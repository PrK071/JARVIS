"""Model-level ternary quantization.

Replaces nn.Linear layers with TernaryLinear (PRTD-decomposed).
Works on HuggingFace models — walks the module tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from tern.decompose import PRTD, decompose_matrix
from tern.pack import suggest_group


@dataclass
class QuantResult:
    n_layers: int = 0
    nrmse_total: float = 0.0
    layer_stats: list[dict] = field(default_factory=list)

    @property
    def mean_nrmse(self) -> float:
        return self.nrmse_total / max(self.n_layers, 1)


class TernaryLinear(nn.Module):
    """Linear layer backed by K-plane ternary decomposition.

    Supports two modes:
      - materialized=False: codes stored, dequant on forward (memory efficient)
      - materialized=True: precomputed float weight (faster, more memory)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, group: int = 256):
        super().__init__()
        self.in_features = in_features
        self.original_in_features = in_features
        self.out_features = out_features
        self.group = group
        self._materialized = False
        self._weight_cache: torch.Tensor | None = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        K: int = 3,
        group: int = 256,
        n_iter: int = 0,
        device: torch.device | None = None,
    ) -> TernaryLinear:
        module = cls(linear.in_features, linear.out_features, linear.bias is not None, group)
        module.original_in_features = linear.in_features
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)

        weight = linear.weight.data
        if device is not None:
            weight = weight.to(device)

        prtd = decompose_matrix(weight, K=K, group=group, n_iter=n_iter)

        for k in range(K):
            module.register_buffer(f'code_{k}', prtd.codes[k].clone())
            module.register_buffer(f'scale_{k}', prtd.scales[k].clone())

        module._K = K
        module._prtd_cache = prtd
        return module

    @property
    def K(self) -> int:
        return getattr(self, '_K', 0)

    @property
    def prtd(self):
        if hasattr(self, '_prtd_cache') and getattr(self, '_prtd_cache', None) is not None:
            return self._prtd_cache
        from tern.decompose import PRTD
        codes = [getattr(self, f'code_{k}') for k in range(self._K)]
        scales = [getattr(self, f'scale_{k}') for k in range(self._K)]
        return PRTD(codes=codes, scales=scales, group=self.group)

    def materialize_(self):
        self._weight_cache = self.prtd.dequantize()
        self._materialized = True

    def dematerialize_(self):
        self._weight_cache = None
        self._materialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._materialized and self._weight_cache is not None:
            w = self._weight_cache.to(x.device, x.dtype)
        else:
            w = self.prtd.dequantize(dtype=x.dtype, original_in=self.original_in_features).to(x.device)
        return F.linear(x, w, self.bias)


_SKIP_PATTERNS = (
    'lm_head', 'embed_tokens', 'embed.', 'wte',
    'norm', 'layernorm', 'rmsnorm',
    '.gate.', 'gate_inp', 'router', 'gating',
)


def _should_skip(name: str) -> bool:
    name_lower = name.lower()
    for pattern in _SKIP_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def quantize_model(
    model: nn.Module,
    K: int = 3,
    group: int = 256,
    n_iter: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> QuantResult:
    """Walk model tree, replace nn.Linear with TernaryLinear.

    Skips: embeddings, norms, routers, lm_head, small layers (<64 dims).
    Weights are internally padded to group-multiples; original dims preserved.
    """
    result = QuantResult()
    replacements: list[tuple[str, nn.Module, TernaryLinear]] = []

    for name, module in model.named_modules():
        if _should_skip(name):
            continue
        if isinstance(module, nn.Linear):
            if min(module.in_features, module.out_features) < 64:
                continue
            try:
                tl = TernaryLinear.from_linear(module, K=K, group=group, n_iter=n_iter, device=device)
                padded_weight = torch.nn.functional.pad(
                    module.weight.data,
                    (0, tl.prtd.in_features - module.in_features)
                )
                nrmse = tl.prtd.nrmse(padded_weight)
                replacements.append((name, module, tl))
                result.n_layers += 1
                result.nrmse_total += nrmse
                result.layer_stats.append({
                    'name': name,
                    'shape': list(module.weight.shape),
                    'nrmse': nrmse,
                })
                if verbose:
                    print(f"  {name}: NRMSE={nrmse:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  skip {name}: {e}")

    for name, _, tl in replacements:
        parent = model
        parts = name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], tl)

    return result
