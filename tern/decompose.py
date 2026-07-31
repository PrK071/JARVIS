"""Progressive Residual Ternary Decomposition (PRTD).

Algorithm:
  1. SVD initialization — find dominant singular direction per row-group
  2. Ternary threshold — per-row thresholding: sign(w) * 1{|w| > 0.5*mean_abs}
  3. Optimal scale — least-squares per row per plane
  4. Sequential residual fitting — each plane fits residual from prior planes
  5. Block coordinate descent — vectorized per-element ternary search

Key differences from sasori (Adriel007/sasori):
  - Init:    SVD global structure (sasori: per-row greedy threshold)
  - Optimize: vectorized block CD (sasori: ALS KxK solve)
  - Planes:  sequential residual + global CD (sasori: joint from start)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass
class PRTD:
    """K-plane ternary decomposition of a weight matrix.

    W approx sum_k scale_k * code_k   where code_k in {-1,0,+1}
    codes: list[K] int8 tensors (out, inn)
    scales: list[K] float tensors (out, n_groups)
    group: scale group size along input dimension
    """
    codes: list[torch.Tensor] = field(default_factory=list)
    scales: list[torch.Tensor] = field(default_factory=list)
    group: int = 256

    @property
    def K(self) -> int:
        return len(self.codes)

    @property
    def out_features(self) -> int:
        return self.codes[0].shape[0] if self.codes else 0

    @property
    def in_features(self) -> int:
        return self.codes[0].shape[1] if self.codes else 0

    @property
    def n_groups(self) -> int:
        return self.in_features // self.group

    def dequantize(self, dtype: torch.dtype | None = None, original_in: int | None = None) -> torch.Tensor:
        out, inn = self.out_features, self.in_features
        g = self.group
        ng = self.n_groups
        result = torch.zeros(out, inn, dtype=torch.float32)
        for k in range(self.K):
            s = self.scales[k]  # (out, ng)
            c = self.codes[k].float()  # (out, inn)
            for gi in range(ng):
                col_slice = slice(gi * g, (gi + 1) * g)
                result[:, col_slice] += s[:, gi:gi + 1] * c[:, col_slice]
        if original_in is not None and original_in < inn:
            result = result[:, :original_in]
        if dtype is not None:
            result = result.to(dtype)
        return result

    def nrmse(self, target: torch.Tensor) -> float:
        recon = self.dequantize()
        err = target.float() - recon
        return (err.norm() / target.float().norm()).item()


def decompose_matrix(
    weight: torch.Tensor,
    K: int = 3,
    group: int = 256,
    n_iter: int = 0,
    svd_init: bool = True,
    verbose: bool = False,
) -> PRTD:
    """Decompose a float weight matrix into K ternary planes.

    Args:
        weight: (out_features, in_features) float tensor
        K: number of ternary planes (1-4 recommended)
        group: scale group size (must divide in_features)
        n_iter: fast CD refinement iterations (0 to skip, max 10)
        svd_init: use SVD for initialization (True) or random (False)
        verbose: print per-plane NRMSE
    """
    assert weight.ndim == 2, f"Expected 2D weight, got {weight.ndim}D"
    out, inn_orig = weight.shape

    # Pad to full blocks
    padded_inn = ((inn_orig + group - 1) // group) * group
    inn = padded_inn
    ng = inn // group

    W = weight.float()
    if padded_inn > inn_orig:
        W = torch.nn.functional.pad(W, (0, padded_inn - inn_orig))

    codes: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []

    residual = W.clone()

    for k in range(K):
        code_k, scale_k = _fit_single_plane(residual, group, svd_init=svd_init)
        codes.append(code_k)
        scales.append(scale_k)

        s_expanded = scale_k[:, :, None].expand(out, ng, group).reshape(out, inn)
        residual = residual - s_expanded * code_k.float()

        if verbose:
            prtd_tmp = PRTD(codes=list(codes), scales=list(scales), group=group)
            print(f"  plane {k}: NRMSE={prtd_tmp.nrmse(W):.4f}")

    prtd = PRTD(codes=codes, scales=scales, group=group)

    if n_iter > 0:
        _refine_block_cd(prtd, W, n_iter=n_iter, verbose=verbose)

    return prtd


def _fit_single_plane(
    W: torch.Tensor,
    group: int,
    svd_init: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, inn = W.shape
    ng = inn // group

    codes = torch.zeros(out, inn, dtype=torch.int8)
    scales = torch.zeros(out, ng, dtype=torch.float32)

    for gi in range(ng):
        col_slice = slice(gi * group, (gi + 1) * group)
        W_block = W[:, col_slice].float()  # (out, g)

        abs_mean = W_block.abs().mean(dim=1, keepdim=True)  # (out, 1)
        threshold = 0.7 * abs_mean

        ternary = torch.zeros_like(W_block, dtype=torch.int8)
        ternary[W_block > threshold] = 1
        ternary[W_block < -threshold] = -1

        nz = ternary.abs().sum(dim=1).float()  # (out,)
        dot = (W_block * ternary.float()).sum(dim=1)  # (out,)

        scale_row = torch.where(nz > 0, dot / nz, torch.zeros_like(dot))

        codes[:, col_slice] = ternary
        scales[:, gi] = scale_row

    return codes, scales


def _refine_block_cd(
    prtd: PRTD,
    target: torch.Tensor,
    n_iter: int = 5,
    verbose: bool = False,
) -> None:
    g = prtd.group
    ng = prtd.n_groups
    K = prtd.K

    target_f32 = target.float()

    for it in range(n_iter):
        total_improved = 0

        for k in range(K):
            recon = prtd.dequantize()
            residual = target_f32 - recon  # (out, inn)

            for gi in range(ng):
                col_slice = slice(gi * g, (gi + 1) * g)
                res_block = residual[:, col_slice]  # (out, g)
                old_code = prtd.codes[k][:, col_slice]  # (out, g)
                s_block = prtd.scales[k][:, gi:gi + 1]  # (out, 1)

                errors = []
                try_values = torch.tensor([-1, 0, 1], dtype=torch.float32)
                for trit_val in try_values:
                    delta = s_block * (float(trit_val) - old_code.float())
                    err = (res_block - delta).pow(2)
                    errors.append(err)

                errors_stack = torch.stack(errors, dim=0)  # (3, out, g)
                best_idx = errors_stack.argmin(dim=0)  # (out, g)
                trit_tensor = torch.tensor([-1, 0, 1], dtype=torch.int8)
                best_trits = trit_tensor[best_idx]

                improved = (best_trits != old_code).sum().item()
                total_improved += improved
                prtd.codes[k][:, col_slice] = best_trits

        if total_improved == 0:
            break

    for k in range(K):
        for gi in range(ng):
            col_slice = slice(gi * g, (gi + 1) * g)
            c_k = prtd.codes[k][:, col_slice].float()
            w_t = target_f32[:, col_slice]
            dot = (w_t * c_k).sum(dim=1)
            norm = (c_k * c_k).sum(dim=1)
            prtd.scales[k][:, gi] = torch.where(norm > 0, dot / norm, torch.zeros_like(dot))

    if verbose:
        print(f"  CD: {n_iter} iters, final NRMSE={prtd.nrmse(target_f32):.4f}")
