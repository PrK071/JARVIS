"""Tests for model quantization."""

import torch
import torch.nn as nn
import pytest
from tern.quantize import TernaryLinear, quantize_model, QuantResult


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 256)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(256, 256, bias=False),
                'k_proj': nn.Linear(256, 256, bias=False),
                'v_proj': nn.Linear(256, 256, bias=False),
                'o_proj': nn.Linear(256, 256, bias=False),
                'gate_proj': nn.Linear(256, 512, bias=False),
                'up_proj': nn.Linear(256, 512, bias=False),
                'down_proj': nn.Linear(512, 256, bias=False),
            })
        ])
        self.norm = nn.LayerNorm(256)
        self.lm_head = nn.Linear(256, 100, bias=False)

    def forward(self, x):
        return self.lm_head(x)


@pytest.fixture
def toy_model():
    return ToyModel()


def test_quantize_model(toy_model):
    result = quantize_model(toy_model, K=2, group=256, n_iter=0)

    assert result.n_layers == 7  # all 7 projection layers
    assert result.mean_nrmse < 1.0

    assert isinstance(toy_model.layers[0].q_proj, TernaryLinear)
    assert isinstance(toy_model.layers[0].v_proj, TernaryLinear)
    assert isinstance(toy_model.layers[0].gate_proj, TernaryLinear)
    assert isinstance(toy_model.layers[0].down_proj, TernaryLinear)

    # Should NOT be quantized
    assert isinstance(toy_model.lm_head, nn.Linear)
    assert isinstance(toy_model.norm, nn.LayerNorm)
    assert isinstance(toy_model.embed, nn.Embedding)


def test_ternary_linear_forward(toy_model):
    quantize_model(toy_model, K=2, group=256, n_iter=0)

    x = torch.randint(0, 100, (4, 32))
    emb = toy_model.embed(x)

    for layer in toy_model.layers:
        q = layer.q_proj(emb)
        assert q.shape == (4, 32, 256)
        assert not torch.isnan(q).any()


def test_ternary_linear_materialize():
    lin = nn.Linear(128, 256, bias=True)
    tl = TernaryLinear.from_linear(lin, K=2, group=128, n_iter=0)

    x = torch.randn(4, 128)
    y1 = tl(x)

    tl.materialize_()
    y2 = tl(x)
    diff = (y1 - y2).abs().max().item()
    assert diff < 0.1, f"materialize drift: {diff:.6f}"


def test_skip_rules(toy_model):
    result = quantize_model(toy_model, K=2, group=256, n_iter=0)
    names = [s['name'] for s in result.layer_stats]

    assert 'embed' not in str(names)
    assert 'lm_head' not in str(names)
    assert 'norm' not in str(names)
