"""Tests for PRTD decomposition."""

import torch
import pytest
from tern.decompose import PRTD, decompose_matrix


@pytest.mark.parametrize("shape,K", [
    ((64, 256), 2),
    ((32, 256), 3),
    ((32, 256), 1),
])
def test_decompose_reconstruct(shape, K):
    out, inn = shape
    W = torch.randn(out, inn) * 2.0
    W[torch.rand(out, inn) > 0.95] *= 10.0

    prtd = decompose_matrix(W, K=K, group=256, n_iter=0)
    recon = prtd.dequantize()

    nrmse = (recon - W).norm().item() / W.norm().item()
    assert nrmse < 1.0, f"NRMSE {nrmse:.4f} too high"
    assert prtd.K == K
    assert prtd.out_features == out
    assert prtd.in_features == inn
    for c in prtd.codes:
        assert c.dtype == torch.int8
        assert set(c.unique().tolist()).issubset({-1, 0, 1})


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_nrmse_monotonic(K):
    W = torch.randn(32, 256)
    prtd = decompose_matrix(W, K=K, group=256, n_iter=0)
    err = prtd.nrmse(W)

    if K < 4:
        prtd_more = decompose_matrix(W, K=K+1, group=256, n_iter=0)
        err_more = prtd_more.nrmse(W)
        assert err_more <= err * 1.01, f"K={K+1} error {err_more:.4f} > K={K} error {err:.4f}"


def test_dequantize_dtype():
    W = torch.randn(32, 256)
    prtd = decompose_matrix(W, K=3, group=256, n_iter=0)

    f32 = prtd.dequantize(dtype=torch.float32)
    f16 = prtd.dequantize(dtype=torch.float16)
    assert f32.dtype == torch.float32
    assert f16.dtype == torch.float16


def test_zero_weight():
    W = torch.zeros(16, 256)
    prtd = decompose_matrix(W, K=3, group=256, n_iter=0)
    recon = prtd.dequantize()
    assert recon.abs().max().item() < 1e-6


def test_prtd_properties():
    out, inn = 32, 512
    prtd = PRTD(
        codes=[torch.randint(-1, 2, (out, inn), dtype=torch.int8) for _ in range(3)],
        scales=[torch.randn(out, 2) for _ in range(3)],
        group=256,
    )
    assert prtd.K == 3
    assert prtd.out_features == out
    assert prtd.in_features == inn
    assert prtd.n_groups == 2
