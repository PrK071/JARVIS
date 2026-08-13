"""Tests for binary packing."""

import torch
import pytest
from tern.decompose import PRTD, decompose_matrix
from tern.pack import pack_kplane, unpack_kplane, bytes_per_block


@pytest.mark.parametrize("K,group", [
    (1, 256), (2, 256), (3, 256), (4, 256),
    (2, 128), (3, 128),
])
def test_pack_roundtrip(K, group):
    out, inn = 32, group * 2
    W = torch.randn(out, inn)

    prtd = decompose_matrix(W, K=K, group=group, n_iter=0)
    packed = pack_kplane(prtd, group)

    expected_size = prtd.out_features * prtd.n_groups * bytes_per_block(group, K)
    assert len(packed) == expected_size, \
        f"size mismatch: {len(packed)} vs expected {expected_size}"

    unpacked = unpack_kplane(packed, out, inn, group, K)
    assert unpacked.K == K
    assert unpacked.out_features == out
    assert unpacked.in_features == inn

    recon_orig = prtd.dequantize()
    recon_unpacked = unpacked.dequantize()

    max_diff = (recon_orig - recon_unpacked).abs().max().item()
    assert max_diff < 0.02, f"pack/unpack drift: max_diff={max_diff:.6f}"


def test_bytes_per_block():
    assert bytes_per_block(256, 1) == 1 * 64 + 2   # 66
    assert bytes_per_block(256, 2) == 2 * 64 + 4   # 132
    assert bytes_per_block(256, 3) == 3 * 64 + 6   # 198
    assert bytes_per_block(256, 4) == 4 * 64 + 8   # 264
    assert bytes_per_block(128, 2) == 2 * 32 + 4   # 68
    assert bytes_per_block(64, 2) == 2 * 16 + 4    # 36
