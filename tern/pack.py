"""Binary packing for TQ{K}P ternary weights.

Two plane layouts:
  - strided (tq2_0 compatible): group must be multiple of 128
  - flat: any group size, sequential 2-bit packing

Block layout:
  Per (row, block) of `g` input elements:
    uint8 qs1[g/4] ... uint8 qsK[g/4]  — K planes, 2 bits/trit
    fp16  d1 ... dK                      — K per-block scales
  bytes_per_block = K * g/4 + 2 * K

Strided layout (tq2_0 compatible):
  Logical position p -> byte (p/128)*32 + (p%32), bit-pair (p/32)%4

Flat layout:
  Sequential 4 trits per byte, bytes in order.
"""

from __future__ import annotations

import struct
import numpy as np
import torch


def bytes_per_block(group: int, K: int) -> int:
    return K * (group // 4) + 2 * K


def _pack_plane_flat(codes_2d: torch.Tensor) -> np.ndarray:
    """Pack 2D ternary into flat 2-bit layout. Vectorized numpy."""
    flat = codes_2d.numpy().ravel().astype(np.int32) + 1  # trit+1 ∈ {0,1,2}
    n = len(flat)
    npad = (4 - (n % 4)) % 4
    if npad:
        flat = np.pad(flat, (0, npad), constant_values=1)  # pad with 0→trit 1
    n_bytes = len(flat) // 4
    reshaped = flat.reshape(n_bytes, 4)  # (n_bytes, 4)
    packed = (reshaped[:, 0] | (reshaped[:, 1] << 2) |
              (reshaped[:, 2] << 4) | (reshaped[:, 3] << 6)).astype(np.uint8)
    return packed


def _unpack_plane_flat(packed: np.ndarray, out: int, inn: int) -> torch.Tensor:
    """Unpack flat 2-bit layout. Vectorized numpy."""
    total = out * inn
    n_bytes = (total + 3) // 4
    packed = packed[:n_bytes].astype(np.uint32)
    vals = np.zeros(n_bytes * 4, dtype=np.int8)
    for bp in range(4):
        shift = bp * 2
        vals[bp::4] = ((packed >> shift) & 3).astype(np.int8) - 1
    return torch.from_numpy(vals[:total].reshape(out, inn).copy()).to(torch.int8)


def _pack_plane_strided(codes_2d: torch.Tensor, group: int) -> np.ndarray:
    """Pack 2D ternary into strided tq2_0 plane layout. Vectorized."""
    out, inn = codes_2d.shape
    ng = inn // group
    qb = group // 4
    blocks = codes_2d.numpy().reshape(out * ng, group).astype(np.uint8) + 1

    byte_index = np.arange(qb)
    base_position = (byte_index // 32) * 128 + byte_index % 32
    packed = (
        blocks[:, base_position]
        | (blocks[:, base_position + 32] << 2)
        | (blocks[:, base_position + 64] << 4)
        | (blocks[:, base_position + 96] << 6)
    ).astype(np.uint8)
    return packed.ravel()


def _unpack_plane_strided(packed: np.ndarray, out: int, inn: int, group: int) -> torch.Tensor:
    ng = inn // group
    qb = group // 4
    blocks = packed.reshape(out * ng, qb)
    codes = np.empty((out * ng, group), dtype=np.int8)

    byte_index = np.arange(qb)
    base_position = (byte_index // 32) * 128 + byte_index % 32
    for bit_pair in range(4):
        positions = base_position + bit_pair * 32
        codes[:, positions] = (
            ((blocks >> (bit_pair * 2)) & 3).astype(np.int8) - 1
        )
    return torch.from_numpy(codes.reshape(out, inn).copy())


def _pack_plane(codes_2d: torch.Tensor, group: int) -> np.ndarray:
    """Pack a ternary plane: strided if group%128==0, flat otherwise."""
    if group % 128 == 0:
        return _pack_plane_strided(codes_2d, group)
    return _pack_plane_flat(codes_2d)


def _unpack_plane(packed: np.ndarray, out: int, inn: int, group: int) -> torch.Tensor:
    """Unpack: strided if group%128==0, flat otherwise."""
    if group % 128 == 0:
        return _unpack_plane_strided(packed, out, inn, group)
    return _unpack_plane_flat(packed, out, inn)


def pack_kplane(prtd, group: int) -> bytes:
    """Pack PRTD into binary blob (sasori-compatible TQ{K}P block-interleaved format).

    Layout per (row, block): K strided planes interleaved, then K fp16 scales.
    Block size = K*g/4 (trits) + 2*K (scales).
    """
    K = len(prtd.codes)
    out, inn = prtd.codes[0].shape
    ng = inn // group
    qb = group // 4
    n_blocks = out * ng

    packed_planes = [
        _pack_plane(prtd.codes[k], group).reshape(n_blocks, qb)
        for k in range(K)
    ]
    scales = np.stack(
        [scale.detach().cpu().numpy() for scale in prtd.scales],
        axis=-1,
    ).astype("<f2", copy=False)
    scale_bytes = scales.view(np.uint8).reshape(n_blocks, 2 * K)

    return np.concatenate([*packed_planes, scale_bytes], axis=1).tobytes()


def unpack_kplane(data: bytes, out: int, inn: int, group: int, K: int, device='cpu'):
    """Unpack binary TQ{K}P blob (block-interleaved) back to PRTD."""
    from tern.decompose import PRTD

    ng = inn // group
    qb = group // 4
    bpb = K * qb + 2 * K

    expected_size = out * ng * bpb
    assert len(data) == expected_size, \
        f"Data size mismatch: got {len(data)}, expected {expected_size}"

    # Extract plane data and scales
    codes_np = [np.zeros(out * ng * qb, dtype=np.uint8) for _ in range(K)]
    scales_list = [torch.zeros(out, ng, dtype=torch.float32) for _ in range(K)]

    for r in range(out):
        for gi in range(ng):
            block_off = (r * ng + gi) * bpb
            for k in range(K):
                # Copy plane trit data
                plane_off = (r * ng + gi) * qb
                src_start = block_off + k * qb
                codes_np[k][plane_off:plane_off + qb] = np.frombuffer(
                    data[src_start:src_start + qb], dtype=np.uint8
                )
                # Read scale
                scale_off = block_off + K * qb + k * 2
                s = struct.unpack('<e', data[scale_off:scale_off + 2])[0]
                scales_list[k][r, gi] = s

    codes = []
    for k in range(K):
        code_k = _unpack_plane(codes_np[k], out, inn, group)
        codes.append(code_k)

    if device != 'cpu':
        codes = [c.to(device) for c in codes]
        scales_list = [s.to(device) for s in scales_list]

    return PRTD(codes=codes, scales=scales_list, group=group)


def suggest_group(in_features: int, preferred: int = 256) -> int:
    """Find largest group size <= preferred that divides in_features and is >= 32."""
    for g in (preferred, 128, 64, 32):
        if in_features % g == 0:
            return g
    for g in range(min(in_features, 256), 0, -1):
        if in_features % g == 0 and g >= 32:
            return g
    return max(1, in_features)  # fallback: treat as single group
