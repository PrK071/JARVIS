"""tern — local ternary LLM inference toolkit.

Progressive Residual Ternary Decomposition (PRTD): SVD-init, coordinate-descent
refinement.  Different from sasori's ALS joint optimization.

Inspiration: sasori (Adriel007/sasori) — post-hoc multi-plane ternarization.
"""
from tern.decompose import PRTD, decompose_matrix
from tern.quantize import quantize_model, QuantResult
from tern.pack import pack_kplane, unpack_kplane, bytes_per_block

__all__ = [
    "PRTD",
    "decompose_matrix",
    "quantize_model",
    "QuantResult",
    "pack_kplane",
    "unpack_kplane",
    "bytes_per_block",
]
