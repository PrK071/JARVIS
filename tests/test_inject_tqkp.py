from dataclasses import dataclass

from scripts.inject_tqkp import _align, _new_layout


@dataclass
class FakeTensor:
    name: str
    n_bytes: int


def test_align():
    assert _align(0, 32) == 0
    assert _align(1, 32) == 32
    assert _align(32, 32) == 32
    assert _align(33, 32) == 64


def test_new_layout_uses_relative_aligned_offsets():
    tensors = [
        FakeTensor("first", 35),
        FakeTensor("replace", 17),
        FakeTensor("last", 4),
    ]
    offsets, sizes, total = _new_layout(tensors, {"replace": 5}, 32)

    assert offsets == [0, 64, 96]
    assert sizes == [35, 5, 4]
    assert total == 100
