"""Read GGUF header key/value metadata without loading tensor data.

Used by local model runtime gates to prove quantization/grouping and
architecture facts from the file itself instead of inferring them from names.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# ggml_type ids used by GGUF tensor headers.
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    36: "MXFP4",
    39: "Q2_0",
}

VALUE_UINT8 = 0
VALUE_INT8 = 1
VALUE_UINT16 = 2
VALUE_INT16 = 3
VALUE_UINT32 = 4
VALUE_INT32 = 5
VALUE_FLOAT32 = 6
VALUE_BOOL = 7
VALUE_STRING = 8
VALUE_ARRAY = 9
VALUE_UINT64 = 10
VALUE_INT64 = 11
VALUE_FLOAT64 = 12

SCALARS = {
    VALUE_UINT8: ("<B", 1),
    VALUE_INT8: ("<b", 1),
    VALUE_UINT16: ("<H", 2),
    VALUE_INT16: ("<h", 2),
    VALUE_UINT32: ("<I", 4),
    VALUE_INT32: ("<i", 4),
    VALUE_FLOAT32: ("<f", 4),
    VALUE_BOOL: ("<?", 1),
    VALUE_UINT64: ("<Q", 8),
    VALUE_INT64: ("<q", 8),
    VALUE_FLOAT64: ("<d", 8),
}


def _read(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of GGUF header")
    return data


def _read_scalar(handle: BinaryIO, value_type: int) -> Any:
    fmt, size = SCALARS[value_type]
    return struct.unpack(fmt, _read(handle, size))[0]


def _read_string(handle: BinaryIO) -> str:
    length = struct.unpack("<Q", _read(handle, 8))[0]
    return _read(handle, length).decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, value_type: int, *, array_limit: int) -> Any:
    if value_type == VALUE_STRING:
        return _read_string(handle)
    if value_type == VALUE_ARRAY:
        item_type = struct.unpack("<I", _read(handle, 4))[0]
        count = struct.unpack("<Q", _read(handle, 8))[0]
        kept: list[Any] = []
        for index in range(count):
            value = _read_value(handle, item_type, array_limit=array_limit)
            if index < array_limit:
                kept.append(value)
        if count > array_limit:
            return {"array_length": count, "head": kept}
        return kept
    return _read_scalar(handle, value_type)


def read_gguf_metadata(path: Path, *, array_limit: int = 8) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = _read(handle, 4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: magic {magic!r}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", _read(handle, 20))
        metadata: dict[str, Any] = {}
        for _index in range(kv_count):
            key = _read_string(handle)
            value_type = struct.unpack("<I", _read(handle, 4))[0]
            metadata[key] = _read_value(handle, value_type, array_limit=array_limit)
        tensor_types: dict[str, int] = {}
        sample: list[dict[str, Any]] = []
        for index in range(tensor_count):
            name = _read_string(handle)
            dim_count = struct.unpack("<I", _read(handle, 4))[0]
            dims = list(struct.unpack(f"<{dim_count}Q", _read(handle, 8 * dim_count)))
            type_id = struct.unpack("<I", _read(handle, 4))[0]
            offset = struct.unpack("<Q", _read(handle, 8))[0]
            type_name = GGML_TYPE_NAMES.get(type_id, f"UNKNOWN_{type_id}")
            tensor_types[type_name] = tensor_types.get(type_name, 0) + 1
            if index < 4:
                sample.append(
                    {
                        "name": name,
                        "dims": dims,
                        "type": type_name,
                        "offset": offset,
                    }
                )
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "gguf_version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "metadata": metadata,
        "tensor_type_histogram": dict(sorted(tensor_types.items())),
        "tensor_sample": sample,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print GGUF header metadata")
    parser.add_argument("path", type=Path)
    parser.add_argument("--array-limit", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = read_gguf_metadata(args.path, array_limit=args.array_limit)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
