"""Convert selected Qwen linear tensors in a base GGUF to TQ3P.

The base GGUF supplies metadata, tokenizer data, and tensors that remain in
their original format. Quantized tensors are produced one at a time so the
full packed model is never held in memory.
"""

from __future__ import annotations

import argparse
import gc
import os
import struct
from pathlib import Path

import torch

from tern.decompose import decompose_matrix
from tern.pack import bytes_per_block, pack_kplane


GGUF_TYPE_TQ3P = 43
TQ3P_GROUP = 256
TQ3P_PLANES = 3

_MODULE_SUFFIXES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def _gguf_name(module_name: str) -> str | None:
    parts = module_name.split(".")
    if (
        len(parts) >= 4
        and parts[0] == "model"
        and parts[1] == "layers"
        and parts[2].isdigit()
    ):
        suffix = ".".join(parts[3:])
        mapped = _MODULE_SUFFIXES.get(suffix)
        if mapped is not None:
            return f"blk.{int(parts[2])}.{mapped}.weight"
    return None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _new_layout(tensors, replacement_sizes: dict[str, int], alignment: int):
    """Return GGUF-relative offsets and sizes, respecting tensor alignment."""
    offsets: list[int] = []
    sizes: list[int] = []
    cursor = 0
    for tensor in tensors:
        cursor = _align(cursor, alignment)
        offsets.append(cursor)
        size = replacement_sizes.get(tensor.name, tensor.n_bytes)
        sizes.append(size)
        cursor += size
    return offsets, sizes, cursor


def _patch_tensor_info(
    header: bytearray,
    tensors,
    replacement_names: set[str],
    relative_offsets: list[int],
) -> None:
    """Patch tensor type and relative data offset fields in-place."""
    for tensor, relative_offset in zip(tensors, relative_offsets):
        field = tensor.field
        name_length = int(field.parts[0][0])
        dimensions = int(field.parts[2][0])
        type_offset = field.offset + 8 + name_length + 4 + dimensions * 8
        offset_offset = type_offset + 4

        if tensor.name in replacement_names:
            struct.pack_into("<I", header, type_offset, GGUF_TYPE_TQ3P)
        struct.pack_into("<Q", header, offset_offset, relative_offset)


def _module_at(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _release_module(model: torch.nn.Module, module_name: str) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], None)


def validate_output(
    base_gguf: Path,
    output_gguf: Path,
    replacement_names: set[str],
) -> None:
    """Validate custom types, layout bounds, and preserved tensor samples."""
    import aenum
    import numpy as np
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import GGML_QUANT_SIZES

    GGML_QUANT_SIZES[GGUF_TYPE_TQ3P] = (
        TQ3P_GROUP,
        bytes_per_block(TQ3P_GROUP, TQ3P_PLANES),
    )
    if GGUF_TYPE_TQ3P not in {member.value for member in GGMLQuantizationType}:
        aenum.extend_enum(GGMLQuantizationType, "TQ3P", GGUF_TYPE_TQ3P)

    base_reader = GGUFReader(base_gguf)
    output_reader = GGUFReader(output_gguf)
    if len(base_reader.tensors) != len(output_reader.tensors):
        raise RuntimeError("tensor count changed")

    output_size = output_gguf.stat().st_size
    custom_count = 0
    with base_gguf.open("rb") as base_file, output_gguf.open("rb") as output_file:
        for base_tensor, output_tensor in zip(
            base_reader.tensors, output_reader.tensors
        ):
            if base_tensor.name != output_tensor.name:
                raise RuntimeError(
                    f"tensor order changed: {base_tensor.name} != {output_tensor.name}"
                )
            if not np.array_equal(base_tensor.shape, output_tensor.shape):
                raise RuntimeError(f"shape changed: {base_tensor.name}")
            if (output_tensor.data_offset - output_reader.data_offset) % output_reader.alignment:
                raise RuntimeError(f"unaligned tensor: {output_tensor.name}")
            if output_tensor.data_offset + output_tensor.n_bytes > output_size:
                raise RuntimeError(f"tensor exceeds file: {output_tensor.name}")

            if output_tensor.name in replacement_names:
                if int(output_tensor.tensor_type) != GGUF_TYPE_TQ3P:
                    raise RuntimeError(f"wrong type: {output_tensor.name}")
                custom_count += 1
                continue

            if int(base_tensor.tensor_type) != int(output_tensor.tensor_type):
                raise RuntimeError(f"preserved type changed: {output_tensor.name}")
            if base_tensor.n_bytes != output_tensor.n_bytes:
                raise RuntimeError(f"preserved size changed: {output_tensor.name}")

            sample_size = min(4096, base_tensor.n_bytes)
            for position in (0, max(0, base_tensor.n_bytes - sample_size)):
                base_file.seek(base_tensor.data_offset + position)
                output_file.seek(output_tensor.data_offset + position)
                if base_file.read(sample_size) != output_file.read(sample_size):
                    raise RuntimeError(f"preserved data changed: {output_tensor.name}")

    if custom_count != len(replacement_names):
        raise RuntimeError(
            f"custom tensors={custom_count}, expected={len(replacement_names)}"
        )
    print(
        f"Validated: {len(output_reader.tensors)} tensors, "
        f"{custom_count} TQ3P, offsets/data OK"
    )


def convert(base_gguf: Path, output_gguf: Path, model_dir: Path) -> None:
    from gguf import GGUFReader
    from transformers import AutoModelForCausalLM

    print("Loading GGUF metadata...", flush=True)
    reader = GGUFReader(base_gguf)
    tensors = list(reader.tensors)
    data_start = reader.data_offset
    alignment = reader.alignment

    print("Loading Hugging Face model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    module_names: dict[str, str] = {}
    replacement_sizes: dict[str, int] = {}
    for module_name, module in model.named_modules():
        tensor_name = _gguf_name(module_name)
        if tensor_name is None or not isinstance(module, torch.nn.Linear):
            continue
        if module.in_features % TQ3P_GROUP:
            raise ValueError(
                f"{module_name}: in_features={module.in_features} "
                f"is not divisible by group={TQ3P_GROUP}"
            )
        module_names[tensor_name] = module_name
        blocks = module.out_features * (module.in_features // TQ3P_GROUP)
        replacement_sizes[tensor_name] = blocks * bytes_per_block(
            TQ3P_GROUP, TQ3P_PLANES
        )

    available = {tensor.name for tensor in tensors}
    missing = sorted(set(module_names) - available)
    if missing:
        raise ValueError(f"{len(missing)} mapped tensors missing from GGUF: {missing[:5]}")
    print(f"TQ3P tensors: {len(module_names)}", flush=True)

    relative_offsets, sizes, data_bytes = _new_layout(
        tensors, replacement_sizes, alignment
    )

    with base_gguf.open("rb") as source:
        header = bytearray(source.read(data_start))
    _patch_tensor_info(header, tensors, set(module_names), relative_offsets)

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    with base_gguf.open("rb") as source, output_gguf.open("wb") as output:
        output.write(header)

        converted = 0
        for index, (tensor, relative_offset, expected_size) in enumerate(
            zip(tensors, relative_offsets, sizes), start=1
        ):
            target_position = data_start + relative_offset
            padding = target_position - output.tell()
            if padding < 0:
                raise RuntimeError(f"layout overlap before {tensor.name}")
            if padding:
                output.write(b"\0" * padding)

            module_name = module_names.get(tensor.name)
            if module_name is None:
                source.seek(tensor.data_offset)
                remaining = tensor.n_bytes
                while remaining:
                    chunk = source.read(min(remaining, 16 * 1024 * 1024))
                    if not chunk:
                        raise EOFError(f"unexpected EOF while copying {tensor.name}")
                    output.write(chunk)
                    remaining -= len(chunk)
                continue

            module = _module_at(model, module_name)
            prtd = decompose_matrix(
                module.weight.detach(),
                K=TQ3P_PLANES,
                group=TQ3P_GROUP,
                n_iter=0,
            )
            packed = pack_kplane(prtd, TQ3P_GROUP)
            if len(packed) != expected_size:
                raise RuntimeError(
                    f"{tensor.name}: packed={len(packed)}, expected={expected_size}"
                )
            output.write(packed)
            converted += 1
            print(
                f"[{converted:3}/{len(module_names)}] {tensor.name} "
                f"{len(packed) / 1024**2:.1f} MiB",
                flush=True,
            )

            del packed, prtd, module
            _release_module(model, module_name)
            gc.collect()

    expected_file_size = data_start + data_bytes
    actual_file_size = output_gguf.stat().st_size
    if actual_file_size != expected_file_size:
        raise RuntimeError(
            f"output size={actual_file_size}, expected={expected_file_size}"
        )

    old_size = base_gguf.stat().st_size
    print(f"Output: {output_gguf} ({actual_file_size / 1024**3:.2f} GiB)")
    print(
        f"Saved: {(old_size - actual_file_size) / 1024**3:.2f} GiB "
        f"({(old_size - actual_file_size) / old_size * 100:.1f}%)"
    )
    validate_output(base_gguf, output_gguf, set(module_names))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("qwen25-1.5b.base.gguf"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qwen25-1.5b.tq3p.gguf"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("qwen25-1.5b"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert(args.base.resolve(), args.output.resolve(), args.model.resolve())


if __name__ == "__main__":
    main()
