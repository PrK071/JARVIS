"""GGUF export for ternary-quantified models.

Exporta TQ{K}P GGUF com tipos custom (IDs 42-57).
Inclui tensores nao-quantizados (embeddings, norms, output).
Requer llama.cpp com patch sasori para carregar.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _map_tensor_name(name: str) -> str:
    """Map HuggingFace parameter name to GGUF tensor name."""
    return name


def export_f16_gguf(
    model: torch.nn.Module,
    output_path: str,
    metadata: dict | None = None,
) -> None:
    """Export reconstructed weights as F16 GGUF (stock llama.cpp compatible)."""
    try:
        from gguf import GGUFWriter, GGMLQuantizationType
    except ImportError:
        raise ImportError("gguf required: pip install gguf")

    from tern.quantize import TernaryLinear

    metadata = metadata or {}
    arch = metadata.get('general.architecture', 'llama')
    writer = GGUFWriter(output_path, arch)

    for key in metadata:
        value = metadata[key]
        parts = key.split('.')
        block = '.'.join(parts[:-1]) if len(parts) > 1 else 'general'
        field = parts[-1]
        if isinstance(value, str):
            writer.add_string(f'{block}.{field}', value)
        elif isinstance(value, int):
            writer.add_uint32(f'{block}.{field}', value)
        elif isinstance(value, float):
            writer.add_float32(f'{block}.{field}', value)

    exported_parents = set()
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            exported_parents.add(name)
            w = module.prtd.dequantize(dtype=torch.float32, original_in=module.original_in_features).numpy()
            w_f16 = w.astype(np.float16)
            writer.add_tensor(_map_tensor_name(name + '.weight'), w_f16, raw_dtype=GGMLQuantizationType.F16)
            if module.bias is not None:
                b = module.bias.data.cpu().float().numpy()
                writer.add_tensor(_map_tensor_name(name + '.bias'), b, raw_dtype=GGMLQuantizationType.F32)

    for name, param in model.named_parameters():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        if parent_name in exported_parents:
            continue
        tensor_name = _map_tensor_name(name)
        w = param.data.cpu().float().numpy()
        writer.add_tensor(tensor_name, w, raw_dtype=GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def export_tqkp_gguf(
    model: torch.nn.Module,
    output_path: str,
    K: int,
    group: int,
    metadata: dict | None = None,
) -> None:
    try:
        from gguf import GGUFWriter
        from gguf.quants import GGML_QUANT_SIZES
        from gguf.constants import GGMLQuantizationType
        import aenum
    except ImportError:
        raise ImportError("gguf + aenum required: pip install gguf aenum")

    from tern.quantize import TernaryLinear
    from tern.pack import pack_kplane, bytes_per_block

    metadata = metadata or {}
    arch = metadata.get('general.architecture', 'llama')

    type_map = {
        (1, 256): 44, (2, 256): 42, (3, 256): 43, (4, 256): 45,
        (1, 32): 46, (1, 64): 47, (1, 128): 48,
        (2, 32): 49, (2, 64): 50, (2, 128): 51,
        (3, 32): 52, (3, 64): 53, (3, 128): 54,
        (4, 32): 55, (4, 64): 56, (4, 128): 57,
    }

    writer = GGUFWriter(output_path, arch)

    for key, value in metadata.items():
        if key == 'general.architecture':
            continue
        parts = key.split('.')
        block = '.'.join(parts[:-1]) if len(parts) > 1 else 'general'
        field = parts[-1]
        if isinstance(value, str):
            writer.add_string(f'{block}.{field}', value)
        elif isinstance(value, int):
            writer.add_uint32(f'{block}.{field}', value)
        elif isinstance(value, float):
            writer.add_float32(f'{block}.{field}', value)
        elif isinstance(value, list):
            if value and isinstance(value[0], str):
                writer.add_array(f'{block}.{field}', value)
            elif value and isinstance(value[0], (int, float)):
                writer.add_array(f'{block}.{field}', np.array(value, dtype=np.float32 if isinstance(value[0], float) else np.int32))
        elif isinstance(value, list):
            if value and isinstance(value[0], str):
                writer.add_array(f'{block}.{field}', value)
            elif value and isinstance(value[0], (int, float)):
                writer.add_array(f'{block}.{field}', np.array(value, dtype=np.float32 if isinstance(value[0], float) else np.int32))

    exported = set()

    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            tensor_name = _map_tensor_name(name + '.weight')
            layer_group = getattr(module, 'group', group)
            layer_type_id = type_map.get((K, layer_group), 43)
            blk_bytes_layer = bytes_per_block(layer_group, K)

            if layer_type_id not in GGML_QUANT_SIZES:
                GGML_QUANT_SIZES[layer_type_id] = (256, blk_bytes_layer)
            try:
                existing = {e.value for e in GGMLQuantizationType}
                if layer_type_id not in existing:
                    aenum.extend_enum(GGMLQuantizationType, f'TQ{K}P_G{layer_group}', layer_type_id)
            except Exception:
                pass

            prtd = module.prtd
            packed = pack_kplane(prtd, layer_group)
            writer.add_tensor(tensor_name, np.frombuffer(packed, dtype=np.uint8),
                            raw_dtype=layer_type_id)
            exported.add(name)

            if module.bias is not None:
                bias_name = _map_tensor_name(name + '.bias')
                bias_data = module.bias.data.cpu().float().numpy()
                writer.add_tensor(bias_name, bias_data, raw_dtype=GGMLQuantizationType.F32)

    for name, param in model.named_parameters():
        parts = name.rsplit('.', 1)
        parent_name = parts[0] if len(parts) > 1 else ''
        param_name = parts[-1]

        if parent_name in exported:
            continue

        tensor_name = _map_tensor_name(name)
        w = param.data.cpu().float().numpy()
        writer.add_tensor(tensor_name, w, raw_dtype=GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
