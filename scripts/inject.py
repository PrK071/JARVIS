"""Inject ternary-reconstructed weights into base GGUF.

Uses gguf.GGUFReader for reliable parsing, binary patch for data.
"""

import os
import numpy as np
import torch


def _map_hf_to_gguf(name: str) -> str | None:
    parts = name.split('.')
    if name == 'model.embed_tokens.weight': return 'token_embd.weight'
    if name == 'model.norm.weight': return 'output_norm.weight'
    if name == 'lm_head.weight': return 'output.weight'
    if len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'layers' and parts[2].isdigit():
        layer_idx = int(parts[2])
        hf_suffix = '.'.join(parts[3:])
        mapping = {
            'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
            'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_output',
            'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up',
            'mlp.down_proj': 'ffn_down',
        }
        for hf_key, gguf_key in mapping.items():
            if hf_key == hf_suffix:
                return f'blk.{layer_idx}.{gguf_key}.weight'
    return None


def inject_ternary(base_gguf: str, output_gguf: str, quantized_model) -> None:
    from gguf import GGUFReader
    from tern.quantize import TernaryLinear

    reader = GGUFReader(base_gguf)

    # Build replacement map: GGUF tensor name → F16 numpy data
    replacement = {}
    for name, module in quantized_model.named_modules():
        if isinstance(module, TernaryLinear):
            gguf_name = _map_hf_to_gguf(name)
            if gguf_name:
                w = module.prtd.dequantize(
                    dtype=torch.float32,
                    original_in=module.original_in_features
                ).numpy().astype(np.float16)
                replacement[gguf_name] = w

    print(f"Tensors to inject: {len(replacement)}")

    # Read entire base file
    with open(base_gguf, 'rb') as f:
        data = bytearray(f.read())

    # Read tensor offsets from GGUFReader (reliable, no manual parsing)
    tensor_offsets = {}
    for tensor in reader.tensors:
        tensor_offsets[tensor.name] = (tensor.data_offset, tensor.n_bytes)

    # Patch in-place
    patched = 0
    errors = []
    for name, new_w in replacement.items():
        if name not in tensor_offsets:
            print(f"  ERROR: tensor '{name}' not found in GGUF")
            continue
        off, old_size = tensor_offsets[name]
        new_bytes = new_w.tobytes()
        new_size = len(new_bytes)

        if new_size != old_size:
            errors.append((name, old_size, new_size, new_w.shape))
            continue

        data[off:off + new_size] = new_bytes
        patched += 1

    if errors:
        print("SIZE MISMATCHES (tensor cannot be binary-patched):")
        for name, old, new, shape in errors[:5]:
            print(f"  {name}: old={old} new={new} shape={shape}")
        raise ValueError(f"{len(errors)}/{len(replacement)} tensors have wrong size")

    with open(output_gguf, 'wb') as f:
        f.write(data)

    print(f"Patched {patched}/{len(replacement)} tensors in-place")
    print(f"Metadata preserved (tokenizer, types, hyperparams)")
    print(f"Output: {output_gguf} ({os.path.getsize(output_gguf)/1024/1024:.1f} MB)")


def main():
    base_gguf = 'D:/tern/smollm2-135m.base.gguf'
    output_gguf = 'D:/tern/smollm2-135m.injected.gguf'

    import sys
    if len(sys.argv) > 1: base_gguf = sys.argv[1]
    if len(sys.argv) > 2: output_gguf = sys.argv[2]

    from transformers import AutoModelForCausalLM
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'D:/tern/smollm2-135m',
        torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()

    print("Quantizing to ternary...")
    from tern.quantize import quantize_model
    result = quantize_model(model, K=3, verbose=False)
    print(f"Quantized: {result.n_layers} layers, NRMSE={result.mean_nrmse:.4f}")

    inject_ternary(base_gguf, output_gguf, model)
    print("Done!")


if __name__ == '__main__':
    main()
