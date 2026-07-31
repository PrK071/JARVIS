"""Qwen2.5-1.5B-Instruct tensor analysis."""
hidden = 1536
intermediate = 8960
heads = 12
kv_heads = 2
head_dim = hidden // heads
kv_dim = kv_heads * head_dim
vocab = 151936
n_layers = 28

print('=== Qwen2.5-1.5B-Instruct - Tensor Analysis ===')
print(f'hidden_size={hidden}, intermediate={intermediate}')
print(f'heads={heads}, kv_heads={kv_heads}, head_dim={head_dim}')
print(f'vocab={vocab}, n_layers={n_layers}')
print()

total_params = 0
quant_params = 0

def check(name, shape, quantizable=True, skip_reason=''):
    global total_params, quant_params
    p = shape[0] * shape[1]
    total_params += p
    in_dim = shape[1]
    div_ok = in_dim % 256 == 0
    div_str = 'YES' if div_ok else f'NO (rem={in_dim%256})'
    tqkp = 'YES' if (div_ok and quantizable) else 'NO'
    if tqkp == 'YES':
        quant_params += p
    print(f'  {name:<45} ({shape[0]:>6},{shape[1]:<6}) params={p:>12,} in_dim={in_dim:>6} div256={div_str:<10} TQKP={tqkp:<4} {skip_reason}')

check('token_embd.weight (embed)', (hidden, vocab), False, 'embedding')
for i in range(n_layers):
    check(f'blk.{i}.attn_q.weight', (hidden, hidden))
    check(f'blk.{i}.attn_k.weight', (kv_dim, hidden))
    check(f'blk.{i}.attn_v.weight', (kv_dim, hidden))
    check(f'blk.{i}.attn_output.weight', (hidden, hidden))
    check(f'blk.{i}.ffn_gate.weight', (intermediate, hidden))
    check(f'blk.{i}.ffn_up.weight', (intermediate, hidden))
    check(f'blk.{i}.ffn_down.weight', (hidden, intermediate))
    check(f'blk.{i}.attn_norm.weight', (hidden, 1), False, 'norm')
    check(f'blk.{i}.ffn_norm.weight', (hidden, 1), False, 'norm')
check('output.weight (lm_head)', (hidden, vocab), False, 'tied embed')
check('output_norm.weight', (hidden, 1), False, 'norm')

print()
print(f'TOTAL params: {total_params/1e9:.3f}B')
print(f'QUANTIZABLE params: {quant_params/1e9:.3f}B ({quant_params/total_params*100:.1f}%)')
print(f'NON-QUANT params: {(total_params-quant_params)/1e6:.1f}M (embeddings+norms)')
print()

tqkp_bpw = 6.1875  # K=3, g=256
tqkp_size_gb = quant_params * tqkp_bpw / 8 / 1024**3
nq_size_gb = (total_params - quant_params) * 2 / 1024**3  # F16 for embed+norms
overhead_gb = 0.05  # metadata, alignment
print(f'TQKP weights (K=3, {tqkp_bpw} bpw): {tqkp_size_gb:.2f} GB')
print(f'Embeddings+norms (F16):             {nq_size_gb:.3f} GB')
print(f'Metadata + alignment:               ~{overhead_gb:.2f} GB')
print(f'TOTAL GGUF estimate:                {tqkp_size_gb + nq_size_gb + overhead_gb:.2f} GB')
print()
print(f'Baseline F16 size: {total_params * 2 / 1024**3:.2f} GB')
print(f'Saving vs F16: {total_params * 2 / 1024**3 - (tqkp_size_gb + nq_size_gb + overhead_gb):.2f} GB ({(1 - (tqkp_size_gb + nq_size_gb + overhead_gb) / (total_params * 2 / 1024**3)) * 100:.1f}%)')

# Per-tensor type summary
print()
print('=== Summary by tensor type ===')
n_tqkp = n_layers * 7  # q,k,v,o,gate,up,down per layer
n_f16 = 2  # embed + lm_head
n_f32 = n_layers * 2 + 1  # norms
print(f'TQKP tensors: {n_tqkp}')
print(f'F16 tensors:  {n_f16}')
print(f'F32 tensors:  {n_f32}')
print(f'Total:        {n_tqkp + n_f16 + n_f32}')
