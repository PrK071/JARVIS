"""Chat with a ternary-quantized model using pure PyTorch inference.

Loads HuggingFace model + tokenizer, applies ternary quantization in-place,
then runs autoregressive generation. No llama.cpp needed.
"""

from __future__ import annotations

import argparse
import sys
import torch


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    if temperature == 0:
        return logits.argmax(-1)
    return torch.multinomial(probs, 1)


def generate(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 128, temperature: float = 0.7,
    top_p: float = 0.95, top_k: int = 50,
) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device) if 'attention_mask' in enc else None

    eos_id = tokenizer.eos_token_id
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, -1, :]

        next_token = _sample(logits, temperature, top_p, top_k)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == eos_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description='Chat ternario nativo PyTorch')
    parser.add_argument('model_dir', help='diretorio do modelo HuggingFace')
    parser.add_argument('--K', type=int, default=3, help='planos ternarios')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--prompt', '-p')
    parser.add_argument('--quantize', action='store_true', help='aplica quantizacao ternaria antes do chat')
    args = parser.parse_args()

    print(f'Carregando modelo: {args.model_dir}')
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()

    if args.quantize:
        print(f'Quantizando para K={args.K} planos ternarios...')
        from tern.quantize import quantize_model
        result = quantize_model(model, K=args.K, verbose=True)
        print(f'Quantizado: {result.n_layers} camadas, NRMSE medio={result.mean_nrmse:.4f}')

    print(f'Pronto. Modelo: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params\n')

    if args.prompt:
        print(f'Prompt: {args.prompt}')
        print('-' * 40)
        resp = generate(model, tokenizer, args.prompt,
                       max_new_tokens=args.max_tokens,
                       temperature=args.temperature,
                       top_p=args.top_p, top_k=args.top_k)
        print(resp)
        print('-' * 40)
        return 0

    print('Chat pronto. /exit sair, /clear limpar, /quantize re-quantizar.\n')

    while True:
        try:
            user = input('Voce: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ('/exit', '/quit'):
            break
        if user.lower() == '/clear':
            print('[Conversa limpa]\n')
            continue
        if user.lower() == '/quantize' and args.quantize:
            print('Re-quantizando...')
            from tern.quantize import quantize_model
            quantize_model(model, K=args.K, verbose=False)

        print('Assistant: ', end='', flush=True)
        resp = generate(model, tokenizer, user,
                       max_new_tokens=args.max_tokens,
                       temperature=args.temperature,
                       top_p=args.top_p, top_k=args.top_k)
        print(resp, '\n')

    return 0


if __name__ == '__main__':
    sys.exit(main())
