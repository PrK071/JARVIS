"""tern CLI — local ternary LLM toolkit."""

from __future__ import annotations

import argparse
import sys
import os
import json
import tempfile

import torch
import numpy as np


def cmd_quantize(args):
    """Quantize a HuggingFace model to ternary planes, export GGUF."""
    from tern.decompose import decompose_matrix
    from tern.quantize import TernaryLinear
    from tern.gguf_writer import export_f16_gguf, export_tqkp_gguf

    model_path = args.model
    output = args.output
    K = args.K
    group = args.group
    n_iter = args.n_iter

    print(f"Loading model from {model_path}...")
    try:
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    except ImportError:
        print("transformers package required: pip install transformers")
        return 1

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = config.vocab_size
    if vocab_size % 256 != 0:
        vocab_size = ((vocab_size + 255) // 256) * 256

    metadata = {
        'general.architecture': getattr(config, 'model_type', 'llama'),
        'llama.context_length': getattr(config, 'max_position_embeddings', 2048),
        'llama.embedding_length': getattr(config, 'hidden_size', 0),
        'llama.block_count': getattr(config, 'num_hidden_layers', 0),
        'llama.attention.head_count': getattr(config, 'num_attention_heads', 0),
        'llama.attention.head_count_kv': getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 0)),
        'llama.feed_forward_length': getattr(config, 'intermediate_size', 0),
        'llama.attention.layer_norm_rms_epsilon': getattr(config, 'rms_norm_eps', 1e-5),
        'llama.vocab_size': vocab_size,
    }

    if hasattr(config, 'head_dim'):
        metadata['llama.rope.dimension_count'] = config.head_dim
    elif hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
        metadata['llama.rope.dimension_count'] = config.hidden_size // config.num_attention_heads

    if hasattr(config, 'rope_theta'):
        metadata['llama.rope.freq_base'] = float(config.rope_theta)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(tokenizer, 'sp_model'):
            sp = tokenizer.sp_model
            metadata['tokenizer.ggml.model'] = 'llama'
            metadata['tokenizer.ggml.tokens'] = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
            metadata['tokenizer.ggml.scores'] = [float(sp.get_score(i)) for i in range(sp.get_piece_size())]
            metadata['tokenizer.ggml.token_type'] = [int(sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i)) for i in range(sp.get_piece_size())]
            if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                metadata['tokenizer.ggml.bos_token_id'] = int(tokenizer.bos_token_id)
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                metadata['tokenizer.ggml.eos_token_id'] = int(tokenizer.eos_token_id)
        else:
            vocab = tokenizer.get_vocab()
            if vocab:
                max_id = max(vocab.values()) if isinstance(list(vocab.values())[0], int) else len(vocab)
                tokens = [''] * (max_id + 1)
                for token, idx in vocab.items():
                    if isinstance(idx, int) and idx < len(tokens):
                        tokens[idx] = token
                metadata['tokenizer.ggml.model'] = 'gpt2' if hasattr(tokenizer, 'bpe_ranks') else 'llama'
                metadata['tokenizer.ggml.tokens'] = tokens
                if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                    metadata['tokenizer.ggml.bos_token_id'] = int(tokenizer.bos_token_id)
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    metadata['tokenizer.ggml.eos_token_id'] = int(tokenizer.eos_token_id)
    except Exception:
        pass

    print(f"Loading model weights (this may use significant RAM)...")
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()

    print(f"Quantizing to K={K} ternary planes, group={group}...")
    from tern.quantize import quantize_model
    result = quantize_model(model, K=K, group=group, n_iter=n_iter, verbose=True)

    print(f"\nQuantization complete: {result.n_layers} layers, mean NRMSE={result.mean_nrmse:.4f}")

    if args.tqkp:
        print(f"\nExporting TQKP GGUF: {output}")
        export_tqkp_gguf(model, output, K=K, group=group, metadata=metadata)
    else:
        print(f"\nExporting F16 GGUF: {output}")
        export_f16_gguf(model, output, metadata=metadata)

    print(f"Done. Model saved to {output}")

    return 0


def cmd_chat(args):
    """Chat with a ternary model via llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python required: pip install llama-cpp-python")
        return 1

    model_path = args.model
    print(f"Loading model: {model_path}")

    llama = Llama(
        model_path=model_path,
        n_ctx=args.ctx,
        n_threads=args.threads,
        verbose=False,
    )

    print("Chat ready. Type /exit to quit, /clear to reset.\n")
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ('/exit', '/quit'):
            break
        if user_input.lower() == '/clear':
            messages.clear()
            print("[Conversation cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        response = llama.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=True,
        )

        full_response = ""
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                text = delta['content']
                print(text, end="", flush=True)
                full_response += text
        print("\n")

        messages.append({"role": "assistant", "content": full_response})

    return 0


def cmd_serve(args):
    """Start OpenAI-compatible API server via llama-cpp-python."""
    try:
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import Settings
    except ImportError:
        print("llama-cpp-python[server] required: pip install llama-cpp-python[server]")
        return 1

    settings = Settings(
        model=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
        host=args.host,
        port=args.port,
    )
    app = create_app(settings=settings)
    print(f"Server running at http://{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_download(args):
    """Download a small HuggingFace model suitable for ternary quantization."""
    model_id = args.model
    output_dir = args.output or f"./{model_id.split('/')[-1]}"

    print(f"Downloading {model_id} to {output_dir}...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface-hub required: pip install huggingface-hub")
        return 1

    snapshot_download(model_id, local_dir=output_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded to {output_dir}")
    print(f"\nNext: tern quantize {output_dir} -o model.tq3p.gguf --K 3")
    return 0


def cmd_ollama(args):
    """Generate Ollama Modelfile for a ternary model."""
    model_path = os.path.abspath(args.model)
    model_name = args.name or os.path.splitext(os.path.basename(model_path))[0]

    modelfile = f"""FROM {model_path}
TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>
{{{{ .System }}}}<|end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}<|end|>
{{{{ end }}}}<|assistant|>
\"\"\"
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|end|>"
PARAMETER stop "<|assistant|>"
PARAMETER num_ctx {args.ctx}
PARAMETER num_predict {args.max_tokens}
"""

    modelfile_path = os.path.join(os.path.dirname(model_path) or '.', f'{model_name}.Modelfile')
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)

    print(f"Modelfile written: {modelfile_path}")
    print(f"\nImport into Ollama:")
    print(f"  ollama create {model_name} -f {modelfile_path}")
    print(f"Run:")
    print(f"  ollama run {model_name}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='tern — local ternary LLM toolkit')
    sub = parser.add_subparsers(dest='command')

    p_quant = sub.add_parser('quantize', help='quantize HF model to ternary GGUF')
    p_quant.add_argument('model', help='HF model dir or model ID')
    p_quant.add_argument('-o', '--output', default='model.gguf', help='output GGUF path')
    p_quant.add_argument('--K', type=int, default=3, help='number of ternary planes (1-4)')
    p_quant.add_argument('--group', type=int, default=256, help='scale group size')
    p_quant.add_argument('--n-iter', type=int, default=0, help='CD refinement iterations')
    p_quant.add_argument('--tqkp', action='store_true', help='export TQKP ternary GGUF (needs patched llama.cpp)')
    p_quant.add_argument('--f16', action='store_true', help='export F16 GGUF (stock llama.cpp, default)')

    p_chat = sub.add_parser('chat', help='chat with a ternary model')
    p_chat.add_argument('model', help='GGUF model path')
    p_chat.add_argument('--ctx', type=int, default=4096, help='context size')
    p_chat.add_argument('--threads', type=int, default=4, help='CPU threads')
    p_chat.add_argument('--max-tokens', type=int, default=512)
    p_chat.add_argument('--temperature', type=float, default=0.7)
    p_chat.add_argument('--top-p', type=float, default=0.95)

    p_serve = sub.add_parser('serve', help='start API server')
    p_serve.add_argument('model', help='GGUF model path')
    p_serve.add_argument('--host', default='127.0.0.1')
    p_serve.add_argument('--port', type=int, default=8000)
    p_serve.add_argument('--ctx', type=int, default=4096)
    p_serve.add_argument('--threads', type=int, default=4)

    p_dl = sub.add_parser('download', help='download a HF model')
    p_dl.add_argument('model', help='HF model ID (e.g. Qwen/Qwen2.5-0.5B)')
    p_dl.add_argument('-o', '--output', help='output directory')

    p_ollama = sub.add_parser('ollama', help='generate Ollama Modelfile')
    p_ollama.add_argument('model', help='GGUF model path')
    p_ollama.add_argument('--name', help='model name for Ollama')
    p_ollama.add_argument('--ctx', type=int, default=4096)
    p_ollama.add_argument('--max-tokens', type=int, default=2048)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    cmds = {
        'quantize': cmd_quantize,
        'chat': cmd_chat,
        'serve': cmd_serve,
        'download': cmd_download,
        'ollama': cmd_ollama,
    }
    handler = cmds.get(args.command)
    if handler:
        return handler(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
