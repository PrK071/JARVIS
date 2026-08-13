#!/usr/bin/env python3
"""End-to-end demo: download model → quantize to ternary → chat.

Recommended small models for testing:
  - Qwen/Qwen2.5-0.5B        (0.5B params, ~1GB RAM)
  - HuggingFaceTB/SmolLM2-135M  (135M params, ~300MB RAM)
  - microsoft/Phi-3-mini-4k-instruct (3.8B, needs more RAM)

Usage:
  python scripts/demo.py Qwen/Qwen2.5-0.5B --K 3
  python scripts/demo.py HuggingFaceTB/SmolLM2-135M --K 2
"""

import argparse
import os
import sys
import subprocess
import tempfile


def run(cmd: list[str], desc: str = "") -> int:
    print(f"\n{'='*60}")
    print(f"  {desc or ' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='tern demo')
    parser.add_argument('model', default='Qwen/Qwen2.5-0.5B', nargs='?',
                        help='HF model ID (default: Qwen/Qwen2.5-0.5B)')
    parser.add_argument('--K', type=int, default=3, help='ternary planes (1-4)')
    parser.add_argument('--group', type=int, default=256)
    parser.add_argument('--n-iter', type=int, default=25)
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--skip-quantize', action='store_true')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]
    output_gguf = args.output or f"{model_name}.tq{args.K}p.gguf"
    model_dir = f"./{model_name}"

    if not args.skip_download:
        ret = run(['python', '-m', 'tern', 'download', args.model, '-o', model_dir],
                  f"Step 1/3: Download {args.model}")
        if ret != 0:
            return ret

    if not args.skip_quantize:
        ret = run([
            'python', '-m', 'tern', 'quantize', model_dir,
            '-o', output_gguf,
            '--K', str(args.K),
            '--group', str(args.group),
            '--n-iter', str(args.n_iter),
        ], f"Step 2/3: Quantize to K={args.K} ternary planes")
        if ret != 0:
            return ret

    print(f"\n{'='*60}")
    print(f"  Model ready: {output_gguf}")
    print(f"  To chat:    python -m tern chat {output_gguf}")
    print(f"  To serve:   python -m tern serve {output_gguf}")
    print(f"  Ollama:     python -m tern ollama {output_gguf}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
