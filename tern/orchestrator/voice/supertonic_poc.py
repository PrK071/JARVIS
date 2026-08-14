from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from supertonic import TTS


SAMPLE_RATE = 44_100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--items-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--speed", type=float, default=1.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.items_json.read_text(encoding="utf-8"))

    load_started = time.perf_counter()
    tts = TTS(
        model="supertonic-3",
        model_dir=str(args.model_dir),
        auto_download=False,
    )
    load_seconds = time.perf_counter() - load_started
    style = tts.get_voice_style(args.style)

    metrics = []
    for item in items:
        started = time.perf_counter()
        waveform, _duration = tts.synthesize(
            item["text"],
            voice_style=style,
            total_steps=args.steps,
            speed=args.speed,
            lang="pt",
        )
        synthesis_seconds = time.perf_counter() - started
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        sf.write(
            args.output_dir / item["filename"],
            samples,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        metrics.append(
            {
                "filename": item["filename"],
                "synthesis_seconds": synthesis_seconds,
                "audio_seconds": len(samples) / SAMPLE_RATE,
                "peak": float(np.max(np.abs(samples))),
            }
        )

    args.result_json.write_text(
        json.dumps(
            {
                "model": "Supertone/supertonic-3",
                "style": args.style,
                "language": "pt",
                "steps": args.steps,
                "speed": args.speed,
                "sample_rate": SAMPLE_RATE,
                "loading_seconds": load_seconds,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
