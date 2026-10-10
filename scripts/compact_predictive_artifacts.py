"""Explicit maintenance command; complete traces remain recoverable from Git."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tern.orchestrator.predictive.artifacts import compact_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--pattern", default="synthetic-v10-v4-*.json")
    args = parser.parse_args()
    records = {
        path.name: compact_report(path, args.archive, source_commit=args.source_commit,
            source_blob=subprocess.check_output(["git", "show", f"{args.source_commit}:{path.as_posix()}"]))
        for path in sorted(args.results.glob(args.pattern))
    }
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
