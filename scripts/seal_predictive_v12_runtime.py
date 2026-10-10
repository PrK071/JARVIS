"""Record the final engine contents before the single external holdout run."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path("tern/orchestrator/predictive")
    output = Path("tests/data/predictive/results/v12-runtime-freeze.json")
    if output.exists():
        raise SystemExit("Freeze ja registrado; nao sobrescrever.")
    baseline = "601b4e697a0d0ee6d79ca716939e37a41d293902"
    frozen = ("analysis.py", "scoring.py", "service.py", "recovery.py", "grounding.py",
              "policy.py", "semantic.py", "import_graph.py")
    hashes = {str(path): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for path in sorted(root.glob("*.py")) if path.name != "external_evaluation.py"}
    unchanged = {}
    for name in frozen:
        path = root/name
        old = subprocess.check_output(["git", "show", f"{baseline}:{path.as_posix()}"])
        unchanged[name] = hashlib.sha256(old.replace(b"\r\n", b"\n")).hexdigest() == hashes[str(path)]
    if not all(unchanged.values()):
        raise ValueError("Camada congelada mudou sem justificativa.")
    manifest = json.loads(Path("tests/data/predictive/external_v12/manifest.json").read_text())
    output.write_text(json.dumps({"version": 12, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": baseline, "corpus_sha256": manifest["cases_sha256"],
        "runtime_source_sha256": hashes, "unchanged_layers": unchanged,
        "holdout_live_executed_at_freeze": False,
        "allowed_engine_changes": ["SCC symptom bridge gating", "typed symptom diagnostics instead of invented exceptions"],
        "scope": "Consultative only; no simulation, scoring tuning or retrieval-budget changes"}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
