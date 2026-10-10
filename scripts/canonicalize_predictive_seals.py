"""Preserve historical worktree seals and record platform-independent Git seals."""
import json
from pathlib import Path

from tern.orchestrator.predictive.evaluation import predictive_corpus_hash


def main():
    root = Path("tests/data/predictive")
    path = root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    historical = value.setdefault("historical_worktree_seals", {})
    for key in tuple(value):
        if key.endswith("_sha256") and key != "holdout_v6_sha256":
            split = key[:-7]
            historical.setdefault(key, value[key])
            value[key] = predictive_corpus_hash(root, split=split)
    value["hash_encoding"] = "LF canonical bytes; .gitkeep directory markers excluded"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from tern.orchestrator.predictive.robustness import robustness_suite_hash
    historical.setdefault("holdout_v6_sha256", value["holdout_v6_sha256"])
    value["holdout_v6_sha256"] = robustness_suite_hash(split="holdout_v6")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
