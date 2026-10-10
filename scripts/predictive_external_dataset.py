"""Acquire upstream benchmark metadata without executing third-party code."""
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    import pyarrow.parquet as pq
    root = Path(".orchestrator/external-v12")
    root.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    info = json.load(opener.open("https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Lite", timeout=60))
    frozen = Path("tests/data/predictive/external_v12/manifest.json")
    revision = json.loads(frozen.read_text())["dataset_revision"] if frozen.exists() else info["sha"]
    rows = []
    for split in ("dev", "test"):
        url = f"https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/{revision}/data/{split}-00000-of-00001.parquet"
        raw = opener.open(url, timeout=60).read()
        file = root / f"{split}.parquet"
        file.write_bytes(raw)
        values = pq.read_table(file).to_pylist()
        rows.extend(values)
        print(split, len(values), hashlib.sha256(raw).hexdigest())
    (root / "dataset.json").write_text(json.dumps({"revision": revision, "rows": rows}), encoding="utf-8")
    selected = [row for row in rows if row["repo"] in {"psf/requests", "pallets/flask", "marshmallow-code/marshmallow", "pytest-dev/pytest"}]
    for row in selected:
        print(json.dumps({"id": row["instance_id"], "repo": row["repo"],
            "problem": row["problem_statement"][:400], "patch_bytes": len(row["patch"])}))


if __name__ == "__main__":
    main()
