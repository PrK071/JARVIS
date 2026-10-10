"""Independent pre-tuning adjudication; only reads acquired third-party source."""
from __future__ import annotations

import ast
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .prepare_predictive_external_v12 import digest, projection_path, source_hash, symbol_locations


# Original reports and upstream source/test corrections, never model answers.
ADJUDICATION = {
    "pytest-dev__pytest-11148": ("development", "BEHAVIORAL_MISMATCH", "src/_pytest/pathlib.py", None, "import_path", "Importlib mode loads an already cached module again instead of returning sys.modules entry.", ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER"], 10811),
    "pytest-dev__pytest-5495": ("development", "DATA_OR_RETURN_CONTRACT", "src/_pytest/assertion/util.py", None, "_compare_eq_sequence", "Bytes indexing gives integers, causing misleading character comparison output.", ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER", "CORRECT_RETURN_VALUE"], 5260),
    "pytest-dev__pytest-7168": ("development", "EXCEPTION_FAILURE", "src/_pytest/_io/saferepr.py", None, "_format_repr_exception", "Reading obj.__class__ invokes broken __getattribute__ while formatting repr failure.", ["FIX_CONSUMER_CONTRACT", "FIX_PRODUCER", "CORRECT_RETURN_VALUE"], None),
    "sphinx-doc__sphinx-11445": ("development", "BEHAVIORAL_MISMATCH", "sphinx/util/rst.py", None, "prepend_prolog", "Broad docinfo regex mistakes domain roles for metadata, inserting prolog inside heading structure.", ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER", "CORRECT_CONFIGURATION"], None),
    "sphinx-doc__sphinx-7738": ("development", "CONFIGURATION_BEHAVIOR", "sphinx/ext/napoleon/docstring.py", "GoogleDocstring", "_escape_args_and_kwargs", "Trailing underscore is escaped even when strip_signature_backslash is false.", ["CORRECT_CONTROL_FLOW", "CORRECT_CONFIGURATION", "FIX_PRODUCER"], 7734),
    "pallets__flask-4045": ("development", "BEHAVIORAL_MISMATCH", "src/flask/blueprints.py", "Blueprint", "__init__", "Blueprint name accepts dots despite nested blueprint namespace contract.", ["VALIDATE_BOUNDARY", "FIX_CONSUMER_CONTRACT"], None),
    "pytest-dev__pytest-7220": ("holdout", "BEHAVIORAL_MISMATCH", "src/_pytest/nodes.py", "Node", "_repr_failure_py", "Relative failure paths use changed working directory instead of invocation directory.", ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER", "CORRECT_CONFIGURATION"], None),
    "pytest-dev__pytest-7490": ("holdout", "BEHAVIORAL_MISMATCH", "src/_pytest/skipping.py", None, "pytest_runtest_call", "Dynamic xfail mark added during call is not reevaluated after execution.", ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER"], 7486),
    "pytest-dev__pytest-9359": ("holdout", "BEHAVIORAL_MISMATCH", "src/_pytest/_code/source.py", None, "get_statement_startend2", "Statement boundary calculation omits decorator line numbers on Python 3.8+.", ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER", "CORRECT_RETURN_VALUE"], None),
    "sphinx-doc__sphinx-8595": ("holdout", "BEHAVIORAL_MISMATCH", "sphinx/ext/autodoc/__init__.py", "ModuleDocumenter", "get_object_members", "Falsy test conflates explicit empty __all__ with absent __all__.", ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER"], None),
}


def main():
    cache = Path(".orchestrator/external-v13")
    cache.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(Path(".orchestrator/external-v12/dataset.json").read_text(encoding="utf-8"))
    rows = {row["instance_id"]: row for row in dataset["rows"]}
    target = Path("tests/data/predictive/external_v13")
    target.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cases = []
    for case_id, annotation in ADJUDICATION.items():
        split, category, path, owner, name, cause, strategies, issue = annotation
        row = rows[case_id]
        url = f"https://codeload.github.com/{row['repo']}/tar.gz/{row['base_commit']}"
        archive = cache / f"{case_id}.tar.gz"
        if not archive.exists():
            archive.write_bytes(opener.open(url, timeout=120).read())
        root = cache / "snapshots" / case_id
        root.mkdir(parents=True, exist_ok=True)
        total = 0
        with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as tar:
            for member in tar:
                relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                if not member.isfile() or relative.suffix != ".py":
                    continue
                total += member.size
                if member.size > 2_000_000 or total > 64_000_000:
                    raise ValueError("Orcamento de snapshot excedido.")
                dest = projection_path(root, relative)
                raw = tar.extractfile(member).read()
                if dest.exists() and dest.read_bytes() != raw:
                    raise ValueError("Snapshot existente divergente; nao sobrescrever.")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(raw)
        tree = ast.parse((root / path).read_text(encoding="utf-8"))
        locations = symbol_locations(tree, name, owner)
        if len(locations) != 1:
            raise ValueError(f"Origem nao univoca: {case_id}")
        repo_url = f"https://github.com/{row['repo']}"
        cases.append({
            "schema_version": 13, "id": case_id, "split": split, "category": category,
            "repo": row["repo"], "base_commit": row["base_commit"],
            "source_archive_url": url, "source_archive_sha256": digest(archive.read_bytes()),
            "source_python_sha256": source_hash(root), "dataset_revision": dataset["revision"],
            "problem_formulations": [row["problem_statement"]],
            "issue_url": f"{repo_url}/issues/{issue}" if issue else f"{repo_url}/pull/{case_id.rsplit('-', 1)[1]}",
            "corrective_pr_url": f"{repo_url}/pull/{case_id.rsplit('-', 1)[1]}",
            "gold_patch_sha256": digest(row["patch"].encode()),
            "gold_test_patch_sha256": digest(row["test_patch"].encode()),
            "oracle": {"root_path": path, "root_symbol": f"{owner}.{name}" if owner else name,
                       "source_locations": locations, "root_cause": cause,
                       "repair_strategies": strategies, "target_paths": [path],
                       "abstention_expected": False, "import_causal": False,
                       "adjudication_basis": "Original report, pre-fix AST, upstream production and regression-test diffs; no model output.",
                       "limitation": "Location/strategy proxy; no third-party code executed. Blueprint constructor is one valid correction, endpoint assertions are a related co-fix."}})
        print(case_id, flush=True)
    raw = (json.dumps(cases, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    seal = digest(raw)
    if (target / "manifest.json").exists():
        previous = json.loads((target / "manifest.json").read_text())
        if seal != previous["cases_sha256"]:
            raise ValueError("Corpus ja selado; nao sobrescrever.")
        print("Snapshots verificados; seal preservado.")
        return
    (target / "cases.json").write_bytes(raw)
    manifest = {"version": 13, "frozen_at": datetime.now(timezone.utc).isoformat(),
                "engine_commit": "a0a47093fe62e0e8875367a8e2915e1ec8cb67fe",
                "dataset": "princeton-nlp/SWE-bench_Lite", "dataset_revision": dataset["revision"],
                "cases_sha256": seal, "independent_defects": len(cases),
                "development": 6, "holdout": 4, "formulations": len(cases),
                "repositories": sorted({case["repo"] for case in cases}),
                "evaluator_version": 13,
                "holdout_thresholds": {"correct_symbol_in_ledger": .80,
                                       "correct_root_candidate_generated": .75,
                                       "external_root_validity": .75},
                "promotion_limit": "Small purposive sample cannot establish production readiness.",
                "historical_external_v12": "All twelve defects observed; diagnostic only, excluded from promotion.",
                "policy": "No gold patch in runtime input; source-only snapshots; holdout one-shot after hardening."}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(seal)


if __name__ == "__main__":
    main()
