"""Adjudicacao independente e aquisicao de Python, sem executar o projeto."""
import ast
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .prepare_predictive_external_v12 import digest, projection_path, source_hash, symbol_locations


ADJUDICATION = {
    "pytest-dev__pytest-7432": ("development", "src/_pytest/skipping.py", None,
        "pytest_runtest_makereport", "WRITES_ATTRIBUTE", "pytest_runtest_makereport",
        "Location rewriting is incorrectly conditional on xfail handling.", 7392),
    "sphinx-doc__sphinx-8713": ("development", "sphinx/ext/napoleon/docstring.py", "GoogleDocstring",
        "_parse_other_parameters_section", "RETURNS_VALUE", "NumpyDocstring",
        "Other parameters formatting does not honor the configuration used by parameters.", 5560),
    "sphinx-doc__sphinx-8721": ("holdout", "sphinx/ext/viewcode.py", None,
        "collect_pages", "YIELDS_VALUE", None,
        "Incremental page collection omits builder/configuration guards.", 8720),
    "sphinx-doc__sphinx-10451": ("holdout", "sphinx/ext/autodoc/typehints.py", None,
        "modify_field_list", "MUTATES_CONTAINER", None,
        "Documented starred parameter names and annotation keys are not reconciled.", 9648),
}


def main():
    cache = Path(".orchestrator/external-v14")
    target = Path("tests/data/predictive/external_v14")
    if (target / "manifest.json").exists():
        raise ValueError("Corpus ja selado; nao sobrescrever.")
    dataset = json.loads(Path(".orchestrator/external-v12/dataset.json").read_text())
    rows = {row["instance_id"]: row for row in dataset["rows"]}
    old = {case["id"] for version in (12, 13) for case in json.loads(
        Path(f"tests/data/predictive/external_v{version}/cases.json").read_text(encoding="utf-8"))}
    assert not old.intersection(ADJUDICATION)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cache.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, (split, path, owner, name, effect, api, cause, issue) in ADJUDICATION.items():
        row = rows[case_id]
        url = f"https://codeload.github.com/{row['repo']}/tar.gz/{row['base_commit']}"
        archive = cache / f"{case_id}.tar.gz"
        if not archive.exists():
            archive.write_bytes(opener.open(url, timeout=120).read(64_000_001))
        if archive.stat().st_size > 64_000_000:
            raise ValueError("Archive excede limite.")
        root = cache / "snapshots" / case_id
        total = 0
        with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as tar:
            for member in tar:
                relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                if not member.isfile() or relative.suffix != ".py":
                    continue
                total += member.size
                if member.size > 2_000_000 or total > 64_000_000:
                    raise ValueError("Snapshot excede limite.")
                dest = projection_path(root, relative)
                raw = tar.extractfile(member).read()
                if dest.exists() and dest.read_bytes() != raw:
                    raise ValueError("Snapshot divergente.")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(raw)
        locations = symbol_locations(ast.parse((root / path).read_text(encoding="utf-8")), name, owner)
        if len(locations) != 1:
            raise ValueError("Origem nao univoca.")
        cases.append({"schema_version": 14, "id": case_id, "split": split,
            "category": "BEHAVIORAL_MISMATCH", "repo": row["repo"],
            "base_commit": row["base_commit"], "dataset_revision": dataset["revision"],
            "source_archive_url": url, "source_archive_sha256": digest(archive.read_bytes()),
            "source_python_sha256": source_hash(root),
            "problem_formulations": [row["problem_statement"]],
            "issue_url": f"https://github.com/{row['repo']}/issues/{issue}",
            "corrective_pr_url": f"https://github.com/{row['repo']}/pull/{case_id.rsplit('-', 1)[1]}",
            "gold_patch_sha256": digest(row["patch"].encode()),
            "gold_test_patch_sha256": digest(row["test_patch"].encode()),
            "oracle": {"root_path": path, "root_symbol": f"{owner}.{name}" if owner else name,
                "source_locations": locations, "root_cause": cause, "affected_api": api,
                "behavioral_effect": effect, "expected_relation": "API_IMPLEMENTATION_EFFECT",
                "repair_strategies": ["CORRECT_CONTROL_FLOW", "FIX_PRODUCER"],
                "target_paths": [path], "abstention_expected": False, "import_causal": False,
                "adjudication_basis": "Original report and independent upstream production/test corrections.",
                "limitation": "Small purposive corpus. API unknown stays null; correction location is not universal semantic truth. 10451 includes another corrected helper."}})
    raw = (json.dumps(cases, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    target.mkdir(parents=True, exist_ok=True)
    (target / "cases.json").write_bytes(raw)
    manifest = {"version": 14, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "engine_commit": "b00a4813278699735d77c41a593cea99790ff8f7",
        "cases_sha256": digest(raw), "independent_defects": 4, "development": 2,
        "holdout": 2, "formulations": 4, "repositories": ["pytest-dev/pytest", "sphinx-doc/sphinx"],
        "evaluator_version": 14,
        "holdout_thresholds": {"implementation_path_recall": .75, "candidate_generation_recall": .75,
            "resolved_call_precision": 1.0, "external_root_validity": .75},
        "policy": "Holdout one-shot; historical v12/v13 diagnostic only; gold diff never reasoner input.",
        "promotion_limit": "Two held-out defects cannot establish broad external generalization."}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest["cases_sha256"])


if __name__ == "__main__":
    main()
