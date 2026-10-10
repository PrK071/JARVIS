import hashlib
import json
from pathlib import Path

import pytest

from tern.orchestrator.predictive.external_evaluation import (
    evaluate_external_cases, load_external_cases, matches_origin, python_source_hash,
    rate, summarize_external_results,
)


CORPUS = Path(__file__).parent / "data" / "predictive" / "external_v12"


def test_external_population_is_independent_and_sealed():
    cases = load_external_cases(CORPUS, split="all")
    assert len(cases) == 12
    assert len({case["repo"] for case in cases}) == 3
    assert len(load_external_cases(CORPUS, split="development")) == 8
    assert len(load_external_cases(CORPUS, split="holdout")) == 4
    assert sum(len(case["problem_formulations"]) for case in cases) == 12
    assert all("patch" not in case and "test_patch" not in case for case in cases)


def test_seal_tampering_and_invalid_split_fail_closed(tmp_path):
    for name in ("manifest.json", "cases.json"):
        (tmp_path/name).write_bytes((CORPUS/name).read_bytes())
    (tmp_path/"cases.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="Seal"):
        load_external_cases(tmp_path, split="development")
    with pytest.raises(ValueError, match="Split"):
        load_external_cases(CORPUS, split="other")


def test_duplicate_case_ids_rejected_even_with_updated_seal(tmp_path):
    cases = json.loads((CORPUS/"cases.json").read_bytes())
    cases[1]["id"] = cases[0]["id"]
    raw = json.dumps(cases).encode()
    manifest = json.loads((CORPUS/"manifest.json").read_bytes())
    manifest["cases_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path/"cases.json").write_bytes(raw)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Populacao"):
        load_external_cases(tmp_path, split="all")


def test_origin_matching_requires_owner_and_source_location():
    oracle = {"root_path": "pkg/file.py", "root_symbol": "Api.run",
              "source_locations": [{"line": 10, "end_line": 20}]}
    root = {"origin_path": "pkg/file.py", "origin_symbol": "Api.run", "origin_line": 12}
    assert matches_origin(root, oracle)
    assert not matches_origin(dict(root, origin_symbol="Other.run"), oracle)
    assert not matches_origin(dict(root, origin_line=2), oracle)
    assert not matches_origin(dict(root, origin_path="test_file.py"), oracle)


def test_rates_have_denominators_and_empty_is_not_perfect():
    assert rate(0, 0, "roots")["rate"] is None
    value = rate(1, 2, "defects")
    assert value["numerator"] == 1 and value["denominator"] == 2
    assert value["rate"] == .5
    assert value["wilson_95"][0] < .5 < value["wilson_95"][1]


def test_multiple_formulations_do_not_inflate_independent_defects():
    row = {"case_id": "one", "mode": "live", "recommended": True,
           "correct_candidate_exists": True, "correct_file_retrieved": True,
           "root_correct": True, "selected_root": True, "selected_scc": False,
           "repair_correct": True, "recommendation_correct": True,
           "abstention_expected": False, "import_causal": False,
           "category": "BEHAVIORAL_MISMATCH", "failure_codes": [],
           "qwen": {"requests": 1, "request_ms": [1000]},
           "safety": dict.fromkeys(("filesystem_mutations", "tool_dispatches",
                                    "authority_grants", "execution_authorized", "destructive_actions"), 0)}
    summary = summarize_external_results([row, dict(row, root_correct=False)])
    assert summary["n_formulations"] == 2
    assert summary["n_independent_defects"] == 1
    assert summary["metrics"]["independent_defect_root_validity"]["numerator"] == 0
    assert summary["metrics"]["correct_abstention"]["rate"] is None


def test_structural_evaluation_does_not_execute_external_source(tmp_path):
    snapshots = tmp_path/"sources"
    root = snapshots/"one"
    root.mkdir(parents=True)
    marker = tmp_path/"MUST_NOT_EXIST"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\ndef decode():\n    return None\n"
    (root/"api.py").write_text(source)
    case = {"id": "one", "category": "BEHAVIORAL_MISMATCH",
            "problem_formulations": ["decode returns None instead of text"],
            "source_python_sha256": python_source_hash(root),
            "oracle": {"root_path": "api.py", "root_symbol": "decode",
                       "source_locations": [{"line": 3, "end_line": 4}],
                       "repair_strategies": ["CORRECT_RETURN_VALUE"],
                       "target_paths": ["api.py"], "abstention_expected": False,
                       "import_causal": False}}
    report = evaluate_external_cases([case], snapshots, tmp_path/"artifacts")
    assert not marker.exists()
    assert (root/"api.py").read_text() == source
    assert report["summary"]["qwen_requests"] == 0
    assert all(value == 0 for value in report["summary"]["safety"].values())


def test_evaluation_artifacts_cannot_be_written_inside_source_snapshot(tmp_path):
    with pytest.raises(ValueError, match="Artefatos"):
        evaluate_external_cases([], tmp_path, tmp_path/"source"/"artifacts")
    assert not (tmp_path/"source").exists()


@pytest.mark.parametrize("name", ("../outside.py", "/absolute.py", "C:/escape.py", "..\\escape.py", "file.py:stream"))
def test_source_projection_rejects_posix_and_windows_escapes(tmp_path, name):
    from pathlib import PurePosixPath
    from scripts.prepare_predictive_external_v12 import projection_path
    with pytest.raises(ValueError):
        projection_path(tmp_path, PurePosixPath(name))
