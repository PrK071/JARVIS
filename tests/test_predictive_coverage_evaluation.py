from dataclasses import replace
from pathlib import Path

from tern.orchestrator.predictive.coverage_evaluation import layer_coverage, summarize_layer_coverage
from tern.orchestrator.predictive.external_evaluation import load_external_cases
from tern.orchestrator.predictive.causal import build_causal_slice
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.models import EvidenceExcerpt
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2, ProjectCandidateGenerator
from tern.orchestrator.security import PathPolicy


def test_new_external_corpus_sealed_independent_from_historical():
    base = Path(__file__).parent / "data/predictive"
    cases = load_external_cases(base / "external_v13", split="all")
    historical = load_external_cases(base / "external_v12", split="all")
    assert len(cases) == 10 and len({item["repo"] for item in cases}) == 3
    assert not ({item["id"] for item in cases} & {item["id"] for item in historical})
    assert len(load_external_cases(base / "external_v13", split="development")) == 6
    assert len(load_external_cases(base / "external_v13", split="holdout")) == 4


def test_file_hit_is_not_symbol_or_fact_hit(tmp_path):
    source = "class Api:\n    def unrelated(self):\n        return 1\n\n    def decode(self):\n        return None\n"
    (tmp_path / "api.py").write_text(source)
    policy = PathPolicy((tmp_path,))
    snapshot = ProjectIndexBuilderV2(tmp_path, path_policy=policy).build()
    context = build_problem_context("api.py", snapshot, ProjectCandidateGenerator().generate("api.py", snapshot), policy)
    context = replace(context, evidence=(EvidenceExcerpt("api.py:1-3", "api.py", 1, 3, "HARD", (), source.split("\n\n")[0]),))
    context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
    causal, roots = build_causal_slice(context, snapshot, policy)
    oracle = {"root_path": "api.py", "root_symbol": "Api.decode", "source_locations": [{"line": 5, "end_line": 6}], "repair_strategies": [], "target_paths": []}
    value = layer_coverage(snapshot, context, causal, roots, oracle)
    assert value["stages"]["correct_file_retrieved"]
    assert value["stages"]["correct_symbol_in_project_index"]
    assert not value["stages"]["correct_symbol_selected"]
    assert not value["stages"]["correct_symbol_in_ledger"]
    assert not value["stages"]["correct_symbol_facts_in_ledger"]
    assert value["first_loss"] == "SYMBOL_WINDOW_MISSED"
    assert value["stages"]["correct_root_selected"] is None
    summary = summarize_layer_coverage([{"layer_coverage": value}])
    assert summary["metrics"]["file_to_symbol_conversion"]["denominator"] == 1
    assert summary["metrics"]["symbol_to_ledger_conversion"]["rate"] is None
