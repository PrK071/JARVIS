from dataclasses import replace

import pytest

from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.oracle_coverage_evaluation import insert_oracle_symbol, oracle_interventions
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2, ProjectCandidateGenerator
from tern.orchestrator.security import PathPolicy


def test_oracle_modes_separate_symbol_fact_and_runtime_input(tmp_path):
    (tmp_path / "api.py").write_text("def unrelated():\n    return 1\ndef decode():\n    return None\n")
    policy = PathPolicy((tmp_path,))
    snapshot = ProjectIndexBuilderV2(tmp_path, path_policy=policy).build()
    problem = "api.py unexpected behavior"
    context = build_problem_context(problem, snapshot, ProjectCandidateGenerator().generate(problem, snapshot), policy)
    context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
    oracle = {"root_path": "api.py", "root_symbol": "decode", "source_locations": [{"line": 3, "end_line": 4}],
              "repair_strategies": ["CORRECT_RETURN_VALUE"], "target_paths": ["api.py"]}
    inserted = insert_oracle_symbol(context, snapshot, policy, oracle)
    assert context.problem == inserted.problem == problem
    assert not context.relevant_symbol_contexts
    assert inserted.relevant_symbol_contexts[-1].selection_reason == "EVALUATION_ORACLE_SYMBOL"
    value = oracle_interventions(context, snapshot, policy, oracle)
    assert value["experimental_only"] and value["not_in_readiness_accuracy"]
    assert [row["mode"] for row in value["experiments"]] == [
        "A_NORMAL", "B_ORACLE_SYMBOL", "C_ORACLE_AST_FACTS", "D_ORACLE_ROOT"]
    assert all(row["qwen_requests"] == 0 for row in value["experiments"])
    assert value["experiments"][2]["inserted_fact_ids"]


def test_missing_grounded_oracle_root_not_fabricated(tmp_path):
    (tmp_path / "api.py").write_text("def mutate(out):\n    out.append(1)\n")
    policy = PathPolicy((tmp_path,))
    snapshot = ProjectIndexBuilderV2(tmp_path, path_policy=policy).build()
    problem = "api.py wrong observable state"
    context = build_problem_context(problem, snapshot, ProjectCandidateGenerator().generate(problem, snapshot), policy)
    context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
    oracle = {"root_path": "api.py", "root_symbol": "mutate", "source_locations": [{"line": 1, "end_line": 2}],
              "repair_strategies": [], "target_paths": ["api.py"]}
    class Never:
        def chat(self, *args, **kwargs):
            raise AssertionError("No unsupported oracle root may call model")
    value = oracle_interventions(context, snapshot, policy, oracle, reasoner=Never())
    assert value["experiments"][-1]["status"] == "ORACLE_ROOT_NOT_REPRESENTABLE"
    assert value["experiments"][-1]["qwen_requests"] == 0
    with pytest.raises(ValueError):
        insert_oracle_symbol(context, snapshot, policy, dict(oracle, root_symbol="missing"))
