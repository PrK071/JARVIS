from __future__ import annotations

import json
from pathlib import Path

from tern.orchestrator.local_model_reasoning_eval import (
    AgenticReasoningEvaluator,
    load_reasoning_cases,
    reasoning_schema,
)
from tern.orchestrator.local_model_runtime import GenerationResult, InferenceObservation


CORPUS = Path(__file__).parent / "data" / "local_model_agentic_reasoning_diagnostic.jsonl"


class ExpectedRuntime:
    def __init__(self, cases):
        self.values = iter(cases)

    def generate_structured(self, _messages, **_kwargs):
        case = next(self.values)
        expected = case.expected
        kinds = expected["required_steps"]
        factor_values = set(expected["required_factors"])
        verification_values = set(expected["required_verification"])
        uncertainty_values = set(expected["required_uncertainties"])
        parsed = {
            "factors": {
                name: name in factor_values
                for name in reasoning_schema()["properties"]["factors"]["required"]
            },
            "steps": [
                {"id": index, "kind": kind, "depends_on": [] if index == 1 else [index - 1]}
                for index, kind in enumerate(kinds, 1)
            ],
            "verification": {
                name: name in verification_values
                for name in reasoning_schema()["properties"]["verification"]["required"]
            },
            "uncertainties": {
                name: name in uncertainty_values
                for name in reasoning_schema()["properties"]["uncertainties"]["required"]
            },
        }
        observation = InferenceObservation(
            "id", "model", "runtime", "1", "schema", "mode", 10.0, 10, 5,
            False, False, "passed", None, "stop", None,
        )
        return GenerationResult(json.dumps(parsed), parsed, {}, observation)


def test_reasoning_corpus_is_separate_structured_and_has_no_selection_label():
    cases = load_reasoning_cases(CORPUS)
    assert len(cases) == 6
    assert {case.category for case in cases} == {
        "multi_agent_discrimination",
        "decomposition_ordering",
        "constraint_reasoning",
        "verification_reasoning",
        "hypothesis_ordering",
        "failure_interpretation",
    }
    assert all("proposed_agent" not in case.expected for case in cases)
    assert "proposed_agent" not in reasoning_schema()["properties"]


def test_reasoning_evaluator_scores_known_truth_without_actions():
    cases = load_reasoning_cases(CORPUS)
    report = AgenticReasoningEvaluator(ExpectedRuntime(cases)).run(cases)
    assert report["structured_validity"] == 1.0
    assert report["automatic_actions"] == 0
    assert set(report["metrics"].values()) == {1.0}
