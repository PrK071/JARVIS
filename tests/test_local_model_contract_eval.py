from __future__ import annotations

from dataclasses import replace

from tern.orchestrator.local_model_contract_eval import (
    StructuredContractEvaluator,
    progressive_contract_cases,
)
from tern.orchestrator.local_model_runtime import (
    GenerationResult,
    InferenceObservation,
    LocalModelRuntimeError,
    RuntimeFailureCode,
)


class FakeRuntime:
    def __init__(self, failure_case=None):
        self.failure_case = failure_case
        self.calls = []

    def generate_structured(self, messages, *, schema, schema_name, **kwargs):
        self.calls.append((messages, schema, schema_name, kwargs))
        observation = InferenceObservation(
            request_id=schema_name,
            model="candidate",
            runtime="runtime",
            runtime_version="1",
            schema_id=schema_name,
            structured_mode="response_format.json_schema",
            latency_ms=10.0,
            input_tokens=5,
            output_tokens=3,
            retry=False,
            fallback=False,
            validator_result="passed",
            semantic_result=None,
            finish_reason="stop",
            failure_code=None,
        )
        if schema_name == self.failure_case:
            failed = replace(
                observation,
                validator_result="failed",
                failure_code=RuntimeFailureCode.VALIDATOR_FAILURE.value,
            )
            raise LocalModelRuntimeError(
                RuntimeFailureCode.VALIDATOR_FAILURE,
                "invalid",
                observation=failed,
            )
        return GenerationResult("{}", {}, {"usage": {}}, observation)


def test_progressive_contract_has_all_required_schema_stages():
    cases = progressive_contract_cases()
    assert [case.case_id for case in cases] == [
        "trivial_object",
        "enum",
        "optional_field",
        "array",
        "nested_object",
        "tri_state",
        "provenance",
        "grounded_requirements",
        "current_semantic_schema",
    ]


def test_contract_evaluator_reports_metrics_without_executing_tools():
    runtime = FakeRuntime()
    report = StructuredContractEvaluator(runtime).run(repeats=2)
    assert report["request_count"] == 18
    assert report["validity"] == 1.0
    assert report["prompt_tokens"] == 90
    assert report["completion_tokens"] == 54
    assert report["latency_ms"]["p99"] == 10.0
    assert all(call[3].get("temperature") == 0.0 for call in runtime.calls)
    assert all("tools" not in call[3] for call in runtime.calls)


def test_contract_evaluator_preserves_failure_taxonomy():
    report = StructuredContractEvaluator(FakeRuntime("array")).run()
    assert report["validity"] == 8 / 9
    assert report["failure_taxonomy"] == {"VALIDATOR_FAILURE": 1}

