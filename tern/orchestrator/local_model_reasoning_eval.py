from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .client import LlamaClient
from .local_model_runtime import (
    LocalModelRuntimeError,
    OpenAICompatibleLocalRuntime,
    RuntimeDescriptor,
)


FACTORS = (
    "READ_ONLY_CONSTRAINT",
    "NO_TEST_CONSTRAINT",
    "CODE_ANALYSIS_REQUIRED",
    "MUTATION_REQUIRED",
    "TEST_EXECUTION_REQUIRED",
    "SINGLE_ELIGIBLE",
    "MULTIPLE_ELIGIBLE",
    "NO_ELIGIBLE",
    "CONFLICTING_HYPOTHESES",
    "EVIDENCE_FIRST",
    "VERIFICATION_FAILED",
    "SCOPE_VIOLATION",
)
STEP_KINDS = (
    "INSPECT",
    "GATHER_EVIDENCE",
    "COMPARE_HYPOTHESES",
    "ANALYZE",
    "EDIT",
    "TEST",
    "PROPOSE",
    "VERIFY_DESIGN",
    "VERIFY",
    "INTERPRET_RESULT",
    "REPORT",
)
VERIFICATION_FACTS = (
    "NO_MUTATION",
    "NO_TEST_EXECUTION",
    "EXPECTED_FILES",
    "TEST_EXIT_CODE",
    "FORBIDDEN_FILES",
    "VERIFICATION_PLAN",
    "EVIDENCE_SUPPORT",
    "WORKER_CLAIM_NOT_SUFFICIENT",
)
UNCERTAINTIES = (
    "AGENT_TIE_UNRESOLVED",
    "HYPOTHESIS_UNRESOLVED",
    "MUTATION_UNRESOLVED",
    "TEST_EXECUTION_UNRESOLVED",
)


def reasoning_schema() -> dict[str, Any]:
    def flags(names: tuple[str, ...]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {name: {"type": "boolean"} for name in names},
            "required": list(names),
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "factors": flags(FACTORS),
            "steps": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer", "minimum": 1, "maximum": 8},
                        "kind": {"type": "string", "enum": list(STEP_KINDS)},
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1, "maximum": 5},
                            "maxItems": 4,
                        },
                    },
                    "required": ["id", "kind", "depends_on"],
                },
            },
            "verification": flags(VERIFICATION_FACTS),
            "uncertainties": flags(UNCERTAINTIES),
        },
        "required": ["factors", "steps", "verification", "uncertainties"],
    }


@dataclass(frozen=True)
class ReasoningCase:
    case_id: str
    category: str
    input: Mapping[str, Any]
    expected: Mapping[str, Any]


def load_reasoning_cases(path: Path) -> tuple[ReasoningCase, ...]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        cases.append(
            ReasoningCase(value["id"], value["category"], value["input"], value["expected"])
        )
    return tuple(cases)


def _set_metrics(actual: set[str], expected: set[str]) -> tuple[float, float, float]:
    true_positive = len(actual & expected)
    precision = true_positive / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = true_positive / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


class AgenticReasoningEvaluator:
    system_prompt = (
        "Analyze the supplied grounded task without choosing an agent and without "
        "executing anything. Use only the allowed categorical factors. Preserve "
        "UNKNOWN as uncertainty, keep multi-agent ties unresolved, order evidence "
        "before conclusions and objective verification after proposed work. Keep "
        "every array minimal, never repeat an item, and include only values supported "
        "by the input. Use two to five steps; dependencies may reference earlier "
        "step ids only."
    )

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def run(self, cases: tuple[ReasoningCase, ...]) -> dict[str, Any]:
        records = [self._evaluate(case) for case in cases]
        valid = [record for record in records if record["structured_valid"]]
        metric_names = (
            "factor_precision",
            "factor_recall",
            "factor_f1",
            "step_recall",
            "ordering_accuracy",
            "verification_recall",
            "uncertainty_recall",
            "forbidden_factor_accuracy",
        )
        return {
            "case_count": len(records),
            "structured_validity": len(valid) / len(records) if records else 0.0,
            "metrics": {
                name: statistics.fmean(record[name] for record in records)
                if records
                else 0.0
                for name in metric_names
            },
            "latency_ms": {
                "mean": statistics.fmean(record["latency_ms"] for record in records)
                if records
                else None,
                "values": [record["latency_ms"] for record in records],
            },
            "prompt_tokens": sum(record["prompt_tokens"] for record in records),
            "completion_tokens": sum(record["completion_tokens"] for record in records),
            "automatic_actions": 0,
            "records": records,
        }

    def _evaluate(self, case: ReasoningCase) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(case.input, ensure_ascii=False)},
        ]
        try:
            result = self.runtime.generate_structured(
                messages,
                schema=reasoning_schema(),
                schema_name="agentic_reasoning_diagnostic",
                temperature=0.0,
                max_tokens=512,
            )
            actual = result.parsed
            observation = result.observation
        except LocalModelRuntimeError as exc:
            observation = exc.observation
            return {
                "case_id": case.case_id,
                "category": case.category,
                "structured_valid": False,
                "failure_code": exc.code.value,
                "factor_precision": 0.0,
                "factor_recall": 0.0,
                "factor_f1": 0.0,
                "step_recall": 0.0,
                "ordering_accuracy": 0.0,
                "verification_recall": 0.0,
                "uncertainty_recall": 0.0,
                "forbidden_factor_accuracy": 0.0,
                "latency_ms": observation.latency_ms if observation else 0.0,
                "prompt_tokens": observation.input_tokens or 0 if observation else 0,
                "completion_tokens": observation.output_tokens or 0 if observation else 0,
            }
        expected = case.expected
        factors = {name for name, enabled in actual["factors"].items() if enabled}
        required_factors = set(expected["required_factors"])
        factor_precision, factor_recall, factor_f1 = _set_metrics(factors, required_factors)
        kinds = [step["kind"] for step in actual["steps"]]
        required_steps = set(expected["required_steps"])
        step_recall = len(set(kinds) & required_steps) / len(required_steps)
        order_results = []
        for before, after in expected["ordered_pairs"]:
            order_results.append(before in kinds and after in kinds and kinds.index(before) < kinds.index(after))
        verification = {
            name for name, enabled in actual["verification"].items() if enabled
        }
        required_verification = set(expected["required_verification"])
        uncertainties = {
            name for name, enabled in actual["uncertainties"].items() if enabled
        }
        required_uncertainties = set(expected["required_uncertainties"])
        forbidden = set(expected["forbidden_factors"])
        return {
            "case_id": case.case_id,
            "category": case.category,
            "structured_valid": True,
            "failure_code": None,
            "factor_precision": factor_precision,
            "factor_recall": factor_recall,
            "factor_f1": factor_f1,
            "step_recall": step_recall,
            "ordering_accuracy": sum(order_results) / len(order_results) if order_results else 1.0,
            "verification_recall": (
                len(verification & required_verification) / len(required_verification)
                if required_verification
                else 1.0
            ),
            "uncertainty_recall": (
                len(uncertainties & required_uncertainties) / len(required_uncertainties)
                if required_uncertainties
                else 1.0
            ),
            "forbidden_factor_accuracy": 1.0 if factors.isdisjoint(forbidden) else 0.0,
            "actual": actual,
            "latency_ms": observation.latency_ms,
            "prompt_tokens": observation.input_tokens or 0,
            "completion_tokens": observation.output_tokens or 0,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run local model agentic reasoning evaluator")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    runtime = OpenAICompatibleLocalRuntime(
        LlamaClient(args.endpoint, timeout=args.timeout),
        RuntimeDescriptor(args.provider, args.model, args.runtime),
    )
    report = AgenticReasoningEvaluator(runtime).run(load_reasoning_cases(args.corpus))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["structured_validity"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
