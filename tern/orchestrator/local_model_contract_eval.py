from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .client import LlamaClient
from .local_model_runtime import (
    GenerationResult,
    LocalModelRuntime,
    LocalModelRuntimeError,
    OpenAICompatibleLocalRuntime,
    RuntimeDescriptor,
)
from .semantic_pass import semantic_json_schema
from .task_requirement_grounding import grounded_requirement_json_schema


@dataclass(frozen=True)
class StructuredContractCase:
    case_id: str
    schema: Mapping[str, Any]
    messages: tuple[Mapping[str, str], ...]
    max_tokens: int = 160


def _messages(instruction: str) -> tuple[Mapping[str, str], ...]:
    return (
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object and no surrounding text. "
                "Follow the requested field names and values exactly."
            ),
        },
        {"role": "user", "content": instruction},
    )


def progressive_contract_cases() -> tuple[StructuredContractCase, ...]:
    tri_state = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "repository_write": {
                "type": "string",
                "enum": ["TRUE", "FALSE", "UNKNOWN"],
            }
        },
        "required": ["repository_write"],
    }
    provenance = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "requirement": {"type": "string", "enum": ["repository_write"]},
            "value": {"type": "string", "enum": ["TRUE", "FALSE", "UNKNOWN"]},
            "source": {
                "type": "string",
                "enum": ["EXPLICIT_USER", "SEMANTIC_INFERENCE", "INSUFFICIENT_EVIDENCE"],
            },
            "evidence_ref": {"type": ["string", "null"], "maxLength": 80},
        },
        "required": ["requirement", "value", "source", "evidence_ref"],
    }
    semantic_schema = semantic_json_schema()["json_schema"]["schema"]
    grounding_schema = grounded_requirement_json_schema(
        ("repository_write", "test_execution", "web_access")
    )["json_schema"]["schema"]
    return (
        StructuredContractCase(
            "trivial_object",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
            _messages('Return {"ok":true}.'),
            32,
        ),
        StructuredContractCase(
            "enum",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"status": {"type": "string", "enum": ["READY", "BLOCKED"]}},
                "required": ["status"],
            },
            _messages('Return status "READY".'),
            32,
        ),
        StructuredContractCase(
            "optional_field",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "required_value": {"type": "integer"},
                    "optional_note": {"type": ["string", "null"]},
                },
                "required": ["required_value"],
            },
            _messages("Return required_value 7; optional_note may be null."),
            48,
        ),
        StructuredContractCase(
            "array",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 3,
                    }
                },
                "required": ["items"],
            },
            _messages("Return items [1,2]."),
            48,
        ),
        StructuredContractCase(
            "nested_object",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outer": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"inner": {"type": "string"}},
                        "required": ["inner"],
                    }
                },
                "required": ["outer"],
            },
            _messages('Return outer.inner as "value".'),
            48,
        ),
        StructuredContractCase(
            "tri_state",
            tri_state,
            _messages('Return repository_write "UNKNOWN".'),
            48,
        ),
        StructuredContractCase(
            "provenance",
            provenance,
            _messages(
                'Return requirement "repository_write", value "TRUE", source '
                '"EXPLICIT_USER", and evidence_ref "field:operation".'
            ),
            64,
        ),
        StructuredContractCase(
            "grounded_requirements",
            grounding_schema,
            _messages(
                "Return resolved as an empty array, target_scope as repository, "
                "risk_level as MEDIUM, and ambiguity_material as false."
            ),
            128,
        ),
        StructuredContractCase(
            "current_semantic_schema",
            semantic_schema,
            _messages(
                "Represent a direct informational answer with no execution, no agent, "
                "no target reference, no constraints, no continuation, no compound "
                "steps, no ambiguity, and confidence 1."
            ),
            320,
        ),
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class StructuredContractEvaluator:
    def __init__(self, runtime: LocalModelRuntime):
        self.runtime = runtime

    def run(
        self,
        cases: Iterable[StructuredContractCase] | None = None,
        *,
        repeats: int = 1,
    ) -> dict[str, Any]:
        selected = tuple(cases or progressive_contract_cases())
        records: list[dict[str, Any]] = []
        for repeat in range(repeats):
            for case in selected:
                try:
                    result = self.runtime.generate_structured(
                        [dict(message) for message in case.messages],
                        schema=case.schema,
                        schema_name=case.case_id,
                        temperature=0.0,
                        max_tokens=case.max_tokens,
                    )
                    records.append(self._success(case, repeat, result))
                except LocalModelRuntimeError as exc:
                    observation = exc.observation.as_dict() if exc.observation else None
                    records.append(
                        {
                            "case_id": case.case_id,
                            "repeat": repeat,
                            "valid": False,
                            "failure_code": exc.code.value,
                            "error": str(exc)[:1000],
                            "observation": observation,
                        }
                    )
        valid = [record for record in records if record["valid"]]
        latencies = [
            float(record["observation"]["latency_ms"])
            for record in records
            if record.get("observation")
        ]
        failures = Counter(
            record["failure_code"] for record in records if record.get("failure_code")
        )
        return {
            "case_count": len(selected),
            "request_count": len(records),
            "valid_count": len(valid),
            "validity": len(valid) / len(records) if records else 0.0,
            "failure_taxonomy": dict(sorted(failures.items())),
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "p50": _percentile(latencies, 0.50),
                "p90": _percentile(latencies, 0.90),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            },
            "prompt_tokens": sum(
                int(record["observation"].get("input_tokens") or 0)
                for record in records
                if record.get("observation")
            ),
            "completion_tokens": sum(
                int(record["observation"].get("output_tokens") or 0)
                for record in records
                if record.get("observation")
            ),
            "records": records,
        }

    @staticmethod
    def _success(
        case: StructuredContractCase,
        repeat: int,
        result: GenerationResult,
    ) -> dict[str, Any]:
        return {
            "case_id": case.case_id,
            "repeat": repeat,
            "valid": True,
            "failure_code": None,
            "parsed": result.parsed,
            "observation": result.observation.as_dict(),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local model structured contract evaluator")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--runtime-version")
    parser.add_argument("--model-path")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    runtime = OpenAICompatibleLocalRuntime(
        LlamaClient(args.endpoint, timeout=args.timeout),
        RuntimeDescriptor(
            provider=args.provider,
            model=args.model,
            runtime=args.runtime,
            expected_model_path=args.model_path,
            expected_runtime_version=args.runtime_version,
        ),
    )
    compatibility = runtime.assert_compatible()
    report = {
        "compatibility": compatibility,
        "model_info": runtime.model_info(),
        "runtime_info": runtime.runtime_info(),
        "contract": StructuredContractEvaluator(runtime).run(repeats=args.repeats),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["contract"]["validity"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

