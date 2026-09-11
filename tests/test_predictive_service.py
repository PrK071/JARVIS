from __future__ import annotations

import json
from pathlib import Path

from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.security import PathPolicy


def response(value):
    return {"choices": [{"message": {"content": json.dumps(value)}}]}


class FakeReasoner:
    def chat(self, messages, **_kwargs):
        payload = json.loads(messages[-1]["content"])
        root = next(
            item for item in payload["root_cause_candidates"]
            if item["origin_path"] == "pkg/calc.py" and item["cause_kind"] == "RETURN_CONTRACT"
        )
        return response(
            {"selections": [{
                "root_cause_id": root["id"],
                "claim": "an operand may be None",
                "strategies": [{
                    "kind": "VALIDATE_BOUNDARY",
                    "target_file": "pkg/calc.py",
                    "target_symbol": "add",
                    "rationale": "Validate operands before addition",
                }],
            }]}
        )


def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (root / "tests" / "test_calc.py").write_text(
        "from pkg.calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return root


def file_state(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_service_reuses_project_intelligence_and_never_mutates_project(tmp_path):
    root = repository(tmp_path)
    before = file_state(root)
    service = PredictiveDecisionService(
        FakeReasoner(), path_policy=PathPolicy((root,))
    )
    report = service.predict(
        'Traceback\n  File "pkg/calc.py", line 2, in add\nTypeError: unsupported operand type(s)',
        root,
    )

    assert report.insufficient_evidence is False
    assert report.recommended_candidate_id == "C1"
    assert "pkg/calc.py:2-2" in report.hypotheses[0].evidence_refs
    assert report.requires_approval is True
    assert report.execution_authorized is False
    assert file_state(root) == before


def test_service_returns_insufficient_evidence_without_calling_model(tmp_path):
    root = repository(tmp_path)

    class ForbiddenReasoner:
        def chat(self, *_args, **_kwargs):
            raise AssertionError("model must not be called without strong evidence")

    report = PredictiveDecisionService(
        ForbiddenReasoner(), path_policy=PathPolicy((root,))
    ).predict("something is wrong", root)

    assert report.insufficient_evidence is True
    assert report.recommended_candidate_id is None
    assert report.candidates == ()
    assert report.failure_reason.value == "INSUFFICIENT_STRUCTURAL_EVIDENCE"


def test_service_preserves_reasoner_unavailable_diagnostic(tmp_path):
    root = repository(tmp_path)

    class UnavailableReasoner:
        def chat(self, *_args, **_kwargs):
            raise OSError("offline")

    report = PredictiveDecisionService(
        UnavailableReasoner(), path_policy=PathPolicy((root,))
    ).predict('File "pkg/calc.py", line 2\nTypeError', root)

    assert report.insufficient_evidence is True
    assert report.failure_reason.value == "REASONER_UNAVAILABLE"
    assert "indisponível" in report.recommendation_explanation


def test_forbidden_candidate_is_rejected_and_never_recommended(tmp_path):
    root = repository(tmp_path)

    class UnsafeReasoner(FakeReasoner):
        def chat(self, messages, **kwargs):
            value = super().chat(messages, **kwargs)
            content = json.loads(value["choices"][0]["message"]["content"])
            content["selections"][0]["strategies"][0]["rationale"] = "Delete config.py and bypass validation"
            return response(content)

    report = PredictiveDecisionService(
        UnsafeReasoner(), path_policy=PathPolicy((root,))
    ).predict('File "pkg/calc.py", line 2\nTypeError', root)

    assert report.recommended_candidate_id is None
    assert report.candidates == ()
    assert report.failure_reason.value == "FORBIDDEN_CANDIDATE"
    assert report.rejected_candidates[0].eligible is False
    assert "FORBIDDEN_CANDIDATE" in report.rejected_candidates[0].rejection_reasons


def test_predictive_json_v3_keeps_legacy_report_fields(tmp_path):
    root = repository(tmp_path)
    report = PredictiveDecisionService(
        FakeReasoner(), path_policy=PathPolicy((root,))
    ).predict('File "pkg/calc.py", line 2\nTypeError', root).as_dict()

    assert report["schema_version"] == 5
    assert "root_cause_selections" in report
    assert {
        "problem", "hypotheses", "candidates", "recommended_candidate_id",
        "insufficient_evidence", "requires_approval",
    }.issubset(report)
    assert {"causal_slice", "root_cause_candidates", "repair_strategies"}.issubset(report)
    assert report["execution_authorized"] is False
