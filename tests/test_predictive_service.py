from __future__ import annotations

import json
from pathlib import Path

from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.security import PathPolicy


def response(value):
    return {"choices": [{"message": {"content": json.dumps(value)}}]}


class FakeReasoner:
    def __init__(self):
        self.values = iter(
            [
                response(
                    {
                        "hypotheses": [
                            {
                                "id": "H1",
                                "statement": "add receives None",
                                "confidence_level": "HIGH",
                                "evidence_refs": ["pkg/calc.py:1-2"],
                            }
                        ]
                    }
                ),
                response(
                    {
                        "candidates": [
                            {
                                "id": "C1",
                                "hypothesis_id": "H1",
                                "action": "Validate operands before addition",
                                "expected_outcome": "None is rejected",
                                "evidence_refs": ["pkg/calc.py:1-2"],
                                "required_tests": ["tests/test_calc.py"],
                                "risk_level": "LOW",
                                "cost_level": "LOW",
                                "reversibility_level": "HIGH",
                            }
                        ]
                    }
                ),
            ]
        )

    def chat(self, _messages, **_kwargs):
        return next(self.values)


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
    assert report.hypotheses[0].evidence_refs == ("pkg/calc.py:1-2",)
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
