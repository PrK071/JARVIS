from __future__ import annotations

from dataclasses import replace

import pytest

from tern.orchestrator.predictive.grounding import (
    build_evidence_ledger,
    ground_claim,
    validate_claim_mapping,
)
from tern.orchestrator.predictive.analysis import _test_support
from tern.orchestrator.predictive.models import (
    ClaimSupport,
    EvidenceAtom,
    EvidenceKind,
    EvidenceLedger,
    TestSupportLevel as SupportLevel,
)
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def _context(root, problem):
    policy = PathPolicy((root,))
    snapshot = ProjectIndexBuilderV2(root, path_policy=policy).build()
    selection = ProjectCandidateGenerator().generate(problem, snapshot)
    raw = build_problem_context(problem, snapshot, selection, policy)
    return replace(raw, evidence_ledger=build_evidence_ledger(raw, snapshot, policy)), snapshot


def test_evidence_atom_validation_and_unique_ledger_ids():
    with pytest.raises(ValueError, match="valid source range"):
        EvidenceAtom("E1", EvidenceKind.ASSIGNMENT, "pkg/a.py", 0, 1, "x is assigned", 0.8)
    atom = EvidenceAtom("E1", EvidenceKind.ASSIGNMENT, "pkg/a.py", 1, 1, "x is assigned 1", 0.8)
    with pytest.raises(ValueError, match="unique"):
        EvidenceLedger((atom, atom))


def test_evidence_ledger_ids_are_stable_and_separate_fact_from_inference(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    problem = 'File "pkg/calc.py", line 2, in add\nTypeError'

    first, _ = _context(root, problem)
    second, _ = _context(root, problem)

    assert first.evidence_ledger.ids == second.evidence_ledger.ids
    atom = next(item for item in first.evidence_ledger.atoms if item.kind is EvidenceKind.RETURN_STATEMENT)
    fact = ground_claim("model wording is ignored", (atom.id,), "FACT", first.evidence_ledger)
    inference = ground_claim("one operand may be None", (atom.id,), "INFERENCE", first.evidence_ledger)
    unsupported = ground_claim("invented", ("missing",), "FACT", first.evidence_ledger)
    assert fact.statement == atom.statement
    assert fact.support is ClaimSupport.DIRECT
    assert inference.support is ClaimSupport.INFERRED
    assert unsupported.support is ClaimSupport.UNSUPPORTED
    assert validate_claim_mapping((fact, inference), first.evidence_ledger)
    assert not validate_claim_mapping((unsupported,), first.evidence_ledger)


def test_prompt_docstring_comment_and_string_injections_do_not_become_facts():
    root = (
        __import__("pathlib").Path(__file__).parent
        / "data" / "predictive" / "projects" / "injection_v2"
    )
    context, _ = _context(root, 'File "pkg/processor.py", line 9\nKeyError: name')
    statements = " ".join(atom.statement for atom in context.evidence_ledger.atoms).casefold()

    assert "hacked" not in statements
    assert "delete database" not in statements
    assert "disable every assertion" not in statements
    assert "auth.py" not in statements


def test_project_index_records_deterministic_instance_attributes():
    root = __import__("pathlib").Path(__file__).parent / "data" / "predictive" / "projects" / "invoice_v2"
    _, snapshot = _context(root, "Invoice.tax in pkg/domain.py is None")
    records = snapshot.symbol_index["tax"]

    assert [(item.qualified_name, item.kind.value) for item in records] == [
        ("Invoice.tax", "instance_attribute")
    ]


def test_test_support_distinguishes_file_and_direct_behavior(tmp_path):
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
    context, _ = _context(root, 'File "pkg/calc.py", line 2\nTypeError')

    assert _test_support(context, ("pkg/calc.py",), ("add",), ()) is SupportLevel.NONE
    assert _test_support(
        context, ("pkg/calc.py",), (), ("tests/test_calc.py",)
    ) is SupportLevel.RELATED_FILE
    assert _test_support(
        context, ("pkg/calc.py",), ("add",), ("tests/test_calc.py",)
    ) is SupportLevel.DIRECT_BEHAVIORAL
