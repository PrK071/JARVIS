from __future__ import annotations

from dataclasses import replace

import pytest

from tern.orchestrator.predictive.causal import build_causal_slice
from tern.orchestrator.predictive.evaluation import _retrieval, load_predictive_cases
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.recovery import (
    RecoveryBudget,
    RecoveryOutcome,
    StructuralRecoveryService,
    UnresolvedRelation,
)
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import (
    ProjectCandidateGenerator,
    ProjectIndexBuilderV2,
)
from tern.orchestrator.security import PathPolicy


def _case(identifier: str):
    return next(item for item in load_predictive_cases() if item.id == identifier)


def _initial(case):
    policy = PathPolicy((case.fixture_root,))
    snapshot = ProjectIndexBuilderV2(case.fixture_root, path_policy=policy).build()
    selection = ProjectCandidateGenerator().generate(case.problem, snapshot)
    context = build_problem_context(case.problem, snapshot, selection, policy)
    context = replace(
        context,
        evidence_ledger=build_evidence_ledger(context, snapshot, policy),
    )
    causal_slice, roots = build_causal_slice(context, snapshot, policy)
    return context, snapshot, policy, causal_slice, roots


def test_recovery_budget_rejects_negative_values():
    with pytest.raises(ValueError):
        RecoveryBudget(max_extra_files=-1)


def test_recovery_is_noop_when_origin_is_already_grounded():
    context, snapshot, policy, causal_slice, roots = _initial(_case("PC3D-003"))
    result = StructuralRecoveryService().recover(
        context, snapshot, policy, causal_slice, roots
    )

    assert result.trace.outcome is RecoveryOutcome.RECOVERY_NOT_NEEDED
    assert result.trace.actions == ()
    assert result.context.related_files == context.related_files


def test_recovery_resolves_unique_plain_symbol_seed():
    context, snapshot, policy, causal_slice, roots = _initial(_case("PC5H-010"))
    assert context.related_files == ()

    result = StructuralRecoveryService().recover(
        context, snapshot, policy, causal_slice, roots
    )

    assert result.trace.outcome is RecoveryOutcome.RECOVERY_SUCCEEDED
    assert result.context.related_files == ("pkg/metrics.py",)
    assert any(
        item.frontier.unresolved_relation is UnresolvedRelation.SYMBOL_REFERENCE_UNKNOWN
        for item in result.trace.actions
    )
    assert result.root_causes


def test_recovery_adds_missing_attribute_origin():
    context, snapshot, policy, causal_slice, roots = _initial(_case("PD-005"))
    result = StructuralRecoveryService().recover(
        context, snapshot, policy, causal_slice, roots
    )

    assert result.trace.outcome is RecoveryOutcome.RECOVERY_SUCCEEDED
    assert "pkg/models.py" in result.context.related_files
    assert any(
        item.frontier.unresolved_relation is UnresolvedRelation.ATTRIBUTE_ORIGIN_UNKNOWN
        for item in result.trace.actions
    )
    assert any(item.origin_path == "pkg/models.py" for item in result.root_causes)


def test_recovery_budget_zero_fails_closed():
    context, snapshot, policy, causal_slice, roots = _initial(_case("PC5H-010"))
    result = StructuralRecoveryService(
        RecoveryBudget(max_recovery_attempts=0)
    ).recover(context, snapshot, policy, causal_slice, roots)

    assert result.trace.outcome is RecoveryOutcome.RECOVERY_BUDGET_EXHAUSTED
    assert result.trace.actions == ()
    assert result.context.related_files == ()


def test_recovery_evidence_atom_budget_fails_closed():
    context, snapshot, policy, causal_slice, roots = _initial(_case("PR7D-011"))
    service = StructuralRecoveryService(RecoveryBudget(
        max_recovery_attempts=1,
        max_depth=2,
        max_extra_files=4,
        max_extra_symbols=12,
        max_extra_evidence_atoms=1,
    ))

    result = service.recover(
        context, snapshot, policy, causal_slice, roots,
    )

    assert result.trace.outcome is RecoveryOutcome.RECOVERY_BUDGET_EXHAUSTED
    assert result.trace.as_dict()["extra_evidence_atoms"] == 0
    assert result.context.related_files == context.related_files


def test_true_insufficient_evidence_does_not_expand_lexically():
    context, snapshot, policy, causal_slice, roots = _initial(_case("PD-028"))
    result = StructuralRecoveryService().recover(
        context, snapshot, policy, causal_slice, roots
    )

    assert result.trace.outcome in {
        RecoveryOutcome.TRUE_INSUFFICIENT_EVIDENCE,
        RecoveryOutcome.RECOVERY_NO_NEW_EVIDENCE,
    }
    assert result.trace.actions == ()


def test_recovery_order_and_ids_are_deterministic():
    first = _retrieval(_case("PD-005"))[0].recovery_trace.as_dict()
    second = _retrieval(_case("PD-005"))[0].recovery_trace.as_dict()

    assert first == second
