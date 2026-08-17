from dataclasses import replace

import pytest

from tern.orchestrator.agent_selection import (
    SelectionFactorType,
    SelectionPolicy,
    SelectionSource,
)
from tern.orchestrator.autonomy_foundation import Agent
from tern.orchestrator.execution_gate import (
    ExecutionAuthority,
    ExecutionBlockReason,
    ExecutionGate,
    ExecutionGateInput,
    ExecutionMode,
    MutationBlockReason,
)


ALL = (Agent.LOCAL, Agent.CODEX, Agent.DEEPSEEK)


def facts(**overrides) -> ExecutionGateInput:
    base = dict(
        execution_requested=True,
        candidate_agent=Agent.CODEX,
        selection_source=SelectionSource.DETERMINISTIC_SELECTION,
        eligible_agents=ALL,
        available_eligible_agents=ALL,
        selection_reason_code="DETERMINISTIC_FIT",
        selection_factors=(SelectionFactorType.IMPLEMENTATION_SUPPORT,),
        policy=SelectionPolicy(deepseek_auto_escalation=False),
    )
    base.update(overrides)
    return ExecutionGateInput(**base)


def evaluate(**overrides):
    return ExecutionGate().evaluate(facts(**overrides))


def test_gate_never_carries_live_authority():
    decision = evaluate()
    assert decision.authority is ExecutionAuthority.SHADOW
    assert decision.live_authority is False
    assert decision.mode == "SHADOW"
    assert (
        decision.delegations,
        decision.jobs_created,
        decision.sessions_resolved,
        decision.filesystem_mutations,
    ) == (0, 0, 0, 0)
    assert ExecutionGate.authority is ExecutionAuthority.SHADOW


def test_gate_is_deterministic():
    first = evaluate()
    second = evaluate()
    assert first == second


def test_execution_not_requested_blocks_even_with_a_candidate():
    decision = evaluate(execution_requested=False)
    assert decision.candidate_agent is Agent.CODEX
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.EXECUTION_NOT_REQUESTED
    assert decision.mutation_authorized is False


def test_no_eligible_agent_blocks():
    decision = evaluate(
        candidate_agent=None,
        selection_source=SelectionSource.NO_ELIGIBLE_AGENT,
        eligible_agents=(),
        available_eligible_agents=(),
        selection_reason_code="NO_ELIGIBLE_AGENT",
    )
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.NO_ELIGIBLE_AGENT
    assert decision.provenance_complete is True


def test_no_available_eligible_agent_blocks():
    decision = evaluate(
        candidate_agent=None,
        selection_source=SelectionSource.NO_AVAILABLE_ELIGIBLE_AGENT,
        eligible_agents=(Agent.CODEX,),
        available_eligible_agents=(),
        selection_reason_code="NO_AVAILABLE_ELIGIBLE_AGENT",
    )
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.NO_AVAILABLE_ELIGIBLE_AGENT


def test_unresolved_selection_fails_closed():
    decision = evaluate(
        candidate_agent=None,
        selection_source=SelectionSource.UNRESOLVED,
        selection_reason_code="AMBIGUOUS_REQUIREMENTS",
    )
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.SELECTION_UNRESOLVED
    assert decision.selection_valid is False


def test_invalid_selection_fails_closed():
    decision = evaluate(
        candidate_agent=None,
        selection_source=SelectionSource.INVALID_SELECTION,
        selection_reason_code="MODEL_NAMED_UNKNOWN_AGENT",
    )
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.SELECTION_UNRESOLVED


def test_ineligible_candidate_blocks():
    decision = evaluate(
        candidate_agent=Agent.CODEX,
        eligible_agents=(Agent.DEEPSEEK,),
        available_eligible_agents=(Agent.DEEPSEEK,),
    )
    assert decision.agent_eligible is False
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.INELIGIBLE_AGENT


def test_unavailable_candidate_blocks():
    decision = evaluate(
        candidate_agent=Agent.CODEX,
        eligible_agents=(Agent.CODEX, Agent.DEEPSEEK),
        available_eligible_agents=(Agent.DEEPSEEK,),
    )
    assert decision.agent_eligible is True
    assert decision.agent_available is False
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.AGENT_UNAVAILABLE


def test_explicit_agent_unavailable_keeps_the_requested_agent():
    decision = evaluate(
        selection_source=SelectionSource.EXPLICIT_USER,
        candidate_agent=Agent.CODEX,
        requested_agent=Agent.CODEX,
        requested_agent_source="EXPLICIT_USER",
        eligible_agents=(Agent.CODEX, Agent.DEEPSEEK),
        available_eligible_agents=(Agent.DEEPSEEK,),
        selection_reason_code="REQUESTED_AGENT_UNAVAILABLE",
    )
    assert decision.candidate_agent is Agent.CODEX
    assert decision.agent_available is False
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.REQUESTED_AGENT_UNAVAILABLE


def test_explicit_agent_ineligible_is_reported_as_such():
    decision = evaluate(
        selection_source=SelectionSource.EXPLICIT_USER,
        candidate_agent=Agent.DEEPSEEK,
        requested_agent=Agent.DEEPSEEK,
        eligible_agents=(Agent.CODEX,),
        available_eligible_agents=(Agent.CODEX,),
        selection_reason_code="REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS",
    )
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.REQUESTED_AGENT_INELIGIBLE


def test_deepseek_auto_selected_is_excluded_by_policy():
    decision = evaluate(
        candidate_agent=Agent.DEEPSEEK,
        selection_source=SelectionSource.SEMANTIC_MULTI_AGENT,
        selection_reason_code="BEST_FACTOR_FIT",
    )
    assert decision.agent_eligible is True
    assert decision.agent_available is True
    assert decision.execution_allowed is False
    assert (
        decision.block_reason
        is ExecutionBlockReason.POLICY_EXCLUDED_FROM_AUTO_SELECTION
    )


def test_deepseek_explicit_request_survives_the_auto_policy():
    decision = evaluate(
        candidate_agent=Agent.DEEPSEEK,
        requested_agent=Agent.DEEPSEEK,
        requested_agent_source="EXPLICIT_USER",
        selection_source=SelectionSource.EXPLICIT_USER,
        selection_reason_code="EXPLICIT_AGENT_READY",
    )
    assert decision.execution_allowed is True
    assert decision.block_reason is None


def test_deepseek_auto_selection_allowed_when_policy_enables_it():
    decision = evaluate(
        candidate_agent=Agent.DEEPSEEK,
        selection_source=SelectionSource.DETERMINISTIC_SELECTION,
        policy=SelectionPolicy(deepseek_auto_escalation=True),
    )
    assert decision.execution_allowed is True


def test_read_only_execution_is_allowed_without_mutation():
    decision = evaluate(read_only_required=True, mutation_requested=False)
    assert decision.execution_allowed is True
    assert decision.execution_mode is ExecutionMode.READ_ONLY
    assert decision.mutation_requested is False
    assert decision.mutation_authorized is False
    assert decision.mutation_block_reason is MutationBlockReason.MUTATION_NOT_REQUESTED


def test_mutation_requested_and_authorized():
    decision = evaluate(mutation_requested=True, agent_can_mutate=True)
    assert decision.execution_allowed is True
    assert decision.mutation_authorized is True
    assert decision.execution_mode is ExecutionMode.MUTATION


def test_mutation_requested_but_agent_cannot_mutate():
    decision = evaluate(mutation_requested=True, agent_can_mutate=False)
    assert decision.execution_allowed is True
    assert decision.mutation_authorized is False
    assert decision.mutation_block_reason is MutationBlockReason.AGENT_CANNOT_MUTATE
    assert decision.execution_mode is ExecutionMode.READ_ONLY


def test_mutation_requested_but_confirmation_required():
    decision = evaluate(
        mutation_requested=True,
        agent_can_mutate=True,
        confirmation_required=True,
    )
    assert decision.execution_allowed is True
    assert decision.mutation_authorized is False
    assert decision.mutation_block_reason is MutationBlockReason.CONFIRMATION_REQUIRED


def test_mutation_requested_but_path_policy_unsatisfied():
    decision = evaluate(
        mutation_requested=True,
        agent_can_mutate=True,
        path_policy_satisfied=False,
    )
    assert decision.mutation_authorized is False
    assert (
        decision.mutation_block_reason
        is MutationBlockReason.PATH_POLICY_NOT_SATISFIED
    )


def test_forbid_mutation_constraint_wins_over_request():
    decision = evaluate(
        mutation_requested=True,
        agent_can_mutate=True,
        forbid_mutation=True,
    )
    assert decision.mutation_authorized is False
    assert (
        decision.mutation_block_reason
        is MutationBlockReason.MUTATION_FORBIDDEN_BY_CONSTRAINT
    )


def test_read_only_task_blocks_mutation_even_if_requested():
    decision = evaluate(
        mutation_requested=True,
        agent_can_mutate=True,
        read_only_required=True,
    )
    assert decision.mutation_authorized is False
    assert decision.mutation_block_reason is MutationBlockReason.READ_ONLY_TASK


def test_unresolved_mutation_requirement_never_authorizes():
    decision = evaluate(
        mutation_requested=True,
        mutation_requirement_unresolved=True,
        agent_can_mutate=True,
    )
    assert decision.mutation_authorized is False
    assert (
        decision.mutation_block_reason
        is MutationBlockReason.MUTATION_REQUIREMENT_UNRESOLVED
    )


def test_execution_safety_unresolved_blocks():
    decision = evaluate(
        execution_safe=False,
        unresolved_safety_requirements=("mutation_required",),
    )
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.EXECUTION_SAFETY_UNRESOLVED


def test_legacy_constraint_violation_blocks():
    decision = evaluate(constraint_violation="FORBID_DELEGATION")
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.CONSTRAINT_VIOLATION


def test_missing_provenance_blocks():
    decision = evaluate(selection_reason_code="")
    assert decision.execution_allowed is False
    assert (
        decision.block_reason is ExecutionBlockReason.SELECTION_PROVENANCE_MISSING
    )
    assert decision.provenance_complete is False


def test_blocked_execution_never_authorizes_mutation():
    for override in (
        {"execution_requested": False},
        {"eligible_agents": (), "available_eligible_agents": ()},
        {"available_eligible_agents": ()},
        {"execution_safe": False},
        {"constraint_violation": "FORBID_MUTATION"},
    ):
        decision = evaluate(
            mutation_requested=True,
            agent_can_mutate=True,
            **override,
        )
        assert decision.execution_allowed is False
        assert decision.mutation_authorized is False
        assert decision.execution_mode is ExecutionMode.READ_ONLY


def test_all_block_reasons_are_collected_not_only_the_first():
    decision = evaluate(
        execution_requested=False,
        eligible_agents=(),
        available_eligible_agents=(),
        candidate_agent=None,
        selection_source=SelectionSource.NO_ELIGIBLE_AGENT,
        selection_reason_code="NO_ELIGIBLE_AGENT",
    )
    assert decision.block_reasons == (
        ExecutionBlockReason.EXECUTION_NOT_REQUESTED,
        ExecutionBlockReason.NO_ELIGIBLE_AGENT,
    )


def test_selection_supported_requires_factors_or_deterministic_source():
    unsupported = evaluate(
        selection_factors=(),
        selection_source=SelectionSource.SEMANTIC_MULTI_AGENT,
        selection_reason_code="EQUAL_FIT",
    )
    assert unsupported.selection_supported is False
    supported = evaluate(
        selection_factors=(),
        selection_source=SelectionSource.SINGLE_ELIGIBLE_AGENT,
        selection_reason_code="SINGLE_ELIGIBLE_AGENT",
    )
    assert supported.selection_supported is True


def test_gate_input_is_frozen():
    value = facts()
    with pytest.raises(Exception):
        value.execution_requested = False  # type: ignore[misc]
    assert replace(value, execution_requested=False).execution_requested is False


def test_decision_dict_never_advertises_authority():
    payload = evaluate().as_dict()
    assert payload["mode"] == "SHADOW"
    assert payload["authority"] == "SHADOW"
    assert payload["live_authority"] is False
    assert payload["delegations"] == 0
    assert payload["jobs_created"] == 0
    assert payload["sessions_resolved"] == 0
    assert payload["filesystem_mutations"] == 0
