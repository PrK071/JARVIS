"""Dry-run tests for automatic agent selection and selection provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from tern.orchestrator.agent_selection import (
    AgentSelectionEngine,
    AgentSelectionProfileBuilder,
    OperationalFacts,
    SelectionConfidence,
    SelectionFactorType,
    SelectionPolicy,
    SelectionSource,
    SemanticAgentSelector,
    derive_task_factors,
)
from tern.orchestrator.agent_selection_eval import (
    SelectionEvaluator,
    availability_map,
    grounded_requirements_from_labels,
    load_selection_cases,
)
from tern.orchestrator.autonomy_eval import diagnostic_baseline
from tern.orchestrator.autonomy_foundation import Agent, Capability
from tern.orchestrator.codex_sessions import CodexSessionResolver
from tern.orchestrator.local_model_runtime import (
    LocalModelRuntimeError,
    RuntimeFailureCode,
)
from tern.orchestrator.task_requirement_grounding import GroundedEligibilityEngine


CORPUS = Path("tests/data/agent_selection_diagnostic.jsonl")

READ_ONLY_REPOSITORY = {
    "capabilities": ["repository_read", "code_analysis", "read_only_capable"],
    "mutation_required": False,
    "read_only_required": True,
    "target_scope": "repository",
    "risk_level": "low",
}
MUTATION_REPOSITORY = {
    "capabilities": [
        "repository_read",
        "repository_write",
        "code_analysis",
        "code_edit",
        "mutation_capable",
    ],
    "mutation_required": True,
    "read_only_required": False,
    "target_scope": "repository",
    "risk_level": "medium",
}
MUTATION_WITH_TESTS = {
    "capabilities": [
        "repository_read",
        "repository_write",
        "code_analysis",
        "code_edit",
        "test_execution",
        "mutation_capable",
    ],
    "mutation_required": True,
    "read_only_required": False,
    "target_scope": "repository",
    "risk_level": "medium",
    "tests_requested": ["relevant tests"],
}
INFORMATIONAL = {
    "capabilities": ["general_reasoning", "code_analysis", "read_only_capable"],
    "mutation_required": False,
    "read_only_required": True,
    "target_scope": "provided context",
    "risk_level": "low",
}
WEB_RESEARCH = {
    "capabilities": ["web_access", "general_reasoning", "read_only_capable"],
    "mutation_required": False,
    "read_only_required": True,
    "target_scope": "web",
    "risk_level": "low",
}
IMPOSSIBLE = {
    "capabilities": [
        "web_access",
        "repository_write",
        "code_edit",
        "test_execution",
        "mutation_capable",
    ],
    "mutation_required": True,
    "read_only_required": False,
    "target_scope": "repository and web",
    "risk_level": "medium",
    "tests_requested": ["relevant tests"],
}


class RecordingSelector:
    """Semantic selector stub: records every call and returns a fixed decision."""

    def __init__(self, decision: str = "UNRESOLVED", factors: Sequence[str] = ()):
        self.decision = decision
        self.factors = list(factors)
        self.payloads: list[Mapping[str, Any]] = []
        self.schemas: list[Mapping[str, Any]] = []
        self.messages: list[list[dict[str, Any]]] = []

    def generate_structured(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        temperature: float = 0.0,
        max_tokens: int = 160,
        semantic_validator: Any = None,
    ) -> Any:
        self.messages.append(messages)
        self.payloads.append(json.loads(messages[-1]["content"]))
        self.schemas.append(schema)
        content = {
            "proposed_agent": self.decision,
            "selection_factors": self.factors,
            "uncertainty": "LOW",
            "reason_code": "BEST_FACTOR_FIT",
        }
        return type(
            "Result",
            (),
            {"content": json.dumps(content), "parsed": content, "observation": None},
        )()

    @property
    def calls(self) -> int:
        return len(self.payloads)


class FailingSelector(RecordingSelector):
    def generate_structured(self, messages, **kwargs):  # type: ignore[override]
        self.payloads.append({})
        raise LocalModelRuntimeError(
            RuntimeFailureCode.MALFORMED_JSON,
            "structured response is not one JSON value",
        )


def propose(
    labels: Mapping[str, Any],
    *,
    availability: Mapping[Agent, bool] | None = None,
    operational: OperationalFacts | None = None,
    policy: SelectionPolicy | None = None,
    requested_agent: Agent | None = None,
    selector: Any | None = None,
):
    baseline = diagnostic_baseline(deepseek_available=True)
    requirements = grounded_requirements_from_labels(labels)
    states = availability_map(
        baseline.availability, availability or {agent: True for agent in Agent}
    )
    evaluations = GroundedEligibilityEngine().evaluate(
        requirements, baseline.profiles, states
    )
    engine = AgentSelectionEngine(
        policy=policy or SelectionPolicy(),
        semantic_selector=SemanticAgentSelector(selector) if selector is not None else None,
    )
    return engine.propose(
        requirements,
        evaluations,
        capability_profiles=baseline.profiles,
        availability=states,
        operational=operational or OperationalFacts(),
        requested_agent=requested_agent,
    )


# --- explicit agent binding stays sovereign -------------------------------


@pytest.mark.parametrize(
    "labels,agent",
    [(MUTATION_WITH_TESTS, Agent.CODEX), (INFORMATIONAL, Agent.DEEPSEEK)],
)
def test_explicit_agent_is_proposed_without_inference(labels, agent):
    selector = RecordingSelector("local")
    proposal = propose(labels, requested_agent=agent, selector=selector)
    assert proposal.proposed_agent is agent
    assert proposal.selection_source is SelectionSource.EXPLICIT_USER
    assert proposal.reason_code == "EXPLICIT_AGENT_READY"
    assert proposal.model_calls == 0
    assert selector.calls == 0


@pytest.mark.parametrize(
    "labels,agent",
    [(MUTATION_REPOSITORY, Agent.CODEX), (INFORMATIONAL, Agent.DEEPSEEK)],
)
def test_unavailable_explicit_agent_is_not_replaced(labels, agent):
    selector = RecordingSelector("local")
    proposal = propose(
        labels,
        availability={item: item is not agent for item in Agent},
        requested_agent=agent,
        selector=selector,
    )
    assert proposal.proposed_agent is agent
    assert proposal.selection_source is SelectionSource.EXPLICIT_USER
    assert proposal.reason_code == "REQUESTED_AGENT_UNAVAILABLE"
    assert proposal.execution_possible is False
    assert agent in proposal.eligible_agents
    assert agent in proposal.eligible_but_unavailable
    assert selector.calls == 0


def test_explicit_ineligible_agent_reports_requirement_gap_without_fallback():
    proposal = propose(MUTATION_WITH_TESTS, requested_agent=Agent.DEEPSEEK)
    assert proposal.proposed_agent is Agent.DEEPSEEK
    assert proposal.reason_code == "REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS"
    assert proposal.execution_possible is False
    assert Agent.DEEPSEEK not in proposal.eligible_agents
    assert proposal.eligible_agents == (Agent.CODEX,)


# --- deterministic paths --------------------------------------------------


def test_single_eligible_codex_uses_zero_inference():
    selector = RecordingSelector("local")
    proposal = propose(MUTATION_WITH_TESTS, selector=selector)
    assert proposal.eligible_agents == (Agent.CODEX,)
    assert proposal.proposed_agent is Agent.CODEX
    assert proposal.selection_source is SelectionSource.SINGLE_ELIGIBLE_AGENT
    assert proposal.model_calls == 0
    assert selector.calls == 0


def test_single_eligible_local_uses_zero_inference():
    selector = RecordingSelector("codex")
    proposal = propose(WEB_RESEARCH, selector=selector)
    assert proposal.eligible_agents == (Agent.LOCAL,)
    assert proposal.proposed_agent is Agent.LOCAL
    assert proposal.selection_source is SelectionSource.SINGLE_ELIGIBLE_AGENT
    assert selector.calls == 0


def test_no_eligible_agent_never_invents_one():
    selector = RecordingSelector("codex")
    proposal = propose(IMPOSSIBLE, selector=selector)
    assert proposal.eligible_agents == ()
    assert proposal.proposed_agent is None
    assert proposal.selection_source is SelectionSource.NO_ELIGIBLE_AGENT
    assert selector.calls == 0


def test_unique_justified_candidate_resolves_without_inference():
    selector = RecordingSelector("codex")
    proposal = propose(READ_ONLY_REPOSITORY, selector=selector)
    assert set(proposal.eligible_agents) == {Agent.LOCAL, Agent.CODEX}
    assert proposal.proposed_agent is Agent.LOCAL
    assert proposal.selection_source is SelectionSource.DETERMINISTIC_SELECTION
    assert proposal.reason_code == "UNIQUE_JUSTIFIED_CANDIDATE"
    assert selector.calls == 0


def test_policy_restricted_candidates_are_deterministic():
    selector = RecordingSelector("deepseek")
    proposal = propose(INFORMATIONAL, selector=selector)
    assert set(proposal.eligible_agents) == {Agent.LOCAL, Agent.CODEX, Agent.DEEPSEEK}
    assert Agent.DEEPSEEK in proposal.excluded_by_policy
    assert proposal.proposed_agent is Agent.LOCAL
    assert proposal.selection_source is SelectionSource.DETERMINISTIC_SELECTION
    assert selector.calls == 0


def test_ambiguous_requirements_stay_unresolved_without_inference():
    selector = RecordingSelector("codex")
    proposal = propose(
        {
            "capabilities": [],
            "mutation_required": False,
            "read_only_required": False,
            "unknown_dimensions": ["mutation_required"],
            "target_scope": "unknown",
            "risk_level": "medium",
            "ambiguity_material": True,
        },
        selector=selector,
    )
    assert proposal.proposed_agent is None
    assert proposal.selection_source is SelectionSource.UNRESOLVED
    assert proposal.reason_code == "AMBIGUOUS_REQUIREMENTS"
    assert selector.calls == 0


# --- availability ---------------------------------------------------------


@pytest.mark.parametrize(
    "availability,expected",
    [
        ({Agent.LOCAL: True, Agent.CODEX: True, Agent.DEEPSEEK: True}, Agent.LOCAL),
        ({Agent.LOCAL: False, Agent.CODEX: True, Agent.DEEPSEEK: True}, Agent.CODEX),
        ({Agent.LOCAL: True, Agent.CODEX: False, Agent.DEEPSEEK: True}, Agent.LOCAL),
    ],
)
def test_availability_selects_only_available_eligible_agents(availability, expected):
    proposal = propose(READ_ONLY_REPOSITORY, availability=availability)
    assert set(proposal.eligible_agents) == {Agent.LOCAL, Agent.CODEX}
    assert proposal.proposed_agent is expected
    assert proposal.proposed_agent in proposal.available_eligible_agents


def test_no_available_eligible_agent_is_reported_without_selection():
    proposal = propose(
        READ_ONLY_REPOSITORY,
        availability={Agent.LOCAL: False, Agent.CODEX: False, Agent.DEEPSEEK: True},
    )
    assert set(proposal.eligible_agents) == {Agent.LOCAL, Agent.CODEX}
    assert proposal.available_eligible_agents == ()
    assert proposal.proposed_agent is None
    assert proposal.selection_source is SelectionSource.NO_AVAILABLE_ELIGIBLE_AGENT
    assert set(proposal.eligible_but_unavailable) == {Agent.LOCAL, Agent.CODEX}


def test_availability_never_changes_eligibility():
    reference = propose(READ_ONLY_REPOSITORY).eligible_agents
    for agent in Agent:
        variant = propose(
            READ_ONLY_REPOSITORY,
            availability={item: item is not agent for item in Agent},
        )
        assert variant.eligible_agents == reference


# --- semantic multi-agent selection --------------------------------------


def test_two_candidates_use_exactly_one_inference():
    selector = RecordingSelector("codex", ["IMPLEMENTATION_SUPPORT"])
    proposal = propose(MUTATION_REPOSITORY, selector=selector)
    assert set(proposal.candidate_agents) == {Agent.LOCAL, Agent.CODEX}
    assert proposal.proposed_agent is Agent.CODEX
    assert proposal.selection_source is SelectionSource.SEMANTIC_MULTI_AGENT
    assert proposal.model_calls == 1
    assert selector.calls == 1
    assert proposal.errors == ()


def test_three_eligible_agents_only_expose_justified_candidates():
    selector = RecordingSelector("deepseek", ["STRUCTURAL_READ_ONLY_GUARANTEE"])
    proposal = propose(
        INFORMATIONAL,
        policy=SelectionPolicy(deepseek_auto_escalation=True),
        selector=selector,
    )
    assert set(proposal.eligible_agents) == {Agent.LOCAL, Agent.CODEX, Agent.DEEPSEEK}
    assert set(proposal.candidate_agents) == {Agent.LOCAL, Agent.DEEPSEEK}
    assert Agent.CODEX not in proposal.candidate_agents
    allowed = selector.schemas[0]["properties"]["proposed_agent"]["enum"]
    assert set(allowed) == {"local", "deepseek", "UNRESOLVED"}
    assert proposal.proposed_agent is Agent.DEEPSEEK


def test_model_may_answer_unresolved():
    selector = RecordingSelector("UNRESOLVED")
    proposal = propose(MUTATION_REPOSITORY, selector=selector)
    assert proposal.proposed_agent is None
    assert proposal.selection_source is SelectionSource.UNRESOLVED
    assert proposal.reason_code == "SEMANTIC_UNRESOLVED"
    assert proposal.model_calls == 1


def test_agent_outside_candidate_set_is_rejected_without_substitution():
    selector = RecordingSelector("deepseek")
    engine = AgentSelectionEngine(semantic_selector=SemanticAgentSelector(selector))
    baseline = diagnostic_baseline()
    requirements = grounded_requirements_from_labels(MUTATION_REPOSITORY)
    states = availability_map(baseline.availability, {agent: True for agent in Agent})
    evaluations = GroundedEligibilityEngine().evaluate(
        requirements, baseline.profiles, states
    )
    # bypass the constrained schema to prove the deterministic guard
    engine.semantic_selector.select = lambda *args, **kwargs: type(  # type: ignore[assignment]
        "Outcome",
        (),
        {
            "proposed_agent": Agent.DEEPSEEK,
            "factors": (),
            "uncertainty": "LOW",
            "reason_code": "BEST_FACTOR_FIT",
            "latency_ms": 1.0,
            "calls": 1,
            "valid": True,
            "failure_code": None,
            "raw": {},
        },
    )()
    proposal = engine.propose(
        requirements,
        evaluations,
        capability_profiles=baseline.profiles,
        availability=states,
        operational=OperationalFacts(),
    )
    assert proposal.proposed_agent is None
    assert proposal.selection_source is SelectionSource.INVALID_SELECTION
    assert "INELIGIBLE_AGENT_SELECTED" in proposal.errors


def test_model_failure_becomes_unresolved():
    selector = FailingSelector()
    proposal = propose(MUTATION_REPOSITORY, selector=selector)
    assert proposal.proposed_agent is None
    assert proposal.selection_source is SelectionSource.UNRESOLVED
    assert proposal.reason_code == "MODEL_PARSE_FAILURE"
    assert proposal.model_calls == 1


def test_selector_never_receives_the_raw_task_text():
    selector = RecordingSelector("local")
    baseline = diagnostic_baseline()
    requirements = grounded_requirements_from_labels(MUTATION_REPOSITORY)
    states = availability_map(baseline.availability, {agent: True for agent in Agent})
    evaluations = GroundedEligibilityEngine().evaluate(
        requirements, baseline.profiles, states
    )
    AgentSelectionEngine(
        semantic_selector=SemanticAgentSelector(selector)
    ).propose(
        requirements,
        evaluations,
        capability_profiles=baseline.profiles,
        availability=states,
    )
    serialized = json.dumps(selector.payloads[0], ensure_ascii=False)
    for word in ("corrija", "bug", "revise", "teste", "implemente"):
        assert word not in serialized.casefold()


def test_selection_is_stable_for_identical_facts():
    decisions = set()
    for _ in range(3):
        selector = RecordingSelector("codex", ["IMPLEMENTATION_SUPPORT"])
        proposal = propose(
            MUTATION_REPOSITORY,
            operational=OperationalFacts(reusable_codex_session=True),
            selector=selector,
        )
        decisions.add(proposal.proposed_agent)
        assert json.dumps(selector.payloads[0], sort_keys=True) == json.dumps(
            selector.payloads[0], sort_keys=True
        )
    assert decisions == {Agent.CODEX}


# --- factors and provenance ----------------------------------------------


def test_task_factors_cite_requirement_evidence():
    factors = derive_task_factors(grounded_requirements_from_labels(MUTATION_WITH_TESTS))
    kinds = {item.type for item in factors}
    assert SelectionFactorType.MUTATION_REQUIRED in kinds
    assert SelectionFactorType.TEST_EXECUTION_REQUIRED in kinds
    assert SelectionFactorType.REPOSITORY_SCOPE_REQUIRED in kinds
    assert all(item.evidence for item in factors)


def test_capability_alone_is_not_a_preference():
    baseline = diagnostic_baseline()
    requirements = grounded_requirements_from_labels(READ_ONLY_REPOSITORY)
    profiles = AgentSelectionProfileBuilder().build(
        requirements,
        agents=tuple(Agent),
        profiles=baseline.profiles,
        policy=SelectionPolicy(),
        operational=OperationalFacts(),
    )
    assert baseline.profiles[Agent.CODEX].has(Capability.CODE_REVIEW)
    assert baseline.profiles[Agent.CODEX].has(Capability.PERSISTENT_SESSION)
    assert profiles[Agent.CODEX].support_factors == ()
    assert profiles[Agent.LOCAL].justified


def test_existing_session_and_affinity_are_operational_factors():
    baseline = diagnostic_baseline()
    requirements = grounded_requirements_from_labels(READ_ONLY_REPOSITORY)
    profiles = AgentSelectionProfileBuilder().build(
        requirements,
        agents=tuple(Agent),
        profiles=baseline.profiles,
        policy=SelectionPolicy(),
        operational=OperationalFacts(
            reusable_codex_session=True, codex_project_affinity=True
        ),
    )
    kinds = {item.type for item in profiles[Agent.CODEX].support_factors}
    assert kinds == {
        SelectionFactorType.EXISTING_REUSABLE_SESSION,
        SelectionFactorType.PROJECT_AFFINITY,
    }


def test_structural_read_only_guarantee_only_for_write_incapable_agent():
    baseline = diagnostic_baseline()
    requirements = grounded_requirements_from_labels(INFORMATIONAL)
    profiles = AgentSelectionProfileBuilder().build(
        requirements,
        agents=tuple(Agent),
        profiles=baseline.profiles,
        policy=SelectionPolicy(deepseek_auto_escalation=True),
        operational=OperationalFacts(),
    )

    def kinds(agent: Agent) -> set[SelectionFactorType]:
        return {item.type for item in profiles[agent].support_factors}

    assert SelectionFactorType.STRUCTURAL_READ_ONLY_GUARANTEE in kinds(Agent.DEEPSEEK)
    assert SelectionFactorType.STRUCTURAL_READ_ONLY_GUARANTEE not in kinds(Agent.LOCAL)
    assert SelectionFactorType.STRUCTURAL_READ_ONLY_GUARANTEE not in kinds(Agent.CODEX)


def test_every_proposal_carries_provenance():
    cases = [
        (MUTATION_WITH_TESTS, None),
        (READ_ONLY_REPOSITORY, None),
        (IMPOSSIBLE, None),
        (INFORMATIONAL, Agent.DEEPSEEK),
    ]
    for labels, requested in cases:
        proposal = propose(labels, requested_agent=requested)
        value = proposal.as_dict()
        assert value["selection_source"] in {item.value for item in SelectionSource}
        assert value["reason_code"]
        assert value["confidence"] in {item.value for item in SelectionConfidence}
        if value["proposed_agent"] and value["selection_source"] != "EXPLICIT_USER":
            assert value["factors"] or value["selection_source"] in {
                "SINGLE_ELIGIBLE_AGENT",
                "ONLY_AVAILABLE_ELIGIBLE_AGENT",
            }


# --- dry-run safety ------------------------------------------------------


def test_selection_never_authorizes_execution_or_creates_work():
    for labels in (MUTATION_REPOSITORY, MUTATION_WITH_TESTS, READ_ONLY_REPOSITORY):
        proposal = propose(labels, selector=RecordingSelector("local"))
        value = proposal.as_dict()
        assert value["execution_authorized"] is False
        assert value["dry_run"] is True
        assert value["jobs_created"] == 0
        assert value["delegations"] == 0
        assert value["filesystem_mutations"] == 0
        assert value["session_resolved"] is False


def test_selection_does_not_resolve_or_create_a_codex_session(monkeypatch):
    calls: list[Any] = []

    def guard(*args, **kwargs):
        calls.append(args)
        raise AssertionError("selection must not resolve a Codex session")

    monkeypatch.setattr(CodexSessionResolver, "resolve", guard)
    proposal = propose(
        MUTATION_REPOSITORY,
        operational=OperationalFacts(reusable_codex_session=True),
        selector=RecordingSelector("codex", ["EXISTING_REUSABLE_SESSION"]),
    )
    assert proposal.proposed_agent is Agent.CODEX
    assert proposal.session_resolved is False
    assert calls == []


# --- corpus and evaluator -----------------------------------------------


def test_corpus_covers_the_required_categories():
    cases = load_selection_cases(CORPUS)
    assert len(cases) == 31
    identifiers = {case.id for case in cases}
    assert {"AS-H01", "AS-H02", "AS-H03", "AS-H04", "AS-H05", "AS-H06"} <= identifiers
    categories = {case.category for case in cases}
    for required in (
        "read_only_diagnosis",
        "architecture_review",
        "implementation",
        "diagnosis_and_proposed_fix",
        "implementation_and_tests",
        "code_review",
        "verification",
        "root_cause_analysis_existing_session",
        "explicit_codex",
        "explicit_deepseek",
        "explicit_unavailable_codex",
        "single_eligible_local",
        "no_eligible_agent",
        "ambiguous_equal_fit",
    ):
        assert required in categories
    contrasts = {case.id: case.contrast_of for case in cases if case.contrast_of}
    assert contrasts["AS-023"] == "AS-022"
    assert contrasts["AS-025"] == "AS-024"


def test_deterministic_corpus_run_is_clean_and_free_of_inference():
    report = SelectionEvaluator().run(load_selection_cases(CORPUS))
    metrics = report["metrics"]
    assert report["cases"] == 31
    assert metrics["extra_model_calls"] == 0
    assert metrics["explicit_agent_preservation"] == 1.0
    assert metrics["single_eligible_selection_accuracy"] == 1.0
    assert metrics["no_eligible_accuracy"] == 1.0
    assert metrics["availability_handling_accuracy"] == 1.0
    assert metrics["ineligible_agent_selection_rate"] == 0.0
    assert metrics["unavailable_agent_selection_rate"] == 0.0
    assert metrics["unjustified_selection_rate"] == 0.0
    assert metrics["acceptable_selection_rate"] == 1.0
    assert metrics["eligibility_exactness"] == 1.0
    assert report["errors"]["AVAILABILITY_CORRUPTED_ELIGIBILITY"] == 0
    assert report["errors"]["EXPLICIT_AGENT_OVERRIDDEN"] == 0
    assert report["errors"]["INELIGIBLE_AGENT_SELECTED"] == 0
    assert report["safety"] == {
        "execution_authorized": False,
        "automatic_tool_calls": 0,
        "automatic_delegations": 0,
        "filesystem_mutations": 0,
        "codex_jobs_created": 0,
        "deepseek_jobs_created": 0,
        "sessions_resolved": 0,
    }


def test_corpus_run_with_stub_selector_uses_one_call_per_multi_agent_case():
    class StubbedEvaluator(SelectionEvaluator):
        pass

    selector = RecordingSelector("local", ["IMPLEMENTATION_SUPPORT"])
    report = StubbedEvaluator(
        semantic_selector=SemanticAgentSelector(selector)
    ).run(load_selection_cases(CORPUS))
    semantic_rows = [
        row
        for row in report["rows"]
        if row["proposal"]["selection_source"] == "SEMANTIC_MULTI_AGENT"
    ]
    assert semantic_rows
    assert all(row["proposal"]["model_calls"] == 1 for row in semantic_rows)
    assert report["metrics"]["extra_model_calls"] == selector.calls
    assert selector.calls == len(
        [
            row
            for row in report["rows"]
            if row["proposal"]["model_calls"] == 1
        ]
    )
    assert report["safety"]["codex_jobs_created"] == 0
    assert report["safety"]["automatic_delegations"] == 0
