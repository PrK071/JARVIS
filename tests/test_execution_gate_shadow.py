from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.agent_selection import SelectionPolicy, SelectionSource
from tern.orchestrator.autonomy_eval import diagnostic_baseline
from tern.orchestrator.autonomy_foundation import Agent
from tern.orchestrator.config import load_settings
from tern.orchestrator.decision_policy import SideEffect
from tern.orchestrator.execution_gate import (
    ExecutionAuthority,
    ExecutionBlockReason,
    ExecutionMode,
)
from tern.orchestrator.execution_gate_eval import (
    ShadowGateEvaluator,
    load_shadow_cases,
)
from tern.orchestrator.execution_gate_shadow import (
    DivergenceCode,
    ShadowExecutionObserver,
    capability_baseline_from_registry,
    compare_legacy_and_shadow,
    legacy_facts_from_decision,
)


CORPUS = "tests/data/execution_gate_shadow.jsonl"

TOOL_NAMES = (
    "resolve_project",
    "find_project_files",
    "filesystem_list",
    "filesystem_read_text",
    "filesystem_write_text",
    "filesystem_delete",
    "web_search",
    "review_codex_session",
    "get_codex_job_status",
    "delegate_to_codex",
    "steer_codex_job",
    "cancel_codex_job",
    "review_deepseek_session",
    "delegate_to_deepseek",
)


class Logger:
    def __init__(self):
        self.events = []

    def write_event(self, event, **values):
        self.events.append((event, values))

    def find(self, event):
        return [values for name, values in self.events if name == event]


class ExplodingRegistry:
    """Any executor touched by the shadow path fails the test immediately."""

    def __init__(self):
        self.logger = Logger()
        self.web = SimpleNamespace(begin_research=lambda _text: None)
        self.projects = SimpleNamespace(
            context=lambda: {"active_project": {"id": "tern", "root": r"D:\tern"}},
            context_text=lambda: "Active project: tern",
            projects=lambda: [{"id": "tern", "root": r"D:\tern"}],
        )
        self.codex = SimpleNamespace(
            jobs=SimpleNamespace(
                list=lambda: [],
                create=self._forbidden("codex_jobs_create"),
            ),
            shared_project=lambda: r"D:\tern",
            claim_completed_results=lambda: [],
            sessions=SimpleNamespace(resolve=self._forbidden("session_resolve")),
        )
        self.deepseek = SimpleNamespace(
            client=SimpleNamespace(enabled=True, configured=True),
            status=lambda **_kwargs: {
                "enabled": True,
                "configured": True,
                "active_session": "ds-1",
            },
        )
        self.pending_actions = SimpleNamespace(
            pending=lambda: None,
            prepare=self._forbidden("pending_actions_prepare"),
        )
        self.forbidden_calls: list[str] = []

    def _forbidden(self, label):
        def guard(*_args, **_kwargs):
            self.forbidden_calls.append(label)
            raise AssertionError(f"shadow path called {label}")

        return guard

    def names(self):
        return TOOL_NAMES

    def specs(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in TOOL_NAMES
        ]

    def execute(self, name, arguments, **_kwargs):
        self.forbidden_calls.append(f"execute:{name}")
        raise AssertionError(f"shadow path executed tool {name}")

    def delegate_to_codex(self, *_args, **_kwargs):
        self.forbidden_calls.append("delegate_to_codex")
        raise AssertionError("shadow path delegated to codex")

    def delegate_to_deepseek(self, *_args, **_kwargs):
        self.forbidden_calls.append("delegate_to_deepseek")
        raise AssertionError("shadow path delegated to deepseek")


class AnswerOnlyClient:
    """Never asks for a tool, so any executor call would come from the shadow."""

    supports_structured_output = False

    def __init__(self):
        self.calls = 0

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def observer(**kwargs):
    return ShadowExecutionObserver(
        policy=SelectionPolicy(deepseek_auto_escalation=False),
        **kwargs,
    )


def baseline():
    return diagnostic_baseline(deepseek_available=True)


def settings(**overrides):
    values = {
        "MODEL_MAX_TOOL_CALLS": "4",
        "AGENT_DECISION_FAST_PATH": "false",
        "AGENT_DECISION_SEMANTIC_FIRST": "false",
    }
    values.update(overrides)
    return load_settings(values)


# --- shadow pipeline behaviour -------------------------------------------------


def test_shadow_observation_costs_zero_model_calls():
    observation = observer().observe(
        "corrija o bug no modulo de autenticacao e rode os testes",
        baseline=baseline(),
        execution_requested=True,
    )
    assert observation.model_calls == 0
    assert observation.proposal.model_calls == 0
    assert observation.authority is ExecutionAuthority.SHADOW


def test_shadow_never_reports_side_effects():
    observation = observer().observe(
        "use o Codex para corrigir o bug e alterar o modulo",
        baseline=baseline(),
        execution_requested=True,
    )
    record = observation.provenance_record()
    assert record["live_authority"] is False
    assert record["mode"] == "SHADOW"
    assert record["delegations"] == 0
    assert record["jobs_created"] == 0
    assert record["sessions_resolved"] == 0
    assert record["filesystem_mutations"] == 0
    assert observation.proposal.session_resolved is False


def test_explicit_agent_is_never_substituted_in_shadow():
    observation = observer().observe(
        "use o DeepSeek para explicar esse conceito",
        baseline=baseline(),
        execution_requested=True,
    )
    decision = observation.decision
    assert decision.selection_source is SelectionSource.EXPLICIT_USER
    assert decision.candidate_agent is Agent.DEEPSEEK
    assert decision.requested_agent is Agent.DEEPSEEK
    assert observation.model_calls == 0


def test_meta_question_can_select_but_never_executes():
    observation = observer().observe(
        "qual agente seria melhor para essa tarefa?",
        baseline=baseline(),
        execution_requested=False,
    )
    decision = observation.decision
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.EXECUTION_NOT_REQUESTED
    assert decision.mutation_authorized is False


def test_unavailable_explicit_agent_is_not_replaced_by_a_fallback():
    observation = observer().observe(
        "use o Codex para corrigir esse teste que falha",
        baseline=baseline(),
        execution_requested=True,
        availability_override={Agent.CODEX: False},
    )
    decision = observation.decision
    assert decision.candidate_agent is Agent.CODEX
    assert decision.agent_available is False
    assert decision.execution_allowed is False
    assert decision.block_reason is ExecutionBlockReason.REQUESTED_AGENT_UNAVAILABLE


def test_availability_override_is_a_snapshot_and_keeps_eligibility():
    task = "corrija o bug no modulo de autenticacao e rode os testes"
    full = observer().observe(task, baseline=baseline(), execution_requested=True)
    degraded = observer().observe(
        task,
        baseline=baseline(),
        execution_requested=True,
        availability_override={Agent.CODEX: False},
    )
    assert full.decision.eligible_agents == degraded.decision.eligible_agents
    assert Agent.CODEX not in degraded.decision.available_eligible_agents
    assert degraded.availability_snapshot[Agent.CODEX].available is False


def test_deepseek_auto_selection_is_policy_excluded_but_stays_eligible():
    observation = ShadowExecutionObserver(
        policy=SelectionPolicy(deepseek_auto_escalation=False)
    ).observe(
        "analise conceitualmente essa ideia de arquitetura distribuida",
        baseline=baseline(),
        execution_requested=True,
        availability_override={Agent.CODEX: False, Agent.LOCAL: False},
    )
    decision = observation.decision
    assert Agent.DEEPSEEK in decision.eligible_agents
    assert decision.execution_allowed is False
    assert decision.mutation_authorized is False


def test_read_only_request_never_authorizes_mutation():
    observation = observer().observe(
        "use o Codex para investigar a causa sem modificar nenhum arquivo",
        baseline=baseline(),
        execution_requested=True,
    )
    decision = observation.decision
    assert decision.mutation_requested is False
    assert decision.mutation_authorized is False
    assert decision.execution_mode is ExecutionMode.READ_ONLY


def test_observation_is_deterministic_for_the_same_facts():
    task = "corrija a funcao e execute os testes unitarios do projeto"
    first = observer().observe(task, baseline=baseline(), execution_requested=True)
    second = observer().observe(task, baseline=baseline(), execution_requested=True)
    assert first.decision == second.decision
    assert first.requirements.as_dict() == second.requirements.as_dict()


# --- legacy comparison ---------------------------------------------------------


def legacy_stub(**values):
    base = {
        "intent": SimpleNamespace(value="CODE_EXECUTION"),
        "tools": ("delegate_to_codex",),
        "selected_action": "delegate_to_codex",
        "side_effects": (SideEffect.CODE_EXECUTION,),
        "requested_agent": None,
        "constraint_violation": None,
        "execution_allowed": None,
        "intent_frame": None,
    }
    base.update(values)
    return SimpleNamespace(**base)


def test_legacy_facts_read_the_decision_without_changing_it():
    decision = legacy_stub(requested_agent="codex")
    facts = legacy_facts_from_decision(decision)
    assert facts.agent is Agent.CODEX
    assert facts.requested_agent is Agent.CODEX
    assert facts.execution_allowed is True
    assert decision.selected_action == "delegate_to_codex"


def test_legacy_constraint_violation_is_not_execution():
    facts = legacy_facts_from_decision(
        legacy_stub(constraint_violation="FORBID_DELEGATION")
    )
    assert facts.execution_allowed is False
    assert facts.mutation_behavior is False


def test_divergence_is_recorded_when_legacy_executes_and_shadow_blocks():
    observation = observer().observe(
        "use o Codex para corrigir isso agora e alterar o arquivo",
        baseline=baseline(),
        execution_requested=True,
        availability_override={Agent.CODEX: False},
        legacy=legacy_facts_from_decision(legacy_stub(requested_agent="codex")),
    )
    comparison = observation.comparison
    assert comparison is not None
    assert comparison.legacy_execution_allowed is True
    assert comparison.shadow_execution_allowed is False
    assert DivergenceCode.LEGACY_EXECUTES_SHADOW_BLOCKS in comparison.divergence_codes
    assert comparison.agreement is False


def test_divergence_is_never_repaired():
    observation = observer().observe(
        "use o Codex para corrigir o modulo e alterar os arquivos necessarios",
        baseline=baseline(),
        execution_requested=True,
        legacy=legacy_facts_from_decision(
            legacy_stub(
                selected_action="delegate_to_deepseek",
                tools=("delegate_to_deepseek",),
                side_effects=(SideEffect.REMOTE_GENERATION,),
            )
        ),
    )
    comparison = observation.comparison
    assert comparison is not None
    assert DivergenceCode.AGENT_MISMATCH in comparison.divergence_codes
    # the shadow decision keeps its own answer; nothing was aligned to legacy
    assert observation.decision.candidate_agent is Agent.CODEX


def test_comparison_flags_missing_provenance():
    observation = observer().observe(
        "corrija o bug no modulo de autenticacao e rode os testes",
        baseline=baseline(),
        execution_requested=True,
    )
    broken = observation.decision.__class__(
        candidate_agent=Agent.CODEX,
        execution_requested=True,
        agent_eligible=True,
        agent_available=True,
        selection_valid=True,
        selection_supported=True,
        execution_allowed=True,
        execution_mode=ExecutionMode.READ_ONLY,
        mutation_requested=False,
        mutation_authorized=False,
        block_reason=None,
        block_reasons=(),
        mutation_block_reason=None,
        selection_source=SelectionSource.DETERMINISTIC_SELECTION,
        selection_factors=(),
        selection_reason_code="",
        requested_agent=None,
        requested_agent_source=None,
        eligible_agents=(Agent.CODEX,),
        available_eligible_agents=(Agent.CODEX,),
    )
    comparison = compare_legacy_and_shadow(
        legacy_facts_from_decision(legacy_stub()),
        broken,
    )
    assert DivergenceCode.SELECTION_PROVENANCE_MISSING in comparison.divergence_codes


# --- zero live authority -------------------------------------------------------


def test_shadow_path_touches_no_executor():
    registry = ExplodingRegistry()
    snapshot = capability_baseline_from_registry(registry)
    observation = observer().observe(
        "use o Codex para corrigir o bug e alterar o modulo agora",
        baseline=snapshot,
        execution_requested=True,
        legacy=legacy_facts_from_decision(legacy_stub(requested_agent="codex")),
    )
    assert registry.forbidden_calls == []
    assert observation.decision.candidate_agent is Agent.CODEX
    assert observation.decision.delegations == 0


def test_registry_view_does_not_expose_executors():
    registry = ExplodingRegistry()
    capability_baseline_from_registry(registry)
    from tern.orchestrator.execution_gate_shadow import _ReadOnlyRegistryView

    view = _ReadOnlyRegistryView(registry)
    assert not hasattr(view, "execute")
    assert not hasattr(view, "delegate_to_codex")
    assert not hasattr(view, "pending_actions")
    assert view.names() == TOOL_NAMES


def test_live_run_with_shadow_enabled_calls_no_tool():
    registry = ExplodingRegistry()
    client = AnswerOnlyClient()
    result = Supervisor(settings(), client, registry).run(
        "use o Codex para corrigir o bug e alterar o modulo agora"
    )
    assert result["ok"] is True
    assert registry.forbidden_calls == []
    shadow_events = registry.logger.find("execution_gate_shadow")
    assert len(shadow_events) == 1
    record = shadow_events[0]
    assert record["mode"] == "SHADOW"
    assert record["live_authority"] is False
    assert record["delegations"] == 0
    assert record["jobs_created"] == 0
    assert record["sessions_resolved"] == 0
    assert record["filesystem_mutations"] == 0
    assert registry.logger.find("execution_gate_shadow_error") == []


def test_shadow_record_carries_full_provenance():
    registry = ExplodingRegistry()
    Supervisor(settings(), AnswerOnlyClient(), registry).run(
        "use o Codex para corrigir o bug e alterar o modulo agora"
    )
    record = registry.logger.find("execution_gate_shadow")[0]
    for field in (
        "selection_source",
        "selection_factors",
        "requested_agent",
        "eligible_agents",
        "available_eligible_agents",
        "candidate_agent",
        "execution_requested",
        "execution_allowed_shadow",
        "mutation_requested",
        "mutation_authorized_shadow",
        "block_reason",
    ):
        assert field in record
    assert record["candidate_agent"] == "codex"
    assert record["requested_agent"] == "codex"


def test_shadow_record_does_not_leak_task_text():
    registry = ExplodingRegistry()
    secret = "use o Codex para corrigir o token SUPERSECRETVALUE no modulo"
    Supervisor(settings(), AnswerOnlyClient(), registry).run(secret)
    encoded = json.dumps(registry.logger.find("execution_gate_shadow")[0])
    assert "SUPERSECRETVALUE" not in encoded
    assert "corrigir" not in encoded


def test_shadow_failure_never_breaks_the_live_pipeline(monkeypatch):
    registry = ExplodingRegistry()
    monkeypatch.setattr(
        ShadowExecutionObserver,
        "observe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = Supervisor(settings(), AnswerOnlyClient(), registry).run(
        "use o Codex para corrigir o bug agora"
    )
    assert result["ok"] is True
    assert registry.logger.find("execution_gate_shadow") == []
    errors = registry.logger.find("execution_gate_shadow_error")
    assert errors and errors[0]["error"] == "RuntimeError"


def test_shadow_can_be_disabled_by_configuration():
    registry = ExplodingRegistry()
    agent = Supervisor(
        settings(EXECUTION_GATE_SHADOW="false"), AnswerOnlyClient(), registry
    )
    assert agent.shadow_observer is None
    result = agent.run("use o Codex para corrigir o bug agora")
    assert result["ok"] is True
    assert registry.logger.find("execution_gate_shadow") == []


class ShadowStructuredClient(AnswerOnlyClient):
    def chat(self, _messages, **kwargs):
        self.calls += 1
        response_format = kwargs.get("response_format") or {}
        schema_name = (response_format.get("json_schema") or {}).get("name")
        if schema_name == "shadow_next_action":
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "INSPECT",
                                    "target_agent": None,
                                    "target": "authentication files",
                                    "tool_name": None,
                                    "arguments": {},
                                    "objective": "inspect authentication evidence",
                                    "execution_mode": "READ_ONLY",
                                    "required_capabilities": [],
                                    "reason_code": "REPOSITORY_INSPECTION_REQUIRED",
                                    "evidence_refs": [],
                                    "expected_observation": "repository facts",
                                    "confidence": None,
                                    "short_horizon_hint": None,
                                }
                            )
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def test_phase_175_observer_is_opt_in_and_has_zero_live_effects():
    registry = ExplodingRegistry()
    client = ShadowStructuredClient()
    result = Supervisor(
        settings(ORCHESTRATION_SHADOW_ENABLED="true"), client, registry
    ).run("investigue o bug de autenticacao")

    assert result["ok"] is True
    assert registry.forbidden_calls == []
    records = registry.logger.find("orchestration_shadow")
    assert len(records) == 1
    record = records[0]
    assert record["mode"] == "SHADOW"
    assert record["model_calls"] == 1
    assert set(record["effect_counts"].values()) == {0}
    assert record["records"][0]["authority_shadow_result"]["live_authority"] is False


def test_phase_175_policy_failure_cannot_change_live_result(monkeypatch):
    from tern.orchestrator.orchestration_policy import QwenOrchestrationPolicy

    registry = ExplodingRegistry()
    monkeypatch.setattr(
        QwenOrchestrationPolicy,
        "decide",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = Supervisor(
        settings(ORCHESTRATION_SHADOW_ENABLED="true"),
        AnswerOnlyClient(),
        registry,
    ).run("investigue o bug")
    assert result["ok"] is True
    assert registry.forbidden_calls == []
    assert registry.logger.find("orchestration_shadow_error") == []
    record = registry.logger.find("orchestration_shadow")[0]
    assert record["termination_reason"] == "POLICY_FAILURE"
    assert set(record["effect_counts"].values()) == {0}


# --- corpus level safety targets ----------------------------------------------


@pytest.fixture(scope="module")
def report():
    return ShadowGateEvaluator().run(load_shadow_cases(CORPUS))


def test_corpus_matches_expected_shadow_behaviour(report):
    assert report["metrics"]["corpus_expectation_accuracy"] == 1.0
    assert report["metrics"]["shadow_pipeline_success_rate"] == 1.0
    assert report["cases"] >= 20


def test_corpus_hard_safety_targets(report):
    metrics = report["metrics"]
    assert metrics["ineligible_shadow_execution_rate"] == 0.0
    assert metrics["unavailable_shadow_execution_rate"] == 0.0
    assert metrics["policy_excluded_shadow_execution_rate"] == 0.0
    assert metrics["execution_without_request_shadow_rate"] == 0.0
    assert metrics["mutation_without_authorization_shadow_rate"] == 0.0
    assert metrics["mutation_without_execution_shadow_rate"] == 0.0
    assert metrics["explicit_agent_preservation"] == 1.0
    assert metrics["selection_provenance_completeness"] == 1.0
    assert metrics["bad_selection_factor_rate"] == 0.0
    assert metrics["model_calls"] == 0


def test_corpus_reports_zero_side_effects(report):
    safety = report["safety"]
    assert safety["authority"] == "SHADOW"
    assert safety["live_authority"] is False
    assert safety["shadow_executor_calls"] == 0
    assert safety["shadow_job_creations"] == 0
    assert safety["shadow_session_resolutions"] == 0
    assert safety["shadow_mutations"] == 0


def test_corpus_keeps_divergences_visible(report):
    divergences = report["divergences"]
    assert sum(divergences.values()) > 0
    assert divergences["AGENT_MISMATCH"] > 0
    assert divergences["LEGACY_EXECUTES_SHADOW_BLOCKS"] > 0
    assert report["findings"]


def test_corpus_covers_the_required_categories(report):
    categories = {row["category"] for row in report["rows"]}
    for name in (
        "explicit_agent",
        "explicit_agent_unavailable",
        "meta_question",
        "single_eligible",
        "one_eligible_unavailable",
        "no_available_eligible",
        "read_only",
        "forbid_mutation",
        "policy_exclusion",
        "multi_agent",
        "no_execution_request",
        "constraint_violation",
        "mutation_unauthorized",
        "mutation_authorized",
        "agreement",
        "divergence",
    ):
        assert name in categories
