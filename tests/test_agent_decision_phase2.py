from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from tern.orchestrator.decision_observability import (
    AgentDecisionObserver,
    DecisionTiming,
    latency_summary,
)
from tern.orchestrator.decision_policy import (
    AgentDecisionPolicy,
    Intent,
    SideEffect,
)
from tern.orchestrator.explicit_agent_binding import ExplicitAgentBinding
from tern.orchestrator.routing_eval import evaluate


def fixture_policy(text: str, **fixture):
    policy = AgentDecisionPolicy()
    context = policy.build_context(fixture_context=fixture)
    return policy, context, policy.decide(text, context=context)


def test_shadow_mode_does_not_change_decision(tmp_path):
    policy, context, decision = fixture_policy(
        "o codex terminou?",
        active_project="tern",
        focused_agent="codex",
        codex_job={"status": "running", "job_id": "job-1"},
    )
    observer = AgentDecisionObserver(tmp_path, enabled=True)
    before = decision.as_dict()
    identifier = observer.begin(
        original_input="o codex terminou?",
        normalized_input="o Codex terminou?",
        decision=decision,
        context=context,
        prompt_sizes={},
    )
    observer.complete(
        identifier,
        original_input="o codex terminou?",
        normalized_input="o Codex terminou?",
        decision=decision,
        context=context,
        prompt_sizes={},
        timing={"decision_ms": 1.0},
        tool_calls=1,
        actual_tools=["get_codex_job_status"],
        outcome="success",
    )
    assert decision.as_dict() == before
    event = json.loads(observer.path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["speech_act"] == "STATUS_QUERY"
    assert event["execution_requested"] is False
    assert event["constraints"] == []
    assert event["resolved_reference_type"] == "codex_job"
    assert event["reference_confidence"] > 0
    assert event["followup_type"] == "NEW_REQUEST"


def test_shadow_disabled_does_not_write(tmp_path):
    _policy, context, decision = fixture_policy("isso faz sentido?", active_project="tern")
    observer = AgentDecisionObserver(tmp_path, enabled=False)
    identifier = observer.begin(
        original_input="isso faz sentido?",
        normalized_input="isso faz sentido?",
        decision=decision,
        context=context,
        prompt_sizes={},
    )
    observer.complete(
        identifier,
        original_input="isso faz sentido?",
        normalized_input="isso faz sentido?",
        decision=decision,
        context=context,
        prompt_sizes={},
        timing={},
        tool_calls=0,
        actual_tools=[],
        outcome="direct_answer",
    )
    assert identifier is None
    assert not observer.path.exists()


def test_shadow_redacts_tokens(tmp_path):
    _policy, context, decision = fixture_policy("responde voce mesmo", active_project="tern")
    observer = AgentDecisionObserver(tmp_path, enabled=True)
    secret = "sk-abcdefghijklmnopqrstuv"
    identifier = observer.begin(
        original_input=secret,
        normalized_input=secret,
        decision=decision,
        context=context,
        prompt_sizes={},
    )
    observer.complete(
        identifier,
        original_input=secret,
        normalized_input=secret,
        decision=decision,
        context=context,
        prompt_sizes={},
        timing={},
        tool_calls=0,
        actual_tools=[],
        outcome="direct_answer",
    )
    stored = observer.path.read_text(encoding="utf-8")
    assert secret not in stored
    assert "<redacted>" in stored


def test_timing_metrics_and_percentiles():
    timing = DecisionTiming()
    timing.mark("input_received")
    time.sleep(0.001)
    timing.mark("decision_context_ready")
    timing.mark("decision_ready")
    timing.mark("prompt_ready")
    timing.mark("decision_detected")
    timing.mark("response_ready")
    value = timing.as_dict()
    assert value["context_build_ms"] >= 1
    assert value["qwen_first_token_ms"] is None
    assert not value["qwen_streaming"]
    summary = latency_summary([1, 2, 3, 4, 100])
    assert summary["p50_ms"] == 3
    assert summary["p90_ms"] > 4


def test_fast_path_codex_status_is_read_only():
    policy, context, decision = fixture_policy(
        "terminou?",
        active_project="tern",
        focused_agent="codex",
        codex_job={"status": "running", "job_id": "job-1"},
    )
    shortcut = policy.fast_path(decision, context, "terminou?")
    assert shortcut is not None
    assert shortcut.tool == "get_codex_job_status"
    assert shortcut.side_effect == SideEffect.READ_ONLY
    assert shortcut.arguments["job_id"] == "job-1"


def test_fast_path_codex_status_uses_latest_when_no_job_is_focused():
    policy, context, decision = fixture_policy(
        "qual o status atual da sessão do Codex?",
        active_project="tern",
    )
    shortcut = policy.fast_path(decision, context, "qual o status atual da sessão do Codex?")
    assert shortcut is not None
    assert shortcut.tool == "get_codex_job_status"
    assert shortcut.arguments == {"job_id": None, "latest": True}


def test_fast_path_reads_only_single_focused_file():
    policy, context, decision = fixture_policy(
        "abre ele",
        active_project="tern",
        focused_file=r"D:\tern\config.py",
    )
    shortcut = policy.fast_path(decision, context, "abre ele")
    assert shortcut is not None
    assert shortcut.tool == "filesystem_read_text"
    _policy, ambiguous, value = fixture_policy(
        "abre ele",
        active_project="tern",
        focused_file=r"D:\tern\config.py",
        ambiguous_target=True,
    )
    assert _policy.fast_path(value, ambiguous, "abre ele") is None


def test_explicit_codex_binding_has_direct_handoff_arguments():
    policy = AgentDecisionPolicy()
    context = policy.build_context(fixture_context={"active_project": "tern"})
    prompt = "delegue essa tarefa ao Codex\nRevise o projeto sem ampliar o escopo."
    decision = policy.decide(
        prompt,
        context=context,
        explicit_agent_binding=ExplicitAgentBinding("codex"),
    )

    shortcut = policy.fast_path(decision, context, prompt)

    assert shortcut is not None
    assert shortcut.tool == "delegate_to_codex"
    assert shortcut.reason_code == "explicit_agent_direct_handoff"
    assert shortcut.side_effect == SideEffect.CODE_EXECUTION
    assert shortcut.arguments == {
        "task": prompt,
        "project_path": r"D:\tern",
        "continue_current_thread": True,
    }


def test_automatic_project_mutation_has_direct_handoff_arguments():
    policy, context, decision = fixture_policy(
        "melhore a pagina inicial",
        active_project="tern",
    )

    shortcut = policy.fast_path(decision, context, "melhore a pagina inicial")

    assert shortcut is not None
    assert shortcut.tool == "delegate_to_codex"
    assert shortcut.reason_code == "automatic_mutation_direct_handoff"
    assert shortcut.side_effect == SideEffect.CODE_EXECUTION
    assert shortcut.arguments == {
        "task": "melhore a pagina inicial",
        "project_path": r"D:\tern",
        "continue_current_thread": True,
    }


def test_active_job_mutation_has_direct_steer_arguments():
    policy, context, decision = fixture_policy(
        "adicione tambem testes mobile",
        active_project="tern",
        codex_job={"status": "running", "job_id": "job-1"},
    )

    shortcut = policy.fast_path(decision, context, "adicione tambem testes mobile")

    assert shortcut is not None
    assert shortcut.tool == "steer_codex_job"
    assert shortcut.reason_code == "active_job_mutation_direct_steer"
    assert shortcut.side_effect == SideEffect.CODE_EXECUTION
    assert shortcut.arguments == {
        "instruction": "adicione tambem testes mobile",
        "job_id": "job-1",
        "latest": False,
    }


def test_fast_path_never_cancels_or_generates_consultations():
    cases = [
        (
            "para ele",
            {
                "active_project": "tern",
                "focused_agent": "codex",
                "codex_job": {"status": "running", "job_id": "job-1"},
            },
            Intent.CODEX_CANCEL,
        ),
        (
            "pergunta ao DeepSeek",
            {"active_project": "tern"},
            Intent.DEEPSEEK_DELEGATE,
        ),
    ]
    for text, fixture, intent in cases:
        policy, context, decision = fixture_policy(text, **fixture)
        assert decision.intent == intent
        assert policy.fast_path(decision, context, text) is None


class CachedTools:
    def __init__(self):
        self.context_calls = 0
        self.status_calls = 0
        owner = self

        class Projects:
            def context(self):
                owner.context_calls += 1
                return {
                    "active_project": {"id": "tern", "root": r"D:\tern"},
                    "codex_thread_project": {"id": "tern"},
                }

            def projects(self):
                return [{"id": "tern", "name": "Tern", "root": r"D:\tern"}]

        self.projects = Projects()
        self.codex = SimpleNamespace(
            jobs=SimpleNamespace(list=lambda: [{"job_id": "job-1", "status": "running"}]),
            shared_project=lambda: r"D:\tern",
        )
        self.deepseek = self
        self.pending_actions = SimpleNamespace(pending=lambda: None)
        self.logger = None

    def status(self, **_kwargs):
        self.status_calls += 1
        return {"enabled": True, "configured": False, "active_session": "ds-1"}


def test_context_cache_and_file_invalidation():
    tools = CachedTools()
    policy = AgentDecisionPolicy(tools=tools, context_cache_enabled=True)
    policy.build_context()
    policy.build_context()
    assert tools.context_calls == 1
    policy.record_tool_result("filesystem_read_text", {"path": "x"}, {"ok": False})
    policy.build_context()
    assert tools.context_calls == 2
    assert policy._context_cache_last_reason == "file_focus_changed"


def test_job_and_project_events_invalidate_context_cache():
    tools = CachedTools()
    policy = AgentDecisionPolicy(tools=tools, context_cache_enabled=True)
    policy.build_context()
    policy.record_tool_result("delegate_to_codex", {}, {"ok": True, "job_id": "job-2"})
    assert policy._context_cache_last_reason == "job_started"
    policy.build_context()
    policy.record_tool_result("resolve_project", {}, {"ok": True, "root": r"D:\llama.cpp"})
    assert policy._context_cache_last_reason == "project_changed"


def test_feedback_and_stats_are_local(tmp_path):
    _policy, context, decision = fixture_policy("isso faz sentido?", active_project="tern")
    observer = AgentDecisionObserver(tmp_path, enabled=True)
    identifier = observer.begin(
        original_input="isso faz sentido?",
        normalized_input="isso faz sentido?",
        decision=decision,
        context=context,
        prompt_sizes={},
    )
    observer.complete(
        identifier,
        original_input="isso faz sentido?",
        normalized_input="isso faz sentido?",
        decision=decision,
        context=context,
        prompt_sizes={},
        timing={"decision_ms": 12.0},
        tool_calls=0,
        actual_tools=[],
        outcome="direct_answer",
    )
    feedback = observer.feedback(verdict="wrong", expected="CLARIFY")
    assert feedback["ok"] and feedback["actual"] == "ANSWER_DIRECTLY"
    stats = observer.stats(days=7)
    assert stats["decisions"] == 1
    assert stats["user_corrections"] == 1
    assert stats["latency"]["p50_ms"] == 12


def test_sealed_v2_checksum_is_unchanged_without_evaluating_it():
    root = Path(__file__).parent / "data"
    seal = json.loads((root / "agent_routing_test_v2.seal.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / "agent_routing_test_v2.jsonl").read_bytes()).hexdigest()
    assert str(seal["status"]).startswith("sealed_")
    assert digest == seal["sha256"]
    assert seal["cases"] == 40


def test_agent_routing_regression_gate_is_100_of_100():
    report = evaluate(mode="policy")
    assert report["cases"] == 100
    assert report["passed"] == 100
    assert report["forbidden_tool_calls"] == 0
    assert report["new_turn_violations"] == 0
    assert report["tool_loop_violations"] == 0


def test_v3_is_sealed_and_result_is_preserved_without_reevaluation():
    root = Path(__file__).parent / "data"
    seal = json.loads((root / "agent_routing_test_v3.seal.json").read_text(encoding="utf-8"))
    result = json.loads((root / "agent_routing_test_v3_result.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / "agent_routing_test_v3.jsonl").read_bytes()).hexdigest()
    assert seal["status"] == "sealed_before_first_evaluation"
    assert digest == seal["sha256"] == result["sha256"]
    assert seal["cases"] == result["metrics"]["cases"] == 50
    assert result["metrics"]["passed"] == 17
    assert result["policy_changed_from_results"] is False
