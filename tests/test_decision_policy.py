from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tern.orchestrator.decision_policy import (
    AgentDecisionPolicy,
    Intent,
    SideEffect,
    TOOL_EFFECTS,
)
from tern.orchestrator.projects import normalize_technical_transcript
from tern.orchestrator.routing_eval import (
    CASES_PATH,
    corpus_summary,
    evaluate,
    evaluate_live_qwen,
    format_confusion,
    load_cases,
)


def decide(text: str, **context):
    return AgentDecisionPolicy().decide(text, fixture_context=context)


@pytest.mark.parametrize(
    ("text", "context", "intent", "tools", "reason"),
    [
        ("essa arquitetura faz sentido?", {}, Intent.ANSWER_DIRECTLY, (), "direct_answer_sufficient"),
        ("onde ta a config da voz?", {}, Intent.LOCAL_SEARCH, ("find_project_files",), "explicit_local_search"),
        ("abre ele", {"focused_file": r"D:\tern\config.py"}, Intent.LOCAL_READ, ("filesystem_read_text",), "existing_file_context"),
        ("o codex terminou?", {}, Intent.CODEX_STATUS, ("get_codex_job_status",), "active_job_status_query"),
        ("oq o codex fez por ultimo?", {}, Intent.CODEX_REVIEW, ("review_codex_session",), "codex_history_query"),
        ("manda o codex corrigir", {"active_project": "tern"}, Intent.CODEX_DELEGATE, ("delegate_to_codex",), "explicit_codex_delegate"),
        ("fala pra ele olhar so warnings", {"focused_agent": "codex", "codex_job": {"status": "running", "job_id": "job-1"}}, Intent.CODEX_STEER, ("steer_codex_job",), "followup_to_active_job"),
        ("para ele", {"focused_agent": "codex", "codex_job": {"status": "running", "job_id": "job-1"}}, Intent.CODEX_CANCEL, ("cancel_codex_job",), "followup_to_active_job"),
        ("pergunta pro deepseek o que ele acha", {}, Intent.DEEPSEEK_DELEGATE, ("delegate_to_deepseek",), "explicit_deepseek_request"),
        ("o que o deepseek falou?", {}, Intent.DEEPSEEK_REVIEW, ("review_deepseek_session",), "deepseek_history_query"),
        ("abre config.py", {"ambiguous_target": True}, Intent.CLARIFY, (), "ambiguous_target"),
        ("qual e o projeto ativo", {"active_project": "tern"}, Intent.PROJECT_RESOLUTION, (), "tool_result_already_available"),
    ],
)
def test_core_decisions(text, context, intent, tools, reason):
    value = AgentDecisionPolicy().decide(text, fixture_context=context)
    assert value.intent == intent
    assert value.tools == tools
    assert value.reason_code == reason
    assert 0 <= value.confidence <= 1


def test_explicit_overrides_focus_and_auto_escalation_is_not_inferred():
    focused = {"focused_agent": "codex", "codex_job": {"status": "running"}}
    assert decide("pergunta ao DeepSeek sobre isso", **focused).intent == Intent.DEEPSEEK_DELEGATE
    direct = decide("isso parece uma arquitetura dificil?", **focused)
    assert "delegate_to_deepseek" not in direct.tools
    own = decide("responde voce mesmo sem usar o codex", **focused)
    assert own.intent == Intent.ANSWER_DIRECTLY
    assert own.user_override == "qwen_only"


def test_status_and_history_never_create_a_codex_turn():
    for text in ("o codex terminou?", "leia a ultima sessao do codex"):
        value = decide(text, active_project="tern")
        assert not value.new_codex_turn
        assert "delegate_to_codex" not in value.tools


def test_deepseek_review_never_calls_api():
    value = decide("o que o deepseek falou por ultimo?", active_project="tern")
    assert value.tools == ("review_deepseek_session",)
    assert "delegate_to_deepseek" not in value.tools


def test_multiagent_sequences_are_compact_and_ordered():
    codex_to_ds = decide("mostra pro deepseek os ultimos 3 turns do codex", active_project="tern")
    assert codex_to_ds.tools == ("review_codex_session", "delegate_to_deepseek")
    ds_to_codex = decide("pergunta ao deepseek e depois manda o codex implementar", active_project="tern")
    assert ds_to_codex.tools == ("delegate_to_deepseek", "delegate_to_codex")
    assert ds_to_codex.new_codex_turn


def test_focus_updates_only_from_real_results_and_reuses_file_content(tmp_path):
    source = tmp_path / "config.py"
    source.write_text("RATE = 2", encoding="utf-8")
    policy = AgentDecisionPolicy()
    policy.record_tool_result(
        "filesystem_read_text",
        {"path": str(source)},
        {"ok": True, "path": str(source), "content": "RATE = 2"},
    )
    context = policy.build_context(fixture_context={"active_project": "tern"})
    value = policy.decide("agora explica esse rate", context=context)
    assert value.intent == Intent.ANSWER_DIRECTLY
    assert value.tools == ()
    assert "RATE = 2" in policy.reusable_context_text()


def test_focus_tracks_agents_jobs_sessions_and_recent_tools():
    policy = AgentDecisionPolicy()
    policy.record_tool_result(
        "delegate_to_codex",
        {"project_path": r"D:\tern"},
        {"ok": True, "job_id": "job-1", "thread_id": "thread-1", "turn_id": "turn-1"},
    )
    assert policy.focus.focused_agent == "codex"
    assert policy.focus.focused_job == "job-1"
    assert policy.focus.last_codex_turn_id == "turn-1"
    policy.record_tool_result(
        "delegate_to_deepseek",
        {"project_path": r"D:\tern"},
        {"ok": True, "session_id": "ds-1"},
    )
    assert policy.focus.focused_agent == "deepseek"
    assert policy.focus.focused_session == "ds-1"
    assert policy.focus.recent_tools == ["delegate_to_codex", "delegate_to_deepseek"]


def test_tool_effects_are_declared_for_decision_tools():
    assert TOOL_EFFECTS["review_codex_session"] == SideEffect.READ_ONLY
    assert TOOL_EFFECTS["delegate_to_codex"] == SideEffect.CODE_EXECUTION
    assert TOOL_EFFECTS["delegate_to_deepseek"] == SideEffect.REMOTE_GENERATION
    value = decide("corrige o Jarvis", active_project="tern")
    assert value.side_effects == (SideEffect.CODE_EXECUTION,)


def test_tool_budget_matches_the_smallest_sufficient_plan():
    assert decide("o codex terminou?").max_tool_calls == 1
    assert decide("essa arquitetura faz sentido?").max_tool_calls == 0
    assert decide("mostra pro deepseek os ultimos 3 turns do codex").max_tool_calls == 2


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("o código ex terminou", "Codex terminou"),
        ("da uma checada na ultima acessao do codex", "sessao"),
        ("pergunta pro dip sique", "DeepSeek"),
        ("manda ele olhar só os uornings", "warnings"),
        ("abre o arquivo do lama ponto cpp", "llama.cpp"),
    ],
)
def test_contextual_stt_normalization_preserves_routing_signals(spoken, expected):
    assert expected.casefold() in normalize_technical_transcript(spoken).casefold()


def test_original_and_routing_transcripts_are_logged(tmp_path):
    class Logger:
        def __init__(self):
            self.events = []

        def write_event(self, event, **values):
            self.events.append((event, values))

    logger = Logger()
    policy = AgentDecisionPolicy(logger=logger)
    original = "pergunta pro dip sique"
    value = policy.decide(original, fixture_context={"active_project": "tern"})
    policy.record_decision(value, original)
    event, fields = logger.events[-1]
    assert event == "decision_made"
    assert fields["original_transcript"] == original
    assert "DeepSeek" in fields["routing_transcript"]
    assert fields["reason_code"] == "explicit_deepseek_request"


def test_routing_corpus_shape_and_fixtures():
    cases = load_cases()
    summary = corpus_summary(cases)
    assert summary == {
        "cases": 100,
        "development": 80,
        "holdout": 20,
        "multi_turn_scenarios": 15,
        "stt_cases": 10,
    }
    assert all({"id", "input", "context", "expected", "split"} <= set(case) for case in cases)
    assert len({case["id"] for case in cases}) == 100


def test_frozen_baseline_matches_recorded_metrics():
    stored = json.loads((CASES_PATH.parent / "agent_routing_baseline.json").read_text(encoding="utf-8"))
    for split in ("development", "holdout"):
        actual = evaluate(mode="legacy", split=split)
        expected = stored[split]
        for key in (
            "cases",
            "passed",
            "overall_accuracy",
            "tool_selection_accuracy",
            "unnecessary_delegations",
            "new_turn_violations",
            "tool_loop_violations",
        ):
            assert actual[key] == expected[key]


def test_policy_meets_development_and_holdout_targets_without_violations():
    development = evaluate(mode="policy", split="development")
    holdout = evaluate(mode="policy", split="holdout")
    assert development["overall_accuracy"] >= 0.95
    assert holdout["overall_accuracy"] >= 0.90
    for report in (development, holdout):
        assert report["new_turn_violations"] == 0
        assert report["forbidden_tool_calls"] == 0
        assert report["tool_loop_violations"] == 0
        assert report["unnecessary_delegations"] / report["cases"] <= 0.02
        assert "expected\\predicted" in format_confusion(report)


def test_live_qwen_eval_is_dry_run_and_records_latency():
    class Client:
        def chat(self, _messages, **_kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "Faz sentido."}}]}

    case = {
        "id": "live-1",
        "split": "development",
        "input": "essa arquitetura faz sentido?",
        "context": {"active_project": "tern"},
        "expected": {
            "intent": "ANSWER_DIRECTLY",
            "tools": [],
            "forbidden_tools": ["delegate_to_codex"],
            "project": "tern",
            "max_tool_calls": 0,
            "new_codex_turn": False,
        },
    }
    report = evaluate_live_qwen(cases=[case], client=Client(), tool_specs=[])
    assert report["passed"] == 1
    assert report["live_qwen"]
    assert report["average_time_to_decision_ms"] >= 0


def test_semantic_live_eval_does_not_construct_tool_registry(monkeypatch):
    from tern.orchestrator import cli

    case = {
        "id": "dry-run",
        "input": "responda",
        "context": {},
        "expected": {"intent": "ANSWER_DIRECTLY", "tools": []},
    }
    manager = SimpleNamespace(
        ensure_llama_server=lambda _wait: {},
        inspect_llama_server=lambda: {"healthy": True},
    )
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(base_url="http://test", timeout=1))
    monkeypatch.setattr(cli, "RuntimeManager", lambda _settings: manager)
    monkeypatch.setattr(cli, "load_cases", lambda: [case])
    monkeypatch.setattr(cli, "LlamaClient", lambda *_args: object())
    monkeypatch.setattr(
        cli,
        "evaluate_live_semantic_qwen",
        lambda **kwargs: {"failed": 0, "received_cases": len(kwargs["cases"])},
    )
    monkeypatch.setattr(
        cli,
        "_registry",
        lambda _settings: (_ for _ in ()).throw(AssertionError("tool registry must stay unused")),
    )

    assert cli.main(["agent-routing-eval", "--live-qwen", "--semantic-first", "--limit", "1", "--json"]) == 0
