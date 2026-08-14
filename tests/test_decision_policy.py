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
    normalize_browser_open_request,
    normalize_contextual_web_open_request,
    normalize_explicit_web_open_request,
    normalize_known_site_open_request,
    normalize_music_open_request,
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
        ("qual o status atual da sessão do Codex?", {}, Intent.CODEX_STATUS, ("get_codex_job_status",), "active_job_status_query"),
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


@pytest.mark.parametrize(
    ("text", "expected_url"),
    [
        ("abra example.com", "https://example.com"),
        ("abra www.example.com", "https://www.example.com"),
        ("abra https://example.com", "https://example.com"),
        ("abra https://www.example.com/", "https://www.example.com/"),
        ("abra xvideos.com", "https://xvideos.com"),
        ("abra www.xvideos.com", "https://www.xvideos.com"),
        ("abra https://xvideos.com", "https://xvideos.com"),
        ("abra https://www.xvideos.com/", "https://www.xvideos.com/"),
        ("abra o xvideos.com", "https://xvideos.com"),
        ("abra o site example.com", "https://example.com"),
    ],
)
def test_explicit_url_open_is_generic_and_launches_browser(text, expected_url):
    value = decide(text)
    assert normalize_explicit_web_open_request(text) == expected_url
    assert value.intent is Intent.WEB_OPEN
    assert value.tools == ("web_open_browser",)
    assert value.target == expected_url
    assert value.reason_code == "explicit_browser_url"


def test_local_filename_is_not_reclassified_as_hostname():
    assert normalize_explicit_web_open_request("abra config.py") is None
    value = decide("abra config.py", ambiguous_target=True)
    assert value.intent is Intent.CLARIFY


@pytest.mark.parametrize(
    ("text", "expected_url"),
    [
        ("abra xvideos", "https://www.xvideos.com/"),
        ("abre o xvideos", "https://www.xvideos.com/"),
        ("acesse o site xvideos", "https://www.xvideos.com/"),
        ("abra pornhub", "https://pt.pornhub.com/"),
        ("abra o pornhub", "https://pt.pornhub.com/"),
        ("abra amazon", "https://www.amazon.com.br/"),
        ("abra a amazon", "https://www.amazon.com.br/"),
    ],
)
def test_known_site_alias_launches_browser_without_url(text, expected_url):
    assert normalize_known_site_open_request(text) == expected_url
    value = decide(text)
    assert value.intent is Intent.WEB_OPEN
    assert value.tools == ("web_open_browser",)
    assert value.target == expected_url
    assert value.reason_code == "known_site_browser_url"
    assert value.intent_frame.followup_type.value == "NEW_REQUEST"


@pytest.mark.parametrize(
    ("text", "platform", "query", "expected_url"),
    [
        (
            "bota pra tocar a musica trust bothbirds",
            "youtube",
            "trust bothbirds",
            "https://www.youtube.com/results?search_query=trust+bothbirds",
        ),
        (
            "toque Trust Bothbirds no Spotify",
            "spotify",
            "Trust Bothbirds",
            "https://open.spotify.com/search/Trust%20Bothbirds",
        ),
        (
            "abra a música Trust do Bothbirds no YouTube",
            "youtube",
            "Trust do Bothbirds",
            "https://www.youtube.com/results?search_query=Trust+do+Bothbirds",
        ),
    ],
)
def test_music_command_opens_requested_platform(text, platform, query, expected_url):
    request = normalize_music_open_request(text)
    assert request is not None
    assert request.platform == platform
    assert request.query == query
    assert request.url == expected_url

    value = decide(text)
    assert value.intent is Intent.WEB_OPEN
    assert value.tools == ("web_open_browser",)
    assert value.target == expected_url
    assert value.reason_code == "music_browser_search"


@pytest.mark.parametrize(
    ("text", "platform", "query"),
    [
        ("toca aí Trust Bothbirds", "youtube", "Trust Bothbirds"),
        ("bote pra tocar a faixa Trust Bothbirds", "youtube", "Trust Bothbirds"),
        ("coloque essa música Trust Bothbirds no Spotify", "spotify", "Trust Bothbirds"),
        ("põe Trust Bothbirds no yt", "youtube", "Trust Bothbirds"),
        ("ponha tal música Trust Bothbirds", "youtube", "Trust Bothbirds"),
        ("reproduz Trust Bothbirds usando o Spotify", "spotify", "Trust Bothbirds"),
        ("quero ouvir Trust Bothbirds", "youtube", "Trust Bothbirds"),
        ("eu queria escutar Trust Bothbirds no Spotify", "spotify", "Trust Bothbirds"),
        ("escuta Trust Bothbirds no YouTube Music", "youtube", "Trust Bothbirds"),
        ("solta o som Trust Bothbirds", "youtube", "Trust Bothbirds"),
    ],
)
def test_music_command_understands_natural_variations(text, platform, query):
    request = normalize_music_open_request(text)
    assert request is not None
    assert request.platform == platform
    assert request.query == query


def test_non_music_command_is_not_reclassified_as_media():
    assert normalize_music_open_request("bota o arquivo no projeto") is None
    assert normalize_music_open_request("coloque isso na pasta") is None


def test_bare_site_label_reuses_only_the_immediately_previous_exact_url():
    prior = SimpleNamespace(last_user_text="abra https://example.com")
    assert normalize_contextual_web_open_request("abra o example", prior) == (
        "https://example.com"
    )
    value = decide(
        "abra o example",
        last_user_text="abra https://example.com",
    )
    assert value.intent is Intent.WEB_OPEN
    assert value.tools == ("web_open_browser",)
    assert value.target == "https://example.com"


@pytest.mark.parametrize(
    "text",
    ["então abra", "entao abre", "agora abra", "pode abrir", "abra isso"],
)
def test_targetless_open_followup_reuses_immediately_previous_exact_url(text):
    prior = SimpleNamespace(last_user_text="abra https://example.com/path")
    assert normalize_contextual_web_open_request(text, prior) == (
        "https://example.com/path"
    )
    value = decide(text, last_user_text="abra https://example.com/path")
    assert value.intent is Intent.WEB_OPEN
    assert value.tools == ("web_open_browser",)
    assert value.target == "https://example.com/path"
    assert value.reason_code == "contextual_browser_url"
    assert value.intent_frame.followup_type.value == "REFERENCE_FOLLOWUP"


def test_targetless_open_without_previous_explicit_url_still_clarifies():
    prior = SimpleNamespace(last_user_text="pesquise sobre Python")
    assert normalize_contextual_web_open_request("então abra", prior) is None


def test_targetless_open_followup_works_with_recorded_conversation_state():
    policy = AgentDecisionPolicy()
    first = policy.decide("abra https://example.com")
    policy.record_decision(first, "abra https://example.com")
    policy.record_tool_result(
        "web_open_browser",
        {"url": "https://example.com"},
        {"ok": True, "url": "https://example.com/"},
    )

    second = policy.decide("então abra")

    assert second.intent is Intent.WEB_OPEN
    assert second.target == "https://example.com"
    assert second.reason_code == "contextual_browser_url"


def test_bare_site_label_never_guesses_a_domain_or_changes_host():
    assert normalize_contextual_web_open_request(
        "abra o outrosite", SimpleNamespace(last_user_text=None)
    ) is None
    prior = SimpleNamespace(last_user_text="abra https://example.com")
    assert normalize_contextual_web_open_request("abra o outrosite", prior) is None


def test_browser_request_reuses_the_previous_explicit_url_without_guessing():
    value = decide(
        "abre ai no meu navegador cara",
        last_user_text="abra https://www.xvideos.com/",
    )
    assert normalize_browser_open_request(
        "abre ai no meu navegador cara",
        SimpleNamespace(
            last_user_text="abra https://www.xvideos.com/",
            recent_entities=(),
        ),
    ) == "https://www.xvideos.com/"
    assert value.intent is Intent.WEB_OPEN
    assert value.tools == ("web_open_browser",)
    assert value.target == "https://www.xvideos.com/"
    assert value.reason_code == "explicit_browser_url"


def test_browser_request_requires_one_explicit_or_validated_url():
    value = decide("abre ai no meu navegador cara")
    assert value.intent is Intent.CLARIFY
    assert value.tools == ()
    assert value.reason_code == "browser_url_missing_or_ambiguous"


def test_explicit_overrides_focus_and_auto_escalation_is_not_inferred():
    focused = {"focused_agent": "codex", "codex_job": {"status": "running"}}
    assert decide("pergunta ao DeepSeek sobre isso", **focused).intent == Intent.DEEPSEEK_DELEGATE
    direct = decide("isso parece uma arquitetura dificil?", **focused)
    assert "delegate_to_deepseek" not in direct.tools
    own = decide("responde voce mesmo sem usar o codex", **focused)
    assert own.intent == Intent.ANSWER_DIRECTLY
    assert own.user_override == "qwen_only"


def test_hardware_queries_use_live_telemetry_tool():
    temperature = decide("qual a temperatura da minha CPU?")
    usb = decide("quantos dispositivos USB estão conectados?")

    assert temperature.intent == Intent.HARDWARE_STATUS
    assert temperature.tools == ("get_hardware_telemetry",)
    assert usb.intent == Intent.HARDWARE_STATUS
    assert usb.tools == ("get_hardware_telemetry",)


def test_application_commands_get_dedicated_tools():
    assert decide("liste os aplicativos instalados").tools == ("list_installed_applications",)
    assert decide("abra o aplicativo Calculadora").tools == ("open_application",)
    assert decide("agende o Chrome para abrir amanhã").tools == ("schedule_application",)


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
