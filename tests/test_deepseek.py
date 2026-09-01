from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tern.orchestrator.agent import Supervisor, _deepseek_intent
from tern.orchestrator.codex import CodexResult
from tern.orchestrator.config import load_settings
from tern.orchestrator.deepseek import DeepSeekClient, DeepSeekError, DeepSeekSessionManager
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry


class Projects:
    def __init__(self, roots: dict[str, Path]):
        self.roots = {key: value.resolve() for key, value in roots.items()}
        self.current = next(iter(self.roots.values()))

    def resolve(self, *, query=None, path_hint=None, **_kwargs):
        if path_hint:
            path = Path(path_hint).resolve()
            if path in self.roots.values():
                self.current = path
                return {"ok": True, "root": str(path), "error": None}
            return {"ok": False, "error": "project_not_registered"}
        if query:
            normalized = str(query).casefold()
            for alias, root in self.roots.items():
                if alias.casefold() in normalized:
                    self.current = root
                    return {"ok": True, "root": str(root), "error": None}
        return {"ok": True, "root": str(self.current), "error": None}


class StubClient:
    def __init__(self, responses=None, *, enabled=True, api_key="secret", model="deepseek-test"):
        self.enabled = enabled
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com"
        self.model = model
        self.timeout_seconds = 5
        self.responses = list(responses or ["ok"])
        self.calls = []

    @property
    def configured(self):
        return bool(self.api_key and self.model)

    def chat(self, messages, *, cancel_event=None):
        self.calls.append(messages)
        content = self.responses.pop(0) if self.responses else "ok"
        return {
            "response": content,
            "model": self.model,
            "usage": {
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
                "cache_hit_tokens": 1,
                "cache_miss_tokens": 3,
                "reasoning_tokens": 1,
            },
            "finish_reason": "stop",
        }


def manager(tmp_path: Path, *, client=None, roots=None, recent=20, context=60_000):
    project = tmp_path / "tern"
    project.mkdir(parents=True, exist_ok=True)
    roots = roots or {"tern": project}
    return DeepSeekSessionManager(
        client=client or StubClient(),
        state_dir=tmp_path / "state",
        projects=Projects(roots),
        max_recent_turns=recent,
        max_context_characters=context,
    ), project.resolve()


class HttpResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.value).encode()


def response_value(content="resposta"):
    return {
        "id": "one",
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "total_tokens": 17,
            "prompt_cache_hit_tokens": 3,
            "prompt_cache_miss_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }


def test_configuration_defaults_enable_current_deepseek_model():
    settings = load_settings({})
    assert settings.deepseek_enabled
    assert settings.deepseek_api_key is None
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert not settings.deepseek_auto_escalation
    assert settings.deepseek_session_max_recent_turns == 20


def test_client_missing_key_and_disabled_do_not_call_http():
    calls = []
    for enabled, key, expected in (
        (False, "secret", "deepseek_disabled"),
        (True, None, "deepseek_api_key_missing"),
    ):
        client = DeepSeekClient(
            enabled=enabled,
            api_key=key,
            base_url="https://api.deepseek.com",
            model="model",
            opener=lambda *_a, **_k: calls.append(True),
        )
        with pytest.raises(DeepSeekError) as caught:
            client.chat([{"role": "user", "content": "oi"}])
        assert caught.value.code == expected
    assert calls == []


def test_http_client_payload_protocol_and_usage():
    captured = {}

    def opener(request, **kwargs):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = kwargs["timeout"]
        return HttpResponse(response_value())

    client = DeepSeekClient(
        enabled=True,
        api_key="super-secret",
        base_url="https://api.deepseek.com",
        model="configured-model",
        timeout_seconds=9,
        opener=opener,
    )
    result = client.chat([{"role": "user", "content": "oi"}])
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["payload"]["model"] == "configured-model"
    assert "tools" not in captured["payload"]
    assert captured["timeout"] == 9
    assert result["usage"] == {
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
        "cache_hit_tokens": 3,
        "cache_miss_tokens": 7,
        "reasoning_tokens": 4,
    }


@pytest.mark.parametrize(
    ("status", "detail", "code"),
    [
        (401, "invalid key", "deepseek_auth_failed"),
        (404, "model missing", "deepseek_model_not_found"),
        (429, "busy", "deepseek_rate_limited"),
        (422, "context length exceeded", "deepseek_context_too_large"),
        (500, "busy", "deepseek_api_error"),
    ],
)
def test_structured_http_errors(status, detail, code):
    def opener(*_args, **_kwargs):
        body = io.BytesIO(json.dumps({"error": {"message": detail}}).encode())
        raise HTTPError("https://api.deepseek.com", status, detail, {}, body)

    client = DeepSeekClient(
        enabled=True,
        api_key="secret",
        base_url="https://api.deepseek.com",
        model="model",
        max_retries=0,
        opener=opener,
    )
    with pytest.raises(DeepSeekError) as caught:
        client.chat([{"role": "user", "content": "oi"}])
    assert caught.value.code == code


def test_limited_retries_for_transient_error():
    attempts = []

    def opener(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise HTTPError("url", 503, "busy", {}, io.BytesIO(b"{}"))
        return HttpResponse(response_value())

    client = DeepSeekClient(
        enabled=True,
        api_key="secret",
        base_url="https://api.deepseek.com",
        model="model",
        max_retries=2,
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    assert client.chat([{"role": "user", "content": "oi"}])["response"] == "resposta"
    assert len(attempts) == 3


def test_invalid_response_is_structured():
    client = DeepSeekClient(
        enabled=True,
        api_key="secret",
        base_url="https://api.deepseek.com",
        model="model",
        opener=lambda *_a, **_k: HttpResponse({"choices": []}),
    )
    with pytest.raises(DeepSeekError) as caught:
        client.chat([{"role": "user", "content": "oi"}])
    assert caught.value.code == "deepseek_invalid_response"


def test_new_session_resume_and_persistence(tmp_path):
    session, project = manager(tmp_path, client=StubClient(["um", "dois"]))
    first = session.delegate("primeira", project_path=str(project), source="human")
    second = session.delegate("segunda", project_path=str(project), source="qwen")
    assert first["session_id"] == second["session_id"]
    stored = json.loads(session.path.read_text(encoding="utf-8"))
    messages = stored["sessions"][0]["messages"]
    assert [item["source"] for item in messages] == ["human", "deepseek", "qwen", "deepseek"]
    assert all({"id", "source", "role", "content", "created_at"} <= item.keys() for item in messages)


def test_new_session_when_continue_is_false(tmp_path):
    session, project = manager(tmp_path, client=StubClient(["um", "dois"]))
    first = session.delegate("primeira", project_path=str(project))
    second = session.delegate(
        "segunda", project_path=str(project), continue_current_session=False
    )
    assert first["session_id"] != second["session_id"]


def test_configured_model_change_starts_new_session_instead_of_mixing_history(tmp_path):
    client = StubClient(["um"], model="model-a")
    session, project = manager(tmp_path, client=client)
    first = session.delegate("primeira", project_path=str(project))
    client.model = "model-b"
    client.responses.append("dois")
    second = session.delegate("segunda", project_path=str(project))
    assert first["session_id"] != second["session_id"]
    assert second["model"] == "model-b"


def test_one_active_session_per_project(tmp_path):
    first_project = tmp_path / "tern"
    second_project = tmp_path / "llama.cpp"
    first_project.mkdir()
    second_project.mkdir()
    session, _ = manager(
        tmp_path,
        client=StubClient(["a", "b", "c"]),
        roots={"tern": first_project, "llama": second_project},
    )
    a = session.delegate("a", project_path=str(first_project))
    b = session.delegate("b", project_path=str(second_project))
    c = session.delegate("c", project_path=str(first_project))
    assert a["session_id"] == c["session_id"] != b["session_id"]


def test_review_reads_persistence_without_api_call(tmp_path):
    client = StubClient(["DS-HUMAN-OK"])
    session, project = manager(tmp_path, client=client)
    session.delegate("Responda apenas DS-HUMAN-OK", project_path=str(project), source="human")
    calls = len(client.calls)
    result = session.review_session(project_path=str(project), turn_limit="10")
    assert result["ok"] and result["last_response"] == "DS-HUMAN-OK"
    assert result["summary_source"][-1]["source"] == "human"
    assert len(client.calls) == calls


@pytest.mark.parametrize("limit,reviewed", [(1, 1), (3, 3), (10, 4)])
def test_review_turn_limits(tmp_path, limit, reviewed):
    session, project = manager(tmp_path, client=StubClient([str(i) for i in range(4)]))
    for index in range(4):
        session.delegate(str(index), project_path=str(project))
    result = session.review_session(project_path=str(project), turn_limit=limit)
    assert result["turns_available"] == 4
    assert result["turns_reviewed"] == reviewed


def test_review_missing_or_empty_session(tmp_path):
    session, project = manager(tmp_path)
    assert session.review_session(project_path=str(project))["error"] == "deepseek_session_not_found"
    created = session.new_session(str(project))
    result = session.review_session(project_path=str(project))
    assert created["ok"] and result["turns_available"] == result["turns_reviewed"] == 0


def test_large_history_rolls_summary_but_preserves_messages(tmp_path):
    client = StubClient([f"r{i}" for i in range(8)])
    session, project = manager(tmp_path, client=client, recent=2)
    for index in range(8):
        session.delegate(f"pergunta {index}", project_path=str(project))
    stored = json.loads(session.path.read_text(encoding="utf-8"))["sessions"][0]
    assert len(stored["messages"]) == 16
    assert stored["summary"]
    assert stored["summary_message_count"] == 12
    # A small unsummarized batch may accompany the exact recent window so no
    # message disappears between summary updates.
    assert len(client.calls[-1]) <= 2 + 4 + 4


def test_context_is_safely_truncated_and_size_logged(tmp_path):
    client = StubClient(["ok"])
    session, project = manager(tmp_path, client=client, context=4_000)
    session.delegate("fim", project_path=str(project), context="x" * 50_000)
    assert sum(len(item["content"]) for item in client.calls[0]) <= 4_000


def test_current_message_too_large_is_rejected_before_api(tmp_path):
    client = StubClient(["unused"])
    session, project = manager(tmp_path, client=client, context=4_000)
    result = session.delegate("x" * 4_000, project_path=str(project))
    assert result["error"] == "deepseek_context_too_large"
    assert client.calls == []


def test_current_message_above_history_limit_is_sent_without_truncation(tmp_path):
    client = StubClient(["ok"])
    session, project = manager(tmp_path, client=client)
    task = "preserve-exact:" + ("x" * 9_000)

    result = session.delegate(task, project_path=str(project))

    assert result["ok"]
    assert client.calls[0][-1] == {"role": "user", "content": task}


def test_current_message_takes_priority_over_temporary_context(tmp_path):
    client = StubClient(["ok"])
    session, project = manager(tmp_path, client=client, context=4_000)
    task = "authoritative:" + ("x" * 3_100)

    result = session.delegate(
        task,
        project_path=str(project),
        context="supplemental:" + ("y" * 20_000),
    )

    assert result["ok"]
    assert client.calls[0][-1] == {"role": "user", "content": task}
    assert sum(len(item["content"]) for item in client.calls[0]) <= 4_000


def test_usage_is_returned_and_persisted(tmp_path):
    session, project = manager(tmp_path)
    result = session.delegate("teste", project_path=str(project))
    stored = json.loads(session.path.read_text(encoding="utf-8"))["sessions"][0]
    assert result["usage"]["reasoning_tokens"] == 1
    assert stored["last_usage"] == result["usage"]


def test_cancel_discards_late_response_and_keeps_session_usable(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class SlowClient(StubClient):
        def chat(self, messages, *, cancel_event=None):
            self.calls.append(messages)
            started.set()
            release.wait(3)
            return super().chat(messages, cancel_event=cancel_event)

    client = SlowClient(["tardia", "valida"])
    session, project = manager(tmp_path, client=client)
    box = {}
    worker = threading.Thread(
        target=lambda: box.update(
            result=session.delegate("lenta", project_path=str(project))
        )
    )
    worker.start()
    assert started.wait(1)
    assert session.cancel(project_path=str(project))["ok"]
    release.set()
    worker.join(3)
    assert box["result"]["error"] == "deepseek_cancelled"
    stored = json.loads(session.path.read_text(encoding="utf-8"))["sessions"][0]
    assert not any(item.get("content") == "tardia" for item in stored["messages"])
    assert session.delegate("nova", project_path=str(project))["ok"]


def test_deepseek_has_no_local_tools_or_filesystem_access(tmp_path):
    client = StubClient(["conselho"])
    session, project = manager(tmp_path, client=client)
    session.delegate("analise", project_path=str(project), context="codigo relevante")
    assert not hasattr(session.client, "filesystem")
    assert all(set(item) == {"role", "content"} for item in client.calls[0])


def test_disabled_or_missing_key_does_not_break_session_storage(tmp_path):
    for client, expected in (
        (StubClient(enabled=False), "deepseek_disabled"),
        (StubClient(api_key=None), "deepseek_api_key_missing"),
    ):
        session, project = manager(tmp_path / expected, client=client)
        result = session.delegate("teste", project_path=str(project))
        assert result["error"] == expected
        assert not session.path.exists()


def test_api_key_never_appears_in_action_log(tmp_path):
    secret = "sk-this-must-never-appear"
    client = StubClient(["ok"], api_key=secret)
    session, project = manager(tmp_path, client=client)
    session.logger = ActionLogger(tmp_path / "actions.jsonl")
    session.delegate("teste", project_path=str(project))
    assert secret not in (tmp_path / "actions.jsonl").read_text(encoding="utf-8")


def test_explicit_intent_distinguishes_delegate_and_review():
    assert _deepseek_intent("Pergunta ao DeepSeek qual abordagem e melhor") == "delegate_to_deepseek"
    assert _deepseek_intent("O que o DeepSeek respondeu por ultimo?") == "review_deepseek_session"
    assert _deepseek_intent("Qual abordagem e melhor?") is None


class FakeCodex:
    timeout = 5

    def claim_completed_results(self):
        return []

    def delegate_to_codex(self, **_kwargs):
        return CodexResult(True, "thread", "turn", "completed", "revisado", None, 1)

    def review_session(self, **_kwargs):
        return {
            "ok": True,
            "thread_id": "thread",
            "summary_source": [{"requested": "x", "result": "feito", "status": "completed"}],
            "new_turn_started": False,
        }


class ModelResponses:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.tool_names = []

    def chat(self, _messages, *, tools=None):
        self.tool_names.append(
            {item["function"]["name"] for item in (tools or [])}
        )
        return {"choices": [{"message": next(self.messages)}], "usage": {}}


def tool_call(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": name,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def integrated_registry(tmp_path, client):
    root = tmp_path / "tern"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    policy = PathPolicy((root,))
    logger = ActionLogger(tmp_path / "actions.jsonl")
    from tern.orchestrator.projects import ProjectRegistry

    projects = ProjectRegistry(policy, tmp_path / "state", codex=FakeCodex())
    deepseek = DeepSeekSessionManager(
        client=client,
        state_dir=tmp_path / "state",
        projects=projects,
        logger=logger,
    )
    return ToolRegistry(
        policy=policy,
        logger=logger,
        codex=FakeCodex(),
        max_output_bytes=131072,
        projects=projects,
        deepseek=deepseek,
    ), root


def test_qwen_to_deepseek_and_human_to_deepseek_share_session(tmp_path):
    client = StubClient(["DEEPSEEK-HUMANO-OK", "DEEPSEEK-QWEN-OK"])
    tools, root = integrated_registry(tmp_path, client)
    human = tools.deepseek.delegate("humano", project_path=str(root), source="human")
    qwen = tools.execute(
        "delegate_to_deepseek",
        {"task": "qwen", "project_path": str(root)},
        context={"user_text": "Pergunte ao DeepSeek no projeto tern"},
    )
    assert human["session_id"] == qwen["session_id"]
    review = tools.execute(
        "review_deepseek_session",
        {"project_path": str(root), "turn_limit": 10},
    )
    assert review["last_response"] == "DEEPSEEK-QWEN-OK"
    assert {turn["source"] for turn in review["summary_source"]} == {"human", "qwen"}


def test_deepseek_tool_receives_the_same_preserved_delegation_protocol(tmp_path):
    client = StubClient(["ok"])
    tools, root = integrated_registry(tmp_path, client)
    original = "Analise apenas auth.py e não proponha alterações fora dele."

    result = tools.execute(
        "delegate_to_deepseek",
        {
            "task": "avalie o repositório inteiro",
            "project_path": str(root),
        },
        context={
            "user_text": "Pergunte ao DeepSeek no projeto tern",
            "original_user_text": original,
            "delegation_references": ["auth.py"],
            "turn_id": "turn-preservation",
        },
    )

    assert result["ok"]
    payload = json.loads(client.calls[0][-1]["content"])
    assert payload["schema"] == "jarvis.delegation_request.v1"
    assert payload["requested_agent"] == "deepseek"
    assert payload["task"] == original
    assert payload["references"] == ["auth.py"]
    assert "avalie o repositório inteiro" not in client.calls[0][-1]["content"]


def test_deepseek_tool_content_is_not_duplicated_in_logs_or_action_history(tmp_path):
    client = StubClient(["private response"])
    tools, root = integrated_registry(tmp_path, client)
    result = tools.execute(
        "delegate_to_deepseek",
        {
            "task": "private question",
            "project_path": str(root),
            "context": "private context",
        },
        context={"user_text": "Pergunte ao DeepSeek no projeto tern"},
    )
    assert result["ok"]
    logs = (tmp_path / "actions.jsonl").read_text(encoding="utf-8")
    pending = (tmp_path / "pending-actions.json").read_text(encoding="utf-8")
    for secret_text in ("private question", "private context", "private response"):
        assert secret_text not in logs
        assert secret_text not in pending


def test_no_automatic_deepseek_tool_is_exposed_by_default(tmp_path):
    tools, _root = integrated_registry(tmp_path, StubClient())
    model = ModelResponses([{"role": "assistant", "content": "resposta local"}])
    result = Supervisor(load_settings({}), model, tools).run("Qual abordagem e melhor?")
    assert result["tool_calls"] == 0
    assert "delegate_to_deepseek" not in model.tool_names[0]
    assert "review_deepseek_session" not in model.tool_names[0]


def test_codex_history_can_be_compacted_then_sent_to_deepseek(tmp_path):
    tools, root = integrated_registry(tmp_path, StubClient(["avaliacao curta"]))
    model = ModelResponses([
        tool_call("review_codex_session", {"project_path": str(root), "turn_limit": 3}),
        tool_call(
            "delegate_to_deepseek",
            {"task": "avalie", "project_path": str(root), "context": "Codex recent activity: feito"},
        ),
        {"role": "assistant", "content": "avaliacao curta"},
    ])
    result = Supervisor(load_settings({"MODEL_MAX_TOOL_CALLS": "4"}), model, tools).run(
        "Mostre para o DeepSeek o que o Codex fez por ultimo e peca uma avaliacao curta"
    )
    assert result["ok"] and result["tool_calls"] == 2, result
    assert len(tools.deepseek.client.calls) == 1
    assert any(
        "Codex recent activity" in item["content"]
        for item in tools.deepseek.client.calls[0]
    )


def test_deepseek_advice_then_codex_is_qwen_mediated(tmp_path):
    tools, root = integrated_registry(tmp_path, StubClient(["sugestao segura"]))
    model = ModelResponses([
        tool_call("delegate_to_deepseek", {"task": "sugira", "project_path": str(root)}),
        tool_call(
            "delegate_to_codex",
            {"task": "Revise a sugestao: sugestao segura", "project_path": str(root), "wait": True},
        ),
        {"role": "assistant", "content": "fluxo concluido"},
    ])
    result = Supervisor(load_settings({"MODEL_MAX_TOOL_CALLS": "4"}), model, tools).run(
        "Pergunte ao DeepSeek e depois peca ao Codex apenas revisar a sugestao"
    )
    assert result["ok"] and result["tool_calls"] == 2
    assert len(tools.deepseek.client.calls) == 1
