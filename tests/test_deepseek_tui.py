from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tern.orchestrator.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    DeepSeekService,
    DeepSeekSessionManager,
)
from tern.orchestrator.deepseek_tui import DeepSeekTUI


class Projects:
    def __init__(self, values):
        self.values = {key: Path(value).resolve() for key, value in values.items()}
        self.current = next(iter(self.values.values()))

    def resolve(self, *, query=None, path_hint=None, **_kwargs):
        if path_hint:
            candidate = Path(path_hint).resolve()
            if candidate in self.values.values():
                self.current = candidate
                return {"ok": True, "root": str(candidate), "error": None}
        if query:
            for key, path in self.values.items():
                if key in query.casefold():
                    self.current = path
                    return {"ok": True, "root": str(path), "error": None}
        return {"ok": True, "root": str(self.current), "error": None}


class StreamingClient:
    enabled = True
    api_key = "secret"
    base_url = "https://api.deepseek.com"
    model = "deepseek-test"
    timeout_seconds = 5

    def __init__(self, responses=None):
        self.responses = list(responses or ["ok"])
        self.calls = []

    @property
    def configured(self):
        return True

    def stream_chat(self, messages, *, on_delta, cancel_event=None):
        self.calls.append(messages)
        response = self.responses.pop(0)
        for part in [response[index : index + 2] for index in range(0, len(response), 2)]:
            if cancel_event and cancel_event.is_set():
                raise DeepSeekError("deepseek_cancelled", "cancelled")
            on_delta(part)
        return {
            "response": response,
            "model": self.model,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "cache_hit_tokens": 1,
                "cache_miss_tokens": 9,
                "reasoning_tokens": 1,
            },
            "finish_reason": "stop",
        }

    def chat(self, messages, *, cancel_event=None):
        return self.stream_chat(messages, on_delta=lambda _value: None, cancel_event=cancel_event)


class Codex:
    def __init__(self):
        self.reads = 0
        self.starts = 0

    def review_session(self, **_kwargs):
        self.reads += 1
        return {
            "ok": True,
            "summary_source": [
                {
                    "requested": "corrigir bridge",
                    "final_response": "bridge corrigido; 5 testes passaram",
                    "status": "completed",
                }
            ],
        }


def create(tmp_path, *, client=None, enabled=True, key="secret"):
    tern = tmp_path / "tern"
    llama = tmp_path / "llama.cpp"
    tern.mkdir()
    llama.mkdir()
    client = client or StreamingClient()
    client.enabled = enabled
    client.api_key = key
    projects = Projects({"tern": tern, "llama": llama})
    manager = DeepSeekSessionManager(
        client=client,
        state_dir=tmp_path / "state",
        projects=projects,
        max_recent_turns=2,
    )
    codex = Codex()
    service = DeepSeekService(manager, project_path=str(tern), codex=codex)
    return manager, service, codex, tern.resolve(), llama.resolve()


class SSE:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)


def test_http_sse_stream_emits_in_place_and_parses_usage():
    chunks = [
        {"model": "model", "choices": [{"delta": {"content": "DS-"}, "finish_reason": None}]},
        {"model": "model", "choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
    ]
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks] + [b"data: [DONE]\n"]
    client = DeepSeekClient(
        enabled=True,
        api_key="secret",
        base_url="https://api.deepseek.com",
        model="model",
        opener=lambda *_a, **_k: SSE(lines),
    )
    deltas = []
    result = client.stream_chat(
        [{"role": "user", "content": "oi"}], on_delta=deltas.append
    )
    assert deltas == ["DS-", "OK"]
    assert result["response"] == "DS-OK"
    assert result["usage"]["input_tokens"] == 4


def test_sse_cancellation_discards_stream():
    event = threading.Event()
    lines = [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"late"}}]}\n',
    ]
    client = DeepSeekClient(
        enabled=True,
        api_key="secret",
        base_url="https://api.deepseek.com",
        model="model",
        opener=lambda *_a, **_k: SSE(lines),
    )

    def delta(_text):
        event.set()

    with pytest.raises(DeepSeekError) as caught:
        client.stream_chat(
            [{"role": "user", "content": "oi"}],
            on_delta=delta,
            cancel_event=event,
        )
    assert caught.value.code == "deepseek_cancelled"


def test_tui_startup_opens_history_without_api_request(tmp_path):
    client = StreamingClient(["unused"])
    _manager, service, _codex, project, _llama = create(tmp_path, client=client)
    tui = DeepSeekTUI(service)
    assert client.calls == []
    assert tui.session["project"] == str(project)
    assert "DeepSeek" in tui.render_text()


@pytest.mark.parametrize(
    ("enabled", "key", "expected"),
    [(False, "secret", "disabled"), (True, None, "API key not configured")],
)
def test_tui_opens_read_only_when_unavailable(tmp_path, enabled, key, expected):
    _manager, service, _codex, _project, _llama = create(
        tmp_path, enabled=enabled, key=key
    )
    tui = DeepSeekTUI(service)
    tui.handle_command("/status")
    assert expected.casefold() in tui.render_text().casefold()


def test_streamed_human_message_persists_and_reopens(tmp_path):
    client = StreamingClient(["DS-TUI-OK"])
    manager, service, _codex, project, _llama = create(tmp_path, client=client)
    deltas = []
    result = service.send("Responda apenas: DS-TUI-OK", on_delta=deltas.append)
    assert result["response"] == "DS-TUI-OK" and "".join(deltas) == "DS-TUI-OK"
    reopened = DeepSeekTUI(DeepSeekService(manager, project_path=str(project)))
    snapshot = reopened.render_text()
    assert "Responda apenas: DS-TUI-OK" in snapshot
    assert "DS-TUI-OK" in snapshot
    assert len(client.calls) == 1


def test_context_attachment_is_temporary_and_codex_read_only(tmp_path):
    client = StreamingClient(["avaliado"])
    manager, service, codex, project, _llama = create(tmp_path, client=client)
    service.open()
    attached = service.attach_codex(3)
    report = service.context_report()
    assert attached["ok"] and codex.reads == 1 and codex.starts == 0
    assert report["attachments"][0]["source"] == "codex_context"
    service.send("avalie", on_delta=lambda _value: None)
    stored = manager.history_messages(project_path=str(project))
    assert all("bridge corrigido" not in message["content"] for message in stored)
    assert service.context_report()["attachments"] == []


def test_clear_context_does_not_clear_history(tmp_path):
    manager, service, _codex, project, _llama = create(tmp_path)
    service.open()
    service.temporary_contexts.append(
        {"content": "temporary", "consumed": False, "label": "x", "estimated_tokens": 3}
    )
    before = manager.history_messages(project_path=str(project))
    assert service.clear_context()["removed"] == 1
    assert manager.history_messages(project_path=str(project)) == before


def test_tui_commands_sessions_new_project_context_usage_and_model(tmp_path):
    manager, service, _codex, tern, llama = create(tmp_path)
    tui = DeepSeekTUI(service)
    original = tui.session["session_id"]
    assert tui.handle_command("/context")["estimate"]
    assert tui.handle_command("/usage")["usage"].get("requests", 0) == 0
    created = tui.handle_command("/new")
    assert created["session_id"] != original
    sessions = tui.handle_command("/sessions")["sessions"]
    assert len(sessions) == 2
    assert tui.handle_command(f"/sessions {original[:8]}")["session_id"] == original
    switched = tui.handle_command("/project llama")
    assert switched["project"] == str(llama)
    assert tui.handle_command("/project tern")["project"] == str(tern)
    assert tui.handle_command("/model")["model"] == "deepseek-test"
    assert not tui.handle_command("/model unknown")["ok"]
    assert tui.handle_command("/history")["ok"]


def test_tui_codex_and_clear_context_commands(tmp_path):
    _manager, service, codex, _project, _llama = create(tmp_path)
    tui = DeepSeekTUI(service)
    assert tui.handle_command("/codex 3")["turns"] == 1
    assert codex.reads == 1 and codex.starts == 0
    assert tui.handle_command("/context")["attachments"]
    assert tui.handle_command("/clear-context")["removed"] == 1


def test_send_codex_is_confirmed_and_qwen_mediated(tmp_path):
    manager, service, _codex, project, _llama = create(
        tmp_path, client=StreamingClient(["recomendacao"])
    )
    service.open()
    service.send("pergunta", on_delta=lambda _value: None)
    calls = []
    tui = DeepSeekTUI(
        DeepSeekService(manager, project_path=str(project)),
        qwen_handler=lambda prompt: calls.append(prompt) or {"ok": True, "answer": "Qwen avaliou"},
    )
    pending = tui.handle_command("/send-codex")
    assert pending["confirmation_required"] and calls == []
    result = tui.handle_command("/send-codex confirm")
    assert result["ok"] and len(calls) == 1
    assert "Recomendacao DeepSeek" in calls[0]


def test_tui_identity_labels_qwen_human_deepseek_and_codex(tmp_path):
    manager, service, _codex, project, _llama = create(tmp_path)
    service.open()
    with manager.path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    session = state["sessions"][0]
    session["messages"] = [
        {"id": "1", "source": "human", "role": "user", "content": "H", "created_at": "now"},
        {"id": "2", "source": "qwen", "role": "user", "content": "Q", "created_at": "now"},
        {"id": "3", "source": "deepseek", "role": "assistant", "content": "D", "created_at": "now"},
        {"id": "4", "source": "codex_context", "role": "system", "content": "C", "created_at": "now"},
    ]
    with manager.lock_path.open("a"):
        pass
    manager._write_unlocked(state)
    tui = DeepSeekTUI(DeepSeekService(manager, project_path=str(project)))
    text = tui.render_text()
    assert all(label in text for label in ("Voce", "Qwen", "DeepSeek", "Contexto do Codex"))
