from __future__ import annotations

import json
import os
from pathlib import Path

from tern.orchestrator.codex_history import (
    MAX_LINE_CHARS,
    CodexCliHistory,
    parse_session_file,
)


def _line(record_type: str, payload: dict, timestamp: str | None = None) -> str:
    return json.dumps(
        {
            "timestamp": timestamp or "2026-09-10T02:20:00.000Z",
            "ordinal": 0,
            "type": record_type,
            "payload": payload,
        },
        ensure_ascii=False,
    )


def _write_session(
    codex_home: Path,
    name: str,
    *,
    session_id: str,
    started_at: str,
    cwd: str,
    user_texts: list[str],
    mtime: float,
) -> Path:
    path = codex_home / "sessions" / "2026" / "09" / "10" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        _line(
            "session_meta",
            {
                "session_id": session_id,
                "id": session_id,
                "timestamp": started_at,
                "cwd": cwd,
                "originator": "codex-tui",
                "cli_version": "0.153.4",
            },
            started_at,
        ),
    ]
    for text in user_texts:
        records.append(
            _line(
                "response_item",
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            )
        )
    records.append(
        _line(
            "response_item",
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "resposta do codex"}],
            },
        )
    )
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _sample_home(tmp_path: Path) -> Path:
    codex_home = tmp_path / ".codex"
    _write_session(
        codex_home,
        "rollout-new.jsonl",
        session_id="session-nova",
        started_at="2026-09-10T02:20:00.000Z",
        cwd="C:\\Users\\User",
        user_texts=["explique recursao"],
        mtime=300.0,
    )
    _write_session(
        codex_home,
        "rollout-old.jsonl",
        session_id="session-antiga",
        started_at="2026-09-01T02:20:00.000Z",
        cwd="D:\\JARVIS",
        user_texts=["<environment_context>\n  <cwd>D:\\JARVIS</cwd>\n</environment_context>", "corrija o bug do bridge"],
        mtime=100.0,
    )
    return codex_home


def test_list_recent_orders_by_mtime_and_summarizes(tmp_path):
    history = CodexCliHistory(_sample_home(tmp_path))

    result = history.list_recent(limit=10)

    assert result["ok"]
    assert result["count"] == 2
    sessions = result["sessions"]
    assert sessions[0]["session_id"] == "session-nova"
    assert sessions[0]["title"] == "explique recursao"
    assert sessions[0]["user_messages"] == 1
    assert sessions[1]["session_id"] == "session-antiga"
    assert sessions[1]["user_messages"] == 1
    assert "<environment_context>" not in sessions[1]["title"]
    assert sessions[1]["title"] == "corrija o bug do bridge"


def test_list_recent_filters_by_project_dir(tmp_path):
    history = CodexCliHistory(_sample_home(tmp_path))

    result = history.list_recent(limit=10, project_dir="D:\\JARVIS")

    assert result["count"] == 1
    assert result["sessions"][0]["session_id"] == "session-antiga"


def test_read_session_by_index_and_session_id(tmp_path):
    history = CodexCliHistory(_sample_home(tmp_path))

    by_index = history.read_session("1")
    assert by_index["ok"]
    assert by_index["session_id"] == "session-nova"
    assert [item["role"] for item in by_index["messages"]] == ["user", "assistant"]

    by_id = history.read_session("session-ant")
    assert by_id["ok"]
    assert by_id["session_id"] == "session-antiga"
    assert by_id["messages"][0]["text"] == "corrija o bug do bridge"


def test_read_session_unknown_selector(tmp_path):
    history = CodexCliHistory(_sample_home(tmp_path))

    assert not history.read_session("99")["ok"]
    assert not history.read_session("nao existe")["ok"]
    assert history.read_session("99")["error"] == "session_not_found"


def test_read_session_turn_limit_keeps_last_user_turns(tmp_path):
    codex_home = tmp_path / ".codex"
    _write_session(
        codex_home,
        "rollout-turns.jsonl",
        session_id="session-turns",
        started_at="2026-09-10T02:20:00.000Z",
        cwd="D:\\JARVIS",
        user_texts=["primeira", "segunda", "terceira"],
        mtime=200.0,
    )
    history = CodexCliHistory(codex_home)

    result = history.read_session("1", turn_limit=2)

    assert result["ok"]
    user_texts = [item["text"] for item in result["messages"] if item["role"] == "user"]
    assert user_texts == ["segunda", "terceira"]


def test_read_session_respects_max_chars(tmp_path):
    codex_home = tmp_path / ".codex"
    _write_session(
        codex_home,
        "rollout-chars.jsonl",
        session_id="session-chars",
        started_at="2026-09-10T02:20:00.000Z",
        cwd="D:\\JARVIS",
        user_texts=["texto longo " * 200],
        mtime=200.0,
    )
    history = CodexCliHistory(codex_home)

    result = history.read_session("1", max_chars=100)

    assert result["ok"]
    assert result["max_chars"] == 100
    assert len(json.dumps(result["messages"])) < 400


def test_giant_system_lines_are_skipped(tmp_path):
    codex_home = tmp_path / ".codex"
    path = codex_home / "sessions" / "2026" / "09" / "10" / "rollout-giant.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    giant = "x" * (MAX_LINE_CHARS + 100)
    lines = [
        _line(
            "session_meta",
            {
                "session_id": "session-giant",
                "timestamp": "2026-09-10T02:20:00.000Z",
                "cwd": "D:\\JARVIS",
                "originator": "codex-tui",
                "cli_version": "0.153.4",
            },
        ),
        _line(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": giant}],
            },
        ),
        _line(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "mensagem normal"}],
            },
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    parsed = parse_session_file(path)

    assert parsed["ok"]
    assert [item["text"] for item in parsed["messages"]] == ["mensagem normal"]


def test_missing_sessions_directory_is_not_an_error(tmp_path):
    history = CodexCliHistory(tmp_path / ".codex")

    assert not history.available()
    assert history.list_recent()["ok"]
    assert history.list_recent()["sessions"] == []
    assert history.read_session("1")["error"] == "no_codex_sessions"
