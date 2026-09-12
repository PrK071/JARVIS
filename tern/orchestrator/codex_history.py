"""Leitura local e read-only das sessoes de conversa do usuario com o Codex CLI.

O Codex CLI persiste cada sessao em ~/.codex/sessions/AAAA/MM/DD/rollout-*.jsonl.
Este modulo le esses arquivos sem iniciar o Codex, sem o App Server compartilhado
e sem escrever nada: so expoe as conversas do usuario (lista resumida e leitura
de uma sessao). Nenhuma thread do Jarvis e alterada aqui.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
DEFAULT_MAX_CHARS = 6000
MAX_MAX_CHARS = 32000
MAX_LINE_CHARS = 4_000_000
LIST_SCAN_LINE_LIMIT = 50_000
MAX_MESSAGE_TEXT_CHARS = 4000

_ENV_CONTEXT_RE = re.compile(
    r"<environment_context>.*?</environment_context>", re.DOTALL
)

_META_PATTERNS: tuple[tuple[str, str], ...] = (
    ("session_id", r'"session_id"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    ("timestamp", r'"timestamp"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    ("cwd", r'"cwd"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    ("originator", r'"originator"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    ("cli_version", r'"cli_version"\s*:\s*"((?:[^"\\]|\\.)*)"'),
)

_SESSION_META_RE = re.compile(r'"type"\s*:\s*"session_meta"')
_USER_ROLE_RE = re.compile(r'"role"\s*:\s*"user"')
_ASSISTANT_ROLE_RE = re.compile(r'"role"\s*:\s*"assistant"')


def default_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".codex"


def _unescape(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value


def _extract_meta(line: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for field, pattern in _META_PATTERNS:
        match = re.search(pattern, line)
        if match:
            meta[field] = _unescape(match.group(1))
    return meta


def _extract_message_text(payload: dict[str, Any], role: str) -> str:
    if not isinstance(payload, dict):
        return ""
    if payload.get("type") != "message" or payload.get("role") != role:
        return ""
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    key = "input_text" if role == "user" else "output_text"
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == key:
            value = item.get("text")
            if isinstance(value, str) and value.strip():
                parts.append(value)
    text = "\n".join(parts)
    if role == "user":
        text = _ENV_CONTEXT_RE.sub("", text)
    return text.strip()


def iter_session_files(codex_home: Path) -> Iterator[Path]:
    sessions_root = codex_home / "sessions"
    if not sessions_root.is_dir():
        return
    for path in sessions_root.rglob("rollout-*.jsonl"):
        if path.is_file():
            yield path


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def sorted_session_files(codex_home: Path) -> list[Path]:
    return sorted(
        iter_session_files(codex_home),
        key=_file_mtime,
        reverse=True,
    )


def parse_session_file(
    path: Path,
    *,
    roles: frozenset[str] = frozenset({"user", "assistant"}),
    line_limit: int | None = None,
) -> dict[str, Any]:
    """Le um rollout.jsonl e extrai meta e mensagens selecionadas.

    Linhas gigantes (instrucoes de sistema e skills) sao ignoradas sem parse
    de JSON para manter o consumo de memoria baixo.
    """
    meta: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    scanned_lines = 0
    truncated_scan = False
    last_activity: str | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                scanned_lines += 1
                if line_limit is not None and scanned_lines > line_limit:
                    truncated_scan = True
                    break
                line = raw.strip()
                if not line:
                    continue
                if _SESSION_META_RE.search(line) and not meta:
                    meta = _extract_meta(line)
                if _USER_ROLE_RE.search(line):
                    role = "user"
                elif _ASSISTANT_ROLE_RE.search(line):
                    role = "assistant"
                else:
                    role = None
                if role is None or role not in roles:
                    continue
                if len(line) > MAX_LINE_CHARS:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload")
                text = _extract_message_text(payload, role)
                if not text:
                    continue
                at = str(record.get("timestamp") or "")
                if at:
                    last_activity = at
                messages.append(
                    {
                        "role": role,
                        "text": text[:MAX_MESSAGE_TEXT_CHARS],
                        "text_truncated": len(text) > MAX_MESSAGE_TEXT_CHARS,
                        "at": at,
                    }
                )
    except OSError:
        return {
            "ok": False,
            "error": "session_file_unreadable",
            "path": str(path),
        }
    return {
        "ok": True,
        "path": str(path),
        "session_id": meta.get("session_id"),
        "started_at": meta.get("timestamp"),
        "cwd": meta.get("cwd"),
        "originator": meta.get("originator"),
        "cli_version": meta.get("cli_version"),
        "messages": messages,
        "scanned_lines": scanned_lines,
        "truncated_scan": truncated_scan,
        "last_activity": last_activity,
    }


def _summarize_user(parsed: dict[str, Any]) -> dict[str, Any]:
    user_messages = [
        item for item in parsed["messages"] if item.get("role") == "user"
    ]
    title = user_messages[0]["text"] if user_messages else ""
    last_user = user_messages[-1]["text"] if user_messages else ""
    assistant_count = sum(
        1 for item in parsed["messages"] if item.get("role") == "assistant"
    )
    return {
        "session_id": parsed.get("session_id"),
        "path": parsed.get("path"),
        "started_at": parsed.get("started_at"),
        "last_activity": parsed.get("last_activity"),
        "cwd": parsed.get("cwd"),
        "originator": parsed.get("originator"),
        "cli_version": parsed.get("cli_version"),
        "title": title[:160],
        "user_messages": len(user_messages),
        "assistant_messages": assistant_count,
        "last_user_message": last_user[:240],
        "truncated_scan": parsed.get("truncated_scan"),
    }


class CodexCliHistory:
    """Fachada read-only sobre o historico local do Codex CLI."""

    def __init__(self, codex_home: Path | None = None):
        self.codex_home = codex_home or default_codex_home()

    def available(self) -> bool:
        return (self.codex_home / "sessions").is_dir()

    def list_recent(
        self,
        *,
        limit: int = DEFAULT_LIMIT,
        project_dir: str | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), MAX_LIMIT))
        sessions: list[dict[str, Any]] = []
        for path in sorted_session_files(self.codex_home):
            parsed = parse_session_file(
                path, roles=frozenset({"user"}), line_limit=LIST_SCAN_LINE_LIMIT
            )
            if not parsed.get("ok"):
                continue
            summary = _summarize_user(parsed)
            if project_dir:
                cwd = summary.get("cwd") or ""
                if os.path.normcase(cwd) != os.path.normcase(str(project_dir)):
                    continue
            sessions.append(summary)
            if len(sessions) >= limit:
                break
        return {
            "ok": True,
            "source": str(self.codex_home / "sessions"),
            "count": len(sessions),
            "sessions": sessions,
        }

    def _resolve_path(self, selector: str) -> dict[str, Any]:
        files = sorted_session_files(self.codex_home)
        if not files:
            return {"ok": False, "error": "no_codex_sessions"}
        if selector.isdigit():
            index = int(selector) - 1
            if 0 <= index < len(files):
                return {"ok": True, "path": str(files[index])}
        candidate = Path(selector)
        if candidate.is_file():
            return {"ok": True, "path": str(candidate)}
        normalized = selector.strip().lower()
        for path in files:
            if normalized in path.name.lower():
                return {"ok": True, "path": str(path)}
        for path in files[:MAX_LIMIT]:
            parsed = parse_session_file(
                path, roles=frozenset({"user"}), line_limit=LIST_SCAN_LINE_LIMIT
            )
            if not parsed.get("ok"):
                continue
            summary = _summarize_user(parsed)
            title = summary["title"].lower()
            session_id = (summary.get("session_id") or "").lower()
            if normalized and (
                normalized in title or session_id.startswith(normalized)
            ):
                return {"ok": True, "path": str(path)}
        return {"ok": False, "error": "session_not_found", "selector": selector}

    def read_session(
        self,
        selector: str,
        *,
        turn_limit: int | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        resolution = self._resolve_path(selector)
        if not resolution.get("ok"):
            return resolution
        path = Path(str(resolution["path"]))
        parsed = parse_session_file(path)
        if not parsed.get("ok"):
            return parsed
        messages = parsed["messages"]
        user_indexes = [
            index
            for index, item in enumerate(messages)
            if item.get("role") == "user"
        ]
        if turn_limit is not None and len(user_indexes) > turn_limit > 0:
            first = user_indexes[-turn_limit]
            messages = messages[first:]
        max_chars = max(1, min(int(max_chars), MAX_MAX_CHARS))
        total = 0
        selected: list[dict[str, Any]] = []
        truncated = False
        for item in reversed(messages):
            text = str(item["text"])
            overhead = len(text) + 12
            if total + overhead > max_chars and selected:
                truncated = True
                break
            selected.append(item)
            total += overhead
        selected.reverse()
        return {
            "ok": True,
            "session_id": parsed.get("session_id"),
            "path": str(path),
            "started_at": parsed.get("started_at"),
            "cwd": parsed.get("cwd"),
            "originator": parsed.get("originator"),
            "cli_version": parsed.get("cli_version"),
            "message_count": len(messages),
            "messages_shown": len(selected),
            "truncated": truncated or any(item.get("text_truncated") for item in selected),
            "turn_limit": turn_limit,
            "max_chars": max_chars,
            "messages": selected,
        }
