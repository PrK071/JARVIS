from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class AccessDenied(PermissionError):
    pass


class ApprovalRequired(PermissionError):
    pass


@dataclass(frozen=True)
class PathPolicy:
    roots: tuple[Path, ...]

    def resolve(self, raw_path: str, *, must_exist: bool = True) -> Path:
        candidate = Path(raw_path).expanduser()
        if ".." in candidate.parts:
            raise AccessDenied("segmento '..' bloqueado")
        resolved = candidate.resolve(strict=must_exist)
        for root in self.roots:
            root_resolved = root.resolve(strict=True)
            try:
                resolved.relative_to(root_resolved)
                return resolved
            except ValueError:
                continue
        raise AccessDenied(f"caminho fora da allowlist: {resolved}")

    def child(self, raw_parent: str, name: str, *, must_exist: bool = False) -> Path:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise AccessDenied("nome de arquivo invalido")
        parent = self.resolve(raw_parent)
        candidate = parent / name
        return self.resolve(str(candidate), must_exist=must_exist)


class ActionLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, *, tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": "tool_result",
            "tool": tool,
            "arguments": self._redact(arguments),
            "result": self._redact(result),
        }
        self._append(record)

    def write_event(self, event: str, **values: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self._redact(values),
        }
        self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def _redact(cls, value: Any, key: str = "") -> Any:
        if re.search(
            r"(?:api[_-]?key|authorization|password|passwd|secret|token|credential)",
            key,
            re.IGNORECASE,
        ):
            return "<redacted>"
        if isinstance(value, dict):
            return {str(k): cls._redact(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._redact(item) for item in value]
        if isinstance(value, str):
            return re.sub(
                r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._~+/-]{12,})",
                "<redacted>",
                value,
            )
        return value


ApprovalCallback = Callable[[str, dict[str, Any]], bool]


def require_approval(callback: ApprovalCallback | None, action: str, arguments: dict[str, Any]) -> None:
    if callback is None or not callback(action, arguments):
        raise ApprovalRequired(f"confirmacao necessaria para {action}")
