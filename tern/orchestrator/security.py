from __future__ import annotations

import json
import os
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
            "tool": tool,
            "arguments": arguments,
            "result": result,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + os.linesep)


ApprovalCallback = Callable[[str, dict[str, Any]], bool]


def require_approval(callback: ApprovalCallback | None, action: str, arguments: dict[str, Any]) -> None:
    if callback is None or not callback(action, arguments):
        raise ApprovalRequired(f"confirmacao necessaria para {action}")
