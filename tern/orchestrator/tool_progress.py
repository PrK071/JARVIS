from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


_TRANSIENT_KEYS = frozenset(
    {
        "time",
        "timestamp",
        "updated_at",
        "last_event_at",
        "duration_seconds",
        "size",
        "bytes",
        "modified_ns",
        "delivery_token",
    }
)
_ENTITY_KEYS = frozenset(
    {
        "project_id",
        "thread_id",
        "turn_id",
        "job_id",
        "session_id",
        "name",
    }
)
_PATH_KEYS = frozenset(
    {
        "path",
        "root",
        "relative_path",
        "project_path",
        "working_directory",
    }
)


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _TRANSIENT_KEYS
        }
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _fingerprint(name: str, arguments: dict[str, Any]) -> str:
    encoded = json.dumps([name, _stable(arguments)], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _result_signature(result: dict[str, Any]) -> str:
    encoded = json.dumps(_stable(result), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _collect(value: Any, *, paths: set[str], entities: set[str], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _collect(child, paths=paths, entities=entities, key=str(child_key))
        return
    if isinstance(value, list):
        for child in value:
            _collect(child, paths=paths, entities=entities, key=key)
        return
    if not isinstance(value, str) or not value:
        return
    if key in _PATH_KEYS or key.endswith("_path"):
        paths.add(value.casefold())
    elif key in _ENTITY_KEYS or key.endswith("_id"):
        entities.add(f"{key}:{value.casefold()}")


@dataclass(frozen=True)
class ProgressObservation:
    fingerprint: str
    result_signature: str
    progress: bool
    new_paths: tuple[str, ...]
    new_entities: tuple[str, ...]
    error: str | None


class ToolProgressTracker:
    """Tracks evidence gained by tool calls during one Qwen turn."""

    def __init__(self) -> None:
        self._history: dict[str, list[ProgressObservation]] = {}
        self._paths: set[str] = set()
        self._entities: set[str] = set()

    def fingerprint(self, name: str, arguments: dict[str, Any]) -> str:
        return _fingerprint(name, arguments)

    def should_block(self, name: str, arguments: dict[str, Any]) -> bool:
        history = self._history.get(_fingerprint(name, arguments), [])
        return bool(
            len(history) >= 2
            and history[-1].result_signature == history[-2].result_signature
            and not history[-1].progress
        )

    def record(
        self,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> ProgressObservation:
        fingerprint = _fingerprint(name, arguments)
        paths: set[str] = set()
        entities: set[str] = set()
        _collect(result, paths=paths, entities=entities)
        new_paths = paths.difference(self._paths)
        new_entities = entities.difference(self._entities)
        signature = _result_signature(result)
        history = self._history.setdefault(fingerprint, [])
        changed_state = bool(history and history[-1].result_signature != signature)
        progress = bool(new_paths or new_entities or changed_state)
        observation = ProgressObservation(
            fingerprint=fingerprint,
            result_signature=signature,
            progress=progress,
            new_paths=tuple(sorted(new_paths)),
            new_entities=tuple(sorted(new_entities)),
            error=str(result.get("error")) if result.get("error") else None,
        )
        history.append(observation)
        self._paths.update(paths)
        self._entities.update(entities)
        return observation

    def equivalent_without_progress(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> bool:
        history = self._history.get(_fingerprint(name, arguments), [])
        return bool(
            len(history) >= 2
            and history[-1].result_signature == history[-2].result_signature
            and not history[-1].progress
        )

    def evidence(self) -> dict[str, list[str]]:
        return {
            "paths": sorted(self._paths),
            "entities": sorted(self._entities),
        }
