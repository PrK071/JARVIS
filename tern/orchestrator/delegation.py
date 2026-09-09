from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


DELEGATION_REQUEST_SCHEMA = "jarvis.delegation_request.v1"


def _stable_values(values: Iterable[Any]) -> tuple[str, ...]:
    normalized = {
        str(value)
        for value in values
        if value is not None and str(value).strip()
    }
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class DelegationRequest:
    """Versioned, deterministic envelope sent to every delegated agent."""

    requested_agent: str
    task: str
    project_path: str | None
    action: str | None = None
    constraints: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    execution_mode: str | None = None
    source: str = "tool_argument"
    requested_agent_source: str | None = None

    @classmethod
    def build(
        cls,
        *,
        requested_agent: str,
        submitted_task: str,
        project_path: str | None,
        context: dict[str, Any],
    ) -> "DelegationRequest":
        original = context.get("original_user_text")
        has_original = isinstance(original, str) and bool(original.strip())
        task = original if has_original else submitted_task
        return cls(
            requested_agent=requested_agent,
            task=task,
            project_path=project_path,
            action=str(context.get("delegation_action") or "") or None,
            constraints=_stable_values(context.get("delegation_constraints") or ()),
            references=_stable_values(context.get("delegation_references") or ()),
            execution_mode=str(context.get("execution_mode") or "") or None,
            source="original_user_text" if has_original else "tool_argument",
            requested_agent_source=(
                str(context.get("requested_agent_source") or "") or None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraints": list(self.constraints),
            "action": self.action,
            "execution_mode": self.execution_mode,
            "references": list(self.references),
            "requested_agent": self.requested_agent,
            "requested_agent_source": self.requested_agent_source,
            "schema": DELEGATION_REQUEST_SCHEMA,
            "scope": {"project_path": self.project_path},
            "source": self.source,
            "task": self.task,
        }

    def serialize(self) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
