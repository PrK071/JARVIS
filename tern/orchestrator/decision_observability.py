from __future__ import annotations

import json
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .security import ActionLogger


def estimate_tokens(value: Any) -> int:
    """Explicit rough estimate for local diagnostics when no tokenizer is loaded."""
    if value in (None, "", [], {}, ()):
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, math.ceil(len(value) / 4)) if value else 0


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "average_ms": sum(values) / len(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
    }


@dataclass
class DecisionTiming:
    started: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)
    tool_execution_ms: float = 0.0
    qwen_request_ms: float = 0.0
    qwen_requests: int = 0
    qwen_first_token_ms: float | None = None

    def mark(self, phase: str) -> None:
        self.marks[phase] = time.perf_counter()

    def elapsed(self, phase: str) -> float | None:
        value = self.marks.get(phase)
        return None if value is None else (value - self.started) * 1000

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_build_ms": _delta(self.marks.get("input_received"), self.marks.get("decision_context_ready")),
            "policy_ms": _delta(self.marks.get("decision_context_ready"), self.marks.get("decision_ready")),
            "semantic_latency_ms": _delta(
                self.marks.get("semantic_request_started"),
                self.marks.get("semantic_request_completed"),
            ),
            "prompt_build_ms": _delta(self.marks.get("decision_ready"), self.marks.get("prompt_ready")),
            "qwen_first_token_ms": self.qwen_first_token_ms,
            "qwen_first_token_available": self.qwen_first_token_ms is not None,
            "qwen_streaming": False,
            "decision_ms": self.elapsed("decision_detected"),
            "first_tool_ms": self.elapsed("tool_call_detected"),
            "tool_execution_ms": self.tool_execution_ms,
            "qwen_request_ms": self.qwen_request_ms,
            "qwen_requests": self.qwen_requests,
            "response_ms": self.elapsed("response_ready"),
            "time_to_first_audio_ms": None,
        }


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000


class AgentDecisionObserver:
    def __init__(self, state_dir: Path, *, enabled: bool):
        self.enabled = enabled
        self.path = state_dir / "agent-decisions.jsonl"
        self.logger = ActionLogger(self.path)

    def begin(
        self,
        *,
        original_input: str,
        normalized_input: str,
        decision: Any,
        context: Any,
        prompt_sizes: dict[str, Any],
    ) -> str | None:
        if not self.enabled:
            return None
        return str(uuid.uuid4())

    def complete(
        self,
        decision_id: str | None,
        *,
        original_input: str,
        normalized_input: str,
        decision: Any,
        context: Any,
        prompt_sizes: dict[str, Any],
        timing: dict[str, Any],
        tool_calls: int,
        actual_tools: list[str],
        outcome: str,
        semantic_result: Any | None = None,
        tool_catalog: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or decision_id is None:
            return
        frame = getattr(decision, "intent_frame", None)
        reference = getattr(decision, "resolved_reference", None)
        semantic = getattr(semantic_result, "decision", None)
        self.logger.write_event(
            "agent_decision_shadow",
            decision_id=decision_id,
            original_input=original_input[:2000],
            normalized_input=normalized_input[:2000],
            intent=decision.intent.value,
            confidence=round(decision.confidence, 3),
            selected_tool=decision.selected_action,
            planned_tools=list(decision.tools),
            actual_tools=list(actual_tools),
            project=decision.project,
            reason_code=decision.reason_code,
            speech_act=(frame.speech_act.value if frame else None),
            execution_requested=(frame.execution_requested if frame else None),
            constraints=([item.value for item in frame.constraints] if frame else []),
            resolved_reference_type=(reference.type if reference else None),
            reference_confidence=(round(reference.confidence, 3) if reference else 0.0),
            followup_type=(frame.followup_type.value if frame else None),
            focused_agent=context.focused_agent,
            focused_project=context.focused_project,
            focused_file=context.focused_file,
            focused_job=context.focused_job,
            focused_session=context.focused_session,
            tool_calls=tool_calls,
            outcome=outcome,
            timing=timing,
            prompt_sizes=prompt_sizes,
            semantic_pass_used=bool(getattr(semantic_result, "used", False)),
            semantic_latency_ms=float(getattr(semantic_result, "latency_ms", 0.0)),
            semantic_parse_valid=bool(getattr(semantic_result, "parse_valid", True)),
            semantic_repair_used=bool(getattr(semantic_result, "repair_used", False)),
            semantic_cache_hit=bool(getattr(semantic_result, "cache_hit", False)),
            semantic_frame=(semantic.as_dict() if semantic else None),
            tool_catalog=tool_catalog or {"allowed": [], "rejected": []},
        )

    def feedback(
        self,
        *,
        verdict: str,
        expected: str | None = None,
    ) -> dict[str, Any]:
        recent = _last_event(self.path, "agent_decision_shadow")
        if recent is None:
            return {"ok": False, "error": "decision_not_found"}
        record = {
            "decision_id": recent.get("decision_id"),
            "actual": recent.get("intent"),
            "expected": expected,
            "verdict": verdict,
            "context": {
                "project": recent.get("project"),
                "focused_agent": recent.get("focused_agent"),
                "focused_project": recent.get("focused_project"),
                "focused_file": recent.get("focused_file"),
                "focused_job": recent.get("focused_job"),
                "focused_session": recent.get("focused_session"),
                "reason_code": recent.get("reason_code"),
            },
        }
        self.logger.write_event("agent_decision_feedback", **record)
        return {"ok": True, **record}

    def stats(self, *, days: int = 7) -> dict[str, Any]:
        if days <= 0:
            raise ValueError("days must be positive")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        decisions: list[dict[str, Any]] = []
        feedback: list[dict[str, Any]] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                    when = datetime.fromisoformat(str(item.get("time")))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if when < cutoff:
                    continue
                if item.get("event") == "agent_decision_shadow":
                    decisions.append(item)
                elif item.get("event") == "agent_decision_feedback":
                    feedback.append(item)
        intents = Counter(str(item.get("intent") or "unknown") for item in decisions)
        outcomes = Counter(str(item.get("outcome") or "unknown") for item in decisions)
        reasons = Counter(str(item.get("reason_code") or "unknown") for item in decisions)
        grouped = {
            "direct": sum(intents[name] for name in ("ANSWER_DIRECTLY", "NO_ACTION")),
            "local_read_search": sum(
                intents[name]
                for name in ("LOCAL_READ", "LOCAL_SEARCH", "LOCAL_ACTION", "PROJECT_RESOLUTION")
            ),
            "codex": sum(count for name, count in intents.items() if name.startswith("CODEX_")),
            "deepseek": sum(count for name, count in intents.items() if name.startswith("DEEPSEEK_")),
            "clarifications": intents["CLARIFY"],
        }
        decision_latency = [
            float(item["timing"]["decision_ms"])
            for item in decisions
            if isinstance(item.get("timing"), dict)
            and isinstance(item["timing"].get("decision_ms"), (int, float))
        ]
        return {
            "ok": True,
            "days": days,
            "decisions": len(decisions),
            "intents": dict(intents),
            "groups": grouped,
            "outcomes": dict(outcomes),
            "latency": latency_summary(decision_latency),
            "tool_loops_prevented": outcomes.get("loop_prevented", 0),
            "tool_errors": outcomes.get("tool_error", 0),
            "user_corrections": sum(item.get("verdict") == "wrong" for item in feedback),
            "reason_codes": dict(reasons.most_common(10)),
        }


def _last_event(path: Path, event: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("event") == event:
            return item
    return None
