from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .projects import normalize_technical_transcript
from .intent_semantics import (
    Constraint,
    ConversationReferenceResolver,
    FollowupType,
    IntentFrame,
    IntentFrameBuilder,
    PlanStep,
    ResolvedReference,
    SpeechAct,
)


class Intent(str, Enum):
    ANSWER_DIRECTLY = "ANSWER_DIRECTLY"
    LOCAL_READ = "LOCAL_READ"
    LOCAL_SEARCH = "LOCAL_SEARCH"
    LOCAL_ACTION = "LOCAL_ACTION"
    CODEX_REVIEW = "CODEX_REVIEW"
    CODEX_STATUS = "CODEX_STATUS"
    CODEX_DELEGATE = "CODEX_DELEGATE"
    CODEX_STEER = "CODEX_STEER"
    CODEX_CANCEL = "CODEX_CANCEL"
    DEEPSEEK_REVIEW = "DEEPSEEK_REVIEW"
    DEEPSEEK_DELEGATE = "DEEPSEEK_DELEGATE"
    PROJECT_RESOLUTION = "PROJECT_RESOLUTION"
    CLARIFY = "CLARIFY"
    NO_ACTION = "NO_ACTION"


class SideEffect(str, Enum):
    READ_ONLY = "READ_ONLY"
    LOCAL_MUTATION = "LOCAL_MUTATION"
    REMOTE_READ = "REMOTE_READ"
    REMOTE_GENERATION = "REMOTE_GENERATION"
    CODE_EXECUTION = "CODE_EXECUTION"


TOOL_EFFECTS: dict[str, SideEffect] = {
    "resolve_project": SideEffect.READ_ONLY,
    "find_project_files": SideEffect.READ_ONLY,
    "filesystem_list": SideEffect.READ_ONLY,
    "filesystem_read_text": SideEffect.READ_ONLY,
    "filesystem_write_text": SideEffect.LOCAL_MUTATION,
    "filesystem_delete": SideEffect.LOCAL_MUTATION,
    "web_search": SideEffect.REMOTE_READ,
    "web_open": SideEffect.REMOTE_READ,
    "web_extract": SideEffect.REMOTE_READ,
    "review_codex_session": SideEffect.READ_ONLY,
    "get_codex_job_status": SideEffect.READ_ONLY,
    "steer_codex_job": SideEffect.CODE_EXECUTION,
    "cancel_codex_job": SideEffect.CODE_EXECUTION,
    "delegate_to_codex": SideEffect.CODE_EXECUTION,
    "review_deepseek_session": SideEffect.READ_ONLY,
    "delegate_to_deepseek": SideEffect.REMOTE_GENERATION,
}


_STRICT_TOOL_INTENTS = {
    Intent.CODEX_REVIEW,
    Intent.CODEX_STATUS,
    Intent.CODEX_STEER,
    Intent.CODEX_CANCEL,
    Intent.DEEPSEEK_REVIEW,
    Intent.DEEPSEEK_DELEGATE,
    Intent.LOCAL_READ,
    Intent.LOCAL_SEARCH,
    Intent.PROJECT_RESOLUTION,
}


_RUNNING_STATES = {
    "queued",
    "starting",
    "running",
    "steering",
    "cancelling",
    "disconnected",
    "reconnecting",
}


def _plain(value: str) -> str:
    corrected = normalize_technical_transcript(value)
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", corrected.casefold())
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _has(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


@dataclass
class ConversationFocus:
    focused_agent: str | None = None
    focused_project: str | None = None
    focused_project_root: str | None = None
    focused_file: str | None = None
    focused_job: str | None = None
    focused_session: str | None = None
    last_codex_turn_id: str | None = None
    last_codex_project: str | None = None
    last_deepseek_project: str | None = None
    file_content_available: bool = False
    file_content_excerpt: str | None = None
    last_user_intent: str | None = None
    last_user_text: str | None = None
    last_answer_excerpt: str | None = None
    recent_tools: list[str] = field(default_factory=list)
    turn_index: int = 0
    recent_entities: list[dict[str, Any]] = field(default_factory=list)

    def remember_entity(
        self,
        entity_type: str,
        identifier: str | None,
        *,
        status: str | None = None,
    ) -> None:
        if not identifier:
            return
        value = {
            "type": entity_type,
            "id": str(identifier),
            "turn_index": self.turn_index,
        }
        if status:
            value["status"] = status
        self.recent_entities = [
            item
            for item in self.recent_entities
            if not (item.get("type") == entity_type and item.get("id") == str(identifier))
        ]
        self.recent_entities.append(value)
        self.recent_entities = self.recent_entities[-10:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "focused_agent": self.focused_agent,
            "focused_project": self.focused_project,
            "focused_project_root": self.focused_project_root,
            "focused_file": self.focused_file,
            "focused_job": self.focused_job,
            "focused_session": self.focused_session,
            "last_codex_turn_id": self.last_codex_turn_id,
            "last_codex_project": self.last_codex_project,
            "last_deepseek_project": self.last_deepseek_project,
            "file_content_available": self.file_content_available,
            "last_user_intent": self.last_user_intent,
            "last_user_text": self.last_user_text,
            "recent_tools": list(self.recent_tools),
            "turn_index": self.turn_index,
            "recent_entities": list(self.recent_entities),
        }


@dataclass(frozen=True)
class DecisionContext:
    active_project: str | None
    project_root: str | None
    known_projects: tuple[dict[str, Any], ...]
    codex_job_status: str | None
    codex_job_id: str | None
    codex_running_jobs: int
    codex_thread_available: bool
    deepseek_enabled: bool
    deepseek_configured: bool
    deepseek_active_session: str | None
    pending_action: str | None
    focused_agent: str | None
    focused_project: str | None
    focused_project_root: str | None
    focused_file: str | None
    focused_job: str | None
    focused_session: str | None
    content_available: bool
    ambiguous_target: bool
    recent_tools: tuple[str, ...]
    last_user_intent: str | None
    last_user_text: str | None
    turn_index: int = 0
    recent_entities: tuple[dict[str, Any], ...] = ()

    def prompt_text(self) -> str:
        return (
            "Decision context (local, compact):\n"
            f"- active_project: {self.active_project or '-'}\n"
            f"- project_root: {self.project_root or '-'}\n"
            f"- codex_job: {self.codex_job_status or 'none'}"
            f" ({self.codex_job_id or '-'})\n"
            f"- codex_running_jobs: {self.codex_running_jobs}\n"
            f"- codex_thread_available: {str(self.codex_thread_available).lower()}\n"
            f"- deepseek: enabled={str(self.deepseek_enabled).lower()}, "
            f"configured={str(self.deepseek_configured).lower()}, "
            f"session={self.deepseek_active_session or '-'}\n"
            f"- recent_tools: {', '.join(self.recent_tools) or '-'}\n"
            f"- focused_agent: {self.focused_agent or '-'}\n"
            f"- focused_file: {self.focused_file or '-'}\n"
            f"- focused_job: {self.focused_job or '-'}\n"
            f"- pending_action: {self.pending_action or 'none'}\n"
            f"- last_user_intent: {self.last_user_intent or '-'}"
            f"\n- recent_entity_types: {', '.join(str(item.get('type')) for item in self.recent_entities[-5:]) or '-'}"
        )


@dataclass(frozen=True)
class Decision:
    intent: Intent
    confidence: float
    project: str | None
    project_root: str | None
    target: str | None
    tools: tuple[str, ...]
    reason_code: str
    side_effects: tuple[SideEffect, ...]
    alternatives: tuple[tuple[str, float], ...]
    max_tool_calls: int
    new_codex_turn: bool
    user_override: str | None = None
    intent_frame: IntentFrame | None = None
    resolved_reference: ResolvedReference | None = None
    constraint_violation: str | None = None
    semantic_frame: dict[str, Any] | None = None
    semantic_confidence: float | None = None
    reference_confidence: float | None = None

    @property
    def selected_action(self) -> str | None:
        return self.tools[0] if self.tools else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "project": self.project,
            "project_root": self.project_root,
            "target": self.target,
            "tools": list(self.tools),
            "selected_action": self.selected_action,
            "reason_code": self.reason_code,
            "side_effects": [effect.value for effect in self.side_effects],
            "alternatives": [
                {"intent": intent, "confidence": confidence}
                for intent, confidence in self.alternatives
            ],
            "max_tool_calls": self.max_tool_calls,
            "new_codex_turn": self.new_codex_turn,
            "user_override": self.user_override,
            "intent_frame": self.intent_frame.as_dict() if self.intent_frame else None,
            "resolved_reference": self.resolved_reference.as_dict() if self.resolved_reference else None,
            "constraint_violation": self.constraint_violation,
            "semantic_frame": self.semantic_frame,
            "semantic_confidence": self.semantic_confidence,
            "reference_confidence": self.reference_confidence,
        }


@dataclass(frozen=True)
class FastPath:
    tool: str
    arguments: dict[str, Any]
    reason_code: str
    side_effect: SideEffect = SideEffect.READ_ONLY


def constraint_violation_for_tool(
    tool: str,
    frame: IntentFrame | None,
) -> str | None:
    """Validate explicit semantic constraints before any tool can execute."""
    if frame is None:
        return None
    constraints = set(frame.constraints)
    effect = TOOL_EFFECTS.get(tool)
    if Constraint.FORBID_CODEX in constraints and "codex" in tool:
        return Constraint.FORBID_CODEX.value
    if Constraint.FORBID_DEEPSEEK in constraints and "deepseek" in tool:
        return Constraint.FORBID_DEEPSEEK.value
    if Constraint.FORBID_CANCEL in constraints and tool == "cancel_codex_job":
        return Constraint.FORBID_CANCEL.value
    if Constraint.FORBID_NEW_TURN in constraints and tool == "delegate_to_codex":
        return Constraint.FORBID_NEW_TURN.value
    if Constraint.FORBID_DELEGATION in constraints and tool.startswith("delegate_to_"):
        return Constraint.FORBID_DELEGATION.value
    if Constraint.FORBID_MUTATION in constraints and effect is SideEffect.LOCAL_MUTATION:
        return Constraint.FORBID_MUTATION.value
    if Constraint.ANSWER_SELF in constraints and tool.startswith("delegate_to_"):
        return Constraint.ANSWER_SELF.value
    if Constraint.READ_ONLY in constraints and effect is not SideEffect.READ_ONLY:
        return Constraint.READ_ONLY.value
    if not frame.execution_requested and effect is not SideEffect.READ_ONLY:
        return "EXECUTION_NOT_REQUESTED"
    return None


def check_decision_constraints(decision: Decision) -> str | None:
    if decision.intent_frame and decision.intent_frame.contradictory_constraints:
        return "CONTRADICTORY_CONSTRAINTS"
    for tool in decision.tools:
        violation = constraint_violation_for_tool(tool, decision.intent_frame)
        if violation:
            return violation
    return None


class AgentDecisionPolicy:
    """Decision scaffolding: recommends and bounds actions; Qwen still chooses tool arguments."""

    def __init__(
        self,
        *,
        tools: Any | None = None,
        logger: Any | None = None,
        context_cache_enabled: bool = True,
    ):
        self.tools = tools
        self.logger = logger or getattr(tools, "logger", None)
        self.focus = ConversationFocus()
        self.context_cache_enabled = context_cache_enabled
        self._context_cache: DecisionContext | None = None
        self._context_cache_generation = 0
        self._context_cache_last_reason: str | None = None
        self.reference_resolver = ConversationReferenceResolver()
        self.frame_builder = IntentFrameBuilder(self.reference_resolver)
        self._active_frame: IntentFrame | None = None
        self._active_reference: ResolvedReference | None = None
        self._active_semantic: Any | None = None

    def invalidate_context(self, reason: str) -> None:
        self._context_cache = None
        self._context_cache_generation += 1
        self._context_cache_last_reason = reason
        writer = getattr(self.logger, "write_event", None)
        if callable(writer):
            writer(
                "agent_decision_context_invalidated",
                reason=reason,
                generation=self._context_cache_generation,
            )

    def _focus_context(self, value: DecisionContext) -> DecisionContext:
        return replace(
            value,
            focused_agent=self.focus.focused_agent,
            focused_project=self.focus.focused_project or value.active_project,
            focused_project_root=self.focus.focused_project_root or value.project_root,
            focused_file=self.focus.focused_file,
            focused_job=self.focus.focused_job or value.codex_job_id,
            focused_session=self.focus.focused_session,
            content_available=self.focus.file_content_available,
            recent_tools=tuple(self.focus.recent_tools[-5:]),
            last_user_intent=self.focus.last_user_intent,
            last_user_text=self.focus.last_user_text,
            turn_index=self.focus.turn_index,
            recent_entities=tuple(self.focus.recent_entities[-10:]),
        )

    def _fixture_context(self, fixture: dict[str, Any]) -> DecisionContext:
        active = fixture.get("active_project")
        roots = {
            "tern": r"D:\tern",
            "llama.cpp": r"D:\llama.cpp",
            "sasori_review": r"D:\sasori_review",
        }
        project_root = roots.get(str(active)) if active else None
        job = fixture.get("codex_job") if isinstance(fixture.get("codex_job"), dict) else {}
        deepseek = fixture.get("deepseek") if isinstance(fixture.get("deepseek"), dict) else {}
        return DecisionContext(
            active_project=str(active) if active else None,
            project_root=project_root,
            known_projects=tuple(
                {"id": name, "name": name, "root": root, "aliases": [name]}
                for name, root in roots.items()
            ),
            codex_job_status=str(job.get("status")) if job.get("status") else None,
            codex_job_id=str(job.get("job_id")) if job.get("job_id") else None,
            codex_running_jobs=1 if job.get("status") in _RUNNING_STATES else 0,
            codex_thread_available=bool(fixture.get("codex_thread_available", True)),
            deepseek_enabled=bool(deepseek.get("enabled", True)),
            deepseek_configured=bool(deepseek.get("configured", False)),
            deepseek_active_session=str(deepseek.get("session_id")) if deepseek.get("session_id") else None,
            pending_action=str(fixture.get("pending_action")) if fixture.get("pending_action") else None,
            focused_agent=fixture.get("focused_agent") or self.focus.focused_agent,
            focused_project=fixture.get("focused_project") or active or self.focus.focused_project,
            focused_project_root=fixture.get("focused_project_root") or project_root or self.focus.focused_project_root,
            focused_file=fixture.get("focused_file") or self.focus.focused_file,
            focused_job=fixture.get("focused_job") or job.get("job_id") or self.focus.focused_job,
            focused_session=fixture.get("focused_session") or self.focus.focused_session,
            content_available=bool(
                fixture.get("content_available", self.focus.file_content_available)
            ),
            ambiguous_target=bool(fixture.get("ambiguous_target", False)),
            recent_tools=tuple(fixture.get("recent_tools") or self.focus.recent_tools[-5:]),
            last_user_intent=fixture.get("last_user_intent") or self.focus.last_user_intent,
            last_user_text=fixture.get("last_user_text") or self.focus.last_user_text,
            turn_index=int(fixture.get("turn_index", self.focus.turn_index) or 0),
            recent_entities=tuple(fixture.get("recent_entities") or self.focus.recent_entities[-10:]),
        )

    def build_context(self, *, fixture_context: dict[str, Any] | None = None) -> DecisionContext:
        if fixture_context is not None:
            return self._fixture_context(fixture_context)
        if self.tools is None:
            return self._fixture_context({})
        if self.context_cache_enabled and self._context_cache is not None:
            return self._focus_context(self._context_cache)
        project_value = self.tools.projects.context()
        active = project_value.get("active_project") or {}
        jobs: list[dict[str, Any]] = []
        job_store = getattr(self.tools.codex, "jobs", None)
        if job_store is not None:
            try:
                jobs = [item for item in job_store.list() if isinstance(item, dict)]
            except Exception:
                jobs = []
        running = [item for item in jobs if item.get("status") in _RUNNING_STATES]
        latest = (running or jobs)[-1] if (running or jobs) else {}
        deepseek_status: dict[str, Any] = {}
        if getattr(self.tools, "deepseek", None) is not None:
            try:
                deepseek_status = self.tools.deepseek.status(
                    project_path=active.get("root")
                )
            except Exception:
                deepseek_status = {}
        try:
            pending = self.tools.pending_actions.pending()
        except Exception:
            pending = None
        thread_available = bool(
            project_value.get("codex_thread_project")
            or getattr(self.tools.codex, "shared_project", lambda: None)()
        )
        value = DecisionContext(
            active_project=str(active.get("id") or active.get("name")) if active else None,
            project_root=str(active.get("root")) if active.get("root") else None,
            known_projects=tuple(self.tools.projects.projects()),
            codex_job_status=str(latest.get("status")) if latest.get("status") else None,
            codex_job_id=str(latest.get("job_id")) if latest.get("job_id") else None,
            codex_running_jobs=len(running),
            codex_thread_available=thread_available,
            deepseek_enabled=bool(deepseek_status.get("enabled")),
            deepseek_configured=bool(deepseek_status.get("configured")),
            deepseek_active_session=str(deepseek_status.get("active_session")) if deepseek_status.get("active_session") else None,
            pending_action=str(pending.get("action_id")) if isinstance(pending, dict) else None,
            focused_agent=self.focus.focused_agent,
            focused_project=self.focus.focused_project or (str(active.get("id")) if active else None),
            focused_project_root=self.focus.focused_project_root or (str(active.get("root")) if active else None),
            focused_file=self.focus.focused_file,
            focused_job=self.focus.focused_job or (str(latest.get("job_id")) if latest.get("job_id") else None),
            focused_session=self.focus.focused_session,
            content_available=self.focus.file_content_available,
            ambiguous_target=False,
            recent_tools=tuple(self.focus.recent_tools[-5:]),
            last_user_intent=self.focus.last_user_intent,
            last_user_text=self.focus.last_user_text,
            turn_index=self.focus.turn_index,
            recent_entities=tuple(self.focus.recent_entities[-10:]),
        )
        if self.context_cache_enabled:
            self._context_cache = value
        return self._focus_context(value)

    @staticmethod
    def _project(text: str, context: DecisionContext) -> tuple[str | None, str | None, bool]:
        aliases = (
            ("llama.cpp", ("llama", "lama ponto cpp", "lama cpp")),
            ("sasori_review", ("sasori", "sasori review")),
            ("tern", ("jarvis", "tern", "orquestrador")),
        )
        selected = None
        explicit = False
        for project, values in aliases:
            if _has(text, values):
                selected = project
                explicit = True
                break
        if selected is None:
            selected = context.focused_project or context.active_project
        root = None
        for item in context.known_projects:
            candidates = {
                str(item.get("id") or "").casefold(),
                str(item.get("name") or "").casefold(),
                Path(str(item.get("root") or ".")).name.casefold(),
            }
            if selected and str(selected).casefold() in candidates:
                root = str(item.get("root") or "") or None
                selected = str(item.get("id") or selected)
                break
        root = root or context.focused_project_root or context.project_root
        return selected, root, explicit

    def _decision(
        self,
        intent: Intent,
        confidence: float,
        project: str | None,
        root: str | None,
        tools: tuple[str, ...],
        reason: str,
        *,
        target: str | None = None,
        alternatives: tuple[tuple[str, float], ...] = (),
        override: str | None = None,
    ) -> Decision:
        value = Decision(
            intent=intent,
            confidence=confidence,
            project=project,
            project_root=root,
            target=target,
            tools=tools,
            reason_code=reason,
            side_effects=tuple(TOOL_EFFECTS[tool] for tool in tools if tool in TOOL_EFFECTS),
            alternatives=alternatives,
            max_tool_calls=len(tools),
            new_codex_turn="delegate_to_codex" in tools,
            user_override=override,
            intent_frame=self._active_frame,
            resolved_reference=self._active_reference,
            semantic_frame=(self._active_semantic.as_dict() if self._active_semantic else None),
            semantic_confidence=(float(self._active_semantic.confidence) if self._active_semantic else None),
            reference_confidence=(self._active_reference.confidence if self._active_reference else None),
        )
        violation = check_decision_constraints(value)
        if violation and intent is not Intent.CLARIFY:
            return Decision(
                intent=Intent.CLARIFY,
                confidence=0.99,
                project=project,
                project_root=root,
                target=target,
                tools=(),
                reason_code="explicit_constraint_conflict",
                side_effects=(),
                alternatives=((intent.value, confidence),),
                max_tool_calls=0,
                new_codex_turn=False,
                user_override=override,
                intent_frame=self._active_frame,
                resolved_reference=self._active_reference,
                constraint_violation=violation,
                semantic_frame=(self._active_semantic.as_dict() if self._active_semantic else None),
                semantic_confidence=(float(self._active_semantic.confidence) if self._active_semantic else None),
                reference_confidence=(self._active_reference.confidence if self._active_reference else None),
            )
        return value

    def _semantic_frame_from_qwen(self, semantic: Any) -> IntentFrame:
        return IntentFrame(
            speech_act=semantic.speech_act,
            operation=semantic.operation,
            agent=semantic.agent,
            target=semantic.target_reference,
            polarity="positive",
            execution_requested=semantic.execution_requested,
            continuation=semantic.continuation,
            constraints=tuple(semantic.constraints),
            confidence=semantic.confidence,
            followup_type=semantic.followup_type,
            plan=tuple(
                PlanStep(
                    step.operation,
                    step.agent,
                    step.target_reference,
                )
                for step in semantic.compound_plan
            ),
        )

    def _decision_from_semantic(
        self,
        semantic: Any,
        *,
        context: DecisionContext,
        project: str | None,
        root: str | None,
        explicit_project: bool,
    ) -> Decision:
        reference = self._active_reference or ResolvedReference(None, None, 0.0)
        if semantic.ambiguity_present or reference.ambiguous:
            return self._decision(
                Intent.CLARIFY,
                max(0.75, semantic.confidence),
                project,
                root,
                (),
                "semantic_reference_ambiguous",
                target=reference.id,
            )

        if reference.type == "project" and reference.id:
            project = reference.id
            for item in context.known_projects:
                if str(item.get("id") or item.get("name")) == project:
                    root = str(item.get("root") or "") or root
                    break

        tool_for_intent = {
            Intent.CODEX_STATUS: "get_codex_job_status",
            Intent.CODEX_REVIEW: "review_codex_session",
            Intent.CODEX_DELEGATE: "delegate_to_codex",
            Intent.CODEX_STEER: "steer_codex_job",
            Intent.CODEX_CANCEL: "cancel_codex_job",
            Intent.DEEPSEEK_REVIEW: "review_deepseek_session",
            Intent.DEEPSEEK_DELEGATE: "delegate_to_deepseek",
            Intent.PROJECT_RESOLUTION: "resolve_project",
            Intent.LOCAL_SEARCH: "find_project_files",
            Intent.LOCAL_ACTION: "filesystem_write_text",
        }
        tools: list[str] = []
        if semantic.compound_plan:
            for step in semantic.compound_plan:
                tool = tool_for_intent.get(step.intent)
                if step.intent is Intent.LOCAL_READ:
                    tool = "filesystem_read_text" if reference.type == "file" else "find_project_files"
                if tool and tool not in tools:
                    tools.append(tool)
        elif semantic.primary_intent is Intent.LOCAL_READ:
            if reference.type == "file" and reference.id:
                tools = ["filesystem_read_text"]
            else:
                prefix = ["resolve_project"] if explicit_project and project != context.active_project else []
                tools = [*prefix, "find_project_files", "filesystem_read_text"]
        else:
            tool = tool_for_intent.get(semantic.primary_intent)
            if semantic.primary_intent is Intent.LOCAL_ACTION and semantic.operation == "delete":
                tool = "filesystem_delete"
            if tool:
                tools = [tool]

        if semantic.primary_intent in {Intent.ANSWER_DIRECTLY, Intent.NO_ACTION, Intent.CLARIFY}:
            tools = []
        target = reference.id or semantic.target_reference
        return self._decision(
            semantic.primary_intent,
            semantic.confidence,
            project,
            root,
            tuple(tools),
            "qwen_semantic_frame",
            target=target,
            override="semantic_first",
        )

    def _semantic_route(
        self,
        *,
        text: str,
        context: DecisionContext,
        project: str | None,
        root: str | None,
        explicit_project: bool,
    ) -> Decision | None:
        frame = self._active_frame
        reference = self._active_reference
        if frame is None or reference is None:
            return None
        constraints = set(frame.constraints)
        job_active = context.codex_job_status in _RUNNING_STATES or context.codex_running_jobs > 0

        if frame.contradictory_constraints or (
            (reference.ambiguous or context.ambiguous_target) and not frame.plan
        ):
            return self._decision(
                Intent.CLARIFY,
                0.96,
                project,
                root,
                (),
                "contradictory_constraints" if frame.contradictory_constraints else "ambiguous_target" if context.ambiguous_target else "multiple_plausible_referents",
                target=reference.id,
            )

        if Constraint.ANSWER_SELF in constraints:
            return self._decision(
                Intent.ANSWER_DIRECTLY,
                0.99,
                project,
                root,
                (),
                "explicit_qwen_override",
                override="qwen_only",
            )

        if frame.speech_act is SpeechAct.CORRECTION:
            return self._decision(
                Intent.ANSWER_DIRECTLY,
                0.97,
                project,
                root,
                (),
                "user_reference_correction",
                target=reference.id,
            )

        if (
            not frame.execution_requested
            and frame.operation == "answer"
            and frame.agent in {"codex", "deepseek"}
            and ("codex" in text or "deepseek" in text or "consultor" in text)
        ):
            return self._decision(
                Intent.ANSWER_DIRECTLY,
                0.96,
                project,
                root,
                (),
                "agent_mention_not_request",
            )

        if frame.operation in {"explain", "explain_cancel"} and not frame.execution_requested:
            return self._decision(
                Intent.ANSWER_DIRECTLY,
                0.98,
                project,
                root,
                (),
                "question_about_action" if frame.operation == "explain_cancel" else "direct_answer_sufficient",
            )

        if frame.plan:
            tools: list[str] = []
            for step in frame.plan:
                tool = {
                    ("review", "codex"): "review_codex_session",
                    ("delegate", "deepseek"): "delegate_to_deepseek",
                    ("delegate", "codex"): "delegate_to_codex",
                }.get((step.operation, step.agent))
                if tool and tool not in tools:
                    tools.append(tool)
            if tools:
                return self._decision(
                    Intent.DEEPSEEK_DELEGATE if "delegate_to_deepseek" in tools else Intent.CODEX_DELEGATE,
                    0.97,
                    project,
                    root,
                    tuple(tools),
                    "ordered_compound_action",
                    target=frame.target,
                )

        if frame.operation == "status":
            if reference.type == "codex_job" or (
                reference.type is None and (context.focused_agent == "codex" or job_active)
            ):
                return self._decision(
                    Intent.CODEX_STATUS,
                    0.98,
                    project,
                    root,
                    ("get_codex_job_status",),
                    "active_job_status_query",
                    target=reference.id or context.focused_job or context.codex_job_id,
                )

        if frame.operation == "review":
            asks_for_opinion = bool(
                re.search(r"\b(?:voce|sua opiniao|o que acha|o que achou|me explica|explique|resume|resuma)\b", text)
            )
            if context.content_available and asks_for_opinion:
                return self._decision(
                    Intent.ANSWER_DIRECTLY,
                    0.98,
                    project,
                    root,
                    (),
                    "tool_result_already_available",
                    target=reference.id,
                )
            if reference.type == "deepseek_session" or frame.agent == "deepseek":
                return self._decision(
                    Intent.DEEPSEEK_REVIEW,
                    0.97,
                    project,
                    root,
                    ("review_deepseek_session",),
                    "deepseek_history_query",
                    target=reference.id or context.focused_session,
                )
            if reference.type in {"codex_session", "codex_job"} or frame.agent == "codex":
                return self._decision(
                    Intent.CODEX_REVIEW,
                    0.97,
                    project,
                    root,
                    ("review_codex_session",),
                    "codex_history_query",
                    target=reference.id or context.focused_session,
                )

        if frame.operation == "read" and reference.type == "file":
            return self._decision(
                Intent.LOCAL_READ,
                0.98,
                project,
                root,
                ("filesystem_read_text",),
                "existing_file_context",
                target=reference.id,
            )
        if frame.operation == "read":
            prefix = ("resolve_project",) if explicit_project and project != context.active_project else ()
            return self._decision(
                Intent.LOCAL_READ,
                0.92,
                project,
                root,
                prefix + ("find_project_files", "filesystem_read_text"),
                "explicit_local_read",
                target=reference.id,
            )

        if frame.operation == "cancel" and frame.execution_requested:
            if frame.agent == "codex" and job_active:
                return self._decision(
                    Intent.CODEX_CANCEL,
                    0.99,
                    project,
                    root,
                    ("cancel_codex_job",),
                    "explicit_codex_cancel" if "codex" in text else "followup_to_active_job",
                    target=reference.id or context.focused_job or context.codex_job_id,
                )

        if frame.operation == "steer" and frame.execution_requested and frame.agent == "codex" and job_active:
            return self._decision(
                Intent.CODEX_STEER,
                0.98,
                project,
                root,
                ("steer_codex_job",),
                "followup_to_active_job",
                target=reference.id or context.focused_job or context.codex_job_id,
            )

        if frame.operation == "delegate" and frame.execution_requested:
            if frame.agent == "deepseek":
                return self._decision(
                    Intent.DEEPSEEK_DELEGATE,
                    0.99,
                    project,
                    root,
                    ("delegate_to_deepseek",),
                    "explicit_deepseek_request",
                    override="deepseek_explicit",
                )
            if frame.agent == "codex":
                if project is None:
                    return self._decision(Intent.CLARIFY, 0.82, None, None, (), "ambiguous_target")
                return self._decision(
                    Intent.CODEX_DELEGATE,
                    0.98,
                    project,
                    root,
                    ("delegate_to_codex",),
                    "explicit_codex_delegate",
                    target=root,
                )
        return None

    def decide(
        self,
        user_text: str,
        *,
        context: DecisionContext | None = None,
        fixture_context: dict[str, Any] | None = None,
        semantic_decision: Any | None = None,
    ) -> Decision:
        context = context or self.build_context(fixture_context=fixture_context)
        text = _plain(user_text)
        project, root, explicit_project = self._project(text, context)
        self._active_semantic = semantic_decision
        if semantic_decision is not None:
            self._active_frame = self._semantic_frame_from_qwen(semantic_decision)
            self._active_reference = self.reference_resolver.resolve_typed(
                semantic_decision.target_type,
                semantic_decision.target_reference,
                context,
            )
            return self._decision_from_semantic(
                semantic_decision,
                context=context,
                project=project,
                root=root,
                explicit_project=explicit_project,
            )
        self._active_semantic = None
        self._active_frame, self._active_reference = self.frame_builder.build(text, context)
        semantic = self._semantic_route(
            text=text,
            context=context,
            project=project,
            root=root,
            explicit_project=explicit_project,
        )
        if semantic is not None:
            return semantic
        job_active = context.codex_job_status in _RUNNING_STATES or context.codex_running_jobs > 0
        explicit_codex = "codex" in text
        explicit_deepseek = "deepseek" in text

        if _has(text, ("nao faca nada", "so me diga se entendeu", "apenas confirme que entendeu")):
            return self._decision(Intent.NO_ACTION, 0.99, project, root, (), "explicit_no_action", override="no_action")

        self_override = _has(text, ("responde voce mesmo", "responda voce mesmo", "nao use o codex", "sem usar o codex"))
        if self_override and not explicit_deepseek:
            return self._decision(Intent.ANSWER_DIRECTLY, 0.99, project, root, (), "explicit_qwen_override", override="qwen_only")

        status_signal = _has(
            text,
            (
                "terminou",
                "ja acabou",
                "ainda esta trabalhando",
                "ainda ta trabalhando",
                "ainda ta fazendo",
                "como esta aquela tarefa",
                "como ta aquela tarefa",
                "status da tarefa",
                "e ai terminou",
            ),
        )
        history_signal = _has(
            text,
            (
                "o que o codex fez",
                "oq o codex fez",
                "o que ele fez",
                "oq ele fez",
                "ultima sessao",
                "ultimos tres turns",
                "ultimos 3 turns",
                "ultima atividade",
                "o que aconteceu nos ultimos",
                "ultima solucao do codex",
                "informacoes do codex",
            ),
        ) or ("turn" in text and "ultim" in text)
        cancel_signal = _has(
            text, ("cancela", "cancelar", "interrompa", "para ele", "manda ele parar")
        ) or bool(re.search(r"\bcodex\s+(?:pare|para)\s*$", text))
        steer_signal = _has(
            text,
            (
                "fala pra ele",
                "fala pro codex",
                "avisa ele",
                "avise ao codex",
                "manda ele olhar",
                "manda ele incluir",
                "e fala pra ele",
                "so os warnings",
                "so os uornings",
                "nao mexer",
            ),
        )

        if explicit_deepseek:
            deepseek_review = _has(
                text,
                (
                    "o que o deepseek falou",
                    "oq o deepseek falou",
                    "deepseek respondeu",
                    "resposta dele",
                    "ultima conversa com o deepseek",
                    "deepseek sugeriu",
                ),
            ) or _has(text, ("falou o que", "respondeu o que", "sugeriu o que"))
            deepseek_review = deepseek_review and not _has(
                text,
                ("pergunta", "manda", "mostra", "consulta", "segunda opiniao"),
            )
            if deepseek_review:
                return self._decision(
                    Intent.DEEPSEEK_REVIEW,
                    0.98,
                    project,
                    root,
                    ("review_deepseek_session",),
                    "deepseek_history_query",
                    target=context.focused_session,
                )
            codex_context = explicit_codex and history_signal
            then_codex = explicit_codex and _has(
                text, ("depois", "implementar", "implemente", "manda o codex", "codex revisar")
            )
            tools: list[str] = []
            if codex_context:
                tools.append("review_codex_session")
            tools.append("delegate_to_deepseek")
            if then_codex:
                tools.append("delegate_to_codex")
            return self._decision(
                Intent.DEEPSEEK_DELEGATE,
                0.99,
                project,
                root,
                tuple(tools),
                "explicit_deepseek_request",
                alternatives=((Intent.ANSWER_DIRECTLY.value, 0.05),),
                override="deepseek_explicit",
            )

        if cancel_signal and (explicit_codex or context.focused_agent == "codex"):
            if job_active:
                return self._decision(Intent.CODEX_CANCEL, 0.99, project, root, ("cancel_codex_job",), "followup_to_active_job", target=context.focused_job or context.codex_job_id)
            return self._decision(Intent.CODEX_STATUS, 0.72, project, root, ("get_codex_job_status",), "codex_job_state_required", target=context.focused_job)

        if steer_signal and context.focused_agent == "codex" and job_active:
            return self._decision(Intent.CODEX_STEER, 0.98, project, root, ("steer_codex_job",), "followup_to_active_job", target=context.focused_job or context.codex_job_id)

        if status_signal and (explicit_codex or context.focused_agent == "codex" or job_active):
            return self._decision(Intent.CODEX_STATUS, 0.97, project, root, ("get_codex_job_status",), "active_job_status_query", target=context.focused_job or context.codex_job_id, alternatives=((Intent.CODEX_REVIEW.value, 0.10),))

        if history_signal and (explicit_codex or context.focused_agent == "codex"):
            return self._decision(Intent.CODEX_REVIEW, 0.97, project, root, ("review_codex_session",), "codex_history_query", target=context.focused_session)

        if context.focused_agent == "deepseek" and _has(
            text, ("o que ele respondeu", "oq ele respondeu", "o que ele falou", "oq ele falou")
        ):
            return self._decision(Intent.DEEPSEEK_REVIEW, 0.94, project, root, ("review_deepseek_session",), "focused_deepseek_followup", target=context.focused_session)

        if context.ambiguous_target:
            return self._decision(Intent.CLARIFY, 0.90, project, root, (), "ambiguous_target", target=context.focused_file, alternatives=((Intent.LOCAL_READ.value, 0.35),))

        mutation = _has(
            text,
            (
                "corrige",
                "corrigir",
                "implementa",
                "implemente",
                "adiciona",
                "arruma",
                "conserta",
                "roda os testes e",
                "executa os testes e",
            ),
        )
        explicit_codex_action = explicit_codex and _has(
            text, ("peca", "manda", "faz", "use", "rodar", "revisar", "corrigir", "implementar", "verifica")
        )
        if mutation or explicit_codex_action:
            if project is None:
                return self._decision(Intent.CLARIFY, 0.82, None, None, (), "ambiguous_target")
            return self._decision(Intent.CODEX_DELEGATE, 0.96 if explicit_codex_action else 0.90, project, root, ("delegate_to_codex",), "explicit_codex_delegate" if explicit_codex_action else "project_mutation_requires_execution", target=root)

        if _has(text, ("onde eu tava mexendo", "onde eu estava mexendo")) and context.focused_file:
            return self._decision(Intent.ANSWER_DIRECTLY, 0.97, project, root, (), "existing_file_context", target=context.focused_file)

        read_signal = _has(text, ("leia", "abre", "abra", "mostre o conteudo", "ve aquela configuracao", "revisa a configuracao", "revise a configuracao"))
        pronoun_file = _has(text, ("abre ele", "abre ela", "abre esse", "abre aquele", "aquele arquivo"))
        if pronoun_file and context.focused_file:
            return self._decision(Intent.LOCAL_READ, 0.98, project, root, ("filesystem_read_text",), "existing_file_context", target=context.focused_file)

        explain_signal = _has(
            text,
            (
                "explica",
                "explique",
                "o que significa",
                "oq significa",
                "qual a diferenca",
                "faz sentido",
                "parece correto",
                "qual nome",
                "resume",
                "avalie essa ideia",
                "arquitetura",
                "traceback",
            ),
        ) or bool(
            re.search(
                r"\bentre\b.+\b(?:qual|melhor|escolh(?:e|eria|o))\b|"
                r"\bqual\b.+\b(?:escolh(?:e|eria)|prefer(?:e|iria))\b",
                text,
            )
        )
        if explain_signal and context.content_available:
            return self._decision(Intent.ANSWER_DIRECTLY, 0.95, project, root, (), "tool_result_already_available", target=context.focused_file)

        search_signal = _has(
            text,
            (
                "onde fica",
                "onde ta",
                "onde esta",
                "localiza",
                "localizar",
                "procura",
                "encontre",
                "qual arquivo",
                "mostra os arquivos",
                "ve em qual arquivo",
            ),
        )
        project_resolution = _has(text, ("resolve o projeto", "olha o llama", "olha o sasori", "qual e o projeto ativo"))
        if project_resolution:
            if _has(text, ("qual e o projeto ativo",)) and context.active_project:
                return self._decision(Intent.PROJECT_RESOLUTION, 0.98, project, root, (), "tool_result_already_available", target=root)
            return self._decision(Intent.PROJECT_RESOLUTION, 0.93, project, root, ("resolve_project",), "known_project_alias" if explicit_project else "project_resolution_required", target=root)

        if read_signal:
            if context.focused_file and _has(text, ("ele", "ela", "esse", "aquele")):
                tools = ("filesystem_read_text",)
                target = context.focused_file
            else:
                prefix = ("resolve_project",) if explicit_project and project != context.active_project else ()
                tools = prefix + ("find_project_files", "filesystem_read_text")
                target = context.focused_file
            return self._decision(Intent.LOCAL_READ, 0.91, project, root, tools, "explicit_local_read", target=target)

        if search_signal:
            prefix = ("resolve_project",) if explicit_project and project != context.active_project else ()
            return self._decision(Intent.LOCAL_SEARCH, 0.92, project, root, prefix + ("find_project_files",), "known_project_alias" if explicit_project else "explicit_local_search", target=root)

        if explain_signal:
            return self._decision(Intent.ANSWER_DIRECTLY, 0.86, project, root, (), "direct_answer_sufficient")

        if _has(text, ("apaga o arquivo", "delete o arquivo")):
            return self._decision(Intent.LOCAL_ACTION, 0.88, project, root, ("filesystem_delete",), "explicit_local_mutation", target=context.focused_file)

        return self._decision(
            Intent.ANSWER_DIRECTLY,
            0.64,
            project,
            root,
            (),
            "direct_answer_sufficient",
            alternatives=((Intent.LOCAL_SEARCH.value, 0.22), (Intent.CLARIFY.value, 0.12)),
        )

    def record_decision(self, decision: Decision, user_text: str) -> None:
        self.focus.turn_index += 1
        self.focus.last_user_intent = decision.intent.value
        self.focus.last_user_text = user_text[:1000]
        if decision.project:
            self.focus.focused_project = decision.project
            self.focus.focused_project_root = decision.project_root
            self.focus.remember_entity("project", decision.project_root or decision.project)
        if (
            decision.intent_frame
            and decision.intent_frame.speech_act is SpeechAct.CORRECTION
            and decision.intent_frame.agent in {"codex", "deepseek"}
        ):
            self.focus.focused_agent = decision.intent_frame.agent
        if (
            decision.intent_frame
            and decision.intent_frame.speech_act is SpeechAct.CORRECTION
            and decision.resolved_reference
            and decision.resolved_reference.type == "file"
            and decision.resolved_reference.id
        ):
            self.focus.focused_file = decision.resolved_reference.id
        writer = getattr(self.logger, "write_event", None)
        if callable(writer):
            writer(
                "decision_made",
                intent=decision.intent.value,
                confidence=round(decision.confidence, 3),
                selected_action=decision.selected_action,
                project=decision.project_root or decision.project,
                alternatives=[
                    {"intent": intent, "confidence": confidence}
                    for intent, confidence in decision.alternatives
                ],
                reason_code=decision.reason_code,
                side_effects=[effect.value for effect in decision.side_effects],
                max_tool_calls=decision.max_tool_calls,
                user_override=decision.user_override,
                original_transcript=user_text,
                routing_transcript=normalize_technical_transcript(user_text),
                speech_act=(decision.intent_frame.speech_act.value if decision.intent_frame else None),
                execution_requested=(decision.intent_frame.execution_requested if decision.intent_frame else None),
                constraints=([value.value for value in decision.intent_frame.constraints] if decision.intent_frame else []),
                resolved_reference_type=(decision.resolved_reference.type if decision.resolved_reference else None),
                reference_confidence=(decision.resolved_reference.confidence if decision.resolved_reference else 0.0),
                followup_type=(decision.intent_frame.followup_type.value if decision.intent_frame else None),
            )

    def safe_fallback_decision(self, decision: Decision) -> Decision:
        """Keep only unequivocal reads after two failed semantic parses."""
        if (
            decision.tools
            and all(TOOL_EFFECTS.get(tool) is SideEffect.READ_ONLY for tool in decision.tools)
            and not (decision.resolved_reference and decision.resolved_reference.ambiguous)
            and decision.confidence >= 0.95
        ):
            return replace(
                decision,
                reason_code="semantic_parse_failed_safe_read",
                user_override="semantic_safe_fallback",
            )
        if decision.intent in {Intent.ANSWER_DIRECTLY, Intent.NO_ACTION} and not decision.tools:
            return replace(
                decision,
                reason_code="semantic_parse_failed_direct_fallback",
                user_override="semantic_safe_fallback",
            )
        return Decision(
            intent=Intent.CLARIFY,
            confidence=1.0,
            project=decision.project,
            project_root=decision.project_root,
            target=decision.target,
            tools=(),
            reason_code="semantic_parse_failed_safe_fallback",
            side_effects=(),
            alternatives=((decision.intent.value, decision.confidence),),
            max_tool_calls=0,
            new_codex_turn=False,
            user_override="semantic_safe_fallback",
            intent_frame=decision.intent_frame,
            resolved_reference=decision.resolved_reference,
            semantic_frame=None,
            semantic_confidence=None,
            reference_confidence=decision.reference_confidence,
        )

    def fast_path(
        self,
        decision: Decision,
        context: DecisionContext,
        user_text: str,
    ) -> FastPath | None:
        """Return only deterministic, single-entity, read-only shortcuts."""
        if decision.confidence < 0.95 or len(decision.tools) != 1:
            return None
        tool = decision.tools[0]
        if constraint_violation_for_tool(tool, decision.intent_frame):
            return None
        if TOOL_EFFECTS.get(tool) is not SideEffect.READ_ONLY:
            return None
        if context.ambiguous_target:
            return None
        if tool == "get_codex_job_status":
            if context.codex_running_jobs != 1 or not context.codex_job_id:
                return None
            return FastPath(
                tool,
                {"job_id": context.codex_job_id, "latest": False},
                "single_active_codex_job_read",
            )
        turn_limit = _turn_limit(user_text)
        if tool == "review_codex_session":
            if not context.codex_thread_available:
                return None
            return FastPath(
                tool,
                {"turn_limit": turn_limit},
                "single_codex_thread_read",
            )
        if tool == "review_deepseek_session":
            session = context.focused_session or context.deepseek_active_session
            if not session:
                return None
            return FastPath(
                tool,
                {"project_path": context.project_root, "turn_limit": turn_limit},
                "single_deepseek_session_read",
            )
        if tool == "filesystem_read_text":
            if not context.focused_file:
                return None
            return FastPath(
                tool,
                {"path": context.focused_file, "max_bytes": 131072},
                "single_focused_file_read",
            )
        return None

    def record_tool_result(
        self,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        invalidation_reason = {
            "resolve_project": "project_changed",
            "find_project_files": "file_focus_changed",
            "filesystem_read_text": "file_focus_changed",
            "delegate_to_codex": "job_started",
            "get_codex_job_status": "job_status_changed",
            "steer_codex_job": "job_changed",
            "cancel_codex_job": "job_cancelled",
            "review_codex_session": "codex_session_changed",
            "delegate_to_deepseek": "deepseek_session_changed",
            "review_deepseek_session": "deepseek_session_changed",
        }.get(name)
        if invalidation_reason:
            self.invalidate_context(invalidation_reason)
        if result.get("error") == "action_pending":
            self.invalidate_context("pending_action_changed")
        self.focus.recent_tools = [*self.focus.recent_tools, name][-5:]
        project = arguments.get("project_path") or result.get("root")
        if project:
            self.focus.focused_project_root = str(project)
            self.focus.focused_project = Path(str(project)).name or str(project)
        if name == "find_project_files" and result.get("ok"):
            values = result.get("results") if isinstance(result.get("results"), list) else []
            if values and not result.get("ambiguous"):
                root = str(result.get("root") or self.focus.focused_project_root or "")
                relative = str(values[0].get("path") or "")
                if root and relative:
                    self.focus.focused_file = str((Path(root) / relative).resolve())
                    self.focus.remember_entity("file", self.focus.focused_file)
        elif name == "filesystem_read_text" and result.get("ok"):
            self.focus.focused_file = str(result.get("path") or arguments.get("path") or "") or self.focus.focused_file
            self.focus.file_content_available = True
            content = str(result.get("content") or "")
            self.focus.file_content_excerpt = content[:6000]
            self.focus.remember_entity("file", self.focus.focused_file)
        elif name in {"delegate_to_codex", "get_codex_job_status", "steer_codex_job", "cancel_codex_job", "review_codex_session"}:
            self.focus.focused_agent = "codex"
            self.focus.focused_job = str(result.get("job_id") or self.focus.focused_job or "") or None
            self.focus.focused_session = str(result.get("thread_id") or self.focus.focused_session or "") or None
            self.focus.last_codex_turn_id = str(result.get("turn_id") or self.focus.last_codex_turn_id or "") or None
            self.focus.last_codex_project = str(project or self.focus.last_codex_project or "") or None
            self.focus.remember_entity(
                "codex_job",
                self.focus.focused_job,
                status=str(result.get("status") or "") or None,
            )
            self.focus.remember_entity("codex_session", self.focus.focused_session)
        elif name in {"delegate_to_deepseek", "review_deepseek_session"}:
            self.focus.focused_agent = "deepseek"
            self.focus.focused_session = str(result.get("session_id") or self.focus.focused_session or "") or None
            self.focus.last_deepseek_project = str(project or self.focus.last_deepseek_project or "") or None
            self.focus.remember_entity("deepseek_session", self.focus.focused_session)

    def record_answer(self, answer: str) -> None:
        self.focus.last_answer_excerpt = answer[:2000]

    def reusable_context_text(self) -> str:
        values = []
        if self.focus.last_user_text:
            values.append(f"Previous user request: {self.focus.last_user_text}")
        if self.focus.last_answer_excerpt:
            values.append(f"Previous answer excerpt: {self.focus.last_answer_excerpt}")
        if self.focus.focused_file:
            values.append(f"Focused file: {self.focus.focused_file}")
        if self.focus.file_content_excerpt:
            values.append(f"Last read content excerpt:\n{self.focus.file_content_excerpt}")
        return "\n".join(values)


def tool_specs_for_decision(
    specs: list[dict[str, Any]],
    decision: Decision,
) -> list[dict[str, Any]]:
    """Expose the least-capable catalogue sufficient for a confident decision."""
    specs = [
        item
        for item in specs
        if not constraint_violation_for_tool(
            str(item.get("function", {}).get("name") or ""),
            decision.intent_frame,
        )
    ]
    if decision.confidence < 0.80:
        blocked = {
            "delegate_to_deepseek",
            "review_deepseek_session",
            "delegate_to_codex",
        }
        return [
            item
            for item in specs
            if item.get("function", {}).get("name") not in blocked
        ]
    if decision.intent in {Intent.ANSWER_DIRECTLY, Intent.NO_ACTION, Intent.CLARIFY}:
        return []
    if decision.intent == Intent.CODEX_DELEGATE:
        # A mutation/development request is already resolved to Codex. Keeping
        # unrelated schemas visible made the model re-resolve the project instead
        # of delegating, while the Codex task itself can perform required reads.
        allowed = set(decision.tools)
        return [
            item
            for item in specs
            if item.get("function", {}).get("name") in allowed
        ]
    if decision.intent in _STRICT_TOOL_INTENTS:
        allowed = set(decision.tools)
        return [
            item
            for item in specs
            if item.get("function", {}).get("name") in allowed
        ]
    return list(specs)


def tool_catalog_audit(
    specs: list[dict[str, Any]],
    decision: Decision,
) -> dict[str, Any]:
    """Explain deterministic catalogue filtering without exposing model reasoning."""
    allowed_specs = tool_specs_for_decision(specs, decision)
    allowed = {
        str(item.get("function", {}).get("name") or "")
        for item in allowed_specs
    }
    rejected: list[dict[str, str]] = []
    for item in specs:
        name = str(item.get("function", {}).get("name") or "")
        if not name or name in allowed:
            continue
        constraint = constraint_violation_for_tool(name, decision.intent_frame)
        if constraint:
            reason = f"constraint:{constraint}"
        elif decision.intent in {Intent.ANSWER_DIRECTLY, Intent.NO_ACTION, Intent.CLARIFY}:
            reason = "no_tool_intent"
        elif decision.confidence < 0.80 and name.startswith("delegate_to_"):
            reason = "low_confidence_generation"
        else:
            reason = "not_required_by_semantic_frame"
        rejected.append({"tool": name, "reason": reason})
    return {"allowed": sorted(allowed), "rejected": rejected}


def _turn_limit(value: str) -> int:
    normalized = _plain(value)
    match = re.search(r"\b([1-9]|[1-4][0-9]|50)\b", normalized)
    if match:
        return int(match.group(1))
    words = {
        "um": 1,
        "uma": 1,
        "dois": 2,
        "duas": 2,
        "tres": 3,
        "quatro": 4,
        "cinco": 5,
    }
    for word, number in words.items():
        if re.search(rf"\b{word}\b", normalized):
            return number
    return 10
