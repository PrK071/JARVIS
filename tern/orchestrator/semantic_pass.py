from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .decision_policy import Intent
from .intent_semantics import Constraint, FollowupType, SpeechAct


TARGET_TYPES = {
    "none",
    "codex_job",
    "codex_session",
    "deepseek_session",
    "file",
    "project",
    "agent_response",
    "tool_result",
    "task",
    "generation",
}
AGENTS = {None, "qwen", "codex", "deepseek", "filesystem", "project"}
OPERATIONS = {
    "answer",
    "read",
    "delete",
    "search",
    "status",
    "review",
    "delegate",
    "steer",
    "cancel",
    "resolve_project",
    "clarify",
    "no_action",
}
CONDITIONS = {None, "positive_recommendation"}


SEMANTIC_SYSTEM_PROMPT = """You are the semantic interpretation stage of a local assistant.
Interpret the user's Portuguese request. Do not answer it and do not execute anything.
Return only one compact JSON object required by the supplied schema. No prose or
whitespace formatting.

Distinguish mention from request, question from command, negated action from requested
action, status/history from a new task, and direct answer from delegation. Explicit
constraints override inferred preferences. Never invent an id, path, thread, session,
job, or tool result. Use a semantic reference such as focused_file, latest_codex_job,
shared_codex_session, active_deepseek_session, previous_agent_response,
previous_entity, other_candidate, active_project, or user_mentioned_target.
When execution_requested is false, never choose a side-effect primary_intent. Use
ANSWER_DIRECTLY for explanations and meta-discussion, even when they mention a
side-effect operation or agent. Make primary_intent, operation, speech_act, agent,
and execution_requested consistent with one another.

Principle examples:
- "Como cancelo o Codex?" has speech_act EXPLANATION_REQUEST, primary_intent
  ANSWER_DIRECTLY, operation answer, agent qwen, execution_requested=false.
- "Cancela o Codex." is a command; execution_requested=true.
- "Não pergunta ao DeepSeek. O que você acha?" forbids DeepSeek and requests Qwen.
- In "Pergunta ao DeepSeek se não seria melhor X", the negation is question content,
  not a prohibition on DeepSeek.
- "Pergunta ao DeepSeek e depois manda o Codex implementar" is an ORDERED two-step plan.
"""


def semantic_json_schema() -> dict[str, Any]:
    step = {
        "type": "object",
        "additionalProperties": False,
        "required": ["intent", "operation", "agent", "target_type", "target_reference", "condition"],
        "properties": {
            "intent": {"type": "string", "enum": [item.value for item in Intent]},
            "operation": {"type": "string", "enum": sorted(OPERATIONS)},
            "agent": {"type": ["string", "null"], "enum": [None, "qwen", "codex", "deepseek", "filesystem", "project"]},
            "target_type": {"type": "string", "enum": sorted(TARGET_TYPES)},
            "target_reference": {"type": ["string", "null"], "maxLength": 200},
            "condition": {"type": ["string", "null"], "enum": [None, "positive_recommendation"]},
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "speech_act", "primary_intent", "operation", "execution_requested",
            "agent", "target", "constraints", "followup_type", "continuation",
            "compound_plan", "ambiguity", "confidence",
        ],
        "properties": {
            "speech_act": {"type": "string", "enum": [item.value for item in SpeechAct]},
            "primary_intent": {"type": "string", "enum": [item.value for item in Intent]},
            "operation": {"type": "string", "enum": sorted(OPERATIONS)},
            "execution_requested": {"type": "boolean"},
            "agent": {"type": ["string", "null"], "enum": [None, "qwen", "codex", "deepseek", "filesystem", "project"]},
            "target": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "reference"],
                "properties": {
                    "type": {"type": "string", "enum": sorted(TARGET_TYPES)},
                    "reference": {"type": ["string", "null"], "maxLength": 200},
                },
            },
            "constraints": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": [item.value for item in Constraint]},
                "maxItems": len(Constraint),
            },
            "followup_type": {"type": "string", "enum": [item.value for item in FollowupType]},
            "continuation": {"type": "boolean"},
            "compound_plan": {"type": "array", "items": step, "maxItems": 4},
            "ambiguity": {
                "type": "object",
                "additionalProperties": False,
                "required": ["present", "candidates"],
                "properties": {
                    "present": {"type": "boolean"},
                    "candidates": {"type": "array", "items": {"type": "string", "maxLength": 100}, "maxItems": 5},
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "semantic_decision", "strict": True, "schema": schema},
    }


@dataclass(frozen=True)
class SemanticPlanStep:
    intent: Intent
    operation: str
    agent: str | None
    target_type: str
    target_reference: str | None
    condition: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "operation": self.operation,
            "agent": self.agent,
            "target_type": self.target_type,
            "target_reference": self.target_reference,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class SemanticDecision:
    speech_act: SpeechAct
    primary_intent: Intent
    operation: str
    execution_requested: bool
    agent: str | None
    target_type: str
    target_reference: str | None
    constraints: tuple[Constraint, ...]
    followup_type: FollowupType
    continuation: bool
    compound_plan: tuple[SemanticPlanStep, ...]
    ambiguity_present: bool
    ambiguity_candidates: tuple[str, ...]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "speech_act": self.speech_act.value,
            "primary_intent": self.primary_intent.value,
            "operation": self.operation,
            "execution_requested": self.execution_requested,
            "agent": self.agent,
            "target": {"type": self.target_type, "reference": self.target_reference},
            "constraints": [item.value for item in self.constraints],
            "followup_type": self.followup_type.value,
            "continuation": self.continuation,
            "compound_plan": [item.as_dict() for item in self.compound_plan],
            "ambiguity": {
                "present": self.ambiguity_present,
                "candidates": list(self.ambiguity_candidates),
            },
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SemanticPassResult:
    used: bool
    decision: SemanticDecision | None
    latency_ms: float
    parse_valid: bool
    repair_used: bool
    cache_hit: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "latency_ms": round(self.latency_ms, 3),
            "parse_valid": self.parse_valid,
            "repair_used": self.repair_used,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "semantic_frame": self.decision.as_dict() if self.decision else None,
        }


class SemanticValidationError(ValueError):
    pass


def _enum(enum_type: Any, value: Any, field: str) -> Any:
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise SemanticValidationError(f"{field}: invalid enum {value!r}") from exc


def _validate_reference(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 200:
        raise SemanticValidationError("target.reference must be string|null <= 200 chars")
    if re.search(r"(?:^[A-Za-z]:[\\/]|[0-9a-f]{8}-[0-9a-f-]{27,})", value, re.I):
        raise SemanticValidationError("target.reference must be semantic, not a concrete path/id")
    return value


def validate_semantic_decision(raw: Any) -> SemanticDecision:
    if not isinstance(raw, dict):
        raise SemanticValidationError("root must be an object")
    required = {
        "speech_act", "primary_intent", "operation", "execution_requested", "agent",
        "target", "constraints", "followup_type", "continuation", "compound_plan",
        "ambiguity", "confidence",
    }
    missing = required - set(raw)
    if missing:
        raise SemanticValidationError(f"missing fields: {sorted(missing)}")
    extra = set(raw) - required
    if extra:
        raise SemanticValidationError(f"unexpected fields: {sorted(extra)}")
    speech = _enum(SpeechAct, raw["speech_act"], "speech_act")
    intent = _enum(Intent, raw["primary_intent"], "primary_intent")
    operation = str(raw["operation"])
    if operation not in OPERATIONS:
        raise SemanticValidationError(f"operation: invalid {operation!r}")
    execution = raw["execution_requested"]
    if not isinstance(execution, bool):
        raise SemanticValidationError("execution_requested must be boolean")
    agent = raw["agent"]
    if agent not in AGENTS:
        raise SemanticValidationError(f"agent: invalid {agent!r}")
    target = raw["target"]
    if (
        not isinstance(target, dict)
        or set(target) != {"type", "reference"}
        or target.get("type") not in TARGET_TYPES
    ):
        raise SemanticValidationError("target.type is invalid")
    target_reference = _validate_reference(target.get("reference"))
    raw_constraints = raw["constraints"]
    if not isinstance(raw_constraints, list):
        raise SemanticValidationError("constraints must be an array")
    constraints = tuple(_enum(Constraint, item, "constraints") for item in raw_constraints)
    if len(constraints) != len(set(constraints)):
        raise SemanticValidationError("constraints must not contain duplicates")
    constraint_set = set(constraints)
    if Constraint.ANSWER_SELF in constraint_set and (
        Constraint.WAIT_FOR_RESULT in constraint_set
        or intent in {Intent.CODEX_DELEGATE, Intent.DEEPSEEK_DELEGATE}
    ):
        raise SemanticValidationError("ANSWER_SELF conflicts with delegation")
    followup = _enum(FollowupType, raw["followup_type"], "followup_type")
    if not isinstance(raw["continuation"], bool):
        raise SemanticValidationError("continuation must be boolean")
    plan_raw = raw["compound_plan"]
    if not isinstance(plan_raw, list) or len(plan_raw) > 4:
        raise SemanticValidationError("compound_plan must be an array with <= 4 steps")
    plan: list[SemanticPlanStep] = []
    for number, item in enumerate(plan_raw):
        if not isinstance(item, dict):
            raise SemanticValidationError(f"compound_plan[{number}] must be object")
        step_fields = {"intent", "operation", "agent", "target_type", "target_reference", "condition"}
        if set(item) != step_fields:
            raise SemanticValidationError(f"compound_plan[{number}] fields do not match schema")
        step_operation = str(item.get("operation"))
        step_agent = item.get("agent")
        target_type = str(item.get("target_type"))
        condition = item.get("condition")
        if step_operation not in OPERATIONS or step_agent not in AGENTS:
            raise SemanticValidationError(f"compound_plan[{number}] operation/agent invalid")
        if target_type not in TARGET_TYPES or condition not in CONDITIONS:
            raise SemanticValidationError(f"compound_plan[{number}] target/condition invalid")
        plan.append(
            SemanticPlanStep(
                _enum(Intent, item.get("intent"), f"compound_plan[{number}].intent"),
                step_operation,
                step_agent,
                target_type,
                _validate_reference(item.get("target_reference")),
                condition,
            )
        )
    if plan and Constraint.ORDERED not in constraint_set:
        raise SemanticValidationError("compound_plan requires ORDERED constraint")
    for number, step in enumerate(plan):
        if step.condition is None:
            continue
        if number == 0:
            raise SemanticValidationError("compound_plan[0] cannot have a condition")
        if step.condition == "positive_recommendation":
            prior_deepseek = any(
                previous.intent is Intent.DEEPSEEK_DELEGATE
                for previous in plan[:number]
            )
            if step.intent is not Intent.CODEX_DELEGATE or not prior_deepseek:
                raise SemanticValidationError(
                    "positive_recommendation requires prior DeepSeek then Codex delegation"
                )
    ambiguity = raw["ambiguity"]
    if (
        not isinstance(ambiguity, dict)
        or set(ambiguity) != {"present", "candidates"}
        or not isinstance(ambiguity.get("present"), bool)
    ):
        raise SemanticValidationError("ambiguity must contain boolean present")
    candidates = ambiguity.get("candidates")
    if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
        raise SemanticValidationError("ambiguity.candidates must be strings")
    confidence = raw["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
        raise SemanticValidationError("confidence must be between 0 and 1")
    if not execution and intent in {
        Intent.CODEX_DELEGATE, Intent.CODEX_STEER, Intent.CODEX_CANCEL,
        Intent.DEEPSEEK_DELEGATE, Intent.LOCAL_ACTION,
    }:
        raise SemanticValidationError("side-effect intent requires execution_requested=true")
    if plan and not execution:
        if any(
            step.intent in {
                Intent.CODEX_DELEGATE,
                Intent.CODEX_STEER,
                Intent.CODEX_CANCEL,
                Intent.DEEPSEEK_DELEGATE,
                Intent.LOCAL_ACTION,
            }
            for step in plan
        ):
            raise SemanticValidationError("side-effect compound plan requires execution_requested=true")
    forbidden = set(constraints)
    if intent in {Intent.CODEX_DELEGATE, Intent.CODEX_STEER, Intent.CODEX_CANCEL} and Constraint.FORBID_CODEX in forbidden:
        raise SemanticValidationError("Codex intent conflicts with FORBID_CODEX")
    if intent is Intent.DEEPSEEK_DELEGATE and Constraint.FORBID_DEEPSEEK in forbidden:
        raise SemanticValidationError("DeepSeek intent conflicts with FORBID_DEEPSEEK")
    for step in plan:
        if step.intent in {Intent.CODEX_DELEGATE, Intent.CODEX_STEER, Intent.CODEX_CANCEL} and Constraint.FORBID_CODEX in forbidden:
            raise SemanticValidationError("compound Codex step conflicts with FORBID_CODEX")
        if step.intent is Intent.DEEPSEEK_DELEGATE and Constraint.FORBID_DEEPSEEK in forbidden:
            raise SemanticValidationError("compound DeepSeek step conflicts with FORBID_DEEPSEEK")
    return SemanticDecision(
        speech,
        intent,
        operation,
        execution,
        agent,
        str(target["type"]),
        target_reference,
        constraints,
        followup,
        raw["continuation"],
        tuple(plan),
        ambiguity["present"],
        tuple(candidates[:5]),
        float(confidence),
    )


def _response_content(response: dict[str, Any]) -> str:
    try:
        value = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SemanticValidationError("response missing choices[0].message.content") from exc
    if not isinstance(value, str):
        raise SemanticValidationError("semantic response content must be string")
    return value.strip()


def semantic_context_payload(context: Any) -> dict[str, Any]:
    return {
        "active_project": getattr(context, "active_project", None),
        "project_root": getattr(context, "project_root", None),
        "codex_job": {
            "status": getattr(context, "codex_job_status", None),
            "available": bool(getattr(context, "codex_job_id", None)),
            "running_count": getattr(context, "codex_running_jobs", 0),
        },
        "codex_thread_available": getattr(context, "codex_thread_available", False),
        "deepseek": {
            "enabled": getattr(context, "deepseek_enabled", False),
            "configured": getattr(context, "deepseek_configured", False),
            "active_session": bool(getattr(context, "deepseek_active_session", None)),
        },
        "focus": {
            "agent": getattr(context, "focused_agent", None),
            "project": getattr(context, "focused_project", None),
            "file_available": bool(getattr(context, "focused_file", None)),
            "job_available": bool(getattr(context, "focused_job", None)),
            "session_available": bool(getattr(context, "focused_session", None)),
            "content_available": getattr(context, "content_available", False),
        },
        "recent_entity_types": [
            str(item.get("type"))
            for item in tuple(getattr(context, "recent_entities", ()) or ())[-8:]
            if isinstance(item, dict)
        ],
        "recent_tools": list(tuple(getattr(context, "recent_tools", ()) or ())[-5:]),
        "last_user_intent": getattr(context, "last_user_intent", None),
        "last_user_text": (getattr(context, "last_user_text", None) or "")[:500] or None,
        "pending_action": bool(getattr(context, "pending_action", None)),
    }


class QwenSemanticInterpreter:
    system_prompt = SEMANTIC_SYSTEM_PROMPT

    def __init__(self, client: Any, *, cache_size: int = 64):
        self.client = client
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, SemanticDecision] = OrderedDict()

    @staticmethod
    def needs_semantic_pass(text: str, context: Any) -> bool:
        normalized = text.casefold()
        if re.search(
            r"\b(?:codex|deepseek|arquivo|projeto|sess[aã]o|job|tarefa|turn|"
            r"n[aã]o|sem|ele|ela|isso|esse|essa|aquilo|anterior|outro|"
            r"manda|fa[çc]a|pergunta|consulta|cancela|para|abre|leia|procura|"
            r"corrige|implementa|revisa|continua|depois|ent[aã]o|mas|por[eé]m)\b",
            normalized,
        ):
            return True
        return bool(
            getattr(context, "ambiguous_target", False)
            or getattr(context, "focused_job", None)
            or getattr(context, "focused_file", None)
            or getattr(context, "focused_session", None)
        ) and len(normalized.split()) <= 8

    @staticmethod
    def _fingerprint(original: str, normalized: str, context: Any) -> str:
        payload = {
            "original": original,
            "normalized": normalized,
            "context": semantic_context_payload(context),
            "context_identity": {
                "project": getattr(context, "active_project", None),
                "focused_project": getattr(context, "focused_project", None),
                "focused_file": getattr(context, "focused_file", None),
                "focused_job": getattr(context, "focused_job", None),
                "focused_session": getattr(context, "focused_session", None),
                "job_status": getattr(context, "codex_job_status", None),
                "pending_action": getattr(context, "pending_action", None),
            },
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def interpret(self, original: str, normalized: str, context: Any) -> SemanticPassResult:
        started = time.perf_counter()
        key = self._fingerprint(original, normalized, context)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return SemanticPassResult(True, cached, 0.0, True, False, True)
        schema = semantic_json_schema()
        input_payload = {
            "original_input": original,
            "normalized_input": normalized,
            "decision_context": semantic_context_payload(context),
        }
        messages = [
            {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)},
        ]
        invalid_content = ""
        error = "semantic_parse_failed"
        for attempt in range(2):
            try:
                response = self.client.chat(
                    messages,
                    tools=None,
                    response_format=schema,
                    temperature=0.0,
                    max_tokens=320,
                )
                invalid_content = _response_content(response)
                raw = json.loads(invalid_content)
                decision = validate_semantic_decision(raw)
            except (SemanticValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
                error = str(exc)[:1000]
                if attempt == 0:
                    messages = [
                        {"role": "system", "content": "Repair one invalid semantic JSON object. Return only valid JSON matching the supplied schema."},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "schema_error": error,
                                    "invalid_object": invalid_content[:4000],
                                    "expected_schema": schema["json_schema"]["schema"],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ]
                    continue
                return SemanticPassResult(
                    True,
                    None,
                    (time.perf_counter() - started) * 1000,
                    False,
                    True,
                    False,
                    "semantic_parse_failed",
                )
            self._cache[key] = decision
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return SemanticPassResult(
                True,
                decision,
                (time.perf_counter() - started) * 1000,
                True,
                attempt == 1,
                False,
            )
        return SemanticPassResult(True, None, (time.perf_counter() - started) * 1000, False, True, False, error)

    @staticmethod
    def skipped() -> SemanticPassResult:
        return SemanticPassResult(False, None, 0.0, True, False, False)
