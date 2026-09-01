"""One-step orchestration policies and deterministic NextAction validation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jsonschema import ValidationError, validate as validate_json_schema

from .autonomy_foundation import Agent, Capability
from .execution_authority import BoundedLiveRiskMatrix
from .execution_gate import ExecutionMode
from .orchestration_contracts import (
    NextAction,
    NextActionType,
    OrchestrationReasonCode,
    UserGoal,
    WorldState,
)
from .orchestration_fast_path import AllowedActionSpace
from .task_requirement_grounding import RequirementValue


ORCHESTRATION_POLICY_SYSTEM_PROMPT = """Choose exactly one best NEXT action as JSON.
WorldState is factual; never claim an unobserved result or grant permission.
ExecutionAuthority decides permission. Preserve explicit agents, never use forbidden
agents, and treat permitted agents as optional. READ_ONLY forbids mutation. Prefer
INSPECT over ASK_USER when an available read can reduce uncertainty. WAIT requires a
real running dependency. Mutation language alone does not prove target/path state;
inspect first when those facts are missing.

Include action, objective, reason_code and only fields relevant to that action:
- DELEGATE: target_agent + execution_mode; never tool_name.
- EXECUTE: tool_name + execution_mode; never target_agent.
- INSPECT: target or tool_name; never target_agent or MUTATION.
- ASK_USER/RESPOND/STOP: no agent, tool, mode or arguments.
Keep objective concise. Return JSON only, never a multi-step plan."""


_POLICY_REASON_CODES = (
    OrchestrationReasonCode.INSUFFICIENT_INFORMATION,
    OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED,
    OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
    OrchestrationReasonCode.CODE_MUTATION_REQUIRED,
    OrchestrationReasonCode.USER_INPUT_REQUIRED,
    OrchestrationReasonCode.GOAL_COMPLETED,
    OrchestrationReasonCode.GOAL_IMPOSSIBLE,
    OrchestrationReasonCode.AGENT_UNAVAILABLE,
    OrchestrationReasonCode.AGENT_INELIGIBLE,
    OrchestrationReasonCode.AUTHORITY_BLOCKED,
    OrchestrationReasonCode.SUFFICIENT_INFORMATION,
    OrchestrationReasonCode.EXPLICIT_AGENT_REQUIRED,
    OrchestrationReasonCode.DEPENDENCY_IN_PROGRESS,
    OrchestrationReasonCode.SAFETY_TERMINAL,
)


class OrchestrationPolicy(Protocol):
    model_version: str

    def decide(
        self,
        user_goal: UserGoal,
        world_state: WorldState,
        allowed_actions: AllowedActionSpace | None = None,
    ) -> NextAction:
        """Return one proposed action and perform no external action."""


class PolicyOutputError(ValueError):
    """The model returned an object outside the closed NextAction contract."""


@dataclass(frozen=True)
class PolicyCallStats:
    inference_ms: float
    model_calls: int
    prompt_build_ms: float = 0.0
    tokenization_ms: float | None = None
    prefill_ms: float | None = None
    generation_ms: float | None = None
    structured_parse_ms: float = 0.0
    retry_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    context_size: int = 0
    cache_hit: bool = False

    @property
    def total_policy_ms(self) -> float:
        return self.inference_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_policy_ms": self.total_policy_ms,
            "prompt_build_ms": self.prompt_build_ms,
            "tokenization_ms": self.tokenization_ms,
            "prefill_ms": self.prefill_ms,
            "generation_ms": self.generation_ms,
            "structured_parse_ms": self.structured_parse_ms,
            "retry_ms": self.retry_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_size": self.context_size,
            "model_calls": self.model_calls,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class ActionValidationResult:
    valid: bool
    violations: tuple[str, ...]
    critical_violations: tuple[str, ...]
    validation_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": list(self.violations),
            "critical_violations": list(self.critical_violations),
            "validation_ms": self.validation_ms,
        }


class NextActionValidator:
    """Validate structure and reality-bound facts without authorizing execution."""

    _TERMINAL_ACTIONS = frozenset(
        {NextActionType.ASK_USER, NextActionType.RESPOND, NextActionType.STOP}
    )

    def validate(
        self,
        action: NextAction,
        goal: UserGoal,
        state: WorldState,
        allowed: AllowedActionSpace | None = None,
    ) -> ActionValidationResult:
        started = time.perf_counter()
        violations: list[str] = []
        critical: list[str] = []

        if action.reason_code is OrchestrationReasonCode.INVALID_ACTION:
            violations.append("INVALID_REASON_CODE")

        if allowed is not None:
            if action.action not in allowed.action_types:
                violations.append("ACTION_OUTSIDE_AVAILABLE_ACTIONS")
                critical.append("POLICY_EXCLUDED_EXECUTION")
            if action.tool_name and action.tool_name not in allowed.tools:
                violations.append("TOOL_OUTSIDE_AVAILABLE_ACTIONS")
                critical.append("POLICY_EXCLUDED_EXECUTION")
            if (
                action.target_agent is not None
                and action.target_agent not in allowed.candidate_agents
            ):
                violations.append("AGENT_OUTSIDE_AVAILABLE_ACTIONS")
                critical.append("POLICY_EXCLUDED_EXECUTION")
            if action.target_agent is not None and action.execution_mode is not None:
                modes = allowed.agent_modes.get(action.target_agent, ())
                if action.execution_mode not in modes:
                    violations.append("MODE_OUTSIDE_AVAILABLE_ACTIONS")
                    critical.append("POLICY_EXCLUDED_EXECUTION")
            if (
                action.action in {NextActionType.INSPECT, NextActionType.EXECUTE}
                and action.tool_name in allowed.tool_schemas
            ):
                tool_policy = BoundedLiveRiskMatrix.tool_policy(action.tool_name)
                if (
                    action.execution_mode is ExecutionMode.MUTATION
                    and not tool_policy.mutation
                ):
                    violations.append("TOOL_MODE_MISMATCH")
                schema = allowed.tool_schemas[action.tool_name]
                if schema:
                    try:
                        validate_json_schema(dict(action.arguments), dict(schema))
                    except ValidationError as exc:
                        violations.append("TOOL_ARGUMENTS_INVALID")
                        violations.append(
                            f"TOOL_ARGUMENT_ERROR:{str(exc.message)[:240]}"
                        )

        if action.action is NextActionType.DELEGATE:
            if action.target_agent is None:
                violations.append("DELEGATE_REQUIRES_AGENT")
            if action.tool_name is not None:
                violations.append("DELEGATE_FORBIDS_TOOL")
            if action.execution_mode is None:
                violations.append("DELEGATE_REQUIRES_MODE")
        elif action.action is NextActionType.EXECUTE:
            if action.tool_name is None:
                violations.append("EXECUTE_REQUIRES_TOOL")
            if action.target_agent is not None:
                violations.append("EXECUTE_FORBIDS_AGENT")
            if action.execution_mode is None:
                violations.append("EXECUTE_REQUIRES_MODE")
        elif action.action is NextActionType.INSPECT:
            if action.target is None and action.tool_name is None:
                violations.append("INSPECT_REQUIRES_TARGET")
            if action.target_agent is not None:
                violations.append("INSPECT_FORBIDS_AGENT")
            if action.execution_mode is ExecutionMode.MUTATION:
                violations.append("INSPECT_CANNOT_MUTATE")
        elif action.action is NextActionType.WAIT:
            if action.target_agent is not None:
                violations.append("WAIT_FORBIDS_AGENT")
            if action.tool_name is not None:
                violations.append("WAIT_FORBIDS_TOOL")
            if action.execution_mode is not None:
                violations.append("WAIT_FORBIDS_MODE")
            in_progress = any(
                job.status.casefold()
                in {
                    "queued",
                    "starting",
                    "running",
                    "steering",
                    "cancelling",
                    "reconnecting",
                }
                for job in state.jobs
            )
            if not in_progress:
                violations.append("WAIT_WITHOUT_ACTIVE_DEPENDENCY")
        elif action.action in self._TERMINAL_ACTIONS:
            if action.target_agent is not None:
                violations.append(f"{action.action.value}_FORBIDS_AGENT")
            if action.tool_name is not None:
                violations.append(f"{action.action.value}_FORBIDS_TOOL")
            if action.execution_mode is not None:
                violations.append(f"{action.action.value}_FORBIDS_MODE")
            if (
                action.action is NextActionType.STOP
                and action.reason_code is OrchestrationReasonCode.GOAL_COMPLETED
            ):
                violations.append("STOP_CANNOT_CLAIM_GOAL_COMPLETED")

        agent_state = None
        if action.target_agent is not None:
            agent_state = state.agents.get(action.target_agent)
            if agent_state is None or not agent_state.availability_known:
                violations.append("AGENT_AVAILABILITY_UNKNOWN")
                critical.append("FABRICATED_CAPABILITY")
            elif not agent_state.available:
                violations.append("AGENT_UNAVAILABLE")
            if agent_state and agent_state.eligible is False:
                violations.append("AGENT_INELIGIBLE")
            if agent_state:
                missing = set(action.required_capabilities) - set(
                    agent_state.capabilities
                )
                if missing:
                    violations.append("WRONG_CAPABILITY")
                    critical.append("FABRICATED_CAPABILITY")

        if action.target_agent in goal.forbidden_agents:
            violations.append("FORBIDDEN_AGENT")
            critical.append("USED_FORBIDDEN_AGENT")
        if (
            goal.explicit_agent is not None
            and action.target_agent is not None
            and action.target_agent is not goal.explicit_agent
        ):
            violations.append("EXPLICIT_AGENT_NOT_PRESERVED")
            critical.extend(("IGNORED_EXPLICIT_AGENT", "SILENT_AGENT_SUBSTITUTION"))
        if (
            goal.mutation_forbidden
            and action.execution_mode is ExecutionMode.MUTATION
        ):
            violations.append("READ_ONLY_CONFLICT")
            critical.append("VIOLATED_READ_ONLY")
        if (
            action.execution_mode is ExecutionMode.MUTATION
            and goal.mutation_required is not RequirementValue.TRUE
        ):
            violations.append("MUTATION_REQUIREMENT_MISSING")
            critical.append("MUTATION_WITHOUT_REQUIREMENT_OR_AUTHORITY")
        if action.tool_name and action.tool_name not in state.tools:
            violations.append("TOOL_NOT_AVAILABLE")
            critical.append("FABRICATED_CAPABILITY")
        if (
            action.action is NextActionType.EXECUTE
            and violations
            and (
                action.execution_mode is ExecutionMode.MUTATION
                or (
                    action.tool_name is not None
                    and BoundedLiveRiskMatrix.tool_policy(action.tool_name).mutation
                )
            )
        ):
            critical.append("UNSAFE_EXECUTE")

        return ActionValidationResult(
            valid=not violations,
            violations=tuple(dict.fromkeys(violations)),
            critical_violations=tuple(dict.fromkeys(critical)),
            validation_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def orchestration_action_json_schema(
    allowed: AllowedActionSpace | None = None,
) -> dict[str, Any]:
    action_types = (
        allowed.action_types if allowed else tuple(NextActionType)
    )
    agents = allowed.candidate_agents if allowed else tuple(Agent)
    modes = (
        tuple(
            dict.fromkeys(
                mode for values in allowed.agent_modes.values() for mode in values
            )
        )
        if allowed
        else tuple(ExecutionMode)
    )
    tools = allowed.tools if allowed else ()
    nullable_agents = [None, *(item.value for item in agents)]
    nullable_modes = [None, *(item.value for item in modes)]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "target_agent",
            "target",
            "tool_name",
            "arguments",
            "objective",
            "execution_mode",
            "reason_code",
        ],
        "properties": {
            "action": {
                "type": "string",
                "enum": [item.value for item in action_types],
            },
            "target_agent": {
                "type": ["string", "null"],
                "enum": nullable_agents,
            },
            "target": {"type": ["string", "null"], "maxLength": 600},
            "tool_name": {
                "type": ["string", "null"],
                "enum": [None, *tools] if allowed else None,
                "maxLength": 160,
            },
            "arguments": {
                "type": "object",
                "maxProperties": 24,
                "additionalProperties": True,
            },
            "objective": {"type": "string", "minLength": 1, "maxLength": 600},
            "execution_mode": {
                "type": ["string", "null"],
                "enum": nullable_modes,
            },
            "required_capabilities": {
                "type": "array",
                "uniqueItems": True,
                "maxItems": len(Capability),
                "items": {
                    "type": "string",
                    "enum": [item.value for item in Capability],
                },
            },
            "reason_code": {
                "type": "string",
                "enum": [item.value for item in _POLICY_REASON_CODES],
            },
            "evidence_refs": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string", "maxLength": 200},
            },
            "expected_observation": {
                "type": ["string", "null"],
                "maxLength": 1000,
            },
            "confidence": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 1,
            },
            "short_horizon_hint": {
                "type": ["string", "null"],
                "maxLength": 600,
            },
        },
    }
    if not allowed:
        schema["properties"]["tool_name"].pop("enum", None)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "shadow_next_action",
            "strict": True,
            "schema": schema,
        },
    }


class QwenOrchestrationPolicy:
    """Structured, one-step policy adapter around the existing local client."""

    model_version = "qwen-local"

    def __init__(
        self,
        client: Any,
        *,
        max_tokens: int = 280,
        mode: str = "shadow",
    ):
        self.client = client
        self.max_tokens = max_tokens
        self.mode = mode
        self.last_call_stats = PolicyCallStats(0.0, 0)

    def decide(
        self,
        user_goal: UserGoal,
        world_state: WorldState,
        allowed_actions: AllowedActionSpace | None = None,
    ) -> NextAction:
        started = time.perf_counter()
        prompt_started = time.perf_counter()
        allowed_actions = allowed_actions or self._default_action_space(world_state)
        payload = {
            "mode": self.mode,
            "goal": self._compact_goal(user_goal),
            "state": self._compact_state(world_state),
            "available_actions": allowed_actions.as_prompt_dict(),
            "instruction": "Choose the single best next action.",
        }
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        prompt_build_ms = round((time.perf_counter() - prompt_started) * 1000, 3)
        messages = [
            {"role": "system", "content": ORCHESTRATION_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        request = {
            "tools": None,
            "response_format": orchestration_action_json_schema(allowed_actions),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        response = self.client.chat(
            messages,
            **request,
        )
        parse_started = time.perf_counter()
        responses = [response]
        retry_ms = 0.0
        try:
            raw = self._content(response)
            action = self._parse(json.loads(raw), user_goal, world_state, self.mode)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            retry_started = time.perf_counter()
            try:
                raw = self._content(response)
            except ValueError:
                raw = ""
            retry_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "The previous object violated the closed schema "
                        f"({type(exc).__name__}). Return one corrected JSON action."
                    ),
                },
            ]
            retry_response = self.client.chat(retry_messages, **request)
            responses.append(retry_response)
            retry_ms = round((time.perf_counter() - retry_started) * 1000, 3)
            try:
                retry_raw = self._content(retry_response)
                action = self._parse(
                    json.loads(retry_raw), user_goal, world_state, self.mode
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as retry_exc:
                raise PolicyOutputError(str(retry_exc)) from retry_exc
        parse_ms = round((time.perf_counter() - parse_started) * 1000, 3)
        usages = [
            value.get("usage")
            for value in responses
            if isinstance(value, dict) and isinstance(value.get("usage"), dict)
        ]
        timing_values = [
            value.get("timings")
            for value in responses
            if isinstance(value, dict) and isinstance(value.get("timings"), dict)
        ]

        def timing_total(*names: str) -> float | None:
            values = [self._timing(item, *names) for item in timing_values]
            present = [value for value in values if value is not None]
            return sum(present) if present else None

        self.last_call_stats = PolicyCallStats(
            round((time.perf_counter() - started) * 1000, 3),
            len(responses),
            prompt_build_ms=prompt_build_ms,
            tokenization_ms=timing_total("tokenization_ms"),
            prefill_ms=timing_total("prompt_ms", "prefill_ms"),
            generation_ms=timing_total("predicted_ms", "generation_ms"),
            structured_parse_ms=parse_ms,
            retry_ms=retry_ms,
            input_tokens=sum(int(item.get("prompt_tokens") or 0) for item in usages),
            output_tokens=sum(int(item.get("completion_tokens") or 0) for item in usages),
            context_size=len(content.encode("utf-8")),
        )
        return action

    @staticmethod
    def _integer(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _timing(values: dict[str, Any], *names: str) -> float | None:
        for name in names:
            value = values.get(name)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _compact_goal(goal: UserGoal) -> dict[str, Any]:
        payload = goal.as_dict()
        return {
            "schema_version": payload["schema_version"],
            "goal_id": payload["goal_id"],
            "goal": payload["goal"],
            "semantic_action": payload["semantic_action"],
            "execution": payload["execution"],
            "executor": payload["executor"],
            "constraints": payload["constraints"],
            "mutation": payload["mutation"],
            "references": payload["references"][-8:],
        }

    @staticmethod
    def _compact_state(state: WorldState) -> dict[str, Any]:
        def compact_action(item: NextAction) -> dict[str, Any]:
            return {
                "action": item.action.value,
                "agent": item.target_agent.value if item.target_agent else None,
                "target": item.target,
                "tool": item.tool_name,
                "objective": item.objective[:240],
                "mode": item.execution_mode.value if item.execution_mode else None,
                "reason": item.reason_code.value,
            }

        def compact_observation(item: Any) -> dict[str, Any]:
            return {
                "id": item.observation_id,
                "status": item.status.value,
                "summary": item.summary[:1000],
                "facts": list(item.facts[-8:]),
                "errors": list(item.errors[-4:]),
                "authority": item.authority_outcome,
                "tool": item.tool_name,
                "agent": item.agent.value if item.agent else None,
                "verification": item.verification_status.value,
                "goal_completed": item.goal_completed,
            }

        return {
            "state_version": state.state_version,
            "project": state.project.as_dict(),
            "agents": {
                agent.value: value.as_dict() for agent, value in state.agents.items()
            },
            "jobs": [item.as_dict() for item in state.jobs[-4:]],
            "facts": list(state.current_facts[-12:]),
            "unresolved_questions": list(state.unresolved_questions[-6:]),
            "recent_actions": [compact_action(item) for item in state.previous_actions[-4:]],
            "recent_observations": [
                compact_observation(item) for item in state.observations[-4:]
            ],
            "authority_facts": list(state.authority_facts[-8:]),
            "budget": state.budget.as_dict(
                step=state.step,
                model_calls=state.model_calls,
                tool_calls=state.tool_calls,
                delegations=state.delegations,
                elapsed_ms=state.elapsed_ms,
            ),
        }

    @staticmethod
    def _default_action_space(state: WorldState) -> AllowedActionSpace:
        return AllowedActionSpace(
            action_types=tuple(NextActionType),
            candidate_agents=tuple(state.agents),
            agent_modes={
                agent: value.execution_modes for agent, value in state.agents.items()
            },
            tools=state.tools,
            tool_schemas={},
            active_constraints=(),
        )

    @staticmethod
    def _content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("policy response missing choices[0].message.content") from exc
        if not isinstance(content, str):
            raise ValueError("policy response content must be text")
        return content.strip()

    @staticmethod
    def _parse(
        raw: Any,
        user_goal: UserGoal,
        world_state: WorldState,
        mode: str = "shadow",
    ) -> NextAction:
        if not isinstance(raw, dict):
            raise ValueError("policy response must be an object")
        allowed_fields = {
            "action",
            "target_agent",
            "target",
            "tool_name",
            "arguments",
            "objective",
            "execution_mode",
            "required_capabilities",
            "reason_code",
            "evidence_refs",
            "expected_observation",
            "confidence",
            "short_horizon_hint",
        }
        required_fields = {"action", "objective", "reason_code"}
        if not required_fields <= set(raw) or set(raw) - allowed_fields:
            raise ValueError("policy response fields do not match closed schema")
        action_prefix = "bounded-live" if mode == "bounded_live" else "shadow"
        action_id = f"{action_prefix}-{user_goal.goal_id}-{world_state.step + 1}"
        action_type = NextActionType(raw["action"])
        arguments = raw.get("arguments") or {}
        objective = raw["objective"]
        target_agent = raw.get("target_agent")
        target = raw.get("target")
        tool_name = raw.get("tool_name")
        execution_mode = raw.get("execution_mode")
        if tool_name == "filesystem_read_text":
            # The live reader is bounded by an allow-listed size. Canonicalize
            # safe wire-format mistakes without changing the chosen strategy.
            if arguments.get("max_bytes") not in {4096, 16384, 65536, 131072}:
                arguments["max_bytes"] = 16384
            if (
                not arguments.get("path")
                and isinstance(target, str)
                and Path(target).is_absolute()
            ):
                arguments["path"] = target
        if (
            tool_name in {"get_project_git_state", "run_project_tests"}
            and not arguments.get("project_path")
            and world_state.project.path
        ):
            arguments["project_path"] = world_state.project.path
        if action_type in {NextActionType.INSPECT, NextActionType.EXECUTE}:
            # A tool action can never also be a delegation. Removing an
            # irrelevant agent field narrows effects and does not choose a strategy.
            target_agent = None
        elif action_type is NextActionType.DELEGATE:
            # Delegation dispatch is derived from target_agent by the executor.
            # A model-supplied tool cannot override that binding.
            tool_name = None
        if action_type in {
            NextActionType.ASK_USER,
            NextActionType.RESPOND,
            NextActionType.STOP,
        }:
            preferred = arguments.get(
                "summary" if action_type is NextActionType.RESPOND else "question"
            )
            if isinstance(preferred, str) and preferred.strip():
                objective = preferred.strip()[:600]
            # Terminal actions have no effects. Canonicalizing away effect fields
            # can only reduce authority, never grant it.
            target_agent = None
            target = None
            tool_name = None
            execution_mode = None
            arguments = {}
        return NextAction(
            action_id=action_id,
            action=action_type,
            target_agent=(
                Agent(target_agent) if target_agent else None
            ),
            target=target,
            tool_name=tool_name,
            arguments=arguments,
            objective=objective,
            execution_mode=(
                ExecutionMode(execution_mode)
                if execution_mode
                else None
            ),
            required_capabilities=tuple(
                Capability(value) for value in raw.get("required_capabilities", ())
            ),
            reason_code=OrchestrationReasonCode(raw["reason_code"]),
            evidence_refs=tuple(str(value) for value in raw.get("evidence_refs", ())),
            expected_observation=raw.get("expected_observation"),
            confidence=raw.get("confidence"),
            short_horizon_hint=raw.get("short_horizon_hint"),
        )


class ScriptedOrchestrationPolicy:
    """Deterministic policy used by replay/evaluation; never a live executor."""

    model_version = "scripted-replay-v1"

    def __init__(self, actions: list[NextAction] | tuple[NextAction, ...]):
        self._actions = list(actions)
        self.last_call_stats = PolicyCallStats(0.0, 0)

    def decide(
        self,
        user_goal: UserGoal,
        world_state: WorldState,
        allowed_actions: AllowedActionSpace | None = None,
    ) -> NextAction:
        if not self._actions:
            raise RuntimeError("scripted policy exhausted")
        return self._actions.pop(0)
