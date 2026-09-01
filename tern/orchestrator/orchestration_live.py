"""Bounded-live executor adapter: authority first, existing executors second."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .autonomy_foundation import Agent
from .execution_authority import (
    BoundedAuthorityDecision,
    BoundedAuthorityFacts,
    BoundedLiveExecutionAuthority,
    BoundedLiveRiskMatrix,
    OrchestrationMode,
    probe_agent_availability,
)
from .execution_gate import ExecutionMode
from .orchestration_contracts import (
    NextAction,
    NextActionType,
    Observation,
    ObservationSource,
    ObservationStatus,
    UserGoal,
    VerificationStatus,
    WorldState,
)
from .orchestration_policy import ActionValidationResult


@dataclass(frozen=True)
class LiveSinkTiming:
    authority_ms: float
    execution_ms: float
    total_step_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "authority_ms": self.authority_ms,
            "execution_ms": self.execution_ms,
            "total_step_ms": self.total_step_ms,
        }


class LiveActionSink:
    """Translate an authorized proposal into one existing ToolRegistry call."""

    mode = "BOUNDED_LIVE"
    _DIRECT_VERIFICATION_TOOLS = frozenset(
        {
            "resolve_project",
            "find_project_files",
            "filesystem_list",
            "filesystem_read_text",
            "get_project_git_state",
            "review_codex_session",
            "review_deepseek_session",
            "get_codex_job_status",
        }
    )

    def __init__(
        self,
        registry: Any,
        goal: UserGoal,
        *,
        run_id: str,
        conversation_id: str,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        authority: BoundedLiveExecutionAuthority | None = None,
        original_user_text: str | None = None,
        focused_codex_thread_id: str | None = None,
    ):
        self.registry = registry
        self.goal = goal
        self.run_id = run_id
        self.conversation_id = conversation_id
        self.event_callback = event_callback
        self.original_user_text = original_user_text or goal.summary
        self.focused_codex_thread_id = focused_codex_thread_id
        self.authority = authority or BoundedLiveExecutionAuthority(
            OrchestrationMode.BOUNDED_LIVE
        )
        self._records: list[tuple[NextAction, Observation]] = []
        self._executed_fingerprints: set[str] = set()
        self._effect_counts = {
            "tools_executed": 0,
            "delegations": 0,
            "jobs_created": 0,
            "sessions_created": 0,
            "filesystem_mutations": 0,
            "git_mutations": 0,
            "external_effects": 0,
            "live_decisions_changed": 0,
        }
        self.last_timing = LiveSinkTiming(0.0, 0.0, 0.0)

    @property
    def effect_counts(self) -> dict[str, int]:
        return dict(self._effect_counts)

    @property
    def records(self) -> tuple[tuple[NextAction, Observation], ...]:
        return tuple(self._records)

    def record(
        self,
        action: NextAction,
        state: WorldState,
        validation: ActionValidationResult,
    ) -> tuple[Observation, BoundedAuthorityDecision]:
        step_started = time.perf_counter()
        effect_candidate = action.action in {
            NextActionType.INSPECT,
            NextActionType.DELEGATE,
            NextActionType.EXECUTE,
            NextActionType.WAIT,
        }
        fingerprint = action.fingerprint()
        tool_name = self._tool_name(action)
        authority_started = time.perf_counter()
        decision = self.authority.evaluate(
            action,
            self.goal,
            state,
            BoundedAuthorityFacts(
                structural_valid=validation.valid,
                tool_available=(
                    tool_name in set(self.registry.names()) if tool_name else True
                ),
                path_allowed=self._path_allowed(action, state),
                duplicate_action=(
                    effect_candidate and fingerprint in self._executed_fingerprints
                ),
                goal_evidence_sufficient=self._has_evidence(state),
                goal_verified=self._goal_verified(state),
            ),
        )
        authority_ms = round((time.perf_counter() - authority_started) * 1000, 3)
        if not decision.allowed:
            observation = self._blocked_observation(action, decision, validation)
            self._records.append((action, observation))
            self.last_timing = LiveSinkTiming(
                authority_ms,
                0.0,
                round((time.perf_counter() - step_started) * 1000, 3),
            )
            return observation, decision

        if effect_candidate:
            self._executed_fingerprints.add(fingerprint)
        execution_started = time.perf_counter()
        observation = self._execute_allowed(action, state, decision)
        execution_ms = round((time.perf_counter() - execution_started) * 1000, 3)
        self._records.append((action, observation))
        self.last_timing = LiveSinkTiming(
            authority_ms,
            execution_ms,
            round((time.perf_counter() - step_started) * 1000, 3),
        )
        return observation, decision

    def _execute_allowed(
        self,
        action: NextAction,
        state: WorldState,
        decision: BoundedAuthorityDecision,
    ) -> Observation:
        if action.action is NextActionType.ASK_USER:
            return self._control_observation(
                action, ObservationStatus.PENDING, action.objective
            )
        if action.action is NextActionType.RESPOND:
            return self._control_observation(
                action,
                ObservationStatus.SUCCESS,
                action.objective,
                goal_completed=True,
                verification=(
                    VerificationStatus.VERIFIED
                    if self._goal_verified(state)
                    else VerificationStatus.NOT_APPLICABLE
                ),
            )
        if action.action is NextActionType.STOP:
            return self._control_observation(
                action, ObservationStatus.BLOCKED, action.objective
            )

        if action.action is NextActionType.DELEGATE:
            sample = probe_agent_availability(
                self.registry,
                action.target_agent,
                source="bounded_live_dispatch",
            )
            if not sample.available:
                return Observation(
                    observation_id=f"live-{action.action_id}",
                    source=ObservationSource.EXECUTION_AUTHORITY,
                    action_id=action.action_id,
                    status=ObservationStatus.BLOCKED,
                    summary="Agent availability changed before dispatch.",
                    errors=(sample.reason or "AGENT_UNAVAILABLE",),
                    authority_outcome="AVAILABILITY_CHANGED_BEFORE_DISPATCH",
                    agent=action.target_agent,
                )

        tool_name = self._tool_name(action)
        arguments = self._arguments(action, state)
        context = {
            "conversation_id": self.conversation_id,
            # PendingActionStore deduplicates within a turn. A receding-horizon
            # run contains multiple authoritative actions, so scope the turn to
            # the NextAction while retaining the parent run for provenance.
            # This permits a verification command after a mutation without
            # weakening exactly-once for the individual action.
            "turn_id": f"{self.run_id}:{action.action_id}",
            "orchestration_run_id": self.run_id,
            "user_text": self.goal.summary,
            "original_user_text": self.original_user_text,
            "focused_codex_thread_id": self.focused_codex_thread_id,
            "requested_agent_source": (
                self.goal.agent_source.value if self.goal.agent_source else "orchestration_policy"
            ),
            "delegation_action": action.objective,
            "delegation_constraints": self.goal.constraints,
            "delegation_references": self.goal.references,
            "execution_mode": (
                action.execution_mode.value if action.execution_mode else None
            ),
            # Trusted in-process capability: ToolRegistry uses the concrete
            # authority decision (not model JSON or user text) to avoid asking
            # for the same medium-risk overwrite permission twice. High-risk
            # confirmations remain owned by the registry/pending-action path.
            "_bounded_live_authority_decision": decision,
        }
        result = self.registry.execute(
            tool_name,
            arguments,
            context=context,
            event_callback=self.event_callback,
        )
        self._effect_counts["tools_executed"] += 1
        if action.action is NextActionType.DELEGATE:
            self._effect_counts["delegations"] += 1
        if decision.policy.mutation and result.get("ok"):
            self._effect_counts["filesystem_mutations"] += 1
        if result.get("job_id"):
            self._effect_counts["jobs_created"] += 1
        if result.get("session_id") or result.get("thread_id"):
            self._effect_counts["sessions_created"] += 1
        return self._observation_from_result(
            action, tool_name, result, decision.policy.mutation, state
        )

    def _tool_name(self, action: NextAction) -> str | None:
        if action.action is NextActionType.DELEGATE and action.target_agent:
            return f"delegate_to_{action.target_agent.value}"
        if action.action is NextActionType.WAIT:
            return "get_codex_job_status"
        return action.tool_name

    def _arguments(self, action: NextAction, state: WorldState) -> dict[str, Any]:
        arguments = dict(action.arguments)
        if action.action is NextActionType.DELEGATE:
            # The policy chooses the agent and objective. The executor owns the
            # authoritative envelope: model-supplied task/session/tool fields
            # cannot override provenance, affinity, or the selected project.
            project_path = state.project.path or arguments.get("project_path")
            arguments = {
                "task": action.objective,
                "project_path": project_path,
            }
            if action.target_agent is Agent.CODEX:
                arguments.update(
                    continue_current_thread=True,
                    thread_id=None,
                    wait=True,
                )
            elif action.target_agent is Agent.DEEPSEEK:
                if project_path is None:
                    arguments["project_path"] = None
                arguments.update(
                    continue_current_session=True,
                    context="\n".join(state.current_facts[-12:]) or None,
                )
        elif action.action is NextActionType.WAIT:
            active = next(
                (job for job in state.jobs if job.job_id == action.target),
                state.jobs[-1] if state.jobs else None,
            )
            arguments = {
                "job_id": active.job_id if active else action.target,
                "latest": active is None and action.target is None,
            }
        return arguments

    def _path_allowed(self, action: NextAction, state: WorldState) -> bool:
        candidates = [
            action.arguments.get(name)
            for name in ("path", "project_path", "directory", "working_directory")
        ]
        if action.action is NextActionType.DELEGATE:
            candidates.append(action.arguments.get("project_path") or state.project.path)
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.registry.policy.resolve(str(candidate), must_exist=True)
            except Exception:
                return False
        return True

    @staticmethod
    def _has_evidence(state: WorldState) -> bool:
        return bool(
            state.current_facts
            or any(
                item.status is ObservationStatus.SUCCESS
                for item in state.observations
            )
        )

    @staticmethod
    def _goal_verified(state: WorldState) -> bool:
        return any(
            item.verification_status is VerificationStatus.VERIFIED
            for item in state.observations
        )

    @staticmethod
    def _blocked_observation(
        action: NextAction,
        decision: BoundedAuthorityDecision,
        validation: ActionValidationResult,
    ) -> Observation:
        reason = decision.block_reason.value if decision.block_reason else "AUTHORITY_BLOCKED"
        return Observation(
            observation_id=f"authority-{action.action_id}",
            source=ObservationSource.EXECUTION_AUTHORITY,
            action_id=action.action_id,
            status=ObservationStatus.BLOCKED,
            summary=f"ExecutionAuthority blocked the proposed action: {reason}.",
            errors=tuple(dict.fromkeys((reason, *validation.violations))),
            authority_outcome=reason,
            tool_name=action.tool_name,
            agent=action.target_agent,
        )

    @staticmethod
    def _control_observation(
        action: NextAction,
        status: ObservationStatus,
        summary: str,
        *,
        goal_completed: bool = False,
        verification: VerificationStatus = VerificationStatus.NOT_APPLICABLE,
    ) -> Observation:
        return Observation(
            observation_id=f"control-{action.action_id}",
            source=ObservationSource.LIVE_TOOL,
            action_id=action.action_id,
            status=status,
            summary=summary,
            authority_outcome="ALLOW",
            verification_status=verification,
            goal_completed=goal_completed,
        )

    @staticmethod
    def _observation_from_result(
        action: NextAction,
        tool_name: str,
        result: Mapping[str, Any],
        mutation: bool,
        state: WorldState,
    ) -> Observation:
        ok = bool(result.get("ok"))
        status_value = str(result.get("status") or "").casefold()
        pending = status_value in {
            "queued",
            "starting",
            "running",
            "steering",
            "reconnecting",
        }
        test_command_completed = bool(
            tool_name == "run_project_tests"
            and result.get("returncode") is not None
        )
        operational_success = ok or test_command_completed
        status = (
            ObservationStatus.PENDING
            if ok and pending
            else ObservationStatus.SUCCESS
            if operational_success
            else ObservationStatus.FAILURE
        )
        summary_value = next(
            (
                result.get(key)
                for key in (
                    "response",
                    "result",
                    "message",
                    "summary",
                    "content",
                    "output",
                    "diff_stat",
                )
                if isinstance(result.get(key), str) and result.get(key)
            ),
            None,
        )
        if summary_value is None:
            summary_value = json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key
                    in {
                        "ok",
                        "status",
                        "job_id",
                        "session_id",
                        "thread_id",
                        "path",
                        "root",
                        "project_id",
                        "name",
                        "branch",
                        "working_tree",
                        "changed_files",
                        "status",
                        "matches",
                        "entries",
                        "files",
                        "alternatives",
                        "returncode",
                        "passed",
                        "failed",
                    }
                },
                ensure_ascii=False,
            )
        summary = str(summary_value)[:1600] or f"{tool_name} completed"
        facts = [
            f"tool:{tool_name}:{'ok' if operational_success else 'failed'}"
        ]
        job_id = result.get("job_id")
        if job_id:
            facts.append(f"job_status:{job_id}:{result.get('status') or 'unknown'}")
        if result.get("returncode") is not None:
            facts.append(f"returncode:{result.get('returncode')}")
        project_root = result.get("root")
        if project_root:
            facts.append(f"project_path:{project_root}")
        if result.get("project_id"):
            facts.append(f"project_id:{result['project_id']}")
        if result.get("branch"):
            facts.append(f"git_branch:{result['branch']}")
        if result.get("working_tree"):
            facts.append(f"working_tree:{result['working_tree']}")
        artifacts = tuple(
            str(value)
            for value in (
                result.get("path"),
                result.get("root"),
                result.get("job_id"),
                result.get("session_id"),
                result.get("thread_id"),
            )
            if value
        )
        state_changes = ()
        if mutation and ok:
            changes = [f"mutation:{tool_name}"]
            if result.get("path"):
                changes.append(f"path:{result['path']}")
            state_changes = tuple(changes)
        verified = bool(
            ok
            and (
                tool_name in LiveActionSink._DIRECT_VERIFICATION_TOOLS
                or (
                    tool_name == "run_project_tests"
                    and int(result.get("returncode", 1)) == 0
                )
            )
        )
        verification = (
            VerificationStatus.VERIFIED
            if verified
            else VerificationStatus.FAILED
            if test_command_completed and int(result.get("returncode", 1)) != 0
            else VerificationStatus.UNVERIFIED
        )
        errors = () if operational_success else (
            str(result.get("error") or "TOOL_FAILED"),
            str(result.get("message") or "")[:500],
        )
        source = (
            ObservationSource.LIVE_AGENT
            if action.action is NextActionType.DELEGATE
            else ObservationSource.LIVE_TOOL
        )
        return Observation(
            observation_id=f"live-{action.action_id}",
            source=source,
            action_id=action.action_id,
            status=status,
            summary=summary,
            facts=tuple(facts),
            artifacts=artifacts,
            state_changes=state_changes,
            errors=tuple(value for value in errors if value),
            authority_outcome="ALLOW",
            tool_name=tool_name,
            agent=action.target_agent,
            verification_status=verification,
            goal_completed=bool(verified and any(item.state_changes for item in state.observations)),
        )


class BoundedLiveOrchestrationRunner:
    """Build factual state and run the policy through the bounded-live sink."""

    def __init__(
        self,
        client: Any,
        registry: Any,
        *,
        budget: Any,
        policy: Any | None = None,
        fast_path_enabled: bool = True,
        decision_cache_enabled: bool = True,
        decision_cache_max_entries: int = 128,
    ):
        # Local imports keep the executor adapter independent from the shadow
        # loop at module import time and avoid coupling the loop to this runtime.
        from .orchestration_fast_path import (
            ActionSpaceBuilder,
            OrchestrationDecisionCache,
            OrchestrationFastPath,
        )
        from .orchestration_policy import QwenOrchestrationPolicy
        from .orchestration_state import WorldStateBuilder
        from .user_goal import UserGoalBuilder

        self.registry = registry
        self.budget = budget
        self.policy = policy or QwenOrchestrationPolicy(
            client, mode=OrchestrationMode.BOUNDED_LIVE.value
        )
        self.goal_builder = UserGoalBuilder()
        self.state_builder = WorldStateBuilder()
        self.action_space_builder = ActionSpaceBuilder()
        self.fast_path = OrchestrationFastPath() if fast_path_enabled else None
        self.decision_cache = (
            OrchestrationDecisionCache(decision_cache_max_entries)
            if decision_cache_enabled
            else None
        )

    def run(
        self,
        user_text: str,
        *,
        runtime_context: Any,
        conversation_id: str,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        from .execution_gate_shadow import capability_baseline_from_registry
        from .orchestration_loop import OrchestrationLoop

        run_id = f"orchestration-{uuid.uuid4()}"
        run_started = time.perf_counter()
        goal = self.goal_builder.build(user_text, context=runtime_context)
        codex_sample = probe_agent_availability(
            self.registry,
            Agent.CODEX,
            source="bounded_live_world_state",
        )
        baseline = capability_baseline_from_registry(
            self.registry,
            local_model_available=True,
            codex_available=codex_sample.available,
        )
        context_project_root = (
            getattr(runtime_context, "focused_project_root", None)
            or getattr(runtime_context, "project_root", None)
        )
        project_root = self._explicit_project_root(user_text) or context_project_root
        same_context_project = self._same_path(project_root, context_project_root)
        focused_agent = getattr(runtime_context, "focused_agent", None)
        focused_session = getattr(runtime_context, "focused_session", None)
        focused_thread = (
            focused_session
            if same_context_project and focused_agent == Agent.CODEX.value
            else None
        )
        authority_facts = [
            "BOUNDED_LIVE",
            "EXECUTION_AUTHORITY_REQUIRED",
            "QWEN_CANNOT_GRANT_PERMISSION",
        ]
        if focused_thread:
            authority_facts.append("FOCUSED_CODEX_SESSION")
        if (
            same_context_project
            and focused_agent == Agent.DEEPSEEK.value
            and focused_session
        ):
            authority_facts.append("FOCUSED_DEEPSEEK_SESSION")
        project_snapshot = (
            {
                "project_path": project_root,
                "git_status": "unknown",
                "known_test_state": "unknown",
            }
            if project_root
            else None
        )
        job_id = (
            getattr(runtime_context, "codex_job_id", None)
            if same_context_project
            else None
        )
        jobs = (
            (
                {
                    "job_id": job_id,
                    "agent": Agent.CODEX.value,
                    "status": getattr(runtime_context, "codex_job_status", None)
                    or "unknown",
                },
            )
            if job_id
            else ()
        )
        built = self.state_builder.build(
            goal,
            baseline=baseline,
            project_snapshot=project_snapshot,
            tool_names=self.registry.names(),
            jobs=jobs,
            authority_facts=tuple(authority_facts),
            budget=self.budget,
        )
        sink = LiveActionSink(
            self.registry,
            goal,
            run_id=run_id,
            conversation_id=conversation_id,
            event_callback=event_callback,
            original_user_text=user_text,
            focused_codex_thread_id=focused_thread,
        )
        result = OrchestrationLoop(
            self.policy,
            sink=sink,
            action_space_builder=self.action_space_builder,
            tool_specs=self.registry.specs(),
            fast_path=self.fast_path,
            decision_cache=self.decision_cache,
        ).run(goal, built.state)
        record = result.as_dict()
        record["run_id"] = run_id
        record["world_state_build_ms"] = built.build_ms
        record["telemetry"] = self._telemetry(
            result,
            run_id=run_id,
            build_ms=built.build_ms,
            wall_ms=round((time.perf_counter() - run_started) * 1000, 3),
        )
        record["learning_cases"] = self._learning_cases(goal, result)
        logger = getattr(self.registry, "logger", None)
        write_event = getattr(logger, "write_event", None)
        if callable(write_event):
            write_event("orchestration_bounded_live", **record)
            for case in record["learning_cases"]:
                write_event("orchestration_learning_case", run_id=run_id, **case)
        answer = self._answer(result)
        return {
            "ok": True,
            "answer": answer,
            "tool_calls": result.final_state.tool_calls,
            "usage": self._usage(result),
            "decision": {
                "intent": "ORCHESTRATION",
                "reason_code": result.termination_reason,
                "orchestration": record,
            },
            "orchestration": record,
        }

    def _explicit_project_root(self, user_text: str) -> str | None:
        for raw in re.findall(r"(?i)(?:[A-Z]:\\[^\s,;]+)", user_text):
            candidate = raw.rstrip(".?!:)")
            try:
                resolved = self.registry.policy.resolve(candidate, must_exist=True)
            except Exception:
                continue
            if resolved.is_file():
                resolved = resolved.parent
            return str(resolved)
        return None

    @staticmethod
    def _same_path(left: str | None, right: str | None) -> bool:
        if not left or not right:
            return False
        try:
            return Path(left).resolve() == Path(right).resolve()
        except (OSError, ValueError):
            return False

    @staticmethod
    def _answer(result: Any) -> str:
        if not result.records:
            return (
                "Não foi possível iniciar a orquestração: "
                f"{result.termination_reason}."
            )
        last = result.records[-1]
        if result.termination_reason == "GOAL_COMPLETED":
            verified = any(
                item.observation.verification_status is VerificationStatus.VERIFIED
                for item in result.records
            )
            prefix = "Objetivo concluído e verificado." if verified else "Objetivo concluído."
            detail = last.observation.summary or last.next_action.objective
            return f"{prefix}\n\n{detail}" if detail else prefix
        return last.next_action.objective or last.observation.summary

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * percentile))
        return round(ordered[index], 3)

    def _telemetry(
        self,
        result: Any,
        *,
        run_id: str,
        build_ms: float,
        wall_ms: float,
    ) -> dict[str, Any]:
        records = result.records
        policy_latencies = [
            item.policy_inference_ms
            for item in records
            if item.decision_source == "POLICY"
        ]
        policy_stats = [dict(item.policy_stats or {}) for item in records]
        authority_blocks = sum(
            not bool(getattr(item.authority_shadow_result, "allowed", True))
            for item in records
        )
        interventions = sum(
            item.next_action.action is NextActionType.ASK_USER for item in records
        )
        mutations = sum(bool(item.observation.state_changes) for item in records)
        verifications = sum(
            item.observation.verification_status is VerificationStatus.VERIFIED
            for item in records
        )
        completed = bool(
            result.termination_reason == "GOAL_COMPLETED"
            and records
            and records[-1].next_action.action is NextActionType.RESPOND
            and records[-1].observation.status is ObservationStatus.SUCCESS
        )
        looped = result.termination_reason in {
            "LOOP_DETECTED",
            "NO_PROGRESS",
            "REPEATED_ACTION",
        }
        telemetry = dict(result.telemetry or {})
        critical = tuple(result.critical_shadow_violations)
        blocked_indexes = [
            index
            for index, item in enumerate(records)
            if not bool(getattr(item.authority_shadow_result, "allowed", True))
        ]
        recovered_blocks = sum(
            any(
                later.observation.status is ObservationStatus.SUCCESS
                for later in records[index + 1 :]
            )
            for index in blocked_indexes
        )
        premature_responses = sum(
            item.observation.authority_outcome == "PREMATURE_RESPONSE"
            for item in records
        )
        premature_mutations = sum(
            "MUTATION_WITHOUT_REQUIREMENT_OR_AUTHORITY"
            in item.validation.critical_violations
            for item in records
        )
        def sum_policy_timing(name: str) -> float:
            return round(
                sum(float(item.get(name) or 0.0) for item in policy_stats),
                3,
            )

        return {
            "run_id": run_id,
            "goal_id": result.goal_id,
            "steps": len(records),
            "qwen_decisions": sum(item.decision_source == "POLICY" for item in records),
            "fast_path_decisions": result.fast_path_decisions,
            "decision_cache_hits": result.decision_cache_hits,
            "decision_cache_misses": result.decision_cache_misses,
            "tool_calls": result.final_state.tool_calls,
            "delegations": result.final_state.delegations,
            "agent_results": sum(
                item.observation.source is ObservationSource.LIVE_AGENT
                for item in records
            ),
            "authority_allows": len(records) - authority_blocks,
            "authority_blocks": authority_blocks,
            "authority_block_recovery_rate": recovered_blocks
            / max(len(blocked_indexes), 1),
            "replans": max(len(records) - 1, 0),
            "user_interventions": interventions,
            "user_intervention_rate": interventions / max(len(records), 1),
            "mutations": mutations,
            "verifications": verifications,
            "loops": int(looped),
            "loop_rate": float(looped),
            "no_progress_events": int(result.termination_reason == "NO_PROGRESS"),
            "world_state_build_ms": build_ms,
            "prompt_build_ms": sum_policy_timing("prompt_build_ms"),
            "tokenization_ms": sum_policy_timing("tokenization_ms"),
            "prefill_ms": sum_policy_timing("prefill_ms"),
            "generation_ms": sum_policy_timing("generation_ms"),
            "structured_parse_ms": sum_policy_timing("structured_parse_ms"),
            "retry_ms": sum_policy_timing("retry_ms"),
            "total_policy_ms": telemetry.get("policy_inference_ms", 0.0),
            "total_step_ms": round(
                sum(item.total_step_ms for item in records),
                3,
            ),
            "policy_time_ms": telemetry.get("policy_inference_ms", 0.0),
            "execution_time_ms": telemetry.get("execution_ms", 0.0),
            "authority_ms": telemetry.get("authority_ms", 0.0),
            "validation_ms": telemetry.get("validation_ms", 0.0),
            "state_reduce_ms": telemetry.get("state_reduce_ms", 0.0),
            "total_time_ms": wall_ms,
            "successful_completion": completed,
            "autonomous_goal_completion": bool(completed and not interventions),
            "verified_goal_completion": bool(completed and verifications),
            "failure_reason": None if completed else result.termination_reason,
            "critical_violations": list(critical),
            "critical_violation_rate": len(critical) / max(len(records), 1),
            "explicit_agent_preserved": not any(
                value in critical
                for value in ("IGNORED_EXPLICIT_AGENT", "SILENT_AGENT_SUBSTITUTION")
            ),
            "read_only_preserved": "VIOLATED_READ_ONLY" not in critical,
            "forbidden_agent_preserved": "USED_FORBIDDEN_AGENT" not in critical,
            "silent_substitutions": int("SILENT_AGENT_SUBSTITUTION" in critical),
            "premature_response_rate": premature_responses / max(len(records), 1),
            "premature_mutation_rate": premature_mutations / max(len(records), 1),
            "policy_latency_p50_ms": self._percentile(policy_latencies, 0.50),
            "policy_latency_p90_ms": self._percentile(policy_latencies, 0.90),
            "policy_latency_p95_ms": self._percentile(policy_latencies, 0.95),
            "input_tokens": sum(
                int(item.get("input_tokens") or 0) for item in policy_stats
            ),
            "output_tokens": sum(
                int(item.get("output_tokens") or 0) for item in policy_stats
            ),
            "context_size": max(
                (int(item.get("context_size") or 0) for item in policy_stats),
                default=0,
            ),
            "model_calls": result.model_calls,
            "fast_path_hit_rate": result.fast_path_decisions / max(len(records), 1),
            "decision_cache_hit_rate": result.decision_cache_hits
            / max(result.decision_cache_hits + result.decision_cache_misses, 1),
        }

    @staticmethod
    def _usage(result: Any) -> dict[str, int]:
        stats = [dict(item.policy_stats or {}) for item in result.records]
        return {
            "prompt_tokens": sum(int(item.get("input_tokens") or 0) for item in stats),
            "completion_tokens": sum(int(item.get("output_tokens") or 0) for item in stats),
            "model_calls": result.model_calls,
        }

    @staticmethod
    def _learning_cases(goal: UserGoal, result: Any) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for index, item in enumerate(result.records):
            labels: list[str] = []
            if item.validation.critical_violations:
                labels.extend(item.validation.critical_violations)
            blocked = not bool(getattr(item.authority_shadow_result, "allowed", True))
            if blocked:
                recovered = any(
                    later.observation.status is ObservationStatus.SUCCESS
                    for later in result.records[index + 1 :]
                )
                labels.append(
                    "AUTHORITY_BLOCK_RECOVERED"
                    if recovered
                    else "AUTHORITY_BLOCK_NOT_RECOVERED"
                )
            if (
                item.next_action.action is NextActionType.ASK_USER
                and item.observation.status is ObservationStatus.SUCCESS
            ):
                labels.append("ASK_USER_TOO_EARLY")
            cases.append(
                {
                    "goal": goal.as_dict(),
                    "world_state_version": item.world_state_version,
                    "chosen": item.next_action.as_dict(),
                    "observation": item.observation.as_dict(),
                    "evaluation": labels or ["UNREVIEWED"],
                }
            )
        if len(result.records) > 1:
            trajectory = (
                "GOOD_MULTI_STEP_TRAJECTORY"
                if result.termination_reason == "GOAL_COMPLETED"
                else "BAD_MULTI_STEP_TRAJECTORY"
            )
            if cases:
                cases[-1]["evaluation"].append(trajectory)
        return cases
