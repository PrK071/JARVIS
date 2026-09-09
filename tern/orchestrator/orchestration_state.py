"""Compact WorldState construction and deterministic observation reduction."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .autonomy_foundation import Agent, CapabilityBaseline
from .execution_gate import ExecutionMode
from .orchestration_contracts import (
    AgentState,
    JobState,
    NextAction,
    NextActionType,
    Observation,
    ObservationStatus,
    OrchestrationBudget,
    ProjectState,
    UserGoal,
    WorldState,
    VerificationStatus,
)
from .project_intelligence import ProjectSnapshot
from .task_requirement_grounding import GroundedAgentEligibility


@dataclass(frozen=True)
class WorldStateBuildResult:
    state: WorldState
    build_ms: float


def _dedupe_bounded(values: Iterable[str], limit: int) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in result:
            result.append(clean)
    return tuple(result[-limit:])


class WorldStateBuilder:
    """Compose immutable facts from existing snapshots; no runtime handles escape."""

    def build(
        self,
        goal: UserGoal,
        *,
        baseline: CapabilityBaseline,
        eligibility: Mapping[Agent, GroundedAgentEligibility] | None = None,
        eligibility_overrides: Mapping[Agent, bool] | None = None,
        project_snapshot: ProjectSnapshot | Mapping[str, Any] | None = None,
        tool_names: Iterable[str] = (),
        jobs: Iterable[Mapping[str, Any]] = (),
        observations: Iterable[Observation] = (),
        previous_actions: Iterable[NextAction] = (),
        current_facts: Iterable[str] = (),
        unresolved_questions: Iterable[str] = (),
        authority_facts: Iterable[str] = ("SHADOW_ONLY", "LIVE_AUTHORITY_UNCHANGED"),
        budget: OrchestrationBudget | None = None,
    ) -> WorldStateBuildResult:
        started = time.perf_counter()
        budget = budget or OrchestrationBudget()
        state = WorldState(
            goal_id=goal.goal_id,
            state_version=0,
            project=self._project(project_snapshot, budget.max_context_items),
            agents=self._agents(baseline, eligibility, eligibility_overrides),
            tools=_dedupe_bounded(sorted(tool_names), budget.max_context_items),
            observations=tuple(observations)[-budget.max_observations :],
            jobs=self._jobs(jobs, budget.max_context_items),
            previous_actions=tuple(previous_actions)[-budget.max_action_history :],
            current_facts=_dedupe_bounded(current_facts, budget.max_context_items),
            unresolved_questions=_dedupe_bounded(
                unresolved_questions, budget.max_context_items
            ),
            authority_facts=_dedupe_bounded(
                authority_facts, budget.max_context_items
            ),
            budget=budget,
        )
        return WorldStateBuildResult(
            state=state,
            build_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    @staticmethod
    def _project(
        snapshot: ProjectSnapshot | Mapping[str, Any] | None,
        max_items: int,
    ) -> ProjectState:
        if snapshot is None:
            return ProjectState()
        if isinstance(snapshot, ProjectSnapshot):
            compact = snapshot.compact(
                max_entries=min(max_items, 24), max_context_bytes=12_000
            )
        else:
            compact = dict(snapshot)
        path = str(compact.get("project_path") or compact.get("path") or "") or None
        languages = tuple(str(item) for item in compact.get("languages", ()) if item)
        context_refs = _dedupe_bounded(
            (
                *compact.get("important_files", ()),
                *compact.get("entry_points", ()),
                *compact.get("tests", ()),
                *compact.get("configuration_files", ()),
                *compact.get("modified_files", ()),
                *compact.get("untracked_files", ()),
            ),
            max_items,
        )
        return ProjectState(
            project_id=Path(path).name if path else None,
            path=path,
            project_type=",".join(languages[:4]) or None,
            branch=compact.get("git_branch") or compact.get("branch"),
            working_tree=str(compact.get("git_status") or compact.get("working_tree") or "unknown"),
            test_state=str(compact.get("known_test_state") or compact.get("tests_state") or "unknown"),
            context_refs=context_refs,
        )

    @staticmethod
    def _agents(
        baseline: CapabilityBaseline,
        eligibility: Mapping[Agent, GroundedAgentEligibility] | None,
        eligibility_overrides: Mapping[Agent, bool] | None,
    ) -> dict[Agent, AgentState]:
        results: dict[Agent, AgentState] = {}
        for agent, profile in baseline.profiles.items():
            availability = baseline.availability.get(agent)
            grounded = (eligibility or {}).get(agent)
            available = bool(availability and availability.available)
            reason_codes: list[str] = []
            if availability and availability.reason_code:
                reason_codes.append(availability.reason_code)
            if grounded:
                reason_codes.extend(grounded.reason_codes)
            modes = [ExecutionMode.READ_ONLY]
            if any(
                capability.value in {
                    "mutation_capable",
                    "filesystem_write",
                    "repository_write",
                    "code_edit",
                }
                for capability in profile.capabilities
            ):
                modes.append(ExecutionMode.MUTATION)
            results[agent] = AgentState(
                agent=agent,
                availability_known=availability is not None,
                available=available,
                eligible=(
                    grounded.eligible
                    if grounded
                    else (eligibility_overrides or {}).get(agent)
                ),
                capabilities=tuple(sorted(profile.capabilities, key=lambda item: item.value)),
                execution_modes=tuple(modes),
                reason_codes=tuple(dict.fromkeys(reason_codes)),
            )
        return results

    @staticmethod
    def _jobs(values: Iterable[Mapping[str, Any]], limit: int) -> tuple[JobState, ...]:
        jobs: list[JobState] = []
        for raw in values:
            job_id = str(raw.get("job_id") or "").strip()
            status = str(raw.get("status") or "unknown").strip()
            if not job_id:
                continue
            raw_agent = str(raw.get("agent") or "codex").casefold()
            try:
                agent = Agent(raw_agent)
            except ValueError:
                continue
            objective = raw.get("objective_ref") or raw.get("task_summary")
            jobs.append(
                JobState(
                    job_id=job_id,
                    agent=agent,
                    status=status,
                    objective_ref=str(objective)[:240] if objective else None,
                )
            )
        return tuple(jobs[-limit:])


class WorldStateReducer:
    """The only Phase 1.75 component allowed to update WorldState."""

    def reduce(
        self,
        state: WorldState,
        action: NextAction,
        observation: Observation,
    ) -> WorldState:
        if observation.action_id != action.action_id:
            raise ValueError("observation action_id does not match action")
        budget = state.budget
        facts = _dedupe_bounded(
            (*state.current_facts, *observation.facts), budget.max_context_items
        )
        questions = list(state.unresolved_questions)
        if action.action is NextActionType.ASK_USER:
            questions.append(action.objective)
        if observation.goal_completed:
            questions.clear()
        failures = state.failure_count + (
            1
            if observation.status
            in {ObservationStatus.FAILURE, ObservationStatus.BLOCKED}
            else 0
        )
        jobs_by_id = {item.job_id: item for item in state.jobs}
        for fact in observation.facts:
            if not fact.startswith("job_status:"):
                continue
            _, job_id, status = fact.split(":", 2)
            current = jobs_by_id.get(job_id)
            agent = observation.agent or (current.agent if current else Agent.CODEX)
            jobs_by_id[job_id] = JobState(
                job_id=job_id,
                agent=agent,
                status=status,
                objective_ref=(current.objective_ref if current else action.objective[:240]),
            )
        project = state.project
        project_changes: dict[str, Any] = {}
        for fact in observation.facts:
            if fact.startswith("project_path:"):
                project_changes["path"] = fact.split(":", 1)[1]
            elif fact.startswith("project_id:"):
                project_changes["project_id"] = fact.split(":", 1)[1]
            elif fact.startswith("git_branch:"):
                project_changes["branch"] = fact.split(":", 1)[1]
            elif fact.startswith("working_tree:"):
                project_changes["working_tree"] = fact.split(":", 1)[1]
        if project_changes:
            project = replace(project, **project_changes)
        if observation.tool_name == "run_project_tests":
            project = replace(
                project,
                test_state=(
                    "passed"
                    if observation.verification_status is VerificationStatus.VERIFIED
                    else "failed"
                ),
            )
        authority_facts = list(state.authority_facts)
        if observation.authority_outcome and observation.authority_outcome != "ALLOW":
            authority_facts.append(f"AUTHORITY:{observation.authority_outcome}")
        return state.evolve(
            state_version=state.state_version + 1,
            step=state.step + 1,
            failure_count=failures,
            observations=(*state.observations, observation)[-budget.max_observations :],
            previous_actions=(*state.previous_actions, action)[
                -budget.max_action_history :
            ],
            current_facts=facts,
            unresolved_questions=_dedupe_bounded(
                questions, budget.max_context_items
            ),
            jobs=tuple(jobs_by_id.values())[-budget.max_context_items :],
            project=project,
            authority_facts=_dedupe_bounded(
                authority_facts, budget.max_context_items
            ),
        )
