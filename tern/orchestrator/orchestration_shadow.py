"""Opt-in live-path observer adapter for the Phase 1.75 shadow loop."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from .autonomy_foundation import Agent, CapabilityBaseline
from .orchestration_contracts import OrchestrationBudget
from .orchestration_loop import OrchestrationLoop
from .orchestration_policy import QwenOrchestrationPolicy
from .orchestration_state import WorldStateBuilder
from .user_goal import UserGoalBuilder


class ShadowOrchestrationObserver:
    """Consumes pure snapshots and returns a log record; owns no live executor."""

    __slots__ = ("policy", "budget", "goal_builder", "state_builder")

    def __init__(self, client: Any, *, budget: OrchestrationBudget):
        self.policy = QwenOrchestrationPolicy(client)
        self.budget = budget
        self.goal_builder = UserGoalBuilder()
        self.state_builder = WorldStateBuilder()

    def observe_once(
        self,
        user_text: str,
        *,
        baseline: CapabilityBaseline,
        eligibility_overrides: Mapping[Agent, bool] | None,
        intent_frame: Any | None,
        project_snapshot: Mapping[str, Any] | None,
        tool_names: Iterable[str],
        jobs: Iterable[Mapping[str, Any]],
        authority_facts: Iterable[str],
        legacy_facts: Mapping[str, Any],
    ) -> dict[str, Any]:
        goal = self.goal_builder.build(user_text, intent_frame=intent_frame)
        # A live request provides no replay observation. One policy step is enough
        # to observe its current decision without wasting calls on synthetic facts.
        one_step_budget = replace(self.budget, max_steps=1)
        built = self.state_builder.build(
            goal,
            baseline=baseline,
            eligibility_overrides=eligibility_overrides,
            project_snapshot=project_snapshot,
            tool_names=tool_names,
            jobs=jobs,
            authority_facts=authority_facts,
            budget=one_step_budget,
        )
        result = OrchestrationLoop(self.policy).run(goal, built.state)
        record = result.as_dict()
        record["world_state_build_ms"] = built.build_ms
        record["legacy_comparison"] = self._compare_legacy(
            legacy_facts, result.records[0].next_action.as_dict() if result.records else None
        )
        return record

    @staticmethod
    def _compare_legacy(
        legacy: Mapping[str, Any], shadow: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if shadow is None:
            return {"classification": "ambiguous", "reason": "NO_SHADOW_ACTION"}
        legacy_action = str(legacy.get("selected_action") or "")
        legacy_agent = (
            "codex"
            if "codex" in legacy_action
            else "deepseek"
            if "deepseek" in legacy_action
            else None
        )
        shadow_agent = shadow.get("target_agent")
        if legacy_agent == shadow_agent:
            classification = "agreement"
        else:
            classification = "divergence"
        return {
            "classification": classification,
            "legacy_intent": legacy.get("intent"),
            "legacy_reason_code": legacy.get("reason_code"),
            "legacy_agent": legacy_agent,
            "shadow_agent": shadow_agent,
        }
