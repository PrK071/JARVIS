from __future__ import annotations

from tern.orchestrator.autonomy_foundation import Agent
from tern.orchestrator.orchestration_contracts import AgentSource, SemanticAction
from tern.orchestrator.task_requirement_grounding import RequirementValue
from tern.orchestrator.user_goal import UserGoalBuilder


def test_generic_repair_requests_execution_without_selecting_codex() -> None:
    goal = UserGoalBuilder().build("resolve esse erro")
    assert goal.semantic_action is SemanticAction.REPAIR
    assert goal.execution_requested is True
    assert goal.explicit_agent is None


def test_explicit_executor_is_preserved() -> None:
    goal = UserGoalBuilder().build("manda pro Codex corrigir isso")
    assert goal.explicit_agent is Agent.CODEX
    assert goal.agent_source is AgentSource.EXPLICIT_USER
    assert goal.mutation_required is RequirementValue.TRUE


def test_conditional_executor_is_permission_not_selection() -> None:
    goal = UserGoalBuilder().build("se precisar usa o Codex")
    assert goal.explicit_agent is None
    assert goal.permitted_agents == (Agent.CODEX,)

    deepseek = UserGoalBuilder().build("pode usar o DeepSeek se ajudar")
    assert deepseek.explicit_agent is None
    assert deepseek.permitted_agents == (Agent.DEEPSEEK,)


def test_forbidden_executor_is_not_misread_as_explicit_requirement() -> None:
    goal = UserGoalBuilder().build("não use o Codex para resolver isso")
    assert goal.explicit_agent is None
    assert Agent.CODEX in goal.forbidden_agents


def test_read_only_is_preserved_as_mutation_constraint() -> None:
    goal = UserGoalBuilder().build("analisa isso sem alterar nada")
    assert goal.semantic_action is SemanticAction.ANALYZE
    assert goal.mutation_forbidden is True
    assert goal.mutation_required is RequirementValue.FALSE


def test_delete_is_semantic_mutation_without_executor_selection() -> None:
    goal = UserGoalBuilder().build("delete o arquivo temporário")
    assert goal.semantic_action is SemanticAction.DELETE_OBJECT
    assert goal.execution_requested is True
    assert goal.mutation_required is RequirementValue.TRUE
    assert goal.explicit_agent is None


def test_explanation_question_does_not_assume_execution_or_mutation() -> None:
    goal = UserGoalBuilder().build("pq o login está quebrado?")
    assert goal.semantic_action is SemanticAction.EXPLAIN
    assert goal.execution_requested is False
    assert goal.mutation_required is RequirementValue.FALSE
