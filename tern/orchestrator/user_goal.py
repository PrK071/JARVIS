"""Deterministic semantic-understanding adapter for the UserGoal contract."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from types import SimpleNamespace
from typing import Any

from .autonomy_foundation import Agent
from .explicit_agent_binding import detect_explicit_agent_binding
from .intent_semantics import Constraint, IntentFrame, IntentFrameBuilder
from .orchestration_contracts import AgentSource, SemanticAction, UserGoal
from .task_requirement_grounding import RequirementValue


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(char for char in normalized if not unicodedata.combining(char)).split()
    )


_QUESTION = re.compile(r"^(?:por que|porque|pq|o que|oq|qual|quais|como)\b")
_PERMISSION = re.compile(
    r"\b(?:se\s+(?:precisar|necessario)|caso\s+(?:precise|necessario)|"
    r"pode\s+usar|se\s+ajudar)\b[^.!?]{0,80}\b(codex|deepseek)\b|"
    r"\b(codex|deepseek)\b[^.!?]{0,50}\bse\s+(?:precisar|ajudar)\b"
)
_FORBIDDEN_AGENT = re.compile(
    r"\b(?:nao|sem)\s+(?:use|usar|chame|chamar|mande|mandar|delegue|delegar)\s+"
    r"(?:o\s+|pro\s+|para\s+o\s+)?(codex|deepseek)\b"
)
_EXECUTION = re.compile(
    r"\b(?:resolve|resolva|corrige|corrija|conserte|arrume|implemente|implementa|"
    r"crie|cria|reescreva|reescreve|melhore|melhora|delete|deleta|deletar|"
    r"apague|apaga|exclua|exclui|remova|remove|limpe|limpa|zere|zera|"
    r"resete|reseta|execute|executa|faca|faz|continue|continua)\b"
)

_SEMANTIC_PATTERNS: tuple[tuple[SemanticAction, re.Pattern[str]], ...] = (
    (SemanticAction.DELETE_OBJECT, re.compile(r"\b(?:delete|deleta|deletar|apague|apaga|exclua|exclui)\b")),
    (SemanticAction.CLEAR_CONTENT, re.compile(r"\b(?:limpe|limpa|esvazie|esvazia)\b")),
    (SemanticAction.RESET_STATE, re.compile(r"\b(?:zere|zera|resete|reseta|reset)\b")),
    (SemanticAction.REMOVE_COMPONENT, re.compile(r"\b(?:remova|remove|retire|retira|corte|corta)\b")),
    (SemanticAction.REPAIR, re.compile(r"\b(?:corrija|corrige|corrigir|conserte|consertar|arrume|arrumar|repare|reparar|resolve|resolva|resolver)\b")),
    (SemanticAction.REWRITE, re.compile(r"\b(?:reescreva|reescreve|reformule|reformula)\b")),
    (SemanticAction.IMPROVE, re.compile(r"\b(?:melhore|melhora|aprimore|otimize|refine)\b")),
    (SemanticAction.CREATE, re.compile(r"\b(?:crie|cria|construa|monte|implemente|implementa)\b")),
    (SemanticAction.ANALYZE, re.compile(r"\b(?:analise|analisa|investigue|investiga|examine|verifique)\b")),
    (SemanticAction.EXPLAIN, re.compile(r"\b(?:explique|explica|por que|porque|pq)\b")),
    (SemanticAction.CONTINUE, re.compile(r"\b(?:continue|continua|prossiga|retome)\b")),
)


class UserGoalBuilder:
    """Build UserGoal without making an executor-selection decision."""

    def __init__(self, frame_builder: IntentFrameBuilder | None = None):
        self.frame_builder = frame_builder or IntentFrameBuilder()

    def build(
        self,
        user_text: str,
        *,
        context: Any | None = None,
        intent_frame: IntentFrame | None = None,
    ) -> UserGoal:
        original = user_text.strip()
        if not original:
            raise ValueError("user_text must not be blank")
        normalized = _plain(original)
        if intent_frame is None:
            context = context or SimpleNamespace(
                active_project=None,
                project_root=None,
                known_projects=(),
            )
            intent_frame, _ = self.frame_builder.build(original, context)

        permitted = self._agents_from_matches(_PERMISSION, normalized)
        forbidden = self._agents_from_matches(_FORBIDDEN_AGENT, normalized)
        explicit_agent = None
        agent_source = None
        if not permitted and not forbidden:
            binding = detect_explicit_agent_binding(original)
            if binding and binding.requested_agent in {"codex", "deepseek"}:
                explicit_agent = Agent(binding.requested_agent)
                agent_source = AgentSource.EXPLICIT_USER

        frame_constraints = set(intent_frame.constraints)
        if Constraint.FORBID_CODEX in frame_constraints:
            forbidden = tuple(dict.fromkeys((*forbidden, Agent.CODEX)))
        if Constraint.FORBID_DEEPSEEK in frame_constraints:
            forbidden = tuple(dict.fromkeys((*forbidden, Agent.DEEPSEEK)))
        mutation_forbidden = bool(
            frame_constraints & {Constraint.FORBID_MUTATION, Constraint.READ_ONLY}
        ) or bool(re.search(r"\bsem\s+(?:alterar|modificar|mexer|editar)\b", normalized))

        semantic_action = self._semantic_action(normalized, intent_frame)
        mutation_required = self._mutation_requirement(
            semantic_action, mutation_forbidden
        )
        question_only = bool(_QUESTION.match(normalized))
        execution_requested = bool(intent_frame.execution_requested or _EXECUTION.search(normalized))
        if question_only and semantic_action in {
            SemanticAction.EXPLAIN,
            SemanticAction.ANALYZE,
            SemanticAction.UNKNOWN,
        }:
            execution_requested = False

        constraints = tuple(item.value for item in intent_frame.constraints)
        evidence = ["user_input:explicit"]
        if explicit_agent:
            evidence.append("executor:explicit_user")
        if permitted:
            evidence.append("executor:conditional_permission")
        if forbidden:
            evidence.append("executor:explicit_prohibition")
        if mutation_forbidden:
            evidence.append("constraint:read_only")

        goal_id = "goal-" + hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
        target = intent_frame.target or original
        return UserGoal(
            goal_id=goal_id,
            summary=original[:1000],
            desired_outcome=target[:1000],
            semantic_action=semantic_action,
            execution_requested=execution_requested,
            explicit_agent=explicit_agent,
            agent_source=agent_source,
            permitted_agents=permitted,
            forbidden_agents=forbidden,
            constraints=constraints,
            mutation_required=mutation_required,
            mutation_forbidden=mutation_forbidden,
            references=tuple(
                value
                for value in (intent_frame.target,)
                if isinstance(value, str) and value.strip()
            ),
            evidence=tuple(evidence),
        )

    @staticmethod
    def _agents_from_matches(
        pattern: re.Pattern[str], normalized: str
    ) -> tuple[Agent, ...]:
        agents: list[Agent] = []
        for match in pattern.finditer(normalized):
            raw = next((value for value in match.groups() if value), None)
            if raw:
                agent = Agent(raw)
                if agent not in agents:
                    agents.append(agent)
        return tuple(agents)

    @staticmethod
    def _semantic_action(normalized: str, frame: IntentFrame) -> SemanticAction:
        if frame.action is not None:
            return SemanticAction(frame.action.value)
        for action, pattern in _SEMANTIC_PATTERNS:
            if pattern.search(normalized):
                return action
        return SemanticAction.UNKNOWN

    @staticmethod
    def _mutation_requirement(
        action: SemanticAction, mutation_forbidden: bool
    ) -> RequirementValue:
        if mutation_forbidden:
            return RequirementValue.FALSE
        if action in {
            SemanticAction.DELETE_OBJECT,
            SemanticAction.REMOVE_COMPONENT,
            SemanticAction.CLEAR_CONTENT,
            SemanticAction.RESET_STATE,
            SemanticAction.REWRITE,
            SemanticAction.CREATE,
            SemanticAction.IMPROVE,
            SemanticAction.REPAIR,
        }:
            return RequirementValue.TRUE
        if action in {SemanticAction.ANALYZE, SemanticAction.EXPLAIN}:
            return RequirementValue.FALSE
        return RequirementValue.UNKNOWN
