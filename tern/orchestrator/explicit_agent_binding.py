from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


EXPLICIT_USER_SOURCE = "explicit_user"
SUPPORTED_REQUESTED_AGENTS = frozenset({"codex", "deepseek"})


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(character for character in normalized if not unicodedata.combining(character)).split()
    )


_AGENT = r"(?P<agent>codex|deepseek)"
_TRANSFER_VERB = (
    r"(?:mand(?:a|e|ar)|envi(?:a|e|ar)|encaminh(?:a|e|ar)|"
    r"pass(?:a|e|ar)|deleg(?:a|ue|ar)|entreg(?:a|ue|ar)|"
    r"submet(?:a|e|er)|direcion(?:a|e|ar)|lev(?:a|e|ar)|jog(?:a|ue|ar))"
)
_INVOKE_VERB = (
    r"(?:pec(?:a|am|o)|ped(?:e|ir)|pergunt(?:a|e|ar)|solicit(?:a|e|ar)|instr(?:ua|ui|uir)|"
    r"fal(?:a|e|ar)|"
    r"cham(?:a|e|ar)|acion(?:a|e|ar)|us(?:a|e|ar)|utiliz(?:a|e|ar)|"
    r"consult(?:a|e|ar)|coloc(?:a|que|ar)|bot(?:a|e|ar)|poe|ponh(?:a|am))"
)
_EXECUTOR_ACTION = (
    r"(?:analis(?:a|e|ar)|revis(?:a|e|ar)|trabalh(?:a|e|ar)|"
    r"faz|faca|fazer|cri(?:a|e|ar)|corrij(?:a|am|ir)|implement(?:a|e|ar)|"
    r"avali(?:a|e|ar)|resolv(?:a|e|er)|verific(?:a|que|ar)|olh(?:a|e|ar))"
)

_CONTEXTUAL_PRONOUN_EXECUTOR = re.compile(
    rf"(?:^|[,;.!?]\s*)(?:por favor[,;:]?\s+)?"
    rf"(?:fal(?:a|e|ar)|diz|diga|mand(?:a|e|ar)|pec(?:a|am|o))\b"
    rf".{{0,20}}?\b(?:pro|pra|para|ao|a)\s+(?:ele|ela)\b"
    rf".{{0,30}}?\b{_EXECUTOR_ACTION}\b"
)


_BINDING_PATTERNS = (
    re.compile(
        rf"\b{_TRANSFER_VERB}\b.{{0,90}}?\b(?:para|pro|pra|ao|a)\s+(?:o\s+)?{_AGENT}\b"
    ),
    re.compile(
        rf"\b{_TRANSFER_VERB}\b\s+(?:o\s+)?{_AGENT}\b"
        rf"(?=\s+(?:para|pra|a|que)\b|\s+{_EXECUTOR_ACTION}\b|\s*$|[,.!?;:])"
    ),
    re.compile(
        rf"\b{_INVOKE_VERB}\b.{{0,50}}?\b(?:ao|pro|pra|para)\s+(?:o\s+)?{_AGENT}\b"
    ),
    re.compile(
        rf"\b(?:cham(?:a|e|ar)|acion(?:a|e|ar)|us(?:a|e|ar)|utiliz(?:a|e|ar)|"
        rf"consult(?:a|e|ar)|coloc(?:a|que|ar)|bot(?:a|e|ar)|poe|ponh(?:a|am))"
        rf"\b\s+(?:o\s+)?{_AGENT}\b"
        rf"(?=\s+(?:para|pra|a|que)\b|\s+{_EXECUTOR_ACTION}\b|\s*$|[,.!?;:])"
    ),
    re.compile(
        rf"^(?:por favor[,;:]?\s+)?{_AGENT}\s*[,;:]\s*{_EXECUTOR_ACTION}\b"
    ),
)

_META_PREFIX = re.compile(
    r"^(?:como|por que|porque|quando|onde|quem|qual|quais|o que|oq|"
    r"seria possivel|e possivel|posso|poderia|devo|se eu)\b"
)
_NEGATION = re.compile(r"\b(?:nao|nunca|jamais|sem)\b")


@dataclass(frozen=True)
class ExplicitAgentBinding:
    requested_agent: str
    requested_agent_source: str = EXPLICIT_USER_SOURCE
    evidence: str = "executor_clause"

    def as_dict(self) -> dict[str, str]:
        return {
            "requested_agent": self.requested_agent,
            "requested_agent_source": self.requested_agent_source,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class AgentRuntimeAvailability:
    tool: str
    tool_registered: bool
    tool_available: bool
    execution_allowed: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_registered": self.tool_registered,
            "tool_available": self.tool_available,
            "execution_allowed": self.execution_allowed,
            "reason": self.reason,
        }


def _positive_binding_agents(normalized: str) -> set[str]:
    matches: list[tuple[str, int, int]] = []
    for pattern in _BINDING_PATTERNS:
        for match in pattern.finditer(normalized):
            agent = match.group("agent")
            prefix = normalized[max(0, match.start() - 24) : match.start()]
            matched_text = match.group(0)
            before_agent = matched_text[: match.start("agent") - match.start()]
            suffix = normalized[match.end() :].strip(" ,.!?;:")
            if _NEGATION.search(prefix) or _NEGATION.search(before_agent):
                continue
            if suffix == "para":
                # In voice input, "fala pro Codex para" can mean "tell Codex
                # to stop" and has no delegation task yet.
                continue
            matches.append((agent, match.start(), match.end()))

    agents = {agent for agent, _start, _end in matches}
    return {
        agent
        for agent in agents
        if not re.search(
            rf"\b(?:nao|nunca|jamais|sem)\b.{{0,60}}\b{agent}\b",
            normalized,
        )
    }


def detect_explicit_agent_binding(text: str) -> ExplicitAgentBinding | None:
    """Bind a named executor only from a positive delegation clause.

    This detector does not select a model and does not route a request. It only
    preserves the executor explicitly named by the user. A bare mention,
    comparison, availability question, hypothetical, or negated invocation does
    not produce a binding.
    """

    normalized_lines = [_plain(line) for line in text.splitlines() if _plain(line)]
    normalized = _plain(text)
    if not normalized or _META_PREFIX.match(normalized):
        return None

    # A leading handoff command owns the following task body. Agent names in
    # examples, requirements, or quoted corpus entries must not erase it.
    if normalized_lines:
        leading = normalized_lines[0]
        leading_named_agents = {
            agent
            for agent in SUPPORTED_REQUESTED_AGENTS
            if re.search(rf"\b{agent}\b", leading)
        }
        leading_binding_agents = _positive_binding_agents(leading)
        if (
            len(leading_named_agents) == 1
            and leading_binding_agents == leading_named_agents
        ):
            return ExplicitAgentBinding(next(iter(leading_binding_agents)))

    named_agents = {
        agent for agent in SUPPORTED_REQUESTED_AGENTS if re.search(rf"\b{agent}\b", normalized)
    }
    if len(named_agents) > 1:
        # Compound/cross-agent plans remain owned by the existing semantic plan.
        return None
    if len(named_agents) == 1 and _CONTEXTUAL_PRONOUN_EXECUTOR.search(normalized):
        agent = next(iter(named_agents))
        if not re.search(
            rf"\b(?:nao|nunca|jamais|sem)\b.{{0,60}}\b{agent}\b",
            normalized,
        ):
            return ExplicitAgentBinding(
                agent,
                evidence="contextual_executor_pronoun",
            )

    agents = _positive_binding_agents(normalized)
    if len(agents) != 1:
        return None
    agent = next(iter(agents))
    return ExplicitAgentBinding(agent)


def availability_for_requested_agent(
    binding: ExplicitAgentBinding,
    context: Any,
    registered_tools: Iterable[str],
) -> AgentRuntimeAvailability:
    tool = f"delegate_to_{binding.requested_agent}"
    registered = tool in set(registered_tools)
    if not registered:
        return AgentRuntimeAvailability(tool, False, False, False, "tool_not_registered")
    if binding.requested_agent == "deepseek":
        if not bool(getattr(context, "deepseek_enabled", False)):
            return AgentRuntimeAvailability(tool, True, False, False, "agent_disabled")
        if not bool(getattr(context, "deepseek_configured", False)):
            return AgentRuntimeAvailability(tool, True, False, False, "agent_not_configured")
    return AgentRuntimeAvailability(tool, True, True, True)
