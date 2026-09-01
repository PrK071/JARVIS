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
_AGENT_REFERENCE = rf"(?:o\s+)?(?:(?:agente|modelo)\s+)?{_AGENT}"
_TRANSFER_VERB = (
    r"(?:mand(?:a|e|ar)|envi(?:a|e|ar)|encaminh(?:a|e|ar)|"
    r"pass(?:a|e|ar)|deleg(?:a|ue|ar)|entreg(?:a|ue|ar)|"
    r"submet(?:a|e|er)|direcion(?:a|e|ar)|redirecion(?:a|e|ar)|"
    r"transf(?:ere|erir|ira)|repass(?:a|e|ar)|atribu(?:a|i|ir)|"
    r"design(?:a|e|ar)|encarreg(?:a|ue|ar)|incumb(?:a|e|ir)|"
    r"comission(?:a|e|ar)|deix(?:a|e|ar)|lev(?:a|e|ar)|"
    r"jog(?:a|ue|ar)|tac(?:a|ar|e)|d(?:a|e)|faz|faca|fazer)"
)
_INVOKE_VERB = (
    r"(?:pec(?:a|am|o)|ped(?:e|ir)|pergunt(?:a|e|ar)|question(?:a|e|ar)|"
    r"solicit(?:a|e|ar)|diz|diga|dizer|avis(?:a|e|ar)|inform(?:a|e|ar)|"
    r"comunic(?:a|ar|e)|comunique|notific(?:a|ar|e)|notifique|cont(?:a|e|ar)|"
    r"explic(?:a|ar|e)|explique|orient(?:a|e|ar)|instr(?:ua|ui|uir)|"
    r"fal(?:a|e|ar)|"
    r"cham(?:a|e|ar)|acion(?:a|e|ar)|us(?:a|e|ar)|utiliz(?:a|e|ar)|"
    r"consult(?:a|e|ar)|coloc(?:a|que|ar)|bot(?:a|e|ar)|poe|ponh(?:a|am))"
)
_EXECUTOR_ACTION = (
    r"(?:analis(?:a|e|ar)|avali(?:a|e|ar)|examin(?:a|e|ar)|investig(?:a|ue|ar)|"
    r"inspecion(?:a|e|ar)|verific(?:a|que|ar)|confer(?:e|ir)|chec(?:a|ar|que)|"
    r"test(?:a|e|ar)|diagnostic(?:a|ar|e|que)|audit(?:a|e|ar)|compar(?:a|e|ar)|"
    r"revis(?:a|e|ar)|trabalh(?:a|e|ar|ando)|cuid(?:a|e|ar|ando)|"
    r"faz|faca|fazer|execut(?:a|e|ar)|realiz(?:a|e|ar)|efetu(?:a|e|ar)|"
    r"produz|produza|produzir|cri(?:a|e|ar)|constru(?:a|ir|i)|mont(?:a|e|ar)|"
    r"prepar(?:a|e|ar)|elabor(?:a|e|ar)|desenvolv(?:a|e|er)|"
    r"implement(?:a|e|ar)|aplic(?:a|ar|e|que)|ger(?:a|e|ar)|"
    r"corr(?:ig(?:a|e|ir)|ij(?:a|am))|resolv(?:a|e|er)|trat(?:a|e|ar)|"
    r"process(?:a|e|ar)|complet(?:a|e|ar)|termin(?:a|e|ar)|finaliz(?:a|e|ar)|"
    r"providenci(?:a|e|ar)|melhor(?:a|e|ar)|aperfeico(?:a|e|ar)|"
    r"otimiz(?:a|e|ar)|refin(?:a|e|ar)|evolu(?:a|i|ir)|ajust(?:a|e|ar)|"
    r"consert(?:a|e|ar)|arrum(?:a|e|ar)|reformul(?:a|e|ar)|"
    r"reestrutur(?:a|e|ar)|reorganiz(?:a|e|ar)|moderniz(?:a|e|ar)|"
    r"atualiz(?:a|e|ar)|simplific(?:a|ar|e|que)|escrev(?:a|e|er)|"
    r"redij(?:a|ir)|reescrev(?:a|e|er)|resum(?:a|e|ir)|document(?:a|e|ar)|"
    r"planej(?:a|e|ar)|organiz(?:a|e|ar)|estrutur(?:a|e|ar)|"
    r"arquitet(?:a|e|ar)|projet(?:a|e|ar)|defin(?:a|e|ir)|"
    r"pesquis(?:a|e|ar)|procur(?:a|e|ar)|busc(?:a|ar|e|que)|"
    r"encontr(?:a|e|ar)|descubr(?:a|ir)|localiz(?:a|e|ar)|identific(?:a|ar|e|que)|"
    r"program(?:a|e|ar)|codific(?:a|ar|e|que)|debug(?:a|ue|ar)|"
    r"depur(?:a|e|ar)|refator(?:a|e|ar)|compil(?:a|e|ar)|integr(?:a|e|ar)|"
    r"continu(?:a|e|ar)|prossig(?:a|ue|uir)|avanc(?:a|ar|e)|retom(?:a|e|ar)|"
    r"delet(?:a|e|ar)|apag(?:a|ue|ar)|exclu(?:a|i|ir)|remov(?:a|e|er)|"
    r"retir(?:a|e|ar)|elimin(?:a|e|ar)|suprim(?:a|e|ir)|descart(?:a|e|ar)|"
    r"extingu(?:a|e|ir)|erradic(?:a|e|ar)|tir(?:a|e|ar)|cort(?:a|e|ar)|"
    r"limp(?:a|e|ar)|esvazi(?:a|e|ar)|zer(?:a|e|ar)|reset(?:a|e|ar)|"
    r"desinstal(?:a|e|ar)|livr(?:a|e|ar)|"
    r"olh(?:a|e|ar))"
)

_CONTEXTUAL_PRONOUN_EXECUTOR = re.compile(
    rf"(?:^|[,;.!?]\s*)(?:por favor[,;:]?\s+)?"
    rf"(?:{_TRANSFER_VERB}|{_INVOKE_VERB})\b"
    rf".{{0,20}}?\b(?:(?:pro|pra|para|ao|a)\s+)?(?:ele|ela|nele|nela)\b"
    rf".{{0,30}}?\b{_EXECUTOR_ACTION}\b"
)


_BINDING_PATTERNS = (
    re.compile(
        rf"\b{_TRANSFER_VERB}\b.{{0,90}}?\b(?:para|pro|pra|ao|a|no|na|com)\s+{_AGENT_REFERENCE}\b"
    ),
    re.compile(
        rf"\b{_TRANSFER_VERB}\b\s+{_AGENT_REFERENCE}\b"
        rf"(?=\s+(?:para|pra|a|que|de|sobre|disso|nisso)\b|\s+{_EXECUTOR_ACTION}\b|\s*$|[,.!?;:])"
    ),
    re.compile(
        rf"\b{_INVOKE_VERB}\b.{{0,50}}?\b(?:ao|pro|pra|para|com)\s+{_AGENT_REFERENCE}\b"
    ),
    re.compile(
        rf"\b{_INVOKE_VERB}"
        rf"\b\s+{_AGENT_REFERENCE}\b"
        rf"(?=\s+(?:para|pra|a|que|de|sobre|disso|nisso)\b|\s+{_EXECUTOR_ACTION}\b|\s*$|[,.!?;:])"
    ),
    re.compile(
        rf"^(?:por favor[,;:]?\s+)?(?:vai\s+la\s+e\s+)?"
        rf"convers(?:a|e|ar)\s+com\s+{_AGENT_REFERENCE}\b"
    ),
    re.compile(
        rf"^(?:por favor[,;:]?\s+)?{_AGENT_REFERENCE}\s*[,;:]\s*{_EXECUTOR_ACTION}\b"
    ),
    re.compile(
        rf"\b(?:quero|preciso)\s+(?:que\s+)?{_AGENT_REFERENCE}\s+"
        rf"(?:{_EXECUTOR_ACTION}|trabalhando|cuidando)\b"
    ),
    re.compile(
        rf"\b(?:isso|isto|essa\s+tarefa|esta\s+tarefa|a\s+tarefa)\s+"
        rf"(?:e\s+(?:pro|pra|para)|fica\s+com)\s+{_AGENT_REFERENCE}\b"
    ),
    re.compile(
        rf"\b{_AGENT_REFERENCE}\s+pode\s+(?:{_EXECUTOR_ACTION})\b"
    ),
)

_META_PREFIX = re.compile(
    r"^(?:como|por que|porque|quando|onde|quem|qual|quais|o que|oq|"
    r"seria possivel|e possivel|posso|poderia|devo|se eu)\b"
)
_CORRECTION_PREFIX = re.compile(
    r"^(?:nao[,;]?\s+)?(?:quis dizer|eu quis dizer|nao era|eu (?:tava|estava) falando)\b"
)
_NEGATION = re.compile(r"\b(?:nao|nunca|jamais|sem)\b")
_NEGATED_BINDING = re.compile(
    rf"\b(?:nao|nunca|jamais|sem)\s+"
    rf"(?:(?:e|era|seria)\s+(?:pra|para)\s+)?"
    rf"(?:(?:quero|quer|deve|precisa)\s+(?:que\s+)?)?"
    rf"(?:{_TRANSFER_VERB}|{_INVOKE_VERB})\b"
    rf".{{0,90}}?\b(?P<negated_agent>codex|deepseek)\b"
)


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
            suffix = normalized[match.end() :].strip(" ,.!?;:")
            if re.search(
                r"\b(?:nao|nunca|jamais|sem)\s+"
                r"(?:(?:e|era|seria)\s+(?:pra|para)\s+)?"
                r"(?:(?:quero|quer|deve|precisa)\s+(?:que\s+)?)?$",
                prefix,
            ):
                continue
            if suffix == "para":
                # In voice input, "fala pro Codex para" can mean "tell Codex
                # to stop" and has no delegation task yet.
                continue
            matches.append((agent, match.start(), match.end()))

    agents = {agent for agent, _start, _end in matches}
    negated_agents = {
        match.group("negated_agent")
        for match in _NEGATED_BINDING.finditer(normalized)
    }
    return agents - negated_agents


def detect_explicit_agent_binding(
    text: str,
    *,
    focused_agent: str | None = None,
) -> ExplicitAgentBinding | None:
    """Bind a named executor only from a positive delegation clause.

    This detector does not select a model and does not route a request. It only
    preserves the executor explicitly named by the user. A bare mention,
    comparison, availability question, hypothetical, or negated invocation does
    not produce a binding.
    """

    normalized_lines = [_plain(line) for line in text.splitlines() if _plain(line)]
    normalized = _plain(text)
    if (
        not normalized
        or _META_PREFIX.match(normalized)
        or _CORRECTION_PREFIX.match(normalized)
    ):
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
    contextual_pronoun = _CONTEXTUAL_PRONOUN_EXECUTOR.search(normalized)
    pronoun_is_meta = "?" in normalized or bool(
        re.search(r"\b(?:como|quando|se eu)\b.{0,45}\b(?:ele|ela|nele|nela)\b", normalized)
    )
    if len(named_agents) == 1 and contextual_pronoun and not pronoun_is_meta:
        agent = next(iter(named_agents))
        if not re.search(
            rf"\b(?:nao|nunca|jamais|sem)\b.{{0,60}}\b{agent}\b",
            normalized,
        ):
            return ExplicitAgentBinding(
                agent,
                evidence="contextual_executor_pronoun",
            )

    if (
        not named_agents
        and focused_agent in SUPPORTED_REQUESTED_AGENTS
        and contextual_pronoun
        and not pronoun_is_meta
        and not _NEGATION.search(normalized[: contextual_pronoun.end()])
    ):
        return ExplicitAgentBinding(
            str(focused_agent),
            evidence="focused_executor_pronoun",
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
