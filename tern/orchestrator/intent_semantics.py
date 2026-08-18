from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class SpeechAct(str, Enum):
    QUESTION = "QUESTION"
    REQUEST = "REQUEST"
    COMMAND = "COMMAND"
    EXPLANATION_REQUEST = "EXPLANATION_REQUEST"
    STATUS_QUERY = "STATUS_QUERY"
    REFERENCE_QUERY = "REFERENCE_QUERY"
    CONFIRMATION = "CONFIRMATION"
    CORRECTION = "CORRECTION"


class Constraint(str, Enum):
    FORBID_CODEX = "FORBID_CODEX"
    FORBID_DEEPSEEK = "FORBID_DEEPSEEK"
    FORBID_MUTATION = "FORBID_MUTATION"
    FORBID_NEW_TURN = "FORBID_NEW_TURN"
    FORBID_CANCEL = "FORBID_CANCEL"
    FORBID_DELEGATION = "FORBID_DELEGATION"
    ANSWER_SELF = "ANSWER_SELF"
    READ_ONLY = "READ_ONLY"
    BACKGROUND = "BACKGROUND"
    WAIT_FOR_RESULT = "WAIT_FOR_RESULT"
    ORDERED = "ORDERED"


class FollowupType(str, Enum):
    NEW_REQUEST = "NEW_REQUEST"
    CONTINUATION = "CONTINUATION"
    MODIFICATION = "MODIFICATION"
    STATUS_FOLLOWUP = "STATUS_FOLLOWUP"
    REFERENCE_FOLLOWUP = "REFERENCE_FOLLOWUP"
    CORRECTION = "CORRECTION"


@dataclass(frozen=True)
class PlanStep:
    operation: str
    agent: str | None = None
    target: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "agent": self.agent,
            "target": self.target,
        }


@dataclass(frozen=True)
class ReferenceCandidate:
    type: str
    id: str | None
    score: float
    signals: tuple[str, ...]
    turn_distance: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "score": round(self.score, 3),
            "signals": list(self.signals),
            "turn_distance": self.turn_distance,
        }


@dataclass(frozen=True)
class ResolvedReference:
    type: str | None
    id: str | None
    confidence: float
    signals: tuple[str, ...] = ()
    ambiguous: bool = False
    candidates: tuple[ReferenceCandidate, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "confidence": round(self.confidence, 3),
            "signals": list(self.signals),
            "ambiguous": self.ambiguous,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class IntentFrame:
    speech_act: SpeechAct
    operation: str
    agent: str | None
    target: str | None
    polarity: str
    execution_requested: bool
    continuation: bool
    constraints: tuple[Constraint, ...]
    confidence: float
    followup_type: FollowupType
    plan: tuple[PlanStep, ...] = ()
    contradictory_constraints: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "speech_act": self.speech_act.value,
            "operation": self.operation,
            "agent": self.agent,
            "target": self.target,
            "polarity": self.polarity,
            "execution_requested": self.execution_requested,
            "continuation": self.continuation,
            "constraints": [value.value for value in self.constraints],
            "confidence": round(self.confidence, 3),
            "followup_type": self.followup_type.value,
            "plan": [step.as_dict() for step in self.plan],
            "contradictory_constraints": list(self.contradictory_constraints),
        }


_REFERENCE_WORDS = (
    " ele ", " ela ", " isso", " esse", " essa", " aquele", " aquela",
    " aquilo", " la ", " anterior", " ultimo", " ultima", " outro", " outra",
    " trabalho", " tarefa", " resposta", " arquivo", " projeto", " processo",
)

_STATUS_TERMS = (
    "terminou", "acabou", "concluiu", "ficou pronto", "ja saiu", "saiu aquilo",
    "ainda esta", "ainda ta", "ficou rodando", "esta rodando", "ta rodando",
    "estado atual", "status", "me atualiza", "esta vivo", "ta vivo",
    "como esta", "como ta",
    "andamento", "ja foi",
)

_HISTORY_TERMS = (
    "o que fez", "oq fez", "o que ele fez", "oq ele fez", "conclusoes",
    "codex fez", "deepseek falou", "deepseek respondeu", "consultor falou",
    "o que saiu", "ultima ressalva", "resposta anterior", "o que falou",
    "oq falou", "respondeu", "sessao", "historico", "ultimo trabalho",
    "execucao anterior", "concluiu ontem", "resultado anterior",
    "conversa", "sugeriu", "resposta dele",
    "ultimo resultado", "ultima conclusao", "ressalva",
    "ultimas informacoes", "ultimos dados",
)

_META_PREFIXES = (
    "como ", "por que ", "porque ", "o que acontece se", "oq acontece se",
    "seria possivel", "e possivel", "ele consegue", "ela consegue",
    "conseguiria", "qual ferramenta", "quando devo", "quero saber como",
    "me explica como", "explique como",
)

# Short interrogatives only signal meta-discussion when they open the sentence.
# "como funciona a delegacao" is meta; "peca ao deepseek para ver como esta o
# firebase" carries "como" inside the delegated content and stays a command.
_META_OPENERS = ("como ", "por que ", "porque ")
_META_ANYWHERE = tuple(value for value in _META_PREFIXES if value not in _META_OPENERS)
# "quero entender como se cancela" is meta; the interrogative belongs to an
# explanation request, not to the content of a task delegated to an agent.
_META_EXPLANATION = re.compile(
    r"\b(?:entender|compreender|saber|aprender|explic\w+|duvida|curiosidade)\b"
    r"(?:\s+\w+){0,3}\s+(?:como|por que|porque)\b"
)
_DELEGATION_VERBS = re.compile(
    r"\b(?:peca|pede|pergunte|pergunta|manda|mande|consulte|consulta|"
    r"deleg(?:a|ue|ar|ando)|encaminh(?:a|e|ar|ando)|acion(?:a|e|ar|ando))\b"
)

_ACTION_VERBS = (
    "manda", "mande", "peca", "pede", "pergunta", "pergunte", "consulta", "consulte",
    "cancela", "cancele", "para ", "pare ", "faz ", "faca ", "poe ",
    "aplica", "aplique", "corrige", "corrija", "implementa", "implemente",
    "roda", "rode", "adiciona", "adicione", "verifica", "verifique", "revisar",
    "abre", "abra", "leia", "ler ", "procura", "procure", "localiza",
    "mostra", "mostre", "veja", "revisa", "revise", "avisa", "avise", "fala", "fale",
    "delega", "delegue", "delegar", "encaminha", "encaminhe", "encaminhar",
    "aciona", "acione", "acionar",
)


def _contains(text: str, values: Iterable[str]) -> bool:
    padded = f" {text} "
    return any(value in text or value in padded for value in values)


def _meta_signal(text: str) -> bool:
    """Detect meta-discussion about capabilities, not content of a delegated task."""
    if text.lstrip().startswith(_META_OPENERS):
        return True
    if _META_EXPLANATION.search(text):
        return True
    return _contains(text, _META_ANYWHERE)


_NEGATED_AGENT_CLAUSE = re.compile(
    r"\b(?:sem|nao)\s+(?:\w+\s+){0,4}?(?:codex|deepseek|consultor)\b"
)


def _without_negated_clauses(text: str) -> str:
    """A verb inside 'sem/nao ... <agente>' is a prohibition, not an action request."""
    return _NEGATED_AGENT_CLAUSE.sub(" ", text)


def _negated_action(text: str, action: str, agent: str | None = None) -> bool:
    """Recognize only negation governing an action; complement 'se não' is ignored."""
    agent_part = rf"[^,;.]{{0,40}}{agent}" if agent else ""
    patterns = (
        rf"\bnao\s+(?:precisa\s+)?(?:de\s+)?{action}\w*{agent_part}",
        rf"\bsem\s+(?:{action}\w*\s+)?{agent or action}",
    )
    return any(re.search(pattern, text) for pattern in patterns)


class ConversationReferenceResolver:
    """Ranks recent entities by type compatibility and recency without side effects."""

    def resolve(self, text: str, context: Any) -> ResolvedReference:
        explicit_codex = "codex" in text
        explicit_deepseek = "deepseek" in text or "consultor" in text
        reference_signal = _contains(text, _REFERENCE_WORDS)
        status_signal = _contains(text, _STATUS_TERMS)
        if (
            re.fullmatch(r"e ai\??", text)
            and (
                getattr(context, "focused_agent", None) == "codex"
                or bool(getattr(context, "codex_running_jobs", 0))
            )
        ):
            status_signal = True
        history_signal = _contains(text, _HISTORY_TERMS)
        file_signal = bool(re.search(r"\b(?:abre|abra|mostra|mostre|leia|conteudo|arquivo)\b", text))
        project_signal = "projeto" in text or " la " in f" {text} "
        candidates: list[ReferenceCandidate] = []
        turn_index = int(getattr(context, "turn_index", 0) or 0)
        recent_entities = tuple(getattr(context, "recent_entities", ()) or ())

        def distance_for(kind: str, identifier: Any) -> int:
            for item in reversed(recent_entities):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == kind and str(item.get("id") or item.get("value")) == str(identifier):
                    return max(0, turn_index - int(item.get("turn_index", turn_index)))
            return 0

        def add(kind: str, identifier: Any, base: float, *signals: str, distance: int = 0) -> None:
            if not identifier:
                return
            score = max(0.0, base - min(max(distance, 0), 10) * 0.045)
            candidates.append(
                ReferenceCandidate(kind, str(identifier), min(score, 0.99), tuple(signals), distance)
            )

        job_id = getattr(context, "focused_job", None) or getattr(context, "codex_job_id", None)
        job_running = bool(getattr(context, "codex_running_jobs", 0))
        if job_id:
            score = 0.42
            signals = ["focused_job"]
            if status_signal:
                score += 0.40
                signals.append("status_verb_compatibility")
            if explicit_codex:
                score += 0.12
                signals.append("explicit_codex")
            if job_running:
                score += 0.10
                signals.append("running_state")
            add("codex_job", job_id, score, *signals, distance=distance_for("codex_job", job_id))

        session_id = getattr(context, "focused_session", None)
        focused_agent = getattr(context, "focused_agent", None)
        codex_session_id = session_id if focused_agent == "codex" else None
        if codex_session_id or focused_agent == "codex" or explicit_codex:
            score = 0.32
            signals = ["focused_codex_session" if focused_agent == "codex" else "codex_mention"]
            if history_signal:
                score += 0.44
                signals.append("history_verb_compatibility")
            if explicit_codex:
                score += 0.15
                signals.append("explicit_codex")
            codex_id = codex_session_id or "shared"
            add("codex_session", codex_id, score, *signals, distance=distance_for("codex_session", codex_id))

        deepseek_session = (
            session_id if focused_agent == "deepseek" else None
        ) or getattr(context, "deepseek_active_session", None)
        if deepseek_session or focused_agent == "deepseek" or explicit_deepseek:
            score = 0.34
            signals = ["focused_deepseek_session" if focused_agent == "deepseek" else "deepseek_mention"]
            if history_signal or "falou" in text or "resposta" in text or "ressalva" in text:
                score += 0.43
                signals.append("response_verb_compatibility")
            if explicit_deepseek:
                score += 0.15
                signals.append("explicit_deepseek")
            deepseek_id = deepseek_session or "active"
            add("deepseek_session", deepseek_id, score, *signals, distance=distance_for("deepseek_session", deepseek_id))

        focused_file = getattr(context, "focused_file", None)
        if focused_file:
            score = 0.40
            signals = ["focused_file"]
            if file_signal:
                score += 0.46
                signals.append("file_verb_compatibility")
            if "arquivo" in text:
                score += 0.15
                signals.append("explicit_file_type")
            add("file", focused_file, score, *signals, distance=distance_for("file", focused_file))

        focused_project = getattr(context, "focused_project", None) or getattr(context, "active_project", None)
        if focused_project and project_signal:
            add("project", focused_project, 0.70, "project_verb_compatibility")

        for raw in recent_entities:
            if not isinstance(raw, dict) or not raw.get("type"):
                continue
            kind = str(raw["type"])
            identifier = raw.get("id") or raw.get("value")
            distance = max(0, turn_index - int(raw.get("turn_index", turn_index)))
            compatibility = 0.34
            signals = ["recent_entity"]
            if status_signal and kind in {"codex_job", "generation", "task"}:
                compatibility += 0.43
                signals.append("status_verb_compatibility")
            if history_signal and kind in {"codex_session", "deepseek_session", "tool_result", "agent_response"}:
                compatibility += 0.40
                signals.append("history_verb_compatibility")
            if file_signal and kind == "file":
                compatibility += 0.45
                signals.append("file_verb_compatibility")
            add(kind, identifier, compatibility, *signals, distance=distance)

        # Keep the best representation of each entity.
        best: dict[tuple[str, str | None], ReferenceCandidate] = {}
        for candidate in candidates:
            key = (candidate.type, candidate.id)
            if key not in best or candidate.score > best[key].score:
                best[key] = candidate
        ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
        if not ranked:
            return ResolvedReference(None, None, 0.0)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        if any(value in text for value in ("outro", "outra", "anterior", "de antes")):
            alternatives = [candidate for candidate in ranked[1:] if candidate.type == top.type]
            if alternatives:
                selected = alternatives[0]
                return ResolvedReference(
                    selected.type,
                    selected.id,
                    max(0.70, selected.score),
                    (*selected.signals, "explicit_other_correction"),
                    False,
                    tuple(ranked[:5]),
                )
        ambiguous = bool(
            reference_signal
            and second
            and top.score >= 0.60
            and second.score >= 0.60
            and top.score - second.score < 0.08
            and not ("arquivo" in text and top.type == "file")
        )
        if not reference_signal and not (status_signal or history_signal or file_signal):
            return ResolvedReference(None, None, 0.0, candidates=tuple(ranked[:5]))
        return ResolvedReference(
            None if ambiguous else top.type,
            None if ambiguous else top.id,
            top.score,
            top.signals,
            ambiguous,
            tuple(ranked[:5]),
        )

    def resolve_typed(
        self,
        target_type: str,
        reference: str | None,
        context: Any,
    ) -> ResolvedReference:
        """Resolve an LLM-supplied semantic type to a concrete local entity."""
        if target_type == "none":
            return ResolvedReference(None, None, 1.0)
        if target_type == "url" and reference:
            return ResolvedReference(
                "url", reference, 1.0, ("semantic_url",)
            )
        cue = {
            "codex_job": "a tarefa terminou status",
            "codex_session": "o que o codex fez na sessao",
            "deepseek_session": "o que o deepseek falou na resposta",
            "agent_response": "o que ele falou na resposta",
            "file": "abre esse arquivo",
            "project": "esse projeto anterior",
            "tool_result": "resultado anterior",
            "task": "essa tarefa terminou",
            "generation": "essa geracao terminou",
        }.get(target_type, reference or "")
        resolved = self.resolve(f"{cue} {reference or ''}".strip(), context)
        candidates = list(resolved.candidates)
        compatible_types = {
            "agent_response": {"codex_session", "deepseek_session", "agent_response", "tool_result"},
            "task": {"codex_job", "task"},
            "generation": {"generation"},
        }.get(target_type, {target_type})
        compatible = [item for item in candidates if item.type in compatible_types]
        if reference in {"other_candidate", "previous_entity"} and len(compatible) > 1:
            selected = compatible[1]
        elif compatible:
            selected = compatible[0]
        else:
            return ResolvedReference(None, None, 0.0, ("typed_target_not_found",), False, tuple(candidates[:5]))
        ambiguous = bool(
            len(compatible) > 1
            and reference not in {"other_candidate", "previous_entity", "latest_codex_job", "focused_file", "active_deepseek_session", "shared_codex_session", "active_project"}
            and compatible[0].score - compatible[1].score < 0.08
        )
        return ResolvedReference(
            None if ambiguous else selected.type,
            None if ambiguous else selected.id,
            selected.score,
            (*selected.signals, "semantic_target_type"),
            ambiguous,
            tuple(candidates[:5]),
        )


class IntentFrameBuilder:
    """Builds a compact semantic representation; it does not select or run tools."""

    def __init__(self, resolver: ConversationReferenceResolver | None = None):
        self.resolver = resolver or ConversationReferenceResolver()

    def build(self, text: str, context: Any) -> tuple[IntentFrame, ResolvedReference]:
        reference = self.resolver.resolve(text, context)
        explicit_codex = "codex" in text
        explicit_deepseek = "deepseek" in text or "consultor" in text
        focused_agent = getattr(context, "focused_agent", None)
        agent = "deepseek" if explicit_deepseek else "codex" if explicit_codex else focused_agent
        status_signal = _contains(text, _STATUS_TERMS)
        if (
            re.fullmatch(r"e ai\??", text)
            and (
                focused_agent == "codex"
                or bool(getattr(context, "codex_running_jobs", 0))
            )
        ):
            status_signal = True
        history_signal = _contains(text, _HISTORY_TERMS)
        correction = bool(re.search(r"^(?:nao[,;]?\s+)?(?:o outro|a outra|nao era|quis dizer|eu (?:tava|estava) falando)", text))
        meta = _meta_signal(text)
        question = "?" in text or meta or bool(re.match(r"^(?:como|por que|porque|qual|quais|o que|oq|ele|ela|ja|ainda)\b", text))
        action_signal = _contains(_without_negated_clauses(text), _ACTION_VERBS)

        constraints: set[Constraint] = set()
        if _negated_action(text, r"(?:pergunt|consult|cham|us|deleg|encaminh|acion)", "deepseek") or re.search(
            r"\bsem\s+(?:consultar|usar|delegar|encaminhar|perguntar)\s+"
            r"(?:(?:ao?|pro|para)\s+)?(?:o\s+)?deepseek\b",
            text,
        ):
            constraints.add(Constraint.FORBID_DEEPSEEK)
        if _negated_action(text, r"(?:mand|encaminh|us|deleg|acion)", "codex") or re.search(
            r"\bsem\s+(?:mandar|usar|delegar|encaminhar)\s+"
            r"(?:(?:ao?|pro|para)\s+)?(?:o\s+)?codex\b",
            text,
        ):
            constraints.add(Constraint.FORBID_CODEX)
        if re.search(r"\bnao\s+(?:cancela|cancele|cancelar|para|pare)\b", text) or re.search(
            r"\bpara\s+(?:o\s+)?codex\?\s*nao\b", text
        ):
            constraints.add(Constraint.FORBID_CANCEL)
        if _contains(text, ("nao executa de novo", "nao execute de novo", "nao execute nada novamente", "nao cria outro turn", "nao crie outro turn", "nao manda outra tarefa", "sem nova tarefa", "sem criar outro turn", "sem criar uma tarefa nova")):
            constraints.add(Constraint.FORBID_NEW_TURN)
        if _contains(text, ("nao pergunta pra ninguem", "nao pergunta para ninguem", "sem consultar nenhum outro modelo", "responde por conta propria", "responda por conta propria", "responde voce", "responda voce", "quero sua opiniao", "a pergunta e para voce", "a pergunta e pra voce")):
            constraints.update({Constraint.ANSWER_SELF, Constraint.FORBID_DELEGATION})
            if "ninguem" in text or "nenhum outro modelo" in text:
                constraints.update({Constraint.FORBID_CODEX, Constraint.FORBID_DEEPSEEK})
        if Constraint.FORBID_DEEPSEEK in constraints and _contains(text, ("o que voce acha", "o que voce achou", "sua opiniao", "sua avaliacao")):
            constraints.update({Constraint.ANSWER_SELF, Constraint.FORBID_DELEGATION})
        if _contains(text, ("sem alterar", "sem modificar", "nao mexe", "nao mude")):
            constraints.add(Constraint.FORBID_MUTATION)
        if _contains(text, ("so leitura", "somente leitura")):
            constraints.update({Constraint.READ_ONLY, Constraint.FORBID_MUTATION})
        if _contains(text, ("sem esperar", "segundo plano", "em background")):
            constraints.add(Constraint.BACKGROUND)
        if _contains(text, ("espera o resultado", "aguarda o resultado", "aguarde o resultado")):
            constraints.add(Constraint.WAIT_FOR_RESULT)

        explicit_generation = bool(
            explicit_deepseek
            and (
                re.search(
                    r"\b(?:pergunta|pergunte|peca|pede|consulta|consulte|manda|mande|"
                    r"mostra|mostre|veja|ve|deleg(?:a|ue|ar|ando)|"
                    r"encaminh(?:a|e|ar|ando)|acion(?:a|e|ar|ando))\b",
                    text,
                )
                or "segunda opiniao" in text
            )
            and not _negated_action(text, r"(?:pergunt|consult|cham|us|deleg|encaminh|acion)", "deepseek")
            and Constraint.FORBID_DEEPSEEK not in constraints
            and not re.search(r"\ba pergunta\s+(?:e|era|foi)\b", text)
        )
        if explicit_generation and re.search(
            r"\b(?:pergunta|pergunte|peca|pede|consulta|consulte)\b.{0,45}\bdeepseek\b.{0,30}\bse\b",
            text,
        ):
            # Capability wording inside the delegated question is content, not
            # meta-discussion about whether to invoke the named consultant.
            meta = False

        if correction:
            speech_act = SpeechAct.CORRECTION
        elif explicit_generation and not meta:
            speech_act = SpeechAct.COMMAND
        elif status_signal:
            speech_act = SpeechAct.STATUS_QUERY
        elif history_signal or (reference.type in {"codex_session", "deepseek_session"} and question):
            speech_act = SpeechAct.REFERENCE_QUERY
        elif meta or (_contains(text, ("explica", "explique")) and question):
            speech_act = SpeechAct.EXPLANATION_REQUEST
        elif action_signal:
            speech_act = SpeechAct.COMMAND
        elif question:
            speech_act = SpeechAct.QUESTION
        else:
            speech_act = SpeechAct.REQUEST

        execution_requested = bool(action_signal and not meta)
        # Read/status/history are information retrieval, not side-effect requests.
        if speech_act in {SpeechAct.STATUS_QUERY, SpeechAct.REFERENCE_QUERY, SpeechAct.EXPLANATION_REQUEST}:
            execution_requested = False
        elif speech_act is SpeechAct.QUESTION and not action_signal:
            execution_requested = False
        if explicit_generation and not meta:
            execution_requested = True
        if _contains(text, ("quero cancelar", "quero que cancele", "pode cancelar")) and not meta:
            execution_requested = True

        operation = "answer"
        if correction:
            operation = "correct_reference"
        elif explicit_generation and not meta:
            operation = "delegate"
            agent = "deepseek"
        elif status_signal:
            operation = "status"
        elif meta:
            operation = "explain"
        elif history_signal:
            operation = "review"
        elif re.search(r"\b(?:cancela|cancele|cancelar|parar|pare)\b|^para\s+(?:o\s+codex|ele)\b|\bcodex\s+para$", text):
            operation = "explain_cancel" if not execution_requested else "cancel"
        elif focused_agent == "codex" and re.search(r"\b(?:fala|avisa|avise|manda).{0,30}\b(?:olha|olhar|inclui|incluir|warnings|uornings|nao mexer|evitar|evite)\b", text):
            operation = "steer"
            agent = "codex"
        elif focused_agent == "codex" and re.search(r"^(?:e\s+)?(?:os\s+)?(?:warnings|uornings|avisos|falhas)\??$", text):
            operation = "steer"
            agent = "codex"
            execution_requested = True
        elif (explicit_codex or focused_agent == "codex") and re.search(r"\b(?:cancela|cancele|manda).{0,18}\b(?:parar|pare)\b|^para\s+(?:o\s+codex|ele)\b|\bcodex\s+para$", text):
            operation = "cancel"
            agent = "codex"
        elif (explicit_codex or focused_agent == "codex") and re.search(r"\b(?:manda|mande|poe|faz|faca|aplica|aplique|corrige|corrija|implementa|implemente|roda|rode|execute|verifica|verifique|adiciona|adicione|revisar)\b", text):
            operation = "delegate"
            agent = "codex"
        elif (explicit_codex or focused_agent == "codex") and re.search(r"\b(?:fala|avisa|avise|olha|inclui|incluir|nao mexer)\b", text):
            operation = "steer"
            agent = "codex"
        elif re.search(r"\b(?:abre|abra|leia|ver o conteudo|mostra esse|mostre esse|mostra ela|mostra ele)\b", text) or (
            reference.type == "file" and re.search(r"\b(?:mostra|mostre|abre|abra)\b", text)
        ):
            operation = "read"
        elif re.search(r"\b(?:onde|localiza|procura|encontre|qual arquivo)\b", text):
            operation = "search"

        followup = FollowupType.NEW_REQUEST
        has_reference = _contains(text, _REFERENCE_WORDS)
        if correction:
            followup = FollowupType.CORRECTION
        elif status_signal and (has_reference or not explicit_codex):
            followup = FollowupType.STATUS_FOLLOWUP
        elif operation == "steer" and focused_agent == "codex":
            followup = FollowupType.MODIFICATION
        elif has_reference or (history_signal and not (explicit_codex or explicit_deepseek)):
            followup = FollowupType.REFERENCE_FOLLOWUP
        elif text.startswith(("e ", "agora ", "tambem ")):
            followup = FollowupType.CONTINUATION

        plan: list[PlanStep] = []
        if explicit_deepseek and explicit_codex:
            needs_codex_context = bool(
                re.search(r"(?:o que|oq|resultado|conclus|trabalho|sessao|turns?).{0,35}codex|codex.{0,35}(?:fez|conclu|resultado|turn)|ultima\s+solucao.{0,20}codex", text)
            )
            codex_after = bool(re.search(r"(?:depois|em seguida|entao).{0,45}codex|manda.{0,20}codex.{0,25}(?:implement|revis|corrig)", text))
            if needs_codex_context:
                plan.append(PlanStep("review", "codex", "last_result"))
            if operation == "delegate" and agent == "deepseek":
                plan.append(PlanStep("delegate", "deepseek", "request"))
            if codex_after and Constraint.FORBID_CODEX not in constraints:
                plan.append(PlanStep("delegate", "codex", "deepseek_result"))
            if len(plan) > 1:
                constraints.add(Constraint.ORDERED)

        positive_codex = bool(
            explicit_codex and agent == "codex" and execution_requested and operation in {"delegate", "cancel", "steer"}
        )
        positive_deepseek = bool(explicit_deepseek and execution_requested and operation == "delegate")
        contradictions: list[str] = []
        if positive_codex and Constraint.FORBID_CODEX in constraints:
            contradictions.append("CODEX requested and forbidden")
        if positive_deepseek and Constraint.FORBID_DEEPSEEK in constraints:
            contradictions.append("DEEPSEEK requested and forbidden")
        if operation == "cancel" and Constraint.FORBID_CANCEL in constraints:
            contradictions.append("CANCEL requested and forbidden")

        # A negated clause followed by a different positive instruction is not a contradiction.
        if Constraint.FORBID_CANCEL in constraints and status_signal:
            contradictions = [item for item in contradictions if not item.startswith("CANCEL")]
            operation = "status"
            speech_act = SpeechAct.STATUS_QUERY
            execution_requested = False
        if Constraint.ANSWER_SELF in constraints:
            operation = "answer"
            agent = "qwen"
            execution_requested = False

        confidence = 0.95 if explicit_codex or explicit_deepseek or constraints else 0.84
        if reference.ambiguous:
            confidence = min(confidence, 0.55)
        return (
            IntentFrame(
                speech_act=speech_act,
                operation=operation,
                agent=agent,
                target=reference.id,
                polarity="positive",
                execution_requested=execution_requested,
                continuation=followup is not FollowupType.NEW_REQUEST,
                constraints=tuple(sorted(constraints, key=lambda value: value.value)),
                confidence=confidence,
                followup_type=followup,
                plan=tuple(plan),
                contradictory_constraints=tuple(contradictions),
            ),
            reference,
        )
