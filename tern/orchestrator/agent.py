from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.parse
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable

from .client import LlamaClient, ServerError
from .config import Settings
from .decision_policy import (
    AgentDecisionPolicy,
    constraint_violation_for_tool,
    tool_catalog_audit,
    tool_specs_for_decision,
)
from .decision_observability import (
    AgentDecisionObserver,
    DecisionTiming,
    estimate_tokens,
)
from .explicit_agent_binding import (
    availability_for_requested_agent,
    detect_explicit_agent_binding,
)
from .prompt import SYSTEM_PROMPT
from .projects import normalize_technical_transcript
from .semantic_pass import QwenSemanticInterpreter
from .tool_progress import ToolProgressTracker
from .tools import ToolRegistry


_SINGLE_CALL_TOOLS = frozenset(
    {
        "delegate_to_codex",
        "review_codex_session",
        "get_codex_job_status",
        "cancel_codex_job",
        "steer_codex_job",
        "delegate_to_deepseek",
        "review_deepseek_session",
        "get_hardware_telemetry",
    }
)


_WEB_THREAT_EXPLANATIONS = {
    "ACTIVE_LOCAL_NETWORK_REFERENCE": (
        "a página tentou carregar ou contatar recurso de rede local/reservada"
    ),
    "ACTIVE_FILE_SCHEME_REFERENCE": (
        "a página tentou acessar recurso local pelo protocolo file:"
    ),
    "KNOWN_MALICIOUS_HOST": (
        "o host já foi bloqueado por uma detecção determinística anterior"
    ),
}


def _display_web_target(value: object) -> str:
    """Show a useful destination without credentials, query data, or fragments."""
    raw = str(value or "")
    try:
        parsed = urllib.parse.urlsplit(raw)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urllib.parse.urlunsplit(
            (parsed.scheme, f"{host}{port}", parsed.path, "", "")
        )
    except ValueError:
        return "destino solicitado"


def _web_safety_report(
    tool: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    """Create a deterministic user report when browser launch is denied."""
    if tool != "web_open_browser" or result.get("error") != "web_access_denied":
        return None
    details = result.get("details")
    codes = details.get("codes", []) if isinstance(details, dict) else []
    explanations = [
        _WEB_THREAT_EXPLANATIONS[str(code)]
        for code in codes
        if str(code) in _WEB_THREAT_EXPLANATIONS
    ]
    reason = "; ".join(explanations) or str(
        result.get("message") or "destino reprovado pela política de segurança"
    )
    return (
        "Não abri o site.\n\n"
        "Relatório de segurança:\n"
        f"- Destino: `{_display_web_target(arguments.get('url'))}`\n"
        "- Resultado: bloqueado antes da abertura da guia.\n"
        f"- Motivo: {reason}.\n"
        "- Controles: protocolo, domínio, DNS/IP, redirecionamentos e "
        "ameaças ativas.\n"
        "- Guia aberta: não."
    )


def _condition_satisfied(condition: str | None, result: dict[str, Any]) -> bool | None:
    """Accept only an explicit, typed result for a conditional plan step."""
    if condition is None:
        return True
    if condition == "positive_recommendation":
        value = result.get("positive_recommendation")
        return value if isinstance(value, bool) else None
    return None


def _normalized(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _deepseek_intent(value: str) -> str | None:
    normalized = _normalized(value)
    if "deepseek" not in normalized:
        return None
    review = re.search(
        r"\b(?:historico|sessao|ultim(?:a|as|o|os)|o que .* (?:falou|respondeu|sugeriu)|"
        r"leia .* sessao|resuma .* turnos?)\b",
        normalized,
    )
    delegation = re.search(
        r"\b(?:pergunt(?:a|e|ar)|peca|mande|mostre|envie|segunda opiniao|"
        r"o que .* acha|revis(?:ar|e)|critic(?:ar|a|e)|consult(?:ar|e))\b",
        normalized,
    )
    if review and not delegation:
        return "review_deepseek_session"
    return "delegate_to_deepseek"


def _is_codex_history_request(value: str) -> bool:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    if "codex" not in normalized:
        return False
    if re.search(
        r"\b(?:peca|mande|solicite|instrua)\s+(?:ao\s+)?codex\b|"
        r"\buse\s+(?:o\s+)?codex\s+para\b|"
        r"\bcodex\s+(?:faca|execute|revise|analise|corrija|implemente)\b",
        normalized,
    ):
        return False
    explicit_history = re.search(
        r"\b(?:historico|sessao|ultim(?:a|as|o|os)|"
        r"o que aconteceu|o que .* fez|informacoes|turns?)\b",
        normalized,
    )
    inspection = re.search(
        r"\b(?:olhada|revis(?:ao|ar|e)|vistori(?:a|ar|e)|"
        r"inspec(?:ao|ionar)|acompanhar|status)\b",
        normalized,
    )
    existing_work = re.search(
        r"\b(?:tarefa|trabalho|execucao|atividade|resultado|turn|sessao)\b",
        normalized,
    )
    return bool(explicit_history or (inspection and existing_work))


def _codex_job_intent(value: str) -> str | None:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    if "codex" not in normalized and not re.search(r"\b(?:ele|tarefa|delegacao)\b", normalized):
        return None
    if re.search(r"\b(?:cancele|cancelar|pare|interrompa)\b", normalized):
        return "cancel_codex_job"
    if re.search(r"\b(?:avise|diga|instrua|direcione)\s+(?:ao\s+)?codex\b", normalized):
        return "steer_codex_job"
    if re.search(
        r"\b(?:ja terminou|ainda esta trabalhando|como esta a tarefa|"
        r"estado da ultima delegacao|status da tarefa)\b",
        normalized,
    ):
        return "get_codex_job_status"
    return None


def _project_lookup_only(value: str) -> bool:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    if re.search(r"\b(?:leia|abra|mostre o conteudo|corrija|edite|modifique|implemente)\b", normalized):
        return False
    return bool(
        re.search(
            r"\b(?:onde fica|localize|localizar|procure|procurar|encontre|"
            r"qual arquivo|arquivo da|arquivo de)\b",
            normalized,
        )
    )


def _hardware_answer(result: dict[str, Any]) -> str:
    parts: list[str] = []
    if result.get("cpu_temperature_available"):
        parts.append(f"Temperatura da CPU: {float(result['cpu_temperature_c']):.1f} °C.")
    else:
        parts.append(
            "Temperatura da CPU: indisponível. O Windows não expôs um sensor "
            "compatível; nenhum valor foi simulado."
        )
    if result.get("usb_available"):
        parts.append(f"Dispositivos USB conectados: {int(result['usb_devices'])}.")
    else:
        parts.append("Contagem de dispositivos USB: indisponível.")
    return "\n".join(parts)


def _codex_status_answer(result: dict[str, Any]) -> str:
    """Traduz estado interno do Codex sem vazar IDs ou metadados técnicos."""
    if not result.get("ok"):
        if result.get("error") == "codex_job_not_found":
            return "Nenhuma tarefa do Codex foi encontrada nesta sessão."
        return "Não consegui consultar a sessão do Codex agora."

    status = str(result.get("status") or "").casefold()
    task = str(result.get("task_summary") or "").strip()
    if status in {"queued", "starting", "running", "reconnecting"}:
        answer = "O Codex está trabalhando na tarefa atual."
    elif status == "completed":
        answer = "A última tarefa do Codex foi concluída. A sessão está pronta para novas instruções."
    elif status in {"cancelled", "canceled", "interrupted"}:
        answer = "A última tarefa do Codex foi interrompida. A sessão está pronta para novas instruções."
    elif status in {"failed", "error", "timed_out", "timeout"}:
        answer = "A última tarefa do Codex falhou. A sessão está pronta para uma nova tentativa."
    else:
        answer = "A sessão do Codex está disponível."
    if task and status in {"queued", "starting", "running", "reconnecting"}:
        answer += f" Tarefa: {task}"
    return answer


class Supervisor:
    def __init__(self, settings: Settings, client: LlamaClient, tools: ToolRegistry):
        self.settings = settings
        self.client = client
        self.tools = tools
        self.decision_policy = AgentDecisionPolicy(
            tools=tools,
            context_cache_enabled=settings.agent_decision_context_cache,
        )
        self.decision_observer = AgentDecisionObserver(
            settings.state_dir,
            enabled=settings.agent_decision_shadow,
        )
        self.semantic_interpreter = QwenSemanticInterpreter(client)
        self.conversation_id = f"jarvis-{uuid.uuid4()}"

    def run(
        self,
        user_text: str,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        trace_id = f"pipeline-{uuid.uuid4()}"
        pipeline_trace: list[dict[str, Any]] = []

        def trace(stage: str, values: dict[str, Any] | None = None, **extra: Any) -> None:
            safe_values = {
                key: value
                for key, value in {**(values or {}), **extra}.items()
                if key
                in {
                    "result",
                    "reason_code",
                    "tool_name",
                    "normalized_host",
                    "decision_type",
                    "registered",
                    "exposed",
                    "web_enabled",
                    "ambiguity_present",
                    "requested_agent",
                    "requested_agent_source",
                    "tool_available",
                    "execution_allowed",
                }
            }
            record = {"stage": stage, **safe_values}
            pipeline_trace.append(record)
            self.tools.logger.write_event(
                "agent_pipeline_trace",
                trace_id=trace_id,
                **record,
            )

        def emit(event: str, **values: Any) -> None:
            if event_callback is not None:
                event_callback(event, values)

        timing = DecisionTiming()
        timing.mark("input_received")
        emit("input_received")
        trace("USER_INPUT", result="received")
        set_web_trace = getattr(self.tools.web, "set_trace_callback", None)
        if callable(set_web_trace):
            set_web_trace(trace)
        original_user_text = user_text
        routing_text = normalize_technical_transcript(user_text)
        explicit_agent_binding = detect_explicit_agent_binding(routing_text)
        trace(
            "EXPLICIT_AGENT_BINDING",
            result="bound" if explicit_agent_binding else "not_bound",
            requested_agent=(
                explicit_agent_binding.requested_agent
                if explicit_agent_binding
                else None
            ),
            requested_agent_source=(
                explicit_agent_binding.requested_agent_source
                if explicit_agent_binding
                else None
            ),
        )
        begin_research = getattr(self.tools.web, "begin_research", None)
        if callable(begin_research):
            begin_research(routing_text)

        project_context = self.tools.projects.context_text()
        decision_context = self.decision_policy.build_context()
        timing.mark("decision_context_ready")
        emit("decision_context_ready")
        semantic_result = QwenSemanticInterpreter.skipped()
        semantic_enabled = (
            self.settings.agent_decision_semantic_first
            and explicit_agent_binding is None
            and (
                isinstance(self.client, LlamaClient)
                or bool(getattr(self.client, "supports_structured_output", False))
            )
            and self.semantic_interpreter.needs_semantic_pass(
                routing_text,
                decision_context,
            )
        )
        if semantic_enabled:
            timing.mark("semantic_request_started")
            semantic_result = self.semantic_interpreter.interpret(
                original_user_text,
                routing_text,
                decision_context,
            )
            timing.mark("semantic_request_completed")
            emit(
                "semantic_decision_ready",
                valid=semantic_result.parse_valid,
                repair_used=semantic_result.repair_used,
                latency_ms=semantic_result.latency_ms,
            )
        semantic_decision = semantic_result.decision
        trace(
            "SEMANTIC_DECISION",
            result=(
                "valid"
                if semantic_result.parse_valid and semantic_decision is not None
                else "skipped"
                if not semantic_result.used
                else "invalid"
            ),
            reason_code=semantic_result.error,
            decision_type=(
                semantic_decision.primary_intent.value
                if semantic_decision is not None
                else None
            ),
            ambiguity_present=(
                semantic_decision.ambiguity_present
                if semantic_decision is not None
                else None
            ),
        )
        decision = self.decision_policy.decide(
            original_user_text,
            context=decision_context,
            semantic_decision=semantic_result.decision,
            explicit_agent_binding=explicit_agent_binding,
        )
        if semantic_result.used and not semantic_result.parse_valid:
            decision = self.decision_policy.safe_fallback_decision(decision)
        all_tool_specs = self.tools.specs()
        registered_tool_names = {
            str(item.get("function", {}).get("name") or "")
            for item in all_tool_specs
        }
        if explicit_agent_binding is not None:
            availability = availability_for_requested_agent(
                explicit_agent_binding,
                decision_context,
                registered_tool_names,
            )
            decision = replace(
                decision,
                tool_registered=availability.tool_registered,
                tool_available=availability.tool_available,
                execution_allowed=(
                    availability.execution_allowed
                    and decision.constraint_violation is None
                ),
                availability_reason=availability.reason,
            )
            trace(
                "RUNTIME_AVAILABILITY",
                result=("available" if availability.tool_available else "unavailable"),
                tool_name=availability.tool,
                requested_agent=explicit_agent_binding.requested_agent,
                tool_available=availability.tool_available,
                execution_allowed=decision.execution_allowed,
                reason_code=availability.reason,
            )
        trace(
            "DECISION_VALIDATION",
            result="blocked" if decision.constraint_violation else "valid",
            reason_code=decision.constraint_violation,
            decision_type=decision.intent.value,
        )
        trace(
            "ROUTE_SELECTED",
            result="selected",
            reason_code=decision.reason_code,
            decision_type=decision.intent.value,
        )
        trace(
            "TOOL_SELECTED",
            result="selected" if decision.selected_action else "none",
            tool_name=decision.selected_action,
            reason_code=(
                None if decision.selected_action else "ROUTE_HAS_NO_TOOL"
            ),
        )
        timing.mark("decision_ready")
        self.decision_policy.record_decision(decision, original_user_text)
        reusable_context = self.decision_policy.reusable_context_text()

        trusted_runtime_context = (
            "\nContexto confiavel do runtime:\n"
            f"- working_directory: {os.getcwd()}\n"
            f"- backend: {self.settings.backend.name}\n"
            f"- local_datetime: {datetime.now().astimezone().isoformat()}\n"
            f"- allowed_roots: {', '.join(str(root) for root in self.settings.allowed_roots)}\n"
            f"- max_tool_calls: {self.settings.max_tool_calls}\n"
            f"- max_attempts: {self.settings.max_attempts}\n"
            "\nProject context:\n"
            f"{project_context}\n"
            "\n"
            f"{decision_context.prompt_text()}\n"
            "\nDecision recommendation (policy scaffolding; choose arguments yourself):\n"
            + json.dumps(decision.as_dict(), ensure_ascii=False)
            + "\n"
            + (f"\nReusable conversation context:\n{reusable_context}\n" if reusable_context else "")
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + trusted_runtime_context},
            {"role": "user", "content": routing_text},
        ]
        timing.mark("prompt_ready")
        emit("prompt_ready")
        completed_claims: list[dict[str, Any]] = []
        claim_results = getattr(self.tools.codex, "claim_completed_results", None)
        if callable(claim_results):
            try:
                completed_claims = claim_results()
            except Exception as exc:
                self.tools.logger.write_event(
                    "codex_result_queue_failed",
                    state="failed",
                    error=str(exc),
                )
        for job in completed_claims:
            stored = job.get("result") if isinstance(job.get("result"), dict) else {}
            event = {
                "event": "codex_job_completed",
                "job_id": job.get("job_id"),
                "thread_id": job.get("thread_id"),
                "turn_id": job.get("turn_id"),
                "task": job.get("task_summary"),
                "status": job.get("status"),
                "result": stored.get("final_response") or stored.get("error"),
                "human_interventions": len(job.get("human_interventions") or []),
                "completed_at": job.get("completed_at"),
                "notify": not bool(job.get("completion_notified")),
            }
            if event["notify"]:
                job_store = getattr(self.tools.codex, "jobs", None)
                if job_store is not None:
                    event["notify"] = bool(
                        job_store.claim_notification(str(job.get("job_id")), "completed")
                    )
            messages[0]["content"] += (
                "\n\nResultado assíncrono confirmado do Codex:\n"
                + json.dumps(event, ensure_ascii=False)
            )
            emit(
                "codex_result_received",
                **{key: value for key, value in event.items() if key != "event"},
            )
            self.decision_policy.record_tool_result(
                "get_codex_job_status",
                {},
                {
                    "ok": True,
                    "job_id": job.get("job_id"),
                    "thread_id": job.get("thread_id"),
                    "turn_id": job.get("turn_id"),
                    "status": job.get("status"),
                },
            )
        agent_turn_id = f"qwen-{uuid.uuid4()}"
        progress = ToolProgressTracker()
        tool_call_counts: dict[str, int] = {}
        calls = 0
        failures = 0
        web_used = False
        web_sources: list[dict[str, str]] = []
        actual_tools: list[str] = []
        observed_outcome: str | None = None
        tools_disabled = False
        tool_specs = tool_specs_for_decision(all_tool_specs, decision)
        exposed_tool_names = {
            str(item.get("function", {}).get("name") or "")
            for item in tool_specs
        }
        web_enabled = bool(
            getattr(getattr(self.tools.web, "config", None), "enabled", True)
        )
        trace(
            "TOOL_AVAILABILITY",
            result=(
                "exposed"
                if decision.selected_action in exposed_tool_names
                else "not_exposed"
                if decision.selected_action
                else "not_selected"
            ),
            tool_name=decision.selected_action,
            registered=(
                decision.selected_action in registered_tool_names
                if decision.selected_action
                else "web_open" in registered_tool_names
            ),
            exposed=(
                decision.selected_action in exposed_tool_names
                if decision.selected_action
                else "web_open" in exposed_tool_names
            ),
        )
        trace(
            "CAPABILITY_CHECK",
            result=(
                "allowed"
                if decision.selected_action in {"web_open", "web_open_browser", "web_extract", "web_search"}
                and web_enabled
                else "blocked"
                if decision.selected_action in {"web_open", "web_open_browser", "web_extract", "web_search"}
                else "not_reached"
            ),
            reason_code=(
                "WEB_DISABLED"
                if decision.selected_action
                in {"web_open", "web_open_browser", "web_extract", "web_search"}
                and not web_enabled
                else None
            ),
            tool_name=decision.selected_action,
            web_enabled=web_enabled,
        )
        semantic_plan = (
            tuple(semantic_result.decision.compound_plan)
            if semantic_result.decision is not None
            else ()
        )
        semantic_plan_index = 0
        if semantic_plan and decision.tools:
            first_tool = decision.tools[0]
            tool_specs = [
                item
                for item in tool_specs
                if item.get("function", {}).get("name") == first_tool
            ]
        decision_tool_budget = (
            decision.max_tool_calls
            if decision.confidence >= 0.80
            else self.settings.max_tool_calls
        )
        prompt_sizes = {
            "estimated": True,
            "system_prompt_tokens": estimate_tokens(SYSTEM_PROMPT),
            "runtime_context_tokens": estimate_tokens(trusted_runtime_context),
            "tool_schema_tokens": estimate_tokens(tool_specs),
            "decision_context_tokens": estimate_tokens(decision_context.prompt_text()),
            "conversation_tokens": estimate_tokens(routing_text + reusable_context),
            "total_input_tokens": estimate_tokens(messages) + estimate_tokens(tool_specs),
            "tool_schemas_exposed": len(tool_specs),
            "tool_schemas_total": len(self.tools.specs()),
            "semantic_prompt_tokens": estimate_tokens(
                getattr(self.semantic_interpreter, "system_prompt", "")
            ),
        }
        catalog_audit = tool_catalog_audit(self.tools.specs(), decision)
        decision_id = self.decision_observer.begin(
            original_input=original_user_text,
            normalized_input=routing_text,
            decision=decision,
            context=decision_context,
            prompt_sizes=prompt_sizes,
        )

        def finish_observation(outcome: str) -> dict[str, Any]:
            timing.mark("response_ready")
            values = timing.as_dict()
            self.decision_observer.complete(
                decision_id,
                original_input=original_user_text,
                normalized_input=routing_text,
                decision=decision,
                context=decision_context,
                prompt_sizes=prompt_sizes,
                timing=values,
                tool_calls=calls,
                actual_tools=actual_tools,
                outcome=outcome,
                semantic_result=semantic_result,
                tool_catalog=catalog_audit,
            )
            emit("agent_timing", **values)
            return values

        def completed(
            answer: str,
            usage: dict[str, Any] | None,
            *,
            response_stage: str = "MODEL_RESPONSE",
        ) -> dict[str, Any]:
            trace(
                response_stage,
                result="completed",
                decision_type=decision.intent.value,
            )
            research_status = getattr(
                self.tools.web, "research_status", lambda: {}
            )()
            research_intent = research_status.get("intent") or {}
            if (
                web_used
                and research_intent.get("intent") == "news"
                and not web_sources
            ):
                answer = (
                    "Não encontrei fontes suficientemente relevantes para "
                    "responder com segurança."
                )
            if web_sources and not any(
                source["url"] in answer for source in web_sources
            ):
                citations = "\n".join(
                    f"- [{source['title']}]({source['url']})"
                    for source in web_sources
                )
                answer = f"{answer.rstrip()}\n\nFontes consultadas:\n{citations}"
            acknowledge = getattr(self.tools.codex, "acknowledge_result", None)
            if callable(acknowledge):
                for job in completed_claims:
                    acknowledge(
                        str(job.get("job_id")),
                        str(job.get("delivery_token")),
                    )
            self.decision_policy.record_answer(answer)
            if observed_outcome is not None:
                outcome = observed_outcome
            elif any(name.startswith("delegate_to_") for name in actual_tools):
                outcome = "delegated"
            elif decision.intent.value == "CLARIFY":
                outcome = "clarified"
            elif not actual_tools:
                outcome = "direct_answer"
            else:
                outcome = "success"
            timing_values = finish_observation(outcome)
            return {
                "ok": True,
                "answer": answer,
                "tool_calls": calls,
                "usage": usage,
                "decision": {
                    "intent": decision.intent.value,
                    "confidence": decision.confidence,
                    "reason_code": decision.reason_code,
                    "requested_agent": decision.requested_agent,
                    "requested_agent_source": decision.requested_agent_source,
                    "tool_registered": decision.tool_registered,
                    "tool_available": decision.tool_available,
                    "execution_allowed": decision.execution_allowed,
                    "availability_reason": decision.availability_reason,
                    "semantic_pass": semantic_result.as_dict(),
                    "fast_path": bool(
                        self.settings.agent_decision_fast_path
                        and self.decision_policy.fast_path(
                            decision, decision_context, original_user_text
                        )
                    ),
                },
                "timing": timing_values,
                "prompt_sizes": prompt_sizes,
                "web": {
                    "used": web_used,
                    "sources": web_sources,
                },
                "trace_id": trace_id,
                "pipeline_trace": list(pipeline_trace),
            }

        if (
            decision.requested_agent_source == "explicit_user"
            and decision.execution_allowed is False
        ):
            observed_outcome = "requested_agent_unavailable"
            label = "DeepSeek" if decision.requested_agent == "deepseek" else "Codex"
            if (
                decision.requested_agent == "deepseek"
                and decision.availability_reason == "agent_not_configured"
            ):
                unavailable = (
                    "DeepSeek está integrado, mas falta configurar "
                    "DEEPSEEK_API_KEY no arquivo .env do projeto. "
                    "Nenhum outro modelo foi usado como substituto."
                )
            elif (
                decision.requested_agent == "deepseek"
                and decision.availability_reason == "agent_disabled"
            ):
                unavailable = (
                    "DeepSeek está desativado. Defina DEEPSEEK_ENABLED=true "
                    "no arquivo .env. Nenhum outro modelo foi usado como substituto."
                )
            else:
                unavailable = (
                    f"A solicitação continua vinculada ao {label}, mas o agente "
                    "não está operacionalmente disponível neste runtime. "
                    "Nenhum outro modelo foi usado como substituto."
                )
            return completed(
                unavailable,
                None,
                response_stage="RUNTIME_UNAVAILABLE",
            )

        fast_path = (
            self.decision_policy.fast_path(
                decision,
                decision_context,
                original_user_text,
            )
            if self.settings.agent_decision_fast_path
            else None
        )
        if fast_path is not None:
            fast_call_id = f"fast-{uuid.uuid4()}"
            timing.mark("decision_detected")
            timing.mark("tool_call_detected")
            emit(
                "tool_call_detected",
                name=fast_path.tool,
                fast_path=True,
            )
            self.tools.logger.write_event(
                "agent_decision_fast_path",
                agent_turn_id=agent_turn_id,
                tool=fast_path.tool,
                arguments=fast_path.arguments,
                reason_code=fast_path.reason_code,
                side_effect=fast_path.side_effect.value,
                state="dispatching",
            )
            synthetic_call = {
                "id": fast_call_id,
                "type": "function",
                "function": {
                    "name": fast_path.tool,
                    "arguments": json.dumps(
                        fast_path.arguments,
                        ensure_ascii=False,
                    ),
                },
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [synthetic_call],
                }
            )
            emit("tool_start", name=fast_path.tool, fast_path=True)
            timing.mark("tool_started")
            tool_started = time.perf_counter()
            fast_result = self.tools.execute(
                fast_path.tool,
                fast_path.arguments,
                context={
                    "turn_id": agent_turn_id,
                    "user_text": routing_text,
                    "original_user_text": original_user_text,
                    "request_fingerprint": progress.fingerprint(
                        fast_path.tool,
                        fast_path.arguments,
                    ),
                    "fast_path": True,
                    "conversation_id": self.conversation_id,
                    "focused_codex_thread_id": (
                        decision_context.focused_session
                        if decision_context.focused_agent == "codex"
                        else None
                    ),
                },
                event_callback=lambda event, values: emit(event, **values),
            )
            elapsed_tool = (time.perf_counter() - tool_started) * 1000
            timing.tool_execution_ms += elapsed_tool
            timing.mark("tool_completed")
            emit(
                "tool_end",
                name=fast_path.tool,
                ok=fast_result.get("ok"),
                fast_path=True,
            )
            actual_tools.append(fast_path.tool)
            calls = 1
            if fast_path.tool.startswith("web_"):
                web_used = True
            if (
                fast_path.tool in {"web_open", "web_open_browser", "web_extract"}
                and fast_result.get("ok")
                and isinstance(fast_result.get("citation"), dict)
            ):
                citation = fast_result["citation"]
                title = citation.get("title")
                url = citation.get("url")
                if isinstance(title, str) and isinstance(url, str):
                    web_sources.append({"title": title, "url": url})
            progress.record(fast_path.tool, fast_path.arguments, fast_result)
            self.decision_policy.record_tool_result(
                fast_path.tool,
                fast_path.arguments,
                fast_result,
            )
            if fast_result.get("error") in {"ActionCancelled", "AccessDenied"}:
                observed_outcome = "user_cancelled"
            elif not fast_result.get("ok"):
                observed_outcome = "tool_error"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": fast_call_id,
                    "name": fast_path.tool,
                    "content": json.dumps(fast_result, ensure_ascii=False),
                }
            )
            tools_disabled = True
            safety_report = _web_safety_report(
                fast_path.tool,
                fast_path.arguments,
                fast_result,
            )
            if safety_report is not None:
                return completed(
                    safety_report,
                    None,
                    response_stage="WEB_SAFETY_REPORT",
                )
            if fast_path.reason_code == "explicit_agent_direct_handoff":
                if not fast_result.get("ok"):
                    detail = str(
                        fast_result.get("message")
                        or fast_result.get("error")
                        or "falha desconhecida"
                    )
                    answer = (
                        "Não foi possível enviar a tarefa diretamente ao agente "
                        f"solicitado: {detail}"
                    )
                elif fast_path.tool == "delegate_to_codex":
                    answer = str(fast_result.get("final_response") or "").strip()
                    if not answer:
                        status = str(fast_result.get("status") or "iniciada")
                        job_id = str(fast_result.get("job_id") or "").strip()
                        answer = f"Tarefa enviada diretamente ao Codex ({status})."
                        if job_id:
                            answer += f" Job: {job_id}."
                else:
                    answer = str(fast_result.get("response") or "").strip()
                    if not answer:
                        answer = "Tarefa enviada diretamente ao DeepSeek."
                return completed(
                    answer,
                    fast_result.get("usage") if isinstance(fast_result.get("usage"), dict) else None,
                    response_stage=(
                        "EXPLICIT_AGENT_DIRECT_HANDOFF"
                        if fast_result.get("ok")
                        else "EXPLICIT_AGENT_HANDOFF_FAILED"
                    ),
                )
            if fast_path.tool == "get_hardware_telemetry":
                return completed(
                    _hardware_answer(fast_result),
                    None,
                    response_stage="HARDWARE_TELEMETRY",
                )
            if fast_path.tool == "get_codex_job_status":
                return completed(
                    _codex_status_answer(fast_result),
                    None,
                    response_stage="CODEX_STATUS",
                )

        while calls < self.settings.max_tool_calls:
            emit("thinking")
            if tools_disabled:
                timing.mark("final_response_started")
                emit("final_response_started")
            timing.mark("qwen_request_started")
            emit("qwen_request_started")
            qwen_started = time.perf_counter()
            try:
                response = self.client.chat(
                    messages,
                    tools=(
                        None
                        if tools_disabled
                        or not tool_specs
                        or calls >= decision_tool_budget
                        else tool_specs
                    ),
                )
            except ServerError:
                timing.qwen_request_ms += (time.perf_counter() - qwen_started) * 1000
                timing.qwen_requests += 1
                failures += 1
                if failures >= self.settings.max_attempts:
                    finish_observation("tool_error")
                    raise
                continue
            timing.qwen_request_ms += (time.perf_counter() - qwen_started) * 1000
            timing.qwen_requests += 1
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if "decision_detected" not in timing.marks:
                timing.mark("decision_detected")
            if tool_calls and "tool_call_detected" not in timing.marks:
                timing.mark("tool_call_detected")
                emit(
                    "tool_call_detected",
                    name=tool_calls[0].get("function", {}).get("name"),
                    fast_path=False,
                )
            self.tools.logger.write_event(
                "qwen_decision",
                agent_turn_id=agent_turn_id,
                delegated=bool(tool_calls),
                tool_names=[
                    call.get("function", {}).get("name", "")
                    for call in tool_calls
                    if isinstance(call, dict)
                ],
                content=message.get("content"),
                prompt=original_user_text,
                routing_text=routing_text,
                state="tool_call" if tool_calls else "final_answer",
            )
            if not tool_calls:
                if "final_response_started" not in timing.marks:
                    timing.marks["final_response_started"] = timing.marks.get(
                        "qwen_request_started",
                        time.perf_counter(),
                    )
                return completed(
                    message.get("content", ""), response.get("usage")
                )
            if tools_disabled:
                self.tools.logger.write_event(
                    "tool_loop_blocked",
                    agent_turn_id=agent_turn_id,
                    tool_names=[
                        call.get("function", {}).get("name", "")
                        for call in tool_calls
                        if isinstance(call, dict)
                    ],
                    reason="tools_disabled",
                    state="blocked",
                )
                timing_values = finish_observation("loop_prevented")
                return {
                    "ok": False,
                    "error": "tool_loop_blocked",
                    "message": "nova chamada de ferramenta bloqueada apos encerramento do fluxo",
                    "tool_calls": calls,
                    "timing": timing_values,
                }
            messages.append(message)
            for call in tool_calls:
                if calls >= self.settings.max_tool_calls:
                    break
                function = call.get("function", {})
                name = function.get("name", "")
                raw_arguments = function.get("arguments", {})
                if tools_disabled:
                    result = {
                        "ok": False,
                        "error": "tools_disabled",
                        "message": "chamada ignorada porque o fluxo de ferramentas foi encerrado",
                    }
                    self.tools.logger.write_event(
                        "tool_call_rejected",
                        agent_turn_id=agent_turn_id,
                        tool=name,
                        state="tools_disabled",
                        error="tools_disabled",
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"blocked_{calls}"),
                            "name": name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    continue
                self.tools.logger.write_event(
                    "tool_call_generated",
                    agent_turn_id=agent_turn_id,
                    tool=name,
                    raw_arguments=raw_arguments,
                    prompt=original_user_text,
                    routing_text=routing_text,
                    state="generated",
                )
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("argumentos devem ser objeto JSON")
                except (json.JSONDecodeError, ValueError) as exc:
                    result = {"ok": False, "error": "invalid_json", "message": str(exc)}
                    arguments = {"_raw": str(raw_arguments)[:4096]}
                    self.tools.logger.write_event(
                        "tool_call_rejected",
                        agent_turn_id=agent_turn_id,
                        tool=name,
                        arguments=arguments,
                        prompt=user_text,
                        state="invalid_json",
                        error=str(exc),
                    )
                    observed_outcome = "tool_error"
                    tools_disabled = True
                else:
                    fingerprint = progress.fingerprint(name, arguments)
                    constraint_violation = constraint_violation_for_tool(
                        name,
                        decision.intent_frame,
                    )
                    if constraint_violation:
                        self.tools.logger.write_event(
                            "tool_call_rejected",
                            agent_turn_id=agent_turn_id,
                            tool=name,
                            reason="semantic_constraint_violation",
                            constraint=constraint_violation,
                            state="blocked",
                        )
                        result = {
                            "ok": False,
                            "error": "decision_constraint_violation",
                            "message": (
                                "Selected action violates explicit user constraint: "
                                f"{constraint_violation}. Choose another valid action."
                            ),
                        }
                        observed_outcome = "clarified"
                        tools_disabled = True
                    elif progress.should_block(name, arguments):
                        self.tools.logger.write_event(
                            "tool_loop_prevented",
                            agent_turn_id=agent_turn_id,
                            tool=name,
                            arguments=arguments,
                            reason="two_equivalent_calls_without_progress",
                            evidence=progress.evidence(),
                            state="blocked",
                        )
                        result = {
                            "ok": False,
                            "error": "tool_loop_prevented",
                            "message": (
                                "Repeated tool call blocked. Previous result is "
                                "already available. Choose another action or answer "
                                "using existing information."
                            ),
                            "evidence": progress.evidence(),
                        }
                        observed_outcome = "loop_prevented"
                        tools_disabled = True
                    elif (
                        name in _SINGLE_CALL_TOOLS
                        and tool_call_counts.get(name, 0) >= 1
                    ):
                        self.tools.logger.write_event(
                            "duplicate_tool_call_blocked",
                            agent_turn_id=agent_turn_id,
                            tool=name,
                            arguments=arguments,
                            reason="single_call_tool_repeated",
                            state="blocked",
                        )
                        timing_values = finish_observation("loop_prevented")
                        return {
                            "ok": False,
                            "error": "duplicate_tool_call_blocked",
                            "message": f"segunda chamada de {name} bloqueada nesta tarefa",
                            "tool_calls": calls,
                            "timing": timing_values,
                        }
                    else:
                        tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
                        self.tools.logger.write_event(
                            "tool_dispatched",
                            agent_turn_id=agent_turn_id,
                            tool=name,
                            arguments=arguments,
                            prompt=original_user_text,
                            routing_text=routing_text,
                            project=(
                                arguments.get("project_path")
                                or arguments.get("working_directory")
                                or arguments.get("project_id")
                            ),
                            state="dispatching",
                        )
                        emit("tool_start", name=name)
                        timing.mark("tool_started")
                        tool_started = time.perf_counter()
                        result = self.tools.execute(
                            name,
                            arguments,
                            context={
                                "turn_id": agent_turn_id,
                                "user_text": routing_text,
                                "original_user_text": original_user_text,
                                "request_fingerprint": fingerprint,
                                "conversation_id": self.conversation_id,
                                "focused_codex_thread_id": (
                                    decision_context.focused_session
                                    if decision_context.focused_agent == "codex"
                                    else None
                                ),
                            },
                            event_callback=lambda event, values: emit(
                                event, **values
                            ),
                        )
                        timing.tool_execution_ms += (
                            time.perf_counter() - tool_started
                        ) * 1000
                        timing.mark("tool_completed")
                        emit("tool_end", name=name, ok=result.get("ok"))
                        actual_tools.append(name)
                        self.decision_policy.record_tool_result(
                            name,
                            arguments,
                            result,
                        )
                        if semantic_plan and result.get("ok"):
                            completed_step_tool = (
                                decision.tools[semantic_plan_index]
                                if semantic_plan_index < len(decision.tools)
                                else None
                            )
                            if name == completed_step_tool and semantic_plan_index + 1 < len(decision.tools):
                                next_step = semantic_plan[semantic_plan_index + 1]
                                condition = _condition_satisfied(next_step.condition, result)
                                if condition is True:
                                    semantic_plan_index += 1
                                    next_tool = decision.tools[semantic_plan_index]
                                    tool_specs = [
                                        item
                                        for item in tool_specs_for_decision(self.tools.specs(), decision)
                                        if item.get("function", {}).get("name") == next_tool
                                    ]
                                else:
                                    tools_disabled = True
                                    observed_outcome = (
                                        "conditional_plan_not_satisfied"
                                        if condition is False
                                        else "conditional_plan_unverified"
                                    )
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": (
                                                "A proxima etapa do plano e condicional e foi bloqueada. "
                                                "Nao chame ferramentas. "
                                                + (
                                                    "A recomendacao nao foi positiva. Explique isso."
                                                    if condition is False
                                                    else "O resultado nao declarou se a recomendacao foi positiva. "
                                                    "Peca confirmacao curta ao usuario."
                                                )
                                            ),
                                        }
                                    )
                        if name == "delegate_to_codex" and result.get("ok"):
                            emit("assistant_analyzing_result", name=name)
                        observation = progress.record(name, arguments, result)
                        self.tools.logger.write_event(
                            "tool_returned",
                            agent_turn_id=agent_turn_id,
                            tool=name,
                            project=(
                                arguments.get("project_path")
                                or arguments.get("working_directory")
                                or result.get("project_id")
                            ),
                            thread_id=(
                                result.get("thread_id") or result.get("session_id")
                            ),
                            turn_id=result.get("turn_id"),
                            state=result.get("status"),
                            error=result.get("error"),
                            progress=observation.progress,
                            new_paths=observation.new_paths,
                            new_entities=observation.new_entities,
                        )
                    if result.get("error") == "ApprovalRequired":
                        timing_values = finish_observation("user_cancelled")
                        return {
                            "ok": False,
                            "error": "approval_required",
                            "message": "acao sensivel cancelada ou nao confirmada",
                            "tool_calls": calls + 1,
                            "timing": timing_values,
                            "web": {
                                "used": web_used,
                                "sources": web_sources,
                            },
                        }
                    if result.get("error") in {
                        "ActionCancelled",
                        "AccessDenied",
                        "invalid_arguments",
                        "unknown_tool",
                        "duplicate_tool_call_blocked",
                        "action_pending",
                    }:
                        tools_disabled = True
                        observed_outcome = (
                            "user_cancelled"
                            if result.get("error") in {"ActionCancelled", "AccessDenied"}
                            else "tool_error"
                        )
                    if result.get("error") == "duplicate_tool_call_blocked":
                        timing_values = finish_observation("tool_error")
                        return {
                            "ok": False,
                            "error": "duplicate_tool_call_blocked",
                            "message": result.get("message"),
                            "action_id": result.get("action_id"),
                            "tool_calls": calls + 1,
                            "timing": timing_values,
                        }
                    if name in {
                        "delegate_to_codex",
                        "get_codex_job_status",
                        "cancel_codex_job",
                        "steer_codex_job",
                    }:
                        tools_disabled = True
                    if name.startswith("web_"):
                        web_used = True
                    if (
                        name in {"web_open", "web_open_browser", "web_extract"}
                        and result.get("ok")
                        and isinstance(result.get("citation"), dict)
                    ):
                        citation = result["citation"]
                        title = citation.get("title")
                        url = citation.get("url")
                        if (
                            isinstance(title, str)
                            and isinstance(url, str)
                            and not any(
                                source["url"] == url for source in web_sources
                            )
                        ):
                            web_sources.append({"title": title, "url": url})
                calls += 1
                if calls >= decision_tool_budget:
                    tools_disabled = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"call_{calls}"),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                safety_report = _web_safety_report(name, arguments, result)
                if safety_report is not None:
                    observed_outcome = "tool_error"
                    return completed(
                        safety_report,
                        None,
                        response_stage="WEB_SAFETY_REPORT",
                    )
                if (
                    not tools_disabled
                    and "arguments" in locals()
                    and isinstance(arguments, dict)
                    and progress.equivalent_without_progress(name, arguments)
                ):
                    self.tools.logger.write_event(
                        "tool_progress_stalled",
                        agent_turn_id=agent_turn_id,
                        tool=name,
                        arguments=arguments,
                        evidence=progress.evidence(),
                        state="replan_required",
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Repeated tool call blocked. Previous result is already "
                                "available. Choose another action or answer using "
                                "existing information."
                            ),
                        }
                    )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Limite de ferramentas atingido. Nao chame ferramentas. "
                    "Responda agora usando somente resultados confirmados acima, "
                    "incluindo citacoes das fontes abertas."
                ),
            }
        )
        try:
            timing.mark("final_response_started")
            emit("final_response_started")
            timing.mark("qwen_request_started")
            emit("qwen_request_started")
            qwen_started = time.perf_counter()
            final_response = self.client.chat(messages, tools=None)
            timing.qwen_request_ms += (time.perf_counter() - qwen_started) * 1000
            timing.qwen_requests += 1
            final_message = final_response.get("choices", [{}])[0].get(
                "message", {}
            )
            answer = final_message.get("content", "")
            if answer:
                return completed(answer, final_response.get("usage"))
        except ServerError:
            timing.qwen_request_ms += (time.perf_counter() - qwen_started) * 1000
            timing.qwen_requests += 1
        timing_values = finish_observation("tool_error")
        return {
            "ok": False,
            "error": "tool_limit",
            "message": f"limite de {self.settings.max_tool_calls} chamadas atingido",
            "tool_calls": calls,
            "timing": timing_values,
            "web": {"used": web_used, "sources": web_sources},
        }
