from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from typing import Any, Callable

from .client import LlamaClient, ServerError
from .config import Settings
from .prompt import SYSTEM_PROMPT
from .projects import normalize_technical_transcript
from .tool_progress import ToolProgressTracker
from .tools import ToolRegistry


_SINGLE_CALL_TOOLS = frozenset(
    {
        "delegate_to_codex",
        "review_codex_session",
        "get_codex_job_status",
        "cancel_codex_job",
        "steer_codex_job",
    }
)


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


class Supervisor:
    def __init__(self, settings: Settings, client: LlamaClient, tools: ToolRegistry):
        self.settings = settings
        self.client = client
        self.tools = tools

    def run(
        self,
        user_text: str,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        def emit(event: str, **values: Any) -> None:
            if event_callback is not None:
                event_callback(event, values)

        original_user_text = user_text
        routing_text = normalize_technical_transcript(user_text)
        begin_research = getattr(self.tools.web, "begin_research", None)
        if callable(begin_research):
            begin_research(routing_text)

        project_context = self.tools.projects.context_text()

        trusted_runtime_context = (
            "\nContexto confiavel do runtime:\n"
            f"- working_directory: {os.getcwd()}\n"
            f"- backend: {self.settings.backend.name}\n"
            f"- allowed_roots: {', '.join(str(root) for root in self.settings.allowed_roots)}\n"
            f"- max_tool_calls: {self.settings.max_tool_calls}\n"
            f"- max_attempts: {self.settings.max_attempts}\n"
            "\nProject context:\n"
            f"{project_context}\n"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + trusted_runtime_context},
            {"role": "user", "content": routing_text},
        ]
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
        agent_turn_id = f"qwen-{uuid.uuid4()}"
        progress = ToolProgressTracker()
        tool_call_counts: dict[str, int] = {}
        calls = 0
        failures = 0
        web_used = False
        web_sources: list[dict[str, str]] = []
        codex_history_request = _is_codex_history_request(routing_text)
        codex_job_intent = _codex_job_intent(routing_text)
        project_lookup_only = _project_lookup_only(routing_text)
        tools_disabled = False
        tool_specs = self.tools.specs()
        if codex_job_intent:
            tool_specs = [
                item
                for item in tool_specs
                if item.get("function", {}).get("name") == codex_job_intent
            ]
        elif codex_history_request:
            tool_specs = [
                item
                for item in tool_specs
                if item.get("function", {}).get("name")
                == "review_codex_session"
            ]
        elif project_lookup_only:
            tool_specs = [
                item
                for item in tool_specs
                if item.get("function", {}).get("name")
                in {"resolve_project", "find_project_files"}
            ]

        def completed(answer: str, usage: dict[str, Any] | None) -> dict[str, Any]:
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
            return {
                "ok": True,
                "answer": answer,
                "tool_calls": calls,
                "usage": usage,
                "web": {
                    "used": web_used,
                    "sources": web_sources,
                },
            }

        while calls < self.settings.max_tool_calls:
            emit("thinking")
            try:
                response = self.client.chat(
                    messages,
                    tools=(
                        None
                        if tools_disabled
                        or (codex_history_request and calls > 0)
                        else tool_specs
                    ),
                )
            except ServerError:
                failures += 1
                if failures >= self.settings.max_attempts:
                    raise
                continue
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls") or []
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
                return {
                    "ok": False,
                    "error": "tool_loop_blocked",
                    "message": "nova chamada de ferramenta bloqueada apos encerramento do fluxo",
                    "tool_calls": calls,
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
                    tools_disabled = True
                else:
                    fingerprint = progress.fingerprint(name, arguments)
                    if progress.should_block(name, arguments):
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
                                "terceira chamada equivalente bloqueada; "
                                "reformule o plano usando as evidencias ja obtidas"
                            ),
                            "evidence": progress.evidence(),
                        }
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
                        return {
                            "ok": False,
                            "error": "duplicate_tool_call_blocked",
                            "message": f"segunda chamada de {name} bloqueada nesta tarefa",
                            "tool_calls": calls,
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
                        result = self.tools.execute(
                            name,
                            arguments,
                            context={
                                "turn_id": agent_turn_id,
                                "user_text": routing_text,
                                "original_user_text": original_user_text,
                                "request_fingerprint": fingerprint,
                            },
                            event_callback=lambda event, values: emit(
                                event, **values
                            ),
                        )
                        emit("tool_end", name=name, ok=result.get("ok"))
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
                        return {
                            "ok": False,
                            "error": "approval_required",
                            "message": "acao sensivel cancelada ou nao confirmada",
                            "tool_calls": calls + 1,
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
                    if result.get("error") == "duplicate_tool_call_blocked":
                        return {
                            "ok": False,
                            "error": "duplicate_tool_call_blocked",
                            "message": result.get("message"),
                            "action_id": result.get("action_id"),
                            "tool_calls": calls + 1,
                        }
                    if codex_history_request:
                        tools_disabled = True
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
                        name in {"web_open", "web_extract"}
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"call_{calls}"),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
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
                                "Duas chamadas equivalentes nao produziram novas "
                                "entidades nem caminhos. Nao repita a chamada. "
                                "Reformule o plano com as evidencias ja obtidas."
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
            final_response = self.client.chat(messages, tools=None)
            final_message = final_response.get("choices", [{}])[0].get(
                "message", {}
            )
            answer = final_message.get("content", "")
            if answer:
                return completed(answer, final_response.get("usage"))
        except ServerError:
            pass
        return {
            "ok": False,
            "error": "tool_limit",
            "message": f"limite de {self.settings.max_tool_calls} chamadas atingido",
            "tool_calls": calls,
            "web": {"used": web_used, "sources": web_sources},
        }
