from __future__ import annotations

import json
import os
from typing import Any, Callable

from .client import LlamaClient, ServerError
from .config import Settings
from .prompt import SYSTEM_PROMPT
from .tools import ToolRegistry


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

        begin_research = getattr(self.tools.web, "begin_research", None)
        if callable(begin_research):
            begin_research(user_text)

        trusted_runtime_context = (
            "\nContexto confiavel do runtime:\n"
            f"- working_directory: {os.getcwd()}\n"
            f"- backend: {self.settings.backend.name}\n"
            f"- allowed_roots: {', '.join(str(root) for root in self.settings.allowed_roots)}\n"
            f"- max_tool_calls: {self.settings.max_tool_calls}\n"
            f"- max_attempts: {self.settings.max_attempts}\n"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + trusted_runtime_context},
            {"role": "user", "content": user_text},
        ]
        fingerprints: set[str] = set()
        calls = 0
        failures = 0
        web_used = False
        web_sources: list[dict[str, str]] = []

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
                response = self.client.chat(messages, tools=self.tools.specs())
            except ServerError:
                failures += 1
                if failures >= self.settings.max_attempts:
                    raise
                continue
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return completed(
                    message.get("content", ""), response.get("usage")
                )
            messages.append(message)
            for call in tool_calls:
                if calls >= self.settings.max_tool_calls:
                    break
                function = call.get("function", {})
                name = function.get("name", "")
                raw_arguments = function.get("arguments", {})
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("argumentos devem ser objeto JSON")
                except (json.JSONDecodeError, ValueError) as exc:
                    result = {"ok": False, "error": "invalid_json", "message": str(exc)}
                    arguments = {"_raw": str(raw_arguments)[:4096]}
                else:
                    fingerprint = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
                    if fingerprint in fingerprints:
                        return {
                            "ok": False,
                            "error": "loop_prevented",
                            "message": "chamada de ferramenta repetida sem nova evidencia",
                            "tool_calls": calls,
                        }
                    fingerprints.add(fingerprint)
                    emit("tool_start", name=name)
                    result = self.tools.execute(name, arguments)
                    emit("tool_end", name=name, ok=result.get("ok"))
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
