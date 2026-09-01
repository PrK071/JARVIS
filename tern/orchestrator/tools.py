from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .codex import CodexRunner
from .applications import ApplicationManager
from .delegation import DelegationRequest
from .deepseek import DeepSeekSessionManager
from .external_agents import AgentDiscovery, ExternalAgentRunner
from .hardware import HardwareMonitor
from .pending_actions import PendingActionStore
from .projects import ProjectRegistry
from .schema import SchemaError, validate
from .security import ActionLogger, ApprovalCallback, PathPolicy
from .web import WebClient, WebConfig, WebError


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

_CODEX_SENSITIVE_PATTERNS = (
    ("install_software", r"\b(?:instal(?:ar|e)|install)\b"),
    (
        "remove_software",
        r"\b(?:desinstal(?:ar|e)|uninstall|remover?\s+(?:software|programa|pacote))\b",
    ),
    (
        "system_change",
        r"\b(?:registro\s+do\s+windows|configura(?:cao|ção)\s+do\s+sistema|system\s+settings?)\b",
    ),
    (
        "administrative",
        r"\b(?:administrador|administrativo|admin(?:istrative)?|elevad[oa]|sudo)\b",
    ),
    (
        "codex_modify_files",
        r"\b(?:apag(?:ar|ue)|delet(?:ar|e)|sobrescrev(?:er|a)|"
        r"corrig(?:ir|e)|corrij(?:a|am)|edit(?:ar|e)|modific(?:ar|e)|implement(?:ar|e)|"
        r"criar?\s+(?:arquivo|codigo|código)|fix|delete|overwrite|modify|edit)\b",
    ),
)

_PASSIVE_PROGRESS_TOOLS = frozenset(
    {
        "filesystem_list",
        "filesystem_read_text",
        "resolve_project",
        "find_project_files",
        "get_project_git_state",
        "web_search",
        "web_open",
        "web_extract",
    }
)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: ToolHandler
    timeout: int

    def openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


class ToolRegistry:
    def __init__(
        self,
        *,
        policy: PathPolicy,
        logger: ActionLogger,
        codex: CodexRunner,
        max_output_bytes: int,
        approval: ApprovalCallback | None = None,
        confirmation_timeout_seconds: int = 300,
        web: WebClient | None = None,
        projects: ProjectRegistry | None = None,
        deepseek: DeepSeekSessionManager | None = None,
        hardware: HardwareMonitor | None = None,
        applications: ApplicationManager | None = None,
        agent_discovery: AgentDiscovery | None = None,
    ):
        self.policy = policy
        self.logger = logger
        self.codex = codex
        self.max_output_bytes = max_output_bytes
        self.approval = approval
        self.web = web or WebClient(WebConfig(enabled=False))
        self.pending_actions = PendingActionStore(
            logger.path.parent,
            timeout_seconds=confirmation_timeout_seconds,
        )
        self.projects = projects or ProjectRegistry(
            policy,
            logger.path.parent,
            codex=codex,
        )
        self.deepseek = deepseek
        self.hardware = hardware or HardwareMonitor()
        self.applications = applications or ApplicationManager()
        self.agent_discovery = agent_discovery or AgentDiscovery()
        self.external_agents = ExternalAgentRunner(
            policy=policy,
            discovery=self.agent_discovery,
            logger=logger,
        )
        self._execution = threading.local()
        self._tools: dict[str, Tool] = {}
        self._external_agent_tools: tuple[str, ...] = ()
        self._register_defaults()
        self.refresh_external_agents()

    def refresh_external_agents(self, *, force: bool = False) -> tuple[str, ...]:
        """Registra uma ferramenta de delegação por agente externo utilizável.

        Chamado na construção e sob demanda: se o usuário abrir uma sessão nova
        (Kiro, Claude, GLM), a ferramenta aparece sem reiniciar o orquestrador.
        Agente ausente não ganha ferramenta, então o modelo não promete o que
        não existe.
        """
        descobertos = self.agent_discovery.discover(force=force)
        desejados = {
            agent.delegation_tool: agent
            for agent in descobertos
            if agent.usable and not agent.spec.native_integration
        }

        for nome in self._external_agent_tools:
            if nome not in desejados:
                self._tools.pop(nome, None)

        for nome, agent in desejados.items():
            if nome in self._tools:
                continue
            self._add_external_agent_tool(nome, agent)

        self._external_agent_tools = tuple(sorted(desejados))
        return self._external_agent_tools

    def _add_external_agent_tool(self, nome: str, agent: Any) -> None:
        identificador = agent.spec.id
        capacidades = ", ".join(sorted(c.value for c in agent.spec.capabilities)) or "nao declaradas"
        self._add(
            nome,
            (
                f"Delega uma tarefa ao agente externo {agent.spec.display_name} "
                f"(detectado nesta máquina, estado {agent.availability.value}). "
                f"Capacidades: {capacidades}. {agent.spec.notes} "
                "Informe task auto-contida e project_path quando a tarefa for de projeto."
            ),
            _object(
                {
                    "task": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "project_path": {
                        "anyOf": [
                            {"type": "string", "minLength": 1, "maxLength": 4096},
                            {"type": "null"},
                        ]
                    },
                },
                ["task"],
            ),
            lambda arguments, _agente=identificador: self.external_agents.run(
                _agente,
                str(arguments.get("task") or ""),
                project_path=arguments.get("project_path") or None,
            ),
            self.external_agents.default_timeout,
        )

    def specs(self) -> list[dict[str, Any]]:
        return [tool.openai() for tool in self._tools.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def _normalize_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(arguments)
        if name == "resolve_project":
            return normalized
        if name == "find_project_files":
            project_id = str(normalized.get("project_id") or "").strip()
            if not any(item["id"] == project_id for item in self.projects.projects()):
                raise ValueError("project_id nao registrado")
            return normalized
        if name in {
            "filesystem_list",
            "filesystem_read_text",
            "filesystem_delete",
        }:
            normalized["path"] = str(self.policy.resolve(str(normalized["path"])))
            return normalized
        if name == "filesystem_write_text":
            normalized["directory"] = str(
                self.policy.resolve(str(normalized["directory"]))
            )
            return normalized
        if name in {"review_codex_session", "review_deepseek_session"} and normalized.get("project_path"):
            normalized["project_path"] = str(
                self.policy.resolve(str(normalized["project_path"]))
            )
            return normalized
        if name == "delegate_to_deepseek":
            requested = str(normalized.get("project_path") or "").strip()
            user_text = str(context.get("user_text") or "")
            resolution = self.projects.resolve(
                query=user_text,
                path_hint=requested if requested and Path(requested).is_absolute() else None,
            )
            if not resolution.get("ok"):
                raise ValueError("nao foi possivel identificar um projeto permitido")
            normalized["project_path"] = str(
                self.policy.resolve(str(resolution["root"]))
            )
            return normalized
        if name in {"get_project_git_state", "run_project_tests"}:
            normalized["project_path"] = str(
                self.policy.resolve(str(normalized["project_path"]))
            )
            return normalized
        if name.startswith("delegate_to_") and name != "delegate_to_codex":
            requested = str(normalized.get("project_path") or "").strip()
            if requested:
                normalized["project_path"] = str(self.policy.resolve(requested))
            return normalized
        if name != "delegate_to_codex":
            return normalized
        user_text = str(context.get("user_text") or "")
        explicit_paths = re.findall(
            r"(?i)(?:[A-Z]:\\[^\s,;]+)",
            user_text,
        )
        project: Path | None = None
        for raw_path in explicit_paths:
            candidate = raw_path.rstrip(".?!:)")
            try:
                resolution = self.projects.resolve(path_hint=candidate)
                if resolution.get("ok"):
                    project = self.policy.resolve(str(resolution["root"]))
                    break
            except Exception:
                continue
        if project is None and not user_text:
            requested = str(normalized.get("project_path") or "").strip()
            if requested:
                try:
                    resolution = self.projects.resolve(path_hint=requested)
                    if resolution.get("ok"):
                        project = self.policy.resolve(str(resolution["root"]))
                except Exception:
                    project = None
        if project is None:
            requested = str(normalized.get("project_path") or "").strip()
            resolution = self.projects.resolve(
                query=user_text,
                path_hint=(
                    requested
                    if not user_text and requested and Path(requested).is_absolute()
                    else None
                ),
            )
            if resolution.get("ok"):
                project = self.policy.resolve(str(resolution["root"]))
        if project is None and not explicit_paths:
            shared_project = getattr(self.codex, "shared_project", None)
            if callable(shared_project):
                try:
                    shared = shared_project()
                    if shared:
                        try:
                            project = self.policy.resolve(str(shared))
                        except (OSError, PermissionError):
                            candidate = Path(shared).expanduser().resolve(strict=False)
                            allowed_roots = [
                                root.expanduser().resolve(strict=False)
                                for root in self.policy.roots
                            ]
                            if any(
                                candidate == root or root in candidate.parents
                                for root in allowed_roots
                            ):
                                project = candidate
                except Exception:
                    project = None
        if project is None:
            raise ValueError("nao foi possivel identificar um projeto permitido")
        normalized["project_path"] = str(project)
        if "wait" not in normalized:
            authoritative_task = str(
                context.get("original_user_text") or normalized.get("task") or ""
            )
            intent = f"{user_text}\n{authoritative_task}".casefold()
            if re.search(r"\b(?:segundo plano|background|nao aguarde|não aguarde)\b", intent):
                normalized["wait"] = False
            elif re.search(r"\b(?:aguarde terminar|espere terminar|wait)\b", intent):
                normalized["wait"] = True
            else:
                normalized["wait"] = not bool(
                    re.search(
                        r"\b(?:implement|corrig|su[ií]te completa|auditoria ampla|"
                        r"instal|benchmark|muitos arquivos|refator|migrar|build)\w*\b",
                        intent,
                    )
                )
        return normalized

    @staticmethod
    def _preserve_delegation_request(
        name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not name.startswith("delegate_to_"):
            return arguments
        normalized = dict(arguments)
        request = DelegationRequest.build(
            requested_agent=name.removeprefix("delegate_to_"),
            submitted_task=str(normalized.get("task") or ""),
            project_path=(
                str(normalized["project_path"])
                if normalized.get("project_path")
                else None
            ),
            context=context,
        )
        normalized["task"] = request.serialize()
        return normalized

    def _confirmation_requirement(
        self,
        name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> str | None:
        user_text = str(context.get("user_text") or "")
        if name == "schedule_application":
            return "schedule_task"
        if name == "filesystem_delete":
            return "delete"
        if name == "filesystem_write_text":
            try:
                destination = self.policy.child(
                    arguments["directory"],
                    arguments["name"],
                    must_exist=False,
                )
                if not destination.exists():
                    return None
                authority = context.get("_bounded_live_authority_decision")
                if bool(
                    getattr(authority, "allowed", False)
                    and getattr(authority, "mutation_authorized", False)
                ):
                    return None
                return "overwrite"
            except Exception:
                return "outside_project"
        if name != "delegate_to_codex":
            return None
        project_path = Path(str(arguments["project_path"])).resolve()
        shared_project = getattr(self.codex, "shared_project", None)
        current_project = shared_project() if callable(shared_project) else project_path
        if current_project is None:
            current_project = Path(__file__).resolve().parents[2]
        if project_path != Path(current_project).resolve():
            return "outside_project"
        task = str(arguments["task"])
        action = self._codex_sensitive_action(task)
        if action in {
            "install_software",
            "remove_software",
            "system_change",
            "administrative",
        }:
            return action
        if re.search(
            r"(?i)\b(?:credenciais?|senhas?|passwords?|api[_ -]?keys?|"
            r"arquivos? pessoais?|irreversivel|irreversível|backup|sincroniza)\b",
            task,
        ):
            return "high_impact"
        if action != "codex_modify_files":
            return None
        explicitly_requested = bool(
            re.search(
                r"(?i)\b(?:corrig(?:ir|e|a)|implementar?|implemente|"
                r"modific(?:ar|e|a)|edit(?:ar|e)|criar?|melhor(?:ar|e)|"
                r"ajust(?:ar|e)|consert(?:ar|e))\b",
                user_text,
            )
        )
        return None if explicitly_requested else "codex_modify_files"

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": "unknown_tool", "message": f"ferramenta inexistente: {name}"}
        context = context or {}
        try:
            validate(arguments, tool.schema)
            normalized = self._normalize_arguments(name, arguments, context)
            validate(normalized, tool.schema)
            normalized = self._preserve_delegation_request(
                name,
                normalized,
                context,
            )
            if name == "delegate_to_codex":
                normalized["_conversation_id"] = str(
                    context.get("conversation_id") or ""
                )
                normalized["_focused_codex_thread_id"] = str(
                    context.get("focused_codex_thread_id") or ""
                )
                normalized["_execution_mode"] = str(
                    context.get("execution_mode") or ""
                )
        except SchemaError as exc:
            result = {"ok": False, "error": "invalid_arguments", "message": str(exc)}
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        else:
            project = normalized.get("project_path") or normalized.get(
                "working_directory"
            )
            risk_arguments = normalized
            if name == "delegate_to_codex":
                risk_arguments = {
                    **normalized,
                    "task": str(arguments.get("task") or normalized.get("task") or ""),
                }
            risk = self._confirmation_requirement(
                name,
                risk_arguments,
                context,
            )
            turn_id = str(context.get("turn_id") or f"direct-{uuid.uuid4()}")
            pending_turn_id = (
                f"{turn_id}:observation-{uuid.uuid4()}"
                if risk is None and name in _PASSIVE_PROGRESS_TOOLS
                else turn_id
            )
            action_id = str(uuid.uuid4())
            record, prepared = self.pending_actions.prepare(
                action_id=action_id,
                tool=name,
                arguments=normalized,
                project=str(project) if project else None,
                risk=risk or "normal",
                confirmation_required=risk is not None,
                turn_id=pending_turn_id,
            )
            if prepared != "created":
                self.logger.write_event(
                    "duplicate_tool_call_blocked",
                    action_id=record.get("action_id"),
                    tool=name,
                    arguments=normalized,
                    project=project,
                    risk=risk or "normal",
                    turn_id=turn_id,
                    reason=prepared,
                    request_fingerprint=record.get("request_fingerprint"),
                )
                result = {
                    "ok": False,
                    "error": "duplicate_tool_call_blocked",
                    "message": "chamada identica ja pendente ou executada neste turno",
                    "action_id": record.get("action_id"),
                    "status": record.get("status"),
                }
            elif risk is not None:
                self.pending_actions.mark_presented(action_id)
                self.logger.write_event(
                    "pending_action_created",
                    action_id=action_id,
                    tool=name,
                    project=project,
                    risk=risk,
                    status="awaiting_confirmation",
                    request_fingerprint=record.get("request_fingerprint"),
                )
                if event_callback is not None:
                    event_callback(
                        "action_pending",
                        {
                            "action_id": action_id,
                            "tool": name,
                            "risk": risk,
                            "project": project,
                        },
                    )
                approval_arguments = {
                    **normalized,
                    "path": str(project or normalized.get("path") or ""),
                    "action_id": action_id,
                    "tool": name,
                    "risk": risk,
                }
                if self.approval is None:
                    self.cancel_pending_action(
                        action_id,
                        reason="confirmation_callback_unavailable",
                    )
                    result = {
                        "ok": False,
                        "error": "ApprovalRequired",
                        "message": f"confirmacao necessaria para {risk}",
                        "action_id": action_id,
                    }
                else:
                    try:
                        approved = self.approval(risk, approval_arguments)
                    except BaseException as exc:
                        self.cancel_pending_action(
                            action_id,
                            reason=f"confirmation_aborted:{type(exc).__name__}",
                        )
                        raise
                if self.approval is not None and not approved:
                    self.cancel_pending_action(
                        action_id,
                        reason="user_cancelled",
                    )
                    result = {
                        "ok": False,
                        "error": "ActionCancelled",
                        "message": "acao pendente cancelada pelo usuario",
                        "action_id": action_id,
                        "status": "cancelled",
                    }
                elif self.approval is not None:
                    result = self.confirm_pending_action(
                        action_id,
                        event_callback=event_callback,
                    )
            else:
                result = self.confirm_pending_action(
                    action_id,
                    event_callback=event_callback,
                )
            arguments = normalized
        if name == "delegate_to_deepseek":
            self.logger.write(
                tool=name,
                arguments={
                    "project_path": arguments.get("project_path"),
                    "continue_current_session": arguments.get("continue_current_session", True),
                    "task_characters": len(str(arguments.get("task") or "")),
                    "context_characters": len(str(arguments.get("context") or "")),
                },
                result={
                    "ok": result.get("ok"),
                    "session_id": result.get("session_id"),
                    "project": result.get("project"),
                    "model": result.get("model"),
                    "response_characters": len(str(result.get("response") or "")),
                    "error": result.get("error"),
                },
            )
        else:
            self.logger.write(tool=name, arguments=arguments, result=result)
        return result

    def confirm_pending_action(
        self,
        action_id: str,
        *,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        try:
            record, claimed = self.pending_actions.claim_execution(action_id)
        except KeyError as exc:
            return {
                "ok": False,
                "error": "pending_action_not_found",
                "message": str(exc),
                "action_id": action_id,
            }
        if not claimed:
            status = str(record.get("status") or "unknown")
            error = (
                "ActionExpired"
                if status == "expired"
                else "duplicate_tool_call_blocked"
            )
            return {
                "ok": False,
                "error": error,
                "message": (
                    "confirmacao expirada"
                    if status == "expired"
                    else "acao ja esta em execucao ou foi finalizada"
                ),
                "action_id": action_id,
                "status": status,
            }
        self.logger.write_event(
            "pending_action_executing",
            action_id=action_id,
            tool=record.get("tool"),
            project=record.get("project"),
            status="executing",
            request_fingerprint=record.get("request_fingerprint"),
        )
        if event_callback is not None and record.get("tool") == "delegate_to_codex":
            event_callback(
                "codex_sending",
                {"action_id": action_id, "project": record.get("project")},
            )
        tool = self._tools.get(str(record.get("tool") or ""))
        if tool is None:
            result = {
                "ok": False,
                "error": "unknown_tool",
                "message": "ferramenta da acao pendente nao existe",
            }
        else:
            self._execution.event_callback = event_callback
            try:
                result = self._invoke_handler(tool, dict(record["arguments"]))
            finally:
                self._execution.event_callback = None
        succeeded = bool(result.get("ok"))
        private_deepseek = record.get("tool") == "delegate_to_deepseek"
        raw_arguments = (
            record.get("arguments")
            if isinstance(record.get("arguments"), dict)
            else {}
        )
        stored_arguments = (
            {
                "project_path": raw_arguments.get("project_path"),
                "continue_current_session": raw_arguments.get(
                    "continue_current_session", True
                ),
                "task_characters": len(str(raw_arguments.get("task") or "")),
                "context_characters": len(str(raw_arguments.get("context") or "")),
            }
            if private_deepseek
            else None
        )
        stored_result = (
            {
                "ok": result.get("ok"),
                "session_id": result.get("session_id"),
                "project": result.get("project"),
                "model": result.get("model"),
                "response_characters": len(str(result.get("response") or "")),
                "error": result.get("error"),
            }
            if private_deepseek
            else result
        )
        self.pending_actions.complete(
            action_id,
            status="completed" if succeeded else "failed",
            result=stored_result,
            error=None if succeeded else str(result.get("error") or "failed"),
            arguments_override=stored_arguments,
        )
        self.logger.write_event(
            "pending_action_finished",
            action_id=action_id,
            tool=record.get("tool"),
            project=record.get("project"),
            status="completed" if succeeded else "failed",
            error=result.get("error"),
        )
        return result

    def cancel_pending_action(self, action_id: str, *, reason: str) -> dict[str, Any]:
        record = self.pending_actions.complete(
            action_id,
            status="cancelled",
            error=reason,
        )
        self.logger.write_event(
            "pending_action_cancelled",
            action_id=action_id,
            tool=record.get("tool"),
            project=record.get("project"),
            status="cancelled",
            reason=reason,
        )
        return record

    def _invoke_handler(self, tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = tool.handler(arguments)
            encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
            if len(encoded) > self.max_output_bytes:
                return {
                    "ok": False,
                    "error": "output_too_large",
                    "message": f"retorno excedeu {self.max_output_bytes} bytes",
                }
            return result
        except WebError as exc:
            result = {"ok": False, "error": exc.code, "message": str(exc)}
            if exc.details:
                result["details"] = exc.details
            return result
        except TimeoutError as exc:
            return {"ok": False, "error": "timeout", "message": str(exc)}
        except Exception as exc:
            return {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }

    def _available_agents_result(self) -> dict[str, Any]:
        """Estado medido dos agentes externos, mais os agentes nativos."""
        self.refresh_external_agents(force=True)
        descobertos = self.agent_discovery.discover()
        nativos = [
            {
                "id": "qwen",
                "name": "Qwen local",
                "availability": "session_active",
                "usable": True,
                "role": "conversa, roteamento e coordenação",
                "delegation_tool": None,
            },
            {
                "id": "codex",
                "name": "Codex",
                "availability": "installed",
                "usable": "delegate_to_codex" in self._tools,
                "role": "programa, edita, testa e executa localmente",
                "delegation_tool": "delegate_to_codex",
            },
            {
                "id": "deepseek",
                "name": "DeepSeek",
                "availability": "api_key_only" if self.deepseek is not None else "absent",
                "usable": "delegate_to_deepseek" in self._tools,
                "role": "consultor: analise, revisao e segunda opiniao",
                "delegation_tool": "delegate_to_deepseek",
            },
        ]
        externos = [
            agent.as_dict()
            for agent in descobertos
            if not agent.spec.native_integration
        ]
        return {
            "ok": True,
            "native_agents": nativos,
            "external_agents": externos,
            "delegation_tools": sorted(
                nome for nome in self._tools if nome.startswith("delegate_to_")
            ),
        }

    def _add(self, name: str, description: str, schema: dict[str, Any], handler: ToolHandler, timeout: int) -> None:
        self._tools[name] = Tool(name, description, schema, handler, timeout)

    def _register_defaults(self) -> None:
        path = {"type": "string", "minLength": 1, "maxLength": 4096}
        self._add(
            "list_available_agents",
            (
                "Lista os agentes de IA detectados nesta máquina agora, com estado real "
                "(session_active, installed, configured_not_installed, api_key_only, absent), "
                "versão, capacidades e o nome da ferramenta de delegação de cada um. "
                "Use antes de afirmar que pode ou não delegar a um agente."
            ),
            _object({}, []),
            lambda _arguments: self._available_agents_result(),
            30,
        )
        nullable_text = {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 4096},
                {"type": "null"},
            ]
        }
        self._add(
            "get_hardware_telemetry",
            (
                "Lê telemetria real e atual do computador: temperatura da CPU "
                "quando um sensor compatível está disponível e quantidade de "
                "dispositivos USB físicos conectados. Nunca simula valores."
            ),
            _object({}, []),
            self._get_hardware_telemetry,
            20,
        )
        self._add(
            "list_installed_applications",
            "Lista aplicativos instalados no menu Iniciar. Não abre nem altera aplicativos.",
            _object(
                {
                    "query": {
                        "anyOf": [
                            {"type": "string", "minLength": 1, "maxLength": 200},
                            {"type": "null"},
                        ]
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                [],
            ),
            self._list_installed_applications,
            20,
        )
        self._add(
            "open_application",
            (
                "Abre aplicativo instalado resolvido pelo menu Iniciar. "
                "Não aceita caminho, comando ou executável arbitrário."
            ),
            _object(
                {"application_name": {"type": "string", "minLength": 1, "maxLength": 200}},
                ["application_name"],
            ),
            self._open_application,
            20,
        )
        self._add(
            "schedule_application",
            (
                "Agenda abertura de aplicativo instalado pelo Agendador de Tarefas do Windows. "
                "start_at deve ser data local ISO, por exemplo 2026-08-14T09:30. "
                "Sempre exige confirmação do usuário."
            ),
            _object(
                {
                    "application_name": {"type": "string", "minLength": 1, "maxLength": 200},
                    "start_at": {"type": "string", "minLength": 16, "maxLength": 40},
                    "recurrence": {"type": "string", "enum": ["once", "daily"]},
                },
                ["application_name", "start_at", "recurrence"],
            ),
            self._schedule_application,
            35,
        )
        self._add(
            "resolve_project",
            (
                "Resolve projeto por caminho, nome, alias, thread Codex ou "
                "projeto ativo. Nao le arquivos, nao modifica e nao delega."
            ),
            _object(
                {
                    "query": nullable_text,
                    "path_hint": nullable_text,
                    "require_unique": {"type": "boolean"},
                },
                [],
            ),
            self._resolve_project,
            15,
        )
        self._add(
            "find_project_files",
            (
                "Localiza arquivos no indice leve de um projeto resolvido. "
                "Use para nomes, descricoes, testes, documentacao e providers; "
                "nao use filesystem_list para a mesma descoberta."
            ),
            _object(
                {
                    "project_id": {"type": "string", "minLength": 1, "maxLength": 100},
                    "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "file_types": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1, "maxLength": 20},
                                "maxItems": 20,
                            },
                            {"type": "null"},
                        ]
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                ["project_id", "query"],
            ),
            self._find_project_files,
            30,
        )
        self._add(
            "get_project_git_state",
            (
                "Inspeciona branch, working tree e diff stat de um projeto Git permitido. "
                "Somente leitura; não faz checkout, commit, reset, push ou alteração."
            ),
            _object({"project_path": path}, ["project_path"]),
            self._get_project_git_state,
            20,
        )
        self._add(
            "run_project_tests",
            (
                "Executa pytest em modo seguro dentro de um projeto permitido. "
                "Aceita apenas um alvo relativo opcional e nunca usa shell."
            ),
            _object(
                {
                    "project_path": path,
                    "target": {
                        "anyOf": [
                            {"type": "string", "minLength": 1, "maxLength": 1000},
                            {"type": "null"},
                        ]
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "enum": [30, 60, 120, 300],
                    },
                },
                ["project_path"],
            ),
            self._run_project_tests,
            310,
        )
        self._add(
            "filesystem_list",
            (
                "Lista uma pasta permitida em uma unica chamada. Pode percorrer "
                "subpastas com profundidade limitada; retorna no maximo 500 entradas."
            ),
            _object(
                {
                    "path": path,
                    "recursive": {"type": "boolean"},
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                ["path"],
            ),
            self._list,
            15,
        )
        self._add(
            "filesystem_read_text",
            "Le um arquivo UTF-8 permitido, com limite explicito de bytes.",
            _object(
                {
                    "path": path,
                    "max_bytes": {"type": "integer", "enum": [4096, 16384, 65536, 131072]},
                },
                ["path", "max_bytes"],
            ),
            self._read,
            15,
        )
        self._add(
            "filesystem_write_text",
            "Cria arquivo UTF-8. Sobrescrever exige confirmacao externa.",
            _object(
                {
                    "directory": path,
                    "name": {"type": "string", "minLength": 1, "maxLength": 255},
                    "content": {"type": "string", "maxLength": 131072},
                },
                ["directory", "name", "content"],
            ),
            self._write,
            15,
        )
        self._add(
            "filesystem_delete",
            "Apaga um unico arquivo permitido; sempre exige confirmacao externa.",
            _object({"path": path}, ["path"]),
            self._delete,
            15,
        )
        self._add(
            "review_codex_session",
            (
                "Consulta o historico real da thread Codex compartilhada com "
                "thread/read. Use para ultimas informacoes, ultimos turns, "
                "revisao ou resumo da sessao. Nunca inicia turn, delega tarefa "
                "ou pesquisa arquivos e logs."
            ),
            _object(
                {
                    "project_path": path,
                    "turn_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                [],
            ),
            self._review_codex_session,
            min(self.codex.timeout, 60),
        )
        self._add(
            "delegate_to_codex",
            (
                "Envia tarefa real ao Codex App Server compartilhado e aguarda "
                "resultado. Retorna tambem intervencoes humanas e cancelamento. "
                "Use para codigo, testes, repositorio, bugs e recursos. Aceita "
                "thread_id somente quando uma identidade exata foi fornecida."
            ),
            _object(
                {
                    "task": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "project_path": path,
                    "continue_current_thread": {"type": "boolean"},
                    "thread_id": {
                        "anyOf": [
                            {"type": "string", "minLength": 1, "maxLength": 100},
                            {"type": "null"},
                        ]
                    },
                    "wait": {"type": "boolean"},
                },
                ["task", "project_path"],
            ),
            self._delegate_to_codex,
            self.codex.timeout,
        )
        if self.deepseek is not None:
            nullable_project = {
                "anyOf": [path, {"type": "null"}],
            }
            nullable_context = {
                "anyOf": [
                    {"type": "string", "minLength": 1, "maxLength": 30000},
                    {"type": "null"},
                ]
            }
            self._add(
                "review_deepseek_session",
                (
                    "Le apenas o historico persistido da sessao DeepSeek do projeto. "
                    "Nao chama a API nem cria uma nova sessao."
                ),
                _object(
                    {
                        "project_path": nullable_project,
                        "turn_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    [],
                ),
                self._review_deepseek_session,
                15,
            )
            self._add(
                "delegate_to_deepseek",
                (
                    "Consulta explicitamente o DeepSeek como agente consultivo, sem "
                    "filesystem ou execucao local. Persiste pergunta e resposta na "
                    "sessao logica compartilhada do projeto."
                ),
                _object(
                    {
                        "task": {"type": "string", "minLength": 1, "maxLength": 20000},
                        "project_path": nullable_project,
                        "continue_current_session": {"type": "boolean"},
                        "context": nullable_context,
                    },
                    ["task"],
                ),
                self._delegate_to_deepseek,
                self.deepseek.client.timeout_seconds,
            )
        nullable_job_id = {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 100},
                {"type": "null"},
            ]
        }
        self._add(
            "get_codex_job_status",
            (
                "Consulta o job Codex persistido sem iniciar turn nem ler o "
                "filesystem. Use para saber se a ultima delegacao terminou."
            ),
            _object(
                {
                    "job_id": nullable_job_id,
                    "latest": {"type": "boolean"},
                },
                [],
            ),
            self._get_codex_job_status,
            30,
        )
        self._add(
            "cancel_codex_job",
            "Interrompe o turn do job Codex ativo sem iniciar outro turn.",
            _object(
                {
                    "job_id": nullable_job_id,
                    "latest": {"type": "boolean"},
                },
                [],
            ),
            self._cancel_codex_job,
            30,
        )
        self._add(
            "steer_codex_job",
            "Envia direcao humana ao mesmo turn de um job Codex ativo.",
            _object(
                {
                    "instruction": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 10000,
                    },
                    "job_id": nullable_job_id,
                    "latest": {"type": "boolean"},
                },
                ["instruction"],
            ),
            self._steer_codex_job,
            30,
        )
        domain = {
            "type": "string",
            "minLength": 1,
            "maxLength": 253,
            "pattern": r"^(?:\*\.)?[A-Za-z0-9.-]+$",
        }
        nullable_page = {
            "anyOf": [
                {"type": "integer", "minimum": 1, "maximum": 10000},
                {"type": "null"},
            ]
        }
        self._add(
            "web_search",
            "Pesquisa a web e retorna resultados normalizados. Snippets servem para escolher fontes; abra a fonte antes de citar.",
            _object(
                {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1000,
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "language": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 6,
                    },
                    "freshness_days": {
                        "anyOf": [
                            {"type": "integer", "minimum": 1, "maximum": 3650},
                            {"type": "null"},
                        ]
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": domain,
                        "maxItems": 20,
                    },
                    "blocked_domains": {
                        "type": "array",
                        "items": domain,
                        "maxItems": 20,
                    },
                },
                ["query"],
            ),
            self._web_search,
            self.web.config.timeout,
        )
        self._add(
            "web_open",
            "Abre uma pagina HTTP/HTTPS permitida, extrai HTML, texto ou PDF e retorna metadados de citacao.",
            _object(
                {
                    "url": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8192,
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1024,
                        "maximum": 65536,
                    },
                    "page_start": nullable_page,
                    "page_end": nullable_page,
                },
                ["url"],
            ),
            self._web_open,
            self.web.config.timeout,
        )
        self._add(
            "web_open_browser",
            (
                "Abre uma URL HTTP/HTTPS ou um arquivo HTML local permitido em "
                "nova guia do navegador ja aberto; "
                "se nenhum existir, usa o navegador padrao do usuario. Faz isso "
                "somente apos validar a allowlist local ou DNS/IP, redirects e ameacas web."
            ),
            _object(
                {
                    "url": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8192,
                    },
                    "local_artifact": {"type": "boolean"},
                },
                ["url"],
            ),
            self._web_open_browser,
            self.web.config.timeout,
        )
        self._add(
            "web_extract",
            "Abre uma fonte e retorna passagens relevantes para uma consulta, mantendo URL e titulo verificaveis.",
            _object(
                {
                    "url": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8192,
                    },
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1000,
                    },
                    "max_passages": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "passage_chars": {
                        "type": "integer",
                        "minimum": 256,
                        "maximum": 4000,
                    },
                    "page_start": nullable_page,
                    "page_end": nullable_page,
                },
                ["url", "query"],
            ),
            self._web_extract,
            self.web.config.timeout,
        )

    def _list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        directory = self.policy.resolve(arguments["path"])
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        recursive = bool(arguments.get("recursive", False))
        max_depth = int(arguments.get("max_depth", 2))
        entries = []

        def append(item: Path) -> None:
            if len(entries) >= 500 or item.is_symlink():
                return
            stat = item.stat()
            entries.append(
                {
                    "name": item.name,
                    "relative_path": item.relative_to(directory).as_posix(),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size,
                }
            )

        if recursive:
            for current, directories, files in os.walk(
                str(directory),
                followlinks=False,
            ):
                current_path = Path(current)
                depth = len(current_path.relative_to(directory).parts)
                directories[:] = sorted(
                    [
                        name
                        for name in directories
                        if not (current_path / name).is_symlink()
                    ],
                    key=str.casefold,
                )
                if depth >= max_depth:
                    directories[:] = []
                for name in directories:
                    append(current_path / name)
                for name in sorted(files, key=str.casefold):
                    append(current_path / name)
                if len(entries) >= 500:
                    break
        else:
            for item in sorted(
                directory.iterdir(),
                key=lambda entry: entry.name.casefold(),
            )[:500]:
                append(item)
        return {
            "ok": True,
            "path": str(directory),
            "recursive": recursive,
            "max_depth": max_depth if recursive else 0,
            "truncated": len(entries) >= 500,
            "entries": entries,
        }

    def _resolve_project(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.projects.resolve(
            query=arguments.get("query"),
            path_hint=arguments.get("path_hint"),
            require_unique=arguments.get("require_unique", True),
        )

    def _find_project_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.projects.find_files(
            project_id=arguments["project_id"],
            query=arguments["query"],
            file_types=arguments.get("file_types"),
            max_results=arguments.get("max_results", 20),
        )

    def _get_project_git_state(self, arguments: dict[str, Any]) -> dict[str, Any]:
        project = self.policy.resolve(arguments["project_path"])
        if not project.is_dir():
            raise NotADirectoryError(project)

        def git(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "-C", str(project), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )

        branch_result = git("branch", "--show-current")
        if branch_result.returncode != 0:
            return {
                "ok": False,
                "error": "NOT_A_GIT_REPOSITORY",
                "message": branch_result.stderr.strip()[:1000],
                "path": str(project),
                "returncode": branch_result.returncode,
            }
        status_result = git("status", "--porcelain=v1")
        diff_result = git("diff", "--stat")
        status_lines = tuple(
            line for line in status_result.stdout.splitlines() if line.strip()
        )
        return {
            "ok": status_result.returncode == 0 and diff_result.returncode == 0,
            "path": str(project),
            "branch": branch_result.stdout.strip() or None,
            "working_tree": "clean" if not status_lines else "dirty",
            "changed_files": len(status_lines),
            "status": list(status_lines[:200]),
            "diff_stat": diff_result.stdout[-8000:],
        }

    def _run_project_tests(self, arguments: dict[str, Any]) -> dict[str, Any]:
        project = self.policy.resolve(arguments["project_path"])
        if not project.is_dir():
            raise NotADirectoryError(project)
        target = arguments.get("target")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        if target:
            target_value = str(target)
            path_value = target_value.split("::", 1)[0]
            raw_target = Path(path_value)
            if raw_target.is_absolute() or ".." in raw_target.parts:
                raise ValueError("test target must be relative to project")
            resolved_target = self.policy.resolve(str(project / raw_target))
            if project != resolved_target and project not in resolved_target.parents:
                raise PermissionError("test target is outside project")
            command.append(target_value)
        timeout_seconds = int(arguments.get("timeout_seconds", 120))
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=str(project),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "error": "TEST_TIMEOUT",
                "message": f"pytest exceeded {timeout_seconds}s",
                "path": str(project),
                "stdout": str(exc.stdout or "")[-self.max_output_bytes :],
                "stderr": str(exc.stderr or "")[-self.max_output_bytes :],
            }
        output = (completed.stdout or "")[-self.max_output_bytes :]
        errors = (completed.stderr or "")[-self.max_output_bytes :]
        return {
            "ok": completed.returncode == 0,
            "path": str(project),
            "target": str(target) if target else None,
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "output": output,
            "stderr": errors,
            "command": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *( [str(target)] if target else [])],
        }

    def _read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        file_path = self.policy.resolve(arguments["path"])
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        limit = min(arguments["max_bytes"], self.max_output_bytes)
        data = file_path.read_bytes()
        if len(data) > limit:
            raise ValueError(f"arquivo excede limite de {limit} bytes")
        resolution = self.projects.resolve(path_hint=str(file_path))
        if resolution.get("ok"):
            relative = file_path.relative_to(Path(str(resolution["root"]))).as_posix()
            self.projects.note_file(str(resolution["project_id"]), relative)
        return {"ok": True, "path": str(file_path), "content": data.decode("utf-8")}

    def _write(self, arguments: dict[str, Any]) -> dict[str, Any]:
        destination = self.policy.child(arguments["directory"], arguments["name"])
        destination.write_text(arguments["content"], encoding="utf-8")
        return {"ok": True, "path": str(destination), "bytes": destination.stat().st_size}

    def _delete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target = self.policy.resolve(arguments["path"])
        if not target.is_file():
            raise ValueError("somente arquivo individual pode ser apagado")
        size = target.stat().st_size
        target.unlink()
        return {"ok": True, "path": str(target), "bytes_deleted": size}

    @staticmethod
    def _codex_sensitive_action(task: str) -> str | None:
        normalized = task.casefold()
        normalized = re.sub(
            r"\b(?:nao|não|never|do not)\s+"
            r"(?:alterar|apagar|deletar|sobrescrever|instalar|desinstalar|"
            r"modificar|editar|modify|delete|install|uninstall)\b",
            "",
            normalized,
        )
        for action, pattern in _CODEX_SENSITIVE_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return action
        return None

    def _delegate_to_codex(self, arguments: dict[str, Any]) -> dict[str, Any]:
        event_callback = getattr(self._execution, "event_callback", None)
        return self.codex.delegate_to_codex(
            task=arguments["task"],
            project_path=arguments["project_path"],
            continue_current_thread=arguments.get(
                "continue_current_thread", True
            ),
            thread_id=arguments.get("thread_id"),
            focused_thread_id=(
                arguments.get("_focused_codex_thread_id") or None
            ),
            conversation_id=arguments.get("_conversation_id") or None,
            wait=arguments.get("wait", True),
            origin="qwen",
            execution_mode=arguments.get("_execution_mode") or None,
            event_callback=event_callback,
        ).as_dict()

    def _get_hardware_telemetry(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        return self.hardware.read()

    def _list_installed_applications(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.applications.list(
            query=arguments.get("query"),
            limit=arguments.get("limit", 50),
        )

    def _open_application(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.applications.open(arguments["application_name"])

    def _schedule_application(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.applications.schedule(
            arguments["application_name"],
            start_at=arguments["start_at"],
            recurrence=arguments["recurrence"],
        )

    def _delegate_to_deepseek(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self.deepseek is not None
        callback = getattr(self._execution, "event_callback", None)
        if callback is not None:
            callback("deepseek_sending", {})
        result = self.deepseek.delegate(
            arguments["task"],
            project_path=arguments.get("project_path"),
            continue_current_session=arguments.get("continue_current_session", True),
            context=arguments.get("context"),
            source="qwen",
        )
        if callback is not None:
            callback("deepseek_completed" if result.get("ok") else "deepseek_failed", result)
        return result

    def _review_deepseek_session(self, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self.deepseek is not None
        return self.deepseek.review_session(
            project_path=arguments.get("project_path"),
            turn_limit=arguments.get("turn_limit", 10),
        )

    def _get_codex_job_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.codex.get_job_status(
            job_id=arguments.get("job_id"),
            latest=arguments.get("latest", False),
        )
        callback = getattr(self._execution, "event_callback", None)
        if callback is not None and result.get("ok"):
            status = str(result.get("status") or "")
            callback(
                "codex_job_status",
                {
                    "job_id": result.get("job_id"),
                    "status": status,
                    "notify": (
                        status in {"running", "starting", "queued", "reconnecting"}
                        and self.codex.jobs.mark_progress_notified(str(result.get("job_id")))
                    ),
                },
            )
        return result

    def _cancel_codex_job(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.codex.cancel_job(
            job_id=arguments.get("job_id"),
            latest=arguments.get("latest", False),
        )
        callback = getattr(self._execution, "event_callback", None)
        if callback is not None and result.get("ok"):
            callback("codex_job_interrupted", result)
        return result

    def _steer_codex_job(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.codex.steer_job(
            arguments["instruction"],
            job_id=arguments.get("job_id"),
            latest=arguments.get("latest", False),
        )
        callback = getattr(self._execution, "event_callback", None)
        if callback is not None and result.get("ok"):
            callback("codex_job_steered", result)
        return result

    def _review_codex_session(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.codex.review_session(
            project_path=arguments.get(
                "project_path",
                str(Path(__file__).resolve().parents[2]),
            ),
            turn_limit=arguments.get("turn_limit", 10),
        )

    def _web_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.web.search(
            query=arguments["query"],
            max_results=arguments.get("max_results", 8),
            language=arguments.get("language", "pt-BR"),
            freshness_days=arguments.get("freshness_days"),
            allowed_domains=arguments.get("allowed_domains", []),
            blocked_domains=arguments.get("blocked_domains", []),
        )

    def _web_open(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.web.open(
            url=arguments["url"],
            max_chars=arguments.get("max_chars", 32768),
            page_start=arguments.get("page_start"),
            page_end=arguments.get("page_end"),
        )

    def _web_open_browser(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = str(arguments["url"])
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.casefold() != "file" or not arguments.get("local_artifact"):
            return self.web.open_in_browser(url=url)
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise WebError("URI de arquivo local invalida")
        raw_path = urllib.parse.unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:/", raw_path):
            raw_path = raw_path[1:]
        target = self.policy.resolve(raw_path)
        if not target.is_file() or target.suffix.casefold() not in {".html", ".htm"}:
            raise WebError("somente arquivos HTML locais permitidos podem ser abertos")
        file_url = target.as_uri()
        if not self.web.browser_opener(file_url):
            raise WebError("o navegador padrao recusou a abertura do arquivo local")
        return {
            "ok": True,
            "url": file_url,
            "title": target.stem,
            "browser_opened": True,
            "local_file": str(target),
            "threat_checked": False,
            "citation": None,
        }

    def _web_extract(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.web.extract(
            url=arguments["url"],
            query=arguments["query"],
            max_passages=arguments.get("max_passages", 5),
            passage_chars=arguments.get("passage_chars", 1200),
            page_start=arguments.get("page_start"),
            page_end=arguments.get("page_end"),
        )
