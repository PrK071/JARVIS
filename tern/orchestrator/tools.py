from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .codex import CodexRunner
from .schema import SchemaError, validate
from .security import ActionLogger, ApprovalCallback, PathPolicy, require_approval
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
        web: WebClient | None = None,
    ):
        self.policy = policy
        self.logger = logger
        self.codex = codex
        self.max_output_bytes = max_output_bytes
        self.approval = approval
        self.web = web or WebClient(WebConfig(enabled=False))
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    def specs(self) -> list[dict[str, Any]]:
        return [tool.openai() for tool in self._tools.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": "unknown_tool", "message": f"ferramenta inexistente: {name}"}
        try:
            validate(arguments, tool.schema)
            result = tool.handler(arguments)
            encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
            if len(encoded) > self.max_output_bytes:
                result = {
                    "ok": False,
                    "error": "output_too_large",
                    "message": f"retorno excedeu {self.max_output_bytes} bytes",
                }
        except SchemaError as exc:
            result = {"ok": False, "error": "invalid_arguments", "message": str(exc)}
        except WebError as exc:
            result = {
                "ok": False,
                "error": exc.code,
                "message": str(exc),
            }
            if exc.details:
                result["details"] = exc.details
        except TimeoutError as exc:
            result = {"ok": False, "error": "timeout", "message": str(exc)}
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        self.logger.write(tool=name, arguments=arguments, result=result)
        return result

    def _add(self, name: str, description: str, schema: dict[str, Any], handler: ToolHandler, timeout: int) -> None:
        self._tools[name] = Tool(name, description, schema, handler, timeout)

    def _register_defaults(self) -> None:
        path = {"type": "string", "minLength": 1, "maxLength": 4096}
        self._add(
            "filesystem_list",
            "Lista uma pasta permitida. Retorna nomes, tipos e tamanhos; no maximo 500 entradas.",
            _object({"path": path}, ["path"]),
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
            "codex_delegate",
            "Delega programacao ao Codex em sandbox workspace-write.",
            _object(
                {
                    "working_directory": path,
                    "task": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "context": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                    "constraints": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                    "validation": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                },
                [
                    "working_directory",
                    "task",
                    "context",
                    "constraints",
                    "acceptance_criteria",
                    "validation",
                ],
            ),
            self._codex_delegate,
            self.codex.timeout,
        )
        self._add(
            "codex_continue",
            "Continua uma sessao Codex existente para correcao ou validacao adicional.",
            _object(
                {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 200},
                    "working_directory": path,
                    "task": {"type": "string", "minLength": 1, "maxLength": 20000},
                },
                ["session_id", "working_directory", "task"],
            ),
            self._codex_continue,
            self.codex.timeout,
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
        entries = []
        for item in sorted(directory.iterdir(), key=lambda entry: entry.name.lower())[:500]:
            stat = item.stat()
            entries.append({"name": item.name, "type": "directory" if item.is_dir() else "file", "size": stat.st_size})
        return {"ok": True, "path": str(directory), "entries": entries}

    def _read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        file_path = self.policy.resolve(arguments["path"])
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        limit = min(arguments["max_bytes"], self.max_output_bytes)
        data = file_path.read_bytes()
        if len(data) > limit:
            raise ValueError(f"arquivo excede limite de {limit} bytes")
        return {"ok": True, "path": str(file_path), "content": data.decode("utf-8")}

    def _write(self, arguments: dict[str, Any]) -> dict[str, Any]:
        destination = self.policy.child(arguments["directory"], arguments["name"])
        if destination.exists():
            require_approval(self.approval, "overwrite", {"path": str(destination)})
        destination.write_text(arguments["content"], encoding="utf-8")
        return {"ok": True, "path": str(destination), "bytes": destination.stat().st_size}

    def _delete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target = self.policy.resolve(arguments["path"])
        if not target.is_file():
            raise ValueError("somente arquivo individual pode ser apagado")
        require_approval(self.approval, "delete", {"path": str(target)})
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

    def _codex_delegate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = self._codex_sensitive_action(arguments["task"])
        if action:
            require_approval(
                self.approval,
                action,
                {"path": arguments["working_directory"]},
            )
        return self.codex.delegate(arguments).as_dict()

    def _codex_continue(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = self._codex_sensitive_action(arguments["task"])
        if action:
            require_approval(
                self.approval,
                action,
                {"path": arguments["working_directory"]},
            )
        return self.codex.continue_session(**arguments).as_dict()

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

    def _web_extract(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.web.extract(
            url=arguments["url"],
            query=arguments["query"],
            max_passages=arguments.get("max_passages", 5),
            passage_chars=arguments.get("passage_chars", 1200),
            page_start=arguments.get("page_start"),
            page_end=arguments.get("page_end"),
        )
