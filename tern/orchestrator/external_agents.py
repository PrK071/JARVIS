"""Descoberta e execução de agentes de IA externos instalados na máquina.

O orquestrador não deve ter agente cravado no código. Aqui cada agente é descrito
declarativamente e a disponibilidade é *medida* no ambiente: executável no PATH,
instalação em local conhecido, sessão em execução, diretório de configuração ou
chave de API. As ferramentas de delegação são registradas somente para os agentes
realmente disponíveis, então o modelo nunca vê — nem promete — um agente ausente.

Nada aqui muta arquivo do usuário. Execução acontece só via ExternalAgentRunner,
sempre com cwd validado pela allowlist e com registro em auditoria.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .autonomy_foundation import Capability
from .security import AccessDenied, ActionLogger, PathPolicy


class AgentAvailability(str, Enum):
    """Estado observado, não suposto."""

    SESSION_ACTIVE = "session_active"
    INSTALLED = "installed"
    CONFIGURED_NOT_INSTALLED = "configured_not_installed"
    API_KEY_ONLY = "api_key_only"
    ABSENT = "absent"


USABLE_STATES = frozenset(
    {
        AgentAvailability.SESSION_ACTIVE,
        AgentAvailability.INSTALLED,
    }
)

_LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
_PROGRAM_FILES = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
_HOME = Path.home()


@dataclass(frozen=True)
class ExternalAgentSpec:
    """Descrição declarativa de um agente externo."""

    id: str
    display_name: str
    aliases: tuple[str, ...]
    executables: tuple[str, ...]
    extra_paths: tuple[Path, ...] = ()
    process_names: tuple[str, ...] = ()
    config_paths: tuple[Path, ...] = ()
    api_key_env: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)
    prompt_template: tuple[str, ...] = ("{task}",)
    capabilities: frozenset[Capability] = frozenset()
    native_integration: bool = False
    notes: str = ""

    def command_for(self, task: str) -> list[str]:
        return [part.format(task=task) for part in self.prompt_template]


@dataclass(frozen=True)
class DiscoveredAgent:
    spec: ExternalAgentSpec
    availability: AgentAvailability
    executable: str | None
    version: str | None
    evidence: tuple[str, ...]
    detected_at: float

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def usable(self) -> bool:
        return self.availability in USABLE_STATES and bool(self.executable)

    @property
    def delegation_tool(self) -> str:
        return f"delegate_to_{self.spec.id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "name": self.spec.display_name,
            "availability": self.availability.value,
            "usable": self.usable,
            "executable": self.executable,
            "version": self.version,
            "capabilities": sorted(capability.value for capability in self.spec.capabilities),
            "native_integration": self.spec.native_integration,
            "delegation_tool": self.delegation_tool if self.usable else None,
            "evidence": list(self.evidence),
            "notes": self.spec.notes,
        }


_CODE_CAPABILITIES = frozenset(
    {
        Capability.CODE_ANALYSIS,
        Capability.CODE_EDIT,
        Capability.CODE_REVIEW,
        Capability.FILESYSTEM_READ,
        Capability.FILESYSTEM_WRITE,
        Capability.REPOSITORY_READ,
        Capability.REPOSITORY_WRITE,
        Capability.TEST_EXECUTION,
        Capability.GENERAL_REASONING,
        Capability.MUTATION,
    }
)

_CONSULTANT_CAPABILITIES = frozenset(
    {
        Capability.CODE_ANALYSIS,
        Capability.CODE_REVIEW,
        Capability.GENERAL_REASONING,
        Capability.READ_ONLY,
    }
)


KNOWN_AGENTS: tuple[ExternalAgentSpec, ...] = (
    ExternalAgentSpec(
        id="kiro",
        display_name="Kiro CLI",
        aliases=("kiro", "kiro-cli", "kiro cli"),
        executables=("kiro-cli", "kiro"),
        extra_paths=(_LOCAL_APP_DATA / "Kiro-Cli" / "kiro-cli.exe",),
        process_names=("kiro-cli", "kiro"),
        config_paths=(_HOME / ".kiro",),
        prompt_template=("chat", "--no-interactive", "{task}"),
        capabilities=_CODE_CAPABILITIES,
        notes="Agente de terminal com ferramentas de arquivo e shell.",
    ),
    ExternalAgentSpec(
        id="claude",
        display_name="Claude Code",
        aliases=("claude", "claude code", "claude-code"),
        executables=("claude",),
        extra_paths=(
            _LOCAL_APP_DATA / "Programs" / "claude" / "claude.exe",
            _HOME / ".local" / "bin" / "claude",
        ),
        process_names=("claude",),
        config_paths=(_HOME / ".claude",),
        api_key_env=("ANTHROPIC_API_KEY",),
        prompt_template=("-p", "{task}"),
        capabilities=_CODE_CAPABILITIES,
        notes="Agente de código da Anthropic.",
    ),
    ExternalAgentSpec(
        id="gemini",
        display_name="Gemini CLI",
        aliases=("gemini", "gemini cli", "gemini-cli"),
        executables=("gemini",),
        extra_paths=(_LOCAL_APP_DATA / "Programs" / "gemini" / "gemini.exe",),
        process_names=("gemini",),
        config_paths=(_HOME / ".gemini",),
        api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        prompt_template=("-p", "{task}"),
        capabilities=_CODE_CAPABILITIES,
        notes="Agente de código do Google.",
    ),
    ExternalAgentSpec(
        id="glm",
        display_name="GLM CLI",
        aliases=("glm", "glm5", "glm 5", "glm5.2", "glm-5.2", "zhipu", "chatglm"),
        executables=("glm", "glm-cli", "zhipu"),
        process_names=("glm", "glm-cli"),
        config_paths=(_HOME / ".glm", _HOME / ".zhipu"),
        api_key_env=("GLM_API_KEY", "ZHIPU_API_KEY", "ZHIPUAI_API_KEY"),
        prompt_template=("{task}",),
        capabilities=_CONSULTANT_CAPABILITIES,
        notes="Agente GLM/Zhipu.",
    ),
    ExternalAgentSpec(
        id="aider",
        display_name="Aider",
        aliases=("aider",),
        executables=("aider",),
        process_names=("aider",),
        config_paths=(_HOME / ".aider",),
        prompt_template=("--message", "{task}", "--yes"),
        capabilities=_CODE_CAPABILITIES,
        notes="Agente de edição de código em terminal.",
    ),
    ExternalAgentSpec(
        id="codex",
        display_name="Codex",
        aliases=("codex", "codex cli"),
        executables=("codex",),
        extra_paths=(_PROGRAM_FILES / "OpenAI" / "Codex" / "bin" / "codex.exe",),
        process_names=("codex",),
        config_paths=(_HOME / ".codex",),
        capabilities=_CODE_CAPABILITIES,
        native_integration=True,
        notes="Já integrado pela ponte nativa; use delegate_to_codex.",
    ),
)


def _running_process_names(runner: Callable[..., subprocess.CompletedProcess[str]]) -> frozenset[str]:
    """Nomes de processos ativos, em minúsculas e sem extensão."""
    if os.name != "nt":
        return frozenset()
    try:
        completed = runner(
            ["tasklist", "/fo", "csv", "/nh"],
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:  # noqa: BLE001 - ausência de tasklist não invalida a descoberta
        return frozenset()
    if completed.returncode != 0 or not completed.stdout:
        return frozenset()

    nomes: set[str] = set()
    for linha in completed.stdout.splitlines():
        if not linha.startswith('"'):
            continue
        nome = linha.split('","', 1)[0].strip('"')
        if nome:
            nomes.add(Path(nome).stem.lower())
    return frozenset(nomes)


class AgentDiscovery:
    """Mede quais agentes externos existem agora. Cacheado por TTL."""

    def __init__(
        self,
        *,
        specs: Sequence[ExternalAgentSpec] = KNOWN_AGENTS,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        environment: Mapping[str, str] | None = None,
        ttl_seconds: float = 60.0,
        version_timeout: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.specs = tuple(specs)
        self._runner = runner
        self._environment = environment if environment is not None else os.environ
        self.ttl_seconds = ttl_seconds
        self.version_timeout = version_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: tuple[DiscoveredAgent, ...] = ()
        self._cached_at: float | None = None

    def discover(self, *, force: bool = False) -> tuple[DiscoveredAgent, ...]:
        with self._lock:
            agora = self._clock()
            if (
                not force
                and self._cached_at is not None
                and agora - self._cached_at < self.ttl_seconds
            ):
                return self._cache

            processos = _running_process_names(self._runner)
            self._cache = tuple(self._inspect(spec, processos, agora) for spec in self.specs)
            self._cached_at = agora
            return self._cache

    def usable(self, *, force: bool = False) -> tuple[DiscoveredAgent, ...]:
        return tuple(agent for agent in self.discover(force=force) if agent.usable)

    def find(self, texto: str, *, force: bool = False) -> DiscoveredAgent | None:
        """Resolve menção do usuário ("kiro-cli", "glm5.2") para um agente."""
        normalizado = texto.strip().lower()
        if not normalizado:
            return None
        for agent in self.discover(force=force):
            if normalizado == agent.spec.id:
                return agent
            if any(alias == normalizado for alias in agent.spec.aliases):
                return agent
        for agent in self.discover(force=force):
            if any(alias in normalizado for alias in agent.spec.aliases):
                return agent
        return None

    def _inspect(
        self,
        spec: ExternalAgentSpec,
        processos: frozenset[str],
        agora: float,
    ) -> DiscoveredAgent:
        evidencias: list[str] = []
        executavel = self._locate(spec, evidencias)
        versao = self._version(executavel, spec, evidencias) if executavel else None

        sessao_ativa = any(nome.lower() in processos for nome in spec.process_names)
        if sessao_ativa:
            evidencias.append("processo em execução")

        configurado = [str(caminho) for caminho in spec.config_paths if caminho.exists()]
        evidencias.extend(f"configuração: {caminho}" for caminho in configurado)

        chaves = [chave for chave in spec.api_key_env if self._environment.get(chave, "").strip()]
        evidencias.extend(f"chave de API: {chave}" for chave in chaves)

        if executavel and sessao_ativa:
            disponibilidade = AgentAvailability.SESSION_ACTIVE
        elif executavel:
            disponibilidade = AgentAvailability.INSTALLED
        elif configurado:
            disponibilidade = AgentAvailability.CONFIGURED_NOT_INSTALLED
        elif chaves:
            disponibilidade = AgentAvailability.API_KEY_ONLY
        else:
            disponibilidade = AgentAvailability.ABSENT

        return DiscoveredAgent(
            spec=spec,
            availability=disponibilidade,
            executable=executavel,
            version=versao,
            evidence=tuple(evidencias),
            detected_at=agora,
        )

    def _locate(self, spec: ExternalAgentSpec, evidencias: list[str]) -> str | None:
        for nome in spec.executables:
            encontrado = shutil.which(nome)
            if encontrado:
                evidencias.append(f"PATH: {encontrado}")
                return encontrado
        for caminho in spec.extra_paths:
            if caminho.is_file():
                evidencias.append(f"instalação: {caminho}")
                return str(caminho)
        return None

    def _version(
        self,
        executavel: str,
        spec: ExternalAgentSpec,
        evidencias: list[str],
    ) -> str | None:
        if not spec.version_args:
            return None
        try:
            completed = self._runner(
                [executavel, *spec.version_args],
                capture_output=True,
                text=True,
                timeout=self.version_timeout,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as erro:  # noqa: BLE001 - versão é informativa
            evidencias.append(f"versão indisponível: {type(erro).__name__}")
            return None
        saida = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
        primeira = next((linha.strip() for linha in saida.splitlines() if linha.strip()), None)
        if primeira:
            evidencias.append(f"versão: {primeira[:120]}")
        return primeira[:120] if primeira else None


_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _clean_output(texto: str) -> str:
    """CLIs de agente escrevem cores e controle de cursor; o modelo só quer o texto."""
    return _ANSI_ESCAPE.sub("", texto).replace("\r", "").strip()


class ExternalAgentRunner:
    """Executa um agente externo como subprocesso, dentro da allowlist."""

    def __init__(
        self,
        *,
        policy: PathPolicy,
        discovery: AgentDiscovery,
        logger: ActionLogger | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        default_timeout: int = 900,
        max_output_chars: int = 20_000,
    ) -> None:
        self.policy = policy
        self.discovery = discovery
        self.logger = logger
        self._runner = runner
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars

    def run(
        self,
        agent_id: str,
        task: str,
        *,
        project_path: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        agent = self.discovery.find(agent_id)
        if agent is None:
            return {
                "ok": False,
                "error": "unknown_agent",
                "message": f"agente desconhecido: {agent_id}",
                "available": [item.spec.id for item in self.discovery.usable()],
            }
        if not agent.usable:
            return {
                "ok": False,
                "error": "agent_unavailable",
                "message": (
                    f"{agent.spec.display_name} não está utilizável agora "
                    f"(estado: {agent.availability.value})"
                ),
                "evidence": list(agent.evidence),
            }

        tarefa = task.strip()
        if not tarefa:
            return {"ok": False, "error": "empty_task", "message": "tarefa vazia"}

        try:
            diretorio = str(self.policy.resolve(project_path)) if project_path else None
        except AccessDenied as erro:
            return {"ok": False, "error": "access_denied", "message": str(erro)}

        comando = [agent.executable, *agent.spec.command_for(tarefa)]
        limite = timeout or self.default_timeout
        inicio = time.monotonic()

        try:
            completed = self._runner(
                comando,
                capture_output=True,
                text=True,
                timeout=limite,
                cwd=diretorio,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            resultado = {
                "ok": False,
                "error": "timeout",
                "message": f"{agent.spec.display_name} excedeu {limite}s",
                "agent": agent.spec.id,
            }
            self._audit(agent, tarefa, diretorio, resultado)
            return resultado
        except OSError as erro:
            resultado = {
                "ok": False,
                "error": "spawn_failed",
                "message": str(erro),
                "agent": agent.spec.id,
            }
            self._audit(agent, tarefa, diretorio, resultado)
            return resultado

        saida = _clean_output(completed.stdout or "")
        erros = _clean_output(completed.stderr or "")
        resultado = {
            "ok": completed.returncode == 0,
            "agent": agent.spec.id,
            "agent_name": agent.spec.display_name,
            "exit_code": completed.returncode,
            "duration_seconds": round(time.monotonic() - inicio, 1),
            "project_path": diretorio,
            "output": saida[: self.max_output_chars],
            "output_truncated": len(saida) > self.max_output_chars,
            "stderr": erros[:2000],
        }
        if completed.returncode != 0:
            resultado["error"] = "agent_failed"
        self._audit(agent, tarefa, diretorio, resultado)
        return resultado

    def _audit(
        self,
        agent: DiscoveredAgent,
        task: str,
        project_path: str | None,
        result: Mapping[str, Any],
    ) -> None:
        if self.logger is None:
            return
        self.logger.write_event(
            "external_agent_delegation",
            agent=agent.spec.id,
            executable=agent.executable,
            project=project_path,
            task=task[:400],
            ok=bool(result.get("ok")),
            exit_code=result.get("exit_code"),
            error=result.get("error"),
        )
