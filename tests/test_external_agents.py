"""Testes da descoberta de agentes de IA e da delegação adaptativa."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tern.orchestrator.autonomy_foundation import Capability
from tern.orchestrator.decision_policy import (
    AgentDecisionPolicy,
    Intent,
    SideEffect,
    is_agent_discovery_request,
    tool_effect,
    tool_specs_for_decision,
)
from tern.orchestrator.external_agents import (
    AgentAvailability,
    AgentDiscovery,
    ExternalAgentRunner,
    ExternalAgentSpec,
)
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry


def spec(**overrides) -> ExternalAgentSpec:
    base = dict(
        id="fake",
        display_name="Fake Agent",
        aliases=("fake", "fake-cli"),
        executables=("fake-cli",),
        prompt_template=("run", "{task}"),
        capabilities=frozenset({Capability.CODE_EDIT}),
    )
    base.update(overrides)
    return ExternalAgentSpec(**base)


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class FakeRunner:
    """Simula tasklist, --version e execução do agente."""

    def __init__(self, *, processos: str = "", versao: str = "fake 1.2.3", exit_code: int = 0):
        self.processos = processos
        self.versao = versao
        self.exit_code = exit_code
        self.chamadas: list[dict[str, object]] = []

    def __call__(self, args, **kwargs):
        self.chamadas.append({"args": list(args), **kwargs})
        if args[0] == "tasklist":
            return completed(self.processos)
        if len(args) > 1 and args[1] == "--version":
            return completed(self.versao)
        return completed("tarefa concluida", self.exit_code)


@pytest.fixture()
def executavel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binario = tmp_path / "fake-cli.exe"
    binario.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "tern.orchestrator.external_agents.shutil.which",
        lambda nome: str(binario) if nome == "fake-cli" else None,
    )
    return binario


def test_agent_installed_is_usable_and_reports_version(executavel: Path):
    discovery = AgentDiscovery(specs=(spec(),), runner=FakeRunner(), environment={})

    agente = discovery.discover()[0]

    assert agente.availability is AgentAvailability.INSTALLED
    assert agente.usable is True
    assert agente.version == "fake 1.2.3"
    assert agente.delegation_tool == "delegate_to_fake"


def test_running_process_is_detected_as_active_session(executavel: Path):
    runner = FakeRunner(processos='"fake-cli.exe","1234","Console"\n')
    discovery = AgentDiscovery(
        specs=(spec(process_names=("fake-cli",)),),
        runner=runner,
        environment={},
    )

    agente = discovery.discover()[0]

    assert agente.availability is AgentAvailability.SESSION_ACTIVE
    assert agente.usable is True
    assert "processo em execução" in agente.evidence


def test_configured_without_executable_is_not_usable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("tern.orchestrator.external_agents.shutil.which", lambda _nome: None)
    configuracao = tmp_path / ".fake"
    configuracao.mkdir()
    discovery = AgentDiscovery(
        specs=(spec(config_paths=(configuracao,)),),
        runner=FakeRunner(),
        environment={},
    )

    agente = discovery.discover()[0]

    assert agente.availability is AgentAvailability.CONFIGURED_NOT_INSTALLED
    assert agente.usable is False
    assert agente.as_dict()["delegation_tool"] is None


def test_api_key_only_is_not_usable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("tern.orchestrator.external_agents.shutil.which", lambda _nome: None)
    discovery = AgentDiscovery(
        specs=(spec(api_key_env=("FAKE_API_KEY",)),),
        runner=FakeRunner(),
        environment={"FAKE_API_KEY": "abc"},
    )

    assert discovery.discover()[0].availability is AgentAvailability.API_KEY_ONLY
    assert discovery.usable() == ()


def test_discovery_cache_respects_ttl(executavel: Path):
    runner = FakeRunner()
    tempo = {"agora": 0.0}
    discovery = AgentDiscovery(
        specs=(spec(),),
        runner=runner,
        environment={},
        ttl_seconds=10.0,
        clock=lambda: tempo["agora"],
    )

    discovery.discover()
    discovery.discover()
    versoes = [c for c in runner.chamadas if "--version" in c["args"]]
    assert len(versoes) == 1

    tempo["agora"] = 11.0
    discovery.discover()
    versoes = [c for c in runner.chamadas if "--version" in c["args"]]
    assert len(versoes) == 2


def test_alias_resolution_accepts_user_wording(executavel: Path):
    discovery = AgentDiscovery(
        specs=(spec(aliases=("glm", "glm5.2", "zhipu"), executables=("fake-cli",)),),
        runner=FakeRunner(),
        environment={},
    )

    assert discovery.find("glm5.2") is not None
    assert discovery.find("ZHIPU") is not None
    assert discovery.find("outro-agente") is None


def test_runner_executes_inside_allowlist(tmp_path: Path, executavel: Path):
    runner = FakeRunner()
    discovery = AgentDiscovery(specs=(spec(),), runner=runner, environment={})
    execucao = ExternalAgentRunner(
        policy=PathPolicy((tmp_path,)),
        discovery=discovery,
        logger=ActionLogger(tmp_path / "actions.jsonl"),
        runner=runner,
    )

    resultado = execucao.run("fake", "revisar arquivo", project_path=str(tmp_path))

    assert resultado["ok"] is True
    assert resultado["output"] == "tarefa concluida"
    ultima = runner.chamadas[-1]
    assert ultima["args"][1:] == ["run", "revisar arquivo"]
    assert ultima["cwd"] == str(tmp_path.resolve())


def test_external_tool_receives_preserved_delegation_request(
    tmp_path: Path,
    executavel: Path,
):
    runner = FakeRunner()
    discovery = AgentDiscovery(specs=(spec(),), runner=runner, environment={})

    class UnusedCodex:
        timeout = 1

    tools = ToolRegistry(
        policy=PathPolicy((tmp_path,)),
        logger=ActionLogger(tmp_path / "actions.jsonl"),
        codex=UnusedCodex(),
        max_output_bytes=131072,
        agent_discovery=discovery,
    )
    tools.external_agents._runner = runner
    original = "Revise somente auth.py; não edite arquivos."

    result = tools.execute(
        "delegate_to_fake",
        {"task": "edite tudo", "project_path": str(tmp_path)},
        context={
            "original_user_text": original,
            "delegation_constraints": ["read_only"],
            "turn_id": "turn-preservation",
        },
    )

    assert result["ok"], result
    payload = json.loads(runner.chamadas[-1]["args"][-1])
    assert payload["schema"] == "jarvis.delegation_request.v1"
    assert payload["requested_agent"] == "fake"
    assert payload["task"] == original
    assert payload["constraints"] == ["read_only"]
    assert "edite tudo" not in runner.chamadas[-1]["args"][-1]


def test_runner_blocks_path_outside_allowlist(tmp_path: Path, executavel: Path):
    runner = FakeRunner()
    discovery = AgentDiscovery(specs=(spec(),), runner=runner, environment={})
    execucao = ExternalAgentRunner(
        policy=PathPolicy((tmp_path / "permitido",)),
        discovery=discovery,
        runner=runner,
    )
    (tmp_path / "permitido").mkdir()
    (tmp_path / "fora").mkdir()

    resultado = execucao.run("fake", "tarefa", project_path=str(tmp_path / "fora"))

    assert resultado["ok"] is False
    assert resultado["error"] == "access_denied"


def test_runner_reports_unavailable_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("tern.orchestrator.external_agents.shutil.which", lambda _nome: None)
    discovery = AgentDiscovery(specs=(spec(),), runner=FakeRunner(), environment={})
    execucao = ExternalAgentRunner(policy=PathPolicy((tmp_path,)), discovery=discovery)

    resultado = execucao.run("fake", "tarefa")

    assert resultado["ok"] is False
    assert resultado["error"] == "agent_unavailable"


def test_runner_audits_delegation(tmp_path: Path, executavel: Path):
    logger = ActionLogger(tmp_path / "actions.jsonl")
    runner = FakeRunner()
    discovery = AgentDiscovery(specs=(spec(),), runner=runner, environment={})
    execucao = ExternalAgentRunner(
        policy=PathPolicy((tmp_path,)),
        discovery=discovery,
        logger=logger,
        runner=runner,
    )

    execucao.run("fake", "tarefa auditada", project_path=str(tmp_path))

    registro = (tmp_path / "actions.jsonl").read_text(encoding="utf-8")
    assert "external_agent_delegation" in registro
    assert "tarefa auditada" in registro


def test_output_is_cleaned_from_ansi_sequences(tmp_path: Path, executavel: Path):
    """CLIs coloridas devolvem escape ANSI; o texto entregue ao modelo deve ser limpo."""

    class ColoredRunner(FakeRunner):
        def __call__(self, args, **kwargs):
            if args[0] == "tasklist" or (len(args) > 1 and args[1] == "--version"):
                return super().__call__(args, **kwargs)
            self.chamadas.append({"args": list(args), **kwargs})
            return completed("\x1b[38;5;141m> \x1b[0mPONG\x1b[?25l\r\n")

    runner = ColoredRunner()
    discovery = AgentDiscovery(specs=(spec(),), runner=runner, environment={})
    execucao = ExternalAgentRunner(
        policy=PathPolicy((tmp_path,)),
        discovery=discovery,
        runner=runner,
    )

    resultado = execucao.run("fake", "ping", project_path=str(tmp_path))

    assert resultado["output"] == "> PONG"
    assert "\x1b" not in resultado["output"]


def test_delegation_tool_effect_defaults_to_code_execution():
    assert tool_effect("delegate_to_kiro") is SideEffect.CODE_EXECUTION
    assert tool_effect("list_available_agents") is SideEffect.READ_ONLY
    assert tool_effect("ferramenta_inexistente") is None


@pytest.mark.parametrize(
    "texto",
    [
        "quais agentes de IA voce tem disponiveis?",
        "liste as IAs instaladas",
        "quais modelos estao disponiveis",
    ],
)
def test_agent_discovery_questions_are_recognized(texto: str):
    assert is_agent_discovery_request(texto) is True


def test_agent_discovery_does_not_catch_unrelated_text():
    assert is_agent_discovery_request("corrija o bug do firebase") is False


class FakeAgentRegistry:
    """Registro mínimo para exercitar o roteamento sem subir o orquestrador."""

    def __init__(self, discovery: AgentDiscovery, ferramentas: tuple[str, ...]):
        self.agent_discovery = discovery
        self._nomes = ferramentas
        self.logger = None

    def names(self) -> tuple[str, ...]:
        return self._nomes

    def specs(self) -> list[dict[str, object]]:
        return [
            {"type": "function", "function": {"name": nome, "description": "", "parameters": {}}}
            for nome in self._nomes
        ]


def routing_policy(executavel_disponivel: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    binario = tmp_path / "fake-cli.exe"
    binario.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "tern.orchestrator.external_agents.shutil.which",
        lambda nome: str(binario) if (executavel_disponivel and nome == "fake-cli") else None,
    )
    discovery = AgentDiscovery(
        specs=(spec(aliases=("fake", "fake-cli"), config_paths=(tmp_path,)),),
        runner=FakeRunner(),
        environment={},
    )
    ferramentas = ("list_available_agents",)
    if executavel_disponivel:
        ferramentas += ("delegate_to_fake",)
    registry = FakeAgentRegistry(discovery, ferramentas)
    return AgentDecisionPolicy(tools=registry, context_cache_enabled=False), registry


def test_explicit_order_routes_to_agent_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    policy, registry = routing_policy(True, monkeypatch, tmp_path)

    decisao = policy.decide(
        "delegue uma tarefa para o fake-cli: listar arquivos", fixture_context={}
    )

    assert decisao.intent is Intent.EXTERNAL_AGENT_DELEGATE
    assert decisao.tools == ("delegate_to_fake",)
    assert decisao.constraint_violation is None
    expostas = [
        item["function"]["name"] for item in tool_specs_for_decision(registry.specs(), decisao)
    ]
    assert expostas == ["delegate_to_fake"]


def test_capability_question_routes_to_inventory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    policy, _ = routing_policy(True, monkeypatch, tmp_path)

    decisao = policy.decide("consegue delegar tarefa para o fake-cli?", fixture_context={})

    assert decisao.intent is Intent.AGENT_DISCOVERY
    assert decisao.tools == ("list_available_agents",)


def test_absent_agent_falls_back_to_inventory_instead_of_promise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    policy, _ = routing_policy(False, monkeypatch, tmp_path)

    decisao = policy.decide("delegue para o fake-cli corrigir o bug", fixture_context={})

    assert decisao.intent is Intent.AGENT_DISCOVERY
    assert decisao.tools == ("list_available_agents",)
