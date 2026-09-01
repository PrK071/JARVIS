from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.autonomy_foundation import (
    Agent,
    AgentCapabilityProfile,
    AgentRuntimeAvailability,
    Capability,
    CapabilityBaseline,
)
from tern.orchestrator.client import LlamaClient
from tern.orchestrator.config import load_settings
from tern.orchestrator.orchestration_contracts import OrchestrationBudget
from tern.orchestrator.orchestration_fast_path import ActionSpaceBuilder
from tern.orchestrator.orchestration_policy import (
    NextActionValidator,
    QwenOrchestrationPolicy,
)
from tern.orchestrator.orchestration_state import WorldStateBuilder
from tern.orchestrator.user_goal import UserGoalBuilder


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int(round((len(ordered) - 1) * fraction))], 3)


def policy_benchmark(endpoint: str, timeout: int, iterations: int) -> dict:
    profile = AgentCapabilityProfile(
        Agent.LOCAL,
        frozenset(
            {
                Capability.REPOSITORY_READ,
                Capability.FILESYSTEM_READ,
                Capability.TEST_EXECUTION,
                Capability.GENERAL_REASONING,
                Capability.READ_ONLY,
            }
        ),
        (),
    )
    baseline = CapabilityBaseline(
        {Agent.LOCAL: profile},
        {Agent.LOCAL: AgentRuntimeAvailability(Agent.LOCAL, True, True, True)},
    )
    goal = UserGoalBuilder().build(
        "investigue por que os testes de autenticação falham sem alterar arquivos"
    )
    tools = (
        "filesystem_read_text",
        "find_project_files",
        "get_project_git_state",
        "run_project_tests",
    )
    project_path = str(Path.cwd())
    tool_specs = (
        {
            "type": "function",
            "function": {
                "name": "filesystem_read_text",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "max_bytes": {"type": "integer"},
                    },
                    "required": ["path", "max_bytes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_project_files",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "project_id": {"type": "string"},
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["project_id", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_project_git_state",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"project_path": {"type": "string"}},
                    "required": ["project_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_project_tests",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "project_path": {"type": "string"},
                        "target": {"type": ["string", "null"]},
                        "timeout_seconds": {
                            "type": "integer",
                            "enum": [30, 60, 120, 300],
                        },
                    },
                    "required": ["project_path"],
                },
            },
        },
    )
    state = WorldStateBuilder().build(
        goal,
        baseline=baseline,
        project_snapshot={"project_path": project_path},
        tool_names=tools,
        budget=OrchestrationBudget(max_steps=4),
        authority_facts=("BOUNDED_LIVE", "EXECUTION_AUTHORITY_REQUIRED"),
    ).state
    allowed = ActionSpaceBuilder().build(goal, state, tool_specs=tool_specs)
    policy = QwenOrchestrationPolicy(
        LlamaClient(endpoint, timeout), mode="bounded_live"
    )
    validator = NextActionValidator()
    latencies: list[float] = []
    stats: list[dict] = []
    actions: list[str] = []
    decisions: list[dict] = []
    valid = 0
    errors: list[str] = []
    for _ in range(max(iterations, 1)):
        started = time.perf_counter()
        try:
            action = policy.decide(goal, state, allowed)
            latencies.append(round((time.perf_counter() - started) * 1000, 3))
            stats.append(policy.last_call_stats.as_dict())
            actions.append(action.action.value)
            validation = validator.validate(action, goal, state, allowed)
            valid += int(validation.valid)
            decisions.append(
                {
                    "action": action.as_dict(),
                    "validation": validation.as_dict(),
                }
            )
        except Exception as exc:  # benchmark must report, not hide, policy failures
            latencies.append(round((time.perf_counter() - started) * 1000, 3))
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "iterations": max(iterations, 1),
        "valid_action_rate": valid / max(iterations, 1),
        "actions": actions,
        "decisions": decisions,
        "errors": errors,
        "total_policy_ms": {
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
        },
        "input_tokens": [item.get("input_tokens") for item in stats],
        "output_tokens": [item.get("output_tokens") for item in stats],
        "context_size": [item.get("context_size") for item in stats],
        "prefill_ms": [item.get("prefill_ms") for item in stats],
        "generation_ms": [item.get("generation_ms") for item in stats],
        "model_calls": sum(int(item.get("model_calls") or 0) for item in stats),
    }


def live_read_only_smoke(settings, project: Path) -> dict:
    from tern.orchestrator.cli import _registry

    values = dict(os.environ)
    values.update(
        {
            "ORCHESTRATION_MODE": "bounded_live",
            "ORCHESTRATION_SHADOW_ENABLED": "false",
            "ORCHESTRATION_SHADOW_MAX_STEPS": "4",
            "ORCHESTRATION_MAX_TOOL_CALLS": "3",
            "ORCHESTRATION_MAX_DELEGATIONS": "1",
        }
    )
    live_settings = load_settings(values)
    resolved = project.resolve()
    if not any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in live_settings.allowed_roots
    ):
        raise ValueError("smoke project is outside configured allowed roots")
    prompt = (
        f"Inspecione o estado Git do projeto em {resolved} e responda com um resumo. "
        "Não altere nenhum arquivo, não execute mutações e não delegue mutação."
    )
    result = Supervisor(
        live_settings,
        LlamaClient(live_settings.base_url, live_settings.timeout),
        _registry(live_settings),
    ).run(prompt)
    orchestration = result.get("orchestration") or {}
    return {
        "ok": result.get("ok"),
        "answer": result.get("answer"),
        "mode": orchestration.get("mode"),
        "termination_reason": orchestration.get("termination_reason"),
        "actions": [
            item.get("next_action", {}).get("action")
            for item in orchestration.get("records", [])
        ],
        "tools": [
            item.get("observation", {}).get("tool_name")
            for item in orchestration.get("records", [])
            if item.get("observation", {}).get("tool_name")
        ],
        "statuses": [
            item.get("observation", {}).get("status")
            for item in orchestration.get("records", [])
        ],
        "telemetry": orchestration.get("telemetry"),
        "critical_violations": orchestration.get("critical_violations"),
    }


def live_repair_smoke(settings, project: Path) -> dict:
    """Run one isolated repair goal without selecting an agent for Qwen."""

    from tern.orchestrator.cli import _registry

    values = dict(os.environ)
    values.update(
        {
            "ORCHESTRATION_MODE": "bounded_live",
            "ORCHESTRATION_SHADOW_ENABLED": "false",
            "ORCHESTRATION_SHADOW_MAX_STEPS": "8",
            "ORCHESTRATION_MAX_TOOL_CALLS": "6",
            "ORCHESTRATION_MAX_DELEGATIONS": "2",
            "ORCHESTRATION_MAX_ELAPSED_SECONDS": "900",
        }
    )
    live_settings = load_settings(values)
    resolved = project.resolve()
    if not any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in live_settings.allowed_roots
    ):
        raise ValueError("repair project is outside configured allowed roots")
    prompt = (
        f"No projeto isolado em {resolved}, descubra por que o teste falha, corrija "
        "o defeito e verifique executando os testes. Você pode alterar somente arquivos "
        "dentro desse projeto. Não altere nenhum arquivo fora dele. Escolha autonomamente "
        "as ferramentas e agentes necessários."
    )
    result = Supervisor(
        live_settings,
        LlamaClient(live_settings.base_url, live_settings.timeout),
        _registry(live_settings),
    ).run(prompt)
    orchestration = result.get("orchestration") or {}
    verification = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=str(resolved),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    return {
        "ok": result.get("ok"),
        "answer": result.get("answer"),
        "mode": orchestration.get("mode"),
        "termination_reason": orchestration.get("termination_reason"),
        "actions": [
            item.get("next_action", {}).get("action")
            for item in orchestration.get("records", [])
        ],
        "tools": [
            item.get("observation", {}).get("tool_name")
            for item in orchestration.get("records", [])
            if item.get("observation", {}).get("tool_name")
        ],
        "agents": [
            item.get("observation", {}).get("agent")
            for item in orchestration.get("records", [])
            if item.get("observation", {}).get("agent")
        ],
        "authority_outcomes": [
            item.get("observation", {}).get("authority_outcome")
            for item in orchestration.get("records", [])
        ],
        "telemetry": orchestration.get("telemetry"),
        "critical_violations": orchestration.get("critical_violations"),
        "external_verification": {
            "passed": verification.returncode == 0,
            "returncode": verification.returncode,
            "output": verification.stdout[-2000:],
            "errors": verification.stderr[-1000:],
        },
    }


def _project_digest(project: Path) -> str:
    digest = hashlib.sha256()
    ignored = {".orchestrator", ".pytest_cache", "__pycache__"}
    for path in sorted(project.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        digest.update(path.relative_to(project).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def live_agent_read_only_smoke(settings, project: Path, agent: str) -> dict:
    """Exercise one real selected-agent delegation and prove no project mutation."""

    from tern.orchestrator.cli import _registry

    values = dict(os.environ)
    values.update(
        {
            "ORCHESTRATION_MODE": "bounded_live",
            "ORCHESTRATION_SHADOW_ENABLED": "false",
            "ORCHESTRATION_SHADOW_MAX_STEPS": "6",
            "ORCHESTRATION_MAX_TOOL_CALLS": "3",
            "ORCHESTRATION_MAX_DELEGATIONS": "1",
            "ORCHESTRATION_MAX_ELAPSED_SECONDS": "900",
        }
    )
    live_settings = load_settings(values)
    resolved = project.resolve()
    if not any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in live_settings.allowed_roots
    ):
        raise ValueError("agent smoke project is outside configured allowed roots")
    before = _project_digest(resolved)
    agent_request = (
        "mande para o Codex analisar a finalidade de calculator.py em modo somente leitura"
        if agent == "codex"
        else (
            "use o DeepSeek para analisar em modo somente leitura esta função Python: "
            "def add(left, right): return left + right"
        )
    )
    prompt = (
        f"No projeto {resolved}, {agent_request} e responda com um resumo curto. "
        "Não altere, crie nem remova arquivos."
    )
    result = Supervisor(
        live_settings,
        LlamaClient(live_settings.base_url, live_settings.timeout),
        _registry(live_settings),
    ).run(prompt)
    after = _project_digest(resolved)
    orchestration = result.get("orchestration") or {}
    records = orchestration.get("records", [])
    return {
        "ok": result.get("ok"),
        "answer": result.get("answer"),
        "mode": orchestration.get("mode"),
        "termination_reason": orchestration.get("termination_reason"),
        "actions": [item.get("next_action", {}).get("action") for item in records],
        "agents": [
            item.get("observation", {}).get("agent")
            for item in records
            if item.get("observation", {}).get("agent")
        ],
        "execution_modes": [
            item.get("next_action", {}).get("execution_mode")
            for item in records
            if item.get("next_action", {}).get("execution_mode")
        ],
        "project_unchanged": before == after,
        "telemetry": orchestration.get("telemetry"),
        "critical_violations": orchestration.get("critical_violations"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-policy-benchmark", action="store_true")
    parser.add_argument("--live-read-only", action="store_true")
    parser.add_argument("--live-repair", action="store_true")
    parser.add_argument("--live-codex-read-only", action="store_true")
    parser.add_argument("--live-deepseek-read-only", action="store_true")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()
    settings = load_settings()
    report = {}
    if not args.skip_policy_benchmark:
        report["policy_benchmark"] = policy_benchmark(
            args.endpoint or settings.base_url,
            args.timeout,
            args.iterations,
        )
    if args.live_read_only:
        report["live_read_only"] = live_read_only_smoke(settings, args.project)
    if args.live_repair:
        report["live_repair"] = live_repair_smoke(settings, args.project)
    if args.live_codex_read_only:
        report["live_codex_read_only"] = live_agent_read_only_smoke(
            settings, args.project, "codex"
        )
    if args.live_deepseek_read_only:
        report["live_deepseek_read_only"] = live_agent_read_only_smoke(
            settings, args.project, "deepseek"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
