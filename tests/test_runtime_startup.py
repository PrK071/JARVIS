from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import tern.orchestrator.runtime as runtime_module
from tern.orchestrator.client import ServerError
from tern.orchestrator.config import load_settings
from tern.orchestrator.runtime import (
    LlamaServerConfigurationMismatch,
    LlamaServerEndpointOccupied,
    RuntimeManager,
)


def configured(tmp_path: Path, *, model_name: str = "expected.gguf"):
    model = tmp_path / model_name
    executable = tmp_path / "llama-server.exe"
    model.write_bytes(b"model")
    executable.write_bytes(b"exe")
    return load_settings(
        {
            "MODEL_PATH": str(model),
            "MODEL_SERVER_EXECUTABLE": str(executable),
            "MODEL_STATE_DIR": str(tmp_path / "state"),
            "MODEL_SERVER_PORT": "18080",
            "MODEL_CONTEXT_SIZE": "16384",
            "MODEL_PARALLEL_SLOTS": "1",
        }
    )


def absent() -> dict:
    return {
        "running": False,
        "occupied": False,
        "healthy": False,
        "recognized": False,
        "compatible": False,
        "pid": None,
        "managed_by_jarvis": False,
        "state_stale": False,
    }


def compatible(*, pid: int = 41, managed: bool = True) -> dict:
    return {
        "running": True,
        "occupied": True,
        "healthy": True,
        "recognized": True,
        "compatible": True,
        "pid": pid,
        "model": "expected.gguf",
        "managed_by_jarvis": managed,
        "mismatches": [],
        "state_stale": False,
    }


def incompatible(*, managed: bool = True) -> dict:
    return {
        **compatible(managed=managed),
        "compatible": False,
        "model": "other.gguf",
        "mismatches": ["model"],
    }


def test_no_server_starts_one(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    monkeypatch.setattr(manager, "inspect_llama_server", absent)
    monkeypatch.setattr(
        manager,
        "_start_llama_server_unlocked",
        lambda _wait: {"started": True, "reused": False, "pid": 42},
    )
    assert manager.ensure_llama_server()["started"] is True


def test_compatible_server_is_reused(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    monkeypatch.setattr(manager, "inspect_llama_server", compatible)
    monkeypatch.setattr(
        manager,
        "_start_llama_server_unlocked",
        lambda _wait: pytest.fail("nao deve iniciar outro servidor"),
    )
    result = manager.ensure_llama_server()
    assert result["reused"] is True and result["started"] is False


def test_incompatible_server_does_not_switch_silently(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    monkeypatch.setattr(manager, "inspect_llama_server", incompatible)
    monkeypatch.setattr(
        manager,
        "switch_llama_model",
        lambda *_args, **_kwargs: pytest.fail("startup nao deve trocar modelo"),
    )
    with pytest.raises(LlamaServerConfigurationMismatch, match="running_model=.*other"):
        manager.ensure_llama_server()


def test_normal_startup_never_calls_switch_model(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    monkeypatch.setattr(manager, "inspect_llama_server", compatible)
    monkeypatch.setattr(
        manager,
        "switch_llama_model",
        lambda *_args, **_kwargs: pytest.fail("switch indevido"),
    )
    assert manager.start()["reused"] is True


def test_explicit_model_switch_stops_then_starts(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    events: list[str] = []
    monkeypatch.setattr(manager, "inspect_llama_server", incompatible)
    monkeypatch.setattr(
        manager,
        "_stop_unlocked",
        lambda **_kwargs: events.append("stop") or {"stopped": True},
    )
    monkeypatch.setattr(
        manager,
        "_start_llama_server_unlocked",
        lambda _wait: events.append("start") or {"started": True},
    )
    result = manager.switch_llama_model()
    assert events == ["stop", "start"] and result["switched"] is True


def test_unknown_process_occupying_port_is_clear_error(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    value = {**absent(), "running": True, "occupied": True, "pid": 77}
    monkeypatch.setattr(manager, "inspect_llama_server", lambda: value)
    with pytest.raises(LlamaServerEndpointOccupied, match="PID 77"):
        manager.ensure_llama_server()


class HealthyClient:
    def __init__(self, _base_url: str, timeout: int):
        self.timeout = timeout

    def health(self):
        return {"status": "ok"}

    def props(self):
        return {
            "model_path": self.model,
            "total_slots": 1,
            "default_generation_settings": {"n_ctx": 16384},
        }


def install_healthy_client(monkeypatch, model: Path):
    HealthyClient.model = str(model)
    monkeypatch.setattr(runtime_module, "LlamaClient", HealthyClient)


def test_healthy_server_is_recognized(tmp_path, monkeypatch):
    settings = configured(tmp_path)
    install_healthy_client(monkeypatch, settings.model_path)
    manager = RuntimeManager(settings)
    monkeypatch.setattr(manager, "_endpoint_pid", lambda: 81)
    monkeypatch.setattr(manager, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(
        manager,
        "_process_info",
        lambda _pid: {
            "executable": str(settings.server_executable),
            "command_line": f'{settings.server_executable} -m "{settings.model_path}" -c 16384 -np 1',
        },
    )
    result = manager.inspect_llama_server()
    assert result["healthy"] and result["recognized"] and result["compatible"]


def test_dead_persisted_process_is_removed_before_start(tmp_path, monkeypatch):
    settings = configured(tmp_path)
    manager = RuntimeManager(settings)
    manager._save({"pid": 999, "managed_by_jarvis": True})
    stale = {**absent(), "state_stale": True}
    monkeypatch.setattr(manager, "inspect_llama_server", lambda: stale)
    monkeypatch.setattr(
        manager,
        "_start_llama_server_unlocked",
        lambda _wait: {"started": True, "reused": False},
    )
    assert manager.ensure_llama_server()["started"]
    assert not manager.state_file.exists()


def test_compatible_manual_process_is_external(tmp_path, monkeypatch):
    settings = configured(tmp_path)
    install_healthy_client(monkeypatch, settings.model_path)
    manager = RuntimeManager(settings)
    monkeypatch.setattr(manager, "_endpoint_pid", lambda: 82)
    monkeypatch.setattr(manager, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(
        manager,
        "_process_info",
        lambda _pid: {
            "executable": str(settings.server_executable),
            "command_line": f'{settings.server_executable} -m "{settings.model_path}" -c 16384 -np 1',
        },
    )
    result = manager.inspect_llama_server()
    assert result["compatible"] and result["managed_by_jarvis"] is False


def test_compatible_managed_process_keeps_ownership(tmp_path, monkeypatch):
    settings = configured(tmp_path)
    install_healthy_client(monkeypatch, settings.model_path)
    manager = RuntimeManager(settings)
    manager._save(
        {
            "pid": 83,
            "managed_by_jarvis": True,
            "model": str(settings.model_path),
            "command": settings.server_command(),
        }
    )
    monkeypatch.setattr(manager, "_endpoint_pid", lambda: 83)
    monkeypatch.setattr(manager, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(manager, "_process_info", lambda _pid: {})
    result = manager.inspect_llama_server()
    assert result["compatible"] and result["managed_by_jarvis"] is True


def test_stop_does_not_kill_external_server(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    monkeypatch.setattr(
        manager,
        "inspect_llama_server",
        lambda: compatible(pid=84, managed=False),
    )
    monkeypatch.setattr(manager, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(
        runtime_module.os,
        "kill",
        lambda *_args: pytest.fail("servidor externo nao deve ser encerrado"),
    )
    result = manager.stop()
    assert not result["stopped"] and result["reason"] == "external_server"


def test_repeated_startup_reuses_same_process(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    monkeypatch.setattr(manager, "inspect_llama_server", lambda: compatible(pid=85))
    first = manager.ensure_llama_server()
    second = manager.ensure_llama_server()
    assert first["pid"] == second["pid"] == 85
    assert first["reused"] and second["reused"]


def test_two_simultaneous_startups_create_one_server(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    state = {"started": False, "starts": 0}

    def inspect():
        return compatible(pid=86) if state["started"] else absent()

    def start(_wait):
        state["starts"] += 1
        time.sleep(0.1)
        state["started"] = True
        return {**compatible(pid=86), "started": True, "reused": False}

    monkeypatch.setattr(manager, "inspect_llama_server", inspect)
    monkeypatch.setattr(manager, "_start_llama_server_unlocked", start)
    results: list[dict] = []
    threads = [threading.Thread(target=lambda: results.append(manager.ensure_llama_server())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert state["starts"] == 1 and len(results) == 2
    assert {item["pid"] for item in results} == {86}


def test_health_failure_does_not_start_over_live_process(tmp_path, monkeypatch):
    manager = RuntimeManager(configured(tmp_path))
    value = {**absent(), "running": True, "occupied": True, "pid": 87}
    monkeypatch.setattr(manager, "inspect_llama_server", lambda: value)
    monkeypatch.setattr(
        manager,
        "_start_llama_server_unlocked",
        lambda _wait: pytest.fail("nao deve duplicar processo sem health"),
    )
    with pytest.raises(LlamaServerEndpointOccupied, match="nao esta saudavel"):
        manager.ensure_llama_server()


def test_expected_model_is_identified_from_props(tmp_path, monkeypatch):
    settings = configured(tmp_path)
    install_healthy_client(monkeypatch, settings.model_path)
    manager = RuntimeManager(settings)
    monkeypatch.setattr(manager, "_endpoint_pid", lambda: 88)
    monkeypatch.setattr(manager, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(manager, "_process_info", lambda _pid: {})
    result = manager.inspect_llama_server()
    assert result["model"] == str(settings.model_path)
    assert result["mismatches"] == []
