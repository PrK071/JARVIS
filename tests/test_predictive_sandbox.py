from __future__ import annotations

import json

import pytest

from tern.orchestrator import cli
from tern.orchestrator.predictive.simulation.sandbox import (
    DockerSandboxProvider,
    SandboxRunSpec,
    UnavailableSandboxProvider,
    detect_sandbox_provider,
)


def test_unavailable_provider_fails_closed_without_dispatch(tmp_path):
    provider = UnavailableSandboxProvider("no secure runtime")
    spec = SandboxRunSpec(tmp_path, ("python", "-V"))

    result = provider.run(spec)

    assert provider.capabilities().ready is False
    assert result.executed is False
    assert result.failure_reason == "no secure runtime"


def test_detection_does_not_fallback_to_host_subprocess(monkeypatch):
    monkeypatch.setattr("tern.orchestrator.predictive.simulation.sandbox.shutil.which", lambda _name: None)

    provider = detect_sandbox_provider()

    assert isinstance(provider, UnavailableSandboxProvider)
    assert provider.capabilities().available is False


def test_docker_command_contains_mandatory_isolation(tmp_path):
    docker = tmp_path / "docker.exe"
    docker.touch()
    provider = DockerSandboxProvider(docker)
    spec = SandboxRunSpec(tmp_path, ("python", "-I", "-c", "print('ok')"))

    command = provider.build_command(spec, "proof")
    rendered = " ".join(command)

    assert "--network none" in rendered
    assert "--read-only" in command
    assert "--cap-drop ALL" in rendered
    assert "no-new-privileges:true" in command
    assert "--pids-limit" in command
    assert "--memory" in command
    assert "--cpus" in command
    assert "--user 65534:65534" in rendered
    assert "dst=/input,readonly" in rendered
    assert "/work:rw,nosuid,nodev" in rendered
    assert "/var/run/docker.sock" not in rendered


def test_docker_provider_refuses_run_before_self_test(tmp_path):
    docker = tmp_path / "docker.exe"
    docker.touch()
    provider = DockerSandboxProvider(docker)

    result = provider.run(SandboxRunSpec(tmp_path, ("python", "-V")))

    assert result.executed is False
    assert result.failure_reason == "SANDBOX_NOT_VERIFIED"


def test_sandbox_environment_rejects_secret_like_names(tmp_path):
    with pytest.raises(ValueError, match="secret-like"):
        SandboxRunSpec(tmp_path, ("python", "-V"), environment=(("OPENAI_API_KEY", "fake"),))


def test_sandbox_cli_reports_unavailable_without_project_execution(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_sandbox_provider", lambda: UnavailableSandboxProvider("Docker executable unavailable"))

    assert cli.main(["predictive-sandbox-check", "--json"]) == 1
    value = json.loads(capsys.readouterr().out)
    assert value["available"] is False
    assert value["ready"] is False
    assert value["reason"] == "Docker executable unavailable"
