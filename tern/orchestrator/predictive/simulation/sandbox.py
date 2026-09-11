from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(frozen=True)
class SandboxCapabilities:
    provider: str
    available: bool
    write_confinement_verified: bool = False
    network_denial_verified: bool = False
    environment_sanitization_verified: bool = False
    child_process_containment_verified: bool = False
    resource_limits_verified: bool = False
    cleanup_verified: bool = False
    reason: str | None = None

    @property
    def ready(self) -> bool:
        return self.available and all((
            self.write_confinement_verified,
            self.network_denial_verified,
            self.environment_sanitization_verified,
            self.child_process_containment_verified,
            self.resource_limits_verified,
            self.cleanup_verified,
        ))

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "available": self.available,
            "ready": self.ready,
            "write_confinement_verified": self.write_confinement_verified,
            "network_denial_verified": self.network_denial_verified,
            "environment_sanitization_verified": self.environment_sanitization_verified,
            "child_process_containment_verified": self.child_process_containment_verified,
            "resource_limits_verified": self.resource_limits_verified,
            "cleanup_verified": self.cleanup_verified,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SandboxRunSpec:
    input_dir: Path
    command: tuple[str, ...]
    image: str = "python:3.13-alpine"
    timeout_seconds: float = 30.0
    cpu_limit: float = 1.0
    memory_mb: int = 256
    pids_limit: int = 64
    output_limit_bytes: int = 131_072
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        root = Path(self.input_dir).resolve(strict=True)
        if not root.is_dir() or not self.command or any(not str(item) for item in self.command):
            raise ValueError("sandbox input directory and command are required")
        if self.timeout_seconds <= 0 or self.cpu_limit <= 0:
            raise ValueError("sandbox time and CPU limits must be positive")
        if self.memory_mb < 64 or not 8 <= self.pids_limit <= 512:
            raise ValueError("sandbox memory/PID limits are outside the safe range")
        if not 1024 <= self.output_limit_bytes <= 4_194_304:
            raise ValueError("sandbox output limit is outside the safe range")
        blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "SSH", "AWS")
        if any(any(word in key.upper() for word in blocked) for key, _value in self.environment):
            raise ValueError("secret-like environment variable is forbidden")
        object.__setattr__(self, "input_dir", root)
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        object.__setattr__(self, "environment", tuple((str(key), str(value)) for key, value in self.environment))


@dataclass(frozen=True)
class SandboxRunResult:
    executed: bool
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    output_truncated: bool = False
    cleanup_verified: bool = False
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "executed": self.executed, "exit_code": self.exit_code,
            "stdout": self.stdout, "stderr": self.stderr,
            "timed_out": self.timed_out, "output_truncated": self.output_truncated,
            "cleanup_verified": self.cleanup_verified,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class SandboxSelfTestResult:
    capabilities: SandboxCapabilities
    diagnostics: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "capabilities": self.capabilities.as_dict(),
            "diagnostics": list(self.diagnostics),
        }


class SandboxProvider(Protocol):
    def capabilities(self) -> SandboxCapabilities: ...
    def self_test(self) -> SandboxSelfTestResult: ...
    def run(self, spec: SandboxRunSpec) -> SandboxRunResult: ...


class UnavailableSandboxProvider:
    def __init__(self, reason: str = "secure sandbox unavailable") -> None:
        self._capabilities = SandboxCapabilities("unavailable", False, reason=reason)

    def capabilities(self) -> SandboxCapabilities:
        return self._capabilities

    def self_test(self) -> SandboxSelfTestResult:
        return SandboxSelfTestResult(self._capabilities, (self._capabilities.reason or "unavailable",))

    def run(self, spec: SandboxRunSpec) -> SandboxRunResult:
        return SandboxRunResult(False, None, failure_reason=self._capabilities.reason)


def _sanitized_host_environment(docker_path: Path) -> dict[str, str]:
    result = {"PATH": str(docker_path.parent)}
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
            if os.environ.get(key):
                result[key] = os.environ[key]
    return result


class DockerSandboxProvider:
    """Docker-backed execution that is unavailable until its self-test passes."""

    def __init__(self, docker_path: str | Path, *, image: str = "python:3.13-alpine") -> None:
        self.docker_path = Path(docker_path).resolve(strict=True)
        self.image = image
        self._capabilities = SandboxCapabilities(
            "docker", False, reason="sandbox self-test has not passed"
        )

    def capabilities(self) -> SandboxCapabilities:
        return self._capabilities

    def _host_environment(self) -> dict[str, str]:
        return _sanitized_host_environment(self.docker_path)

    def build_command(self, spec: SandboxRunSpec, container_name: str) -> tuple[str, ...]:
        environment = (
            ("HOME", "/nonexistent"),
            ("PATH", "/usr/local/bin:/usr/bin:/bin"),
            ("PYTHONNOUSERSITE", "1"),
            ("PYTHONDONTWRITEBYTECODE", "1"),
            *spec.environment,
        )
        command = [
            str(self.docker_path), "run", "--name", container_name, "--rm",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(spec.pids_limit),
            "--memory", f"{spec.memory_mb}m", "--cpus", str(spec.cpu_limit),
            "--user", "65534:65534",
            "--mount", f"type=bind,src={spec.input_dir},dst=/input,readonly",
            "--tmpfs", "/work:rw,nosuid,nodev,size=67108864,mode=1777",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=33554432,mode=1777",
            "--workdir", "/work",
        ]
        for key, value in environment:
            command.extend(("--env", f"{key}={value}"))
        command.extend((
            spec.image,
            "sh", "-c", 'cp -R /input/. /work/ && cd /work && exec "$@"',
            "sandbox-runner", *spec.command,
        ))
        return tuple(command)

    @staticmethod
    def _read_stream(stream, limit: int, output: bytearray, truncated: list[bool]) -> None:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            remaining = max(0, limit - len(output))
            output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True

    def _container_absent(self, name: str) -> bool:
        completed = subprocess.run(
            [str(self.docker_path), "inspect", name], capture_output=True,
            timeout=10, env=self._host_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return completed.returncode != 0

    def _run_unchecked(self, spec: SandboxRunSpec) -> SandboxRunResult:
        name = f"jarvis-predictive-{uuid.uuid4().hex[:12]}"
        command = self.build_command(spec, name)
        stdout, stderr = bytearray(), bytearray()
        truncated = [False]
        timed_out = False
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=self._host_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            threads = [
                threading.Thread(target=self._read_stream, args=(process.stdout, spec.output_limit_bytes, stdout, truncated), daemon=True),
                threading.Thread(target=self._read_stream, args=(process.stderr, spec.output_limit_bytes, stderr, truncated), daemon=True),
            ]
            for thread in threads:
                thread.start()
            try:
                exit_code = process.wait(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(
                    [str(self.docker_path), "kill", name], capture_output=True,
                    timeout=10, env=self._host_environment(),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                process.kill()
                exit_code = process.wait(timeout=10)
            for thread in threads:
                thread.join(timeout=5)
            subprocess.run(
                [str(self.docker_path), "rm", "-f", name], capture_output=True,
                timeout=10, env=self._host_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            cleanup = self._container_absent(name)
            return SandboxRunResult(
                True, exit_code,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                timed_out, truncated[0], cleanup,
                "SANDBOX_TIMEOUT" if timed_out else None,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return SandboxRunResult(False, None, failure_reason=f"DOCKER_ERROR:{type(exc).__name__}")

    def self_test(self) -> SandboxSelfTestResult:
        try:
            version = subprocess.run(
                [str(self.docker_path), "version", "--format", "{{.Server.Version}}"],
                capture_output=True, timeout=15, env=self._host_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            image = subprocess.run(
                [str(self.docker_path), "image", "inspect", self.image],
                capture_output=True, timeout=15, env=self._host_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            version = image = None
        if not version or version.returncode or not image or image.returncode:
            reason = "Docker daemon or local sandbox image unavailable; image pulls are never automatic"
            self._capabilities = SandboxCapabilities("docker", False, reason=reason)
            return SandboxSelfTestResult(self._capabilities, (reason,))
        with tempfile.TemporaryDirectory(prefix="jarvis-sandbox-selftest-") as directory:
            root = Path(directory)
            sentinel = root / "sentinel.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            script = (
                "import json,os,socket,pathlib; "
                "write_blocked=False; network_blocked=False; "
                "\ntry: pathlib.Path('/input/escape').write_text('bad')\nexcept OSError: write_blocked=True; "
                "\ntry: socket.create_connection(('1.1.1.1',53),0.5)\nexcept OSError: network_blocked=True; "
                "\npathlib.Path('/work/ok').write_text('ok'); "
                "print(json.dumps({'write':write_blocked,'network':network_blocked,'env':os.environ.get('HOME')=='/nonexistent' and 'JARVIS_FAKE_SECRET' not in os.environ,'work':pathlib.Path('/work/ok').exists()}))"
            )
            spec = SandboxRunSpec(root, ("python", "-I", "-c", script), image=self.image, timeout_seconds=10)
            result = self._run_unchecked(spec)
            try:
                observed = json.loads(result.stdout.strip().splitlines()[-1]) if result.exit_code == 0 else {}
            except (IndexError, json.JSONDecodeError):
                observed = {}
            confinement = bool(observed.get("write") and observed.get("work") and sentinel.read_text(encoding="utf-8") == "unchanged" and not (root / "escape").exists())
            network = bool(observed.get("network"))
            environment = bool(observed.get("env"))
            command_flags = self.build_command(spec, "proof")
            limits = all(flag in command_flags for flag in ("--pids-limit", "--memory", "--cpus", "--read-only"))
            timeout_spec = SandboxRunSpec(root, ("python", "-c", "import subprocess,time; subprocess.Popen(['sleep','30']); time.sleep(30)"), image=self.image, timeout_seconds=1)
            timeout_result = self._run_unchecked(timeout_spec)
            process_containment = timeout_result.timed_out and timeout_result.cleanup_verified
            cleanup = result.cleanup_verified and timeout_result.cleanup_verified
            self._capabilities = SandboxCapabilities(
                "docker",
                all((confinement, network, environment, process_containment, limits, cleanup)),
                confinement, network, environment, process_containment, limits, cleanup,
                None if all((confinement, network, environment, process_containment, limits, cleanup)) else "one or more sandbox self-tests failed",
            )
            return SandboxSelfTestResult(self._capabilities, tuple(
                name for name, passed in {
                    "write_confinement": confinement, "network_denial": network,
                    "environment_sanitization": environment,
                    "child_process_containment": process_containment,
                    "resource_limits": limits, "cleanup": cleanup,
                }.items() if not passed
            ))

    def run(self, spec: SandboxRunSpec) -> SandboxRunResult:
        if not self._capabilities.ready:
            return SandboxRunResult(False, None, failure_reason="SANDBOX_NOT_VERIFIED")
        return self._run_unchecked(spec)


def detect_sandbox_provider(*, image: str = "python:3.13-alpine") -> SandboxProvider:
    executable = shutil.which("docker")
    if not executable:
        return UnavailableSandboxProvider("Docker executable unavailable")
    provider = DockerSandboxProvider(executable, image=image)
    result = provider.self_test()
    return provider if result.capabilities.ready else UnavailableSandboxProvider(
        result.capabilities.reason or "Docker sandbox self-test failed"
    )
