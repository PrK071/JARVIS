from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from .workspace import copy_scoped_snapshot

_IMAGE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/:\-]*@sha256:[0-9a-f]{64}\Z")
_ALLOWED_ENV = {"LANG", "LC_ALL", "TZ", "PYTHONHASHSEED"}


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
    private_host_paths_verified: bool = False
    traversal_symlink_verified: bool = False
    timeout_verified: bool = False
    non_root_seccomp_verified: bool = False
    image_digest: str | None = None

    @property
    def ready(self) -> bool:
        return self.available and all((
            self.write_confinement_verified, self.network_denial_verified,
            self.environment_sanitization_verified,
            self.child_process_containment_verified, self.resource_limits_verified,
            self.cleanup_verified, self.private_host_paths_verified,
            self.traversal_symlink_verified, self.timeout_verified,
            self.non_root_seccomp_verified,
        ))

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"ready": self.ready}


@dataclass(frozen=True)
class SandboxRunSpec:
    input_dir: Path
    command: tuple[str, ...]
    image: str | None = None
    timeout_seconds: float = 30.0
    cpu_limit: float = 1.0
    memory_mb: int = 256
    pids_limit: int = 64
    output_limit_bytes: int = 131_072
    environment: tuple[tuple[str, str], ...] = ()
    input_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        root = Path(self.input_dir).resolve(strict=True)
        if not root.is_dir() or not self.command or any(not item or "\0" in item for item in self.command):
            raise ValueError("sandbox input directory and command are required")
        if not all(math.isfinite(value) and value > 0 for value in (self.timeout_seconds, self.cpu_limit)):
            raise ValueError("sandbox limits must be finite and positive")
        if self.timeout_seconds > 300 or self.cpu_limit > 4:
            raise ValueError("sandbox time/CPU limit exceeded")
        if not 64 <= self.memory_mb <= 2048 or not 8 <= self.pids_limit <= 128:
            raise ValueError("sandbox memory/PID limits are outside the safe range")
        if not 1024 <= self.output_limit_bytes <= 1_048_576:
            raise ValueError("sandbox output limit is outside the safe range")
        if self.image is not None and not _IMAGE.fullmatch(self.image):
            raise ValueError("sandbox image requires immutable sha256 digest")
        if len(dict(self.environment)) != len(self.environment):
            raise ValueError("duplicate environment variable")
        if any(key not in _ALLOWED_ENV or "\0" in value for key, value in self.environment):
            raise ValueError("secret-like or non-allowlisted environment variable is forbidden")
        object.__setattr__(self, "input_dir", root)


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
    wall_seconds: float = 0.0
    setup_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    infrastructure_failure: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxSelfTestResult:
    capabilities: SandboxCapabilities
    diagnostics: tuple[str, ...] = ()
    checks: tuple[tuple[str, bool], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"capabilities": self.capabilities.as_dict(),
                "diagnostics": list(self.diagnostics), "checks": dict(self.checks)}


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
    """Functional checks reduce risk; containers are not an absolute escape proof."""

    def __init__(self, docker_path: str | Path, *, image: str | None = None) -> None:
        self.docker_path = Path(docker_path).resolve(strict=True)
        self.image = image
        self._capabilities = SandboxCapabilities("docker", False, reason="sandbox self-test has not passed")

    def capabilities(self) -> SandboxCapabilities:
        return self._capabilities

    def _control(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.docker_path), *args], capture_output=True, timeout=15,
            env=_sanitized_host_environment(self.docker_path),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def build_command(self, spec: SandboxRunSpec, container_name: str) -> tuple[str, ...]:
        if not self.image or not _IMAGE.fullmatch(self.image):
            raise ValueError("configured trusted image digest is required")
        if spec.image is not None and spec.image != self.image:
            raise ValueError("workload cannot override verified image")
        if not re.fullmatch(r"[a-z0-9-]+", container_name):
            raise ValueError("invalid container name")
        if any(char in str(spec.input_dir) for char in (",", "\n", "\r")):
            raise ValueError("invalid bind mount path")
        env = (("HOME", "/nonexistent"), ("PATH", "/usr/local/bin:/usr/bin:/bin"),
               ("LANG", "C.UTF-8"),
               ("PYTHONNOUSERSITE", "1"), ("PYTHONDONTWRITEBYTECODE", "1"),
               ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"), *spec.environment)
        command = [
            str(self.docker_path), "run", "--pull", "never", "--name", container_name,
            "--no-healthcheck",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(spec.pids_limit), "--memory", f"{spec.memory_mb}m",
            "--memory-swap", f"{spec.memory_mb}m", "--cpus", str(spec.cpu_limit),
            "--ulimit", "core=0:0", "--user", "65534:65534",
            "--mount", f"type=bind,src={spec.input_dir},dst=/input,readonly",
            "--tmpfs", "/work:rw,nosuid,nodev,size=67108864,mode=1777",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=16777216,mode=1777",
            "--workdir", "/work", "--entrypoint", "/bin/sh",
        ]
        for key, value in env:
            command.extend(("--env", f"{key}={value}"))
        command.extend((self.image, "-c",
            'cp -R /input/. /work/ && exec /usr/bin/env -i "$@"',
            "runner", *(f"{key}={value}" for key, value in env), *spec.command))
        return tuple(command)

    @staticmethod
    def _read_stream(stream, limit: int, output: bytearray, truncated: list[bool]) -> None:
        try:
            while chunk := stream.read(8192):
                remaining = max(0, limit - len(output))
                output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[0] = True
        finally:
            stream.close()

    def _container_absent(self, name: str) -> bool:
        result = self._control("container", "ls", "--all", "--quiet", "--filter", f"name=^{name}$")
        return result.returncode == 0 and not result.stdout.strip()

    def _run_unchecked(self, spec: SandboxRunSpec) -> SandboxRunResult:
        started = time.monotonic()
        name = f"jarvis-predictive-{uuid.uuid4().hex}"
        stdout, stderr = bytearray(), bytearray()
        truncated = [False]
        process = None
        threads: list[threading.Thread] = []
        timed_out = False
        exit_code = None
        cleanup = False
        failure = None
        try:
            process = subprocess.Popen(
                self.build_command(spec, name), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_sanitized_host_environment(self.docker_path),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for stream, buffer in ((process.stdout, stdout), (process.stderr, stderr)):
                thread = threading.Thread(target=self._read_stream,
                    args=(stream, spec.output_limit_bytes // 2, buffer, truncated), daemon=True)
                thread.start()
                threads.append(thread)
            try:
                exit_code = process.wait(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                failure = "SANDBOX_TIMEOUT"
        except (OSError, ValueError, subprocess.SubprocessError):
            failure = "SANDBOX_INFRASTRUCTURE_ERROR"
        finally:
            cleanup_start = time.monotonic()
            try:
                self._control("rm", "--force", name)
                cleanup = True
            except (OSError, subprocess.SubprocessError):
                cleanup = False
            if process is not None and process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.SubprocessError:
                    cleanup = False
            for thread in threads:
                thread.join(timeout=5)
                if thread.is_alive():
                    cleanup = False
            try:
                cleanup = self._container_absent(name) and cleanup
            except (OSError, subprocess.SubprocessError):
                cleanup = False
            cleanup_seconds = time.monotonic() - cleanup_start
        infrastructure = failure == "SANDBOX_INFRASTRUCTURE_ERROR" or exit_code in {125, 126, 127} or not cleanup
        if not cleanup:
            failure = "SANDBOX_CLEANUP_FAILED"
        if infrastructure:
            self._capabilities = SandboxCapabilities("docker", False, reason=failure or "runtime failure")
        return SandboxRunResult(
            process is not None, exit_code, stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"), timed_out, truncated[0], cleanup,
            failure, time.monotonic() - started, cleanup_seconds=cleanup_seconds,
            infrastructure_failure=infrastructure,
        )

    def self_test(self) -> SandboxSelfTestResult:
        self._capabilities = SandboxCapabilities("docker", False, reason="self-test pending")
        if not self.image or not _IMAGE.fullmatch(self.image):
            reason = "trusted local image digest required; no automatic pull or build"
            self._capabilities = SandboxCapabilities("docker", False, reason=reason)
            return SandboxSelfTestResult(self._capabilities, (reason,))
        try:
            server = self._control("info", "--format", "{{json .}}")
            image = self._control("image", "inspect", self.image)
            info = json.loads(server.stdout) if server.returncode == 0 else {}
            image_info = json.loads(image.stdout) if image.returncode == 0 else []
            supported = info.get("OSType") == "linux" and any(
                "seccomp" in option and "profile=builtin" in option
                for option in info.get("SecurityOptions", [])
            )
            if not supported or not image_info or image_info[0].get("Config", {}).get("Volumes"):
                raise ValueError("Linux Docker with seccomp and local image required")
            with tempfile.TemporaryDirectory(prefix="jarvis-selftest-") as directory:
                root = Path(directory)
                probe = Path(__file__).with_name("selftest_probe.py").read_text(encoding="utf-8")
                spec = SandboxRunSpec(root, ("python", "-I", "-c", probe), timeout_seconds=15,
                                      memory_mb=64, pids_limit=32, cpu_limit=0.5)
                result = self._run_unchecked(spec)
                observed = json.loads(result.stdout) if result.exit_code == 0 and not result.infrastructure_failure else {}
                timeout_result = self._run_unchecked(replace(spec,
                    command=("python", "-I", "-c", "import subprocess,time; subprocess.Popen(['python','-c','import time;time.sleep(60)']);time.sleep(60)"),
                    timeout_seconds=1))
                memory_result = self._run_unchecked(replace(spec,
                    command=("python", "-I", "-c", "x=bytearray(128*1024*1024);print('unexpected')")))
                checks = {name: observed.get(name) is True for name in (
                    "write", "network", "environment", "private_paths", "traversal_symlink", "non_root_seccomp", "resource_limits")}
                checks["resource_limits"] &= memory_result.exit_code == 137 and memory_result.cleanup_verified
                checks["timeout"] = timeout_result.timed_out
                checks["process_containment"] = timeout_result.timed_out and timeout_result.cleanup_verified
                checks["cleanup"] = all(r.cleanup_verified for r in (result, timeout_result, memory_result))
                passed = all(checks.values())
                self._capabilities = SandboxCapabilities(
                    "docker", passed, checks["write"], checks["network"], checks["environment"],
                    checks["process_containment"], checks["resource_limits"], checks["cleanup"],
                    None if passed else "mandatory functional self-test failed",
                    checks["private_paths"], checks["traversal_symlink"], checks["timeout"],
                    checks["non_root_seccomp"], self.image)
                return SandboxSelfTestResult(self._capabilities,
                    tuple(name for name, ok in checks.items() if not ok), tuple(checks.items()))
        except (OSError, ValueError, subprocess.SubprocessError):
            reason = "Docker daemon, Linux/seccomp or trusted local image unavailable"
            self._capabilities = SandboxCapabilities("docker", False, reason=reason)
            return SandboxSelfTestResult(self._capabilities, (reason,))

    def run(self, spec: SandboxRunSpec) -> SandboxRunResult:
        if not self._capabilities.ready:
            return SandboxRunResult(False, None, failure_reason="SANDBOX_NOT_VERIFIED")
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="jarvis-input-") as directory:
                snapshot = Path(directory)
                copy_scoped_snapshot(spec.input_dir, snapshot, spec.input_files)
                setup = time.monotonic() - started
                result = self._run_unchecked(replace(spec, input_dir=snapshot))
                return replace(result, setup_seconds=setup)
        except (OSError, ValueError):
            return SandboxRunResult(False, None, failure_reason="SNAPSHOT_SCOPE_REJECTED", infrastructure_failure=True)


def detect_sandbox_provider(*, image: str | None = None) -> SandboxProvider:
    executable = shutil.which("docker")
    if not executable:
        return UnavailableSandboxProvider("Docker executable unavailable")
    provider = DockerSandboxProvider(executable, image=image or os.environ.get("JARVIS_SANDBOX_IMAGE"))
    provider.self_test()
    return provider
