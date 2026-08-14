from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .client import LlamaClient, ServerError
from .codex_state import FileMutex
from .config import Settings, assert_runtime_ready


class LlamaServerConfigurationMismatch(RuntimeError):
    pass


class LlamaServerEndpointOccupied(RuntimeError):
    pass


class RuntimeManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state_file = settings.state_dir / "server.json"
        self.lock_file = settings.state_dir / "server-start.lock"
        self.log_file = settings.state_dir / "llama-server.log"

    def inspect_llama_server(self) -> dict[str, Any]:
        """Inspect configured endpoint without assuming Jarvis owns its process."""
        state = self._state()
        state_pid = self._integer(state.get("pid"))
        state_pid_alive = bool(state_pid and self._pid_exists(state_pid))
        endpoint_pid = self._endpoint_pid()
        pid = endpoint_pid or (state_pid if state_pid_alive else None)
        process = self._process_info(pid) if pid else {}

        client = LlamaClient(self.settings.base_url, timeout=2)
        health: dict[str, Any] | None = None
        props: dict[str, Any] | None = None
        try:
            health = client.health()
        except Exception:
            pass
        healthy = bool(health and health.get("status") == "ok")
        if healthy:
            try:
                props = client.props()
            except Exception:
                pass

        command = self._command_tokens(process.get("command_line"))
        if not command and state_pid_alive and state_pid == pid:
            command = self._command_tokens(state.get("command"))
        command_values = self._command_values(command)
        executable = str(
            process.get("executable")
            or (state.get("executable") if state_pid_alive and state_pid == pid else "")
            or (command[0] if command else "")
        )
        props_model = self._first_text(
            (props or {}).get("model_path"),
            (props or {}).get("model_alias"),
        )
        model = self._first_text(
            props_model,
            command_values.get("model"),
            state.get("model") if state_pid_alive and state_pid == pid else None,
        )
        context_size = self._integer(
            ((props or {}).get("default_generation_settings") or {}).get("n_ctx")
        ) or self._integer(command_values.get("context_size"))
        parallel_slots = self._integer((props or {}).get("total_slots")) or self._integer(
            command_values.get("parallel_slots")
        )
        managed = bool(
            pid
            and state_pid == pid
            and state_pid_alive
            and state.get("managed_by_jarvis", True)
        )
        recognized = bool(
            props_model
            or "llama-server" in Path(executable).name.casefold()
            or "llama-server" in str(process.get("command_line") or "").casefold()
        )

        parameters = {
            key: value
            for key, value in {
                "context_size": context_size,
                "parallel_slots": parallel_slots,
                "flash_attention": command_values.get("flash_attention"),
                "kv_cache_k": command_values.get("kv_cache_k"),
                "kv_cache_v": command_values.get("kv_cache_v"),
                "gpu_layers": self._integer(command_values.get("gpu_layers")),
                "reasoning": command_values.get("reasoning"),
            }.items()
            if value is not None
        }
        mismatches = self._configuration_mismatches(model, parameters)
        running = bool((pid and self._pid_exists(pid)) or healthy)
        occupied = bool(running or endpoint_pid)
        return {
            "running": running,
            "occupied": occupied,
            "healthy": healthy,
            "recognized": recognized,
            "compatible": healthy and recognized and not mismatches,
            "pid": pid,
            "endpoint": self.settings.base_url,
            "backend": state.get("backend") if managed else None,
            "model": model,
            "context_size": context_size,
            "parameters": parameters,
            "managed_by_jarvis": managed,
            "mismatches": mismatches,
            "health": health,
            "log": str(self.log_file),
            "state_stale": bool(state_pid and not state_pid_alive),
            "process_executable": executable or None,
        }

    def status(self) -> dict[str, Any]:
        return self.inspect_llama_server()

    def ensure_llama_server(self, wait_seconds: int = 180) -> dict[str, Any]:
        """Reuse one compatible server; start only when endpoint is free."""
        with FileMutex(self.lock_file, timeout=max(30, wait_seconds)):
            existing = self.inspect_llama_server()
            if existing["occupied"] and existing["recognized"] and not existing["healthy"]:
                deadline = time.monotonic() + min(wait_seconds, 30)
                while time.monotonic() < deadline:
                    time.sleep(0.5)
                    existing = self.inspect_llama_server()
                    if existing["healthy"] or not existing["occupied"]:
                        break
            if existing["healthy"]:
                if not existing["recognized"]:
                    raise LlamaServerEndpointOccupied(
                        f"endpoint {self.settings.base_url} responde, mas nao foi reconhecido como llama-server"
                    )
                if existing["compatible"]:
                    if existing.get("state_stale"):
                        self.state_file.unlink(missing_ok=True)
                    return {**existing, "started": False, "reused": True}
                running_model = existing.get("model") or "modelo nao identificado"
                requested_model = str(self.settings.model_path)
                details = ", ".join(existing.get("mismatches") or [])
                raise LlamaServerConfigurationMismatch(
                    "llama-server existente usa configuracao diferente; "
                    f"running_model={running_model}; requested_model={requested_model}"
                    + (f"; divergencias={details}" if details else "")
                    + ". Startup normal nao troca modelo. Use uma acao explicita de troca."
                )
            if existing["occupied"]:
                pid = existing.get("pid") or "desconhecido"
                raise LlamaServerEndpointOccupied(
                    f"endpoint {self.settings.base_url} esta ocupado pelo PID {pid}, "
                    "mas o llama-server nao esta saudavel"
                )
            if existing.get("state_stale"):
                self.state_file.unlink(missing_ok=True)
            return self._start_llama_server_unlocked(wait_seconds)

    def start_llama_server(self, wait_seconds: int = 180) -> dict[str, Any]:
        """Start a new managed server only when no process occupies the endpoint."""
        with FileMutex(self.lock_file, timeout=max(30, wait_seconds)):
            existing = self.inspect_llama_server()
            if existing["occupied"] or existing["healthy"]:
                raise LlamaServerEndpointOccupied(
                    f"endpoint {self.settings.base_url} ja esta em uso; use ensure_llama_server()"
                )
            if existing.get("state_stale"):
                self.state_file.unlink(missing_ok=True)
            return self._start_llama_server_unlocked(wait_seconds)

    def start(self, wait_seconds: int = 180) -> dict[str, Any]:
        """Backward-compatible startup entry point: ensure, never switch."""
        return self.ensure_llama_server(wait_seconds)

    def switch_llama_model(
        self,
        wait_seconds: int = 180,
        *,
        allow_external_stop: bool = False,
    ) -> dict[str, Any]:
        """Explicit stop/change/start operation; never used by normal startup."""
        with FileMutex(self.lock_file, timeout=max(30, wait_seconds)):
            existing = self.inspect_llama_server()
            if existing["healthy"] and existing["compatible"]:
                return {**existing, "started": False, "reused": True, "switched": False}
            if existing["occupied"]:
                if not existing["managed_by_jarvis"] and not allow_external_stop:
                    raise RuntimeError(
                        "llama-server externo nao sera encerrado sem autorizacao explicita"
                    )
                self._stop_unlocked(
                    wait_seconds=min(wait_seconds, 20),
                    allow_external=allow_external_stop,
                )
            result = self._start_llama_server_unlocked(wait_seconds)
            return {**result, "switched": True}

    def _start_llama_server_unlocked(self, wait_seconds: int) -> dict[str, Any]:
        assert_runtime_ready(self.settings)
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        command = self.settings.server_command()
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        with self.log_file.open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=str(self.settings.server_executable.parent),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=creationflags,
            )
        self._save(
            {
                "pid": process.pid,
                "endpoint": self.settings.base_url,
                "managed_by_jarvis": True,
                "backend": self.settings.backend.name,
                "model": str(self.settings.model_path),
                "executable": str(self.settings.server_executable),
                "command": command,
                "started_at": time.time(),
            }
        )
        deadline = time.monotonic() + wait_seconds
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = self._log_tail()
                self.state_file.unlink(missing_ok=True)
                raise RuntimeError(
                    f"llama-server encerrou com codigo {process.returncode}\n{tail}"
                )
            try:
                health = LlamaClient(self.settings.base_url, timeout=2).health()
                if health.get("status") == "ok":
                    inspected = self.inspect_llama_server()
                    return {**inspected, "started": True, "reused": False}
            except ServerError as exc:
                last_error = str(exc)
            time.sleep(0.5)
        self._stop_unlocked(wait_seconds=20)
        raise TimeoutError(
            f"llama-server nao ficou pronto em {wait_seconds}s: {last_error}"
        )

    def stop(self, wait_seconds: int = 20) -> dict[str, Any]:
        with FileMutex(self.lock_file, timeout=max(30, wait_seconds)):
            return self._stop_unlocked(wait_seconds=wait_seconds)

    def _stop_unlocked(
        self,
        *,
        wait_seconds: int,
        allow_external: bool = False,
    ) -> dict[str, Any]:
        inspected = self.inspect_llama_server()
        pid = self._integer(inspected.get("pid"))
        if not pid or not self._pid_exists(pid):
            self.state_file.unlink(missing_ok=True)
            return {"stopped": False, "reason": "not_running"}
        if not inspected["managed_by_jarvis"] and not allow_external:
            return {
                "stopped": False,
                "reason": "external_server",
                "pid": pid,
                "managed_by_jarvis": False,
            }
        graceful = True
        try:
            if os.name == "nt":
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            graceful = False
        deadline = time.monotonic() + min(wait_seconds, 10)
        while time.monotonic() < deadline and self._pid_exists(pid):
            time.sleep(0.2)
        if self._pid_exists(pid):
            if os.name != "nt":
                raise RuntimeError(f"processo {pid} nao encerrou no prazo")
            graceful = False
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0 and self._pid_exists(pid):
                raise RuntimeError(
                    f"falha ao encerrar processo {pid}: {result.stderr.strip()}"
                )
        if inspected["managed_by_jarvis"]:
            self.state_file.unlink(missing_ok=True)
        return {
            "stopped": True,
            "pid": pid,
            "graceful": graceful,
            "managed_by_jarvis": inspected["managed_by_jarvis"],
        }

    def _configuration_mismatches(
        self,
        model: str | None,
        parameters: dict[str, Any],
    ) -> list[str]:
        mismatches: list[str] = []
        if not model:
            mismatches.append("running_model=unknown")
        elif not self._same_path(model, self.settings.model_path):
            mismatches.append("model")
        desired = {
            "context_size": self.settings.context_size,
            "parallel_slots": self.settings.parallel_slots,
            "flash_attention": self.settings.flash_attention,
            "kv_cache_k": self.settings.kv_cache_k,
            "kv_cache_v": self.settings.kv_cache_v,
            "gpu_layers": self.settings.gpu_layers,
            "reasoning": self.settings.reasoning,
        }
        for key, expected in desired.items():
            actual = parameters.get(key)
            if actual is not None and str(actual).casefold() != str(expected).casefold():
                mismatches.append(key)
        return mismatches

    def _endpoint_pid(self) -> int | None:
        if os.name != "nt":
            return None
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True,
                timeout=5,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        suffix = f":{self.settings.server_port}"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0].casefold() != "tcp":
                continue
            if parts[3].casefold() != "listening" or not parts[1].endswith(suffix):
                continue
            return self._integer(parts[4])
        return None

    def _process_info(self, pid: int) -> dict[str, str]:
        if os.name == "nt":
            script = (
                f"Get-CimInstance Win32_Process -Filter \"ProcessId = {int(pid)}\" | "
                "Select-Object ExecutablePath,CommandLine | ConvertTo-Json -Compress"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                value = json.loads(result.stdout) if result.stdout.strip() else {}
                return {
                    "executable": str(value.get("ExecutablePath") or ""),
                    "command_line": str(value.get("CommandLine") or ""),
                }
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                return {}
        try:
            executable = os.readlink(f"/proc/{pid}/exe")
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            return {"executable": executable, "command_line": command_line}
        except OSError:
            return {}

    @staticmethod
    def _command_tokens(command: Any) -> list[str]:
        if isinstance(command, list):
            return [str(item) for item in command]
        if not isinstance(command, str) or not command.strip():
            return []
        try:
            return [item.strip('"') for item in shlex.split(command, posix=False)]
        except ValueError:
            return []

    @staticmethod
    def _command_values(command: list[str]) -> dict[str, str]:
        flags = {
            "-m": "model",
            "--model": "model",
            "-c": "context_size",
            "--ctx-size": "context_size",
            "-np": "parallel_slots",
            "--parallel": "parallel_slots",
            "-fa": "flash_attention",
            "-ctk": "kv_cache_k",
            "-ctv": "kv_cache_v",
            "-ngl": "gpu_layers",
            "--reasoning": "reasoning",
        }
        values: dict[str, str] = {}
        for index, token in enumerate(command[:-1]):
            key = flags.get(token.casefold())
            if key:
                values[key] = command[index + 1].strip('"')
        return values

    def _state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_file.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.state_file)

    def _log_tail(self, length: int = 12000) -> str:
        try:
            return self.log_file.read_text(encoding="utf-8", errors="replace")[-length:]
        except FileNotFoundError:
            return ""

    @staticmethod
    def _same_path(left: str | Path, right: str | Path) -> bool:
        return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(
            os.path.abspath(str(right))
        )

    @staticmethod
    def _first_text(*values: Any) -> str | None:
        return next((str(value) for value in values if value not in {None, ""}), None)

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, pid
            )
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            try:
                if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                ):
                    return False
                return exit_code.value == still_active
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
