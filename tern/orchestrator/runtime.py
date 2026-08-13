from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .client import LlamaClient, ServerError
from .config import Settings, assert_runtime_ready


class RuntimeManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state_file = settings.state_dir / "server.json"
        self.log_file = settings.state_dir / "llama-server.log"

    def status(self) -> dict[str, Any]:
        state = self._state()
        healthy = False
        try:
            health = LlamaClient(self.settings.base_url, timeout=2).health()
            healthy = health.get("status") == "ok"
        except Exception:
            health = None
        pid = state.get("pid")
        return {
            "running": bool(pid and self._pid_exists(pid)),
            "healthy": healthy,
            "pid": pid,
            "backend": state.get("backend"),
            "model": state.get("model"),
            "health": health,
            "log": str(self.log_file),
        }

    def start(self, wait_seconds: int = 180) -> dict[str, Any]:
        assert_runtime_ready(self.settings)
        existing = self.status()
        if existing["running"] or existing["healthy"]:
            if (
                existing["backend"] == self.settings.backend.name
                and existing["model"] == str(self.settings.model_path)
                and existing["healthy"]
            ):
                return {**existing, "started": False}
            raise RuntimeError("llama-server ja esta ativo; pare-o antes de trocar modelo")

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
                raise RuntimeError(f"llama-server encerrou com codigo {process.returncode}\n{tail}")
            try:
                health = LlamaClient(self.settings.base_url, timeout=2).health()
                if health.get("status") == "ok":
                    return {**self.status(), "started": True}
            except ServerError as exc:
                last_error = str(exc)
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server nao ficou pronto em {wait_seconds}s: {last_error}")

    def stop(self, wait_seconds: int = 20) -> dict[str, Any]:
        state = self._state()
        pid = state.get("pid")
        if not pid or not self._pid_exists(pid):
            self.state_file.unlink(missing_ok=True)
            return {"stopped": False, "reason": "not_running"}
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
                raise RuntimeError(f"falha ao encerrar processo {pid}: {result.stderr.strip()}")
        self.state_file.unlink(missing_ok=True)
        return {"stopped": True, "pid": pid, "graceful": graceful}

    def _state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict[str, Any]) -> None:
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.state_file)

    def _log_tail(self, length: int = 12000) -> str:
        try:
            return self.log_file.read_text(encoding="utf-8", errors="replace")[-length:]
        except FileNotFoundError:
            return ""

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            try:
                if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
