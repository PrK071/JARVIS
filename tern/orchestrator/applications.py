from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import unicodedata
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(character for character in normalized if not unicodedata.combining(character)).split()
    )


class ApplicationManager:
    """Resolve installed Start applications without accepting arbitrary executables."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        launcher: Callable[..., Any] = subprocess.Popen,
        platform: str | None = None,
    ) -> None:
        self._runner = runner
        self._launcher = launcher
        self.platform = platform or os.name

    @staticmethod
    def _powershell_command(script: str) -> list[str]:
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ]

    def _run_powershell(self, script: str, *, timeout: int = 20) -> subprocess.CompletedProcess[str]:
        return self._runner(
            self._powershell_command(script),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def installed(self) -> list[dict[str, str]]:
        if self.platform != "nt":
            return []
        script = """
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Get-StartApps | Sort-Object Name | Select-Object Name, AppID | ConvertTo-Json -Compress
"""
        try:
            completed = self._run_powershell(script)
            value = json.loads(completed.stdout.strip() or "[]")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return []
        unique: dict[tuple[str, str], dict[str, str]] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name") or "").strip()
            app_id = str(item.get("AppID") or "").strip()
            if name and app_id:
                unique[(_plain(name), app_id)] = {"name": name, "app_id": app_id}
        return sorted(unique.values(), key=lambda item: _plain(item["name"]))

    def list(self, *, query: str | None = None, limit: int = 50) -> dict[str, Any]:
        applications = self.installed()
        normalized_query = _plain(query or "")
        if normalized_query:
            applications = [
                item for item in applications if normalized_query in _plain(item["name"])
            ]
        return {
            "ok": True,
            "applications": [{"name": item["name"]} for item in applications[:limit]],
            "count": len(applications),
            "truncated": len(applications) > limit,
            "error": None,
        }

    def resolve(self, name: str) -> dict[str, Any]:
        requested = _plain(name)
        if not requested:
            return {"ok": False, "error": "application_name_missing"}
        applications = self.installed()
        exact = [item for item in applications if _plain(item["name"]) == requested]
        candidates = exact or [
            item for item in applications if requested in _plain(item["name"])
        ]
        if not candidates:
            return {"ok": False, "error": "application_not_found", "query": name}
        if len(candidates) > 1:
            return {
                "ok": False,
                "error": "application_ambiguous",
                "query": name,
                "candidates": [item["name"] for item in candidates[:10]],
            }
        return {"ok": True, "application": candidates[0], "error": None}

    def open(self, name: str) -> dict[str, Any]:
        resolved = self.resolve(name)
        if not resolved.get("ok"):
            return resolved
        application = resolved["application"]
        try:
            self._launcher(
                ["explorer.exe", f"shell:AppsFolder\\{application['app_id']}"],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            return {
                "ok": False,
                "error": "application_launch_failed",
                "message": str(exc),
            }
        return {"ok": True, "application": application["name"], "error": None}

    @staticmethod
    def _parse_start_at(value: str) -> datetime:
        start_at = datetime.fromisoformat(value)
        if start_at.tzinfo is not None:
            start_at = start_at.astimezone().replace(tzinfo=None)
        now = datetime.now()
        if start_at <= now + timedelta(seconds=30):
            raise ValueError("scheduled_time_must_be_future")
        if start_at > now + timedelta(days=366):
            raise ValueError("scheduled_time_too_far")
        return start_at

    def schedule(self, name: str, *, start_at: str, recurrence: str = "once") -> dict[str, Any]:
        resolved = self.resolve(name)
        if not resolved.get("ok"):
            return resolved
        if recurrence not in {"once", "daily"}:
            return {"ok": False, "error": "invalid_recurrence"}
        try:
            scheduled = self._parse_start_at(start_at)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        application = resolved["application"]
        safe_name = re.sub(r"[^\w .-]+", "", application["name"], flags=re.UNICODE).strip()[:48]
        task_name = f"Jarvis - {safe_name or 'Aplicativo'} - {uuid.uuid4().hex[:8]}"
        app_id_json = json.dumps(application["app_id"], ensure_ascii=False)
        task_json = json.dumps(task_name, ensure_ascii=False)
        date_json = json.dumps(scheduled.isoformat(timespec="seconds"))
        trigger = (
            f"New-ScheduledTaskTrigger -Daily -At ([datetime]::Parse({date_json}, "
            "[Globalization.CultureInfo]::InvariantCulture))"
            if recurrence == "daily"
            else f"New-ScheduledTaskTrigger -Once -At ([datetime]::Parse({date_json}, "
            "[Globalization.CultureInfo]::InvariantCulture))"
        )
        script = f"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$action = New-ScheduledTaskAction -Execute 'explorer.exe' -Argument ('shell:AppsFolder\\' + {app_id_json})
$trigger = {trigger}
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName {task_json} -Action $action -Trigger $trigger -Principal $principal -Description 'Criada pelo JARVIS' -Force | Out-Null
[pscustomobject]@{{ ok = $true; task_name = {task_json} }} | ConvertTo-Json -Compress
"""
        try:
            completed = self._run_powershell(script, timeout=30)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "Task Scheduler failed")
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return {
                "ok": False,
                "error": "schedule_creation_failed",
                "message": str(exc),
            }
        return {
            "ok": True,
            "application": application["name"],
            "start_at": scheduled.isoformat(timespec="minutes"),
            "recurrence": recurrence,
            "task_name": task_name,
            "error": None,
        }
