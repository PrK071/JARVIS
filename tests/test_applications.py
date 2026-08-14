import subprocess
from datetime import datetime, timedelta

from tern.orchestrator.applications import ApplicationManager


class KnownApplications(ApplicationManager):
    def installed(self):
        return [
            {"name": "Google Chrome", "app_id": "Chrome.App"},
            {"name": "Calculadora", "app_id": "Calculator.App"},
        ]


def test_application_resolution_is_exact_or_unambiguous():
    manager = KnownApplications(platform="nt")
    assert manager.resolve("calculadora")["application"]["app_id"] == "Calculator.App"
    assert manager.resolve("inexistente")["error"] == "application_not_found"


def test_open_uses_resolved_start_app_id_only():
    calls = []
    manager = KnownApplications(
        platform="nt",
        launcher=lambda command, **kwargs: calls.append((command, kwargs)),
    )

    result = manager.open("Google Chrome")

    assert result["ok"]
    assert calls[0][0] == ["explorer.exe", "shell:AppsFolder\\Chrome.App"]


def test_schedule_validates_future_time_and_registers_task():
    calls = []
    manager = KnownApplications(
        platform="nt",
        runner=lambda command, **kwargs: (
            calls.append((command, kwargs))
            or subprocess.CompletedProcess(command, 0, '{"ok":true}', "")
        ),
    )
    start_at = (datetime.now() + timedelta(hours=2)).isoformat(timespec="minutes")

    result = manager.schedule("Calculadora", start_at=start_at, recurrence="once")

    assert result["ok"]
    assert result["application"] == "Calculadora"
    assert result["recurrence"] == "once"
    assert calls


def test_schedule_rejects_past_time():
    manager = KnownApplications(platform="nt")
    start_at = (datetime.now() - timedelta(minutes=1)).isoformat(timespec="minutes")
    assert manager.schedule("Calculadora", start_at=start_at)["error"] == "scheduled_time_must_be_future"
