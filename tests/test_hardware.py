import json
import subprocess
from datetime import datetime, timezone

from tern.orchestrator.hardware import HardwareMonitor


def completed(value):
    return subprocess.CompletedProcess([], 0, json.dumps(value), "")


def test_hardware_monitor_returns_real_provider_values(tmp_path):
    monitor = HardwareMonitor(
        platform="nt",
        cache_path=tmp_path / "missing.json",
        runner=lambda *_args, **_kwargs: completed(
            {
                "cpu_temperature_c": 54.25,
                "cpu_temperature_source": "root\\LibreHardwareMonitor/CPU Package",
                "usb_devices": 4,
                "usb_source": "Windows PnP container IDs",
            }
        ),
    )

    result = monitor.read()

    assert result["ok"]
    assert result["cpu_temperature_c"] == 54.25
    assert result["cpu_temperature_available"]
    assert result["usb_devices"] == 4
    assert result["usb_available"]


def test_hardware_monitor_never_fabricates_missing_temperature(tmp_path):
    monitor = HardwareMonitor(
        platform="nt",
        cache_path=tmp_path / "missing.json",
        runner=lambda *_args, **_kwargs: completed(
            {"cpu_temperature_c": None, "usb_devices": 2}
        ),
    )

    result = monitor.read()

    assert result["ok"]
    assert result["cpu_temperature_c"] is None
    assert not result["cpu_temperature_available"]
    assert result["usb_devices"] == 2


def test_hardware_monitor_reports_unsupported_platform():
    result = HardwareMonitor(platform="posix").read()
    assert not result["ok"]
    assert result["error"] == "unsupported_platform"


def test_hardware_monitor_prefers_fresh_elevated_sensor_cache(tmp_path):
    cache = tmp_path / "hardware.json"
    cache.write_text(
        json.dumps(
            {
                "cpu_temperature_c": 61.5,
                "cpu_temperature_source": "LibreHardwareMonitor/CPU Package",
                "measured_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monitor = HardwareMonitor(
        platform="nt",
        cache_path=cache,
        runner=lambda *_args, **_kwargs: completed(
            {"cpu_temperature_c": None, "usb_devices": 2}
        ),
    )

    result = monitor.read()

    assert result["cpu_temperature_c"] == 61.5
    assert result["cpu_temperature_source"] == "LibreHardwareMonitor/CPU Package"


def test_hardware_monitor_uses_complete_cache_without_powershell(tmp_path):
    cache = tmp_path / "hardware.json"
    cache.write_text(
        json.dumps(
            {
                "cpu_temperature_c": 58.0,
                "cpu_temperature_source": "LibreHardwareMonitor/CPU Package",
                "usb_devices": 3,
                "usb_source": "Windows PnP container IDs",
                "measured_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    def unexpected_runner(*_args, **_kwargs):
        raise AssertionError("PowerShell should not run for a complete fresh cache")

    result = HardwareMonitor(
        platform="nt",
        cache_path=cache,
        runner=unexpected_runner,
    ).read()

    assert result["cpu_temperature_c"] == 58.0
    assert result["usb_devices"] == 3
