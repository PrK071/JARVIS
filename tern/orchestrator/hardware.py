from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_WINDOWS_TELEMETRY_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$temperature = $null
$temperatureSource = $null
foreach ($namespace in @('root\LibreHardwareMonitor', 'root\OpenHardwareMonitor')) {
    $sensors = @(
        Get-CimInstance -Namespace $namespace -ClassName Sensor -ErrorAction SilentlyContinue |
        Where-Object {
            $_.SensorType -eq 'Temperature' -and
            ($_.Identifier -match '/cpu/' -or $_.Name -match '(?i)CPU|Package|Tctl|Tdie')
        }
    )
    if ($sensors.Count -gt 0) {
        $preferred = $sensors |
            Sort-Object @{Expression={ if ($_.Name -match '(?i)CPU Package|Tctl|Tdie') { 0 } else { 1 } }} |
            Select-Object -First 1
        if ($null -ne $preferred.Value) {
            $temperature = [math]::Round([double]$preferred.Value, 1)
            $temperatureSource = "$namespace/$($preferred.Name)"
            break
        }
    }
}

$containers = [System.Collections.Generic.HashSet[string]]::new()
$fallbackIds = [System.Collections.Generic.HashSet[string]]::new()
$devices = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
    Where-Object { $_.InstanceId -match '^USB\\VID_' })
foreach ($device in $devices) {
    $container = (Get-PnpDeviceProperty -InstanceId $device.InstanceId `
        -KeyName 'DEVPKEY_Device_ContainerId' -ErrorAction SilentlyContinue).Data
    if ($null -ne $container -and "$container" -notmatch '^0{8}-0{4}-0{4}-0{4}-0{12}$') {
        [void]$containers.Add("$container")
    } else {
        $normalized = $device.InstanceId -replace '&MI_[0-9A-F]{2}', ''
        [void]$fallbackIds.Add($normalized)
    }
}
$usbCount = $containers.Count + $fallbackIds.Count

[pscustomobject]@{
    cpu_temperature_c = $temperature
    cpu_temperature_source = $temperatureSource
    usb_devices = $usbCount
    usb_source = 'Windows PnP container IDs'
} | ConvertTo-Json -Compress
"""


class HardwareMonitor:
    """Read-only Windows hardware telemetry with explicit unavailable states."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        platform: str | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self._runner = runner
        self.platform = platform or os.name
        local_app_data = os.environ.get("LOCALAPPDATA")
        self.cache_path = Path(cache_path) if cache_path is not None else (
            Path(local_app_data) / "JARVIS" / "hardware.json"
            if local_app_data
            else None
        )

    def _cached_telemetry(
        self,
    ) -> tuple[float | None, str | None, int | None, str | None]:
        if self.cache_path is None or not self.cache_path.is_file():
            return None, None, None, None
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            measured = datetime.fromisoformat(str(value["measured_at"]).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - measured.astimezone(timezone.utc)).total_seconds()
            temperature = float(value["cpu_temperature_c"])
            if not 0 < temperature < 130 or age < -5 or age > 15:
                return None, None, None, None
            raw_usb = value.get("usb_devices")
            usb_devices = max(0, int(raw_usb)) if raw_usb is not None else None
            return (
                temperature,
                str(value.get("cpu_temperature_source") or "LibreHardwareMonitor"),
                usb_devices,
                str(value.get("usb_source") or "Windows PnP container IDs"),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None, None, None, None

    def read(self) -> dict[str, Any]:
        measured_at = datetime.now(timezone.utc).isoformat()
        if self.platform != "nt":
            return {
                "ok": False,
                "cpu_temperature_c": None,
                "cpu_temperature_available": False,
                "cpu_temperature_source": None,
                "usb_devices": None,
                "usb_available": False,
                "usb_source": None,
                "measured_at": measured_at,
                "error": "unsupported_platform",
            }

        cached_temperature, cached_source, cached_usb, cached_usb_source = (
            self._cached_telemetry()
        )
        if cached_temperature is not None and cached_usb is not None:
            return {
                "ok": True,
                "cpu_temperature_c": cached_temperature,
                "cpu_temperature_available": True,
                "cpu_temperature_source": cached_source,
                "usb_devices": cached_usb,
                "usb_available": True,
                "usb_source": cached_usb_source,
                "measured_at": measured_at,
                "warnings": [],
                "error": None,
            }

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = self._runner(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    _WINDOWS_TELEMETRY_SCRIPT,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "ok": False,
                "cpu_temperature_c": None,
                "cpu_temperature_available": False,
                "cpu_temperature_source": None,
                "usb_devices": None,
                "usb_available": False,
                "usb_source": None,
                "measured_at": measured_at,
                "error": "hardware_query_failed",
                "message": str(exc),
            }

        try:
            value = json.loads(completed.stdout.strip())
            if not isinstance(value, dict):
                raise ValueError("hardware response is not an object")
        except (json.JSONDecodeError, ValueError):
            return {
                "ok": False,
                "cpu_temperature_c": None,
                "cpu_temperature_available": False,
                "cpu_temperature_source": None,
                "usb_devices": None,
                "usb_available": False,
                "usb_source": None,
                "measured_at": measured_at,
                "error": "hardware_invalid_response",
            }

        raw_temperature = (
            cached_temperature
            if cached_temperature is not None
            else value.get("cpu_temperature_c")
        )
        try:
            temperature = float(raw_temperature) if raw_temperature is not None else None
        except (TypeError, ValueError):
            temperature = None
        if temperature is not None and not 0 < temperature < 130:
            temperature = None

        raw_usb = cached_usb if cached_usb is not None else value.get("usb_devices")
        try:
            usb_devices = max(0, int(raw_usb)) if raw_usb is not None else None
        except (TypeError, ValueError):
            usb_devices = None

        warnings: list[str] = []
        if temperature is None:
            warnings.append(
                "Sensor de CPU indisponível. Instale e execute LibreHardwareMonitor "
                "ou OpenHardwareMonitor com WMI habilitado."
            )
        if usb_devices is None:
            warnings.append("Inventário USB do Windows indisponível.")
        return {
            "ok": temperature is not None or usb_devices is not None,
            "cpu_temperature_c": temperature,
            "cpu_temperature_available": temperature is not None,
            "cpu_temperature_source": cached_source or value.get("cpu_temperature_source") or None,
            "usb_devices": usb_devices,
            "usb_available": usb_devices is not None,
            "usb_source": cached_usb_source or value.get("usb_source") or None,
            "measured_at": measured_at,
            "warnings": warnings,
            "error": None if temperature is not None or usb_devices is not None else "hardware_unavailable",
        }
