from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import psutil


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _work_root() -> Path:
    root = (
        _project_root()
        / ".orchestrator"
        / "light-ptbr-comparison"
        / ".work"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_helper(
    action: str,
    *,
    voice_id: str | None = None,
    interface: str = "WinRT",
    request: dict[str, Any] | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    work = _work_root()
    token = uuid.uuid4().hex
    result_path = work / f"{token}-result.json"
    request_path = work / f"{token}-request.json"
    script_path = Path(__file__).with_name("windows_speech.ps1")

    def quote(value: str | Path) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    arguments = [
        "-Action",
        quote(action),
        "-ResultJson",
        quote(result_path),
    ]
    if voice_id is not None:
        arguments.extend(
            [
                "-VoiceId",
                quote(voice_id),
                "-Interface",
                quote(interface),
            ]
        )
    if request is not None:
        request_path.write_text(
            json.dumps(request, ensure_ascii=False),
            encoding="utf-8",
        )
        arguments.extend(["-RequestJson", quote(request_path)])
    script_command = (
        "& ([scriptblock]::Create("
        f"[IO.File]::ReadAllText({quote(script_path)}))) "
        + " ".join(arguments)
    )
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        script_command,
    ]
    try:
        started = time.monotonic()
        child = subprocess.Popen(
            command,
            cwd=_project_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process = psutil.Process(child.pid)
        peak_rss = 0
        while child.poll() is None:
            if time.monotonic() - started > timeout_seconds:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                raise TimeoutError(
                    f"helper Windows excedeu {timeout_seconds}s"
                )
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.01)
        stdout, stderr_bytes = child.communicate()
        del stdout
        if child.returncode:
            stderr = stderr_bytes.decode(
                "utf-8", errors="replace"
            ).strip()
            raise RuntimeError(
                f"helper Windows falhou ({child.returncode}): {stderr}"
            )
        result = json.loads(
            result_path.read_text(encoding="utf-8-sig")
        )
        result["process"] = {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss,
        }
        return result
    finally:
        request_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def list_windows_voices() -> dict[str, Any]:
    result = _run_helper("list", timeout_seconds=30)
    for key in ("sapi", "system_speech", "winrt"):
        for voice in result.get(key, []):
            voice["pt_br"] = (
                str(voice.get("locale", "")).casefold() == "pt-br"
            )
    result["pt_br"] = [
        voice
        for key in ("winrt", "sapi", "system_speech")
        for voice in result.get(key, [])
        if voice["pt_br"]
    ]
    return result


def synthesize_windows_voice(
    *,
    voice_id: str,
    interface: str,
    output_directory: Path,
    items: list[dict[str, str]],
    rate: float = 0,
    volume: int = 100,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    if not -10 <= rate <= 10:
        raise ValueError("rate deve estar entre -10 e 10")
    if not 0 <= volume <= 100:
        raise ValueError("volume deve estar entre 0 e 100")
    return _run_helper(
        "synthesize",
        voice_id=voice_id,
        interface=interface,
        request={
            "output_directory": str(output_directory.resolve()),
            "items": items,
            "rate": rate,
            "volume": volume,
        },
        timeout_seconds=timeout_seconds,
    )
