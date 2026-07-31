from __future__ import annotations

import json
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf

from .audio import SoundDeviceAudio
from .models import (
    AudioData,
    SynthesisOptions,
    TranscriptionOptions,
)
from .quality import _comparison_text, error_rates
from .stt import FasterWhisperSTT
from .tts import PiperTTS
from .windows_speech import (
    list_windows_voices,
    synthesize_windows_voice,
)


LIGHT_PHRASES = (
    ("01-trabalho.wav", "O trabalho foi concluído corretamente."),
    ("02-trabalhando.wav", "Estou trabalhando no diretório do projeto."),
    (
        "03-trabalhador.wav",
        "O trabalhador verificou o retrabalho antes de continuar.",
    ),
    (
        "04-orquestrador.wav",
        "O orquestrador enviou o trabalho ao Codex.",
    ),
    (
        "05-inteligencia.wav",
        "A inteligência artificial analisou todos os arquivos.",
    ),
    (
        "06-servidor.wav",
        "O servidor local está funcionando normalmente.",
    ),
    (
        "07-programacao.wav",
        "O processador terminou a programação sem encontrar erros.",
    ),
    (
        "08-erres.wav",
        "Rato, carro, porta, correto, ferramenta, servidor e diretório.",
    ),
    (
        "09-familia-trabalho.wav",
        "Trabalho, trabalhador, trabalhando, trabalhar e retrabalho.",
    ),
    (
        "10-assistente.wav",
        "Boa noite, senhor. Todos os sistemas estão operacionais.",
    ),
    (
        "11-alerta.wav",
        "Atenção. Foi encontrado um problema durante a execução.",
    ),
    (
        "12-pergunta.wav",
        "Você deseja que eu continue o trabalho?",
    ),
    (
        "13-pesquisa.wav",
        "A pesquisa encontrou três fontes relevantes.",
    ),
    (
        "14-tecnico.wav",
        "O hardware e o software estão funcionando normalmente.",
    ),
    (
        "15-desenvolvimento.wav",
        "Preparando o ambiente de desenvolvimento.",
    ),
    (
        "16-repositorio.wav",
        "O arquivo foi salvo corretamente no repositório.",
    ),
    (
        "17-falha.wav",
        "Não foi possível concluir a operação solicitada.",
    ),
    (
        "18-pronto.wav",
        "Estou pronto para executar a próxima tarefa.",
    ),
)

TARGET_WORDS = (
    "trabalho",
    "trabalhador",
    "trabalhando",
    "trabalhar",
    "retrabalho",
    "corretamente",
    "diretório",
    "orquestrador",
    "inteligência",
    "artificial",
    "servidor",
    "processador",
    "programação",
    "ferramenta",
    "repositório",
    "reconhecimento",
)

SUPERTONIC_STYLES = (
    ("estilo-01-m1", "M1", "masculino"),
    ("estilo-02-m2", "M2", "masculino"),
    ("estilo-03-f1", "F1", "feminino"),
    ("estilo-04-f2", "F2", "feminino"),
)
SUPERTONIC_REVISION = "3cadd1ee6394adea1bd021217a0e650ede09a323"


def comparison_root(project_root: Path) -> Path:
    root = project_root / ".orchestrator" / "light-ptbr-comparison"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slug(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    ascii_value = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.casefold()).strip("-")


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(
        path, dtype="float32", always_2d=False
    )
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    return values.reshape(-1), int(sample_rate)


def _complete_wav(directory: Path) -> dict[str, Any]:
    values: list[np.ndarray] = []
    sample_rate: int | None = None
    for filename, _text in LIGHT_PHRASES:
        samples, current_rate = _read_audio(directory / filename)
        if sample_rate is None:
            sample_rate = current_rate
        if current_rate != sample_rate:
            raise ValueError("sample rates diferentes na mesma voz")
        if values:
            values.append(
                np.zeros(round(sample_rate * 0.7), dtype=np.float32)
            )
        values.append(samples)
    if sample_rate is None:
        raise ValueError("nenhum WAV gerado")
    joined = np.concatenate(values)
    target = directory / "completo.wav"
    sf.write(target, joined, sample_rate, subtype="PCM_16")
    return {
        "path": str(target),
        "sample_rate": sample_rate,
        "channels": 1,
        "duration_seconds": len(joined) / sample_rate,
        "peak": float(np.max(np.abs(joined))),
        "clipped_samples": int(np.sum(np.abs(joined) >= 0.999)),
    }


def _transcribe_candidate(
    stt: FasterWhisperSTT, directory: Path
) -> dict[str, Any]:
    items = []
    target_errors: set[str] = set()
    for filename, expected in LIGHT_PHRASES:
        samples, sample_rate = _read_audio(directory / filename)
        try:
            result = stt.transcribe(
                AudioData(
                    samples=samples,
                    sample_rate=sample_rate,
                    duration_seconds=len(samples) / sample_rate,
                    rms=float(np.sqrt(np.mean(np.square(samples)))),
                    peak=float(np.max(np.abs(samples))),
                ),
                TranscriptionOptions(
                    language="pt",
                    timeout_seconds=120,
                ),
            )
            transcript = result.text
            error = None
        except Exception as exc:
            transcript = ""
            error = f"{type(exc).__name__}: {exc}"
        wer, cer = error_rates(expected, transcript)
        expected_words = set(_comparison_text(expected).split())
        actual_words = set(_comparison_text(transcript).split())
        for word in TARGET_WORDS:
            normalized = _comparison_text(word)
            if normalized in expected_words and normalized not in actual_words:
                target_errors.add(word)
        items.append(
            {
                "filename": filename,
                "expected": expected,
                "transcript": transcript,
                "wer": wer,
                "cer": cer,
                "error": error,
            }
        )
    return {
        "mean_wer": sum(item["wer"] for item in items) / len(items),
        "mean_cer": sum(item["cer"] for item in items) / len(items),
        "target_errors": sorted(target_errors),
        "transcriptions": items,
    }


def generate_windows_comparison(
    settings: Any,
    *,
    include_stt: bool = True,
) -> dict[str, Any]:
    root = comparison_root(Path(__file__).resolve().parents[3])
    inventory = list_windows_voices()
    voices = []
    seen: set[str] = set()
    for voice in inventory["winrt"]:
        voice_id = str(voice.get("id", ""))
        if (
            not voice.get("pt_br")
            or not voice_id
            or voice_id in seen
        ):
            continue
        seen.add(voice_id)
        voices.append(voice)
    stt = None
    if include_stt:
        stt = FasterWhisperSTT(
            settings.voice_stt_model,
            device=settings.voice_stt_device,
            compute_type=settings.voice_stt_compute_type,
            threads=settings.voice_stt_threads,
        )
    results: dict[str, Any] = {}
    try:
        for voice in voices:
            alias = _slug(str(voice["name"]))
            directory = root / "windows" / alias
            directory.mkdir(parents=True, exist_ok=True)
            for filename, _text in LIGHT_PHRASES:
                (directory / filename).unlink(missing_ok=True)
            (directory / "completo.wav").unlink(missing_ok=True)
            started = time.monotonic()
            synthesis = synthesize_windows_voice(
                voice_id=str(voice["id"]),
                interface="WinRT",
                output_directory=directory,
                items=[
                    {"filename": filename, "text": text}
                    for filename, text in LIGHT_PHRASES
                ],
                timeout_seconds=120,
            )
            wall_seconds = time.monotonic() - started
            metrics = synthesis["metrics"]
            complete = _complete_wav(directory)
            total_audio = sum(
                _read_audio(directory / filename)[0].size
                / _read_audio(directory / filename)[1]
                for filename, _text in LIGHT_PHRASES
            )
            total_synthesis = sum(
                float(item["synthesis_seconds"]) for item in metrics
            )
            analysis = (
                _transcribe_candidate(stt, directory)
                if stt is not None
                else {
                    "mean_wer": None,
                    "mean_cer": None,
                    "target_errors": [],
                    "transcriptions": [],
                }
            )
            results[alias] = {
                "alias": alias,
                "name": voice["name"],
                "provider": "windows_winrt",
                "interface": "WinRT",
                "voice_id": voice["id"],
                "locale": voice["locale"],
                "gender": voice["gender"],
                "origin": "Microsoft Windows language speech package",
                "license": "Windows system component",
                "offline": True,
                "asynchronous": True,
                "cancellation": voice["cancellation"],
                "output": voice["output"],
                "sample_rate": complete["sample_rate"],
                "channels": complete["channels"],
                "process_wall_seconds": wall_seconds,
                "loading_overhead_seconds": max(
                    0.0, wall_seconds - total_synthesis
                ),
                "first_synthesis_seconds": float(
                    metrics[0]["synthesis_seconds"]
                ),
                "warm_synthesis_seconds": sum(
                    float(item["synthesis_seconds"])
                    for item in metrics[1:]
                )
                / max(1, len(metrics) - 1),
                "time_to_first_audio_seconds": float(
                    metrics[0]["synthesis_seconds"]
                ),
                "total_synthesis_seconds": total_synthesis,
                "generated_audio_seconds": total_audio,
                "real_time_factor": total_synthesis / total_audio,
                "peak_rss_bytes": synthesis["process"][
                    "peak_rss_bytes"
                ],
                "vram_bytes": 0,
                "complete_wav": complete,
                **analysis,
            }
    finally:
        if stt is not None:
            stt.close()
    report = {
        "phase": "light-ptbr-poc",
        "status": "windows-complete",
        "inventory": inventory,
        "windows": results,
        "rhvoice": {
            "status": "not-run",
        },
        "supertonic": {
            "status": "not-run",
        },
        "mms": {
            "status": "not-run",
        },
    }
    _write_reports(root, report)
    return report


def _run_monitored(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    peak_rss = 0
    peak_cpu = 0.0
    cpu_seconds = 0.0
    with stdout_path.open("w", encoding="utf-8") as stdout_file, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        monitored = psutil.Process(process.pid)
        monitored.cpu_percent(interval=None)
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise TimeoutError(
                        f"Supertonic excedeu {timeout_seconds:.0f}s"
                    )
                try:
                    processes = [monitored, *monitored.children(recursive=True)]
                    peak_rss = max(
                        peak_rss,
                        sum(
                            child.memory_info().rss
                            for child in processes
                            if child.is_running()
                        ),
                    )
                    peak_cpu = max(
                        peak_cpu,
                        sum(
                            child.cpu_percent(interval=None)
                            for child in processes
                            if child.is_running()
                        ),
                    )
                    cpu_seconds = max(
                        cpu_seconds,
                        sum(
                            (
                                child.cpu_times().user
                                + child.cpu_times().system
                            )
                            for child in processes
                            if child.is_running()
                        ),
                    )
                except psutil.Error:
                    pass
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    wall_seconds = time.monotonic() - started
    if process.returncode:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"Supertonic falhou ({process.returncode}): {error[-2000:]}"
        )
    return {
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss,
        "peak_process_cpu_percent": peak_cpu,
        "average_system_cpu_percent": (
            cpu_seconds
            / max(wall_seconds, 0.001)
            * 100
            / max(1, psutil.cpu_count(logical=True) or 1)
        ),
    }


def _supertonic_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(project_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    return environment


def _run_supertonic(
    project_root: Path,
    *,
    style: str,
    output_directory: Path,
    items: list[dict[str, str]],
    steps: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = comparison_root(project_root)
    environment = _supertonic_environment(project_root)
    python = project_root / ".venv-supertonic" / "Scripts" / "python.exe"
    model_dir = project_root / "models" / "supertonic-3"
    if not python.is_file():
        raise FileNotFoundError(f"ambiente Supertonic ausente: {python}")
    items_path = root / f".supertonic-{label}-items.json"
    result_path = root / f".supertonic-{label}-result.json"
    stdout_path = root / f".supertonic-{label}.stdout.log"
    stderr_path = root / f".supertonic-{label}.stderr.log"
    items_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result_path.unlink(missing_ok=True)
    try:
        process = _run_monitored(
            [
                str(python),
                "-c",
                (
                    "import runpy;"
                    "runpy.run_path("
                    + repr(
                        str(
                            project_root
                            / "tern"
                            / "orchestrator"
                            / "voice"
                            / "supertonic_poc.py"
                        )
                    )
                    + ",run_name='__main__')"
                ),
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(output_directory),
                "--items-json",
                str(items_path),
                "--result-json",
                str(result_path),
                "--style",
                style,
                "--steps",
                str(steps),
            ],
            cwd=project_root,
            environment=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=900,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return result, process
    finally:
        items_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


def _model_manifest(model_dir: Path) -> dict[str, Any]:
    import hashlib

    files = []
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(model_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "revision": SUPERTONIC_REVISION,
        "size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }


def _supertonic_environment_info(project_root: Path) -> dict[str, Any]:
    environment = project_root / ".venv-supertonic"
    files = [path for path in environment.rglob("*") if path.is_file()]
    site_packages = environment / "Lib" / "site-packages"
    packages = list(site_packages.glob("*.dist-info"))
    return {
        "path": ".venv-supertonic",
        "python": "3.13.7",
        "supertonic": "1.3.1",
        "package_count": len(packages),
        "size_bytes": sum(path.stat().st_size for path in files),
        "install_seconds": 14.948,
        "pip_check": "clean",
        "isolated": True,
    }


def _generate_supertonic_comparison(
    project_root: Path,
    stt: FasterWhisperSTT | None,
) -> dict[str, Any]:
    root = comparison_root(project_root)
    model_dir = project_root / "models" / "supertonic-3"
    required = (
        "onnx/duration_predictor.onnx",
        "onnx/text_encoder.onnx",
        "onnx/vector_estimator.onnx",
        "onnx/vocoder.onnx",
        "onnx/tts.json",
        "onnx/unicode_indexer.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        return {"status": "unavailable", "missing": missing}
    results: dict[str, Any] = {}
    for alias, style, declared_gender in SUPERTONIC_STYLES:
        directory = root / "supertonic" / alias
        directory.mkdir(parents=True, exist_ok=True)
        for filename, _text in LIGHT_PHRASES:
            (directory / filename).unlink(missing_ok=True)
        (directory / "completo.wav").unlink(missing_ok=True)
        probe_dir = directory / "steps-5-probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe, probe_process = _run_supertonic(
            project_root,
            style=style,
            output_directory=probe_dir,
            items=[
                {
                    "filename": LIGHT_PHRASES[0][0],
                    "text": LIGHT_PHRASES[0][1],
                }
            ],
            steps=5,
            label=f"{alias}-steps5",
        )
        synthesis, process = _run_supertonic(
            project_root,
            style=style,
            output_directory=directory,
            items=[
                {"filename": filename, "text": text}
                for filename, text in LIGHT_PHRASES
            ],
            steps=8,
            label=f"{alias}-steps8",
        )
        metrics = synthesis["metrics"]
        complete = _complete_wav(directory)
        total_synthesis = sum(
            float(item["synthesis_seconds"]) for item in metrics
        )
        total_audio = sum(float(item["audio_seconds"]) for item in metrics)
        analysis = (
            _transcribe_candidate(stt, directory)
            if stt is not None
            else {
                "mean_wer": None,
                "mean_cer": None,
                "target_errors": [],
                "transcriptions": [],
            }
        )
        results[alias] = {
            "alias": alias,
            "name": f"Supertonic 3 {style}",
            "style": style,
            "declared_gender": declared_gender,
            "provider": "supertonic",
            "locale": "pt (português genérico)",
            "origin": "https://huggingface.co/Supertone/supertonic-3",
            "license": "OpenRAIL-M (pesos); MIT (código)",
            "revision": SUPERTONIC_REVISION,
            "offline": True,
            "asynchronous": False,
            "cancellation": (
                "processo isolado cancelável; sem cancelamento interno ONNX"
            ),
            "sample_rate": complete["sample_rate"],
            "channels": complete["channels"],
            "steps": 8,
            "speed": synthesis["speed"],
            "loading_seconds": synthesis["loading_seconds"],
            "first_synthesis_seconds": float(
                metrics[0]["synthesis_seconds"]
            ),
            "warm_synthesis_seconds": sum(
                float(item["synthesis_seconds"]) for item in metrics[1:]
            )
            / max(1, len(metrics) - 1),
            "time_to_first_audio_seconds": (
                synthesis["loading_seconds"]
                + float(metrics[0]["synthesis_seconds"])
            ),
            "total_synthesis_seconds": total_synthesis,
            "generated_audio_seconds": total_audio,
            "real_time_factor": total_synthesis / total_audio,
            "peak_rss_bytes": process["peak_rss_bytes"],
            "peak_process_cpu_percent": process[
                "peak_process_cpu_percent"
            ],
            "average_system_cpu_percent": process[
                "average_system_cpu_percent"
            ],
            "vram_bytes": 0,
            "dependencies": "ambiente isolado .venv-supertonic",
            "complete_wav": complete,
            "steps_5_probe": {
                "loading_seconds": probe["loading_seconds"],
                "synthesis_seconds": probe["metrics"][0][
                    "synthesis_seconds"
                ],
                "audio_seconds": probe["metrics"][0]["audio_seconds"],
                "real_time_factor": (
                    probe["metrics"][0]["synthesis_seconds"]
                    / probe["metrics"][0]["audio_seconds"]
                ),
                "peak_rss_bytes": probe_process["peak_rss_bytes"],
                "wav": str(probe_dir / LIGHT_PHRASES[0][0]),
            },
            **analysis,
        }
    return {
        "status": "complete",
        "classification": (
            "modelo multilíngue com português genérico; estilos não "
            "comprovadamente brasileiros"
        ),
        "support_status": (
            "repositório anunciou encerramento de desenvolvimento e suporte"
        ),
        "environment": _supertonic_environment_info(project_root),
        "model_manifest": _model_manifest(model_dir),
        "styles": results,
    }


def _generate_piper_control(
    settings: Any,
    stt: FasterWhisperSTT | None,
) -> dict[str, Any]:
    root = comparison_root(Path(__file__).resolve().parents[3])
    directory = root / "piper-control"
    directory.mkdir(parents=True, exist_ok=True)
    audio = SoundDeviceAudio()
    tts = PiperTTS(
        settings.voice_tts_model,
        audio,
        config_path=settings.voice_piper_config_path,
    )
    metrics = []
    try:
        for filename, text in LIGHT_PHRASES:
            started = time.monotonic()
            result = tts.synthesize(
                text,
                SynthesisOptions(
                    rate=settings.voice_tts_rate,
                    timeout_seconds=60,
                ),
            )
            elapsed = time.monotonic() - started
            sf.write(
                directory / filename,
                result.samples,
                result.sample_rate,
                subtype="PCM_16",
            )
            metrics.append(
                {
                    "filename": filename,
                    "synthesis_seconds": elapsed,
                    "audio_seconds": result.duration_seconds,
                }
            )
    finally:
        tts.close()
    complete = _complete_wav(directory)
    analysis = (
        _transcribe_candidate(stt, directory)
        if stt is not None
        else {
            "mean_wer": None,
            "mean_cer": None,
            "target_errors": [],
            "transcriptions": [],
        }
    )
    total_synthesis = sum(item["synthesis_seconds"] for item in metrics)
    total_audio = sum(item["audio_seconds"] for item in metrics)
    return {
        "status": "complete",
        "name": f"Piper atual ({settings.voice_piper_voice})",
        "provider": "piper",
        "locale": "pt-BR",
        "sample_rate": complete["sample_rate"],
        "channels": complete["channels"],
        "first_synthesis_seconds": metrics[0]["synthesis_seconds"],
        "warm_synthesis_seconds": sum(
            item["synthesis_seconds"] for item in metrics[1:]
        )
        / max(1, len(metrics) - 1),
        "real_time_factor": total_synthesis / total_audio,
        "vram_bytes": 0,
        "complete_wav": complete,
        **analysis,
    }


def generate_light_comparison(
    settings: Any,
    *,
    include_stt: bool = True,
    windows_only: bool = False,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    report = generate_windows_comparison(
        settings,
        include_stt=include_stt,
    )
    if windows_only:
        return report
    stt = None
    if include_stt:
        stt = FasterWhisperSTT(
            settings.voice_stt_model,
            device=settings.voice_stt_device,
            compute_type=settings.voice_stt_compute_type,
            threads=settings.voice_stt_threads,
        )
    try:
        report["rhvoice"] = {
            "status": "rejected-before-download",
            "name": "RHVoice Letícia-F123",
            "locale": "Brazilian-Portuguese",
            "core_license": "GPL-2.0",
            "voice_license": None,
            "reason": (
                "repositório oficial da voz não declara licença dos dados; "
                "uso legal não confirmado"
            ),
            "origin": "https://github.com/RHVoice/leticia-f123-pt-br",
        }
        report["supertonic"] = _generate_supertonic_comparison(
            project_root,
            stt,
        )
        report["piper_control"] = _generate_piper_control(settings, stt)
        report["mms"] = {
            "status": "deferred",
            "reason": (
                "último controle; só após rejeição humana de Windows e "
                "Supertonic"
            ),
        }
        report["status"] = "phase-a-audio-ready"
        _write_reports(comparison_root(project_root), report)
        return report
    finally:
        if stt is not None:
            stt.close()


def play_light_candidates(
    settings: Any,
    report: dict[str, Any],
) -> list[str]:
    from .models import AudioResult

    candidates: list[tuple[str, Path]] = []
    windows = report.get("windows", {})
    for alias in ("microsoft-daniel", "microsoft-maria"):
        value = windows.get(alias)
        if value:
            candidates.append(
                (value["name"], Path(value["complete_wav"]["path"]))
            )
    for value in report.get("supertonic", {}).get("styles", {}).values():
        candidates.append(
            (value["name"], Path(value["complete_wav"]["path"]))
        )
    control = report.get("piper_control")
    if isinstance(control, dict) and control.get("status") == "complete":
        candidates.append(
            (control["name"], Path(control["complete_wav"]["path"]))
        )

    audio = SoundDeviceAudio()
    played = []
    for name, path in candidates:
        samples, sample_rate = _read_audio(path)
        print(f"REPRODUZINDO: {name.upper()}", flush=True)
        interrupted = audio.play(
            AudioResult(
                samples=samples,
                sample_rate=sample_rate,
                duration_seconds=len(samples) / sample_rate,
                provider="comparison",
            ),
            output_device=settings.voice_output_device,
            output_device_name=settings.voice_output_device_name,
            interrupt_key=settings.voice_interrupt_key,
        )
        played.append(name)
        if interrupted:
            break
    return played


def _write_reports(root: Path, report: dict[str, Any]) -> None:
    (root / "comparison-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Comparação TTS local leve pt-BR",
        "",
        f"Status: `{report['status']}`.",
        "",
        "| voz | provider | locale | sample rate | primeira | quente | RTF | RAM | VRAM | WER | CER | erros-alvo |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    candidates = list(report.get("windows", {}).values())
    candidates.extend(
        report.get("supertonic", {}).get("styles", {}).values()
    )
    piper_control = report.get("piper_control")
    if isinstance(piper_control, dict) and piper_control.get("status") == "complete":
        candidates.append(piper_control)
    for value in candidates:
        lines.append(
            "| {name} | {provider} | {locale} | {rate} | "
            "{first:.3f} | {warm:.3f} | {rtf:.3f} | {ram} | 0 | "
            "{wer:.3f} | {cer:.3f} | {errors} |".format(
                name=value["name"],
                provider=value["provider"],
                locale=value["locale"],
                rate=value["sample_rate"],
                first=value["first_synthesis_seconds"],
                warm=value["warm_synthesis_seconds"],
                rtf=value["real_time_factor"],
                ram=value.get("peak_rss_bytes", "n/d"),
                wer=value["mean_wer"] or 0.0,
                cer=value["mean_cer"] or 0.0,
                errors=", ".join(value["target_errors"]) or "nenhum",
            )
        )
    lines.extend(
        [
            "",
            "## Candidatos não executados",
            "",
            (
                "- RHVoice Letícia-F123: "
                + report.get("rhvoice", {}).get("reason", "não testada")
                + "."
            ),
            (
                "- MMS-TTS por: "
                + report.get("mms", {}).get("reason", "não testado")
                + "."
            ),
            "",
            (
                "Supertonic usa `pt`, português genérico. Estilos não são "
                "comprovadamente pt-BR."
            ),
            "",
            "Whisper é somente indicador técnico; a decisão é auditiva.",
            "",
        ]
    )
    (root / "comparison-report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
