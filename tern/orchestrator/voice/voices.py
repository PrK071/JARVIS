from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PiperVoiceSpec:
    alias: str
    model_path: Path
    config_path: Path
    requested_alias: str
    fallback: bool = False

    @property
    def available(self) -> bool:
        return self.model_path.is_file() and self.config_path.is_file()


def piper_voice_aliases(project_root: Path) -> dict[str, tuple[Path, Path]]:
    voice_root = project_root / "models" / "voice"
    piper_root = project_root / "models" / "piper"
    return {
        "faber": (
            voice_root / "pt_BR-faber-medium.onnx",
            voice_root / "pt_BR-faber-medium.onnx.json",
        ),
        "miro": (
            piper_root / "pt_BR-miro" / "miro_pt-BR.onnx",
            piper_root / "pt_BR-miro" / "miro_pt-BR.onnx.json",
        ),
        "jeff": (
            voice_root / "pt_BR-jeff-medium.onnx",
            voice_root / "pt_BR-jeff-medium.onnx.json",
        ),
        "cadu": (
            voice_root / "pt_BR-cadu-medium.onnx",
            voice_root / "pt_BR-cadu-medium.onnx.json",
        ),
        "dii": (
            piper_root / "pt_BR-dii" / "dii_pt-BR.onnx",
            piper_root / "pt_BR-dii" / "dii_pt-BR.onnx.json",
        ),
    }


def validate_piper_voice_pair(
    model_path: Path,
    config_path: Path,
    *,
    require_files: bool = True,
) -> dict[str, object]:
    model_path = model_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if require_files and not model_path.is_file():
        raise ValueError(f"modelo Piper ausente: {model_path}")
    if require_files and not config_path.is_file():
        raise ValueError(f"configuração Piper ausente: {config_path}")
    if config_path.name != model_path.name + ".json":
        raise ValueError("modelo e JSON Piper não formam um par correspondente")
    if not config_path.is_file():
        return {}
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
        sample_rate = int(value["audio"]["sample_rate"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON Piper inválido: {config_path}") from exc
    language = str(
        (value.get("language") or {}).get("code")
        or (value.get("espeak") or {}).get("voice")
        or ""
    ).casefold()
    if language not in {"pt-br", "pt_br"}:
        raise ValueError(
            f"voz Piper não é pt-BR: {language or '<não informado>'}"
        )
    if sample_rate < 8000:
        raise ValueError(f"sample rate Piper inválido: {sample_rate}")
    return {
        "sample_rate": sample_rate,
        "language": language,
        "num_speakers": int(value.get("num_speakers", 1)),
    }


def resolve_piper_voice(
    values: Mapping[str, str],
    project_root: Path,
    *,
    default_alias: str = "miro",
) -> PiperVoiceSpec:
    aliases = piper_voice_aliases(project_root)
    explicit_model = values.get("VOICE_PIPER_MODEL_PATH")
    explicit_config = values.get("VOICE_PIPER_CONFIG_PATH")
    if explicit_config and not explicit_model:
        raise ValueError(
            "VOICE_PIPER_CONFIG_PATH exige VOICE_PIPER_MODEL_PATH"
        )
    if explicit_model:
        model_path = Path(explicit_model).expanduser().resolve()
        config_path = (
            Path(explicit_config).expanduser().resolve()
            if explicit_config
            else Path(str(model_path) + ".json")
        )
        validate_piper_voice_pair(model_path, config_path)
        alias = next(
            (
                name
                for name, pair in aliases.items()
                if pair[0].resolve() == model_path
                and pair[1].resolve() == config_path
            ),
            "custom",
        )
        return PiperVoiceSpec(
            alias=alias,
            model_path=model_path,
            config_path=config_path,
            requested_alias=alias,
        )

    # Preserve the former explicit model variable without making it the new
    # preferred interface.
    legacy_model = values.get("VOICE_TTS_MODEL")
    if legacy_model:
        model_path = Path(legacy_model).expanduser().resolve()
        config_path = Path(str(model_path) + ".json")
        return PiperVoiceSpec(
            alias="custom",
            model_path=model_path,
            config_path=config_path,
            requested_alias="custom",
        )

    requested = (
        values.get("VOICE_PIPER_VOICE")
        or _legacy_alias(values.get("VOICE_TTS_VOICE", ""))
        or default_alias
    ).strip().casefold()
    if requested not in aliases:
        allowed = ", ".join(aliases)
        raise ValueError(
            f"VOICE_PIPER_VOICE inválida: {requested!r}; use {allowed}"
        )

    candidates = [requested]
    if default_alias not in candidates:
        candidates.append(default_alias)
    if "faber" not in candidates:
        candidates.append("faber")
    for alias in candidates:
        model_path, config_path = aliases[alias]
        if model_path.is_file() and config_path.is_file():
            validate_piper_voice_pair(model_path, config_path)
            return PiperVoiceSpec(
                alias=alias,
                model_path=model_path.resolve(),
                config_path=config_path.resolve(),
                requested_alias=requested,
                fallback=alias != requested,
            )

    model_path, config_path = aliases[requested]
    return PiperVoiceSpec(
        alias=requested,
        model_path=model_path.resolve(),
        config_path=config_path.resolve(),
        requested_alias=requested,
    )


def _legacy_alias(value: str) -> str | None:
    normalized = value.strip().casefold()
    for alias in ("faber", "miro", "jeff", "cadu", "dii"):
        if alias in normalized:
            return alias
    return None
