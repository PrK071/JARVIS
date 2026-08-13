from __future__ import annotations

import unicodedata
from typing import Any

from .errors import AudioInputNotFound, AudioOutputNotFound
from .models import DeviceInfo


def list_devices(sounddevice_module: Any) -> list[DeviceInfo]:
    raw_devices = sounddevice_module.query_devices()
    try:
        host_apis = sounddevice_module.query_hostapis()
    except Exception:
        host_apis = ()
    devices = []
    for index, raw in enumerate(raw_devices):
        host_api = None
        host_index = int(raw.get("hostapi", -1))
        if 0 <= host_index < len(host_apis):
            host_api = str(host_apis[host_index].get("name") or "") or None
        devices.append(
            DeviceInfo(
                index=int(raw.get("index", index)),
                name=str(raw.get("name", f"device-{index}")),
                input_channels=int(raw.get("max_input_channels", 0)),
                output_channels=int(raw.get("max_output_channels", 0)),
                default_sample_rate=int(raw.get("default_samplerate", 0)),
                host_api=host_api,
            )
        )
    return devices


def select_device(
    devices: list[DeviceInfo],
    selector: str | int | None,
    *,
    direction: str,
    default_index: int | None = None,
    preferred_name: str | None = None,
) -> DeviceInfo:
    return resolve_device(
        devices,
        selector,
        direction=direction,
        default_index=default_index,
        preferred_name=preferred_name,
    )[0]


def _normalized_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(
            char
            for char in normalized
            if not unicodedata.combining(char)
        ).split()
    )


def resolve_device(
    devices: list[DeviceInfo],
    selector: str | int | None,
    *,
    direction: str,
    default_index: int | None = None,
    preferred_name: str | None = None,
) -> tuple[DeviceInfo, str]:
    channel = (
        (lambda item: item.input_channels)
        if direction == "input"
        else (lambda item: item.output_channels)
    )
    error_type = (
        AudioInputNotFound if direction == "input" else AudioOutputNotFound
    )
    candidates = [item for item in devices if channel(item) > 0]
    name_value = (preferred_name or "").strip()
    legacy_value = str(selector).strip() if selector is not None else ""
    if not name_value and legacy_value and not legacy_value.lstrip("-").isdigit():
        name_value = legacy_value
    if name_value:
        exact_identity = [
            item for item in candidates if item.identity == name_value
        ]
        if len(exact_identity) == 1:
            return exact_identity[0], "exact_name"
        exact_name = [
            item for item in candidates if item.name == name_value
        ]
        if len(exact_name) == 1:
            return exact_name[0], "exact_name"
        if len(exact_name) > 1:
            raise error_type(
                f"dispositivo de {direction} duplicado: {name_value}",
                details={"matches": [item.as_dict() for item in exact_name]},
            )
        normalized = _normalized_name(name_value)
        normalized_matches = [
            item
            for item in candidates
            if _normalized_name(item.identity) == normalized
            or _normalized_name(item.name) == normalized
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0], "normalized_name"
        if len(normalized_matches) > 1:
            raise error_type(
                f"dispositivo de {direction} ambiguo: {name_value}",
                details={
                    "matches": [
                        item.as_dict() for item in normalized_matches
                    ]
                },
            )
        partial = [
            item
            for item in candidates
            if normalized in _normalized_name(item.identity)
        ]
        if len(partial) == 1:
            return partial[0], "normalized_name"
        if len(partial) > 1:
            raise error_type(
                f"dispositivo de {direction} ambiguo: {name_value}",
                details={"matches": [item.as_dict() for item in partial]},
            )
    if legacy_value.lstrip("-").isdigit():
        index = int(legacy_value)
        for item in candidates:
            if item.index == index:
                return item, "explicit_id"
    if default_index is not None:
        for item in candidates:
            if item.index == default_index:
                return item, "default"
    if not candidates:
        raise error_type(f"nenhum dispositivo de {direction} disponivel")
    requested = name_value or legacy_value or "<padrao>"
    raise error_type(
        f"dispositivo de {direction} nao encontrado: {requested}",
        details={"available": [item.as_dict() for item in candidates]},
    )
