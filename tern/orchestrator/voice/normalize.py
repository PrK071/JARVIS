from __future__ import annotations

import json
import re
from pathlib import Path


_CODE_BLOCK_RE = re.compile(r"```([\s\S]*?)```")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_URL_RE = re.compile(r"https?://\S+", re.I)
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s\"<>|]+")
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_PIP_RE = re.compile(r"\bpip(?:3)?\s+install\s+([A-Za-z0-9_.\-\[\],]+)", re.I)


def _spoken_code_content(value: str, *, block: bool = False) -> str:
    content = value.strip()
    if block and "\n" in content:
        first_line, remainder = content.split("\n", 1)
        if re.fullmatch(r"[A-Za-z0-9_+.-]{1,30}", first_line.strip()):
            content = remainder.strip()
    return content.replace("_", " ")


def unwrap_markdown_code(value: str) -> str:
    """Remove Markdown backticks while preserving their content for speech."""
    value = _CODE_BLOCK_RE.sub(
        lambda match: _spoken_code_content(match.group(1), block=True),
        value,
    )
    return _INLINE_CODE_RE.sub(
        lambda match: _spoken_code_content(match.group(1)),
        value,
    )


def _load_lexicon(path: Path | None, provider: str) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    selected = value.get(provider, {}) if isinstance(value, dict) else {}
    return selected if isinstance(selected, dict) else {}


def normalize_for_speech(
    text: str,
    provider: str,
    style: str,
    *,
    lexicon_path: Path | None = None,
) -> str:
    value = text.strip()
    paragraph_token = "TERNPARAGRAPHPAUSE"
    value = re.sub(r"\n\s*\n+", f" {paragraph_token} ", value)
    value = _MARKDOWN_LINK_RE.sub(r"\1", value)
    value = unwrap_markdown_code(value)
    value = _PIP_RE.sub(
        lambda match: (
            f"Instale o pacote {match.group(1).replace('-', ' ')} "
            "usando o comando exibido na tela."
        ),
        value,
    )
    value = _WINDOWS_PATH_RE.sub(_spoken_path, value)
    value = _URL_RE.sub("o link exibido na tela", value)
    value = _HASH_RE.sub("o identificador exibido na tela", value)
    value = re.sub(r"(?m)^\s*#{1,6}\s*", "", value)
    value = re.sub(r"[*_>|~]+", " ", value)
    value = re.sub(r"(?m)^\s*[-+]\s+", "", value)
    value = re.sub(r"(?m)^\s*(\d+)[.)]\s+", r"\1. ", value)
    for source, replacement in _load_lexicon(
        lexicon_path, provider
    ).items():
        value = value.replace(source, replacement)
    replacements = {
        r"\bJSON\b": "jêison",
        r"\bAPI\b": "a pê i",
        r"\bGGUF\b": "gê gê u éfe",
        r"\bCPU\b": "C P U",
        r"\bGPU\b": "G P U",
        r"\bVRAM\b": "V RAM",
        r"\bRAM\b": "RAM",
        r"\bPowerShell\b": "Páuer Chél",
    }
    for pattern, replacement in replacements.items():
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*%", r"\1 por cento", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n+\s*", ". ", value)
    value = value.replace(paragraph_token, "\n\n")
    value = re.sub(r"\.{2,}", ".", value)
    value = value.strip(" .") + ("." if value.strip(" .") else "")
    if style == "jarvis":
        value = re.sub(r"!+", ".", value)
    return value


def _spoken_path(match: re.Match[str]) -> str:
    path = match.group(0).casefold()
    if "\\voice" in path or "\\voz" in path:
        return "o módulo de voz. O caminho completo está na tela"
    return "o caminho completo exibido na tela"
