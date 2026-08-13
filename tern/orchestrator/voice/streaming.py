from __future__ import annotations

import re

from .normalize import unwrap_markdown_code


_URL_RE = re.compile(r"https?://\S+", re.I)
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s,;]+")
_ABBREVIATION_RE = re.compile(
    r"\b(?:Dr|Dra|Sr|Sra|Srta|Prof|Profa|etc|ex|pág|vol|vs)\.",
    re.I,
)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")


def _split_long(value: str, maximum: int) -> list[str]:
    result = []
    remaining = value.strip()
    while len(remaining) > maximum:
        cut = max(
            remaining.rfind("; ", 0, maximum + 1),
            remaining.rfind(", ", 0, maximum + 1),
            remaining.rfind(" ", 0, maximum + 1),
        )
        if cut < maximum // 2:
            cut = maximum
        result.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip(" ,;")
    if remaining:
        result.append(remaining)
    return result


def segment_for_speech(
    text: str,
    *,
    minimum: int = 40,
    maximum: int = 280,
) -> list[str]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("limites de segmento invalidos")
    value = unwrap_markdown_code(text)
    value = _URL_RE.sub(" fonte disponível na tela. ", value)
    value = _WINDOWS_PATH_RE.sub(" caminho disponível na tela ", value)
    value = re.sub(
        r"([.!?])\s*\n\s*\n+",
        r" TERNPARAGRAPH\1 ",
        value,
    )
    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        key = f"TERNPROTECTED{len(protected)}TOKEN"
        protected[key] = match.group(0).replace(".", "TERNPONTO")
        return key

    value = _ABBREVIATION_RE.sub(protect, value)
    value = _DECIMAL_RE.sub("TERNPONTO", value)
    value = re.sub(r"(?m)^\s*[-*•]\s+", "", value)
    value = re.sub(r"(?m)^\s*\d+[.)]\s+", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    raw = [
        item.strip(" \r\n")
        for item in re.split(r"(?<=[.!?])\s+|\n+", value)
        if item.strip()
    ]
    pieces: list[str] = []
    for item in raw:
        for key, original in protected.items():
            item = item.replace(key, original)
        item = item.replace("TERNPONTO", ".")
        item = item.replace("TERNPARAGRAPH", "\n\n")
        pieces.extend(_split_long(item, maximum))
    chunks: list[str] = []
    pending = ""
    for piece in pieces:
        if not pending:
            pending = piece
            continue
        combined = f"{pending} {piece}".strip()
        if len(pending) < minimum and len(combined) <= maximum:
            pending = combined
        else:
            chunks.append(pending)
            pending = piece
    if pending:
        if chunks and len(pending) < minimum:
            combined = f"{chunks[-1]} {pending}"
            if len(combined) <= maximum:
                chunks[-1] = combined
            else:
                chunks.append(pending)
        else:
            chunks.append(pending)
    return [item for item in chunks if item]
