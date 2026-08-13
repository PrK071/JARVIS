from __future__ import annotations

import re
from enum import Enum
from typing import Callable

from .normalize import unwrap_markdown_code


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DANGEROUS_RE = re.compile(
    r"\b("
    r"apag(?:ar|ue)|delet(?:ar|e)|remov(?:er|a)|sobrescrev(?:er|a)|"
    r"format(?:ar|e)|instal(?:ar|e)|desinstal(?:ar|e)|"
    r"administrador|powershell|registro do windows|configura(?:r|ção) do sistema"
    r")\b",
    re.IGNORECASE,
)


class ConfirmationDecision(str, Enum):
    SEND = "send"
    RERECORD = "rerecord"
    CANCEL = "cancel"


class ConsoleIO:
    def write(self, value: str = "") -> None:
        print(value, flush=True)

    def read(self, prompt: str = "") -> str:
        return input(prompt)


def may_be_sensitive(text: str) -> bool:
    return bool(_DANGEROUS_RE.search(text))


def confirm_transcription(
    text: str,
    *,
    required: bool,
    console: ConsoleIO,
) -> ConfirmationDecision:
    if not required:
        return ConfirmationDecision.SEND
    console.write('Você disse:\n"' + text + '"')
    value = console.read(
        "Confirma? [S] enviar  [R] gravar novamente  [C] cancelar: "
    ).strip().casefold()
    if value in {"s", "sim", "send", "enviar"}:
        return ConfirmationDecision.SEND
    if value in {"r", "regravar", "gravar"}:
        return ConfirmationDecision.RERECORD
    return ConfirmationDecision.CANCEL


class VoiceActionApprover:
    def __init__(self, console: ConsoleIO):
        self.console = console

    def __call__(self, action: str, arguments: dict) -> bool:
        path = str(arguments.get("path") or "<não informado>")
        if action == "delete":
            tool = "filesystem_delete"
            impact = "remove arquivo"
            reversible = "não garantido"
        elif action == "overwrite":
            tool = "filesystem_write_text"
            impact = "substitui conteúdo existente"
            reversible = "não garantido"
        elif action == "codex_modify_files":
            tool = "codex_delegate"
            impact = "pode criar ou modificar arquivos no workspace"
            reversible = "depende do controle de versão ou backup"
        elif action == "install_software":
            tool = "codex_delegate"
            impact = "instala software ou pacotes"
            reversible = "pode exigir desinstalação separada"
        elif action == "remove_software":
            tool = "codex_delegate"
            impact = "remove software ou pacotes"
            reversible = "pode exigir reinstalação"
        elif action == "system_change":
            tool = "codex_delegate"
            impact = "altera configuração do sistema"
            reversible = "não garantido"
        elif action == "administrative":
            tool = "codex_delegate"
            impact = "executa operação administrativa"
            reversible = "não garantido"
        else:
            tool = "unknown"
            impact = action
            reversible = "desconhecido"
        self.console.write("[confirmação obrigatória]")
        self.console.write(f"ferramenta: {tool}")
        self.console.write(f"ação: {action}")
        self.console.write(f"caminho: {path}")
        self.console.write(f"impacto: {impact}")
        self.console.write(f"reversão: {reversible}")
        value = self.console.read(
            "Digite CONFIRMAR para executar; qualquer outro texto cancela: "
        )
        return value.strip() == "CONFIRMAR"


def prepare_spoken_text(
    text: str,
    *,
    max_characters: int,
    read_code: bool,
    read_urls: bool,
    summarize_long: bool,
) -> str:
    value = text.strip()
    value = unwrap_markdown_code(value)
    url_found = bool(_URL_RE.search(value))
    if not read_urls:
        value = _URL_RE.sub(" fonte disponível na tela ", value)
        value = re.sub(
            r"(?im)^\s*(fontes?|urls?)\s+(consultadas?|citadas?)?\s*:?\s*$",
            "Fontes disponíveis na tela.",
            value,
        )
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[*#>|_]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    suffixes = []
    if url_found and not read_urls and "fonte disponível" not in value.casefold():
        suffixes.append("Fontes disponíveis na tela.")
    if len(value) <= max_characters or not summarize_long:
        result = value[:max_characters] if len(value) > max_characters else value
    else:
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+", value)
            if item.strip()
        ]
        if not sentences:
            result = value[:max_characters]
        else:
            intro_limit = max_characters * 2 // 3
            intro = []
            length = 0
            for sentence in sentences:
                if intro and length + len(sentence) + 1 > intro_limit:
                    break
                intro.append(sentence)
                length += len(sentence) + 1
            conclusion = sentences[-1]
            if conclusion in intro:
                conclusion = ""
            result = " ".join(intro)
            if conclusion:
                result += " Em resumo: " + conclusion
            result += " Detalhes completos permanecem na tela."
            result = result[:max_characters].rstrip()
    if suffixes:
        suffix = " ".join(suffixes)
        available = max(0, max_characters - len(suffix) - 1)
        result = (result[:available].rstrip() + " " + suffix).strip()
    return result[:max_characters].rstrip()


def prepare_research_spoken_text(
    text: str,
    *,
    source_count: int,
    max_characters: int,
    read_code: bool,
    read_urls: bool,
    summarize_long: bool,
) -> str:
    answer = re.split(
        r"(?im)^\s*fontes?\s+consultadas?\s*:",
        text,
        maxsplit=1,
    )[0]
    spoken = prepare_spoken_text(
        answer,
        max_characters=max_characters,
        read_code=read_code,
        read_urls=read_urls,
        summarize_long=summarize_long,
    )
    if source_count:
        label = "fonte relevante" if source_count == 1 else "fontes relevantes"
        suffix = (
            f" Encontrei {source_count} {label}; "
            "os links estão na tela."
        )
        available = max(0, max_characters - len(suffix))
        spoken = spoken[:available].rstrip() + suffix
    return spoken[:max_characters].strip()
