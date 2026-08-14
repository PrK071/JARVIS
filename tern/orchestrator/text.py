from __future__ import annotations

from typing import Any

from .voice.policy import ConsoleIO


class TextSession:
    def __init__(self, supervisor, *, console: ConsoleIO | None = None):
        self.supervisor = supervisor
        self.console = console or ConsoleIO()

    def run(self, *, once: bool = False) -> dict[str, Any]:
        interactions = 0
        last_result: dict[str, Any] = {"ok": True, "interactions": 0}
        self.console.write("[texto] pronto; /sair encerra")
        while True:
            try:
                prompt = self.console.read("> ")
            except (EOFError, KeyboardInterrupt):
                self.console.write("")
                return last_result
            text = prompt.strip()
            if text.casefold() in {"/sair", "/exit", "sair", "exit", "q"}:
                return last_result
            if not text:
                continue
            self.console.write("[assistente] pensando...")
            result = self.supervisor.run(text, event_callback=self._event)
            interactions += 1
            last_result = {**result, "interactions": interactions}
            if result.get("ok"):
                self.console.write("")
                self.console.write("[assistente]")
                self.console.write(str(result.get("answer") or ""))
            else:
                self.console.write(
                    f"[erro] {result.get('error', 'assistant_error')}: "
                    f"{result.get('message', '')}"
                )
            if once:
                return last_result

    def _event(self, event: str, values: dict[str, Any]) -> None:
        if event == "tool_start":
            name = str(values.get("name") or "")
            if name.startswith("web_"):
                self.console.write("[assistente] pesquisando...")
            elif name == "delegate_to_codex":
                self.console.write("[assistente] enviando tarefa ao Codex...")
            else:
                self.console.write(f"[assistente] executando ferramenta: {name}")
        elif event == "codex_working":
            self.console.write("[Codex] trabalhando...")
        elif event in {"codex_job_completed", "codex_result_received", "codex_completed"}:
            self.console.write("[Codex] tarefa concluida.")
        elif event in {"codex_job_failed", "codex_failed"}:
            self.console.write("[Codex] tarefa falhou.")
