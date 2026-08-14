from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .agent import Supervisor
from .client import LlamaClient


@dataclass(frozen=True)
class ChatTurn:
    speaker: str
    text: str
    detail: str = ""


def result_turn(result: dict[str, Any]) -> ChatTurn:
    if result.get("ok"):
        decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
        detail = " · ".join(
            value
            for value in (str(decision.get("intent") or ""), str(decision.get("reason_code") or ""))
            if value
        )
        return ChatTurn("JARVIS", str(result.get("answer") or "Sem resposta."), detail)
    return ChatTurn("SISTEMA", str(result.get("message") or result.get("error") or "Falha."))


class JarvisUI:
    def __init__(self, *, settings: Any, runtime: Any, registry: Any, supervisor_factory: Callable[[Any, Any, Any], Supervisor] = Supervisor):
        try:
            import tkinter as tk
            from tkinter import scrolledtext, ttk
        except ImportError as exc:
            raise RuntimeError("A interface Jarvis requer tkinter.") from exc
        self.tk, self.scrolledtext, self.ttk = tk, scrolledtext, ttk
        self.settings, self.runtime, self.registry = settings, runtime, registry
        self.supervisor_factory = supervisor_factory
        self.supervisor: Supervisor | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.busy = False
        self.root = tk.Tk()
        self.root.title("JARVIS · Assistente local")
        self.root.geometry("1040x760")
        self.root.minsize(760, 560)
        self._configure_style()
        self._build()
        self._append(ChatTurn("SISTEMA", "Pronto. Escreva uma mensagem para iniciar."))
        self.refresh_status()
        self.root.after(100, self._consume_events)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _configure_style(self) -> None:
        self.root.configure(bg="#10161f")
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Header.TFrame", background="#10161f")
        style.configure("Panel.TFrame", background="#18212d")
        style.configure("Title.TLabel", background="#10161f", foreground="#f2f7ff", font=("Segoe UI", 20, "bold"))
        style.configure("Subtle.TLabel", background="#10161f", foreground="#9fb0c6", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#18212d", foreground="#b8c8dd", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 9))

    def _build(self) -> None:
        tk = self.tk
        header = self.ttk.Frame(self.root, style="Header.TFrame", padding=(24, 20, 24, 12))
        header.pack(fill="x")
        self.ttk.Label(header, text="JARVIS", style="Title.TLabel").pack(anchor="w")
        self.ttk.Label(header, text="Assistente local · Qwen · Codex · projetos", style="Subtle.TLabel").pack(anchor="w", pady=(2, 0))
        status = self.ttk.Frame(self.root, style="Panel.TFrame", padding=(16, 10))
        status.pack(fill="x", padx=24, pady=(0, 12))
        self.status_var, self.project_var = tk.StringVar(value="Verificando Qwen…"), tk.StringVar(value="Projeto: —")
        self.ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.ttk.Label(status, textvariable=self.project_var, style="Status.TLabel").pack(side="right")
        body = self.ttk.Frame(self.root, style="Panel.TFrame", padding=2)
        body.pack(fill="both", expand=True, padx=24)
        self.transcript = self.scrolledtext.ScrolledText(body, wrap="word", bg="#18212d", fg="#e7eef8", insertbackground="#ffffff", relief="flat", borderwidth=0, padx=18, pady=16, font=("Segoe UI", 11), state="disabled")
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_configure("user", foreground="#77d8ff", font=("Segoe UI", 11, "bold"), spacing1=12)
        self.transcript.tag_configure("jarvis", foreground="#d8f5bf", font=("Segoe UI", 11, "bold"), spacing1=12)
        self.transcript.tag_configure("system", foreground="#ffc98b", font=("Segoe UI", 10, "bold"), spacing1=12)
        self.transcript.tag_configure("detail", foreground="#92a4ba", font=("Segoe UI", 9))
        composer = self.ttk.Frame(self.root, style="Header.TFrame", padding=(24, 14, 24, 22))
        composer.pack(fill="x")
        self.input = tk.Text(composer, height=4, wrap="word", bg="#202c3b", fg="#f2f7ff", insertbackground="#ffffff", relief="flat", padx=12, pady=10, font=("Segoe UI", 11))
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Control-Return>", self._send_event)
        actions = self.ttk.Frame(composer, style="Header.TFrame")
        actions.pack(side="right", fill="y", padx=(12, 0))
        self.send_button = self.ttk.Button(actions, text="Enviar", style="Accent.TButton", command=self.send)
        self.send_button.pack(fill="x")
        self.ttk.Button(actions, text="Limpar", command=self.clear).pack(fill="x", pady=(8, 0))
        self.ttk.Button(actions, text="Atualizar", command=self.refresh_status).pack(fill="x", pady=(8, 0))
        self.ttk.Label(actions, text="Ctrl+Enter envia", style="Subtle.TLabel").pack(pady=(10, 0))
        self.input.focus_set()

    def _append(self, turn: ChatTurn) -> None:
        tag = {"VOCÊ": "user", "JARVIS": "jarvis", "SISTEMA": "system"}.get(turn.speaker, "system")
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{turn.speaker}\n", tag)
        self.transcript.insert("end", f"{turn.text.strip()}\n")
        if turn.detail:
            self.transcript.insert("end", f"{turn.detail}\n", "detail")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def clear(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self._append(ChatTurn("SISTEMA", "Janela limpa. Contexto do assistente permanece ativo."))

    def _send_event(self, _event: Any) -> str:
        self.send()
        return "break"

    def send(self) -> None:
        prompt = self.input.get("1.0", "end-1c").strip()
        if not prompt or self.busy:
            return
        self.input.delete("1.0", "end")
        self._append(ChatTurn("VOCÊ", prompt))
        self.busy = True
        self.send_button.configure(state="disabled", text="Pensando…")
        self.events.put(("status", "Iniciando Qwen…"))
        threading.Thread(target=self._run_prompt, args=(prompt,), daemon=True).start()

    def _run_prompt(self, prompt: str) -> None:
        try:
            started = self.runtime.ensure_llama_server(240)
            if not started.get("healthy"):
                self.events.put(("result", {"ok": False, "error": "qwen_indisponivel"}))
                return
            if self.supervisor is None:
                self.supervisor = self.supervisor_factory(self.settings, LlamaClient(self.settings.base_url, self.settings.timeout), self.registry)
            self.events.put(("status", "JARVIS analisando…"))
            self.events.put(("result", self.supervisor.run(prompt)))
        except Exception as exc:
            self.events.put(("result", {"ok": False, "error": "ui_request_failed", "message": str(exc)}))

    def refresh_status(self) -> None:
        threading.Thread(target=self._load_status, daemon=True).start()

    def _load_status(self) -> None:
        try:
            runtime = self.runtime.status()
            project = self.registry.projects.context().get("active_project") or {}
            self.events.put(("runtime", (runtime, project)))
        except Exception as exc:
            self.events.put(("status", f"Status indisponível: {exc}"))

    def _consume_events(self) -> None:
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(str(value))
            elif kind == "runtime":
                runtime, project = value
                state = "Qwen pronto" if runtime.get("healthy") else "Qwen parado"
                self.status_var.set(f"{state} · {runtime.get('endpoint') or self.settings.base_url}")
                self.project_var.set(f"Projeto: {project.get('name') or project.get('id') or '—'}")
            elif kind == "result":
                self._append(result_turn(value))
                self.busy = False
                self.send_button.configure(state="normal", text="Enviar")
                self.refresh_status()
        self.root.after(100, self._consume_events)

    def run(self) -> None:
        self.root.mainloop()


def run_jarvis_ui(*, settings: Any, runtime: Any, registry: Any) -> int:
    JarvisUI(settings=settings, runtime=runtime, registry=registry).run()
    return 0
