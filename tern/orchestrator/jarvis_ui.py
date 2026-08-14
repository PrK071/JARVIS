from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import tkinter as tk
from tkinter import messagebox

from hudkit.animations import CoreReactor, TernaryGrid, WaveformDisplay
from hudkit.base import HUDApplication
from hudkit.theme import HUDTheme
from hudkit.widgets import (
    DecisionBars,
    HUDCommandBar,
    HUDFooter,
    HUDHeader,
    HUDLogTerminal,
    HUDMetric,
    HUDPanel,
    InferenceHistory,
)

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


def clean_chat_text(text: str) -> str:
    """Converte Markdown e metadados internos em texto limpo para o chat."""
    clean_lines: list[str] = []
    hidden_fields = re.compile(
        r"^\s*[-*•]?\s*(?:\*\*)?(?:id da tarefa|thread disponível|thread id|task id)(?:\*\*)?\s*:",
        re.IGNORECASE,
    )
    for raw_line in str(text).replace("\r\n", "\n").split("\n"):
        if hidden_fields.match(raw_line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", raw_line)
        line = re.sub(r"^\s*[*-]\s+", "• ", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        clean_lines.append(line.rstrip())

    while clean_lines and not clean_lines[-1]:
        clean_lines.pop()
    return "\n".join(clean_lines).strip()


class JarvisUI(HUDApplication):
    """JARVIS funcional com HUD baseado no Interface-JARVIS de mucamuca."""

    title = "J.A.R.V.I.S. · Assistente local"
    geometry = "1280x740"
    min_size = (1024, 600)
    QUICK_ACTIONS = [
        ("Status", "status"),
        ("Diagnóstico", "diagnóstico"),
        ("Análise", "analise o projeto atual"),
        ("Codex", "status do codex"),
        ("Hardware", "qual a temperatura da CPU e quantos dispositivos USB estão conectados?"),
    ]

    def __init__(
        self,
        *,
        settings: Any,
        runtime: Any,
        registry: Any,
        supervisor_factory: Callable[[Any, Any, Any], Supervisor] = Supervisor,
    ):
        self.settings, self.runtime, self.registry = settings, runtime, registry
        self.supervisor_factory = supervisor_factory
        self.supervisor: Supervisor | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.busy = False
        self.last_prompt = ""
        self.project_name = "—"
        self.telemetry_loading = False
        self.metrics: dict[str, HUDMetric] = {}
        super().__init__(theme=HUDTheme())
        self.registry.approval = self._approve_action
        self.status_var = tk.StringVar(master=self.root, value="Verificando Qwen…")
        self.project_var = tk.StringVar(master=self.root, value="Projeto: —")
        self._append(ChatTurn("SISTEMA", "Interface J.A.R.V.I.S. inicializada. Aguardando comando."))
        self.refresh_status()
        self.root.after(100, self._consume_events)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def build_left_panel(self, parent: tk.Frame) -> None:
        t = self.theme
        telemetry = HUDPanel(parent, "Telemetria Neural", theme=t)
        telemetry.pack(fill="x", pady=(0, 8))
        self.metrics["layers"] = HUDMetric(telemetry.body, "Camadas ativas", "12 / 12", theme=t)
        self.metrics["layers"].pack(fill="x", pady=2)
        self.metrics["neurons"] = HUDMetric(telemetry.body, "Neurônios ternários", "4.096", theme=t)
        self.metrics["neurons"].pack(fill="x", pady=2)
        self.metrics["convergence"] = HUDMetric(telemetry.body, "Taxa de convergência", "98.7%", theme=t)
        self.metrics["convergence"].pack(fill="x", pady=2)
        self.metrics["usb"] = HUDMetric(telemetry.body, "Dispositivos USB", "LENDO…", theme=t)
        self.metrics["usb"].pack(fill="x", pady=2)
        self.waveform = WaveformDisplay(telemetry.body, theme=t)
        self.waveform.pack(fill="x", pady=(8, 0))

        states = HUDPanel(parent, "Estados Ternários", theme=t)
        states.pack(fill="both", expand=True)
        legend = tk.Frame(states.body, bg=t.bg_panel)
        legend.pack(fill="x", pady=(0, 8))
        for symbol, label, color in (
            ("−1", "Inibição", t.red),
            ("0", "Neutro", t.neutral),
            ("+1", "Ativação", t.green),
        ):
            box = tk.Frame(legend, bg=t.bg_panel, highlightbackground=color, highlightthickness=1)
            box.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(box, text=symbol, bg=t.bg_panel, fg=color, font=("Consolas", 11, "bold")).pack()
            tk.Label(box, text=label, bg=t.bg_panel, fg=t.text_dim, font=t.font_small).pack()
        self.ternary_grid = TernaryGrid(states.body, theme=t)
        self.ternary_grid.pack(pady=4)
        self.ternary_grid.randomize()

    def build_center_panel(self, parent: tk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)
        parent.columnconfigure(0, weight=1)

        core_wrap = tk.Frame(parent, bg=self.theme.bg_deep)
        core_wrap.grid(row=0, column=0, sticky="nsew")
        self.core = CoreReactor(core_wrap, theme=self.theme, size=320)
        self.core.pack(expand=True)

        self.command_bar = HUDCommandBar(
            parent,
            on_submit=self.submit,
            quick_actions=self.QUICK_ACTIONS,
            theme=self.theme,
            placeholder="Comando ou consulta para J.A.R.V.I.S.…",
        )
        self.command_bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.input = self.command_bar.entry

    def build_right_panel(self, parent: tk.Frame) -> None:
        terminal = HUDPanel(parent, "Terminal de Resposta", theme=self.theme)
        terminal.pack(fill="both", expand=True, pady=(0, 8))
        self.log = HUDLogTerminal(terminal.body, theme=self.theme, height=12)
        self.log.pack(fill="both", expand=True)
        self.transcript = self.log.text
        self.transcript.tag_configure("speaker_user", foreground=self.theme.gold, font=("Consolas", 9, "bold"), spacing1=10)
        self.transcript.tag_configure("speaker_jarvis", foreground=self.theme.green, font=("Consolas", 9, "bold"), spacing1=10)
        self.transcript.tag_configure("speaker_system", foreground=self.theme.cyan, font=("Consolas", 9, "bold"), spacing1=10)
        self.transcript.tag_configure("chat_body", foreground=self.theme.text, spacing3=8)

        decision = HUDPanel(parent, "Distribuição de Decisão", theme=self.theme)
        decision.pack(fill="x", pady=(0, 8))
        self.decision_bars = DecisionBars(decision.body, theme=self.theme)
        self.decision_bars.pack(fill="x")

        history = HUDPanel(parent, "Histórico de Inferência", theme=self.theme)
        history.pack(fill="x")
        self.history = InferenceHistory(history.body, theme=self.theme)
        self.history.pack(fill="x")

    def on_ready(self) -> None:
        self.header_widget = HUDHeader(
            self.header,
            "J.A.R.V.I.S.",
            "Just A Rather Very Intelligent System · Qwen · Codex",
            theme=self.theme,
        )
        self.header_widget.pack(fill="both", expand=True)
        self.header_widget.add_status("system", "SISTEMA", "INICIANDO", self.theme.gold)
        self.header_widget.add_status("deepseek", "DEEPSEEK", "VERIFICANDO", self.theme.gold)
        self.header_widget.add_status("core", "CPU", "LENDO…")
        self.header_widget.add_status("clock", "HORA", "--:--:--")

        self.footer_widget = HUDFooter(self.footer, theme=self.theme)
        self.footer_widget.pack(fill="x")
        self._update_footer()
        self.root.after(1000, self._tick_clock)
        self.root.after(200, self._tick_telemetry)
        self.root.after(1500, self._tick_grid)

    def _tick_clock(self) -> None:
        self.header_widget.set_status("clock", datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _tick_telemetry(self) -> None:
        if not self.telemetry_loading:
            self.telemetry_loading = True
            threading.Thread(target=self._load_hardware, daemon=True).start()
        self.root.after(5000, self._tick_telemetry)

    def _load_hardware(self) -> None:
        self.events.put(("hardware", self.registry.hardware.read()))

    def _approve_action(self, risk: str, arguments: dict[str, Any]) -> bool:
        completed = threading.Event()
        result: dict[str, bool] = {"approved": False}
        self.events.put(("approval", (risk, arguments, completed, result)))
        completed.wait(timeout=self.settings.action_confirmation_timeout_seconds)
        return result["approved"]

    def _tick_grid(self) -> None:
        if not self.busy:
            self.ternary_grid.randomize()
        self.root.after(1500, self._tick_grid)

    def _update_footer(self) -> None:
        self.footer_widget.set_text([
            "v0.1.0-TERN",
            f"Latência: {getattr(self, '_latency', '12ms')}",
            "Modo: AUTÔNOMO",
            f"Projeto: {self.project_name}",
        ])

    def _append(self, turn: ChatTurn) -> None:
        speaker_tag = {
            "VOCÊ": "speaker_user",
            "JARVIS": "speaker_jarvis",
            "SISTEMA": "speaker_system",
        }.get(turn.speaker, "speaker_system")
        message = clean_chat_text(turn.text)
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{turn.speaker}\n", speaker_tag)
        self.transcript.insert("end", f"{message}\n\n", "chat_body")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _show_processing(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", "JARVIS\nProcessando…\n\n", "processing")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _remove_processing(self) -> None:
        self.transcript.configure(state="normal")
        ranges = self.transcript.tag_ranges("processing")
        for start, end in zip(reversed(ranges[::2]), reversed(ranges[1::2])):
            self.transcript.delete(start, end)
        self.transcript.configure(state="disabled")

    def clear(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self._append(ChatTurn("SISTEMA", "Terminal limpo. Contexto do assistente permanece ativo."))

    def _send_event(self, event: Any) -> str | None:
        if event.state & 0x0001:
            return None
        self.send()
        return "break"

    def send(self) -> None:
        prompt = self.input.get().strip()
        placeholder = getattr(self.command_bar, "_placeholder", "")
        if not prompt or prompt == placeholder:
            return
        self.input.delete(0, "end")
        self.submit(prompt)

    def submit(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt or self.busy:
            return
        self.last_prompt = prompt
        self._append(ChatTurn("VOCÊ", prompt))
        self.busy = True
        self.command_bar.set_enabled(False)
        self.core.set_processing(True)
        self.waveform.set_processing(True)
        self._show_processing()
        self.events.put(("status", "JARVIS analisando…"))
        threading.Thread(target=self._run_prompt, args=(prompt,), daemon=True).start()

    def _run_prompt(self, prompt: str) -> None:
        try:
            started = self.runtime.ensure_llama_server(240)
            if not started.get("healthy"):
                self.events.put(("result", {"ok": False, "error": "qwen_indisponivel"}))
                return
            if self.supervisor is None:
                self.supervisor = self.supervisor_factory(
                    self.settings,
                    LlamaClient(self.settings.base_url, self.settings.timeout),
                    self.registry,
                )
            self.events.put(("result", self.supervisor.run(prompt)))
        except Exception as exc:
            self.events.put(("result", {"ok": False, "error": "ui_request_failed", "message": str(exc)}))

    def refresh_status(self) -> None:
        threading.Thread(target=self._load_status, daemon=True).start()

    def _load_status(self) -> None:
        try:
            runtime = self.runtime.status()
            project = dict(self.registry.projects.context().get("active_project") or {})
            root = project.get("root")
            if root:
                project["codex_thread_id"] = self.registry.codex.sessions.project_binding(root)
            deepseek = self.registry.deepseek.status(project_path=root) if self.registry.deepseek else {}
            self.events.put(("runtime", (runtime, project, deepseek)))
        except Exception as exc:
            self.events.put(("status", f"Status indisponível: {exc}"))

    @staticmethod
    def _distribution_for(result: dict[str, Any]) -> tuple[int, int, int, int, float]:
        if not result.get("ok"):
            return 72, 18, 10, -1, 72.0
        decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
        reason = f"{decision.get('intent', '')} {decision.get('reason_code', '')}".lower()
        if any(word in reason for word in ("fail", "error", "deny", "block")):
            return 64, 23, 13, -1, 64.0
        if not decision:
            return 18, 64, 18, 0, 64.0
        return 10, 18, 72, 1, 72.0

    def _finish_result(self, result: dict[str, Any]) -> None:
        self._remove_processing()
        self._append(result_turn(result))
        neg, neu, pos, state, confidence = self._distribution_for(result)
        self.decision_bars.set_distribution(neg, neu, pos)
        self.core.set_distribution(neg, neu, pos)
        labels = {-1: "INIBIÇÃO", 0: "NEUTRO", 1: "ATIVAÇÃO"}
        self.core.set_state(labels[state], f"Confiança: {confidence:.1f}%")
        self.core.set_processing(False)
        self.waveform.set_processing(False)
        self.ternary_grid.flash(state)
        if self.last_prompt:
            self.history.add(self.last_prompt, state)
        self.busy = False
        self.command_bar.set_enabled(True)
        self.input.focus_set()

    def _consume_events(self) -> None:
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(str(value))
                self.header_widget.set_status("system", "ANALISANDO" if self.busy else "ATENÇÃO", self.theme.gold)
            elif kind == "runtime":
                runtime, project, deepseek = value
                healthy = bool(runtime.get("healthy"))
                state = "ONLINE" if healthy else "OFFLINE"
                self.status_var.set(f"Qwen {'pronto' if healthy else 'parado'} · {runtime.get('endpoint') or self.settings.base_url}")
                self.header_widget.set_status("system", state, self.theme.green if healthy else self.theme.red)
                deepseek_ready = bool(deepseek.get("enabled") and deepseek.get("configured"))
                deepseek_state = "PRONTO" if deepseek_ready else "SEM CHAVE" if deepseek.get("enabled") else "DESATIVADO"
                self.header_widget.set_status(
                    "deepseek",
                    deepseek_state,
                    self.theme.green if deepseek_ready else self.theme.red,
                )
                project_text = project.get("name") or project.get("id") or "—"
                self.project_name = str(project_text)
                self.project_var.set(f"Projeto: {self.project_name}")
                self._update_footer()
            elif kind == "hardware":
                self.telemetry_loading = False
                if value.get("cpu_temperature_available"):
                    temperature = f"{float(value['cpu_temperature_c']):.1f}°C"
                    self.header_widget.set_status("core", temperature, self.theme.cyan)
                else:
                    self.header_widget.set_status("core", "INDISP.", self.theme.neutral)
                usb = str(value["usb_devices"]) if value.get("usb_available") else "INDISP."
                self.metrics["usb"].set(usb)
            elif kind == "approval":
                risk, arguments, completed, result = value
                if risk == "schedule_task":
                    detail = (
                        f"Aplicativo: {arguments.get('application_name')}\n"
                        f"Horário: {arguments.get('start_at')}\n"
                        f"Recorrência: {arguments.get('recurrence')}"
                    )
                    prompt = "Criar esta tarefa no Agendador do Windows?\n\n" + detail
                else:
                    prompt = f"Autorizar ação {risk}?"
                result["approved"] = messagebox.askyesno(
                    "Confirmação do JARVIS",
                    prompt,
                    parent=self.root,
                )
                completed.set()
            elif kind == "result":
                self._finish_result(value)
                self.refresh_status()
        self.root.after(100, self._consume_events)


def run_jarvis_ui(*, settings: Any, runtime: Any, registry: Any) -> int:
    JarvisUI(settings=settings, runtime=runtime, registry=registry).run()
    return 0
