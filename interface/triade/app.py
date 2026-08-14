"""Interface J.A.R.V.I.S. construída sobre o HUD Kit."""

from __future__ import annotations

import random
from datetime import datetime

import tkinter as tk

from hudkit.base import HUDApplication
from hudkit.theme import HUDTheme
from hudkit.widgets import (
    HUDPanel,
    HUDHeader,
    HUDFooter,
    HUDMetric,
    HUDLogTerminal,
    HUDCommandBar,
    DecisionBars,
    InferenceHistory,
)
from hudkit.animations import CoreReactor, WaveformDisplay, TernaryGrid
from triade.engine import TernaryEngine, STATE_LABELS


class JARVISApp(HUDApplication):
    title = "J.A.R.V.I.S. — Interface Neural"
    geometry = "1280x740"

    QUICK_ACTIONS = [
        ("Status", "status"),
        ("Diagnóstico", "diagnóstico"),
        ("Análise", "analise ternária"),
        ("Reiniciar", "reiniciar núcleo"),
    ]

    def __init__(self) -> None:
        self.engine = TernaryEngine()
        self.processing = False
        self.metrics: dict[str, HUDMetric] = {}
        super().__init__(theme=HUDTheme())

    def build_left_panel(self, parent: tk.Frame) -> None:
        t = self.theme

        tele = HUDPanel(parent, "Telemetria Neural", theme=t)
        tele.pack(fill="x", pady=(0, 8))
        self.metrics["layers"] = HUDMetric(tele.body, "Camadas ativas", "12 / 12", theme=t)
        self.metrics["layers"].pack(fill="x", pady=2)
        self.metrics["neurons"] = HUDMetric(tele.body, "Neurônios ternários", "4.096", theme=t)
        self.metrics["neurons"].pack(fill="x", pady=2)
        self.metrics["convergence"] = HUDMetric(tele.body, "Taxa de convergência", "98.7%", theme=t)
        self.metrics["convergence"].pack(fill="x", pady=2)
        self.waveform = WaveformDisplay(tele.body, theme=t)
        self.waveform.pack(fill="x", pady=(8, 0))

        states = HUDPanel(parent, "Estados Ternários", theme=t)
        states.pack(fill="both", expand=True)
        legend = tk.Frame(states.body, bg=t.bg_panel)
        legend.pack(fill="x", pady=(0, 8))
        for sym, label, color in [("−1", "Inibição", t.red), ("0", "Neutro", t.neutral), ("+1", "Ativação", t.green)]:
            box = tk.Frame(legend, bg=t.bg_panel, highlightbackground=color, highlightthickness=1)
            box.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(box, text=sym, bg=t.bg_panel, fg=color, font=("Consolas", 11, "bold")).pack()
            tk.Label(box, text=label, bg=t.bg_panel, fg=t.text_dim, font=t.font_small).pack()
        self.ternary_grid = TernaryGrid(states.body, theme=t)
        self.ternary_grid.pack(pady=4)

    def build_center_panel(self, parent: tk.Frame) -> None:
        t = self.theme
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)
        parent.columnconfigure(0, weight=1)

        core_wrap = tk.Frame(parent, bg=t.bg_deep)
        core_wrap.grid(row=0, column=0, sticky="nsew")
        self.core = CoreReactor(core_wrap, theme=t, size=320)
        self.core.pack(expand=True)

        self.command_bar = HUDCommandBar(
            parent,
            on_submit=self.handle_command,
            quick_actions=self.QUICK_ACTIONS,
            theme=t,
            placeholder="Comando ou consulta ao modelo ternário...",
        )
        self.command_bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def build_right_panel(self, parent: tk.Frame) -> None:
        t = self.theme

        log_panel = HUDPanel(parent, "Terminal de Resposta", theme=t)
        log_panel.pack(fill="both", expand=True, pady=(0, 8))
        self.log = HUDLogTerminal(log_panel.body, theme=t, height=12)
        self.log.pack(fill="both", expand=True)
        self._log("system", "INIT", "Interface J.A.R.V.I.S. inicializada. Matriz neural carregada.")
        self._log("system", "READY", "Aguardando entrada. Estados: −1 | 0 | +1")

        dist_panel = HUDPanel(parent, "Distribuição de Decisão", theme=t)
        dist_panel.pack(fill="x", pady=(0, 8))
        self.decision_bars = DecisionBars(dist_panel.body, theme=t)
        self.decision_bars.pack(fill="x")

        hist_panel = HUDPanel(parent, "Histórico de Inferência", theme=t)
        hist_panel.pack(fill="x")
        self.history = InferenceHistory(hist_panel.body, theme=t)
        self.history.pack(fill="x")

    def on_ready(self) -> None:
        t = self.theme

        header = HUDHeader(
            self.header,
            "J.A.R.V.I.S.",
            "Just A Rather Very Intelligent System",
            theme=t,
        )
        header.pack(fill="both", expand=True)
        header.add_status("system", "SISTEMA", "ONLINE", t.green)
        header.add_status("core", "NÚCLEO", "37.2°C")
        header.add_status("clock", "HORA", "--:--:--")
        self.header_widget = header

        self.footer_widget = HUDFooter(self.footer, theme=t)
        self.footer_widget.pack(fill="x")
        self._update_footer()

        self.schedule(1000, self._tick_clock)
        self.schedule(2000, self._tick_telemetry)
        self.schedule(1500, self._tick_grid)

    def _log(self, tag: str, ts: str, msg: str) -> None:
        self.log.append(tag, ts, msg)

    def _tick_clock(self) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.header_widget.set_status("clock", now)
        self.schedule(1000, self._tick_clock)

    def _tick_telemetry(self) -> None:
        temp = f"{36.5 + random.random() * 2:.1f}°C"
        conv = f"{95 + random.random() * 4.9:.1f}%"
        self.header_widget.set_status("core", temp)
        self.metrics["convergence"].set(conv)
        self._latency = f"{random.randint(8, 22)}ms"
        self._update_footer()
        self.schedule(2000, self._tick_telemetry)

    def _tick_grid(self) -> None:
        if not self.processing:
            self.ternary_grid.randomize()
        self.schedule(1500, self._tick_grid)

    def _update_footer(self) -> None:
        latency = getattr(self, "_latency", "12ms")
        self.footer_widget.set_text([
            "v1.0.0-JARVIS",
            f"Latência: {latency}",
            "Modo: AUTÔNOMO",
            "J.A.R.V.I.S. HUD",
        ])

    def handle_command(self, text: str) -> None:
        if self.processing:
            return
        self.processing = True
        self.command_bar.set_enabled(False)
        self.core.set_processing(True)
        self.waveform.set_processing(True)

        ts = datetime.now().strftime("%H:%M:%S")
        self._log("user", ts, text)
        self._log("processing", "PROCESSANDO", "Executando inferência ternária...")

        delay = random.randint(700, 1400)
        self.schedule(delay, lambda: self._finish_command(text))

    def _finish_command(self, text: str) -> None:
        inference = self.engine.infer(text)
        telemetry = {
            "convergence": self.metrics["convergence"].value_label.cget("text"),
            "temp": self.header_widget.get_status("core"),
        }
        response = self.engine.respond(text, inference, telemetry)

        self.log.remove_processing()
        ts = datetime.now().strftime("%H:%M:%S")
        self._log("response", ts, response)

        d = inference.distribution
        self.decision_bars.set_distribution(d["neg"], d["neu"], d["pos"])
        self.core.set_distribution(d["neg"], d["neu"], d["pos"])
        self.core.set_state(
            STATE_LABELS[inference.result],
            f"Confiança: {inference.confidence}%",
        )
        self.core.set_processing(False)
        self.history.add(text, inference.result)
        self.ternary_grid.flash(inference.result)

        self.processing = False
        self.command_bar.set_enabled(True)
        self.waveform.set_processing(False)
