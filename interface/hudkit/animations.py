"""Componentes animados em Canvas para o HUD Kit."""

from __future__ import annotations

import math
import random
import tkinter as tk

from hudkit.theme import HUDTheme, DEFAULT_THEME


class WaveformDisplay(tk.Canvas):
    """Osciloscópio animado de telemetria neural."""

    def __init__(self, parent, theme: HUDTheme | None = None, width: int = 220, height: int = 56):
        self.theme = theme or DEFAULT_THEME
        super().__init__(
            parent,
            width=width,
            height=height,
            bg="#010a14",
            highlightbackground=self.theme.border,
            highlightthickness=1,
        )
        self._data = [random.random() * 0.4 for _ in range(80)]
        self.processing = False
        self._animate()

    def set_processing(self, active: bool) -> None:
        self.processing = active

    def _animate(self) -> None:
        self._data.pop(0)
        self._data.append(random.random() * (0.9 if self.processing else 0.4) + (0.1 if self.processing else 0))

        self.delete("all")
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        color = self.theme.gold if self.processing else self.theme.cyan

        points = []
        for i, v in enumerate(self._data):
            x = i / len(self._data) * w
            y = h / 2 + (v - 0.25) * h * 0.75
            points.extend([x, y])

        if len(points) >= 4:
            self.create_line(points, fill=color, width=1.5, smooth=True)

        self.after(50, self._animate)


class CoreReactor(tk.Canvas):
    """Núcleo central com anéis rotativos e arcos ternários."""

    COLORS = {"neg": "#ff3b5c", "neu": "#8899aa", "pos": "#00ff9d"}

    def __init__(self, parent, theme: HUDTheme | None = None, size: int = 300):
        self.theme = theme or DEFAULT_THEME
        self.size = size
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=self.theme.bg_deep,
            highlightthickness=0,
        )
        self.angle_outer = 0.0
        self.angle_mid = 0.0
        self.angle_inner = 0.0
        self.processing = False
        self.distribution = {"neg": 33, "neu": 34, "pos": 33}
        self.state_text = "STANDBY"
        self.confidence_text = "—"
        self._animate()

    def set_processing(self, active: bool) -> None:
        self.processing = active
        if active:
            self.state_text = "PROCESSANDO"
        elif self.state_text == "PROCESSANDO":
            self.state_text = "STANDBY"

    def set_state(self, state: str, confidence: str) -> None:
        self.state_text = state
        self.confidence_text = confidence

    def set_distribution(self, neg: int, neu: int, pos: int) -> None:
        self.distribution = {"neg": neg, "neu": neu, "pos": pos}

    def _animate(self) -> None:
        speed = 0.05 if self.processing else 0.012
        self.angle_outer += speed * 0.4
        self.angle_mid -= speed * 0.6
        self.angle_inner += speed

        self.delete("all")
        cx = cy = self.size / 2
        glow = self.theme.gold if self.processing else self.theme.cyan

        # Glow
        for r, alpha in [(90, 0.08), (70, 0.12), (50, 0.18)]:
            self.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="", fill=glow, stipple="gray50" if alpha < 0.15 else "gray25",
            )

        # Rings
        self._draw_ring(cx, cy, 130, self.angle_outer, self.theme.cyan_dim, dash=(6, 8))
        self._draw_ring(cx, cy, 110, self.angle_mid, self.theme.cyan, width=2)
        self._draw_ring(cx, cy, 92, self.angle_inner, self.theme.cyan, width=1)

        # Ternary arcs
        weights = [
            self.distribution["neg"] / 100,
            self.distribution["neu"] / 100,
            self.distribution["pos"] / 100,
        ]
        keys = ["neg", "neu", "pos"]
        for i, key in enumerate(keys):
            start = self.angle_inner + i * (2 * math.pi / 3)
            end = start + (2 * math.pi / 3) * 0.92
            self.create_arc(
                cx - 78, cy - 78, cx + 78, cy + 78,
                start=math.degrees(start),
                extent=math.degrees(end - start),
                style="arc",
                outline=self.COLORS[key],
                width=int(4 + weights[i] * 10),
            )

        # Center
        self.create_oval(cx - 28, cy - 28, cx + 28, cy + 28, fill="#001830", outline=glow, width=2)
        self.create_text(cx, cy - 4, text="◈", fill=glow, font=("Consolas", 22, "bold"))
        self.create_text(cx, cy + 52, text=self.state_text, fill=glow, font=("Consolas", 9, "bold"))
        self.create_text(cx, cy + 68, text=self.confidence_text, fill=self.theme.text_dim, font=("Segoe UI", 8))

        self.after(33, self._animate)

    def _draw_ring(self, cx, cy, r, angle, color, width=1, dash=None) -> None:
        steps = 48
        points = []
        for i in range(steps + 1):
            a = angle + (i / steps) * 2 * math.pi
            points.extend([cx + math.cos(a) * r, cy + math.sin(a) * r])
        if len(points) >= 4:
            self.create_line(*points, fill=color, width=width, dash=dash)


class TernaryGrid(tk.Frame):
    """Grade 8×8 de células ternárias coloridas."""

    COLORS = {"neg": "#ff3b5c", "neu": "#445566", "pos": "#00ff9d"}

    def __init__(self, parent, theme: HUDTheme | None = None, cols: int = 8, rows: int = 8):
        self.theme = theme or DEFAULT_THEME
        self.cols = cols
        self.rows = rows
        super().__init__(parent, bg=self.theme.bg_panel)
        self._cells: list[tk.Label] = []
        for r in range(rows):
            for c in range(cols):
                cell = tk.Label(self, bg=self.COLORS["neu"], width=2, height=1)
                cell.grid(row=r, column=c, padx=1, pady=1, sticky="nsew")
                self._cells.append(cell)
        for c in range(cols):
            self.columnconfigure(c, weight=1)

    def randomize(self) -> None:
        for cell in self._cells:
            key = random.choice(["neg", "neu", "pos"])
            cell.configure(bg=self.COLORS[key])

    def flash(self, result: int) -> None:
        key = { -1: "neg", 0: "neu", 1: "pos" }.get(result, "neu")
        color = self.COLORS[key]
        for i, cell in enumerate(self._cells):
            self.after(i * 12, lambda c=cell, col=color: c.configure(bg=col))
