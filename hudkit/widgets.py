"""Widgets reutilizáveis estilo HUD para tkinter."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from hudkit.theme import HUDTheme, DEFAULT_THEME


class HUDPanel(tk.Frame):
    """Painel com borda e título estilo módulo HUD."""

    def __init__(self, parent, title: str = "", theme: HUDTheme | None = None, **kwargs):
        self.theme = theme or DEFAULT_THEME
        t = self.theme
        super().__init__(
            parent,
            bg=t.bg_panel,
            highlightbackground=t.border,
            highlightthickness=1,
            **kwargs,
        )
        if title:
            tk.Label(
                self,
                text=f"◆ {title.upper()}",
                bg=t.bg_panel,
                fg=t.cyan,
                font=t.font_title,
                anchor="w",
            ).pack(fill="x", padx=t.pad_x, pady=(t.pad_y, 4))

        self.body = tk.Frame(self, bg=t.bg_panel)
        self.body.pack(fill="both", expand=True, padx=t.pad_x, pady=(0, t.pad_y))


class HUDMetric(tk.Frame):
    """Linha label + valor para telemetria."""

    def __init__(self, parent, label: str, value: str = "—", theme: HUDTheme | None = None):
        self.theme = theme or DEFAULT_THEME
        t = self.theme
        super().__init__(parent, bg=t.bg_panel)
        tk.Label(self, text=label, bg=t.bg_panel, fg=t.text, font=t.font_small).pack(side="left")
        self.value_label = tk.Label(
            self, text=value, bg=t.bg_panel, fg=t.cyan, font=t.font_mono
        )
        self.value_label.pack(side="right")

    def set(self, value: str) -> None:
        self.value_label.configure(text=value)


class HUDButton(tk.Button):
    """Botão com estilo HUD."""

    def __init__(self, parent, text: str, command: Callable | None = None, theme: HUDTheme | None = None, **kwargs):
        t = theme or DEFAULT_THEME
        super().__init__(
            parent,
            text=text.upper(),
            command=command,
            bg=t.bg_input,
            fg=t.text_dim,
            activebackground=t.cyan_dim,
            activeforeground=t.cyan,
            relief="flat",
            font=t.font_small,
            padx=10,
            pady=4,
            cursor="hand2",
            highlightbackground=t.border,
            highlightthickness=1,
            **kwargs,
        )
        self.bind("<Enter>", lambda _e: self.configure(fg=t.cyan, highlightbackground=t.cyan))
        self.bind("<Leave>", lambda _e: self.configure(fg=t.text_dim, highlightbackground=t.border))


class HUDHeader(tk.Frame):
    """Barra superior com marca e indicadores de status."""

    def __init__(self, parent, title: str, subtitle: str = "", theme: HUDTheme | None = None):
        self.theme = theme or DEFAULT_THEME
        t = self.theme
        super().__init__(parent, bg=t.bg_panel)

        brand = tk.Frame(self, bg=t.bg_panel)
        brand.pack(side="left", padx=t.pad_x, pady=t.pad_y)

        tk.Label(brand, text="◈", bg=t.bg_panel, fg=t.cyan, font=("Consolas", 22, "bold")).pack(side="left", padx=(0, 10))
        text_frame = tk.Frame(brand, bg=t.bg_panel)
        text_frame.pack(side="left")
        tk.Label(text_frame, text=title, bg=t.bg_panel, fg=t.cyan, font=t.font_display).pack(anchor="w")
        if subtitle:
            tk.Label(
                text_frame, text=subtitle, bg=t.bg_panel, fg=t.text_dim, font=("Segoe UI", 8)
            ).pack(anchor="w")

        self.status_frame = tk.Frame(self, bg=t.bg_panel)
        self.status_frame.pack(side="right", padx=t.pad_x, pady=t.pad_y)
        self._status_labels: dict[str, tk.Label] = {}

    def add_status(self, key: str, label: str, value: str, color: str | None = None) -> None:
        t = self.theme
        frame = tk.Frame(self.status_frame, bg=t.bg_panel)
        frame.pack(side="left", padx=16)
        tk.Label(frame, text=label, bg=t.bg_panel, fg=t.text_dim, font=("Segoe UI", 7)).pack(anchor="e")
        val = tk.Label(
            frame,
            text=value,
            bg=t.bg_panel,
            fg=color or t.cyan,
            font=t.font_mono,
        )
        val.pack(anchor="e")
        self._status_labels[key] = val

    def set_status(self, key: str, value: str, color: str | None = None) -> None:
        if key in self._status_labels:
            kw = {"text": value}
            if color:
                kw["fg"] = color
            self._status_labels[key].configure(**kw)

    def get_status(self, key: str) -> str:
        if key in self._status_labels:
            return self._status_labels[key].cget("text")
        return ""


class HUDFooter(tk.Frame):
    """Barra inferior com informações do sistema."""

    def __init__(self, parent, theme: HUDTheme | None = None):
        self.theme = theme or DEFAULT_THEME
        t = self.theme
        super().__init__(parent, bg=t.bg_panel)
        self.label = tk.Label(
            self, text="", bg=t.bg_panel, fg=t.text_dim, font=("Segoe UI", 8)
        )
        self.label.pack(pady=6)

    def set_text(self, parts: list[str]) -> None:
        self.label.configure(text="  |  ".join(parts))


class HUDLogTerminal(tk.Frame):
    """Terminal de log com tags coloridas."""

    TAGS = {
        "system": "#00d4ff",
        "user": "#ffd700",
        "response": "#c8e6f5",
        "processing": "#ffd700",
        "error": "#ff3b5c",
    }

    def __init__(self, parent, theme: HUDTheme | None = None, height: int = 14):
        self.theme = theme or DEFAULT_THEME
        t = self.theme
        super().__init__(parent, bg=t.bg_panel)

        self.text = tk.Text(
            self,
            height=height,
            bg="#010a14",
            fg=t.text,
            font=t.font_mono,
            relief="flat",
            wrap="word",
            state="disabled",
            highlightbackground=t.border,
            highlightthickness=1,
            padx=6,
            pady=6,
        )
        scroll = tk.Scrollbar(self, command=self.text.yview, bg=t.bg_panel)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for tag, color in self.TAGS.items():
            self.text.tag_configure(tag, foreground=color)

    def append(self, tag: str, timestamp: str, message: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", f"[{timestamp}] ", tag)
        self.text.insert("end", f"{message}\n", tag)
        self.text.configure(state="disabled")
        self.text.see("end")

    def remove_processing(self) -> None:
        self.text.configure(state="normal")
        content = self.text.get("1.0", "end")
        lines = content.split("\n")
        filtered = [ln for ln in lines if "[PROCESSANDO]" not in ln and ln.strip()]
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(filtered) + ("\n" if filtered else ""))
        self.text.configure(state="disabled")


class HUDCommandBar(tk.Frame):
    """Campo de comando com botão enviar e ações rápidas."""

    def __init__(
        self,
        parent,
        on_submit: Callable[[str], None],
        quick_actions: list[tuple[str, str]] | None = None,
        theme: HUDTheme | None = None,
        placeholder: str = "Comando...",
    ):
        self.theme = theme or DEFAULT_THEME
        t = self.theme
        self.on_submit = on_submit
        super().__init__(parent, bg=t.bg_deep)

        row = tk.Frame(self, bg=t.bg_input, highlightbackground=t.border, highlightthickness=1)
        row.pack(fill="x", pady=(0, 8))

        tk.Label(row, text=">", bg=t.bg_input, fg=t.cyan, font=("Consolas", 12, "bold")).pack(
            side="left", padx=(10, 4), pady=8
        )

        self.entry = tk.Entry(
            row,
            bg=t.bg_input,
            fg=t.text,
            insertbackground=t.cyan,
            relief="flat",
            font=t.font_body,
        )
        self.entry.pack(side="left", fill="x", expand=True, pady=8)
        self.entry.bind("<Return>", self._submit)
        self._placeholder = placeholder
        self.entry.insert(0, placeholder)
        self.entry.configure(fg=t.text_dim)
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<FocusOut>", self._restore_placeholder)

        send = HUDButton(row, "➤", command=self._submit, theme=t)
        send.pack(side="right", padx=6, pady=4)

        if quick_actions:
            actions = tk.Frame(self, bg=t.bg_deep)
            actions.pack(fill="x")
            for label, cmd in quick_actions:
                HUDButton(
                    actions,
                    label,
                    command=lambda c=cmd: self._run_quick(c),
                    theme=t,
                ).pack(side="left", padx=4)

    def _clear_placeholder(self, _event=None) -> None:
        if self.entry.get() == self._placeholder:
            self.entry.delete(0, "end")
            self.entry.configure(fg=self.theme.text)

    def _restore_placeholder(self, _event=None) -> None:
        if not self.entry.get().strip():
            self.entry.insert(0, self._placeholder)
            self.entry.configure(fg=self.theme.text_dim)

    def _submit(self, _event=None) -> None:
        text = self.entry.get().strip()
        if not text or text == self._placeholder:
            return
        self.entry.delete(0, "end")
        self.on_submit(text)

    def _run_quick(self, cmd: str) -> None:
        self.entry.delete(0, "end")
        self.entry.configure(fg=self.theme.text)
        self.on_submit(cmd)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)


class DecisionBars(tk.Frame):
    """Barras de distribuição ternária (−1, 0, +1)."""

    COLORS = {"neg": "#ff3b5c", "neu": "#8899aa", "pos": "#00ff9d"}
    LABELS = {"neg": "−1", "neu": "0", "pos": "+1"}

    def __init__(self, parent, theme: HUDTheme | None = None):
        self.theme = theme or DEFAULT_THEME
        t = self.theme
        super().__init__(parent, bg=t.bg_panel)
        self._bars: dict[str, tk.Canvas] = {}
        self._pcts: dict[str, tk.Label] = {}
        self._values = {"neg": 33, "neu": 34, "pos": 33}

        for key in ("neg", "neu", "pos"):
            row = tk.Frame(self, bg=t.bg_panel)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=self.LABELS[key], bg=t.bg_panel, fg=self.COLORS[key], font=t.font_mono, width=3).pack(
                side="left"
            )
            canvas = tk.Canvas(row, height=10, bg="#001020", highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True, padx=6)
            pct = tk.Label(row, text="33%", bg=t.bg_panel, fg=t.text_dim, font=t.font_mono, width=5)
            pct.pack(side="right")
            self._bars[key] = canvas
            self._pcts[key] = pct

        self.bind("<Configure>", lambda _e: self._draw_all())
        self.after_idle(self._draw_all)

    def set_distribution(self, neg: int, neu: int, pos: int) -> None:
        self._values = {"neg": neg, "neu": neu, "pos": pos}
        self._draw_all()

    def _draw_all(self) -> None:
        for key, canvas in self._bars.items():
            canvas.delete("all")
            w = canvas.winfo_width() or 120
            h = canvas.winfo_height() or 10
            pct = self._values[key]
            fill_w = max(2, int(w * pct / 100))
            canvas.create_rectangle(0, 0, w, h, fill="#001830", outline="")
            canvas.create_rectangle(0, 0, fill_w, h, fill=self.COLORS[key], outline="")
            self._pcts[key].configure(text=f"{pct}%")


class InferenceHistory(tk.Frame):
    """Lista compacta de inferências recentes."""

    COLORS = {-1: "#ff3b5c", 0: "#8899aa", 1: "#00ff9d"}
    SYMBOLS = {-1: "−1", 0: "0", 1: "+1"}

    def __init__(self, parent, theme: HUDTheme | None = None, max_items: int = 8):
        self.theme = theme or DEFAULT_THEME
        self.max_items = max_items
        super().__init__(parent, bg=self.theme.bg_panel)
        self._rows: list[tk.Frame] = []

    def add(self, query: str, result: int) -> None:
        t = self.theme
        row = tk.Frame(self, bg=t.bg_panel)
        row.pack(fill="x", pady=2)
        short = query[:22] + ("…" if len(query) > 22 else "")
        tk.Label(row, text=short, bg=t.bg_panel, fg=t.text_dim, font=t.font_small, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        tk.Label(
            row,
            text=self.SYMBOLS.get(result, "?"),
            bg=t.bg_panel,
            fg=self.COLORS.get(result, t.text),
            font=("Consolas", 10, "bold"),
        ).pack(side="right")
        self._rows.insert(0, row)
        while len(self._rows) > self.max_items:
            old = self._rows.pop()
            old.destroy()
