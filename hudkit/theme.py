"""Paleta, fontes e utilitários visuais do HUD Kit."""

from dataclasses import dataclass, field
import tkinter as tk
import tkinter.font as tkfont


@dataclass
class HUDTheme:
    """Tema visual configurável para interfaces HUD."""

    bg_deep: str = "#020810"
    bg_panel: str = "#041428"
    bg_input: str = "#001428"
    cyan: str = "#00d4ff"
    cyan_dim: str = "#006688"
    gold: str = "#ffd700"
    red: str = "#ff3b5c"
    green: str = "#00ff9d"
    neutral: str = "#8899aa"
    text: str = "#c8e6f5"
    text_dim: str = "#5a7a8f"
    border: str = "#004466"

    font_display: tuple = ("Consolas", 14, "bold")
    font_title: tuple = ("Consolas", 9, "bold")
    font_body: tuple = ("Segoe UI", 10)
    font_small: tuple = ("Segoe UI", 9)
    font_mono: tuple = ("Consolas", 9)

    pad_x: int = 12
    pad_y: int = 8
    border_width: int = 1

    def apply_root(self, root: tk.Tk | tk.Toplevel) -> None:
        root.configure(bg=self.bg_deep)
        root.option_add("*Font", self.font_body)
        root.option_add("*Background", self.bg_deep)
        root.option_add("*Foreground", self.text)
        root.option_add("*selectBackground", self.cyan_dim)
        root.option_add("*selectForeground", self.text)
        root.option_add("*insertBackground", self.cyan)

    def font(self, kind: str = "body", size: int | None = None) -> tkfont.Font:
        mapping = {
            "display": self.font_display,
            "title": self.font_title,
            "body": self.font_body,
            "small": self.font_small,
            "mono": self.font_mono,
        }
        base = mapping.get(kind, self.font_body)
        if size is not None:
            return tkfont.Font(family=base[0], size=size, weight=base[2] if len(base) > 2 else "normal")
        return tkfont.Font(family=base[0], size=base[1], weight=base[2] if len(base) > 2 else "normal")


DEFAULT_THEME = HUDTheme()
