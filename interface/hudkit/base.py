"""Classe base para aplicativos HUD."""

import tkinter as tk
from hudkit.theme import HUDTheme, DEFAULT_THEME


class HUDApplication:
    """
    Janela base com header, área central em 3 colunas e footer.
    Subclasses implementam build_left_panel, build_center_panel e build_right_panel.
    """

    title: str = "HUD Application"
    geometry: str = "1280x720"
    min_size: tuple[int, int] = (1024, 600)

    def __init__(self, theme: HUDTheme | None = None) -> None:
        self.theme = theme or DEFAULT_THEME
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(self.geometry)
        self.root.minsize(*self.min_size)
        self.theme.apply_root(self.root)

        self._header_frame: tk.Frame | None = None
        self._footer_frame: tk.Frame | None = None
        self._left_frame: tk.Frame | None = None
        self._center_frame: tk.Frame | None = None
        self._right_frame: tk.Frame | None = None

        self._build_shell()
        self.build_left_panel(self._left_frame)
        self.build_center_panel(self._center_frame)
        self.build_right_panel(self._right_frame)
        self.on_ready()

    def _build_shell(self) -> None:
        t = self.theme
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._header_frame = tk.Frame(self.root, bg=t.bg_panel, highlightbackground=t.border, highlightthickness=1)
        self._header_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        body = tk.Frame(self.root, bg=t.bg_deep)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        body.columnconfigure(0, weight=0, minsize=260)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=280)
        body.rowconfigure(0, weight=1)

        self._left_frame = tk.Frame(body, bg=t.bg_deep, width=260)
        self._left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._left_frame.grid_propagate(False)

        self._center_frame = tk.Frame(body, bg=t.bg_deep)
        self._center_frame.grid(row=0, column=1, sticky="nsew", padx=6)

        self._right_frame = tk.Frame(body, bg=t.bg_deep, width=280)
        self._right_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self._right_frame.grid_propagate(False)

        self._footer_frame = tk.Frame(self.root, bg=t.bg_panel, highlightbackground=t.border, highlightthickness=1)
        self._footer_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))

    def build_left_panel(self, parent: tk.Frame) -> None:
        pass

    def build_center_panel(self, parent: tk.Frame) -> None:
        pass

    def build_right_panel(self, parent: tk.Frame) -> None:
        pass

    def on_ready(self) -> None:
        """Hook chamado após montagem da interface."""
        pass

    @property
    def header(self) -> tk.Frame:
        assert self._header_frame is not None
        return self._header_frame

    @property
    def footer(self) -> tk.Frame:
        assert self._footer_frame is not None
        return self._footer_frame

    def run(self) -> None:
        self.root.mainloop()

    def schedule(self, ms: int, callback) -> None:
        self.root.after(ms, callback)
