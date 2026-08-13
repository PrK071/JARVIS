"""HUD Kit — framework tkinter para interfaces estilo JARVIS / HUD futurista."""

from hudkit.base import HUDApplication
from hudkit.theme import HUDTheme
from hudkit.widgets import (
    HUDPanel,
    HUDHeader,
    HUDFooter,
    HUDMetric,
    HUDButton,
    HUDLogTerminal,
    HUDCommandBar,
    DecisionBars,
    InferenceHistory,
)
from hudkit.animations import CoreReactor, WaveformDisplay, TernaryGrid

__all__ = [
    "HUDApplication",
    "HUDTheme",
    "HUDPanel",
    "HUDHeader",
    "HUDFooter",
    "HUDMetric",
    "HUDButton",
    "HUDLogTerminal",
    "HUDCommandBar",
    "DecisionBars",
    "InferenceHistory",
    "CoreReactor",
    "WaveformDisplay",
    "TernaryGrid",
]

__version__ = "1.0.0"
