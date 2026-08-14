from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .cli import main as orchestrator_main
else:
    # Allow Windows double-click/direct execution without losing package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tern.orchestrator.cli import main as orchestrator_main


def main(argv: list[str] | None = None) -> int:
    """Run Jarvis, defaulting to the graphical interface."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"--text", "-t"}:
        arguments = ["text", *arguments[1:]]
    elif arguments and arguments[0] == "codex":
        arguments = ["codex-shared-tui", *arguments[1:]]
    return orchestrator_main(arguments or ["ui"])


def gui_main() -> int:
    """Windows GUI entry point that launches without a console window."""
    return orchestrator_main(["ui"])


if __name__ == "__main__":
    raise SystemExit(main())
