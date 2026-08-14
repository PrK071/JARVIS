from __future__ import annotations

import sys

from .cli import main as orchestrator_main


def main(argv: list[str] | None = None) -> int:
    """Run the assistant CLI, defaulting to the continuous voice session."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"--text", "-t"}:
        arguments = ["text", *arguments[1:]]
    elif arguments and arguments[0] == "codex":
        arguments = ["codex-shared-tui", *arguments[1:]]
    return orchestrator_main(arguments or ["voice"])


if __name__ == "__main__":
    raise SystemExit(main())
