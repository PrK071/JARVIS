from __future__ import annotations

import sys

from .cli import main as orchestrator_main


def main(argv: list[str] | None = None) -> int:
    """Run the assistant CLI, defaulting to the continuous voice session."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    return orchestrator_main(arguments or ["voice"])


if __name__ == "__main__":
    raise SystemExit(main())
