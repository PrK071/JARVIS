#!/usr/bin/env python3
"""Ponto de entrada da interface desktop J.A.R.V.I.S."""

from triade.app import JARVISApp


def main() -> None:
    app = JARVISApp()
    app.run()


if __name__ == "__main__":
    main()
