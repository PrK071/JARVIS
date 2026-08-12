#!/usr/bin/env python3
"""Ponto de entrada do T.R.I.A.D.E — aplicativo desktop tkinter."""

from triade.app import TRIADEApp


def main() -> None:
    app = TRIADEApp()
    app.run()


if __name__ == "__main__":
    main()
