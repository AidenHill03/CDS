"""phoenix.py -- Project Phoenix's entry point (the composition root).

The one place that knows about every layer: it starts the sandbox (app/) and
hands it the panels of the research modules (modules/), so that neither
app/ nor cdx/ ever imports a module (see ARCHITECTURE.md). Running
`python -m app.sandbox` still starts the plain sandbox without them.

    PYTHONPATH=cdx/build python phoenix.py

A module whose compiled kernel has not been built is skipped with a message
on stderr, rather than stopping the app from starting: build it with

    cmake -B modules/approximation/build -S modules/approximation
    cmake --build modules/approximation/build --config Release
"""

from __future__ import annotations

import sys


def _module_panels():
    factories = []
    try:
        from modules.approximation.panel import make_panel as approximation_panel
        factories.append(approximation_panel)
    except ImportError as exc:   # kernel not built, or a dependency missing
        print(f"phoenix: Approximation panel unavailable ({exc}); build modules/approximation "
              f"to enable it.", file=sys.stderr)
    return factories


def main() -> None:
    from app.sandbox import main as sandbox_main
    sandbox_main(extra_panels=_module_panels())


if __name__ == "__main__":
    main()
