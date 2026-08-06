"""app/about_dialog.py -- the About dialog: version, author, and a short,
ACCURATE capability/roadmap summary.

The summary is deliberately re-derived from what has actually shipped
(this file's own CAPABILITIES/ROADMAP constants), not copied from
CLAUDE.md's "Current state" section, which describes an earlier snapshot
of the project (pre-P5b/P5c) and would be actively misleading if reused
verbatim here -- an About dialog that's wrong about what the app can do
is worse than no About dialog at all.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from app.version import AUTHOR, PRODUCT_NAME, VERSION

CAPABILITIES = (
    "Julia, parameter-plane, basin, and Green's-function rendering "
    "(both the dynamical G_f(z) and the parameter-plane family "
    "escape-rate function G_M(c))",
    "A term-based rational-map editor -- poles as first-class objects, "
    "live-edited with undo/redo",
    "A family library: six built-in presets plus save/rename/delete for "
    "your own",
    "Dynamical facts (critical points, fixed points, attracting cycles) "
    "with an inline Riemann-Hurwitz check",
    "Critical-point overlay with optional forward-orbit tracing",
    "Click-to-seed orbit tracking with attractor/cycle classification",
    "A colour pipeline with adjustable scaling, four palettes, basin "
    "shading, and equipotential bands with contour lines",
)

ROADMAP = (
    "The simultaneous-approximation construction (Fisher-Hill-Lazebnik-"
    "Thompson) this sandbox exists to support -- not yet started; see "
    "ARCHITECTURE.md's own notes on what it will need.",
)


def about_text() -> str:
    capabilities = "\n".join(f"  • {c}" for c in CAPABILITIES)
    roadmap = "\n".join(f"  • {r}" for r in ROADMAP)
    return (f"{PRODUCT_NAME}\nVersion {VERSION}\n{AUTHOR}\n\n"
            f"A rational-map dynamics sandbox, on the Riemann sphere throughout.\n\n"
            f"Capabilities:\n{capabilities}\n\n"
            f"Roadmap:\n{roadmap}")


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"About {PRODUCT_NAME}")

        layout = QVBoxLayout(self)
        self._label = QLabel(about_text(), self)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, self)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
