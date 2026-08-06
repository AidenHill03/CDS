"""app/metadata_header.py -- a compact, always-visible readout of what is
currently being rendered: map name, formula, domain (parameter vs
dynamical plane), current parameter, and render mode.

Without this, a Newton parameter plane, a Mandelbrot parameter plane, and
a rational-map dynamical plane are all visually indistinguishable in a
screenshot -- P5c's own stated reason for building this. format_metadata_text
below is deliberately a free function, not logic buried inside the widget,
so a future image-export feature can reuse the EXACT same text as an
exported image's caption (P5c's own wording) without needing to touch
this widget at all -- one function is the single source of truth for both
uses, even though only the display side is built here (P5c's task list
has no separate "image export" section; this only makes that text
trivially reusable whenever one exists).
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.session import PARAMETER_PLANE_MODES


def _format_complex(z: complex) -> str:
    return f"{z.real:.6g}{'+' if z.imag >= 0 else ''}{z.imag:.6g}j"


def format_metadata_text(session) -> str:
    is_parameter_plane = session.render_mode in PARAMETER_PLANE_MODES
    domain = "Parameter plane" if is_parameter_plane else "Dynamical plane"
    # On the parameter plane the bound parameter is IGNORED by the render
    # (every pixel IS a parameter -- see cdx::Renderer::render_parameter's
    # own doc comment) -- showing session.param there would look like it
    # means something when it has no effect on what's on screen at all.
    param_part = ("a = pixel (bound parameter has no effect here)" if is_parameter_plane
                 else f"a = {_format_complex(session.param)}")
    formula = session.map.to_formula() or "(empty map)"
    return (f"{session.map.name}: {formula}   |   {domain}   |   {param_part}   |   "
            f"mode: {session.render_mode}")


class MetadataHeader(QWidget):
    def __init__(self, session, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("font-family: monospace; font-weight: bold;")
        layout.addWidget(self._label)

        self.refresh()

    def refresh(self) -> None:
        self._label.setText(format_metadata_text(self.session))
