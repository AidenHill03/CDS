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


def _role_phrase(param_power: int, what: str) -> str:
    if param_power == 1:
        return what
    return f"a^{param_power} multiplies {what}"


def describe_parameter_role(rational_map) -> str:
    """Reads the map's ACTUAL term list to describe what `a` does here --
    never assumed to be an additive constant (P6's own wording). The
    engine binds `a` per-term: PolyTerm.param_power multiplies a poly
    term's coefficient by a^param_power, PoleTerm.param_power does the
    same for a pole's strength, and PoleTerm.location_is_param makes a
    pole's LOCATION equal to `a` outright (see PoleTerm::effective_location
    in rational.cpp -- a full replacement of the stored location, not an
    offset). A map can combine several of these on different terms at
    once, or (newton_cubic(), confirmed directly) depend on `a` not at
    all -- both are read from the terms, not hardcoded per family.
    """
    roles: list[str] = []
    for t in rational_map.poly_terms():
        if t.enabled and t.param_power != 0:
            roles.append(_role_phrase(t.param_power, f"coefficient of z^{t.exponent}"))
    for i, t in enumerate(rational_map.pole_terms()):
        if not t.enabled:
            continue
        name = t.label or f"pole {i + 1}"
        if t.location_is_param:
            roles.append(f"location of {name}")
        if t.param_power != 0:
            roles.append(_role_phrase(t.param_power, f"strength of {name}"))
    if not roles:
        return "unused by this map"
    return "; ".join(roles)


def format_metadata_text(session) -> str:
    is_parameter_plane = session.render_mode in PARAMETER_PLANE_MODES
    domain = "Parameter plane" if is_parameter_plane else "Dynamical plane"
    role = describe_parameter_role(session.map)
    # On the parameter plane the bound parameter is IGNORED by the render
    # (every pixel IS a parameter -- see cdx::Renderer::render_parameter's
    # own doc comment) -- showing session.param there would look like it
    # means something when it has no effect on what's on screen at all.
    # The ROLE description stays meaningful either way: it says what the
    # a-coordinate itself controls, independent of whether this specific
    # render currently uses session.param's bound value.
    param_part = (f"a = pixel [{role}] (bound parameter has no effect here)" if is_parameter_plane
                 else f"a = {_format_complex(session.param)} [{role}]")
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
