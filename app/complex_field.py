"""app/complex_field.py -- a single-line, always-visible complex-number
entry field: the typed half of P6's "symmetry of input" principle for the
parameter `a` and the orbit seed `z0`. Both are complex numbers the user
chooses either by typing here or by clicking a plane; this is the field
side both controls share, so they behave identically rather than each
reinventing parsing/validation/error-display.

PARSING reuses cdx.Expr -- the SAME grammar/interpreter the term-based
sandbox already ships (P5a's expr.hpp) -- rather than hand-rolling a
second complex-literal parser. Restricted to LITERAL values: a compiled
expression is evaluated at two different (z, a) pairs, and rejected
unless both agree, since a plain number's value cannot depend on either
variable -- this is what turns the general "z^5 + a/z^2 - 0.3" grammar
into a literal-only check without cdx.Expr needing to expose its own
opcode list for introspection. Accepts 'i' as the imaginary unit (2-3i,
i, -i), matching cdx.Expr's own grammar -- deliberately NOT the same
convention app.facts_panel's read-only table formatting uses (Python's
built-in complex(), which wants 'j') -- that formatting predates this
field and reuses working code rather than being touched here; a future
pass could unify them, but that is a separate decision.

ON MALFORMED INPUT: the field marks itself invalid (red border + a
tooltip with the parser's own message) and returns without emitting
`committed` -- the caller's model is never touched, never zeroed, and
nothing raises into Qt's event loop.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

import cdx

_INVALID_STYLE = "QLineEdit { border: 1px solid #cc4444; background: #fff0f0; }"


def parse_complex_literal(text: str) -> complex:
    """Parses a plain complex literal: '-0.7269+0.1889i', '2-3i', 'i',
    '-i', '1', '3.2e-1+0i', and so on. Raises ValueError -- with
    cdx.Expr's own parser message on a syntax error, or this function's
    own for the non-finite/depends-on-z-or-a cases -- on anything that
    isn't a plain number. Caller's job to catch it, per this module's own
    convention (see ComplexField._on_editing_finished).
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("enter a complex number")
    expr = cdx.Expr()
    expr.compile(stripped)   # raises ValueError with the parser's own message
    v0 = expr(0j, 0j)
    v1 = expr(1.0 + 1.0j, 1.0 + 1.0j)
    if not (math.isfinite(v0.real) and math.isfinite(v0.imag)):
        raise ValueError("must be a finite complex number")
    if v0 != v1:
        raise ValueError("must be a plain complex number, not an expression in z or a")
    return v0


def format_complex_literal(z: complex) -> str:
    """The display counterpart of parse_complex_literal -- always
    round-trips back through it, using 'i' (never Python's 'j')."""
    if z.imag == 0:
        return f"{z.real:g}"
    real_part = "" if z.real == 0 else f"{z.real:g}"
    sign = "+" if z.imag >= 0 else "-"
    magnitude = abs(z.imag)
    imag_part = "i" if magnitude == 1 else f"{magnitude:g}i"
    if not real_part:
        return f"{sign}{imag_part}" if sign == "-" else imag_part
    return f"{real_part}{sign}{imag_part}"


class ComplexField(QWidget):
    """committed(complex) fires only when a NEWLY TYPED, valid value
    actually differs from the current one -- set_value() (the
    programmatic path: plane clicks, syncing from the model) updates the
    display without re-emitting committed, so the two input methods never
    feed back into each other in a loop.
    """

    committed = Signal(complex)

    def __init__(self, label_text: str, initial: complex, parent: QWidget | None = None):
        super().__init__(parent)
        self._value = initial

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label_text, self))
        self._line_edit = QLineEdit(format_complex_literal(initial), self)
        self._line_edit.setMaximumWidth(160)
        self._line_edit.editingFinished.connect(self._on_editing_finished)
        layout.addWidget(self._line_edit)

    @property
    def value(self) -> complex:
        return self._value

    def set_value(self, z: complex) -> None:
        """Programmatic update (plane click, resync from the model) --
        does NOT emit `committed`, so this never re-triggers whatever
        just set the value in the first place.
        """
        self._value = z
        self._line_edit.setStyleSheet("")
        self._line_edit.setToolTip("")
        self._line_edit.setText(format_complex_literal(z))

    def _on_editing_finished(self) -> None:
        try:
            value = parse_complex_literal(self._line_edit.text())
        except ValueError as exc:
            self._line_edit.setStyleSheet(_INVALID_STYLE)
            self._line_edit.setToolTip(str(exc))
            return   # last valid value untouched; nothing committed
        self._line_edit.setStyleSheet("")
        self._line_edit.setToolTip("")
        self._line_edit.setText(format_complex_literal(value))
        if value != self._value:
            self._value = value
            self.committed.emit(value)
