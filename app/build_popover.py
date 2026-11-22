"""app/build_popover.py -- the Equation tab's "Build ▾" popover.

Choose a polynomial p(z) (type, degree, alpha/beta/lambda when the type
needs them -- each a number or a), and a map built on it (p + a, a·p,
Newton, relaxed Newton, Halley, Chebyshev's method, damped Halley). The
popover previews the typeset map, the expression text it will put in the
f(z) field (written with the parser's own special-polynomial functions, so
it stays editable), and the map's degree and number of free critical
points. Apply fills the field and applies it; Save as family also adds it
to the Library under a generated name ("Relaxed Newton · Hermite H₅").

All the logic is app.family_builder; this is only the form.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from app import family_builder as fb
from app.mathdisplay import FormulaView

RECOMPUTE_DEBOUNCE_MS = 120


class BuildPopover(QFrame):
    """A popup form. `on_apply(expression)` applies an expression to the
    f(z) field; `on_save(expression, name)` applies it and saves it to the
    Library under `name`."""

    def __init__(self, on_apply: Callable[[str], bool], on_save: Callable[[str, str], bool] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("BuildPopover")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._on_apply = on_apply
        self._on_save = on_save
        self._built: fb.Built | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        title = QLabel("POLYNOMIAL p(z)", self)
        title.setProperty("role", "section")
        outer.addWidget(title)
        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        self.type_combo = QComboBox(self)
        for t in fb.POLY_TYPES:
            self.type_combo.addItem(t.label, t.key)
        self.type_combo.setAccessibleName("Polynomial type")
        form.addRow("Type", self.type_combo)
        self.degree_spin = QSpinBox(self)
        self.degree_spin.setRange(1, fb.MAX_DEGREE)
        self.degree_spin.setValue(5)
        self.degree_spin.setAccessibleName("Degree")
        form.addRow("Degree n", self.degree_spin)
        self.symbol_fields: dict[str, QLineEdit] = {}
        self._symbol_labels: dict[str, QLabel] = {}
        for sym in ("alpha", "beta", "lam"):
            field = QLineEdit(fb.SYMBOL_DEFAULT[sym], self)
            field.setProperty("mono", True)
            field.setPlaceholderText("a number, or a")
            field.setAccessibleName(fb.SYMBOL_LABEL[sym])
            label = QLabel(fb.SYMBOL_LABEL[sym], self)
            form.addRow(label, field)
            self.symbol_fields[sym] = field
            self._symbol_labels[sym] = label
        self.custom_field = QLineEdit("z^3 - 1", self)
        self.custom_field.setProperty("mono", True)
        self.custom_field.setAccessibleName("Custom p(z)")
        self._custom_label = QLabel("p(z) =", self)
        form.addRow(self._custom_label, self.custom_field)
        outer.addLayout(form)

        map_title = QLabel("MAP", self)
        map_title.setProperty("role", "section")
        outer.addWidget(map_title)
        self.map_combo = QComboBox(self)
        for m in fb.MAP_KINDS:
            self.map_combo.addItem(f"{m.label}  —  {m.formula}", m.key)
        self.map_combo.setCurrentIndex(self.map_combo.findData("relaxed_newton"))
        self.map_combo.setAccessibleName("Map")
        outer.addWidget(self.map_combo)

        prev_title = QLabel("PREVIEW", self)
        prev_title.setProperty("role", "section")
        outer.addWidget(prev_title)
        self.preview = FormulaView(self, compact=True, px=17, height=56, tooltip=True)
        self.preview.setMinimumWidth(380)
        outer.addWidget(self.preview)
        self.expression_label = QLabel(self)
        self.expression_label.setProperty("mono", True)
        self.expression_label.setProperty("role", "muted")
        self.expression_label.setWordWrap(True)
        self.expression_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.expression_label.setMaximumWidth(460)
        outer.addWidget(self.expression_label)
        self.stats_label = QLabel(self)
        self.stats_label.setProperty("role", "caption")
        outer.addWidget(self.stats_label)
        self.error_label = QLabel(self)
        self.error_label.setProperty("role", "error")
        self.error_label.setWordWrap(True)
        self.error_label.setMaximumWidth(460)
        outer.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.save_button = QPushButton("Save as family", self)
        self.save_button.setToolTip("Apply, and add it to the Library under the generated name")
        self.save_button.clicked.connect(self.save)
        self.save_button.setVisible(on_save is not None)
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.setProperty("role", "primary")
        self.apply_button.clicked.connect(self.apply)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.apply_button)
        outer.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(RECOMPUTE_DEBOUNCE_MS)
        self._timer.timeout.connect(self.recompute)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self.degree_spin.valueChanged.connect(self._schedule)
        self.map_combo.currentIndexChanged.connect(self._schedule)
        for f in list(self.symbol_fields.values()) + [self.custom_field]:
            f.textChanged.connect(self._schedule)
        self._on_type_changed()
        self.recompute()

    # ---- state --------------------------------------------------------------------------------
    def choice(self) -> fb.BuilderChoice:
        return fb.BuilderChoice(
            poly=self.type_combo.currentData(), degree=self.degree_spin.value(),
            alpha=self.symbol_fields["alpha"].text(), beta=self.symbol_fields["beta"].text(),
            lam=self.symbol_fields["lam"].text(), custom=self.custom_field.text(),
            map_kind=self.map_combo.currentData())

    def set_choice(self, c: fb.BuilderChoice) -> None:
        self.type_combo.setCurrentIndex(self.type_combo.findData(c.poly))
        self.degree_spin.setValue(c.degree)
        self.symbol_fields["alpha"].setText(c.alpha)
        self.symbol_fields["beta"].setText(c.beta)
        self.symbol_fields["lam"].setText(c.lam)
        self.custom_field.setText(c.custom)
        self.map_combo.setCurrentIndex(self.map_combo.findData(c.map_kind))
        self.recompute()

    def expression(self) -> str | None:
        return self._built.expression if self._built is not None else None

    def generated_name(self) -> str:
        return fb.family_name(self.choice())

    # ---- reactions ----------------------------------------------------------------------------
    def _on_type_changed(self) -> None:
        t = fb.POLY_BY_KEY[self.type_combo.currentData()]
        for sym, field in self.symbol_fields.items():
            visible = sym in t.symbols
            field.setVisible(visible)
            self._symbol_labels[sym].setVisible(visible)
        custom = t.key == "custom"
        self.custom_field.setVisible(custom)
        self._custom_label.setVisible(custom)
        self.degree_spin.setEnabled(not custom)
        self.degree_spin.setMinimum(t.min_degree)
        self.adjustSize()
        self._schedule()

    def _schedule(self, *_args) -> None:
        self._timer.start()

    def recompute(self) -> None:
        self._timer.stop()
        try:
            built = fb.build(self.choice())
        except fb.BuilderError as e:
            self._built = None
            self.error_label.setText(str(e))
            self.preview.set_source("")
            self.expression_label.setText("")
            self.stats_label.setText("")
            self.apply_button.setEnabled(False)
            self.save_button.setEnabled(False)
            return
        self._built = built
        self.error_label.setText("\n".join(f"Warning: {w}" for w in built.warnings))
        self.preview.set_source(built.expression)
        self.expression_label.setText(built.expression)
        free = "?" if built.free_critical_points is None else str(built.free_critical_points)
        noun = "point" if built.free_critical_points == 1 else "points"
        self.stats_label.setText(f"degree {built.degree} · {free} free critical {noun}"
                                 + ("" if built.parameters else " · no parameter"))
        self.apply_button.setEnabled(True)
        self.save_button.setEnabled(True)

    # ---- actions ------------------------------------------------------------------------------
    def apply(self) -> bool:
        self.recompute()
        if self._built is None:
            return False
        ok = self._on_apply(self._built.expression)
        if ok:
            self.hide()
        return ok

    def save(self) -> bool:
        self.recompute()
        if self._built is None or self._on_save is None:
            return False
        ok = self._on_save(self._built.expression, self.generated_name())
        if ok:
            self.hide()
        return ok

    def popup_below(self, anchor: QWidget) -> None:
        self.recompute()
        self.adjustSize()
        pos = anchor.mapToGlobal(QPoint(0, anchor.height() + 4))
        self.move(pos)
        self.show()
        self.raise_()
