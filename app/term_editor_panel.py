"""app/term_editor_panel.py -- the Term Editor tab.

Live editing of the current RationalMap: two tables (polynomial terms,
pole terms), a live formula display (to_formula(), refreshed after every
edit), add/remove/reorder, a per-term enable checkbox, and undo/redo.

APPLY ON EVERY EDIT, not a button -- the opposite of the Settings tab's
explicit Apply. Term edits are cheap to re-render (nothing like a
resolution change) and this is the actual sandbox: seeing the picture
change as you nudge a coefficient IS the interaction. Every successful
edit calls `on_edit` (see __init__), which SandboxWindow wires to the same
debounced re-render the P5a viewport-drag/zoom path already uses -- so a
burst of edits (dragging a spinbox, typing several digits) coalesces into
one render the same way a scroll burst does, not one render per keystroke.

VALIDATION. RationalMap.add_poly/add_pole (see app.session's
add_poly_term/add_pole_term) can raise ValueError -- a negative exponent,
or a pole location that collides with one that already exists. Every
mutating action here goes through _try_mutate, which shows that message
in the panel's own error label and leaves the map (and the undo stack)
untouched, rather than letting the exception propagate into Qt's event
loop as a crash.

UNDO/REDO. Implemented via whole-map serialize()/deserialize() snapshots,
not a diff/command stack -- RationalMap already has a complete, tested
round-trip for exactly this shape of state, and a term list is small
enough that copying the whole thing on every edit costs nothing worth
optimizing (see CLAUDE.md: measure before optimizing, and there is
nothing here to measure yet).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

import cdx

POLY_COLUMNS = ("On", "Coefficient", "Exponent", "a-power", "Label")
POLE_COLUMNS = ("On", "Location", "Strength", "Order", "a-power", "Loc=a", "Label")

# Column indices, named for readability at the many call sites below that
# need to know which column is which.
POLY_ON, POLY_COEFF, POLY_EXP, POLY_PPOW, POLY_LABEL = range(5)
POLE_ON, POLE_LOC, POLE_STR, POLE_ORDER, POLE_PPOW, POLE_LOCPARAM, POLE_LABEL = range(7)

# Exponent/order/param_power widget ranges. Exponent's floor is negative
# (not 0) even though add_poly now REJECTS a new negative exponent --
# editing an existing term is not gated the same way (see app.session's
# edit_poly_term docstring), and a map loaded from before this restriction
# existed, or from a hand-edited file, can still legitimately have one;
# clamping the widget to >=0 would silently corrupt that term's displayed
# (and, the moment the user touches the spinbox, actual) value the instant
# its row is drawn.
EXPONENT_RANGE = (-50, 50)
ORDER_RANGE = (1, 50)
PARAM_POWER_RANGE = (-20, 20)


def _format_complex(z: complex) -> str:
    # str(complex) always round-trips through complex() exactly, but wraps
    # anything with a nonzero imaginary part in parens ("(1+2j)") -- drop
    # them so the cell shows exactly what a user would type back in.
    s = str(z)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s


def _parse_complex(text: str) -> complex:
    return complex(text.strip())   # may raise ValueError; caller's job to catch it


class TermEditorPanel(QWidget):
    def __init__(self, session, on_edit, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self._on_edit = on_edit
        self._undo_stack: list[str] = []
        self._redo_stack: list[str] = []

        self._build_ui()
        self._refresh_all()

    # ---- construction ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._formula_label = QLabel()
        self._formula_label.setWordWrap(True)
        self._formula_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        layout.addWidget(self._formula_label)

        layout.addWidget(self._build_poly_group())
        layout.addWidget(self._build_pole_group())

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #cc4444;")
        layout.addWidget(self._error_label)

        undo_row = QHBoxLayout()
        self._undo_button = QPushButton("Undo")
        self._undo_button.clicked.connect(self._undo)
        self._redo_button = QPushButton("Redo")
        self._redo_button.clicked.connect(self._redo)
        undo_row.addWidget(self._undo_button)
        undo_row.addWidget(self._redo_button)
        undo_row.addStretch(1)
        layout.addLayout(undo_row)

        layout.addStretch(1)

    def _build_poly_group(self) -> QGroupBox:
        box = QGroupBox("Polynomial Terms")
        v = QVBoxLayout(box)

        table = QTableWidget(0, len(POLY_COLUMNS))
        table.setHorizontalHeaderLabels(POLY_COLUMNS)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.cellChanged.connect(self._on_poly_cell_changed)
        self._poly_table = table
        v.addWidget(table)

        row = QHBoxLayout()
        add_btn = QPushButton("Add Term")
        add_btn.clicked.connect(self._add_poly_term)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_poly_term)
        up_btn = QPushButton("Move Up")
        up_btn.clicked.connect(lambda: self._move_selected_poly_term(-1))
        down_btn = QPushButton("Move Down")
        down_btn.clicked.connect(lambda: self._move_selected_poly_term(1))
        for b in (add_btn, remove_btn, up_btn, down_btn):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _build_pole_group(self) -> QGroupBox:
        box = QGroupBox("Pole Terms")
        v = QVBoxLayout(box)

        table = QTableWidget(0, len(POLE_COLUMNS))
        table.setHorizontalHeaderLabels(POLE_COLUMNS)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.cellChanged.connect(self._on_pole_cell_changed)
        self._pole_table = table
        v.addWidget(table)

        row = QHBoxLayout()
        add_btn = QPushButton("Add Pole")
        add_btn.clicked.connect(self._add_pole_term)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_pole_term)
        up_btn = QPushButton("Move Up")
        up_btn.clicked.connect(lambda: self._move_selected_pole_term(-1))
        down_btn = QPushButton("Move Down")
        down_btn.clicked.connect(lambda: self._move_selected_pole_term(1))
        for b in (add_btn, remove_btn, up_btn, down_btn):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        return box

    # ---- mutation helper: validate, snapshot for undo, report errors inline ------------
    def _try_mutate(self, fn) -> bool:
        """Runs fn() (expected to mutate self.session.map). A snapshot
        taken BEFORE fn() runs is pushed onto the undo stack only if fn()
        completes without raising -- so a rejected edit (ValueError, shown
        in the error label) leaves both the map and the undo history
        exactly as they were, not a no-op entry a user would have to undo
        past.
        """
        snapshot = self.session.map.serialize()
        try:
            fn()
        except ValueError as e:
            self._error_label.setText(str(e))
            return False
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()
        self._error_label.setText("")
        return True

    def _notify_edit(self) -> None:
        self._refresh_all()
        self._on_edit()

    # ---- external resync: session.map was replaced wholesale (e.g. a family load) ------
    def refresh_from_session(self) -> None:
        """Called by SandboxWindow when something OUTSIDE this panel
        replaced self.session.map wholesale (loading a family from the
        Library tab) -- resyncs the tables and formula label, and clears
        undo/redo: a history spanning across an unrelated map isn't
        meaningful undo territory, and every snapshot on the stack was
        relative to a map that self.session.map no longer even is.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh_all()

    # ---- refresh: rebuild both tables + the formula label from session.map -------------
    def _refresh_all(self) -> None:
        self._formula_label.setText(self.session.map.to_formula() or "(empty map)")
        self._refresh_poly_table()
        self._refresh_pole_table()
        self._undo_button.setEnabled(bool(self._undo_stack))
        self._redo_button.setEnabled(bool(self._redo_stack))

    def _refresh_poly_table(self) -> None:
        table = self._poly_table
        table.blockSignals(True)
        terms = self.session.map.poly_terms()
        table.setRowCount(len(terms))
        for row, t in enumerate(terms):
            on = QCheckBox()
            on.setChecked(t.enabled)
            on.stateChanged.connect(lambda state, r=row: self._on_poly_enabled_changed(r, state))
            table.setCellWidget(row, POLY_ON, on)

            table.setItem(row, POLY_COEFF, QTableWidgetItem(_format_complex(t.coeff)))

            exp = QSpinBox()
            exp.setRange(*EXPONENT_RANGE)
            exp.setValue(t.exponent)
            exp.valueChanged.connect(lambda value, r=row: self._on_poly_exponent_changed(r, value))
            table.setCellWidget(row, POLY_EXP, exp)

            ppow = QSpinBox()
            ppow.setRange(*PARAM_POWER_RANGE)
            ppow.setValue(t.param_power)
            ppow.valueChanged.connect(
                lambda value, r=row: self._on_poly_param_power_changed(r, value))
            table.setCellWidget(row, POLY_PPOW, ppow)

            table.setItem(row, POLY_LABEL, QTableWidgetItem(t.label))
        table.blockSignals(False)

    def _refresh_pole_table(self) -> None:
        table = self._pole_table
        table.blockSignals(True)
        terms = self.session.map.pole_terms()
        table.setRowCount(len(terms))
        for row, t in enumerate(terms):
            on = QCheckBox()
            on.setChecked(t.enabled)
            on.stateChanged.connect(lambda state, r=row: self._on_pole_enabled_changed(r, state))
            table.setCellWidget(row, POLE_ON, on)

            table.setItem(row, POLE_LOC, QTableWidgetItem(_format_complex(t.location)))
            table.setItem(row, POLE_STR, QTableWidgetItem(_format_complex(t.strength)))

            order = QSpinBox()
            order.setRange(*ORDER_RANGE)
            order.setValue(t.order)
            order.valueChanged.connect(lambda value, r=row: self._on_pole_order_changed(r, value))
            table.setCellWidget(row, POLE_ORDER, order)

            ppow = QSpinBox()
            ppow.setRange(*PARAM_POWER_RANGE)
            ppow.setValue(t.param_power)
            ppow.valueChanged.connect(
                lambda value, r=row: self._on_pole_param_power_changed(r, value))
            table.setCellWidget(row, POLE_PPOW, ppow)

            loc_param = QCheckBox()
            loc_param.setChecked(t.location_is_param)
            loc_param.stateChanged.connect(
                lambda state, r=row: self._on_pole_location_is_param_changed(r, state))
            table.setCellWidget(row, POLE_LOCPARAM, loc_param)

            table.setItem(row, POLE_LABEL, QTableWidgetItem(t.label))
        table.blockSignals(False)

    # ---- poly: add / remove / move ------------------------------------------------------
    def _add_poly_term(self) -> None:
        if self._try_mutate(lambda: self.session.add_poly_term(1 + 0j, 1, 0)):
            self._notify_edit()

    def _remove_selected_poly_term(self) -> None:
        row = self._poly_table.currentRow()
        if row < 0:
            return
        if self._try_mutate(lambda: self.session.remove_poly_term(row)):
            self._notify_edit()

    def _move_selected_poly_term(self, direction: int) -> None:
        row = self._poly_table.currentRow()
        target = row + direction
        # Checked BEFORE _try_mutate, not via move_poly_term's own return
        # value, so a no-op at the list's boundary never pushes an undo
        # snapshot for a "change" that didn't happen -- _try_mutate always
        # commits its pre-call snapshot when the wrapped call doesn't
        # raise, and move_poly_term never raises, it just returns False.
        if row < 0 or not (0 <= target < len(self.session.map.poly_terms())):
            return
        if self._try_mutate(lambda: self.session.move_poly_term(row, direction)):
            # Reordering never changes what the map evaluates to (it's a
            # sum either way) -- no need to re-render, just redraw the
            # table and keep the formula label in sync.
            self._refresh_all()
            self._poly_table.selectRow(target)

    # ---- pole: add / remove / move -------------------------------------------------------
    # A handful of default locations to try in turn: the origin is the
    # obvious first choice, but it is exactly the one most likely to
    # already be occupied (mandelbrot()-shaped maps, newton_cubic(), ...),
    # so falling back to the next candidate instead of just failing once
    # keeps "Add Pole" usually working on the first click.
    _DEFAULT_POLE_LOCATIONS = (0j, 1 + 0j, 2 + 0j, 3 + 0j, 1j, 2j)

    def _add_pole_term(self) -> None:
        for candidate in self._DEFAULT_POLE_LOCATIONS:
            if self._try_mutate(lambda c=candidate: self.session.add_pole_term(c, 1 + 0j, 1)):
                self._notify_edit()
                return
        self._error_label.setText(
            "every default pole location is already taken -- edit an existing pole's "
            "location instead of adding a new one")

    def _remove_selected_pole_term(self) -> None:
        row = self._pole_table.currentRow()
        if row < 0:
            return
        if self._try_mutate(lambda: self.session.remove_pole_term(row)):
            self._notify_edit()

    def _move_selected_pole_term(self, direction: int) -> None:
        row = self._pole_table.currentRow()
        target = row + direction
        if row < 0 or not (0 <= target < len(self.session.map.pole_terms())):
            return
        if self._try_mutate(lambda: self.session.move_pole_term(row, direction)):
            self._refresh_all()
            self._pole_table.selectRow(target)

    # ---- poly: field edits ---------------------------------------------------------------
    def _on_poly_cell_changed(self, row: int, column: int) -> None:
        if column == POLY_COEFF:
            text = self._poly_table.item(row, POLY_COEFF).text()
            try:
                value = _parse_complex(text)
            except ValueError:
                self._error_label.setText(
                    f"{text!r} is not a valid complex number -- try e.g. 1+2j or -0.5")
                self._refresh_poly_table()
                return
            if self._try_mutate(lambda: self.session.edit_poly_term(row, coeff=value)):
                self._notify_edit()
        elif column == POLY_LABEL:
            text = self._poly_table.item(row, POLY_LABEL).text()
            if self._try_mutate(lambda: self.session.edit_poly_term(row, label=text)):
                self._refresh_all()   # a label change doesn't affect the render

    def _on_poly_enabled_changed(self, row: int, state: int) -> None:
        value = state == Qt.CheckState.Checked.value
        if self._try_mutate(lambda: self.session.edit_poly_term(row, enabled=value)):
            self._notify_edit()

    def _on_poly_exponent_changed(self, row: int, value: int) -> None:
        if self._try_mutate(lambda: self.session.edit_poly_term(row, exponent=value)):
            self._notify_edit()

    def _on_poly_param_power_changed(self, row: int, value: int) -> None:
        if self._try_mutate(lambda: self.session.edit_poly_term(row, param_power=value)):
            self._notify_edit()

    # ---- pole: field edits ----------------------------------------------------------------
    def _on_pole_cell_changed(self, row: int, column: int) -> None:
        if column == POLE_LOC:
            text = self._pole_table.item(row, POLE_LOC).text()
            try:
                value = _parse_complex(text)
            except ValueError:
                self._error_label.setText(
                    f"{text!r} is not a valid complex number -- try e.g. 1+2j or -0.5")
                self._refresh_pole_table()
                return
            if self._try_mutate(lambda: self.session.edit_pole_term(row, location=value)):
                self._notify_edit()
        elif column == POLE_STR:
            text = self._pole_table.item(row, POLE_STR).text()
            try:
                value = _parse_complex(text)
            except ValueError:
                self._error_label.setText(
                    f"{text!r} is not a valid complex number -- try e.g. 1+2j or -0.5")
                self._refresh_pole_table()
                return
            if self._try_mutate(lambda: self.session.edit_pole_term(row, strength=value)):
                self._notify_edit()
        elif column == POLE_LABEL:
            text = self._pole_table.item(row, POLE_LABEL).text()
            if self._try_mutate(lambda: self.session.edit_pole_term(row, label=text)):
                self._refresh_all()

    def _on_pole_enabled_changed(self, row: int, state: int) -> None:
        value = state == Qt.CheckState.Checked.value
        if self._try_mutate(lambda: self.session.edit_pole_term(row, enabled=value)):
            self._notify_edit()

    def _on_pole_order_changed(self, row: int, value: int) -> None:
        if self._try_mutate(lambda: self.session.edit_pole_term(row, order=value)):
            self._notify_edit()

    def _on_pole_param_power_changed(self, row: int, value: int) -> None:
        if self._try_mutate(lambda: self.session.edit_pole_term(row, param_power=value)):
            self._notify_edit()

    def _on_pole_location_is_param_changed(self, row: int, state: int) -> None:
        value = state == Qt.CheckState.Checked.value
        if self._try_mutate(lambda: self.session.edit_pole_term(row, location_is_param=value)):
            self._notify_edit()

    # ---- undo / redo -----------------------------------------------------------------------
    def _restore_snapshot(self, text: str) -> None:
        # cdx.RationalMap.deserialize(text) -> RationalMap, raising
        # ValueError on failure (the Python binding's own translation of
        # the C++ bool+error-string pattern -- see bindings.cpp) -- not
        # reachable in practice here, since every snapshot was itself
        # produced by this same map's own serialize() moments earlier, but
        # caught rather than assumed so a corrupted snapshot reports
        # clearly instead of crashing.
        try:
            self.session.map = cdx.RationalMap.deserialize(text)
        except ValueError as e:
            self._error_label.setText(f"could not restore undo snapshot: {e}")
            return
        self._notify_edit()

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self.session.map.serialize())
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self.session.map.serialize())
        snapshot = self._redo_stack.pop()
        self._restore_snapshot(snapshot)
