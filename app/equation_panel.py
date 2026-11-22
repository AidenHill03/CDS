"""app/equation_panel.py -- the Equation tab (P/Q milestone, Stage 4).

Replaces the old two-table term/pole editor (app/term_editor_panel.py,
retired) with a single text field for f(z) = ..., authored directly the
way the user would write it on paper -- "z^2 + a", "z^3 + a/z^3", a nested
"1/(1/(z+1) + a)", whatever the cdx rational-expression grammar
(cdx.parse_rational_parameters, cdx/rational_parser.hpp) accepts. Every
edit goes through cdx.RationalMap.from_expression (Stage 3), which parses
the text, reduces it to the engine's own single-active-parameter P/Q form,
and replaces session.map wholesale -- there is no more per-term structure
to edit in place.

VALIDATION IS LIVE, APPLYING IS NOT. As the formula text changes, it is
re-parsed on every keystroke (cdx.parse_rational_parameters) purely to (a)
mark the field red with the parser's own message on invalid syntax and
(b) keep the parameter UI (which one is active, fixed values for the
rest) in sync with whatever parameters the CURRENT text actually
references -- but session.map itself is untouched until Return is
pressed in the formula field (the UI redesign removed the Apply button). Unlike the old term editor's per-edit auto-apply (numeric field
edits are always complete/valid the instant they're typed), a formula is
usually SYNTACTICALLY INCOMPLETE for most of the time it's being typed
("z^2 + ", "z^2 + a" one character at a time), and building a full map
additionally needs the active-parameter/fixed-values selections below the
text field, not just valid syntax -- so this follows the Settings tab's
explicit-Apply pattern, not the old term editor's live-apply one.

PARAMETER UI. cdx.parse_rational_parameters(text) is the SAME parameter
list from_expression itself derives; this panel shows it as a table:
name | value | "on plane" radio. The radio picks the ACTIVE parameter (the
one the parameter plane sweeps; auto-selected when there is exactly one),
whose value field is the window's own `a` field (set_active_field); every
OTHER parameter gets a ComplexField for its fixed substitution value -- see cdx.
RationalMap.from_expression's own doc comment for exactly what "active"
and "fixed" mean. Rebuilt whenever the parsed parameter list changes,
preserving already-entered values for names that persist across the edit.

RETAINED: direct pole/zero addition (cdx.RationalMap.add_pole_at/
add_zero_at -- forward root->factor on the current P/Q, NOT a search for a
location with some prescribed dynamical property, which is a separate,
out-of-scope inverse problem -- see their own doc comments).

VALIDATION PATTERN. Every mutating action (_apply, _add_pole, _add_zero,
undo/redo's own snapshot restore) goes through the SAME "try, catch
ValueError, show it in the error label, leave session.map untouched"
shape app.term_editor_panel's own _try_mutate established -- kept as
inline try/excepts here rather than reintroducing that exact helper,
since (unlike the old panel) not every mutating action here shares one
signature (_apply needs the formula text AND the parameter selections;
_add_pole/_add_zero need a single complex location).

UNDO/REDO. Same whole-map serialize()/deserialize() snapshot mechanism as
the old panel -- RationalMap's round trip already covers BOTH
representations (see rational.cpp's own "pq2"/"pqfixed" serialization
lines), so this needed no changes to keep working across the P/Q switch.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget)

import cdx
from app import controls, theme
from app.complex_field import ComplexField
from app.mathdisplay import FormulaView, LivePreview

PREVIEW_PX = 18         # the live preview's base size (full-size typesetting)
PREVIEW_DEBOUNCE_MS = 150


class EquationPanel(QWidget):
    def __init__(self, session, on_edit, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self._on_edit = on_edit
        self._undo_stack: list[str] = []
        self._redo_stack: list[str] = []
        self._current_params: list[str] = []
        self._param_fields: dict[str, ComplexField] = {}

        self._build_ui()
        self._refresh_all()

    # ---- construction ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Two columns (mockup B): the formula on the left, the parameter
        # table on the right; they stack when the dock is narrow.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(24)

        left = QVBoxLayout()
        left.setSpacing(6)
        eq_row = QHBoxLayout()
        fz = QLabel("f(z) =")
        fz.setProperty("mono", True)
        eq_row.addWidget(fz)
        self._formula_edit = QLineEdit()
        self._formula_edit.setProperty("mono", True)
        self._formula_edit.setAccessibleName("f(z)")
        self._formula_edit.textChanged.connect(self._on_formula_text_changed)
        # Return applies (the field validates live; Return commits).
        self._formula_edit.returnPressed.connect(self._apply)
        eq_row.addWidget(self._formula_edit, 1)
        left.addLayout(eq_row)

        # Live typeset preview of what is being typed (debounced). On a parse
        # error the last good preview stays, greyed, and the error shows below.
        self._preview = FormulaView(compact=False, px=PREVIEW_PX)
        self._preview.setAccessibleName("Formula preview")
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._preview_scroll.setWidget(self._preview)
        self._preview_scroll.setWidgetResizable(False)
        self._preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._preview_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        left.addWidget(self._preview_scroll)
        self._live_preview = LivePreview(self._preview, PREVIEW_DEBOUNCE_MS, self)
        self._preview.installEventFilter(self)

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setProperty("role", "error")
        left.addWidget(self._error_label)

        # Non-fatal parser notes (e.g. a special polynomial above degree 25).
        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setProperty("role", "caption")
        self._warning_label.hide()
        left.addWidget(self._warning_label)

        undo_text = controls.key_sequence_label(controls.key_sequences("app.undo")[0])
        self._hint_label = QLabel(f"Press Return to apply · {undo_text} undoes")
        self._hint_label.setProperty("role", "caption")
        left.addWidget(self._hint_label)

        # Add pole / Add zero: one compact row, plus Undo/Redo.
        pz_row = QHBoxLayout()
        pz_row.setSpacing(6)
        self._pole_field = ComplexField("", 0j)
        self._pole_field.setToolTip("Multiply the current map by a pole at this point")
        add_pole_btn = QPushButton("Add pole")
        add_pole_btn.setToolTip("Add a pole at the point on the left (forward root -> factor on P/Q)")
        add_pole_btn.clicked.connect(self._add_pole)
        self._zero_field = ComplexField("", 0j)
        add_zero_btn = QPushButton("Add zero")
        add_zero_btn.setToolTip("Add a zero at the point on the left")
        add_zero_btn.clicked.connect(self._add_zero)
        for w in (self._pole_field, add_pole_btn, self._zero_field, add_zero_btn):
            pz_row.addWidget(w)
        pz_row.addSpacing(12)
        self._undo_button = QPushButton("Undo")
        self._undo_button.clicked.connect(self._undo)
        self._redo_button = QPushButton("Redo")
        self._redo_button.clicked.connect(self._redo)
        pz_row.addWidget(self._undo_button)
        pz_row.addWidget(self._redo_button)
        pz_row.addStretch(1)
        left.addLayout(pz_row)
        left.addStretch(1)
        outer.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(6)
        self._param_grid = QGridLayout()
        self._param_grid.setHorizontalSpacing(10)
        self._param_grid.setVerticalSpacing(6)
        right.addLayout(self._param_grid)
        right.addStretch(1)
        outer.addLayout(right, 2)

        # Which parameter the parameter plane sweeps: the model behind the
        # table's "on plane" radios (kept as a combo so the selection logic
        # below is unchanged; never shown).
        self._active_combo = QComboBox(self)
        self._active_combo.hide()
        self._active_combo.currentTextChanged.connect(self._on_active_param_changed)
        self._plane_group = QButtonGroup(self)
        self._plane_group.setExclusive(True)
        self._plane_group.buttonClicked.connect(
            lambda b: self._active_combo.setCurrentText(b.property("paramName")))
        self._active_field: QWidget | None = None
        self._table_widgets: list[QWidget] = []

    def eventFilter(self, obj, event) -> bool:   # noqa: N802
        if obj is self._preview and event.type() == event.Type.LayoutRequest:
            self._fit_preview()
        return super().eventFilter(obj, event)

    def _fit_preview(self) -> None:
        self._preview.adjustSize()
        bar = self._preview_scroll.horizontalScrollBar()
        extra = bar.sizeHint().height() if self._preview.width() > self._preview_scroll.viewport().width() else 0
        self._preview_scroll.setFixedHeight(self._preview.sizeHint().height() + extra + 2)

    def preview(self) -> FormulaView:
        """The live typeset preview under f(z) = (for tests and snapshots)."""
        return self._preview

    def flush_preview(self) -> None:
        """Applies any pending (debounced) preview update now."""
        self._live_preview.flush()

    def set_active_field(self, field: QWidget) -> None:
        """The value field for the ACTIVE parameter -- the window's own `a`
        field (kept in sync with clicks and nudges on the plane)."""
        self._active_field = field
        self._rebuild_param_table()

    def _rebuild_param_table(self) -> None:
        keep = set(self._param_fields.values()) | {self._active_field}
        for w in self._table_widgets:
            self._param_grid.removeWidget(w)
            if w not in keep:
                w.hide()
                w.deleteLater()
        self._table_widgets = []
        if self._active_field is not None:
            self._param_grid.removeWidget(self._active_field)
        for b in list(self._plane_group.buttons()):
            self._plane_group.removeButton(b)

        def add(widget, row, col):
            self._param_grid.addWidget(widget, row, col)
            if widget is not self._active_field:
                self._table_widgets.append(widget)

        header_param = QLabel("Parameter")
        header_plane = QLabel("On plane")
        for h in (header_param, header_plane):
            h.setProperty("role", "caption")
        add(header_param, 0, 0)
        add(header_plane, 0, 2)
        if not self._current_params:
            none = QLabel("This map has no parameters.")
            none.setProperty("role", "muted")
            add(none, 1, 0)
            if self._active_field is not None:
                self._active_field.hide()
            return
        active = self._active_combo.currentText()
        for row, name in enumerate(self._current_params, start=1):
            label = QLabel(name)
            label.setProperty("mono", True)
            label.setProperty("role", "paramName")
            theme.set_property(label, "active", name == active)
            add(label, row, 0)
            if name == active:
                field = self._active_field
                if field is not None:
                    field.show()
                    self._param_grid.addWidget(field, row, 1)
                else:
                    field = QLabel("(set on the plane)")
                    field.setProperty("role", "muted")
                    add(field, row, 1)
            else:
                field = self._param_fields[name]
                add(field, row, 1)
            radio = QRadioButton("Parameter plane")
            radio.setProperty("paramName", name)
            radio.setChecked(name == active)
            radio.setToolTip(f"Sweep {name} across the parameter plane")
            self._plane_group.addButton(radio)
            add(radio, row, 2)
        self._param_grid.setColumnStretch(1, 1)

    # ---- external resync: session.map was replaced wholesale (e.g. a family load) ------
    def refresh_from_session(self) -> None:
        """Called by SandboxWindow when something OUTSIDE this panel
        replaced self.session.map wholesale (loading a family from the
        Library tab) -- same "history spanning across an unrelated map
        isn't meaningful undo territory" reasoning app.term_editor_panel's
        own refresh_from_session had.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh_all()

    # ---- refresh: rebuild the formula field + parameter UI from session.map ------------
    def _refresh_all(self) -> None:
        self._formula_edit.blockSignals(True)
        self._formula_edit.setText(self.session.map.to_formula())
        self._formula_edit.blockSignals(False)
        self._live_preview.set_now(self.session.map.to_formula())
        self._fit_preview()
        theme.set_invalid(self._formula_edit, False)

        if self.session.map.is_pq_backed():
            # Seed from the map's OWN recorded selections, not a fresh
            # parse -- these are the actual values from_expression was
            # built with, which a re-parse of to_formula()'s (possibly
            # multi-parameter) text could not recover on its own (nothing
            # to tell it WHICH parameter was chosen active).
            active = self.session.map.pq_active_param
            fixed = dict(self.session.map.pq_fixed_params)
            params = sorted(set(fixed) | ({active} if active else set()))
        else:
            # A term-based map (a just-loaded built-in preset, never yet
            # edited through this panel) has no CanonicalRational to seed
            # from -- re-derive the parameter list from its own formula
            # text instead. Every built-in preset parses cleanly this way
            # (to_formula() is a faithful reconstruction, and each has at
            # most one parameter -- from_expression's own auto-active rule
            # applies without needing a fixed-value entry for anything).
            try:
                params = cdx.parse_rational_parameters(self.session.map.to_formula())
            except ValueError:
                params = []
            active = params[0] if len(params) == 1 else ""
            fixed = {}

        self._current_params = params
        self._rebuild_param_ui(active, fixed)
        self._error_label.setText("")
        try:
            self._set_warnings(list(cdx.parse_rational_info(self.session.map.to_formula())["warnings"]))
        except ValueError:
            self._set_warnings([])
        self._undo_button.setEnabled(bool(self._undo_stack))
        self._redo_button.setEnabled(bool(self._redo_stack))

    def _rebuild_param_ui(self, active: str, fixed: dict[str, complex]) -> None:
        self._active_combo.blockSignals(True)
        self._active_combo.clear()
        if self._current_params:
            self._active_combo.addItems(self._current_params)
            idx = self._active_combo.findText(active) if active else -1
            self._active_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._active_combo.blockSignals(False)
        self._rebuild_fixed_param_fields(fixed)

    def _rebuild_fixed_param_fields(self, fixed: dict[str, complex]) -> None:
        for field in self._param_fields.values():
            if field in self._table_widgets:
                self._table_widgets.remove(field)
            self._param_grid.removeWidget(field)
            field.deleteLater()
        self._param_fields = {}
        active = self._active_combo.currentText() if self._current_params else ""
        for name in self._current_params:
            if name == active:
                continue
            field = ComplexField("", fixed.get(name, 0j))
            field.setToolTip(f"The fixed value of {name}")
            self._param_fields[name] = field
        self._rebuild_param_table()

    # ---- live validation: syntax + parameter list, NOT applied to session.map ----------
    def _set_warnings(self, warnings: list[str]) -> None:
        self._warning_label.setText("\n".join(f"Warning: {w}" for w in warnings))
        self._warning_label.setVisible(bool(warnings))

    def warning_text(self) -> str:
        """The parser warnings currently shown (empty when none)."""
        return self._warning_label.text() if self._warning_label.isVisibleTo(self) else ""

    def _on_formula_text_changed(self, text: str) -> None:
        self._live_preview.text_changed(text)
        try:
            info = cdx.parse_rational_info(text)
        except ValueError as e:
            theme.set_invalid(self._formula_edit, True)
            self._error_label.setText(str(e))
            self._set_warnings([])
            return
        params = list(info["parameters"])
        self._set_warnings(list(info["warnings"]))
        theme.set_invalid(self._formula_edit, False)
        self._error_label.setText("")
        if params == self._current_params:
            return   # no parameter-list change -- nothing in the UI needs to move
        current_values = {name: f.value for name, f in self._param_fields.items()}
        current_active = self._active_combo.currentText()
        self._current_params = params
        active = current_active if current_active in params else (
            params[0] if len(params) == 1 else "")
        self._rebuild_param_ui(active, current_values)

    def _on_active_param_changed(self, _text: str) -> None:
        # Preserve whatever values are already sitting in the fixed-value
        # fields (for names that stay non-active) across the rebuild --
        # switching which parameter is active shouldn't discard the
        # others' entered values.
        current_values = {name: f.value for name, f in self._param_fields.items()}
        self._rebuild_fixed_param_fields(current_values)

    # ---- Apply: build the P/Q map, replacing session.map on success only ---------------
    def _apply(self) -> None:
        text = self._formula_edit.text()
        active = self._active_combo.currentText() if self._current_params else ""
        fixed = {name: f.value for name, f in self._param_fields.items()}
        snapshot = self.session.map.serialize()
        try:
            self.session.build_pq_map(text, active, fixed)
        except ValueError as e:
            self._error_label.setText(str(e))
            return
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()
        self._notify_edit()

    def _notify_edit(self) -> None:
        self._refresh_all()
        self._on_edit()

    # ---- pole / zero addition -----------------------------------------------------------
    def _add_pole(self) -> None:
        snapshot = self.session.map.serialize()
        try:
            self.session.add_pq_pole(self._pole_field.value)
        except (ValueError, RuntimeError) as e:
            self._error_label.setText(str(e))
            return
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()
        self._notify_edit()

    def _add_zero(self) -> None:
        snapshot = self.session.map.serialize()
        try:
            self.session.add_pq_zero(self._zero_field.value)
        except (ValueError, RuntimeError) as e:
            self._error_label.setText(str(e))
            return
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()
        self._notify_edit()

    # ---- undo / redo -----------------------------------------------------------------------
    def _restore_snapshot(self, text: str) -> None:
        # Not reachable in practice (every snapshot was produced by this
        # same map's own serialize() moments earlier), but caught rather
        # than assumed, same as app.term_editor_panel's own _restore_snapshot.
        try:
            self.session.map = cdx.RationalMap.deserialize(text)
        except ValueError as e:
            self._error_label.setText(f"could not restore undo snapshot: {e}")
            return
        self._notify_edit()

    def undo(self) -> None:
        """Edit > Undo (when no text field has focus): same as the button."""
        self._undo()

    def redo(self) -> None:
        """Edit > Redo: same as the button."""
        self._redo()

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
