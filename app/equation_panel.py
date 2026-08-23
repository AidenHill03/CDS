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
references -- but session.map itself is untouched until "Apply" is
clicked. Unlike the old term editor's per-edit auto-apply (numeric field
edits are always complete/valid the instant they're typed), a formula is
usually SYNTACTICALLY INCOMPLETE for most of the time it's being typed
("z^2 + ", "z^2 + a" one character at a time), and building a full map
additionally needs the active-parameter/fixed-values selections below the
text field, not just valid syntax -- so this follows the Settings tab's
explicit-Apply pattern, not the old term editor's live-apply one.

PARAMETER UI. cdx.parse_rational_parameters(text) is the SAME parameter
list from_expression itself derives; this panel shows it as an "Active"
combo box (auto-selected when there is exactly one) plus one ComplexField
per OTHER parameter, for its fixed substitution value -- see cdx.
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

from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)

import cdx
from app.complex_field import ComplexField

_INVALID_STYLE = "QLineEdit { border: 1px solid #cc4444; background: #fff0f0; }"


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
        layout = QVBoxLayout(self)

        eq_row = QHBoxLayout()
        eq_row.addWidget(QLabel("f(z) ="))
        self._formula_edit = QLineEdit()
        self._formula_edit.textChanged.connect(self._on_formula_text_changed)
        eq_row.addWidget(self._formula_edit, 1)
        layout.addLayout(eq_row)

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #cc4444;")
        layout.addWidget(self._error_label)

        param_box = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_box)

        active_row = QHBoxLayout()
        active_row.addWidget(QLabel("Active:"))
        self._active_combo = QComboBox()
        self._active_combo.currentTextChanged.connect(self._on_active_param_changed)
        active_row.addWidget(self._active_combo)
        active_row.addStretch(1)
        param_layout.addLayout(active_row)

        # Rebuilt on every parameter-list/active-selection change (see
        # _rebuild_fixed_param_fields) -- one ComplexField per non-active
        # parameter, added directly to this layout rather than into a
        # separate nested container widget, since there is nothing else
        # sharing this space that would need to be told apart from it.
        self._fixed_params_layout = QVBoxLayout()
        param_layout.addLayout(self._fixed_params_layout)

        layout.addWidget(param_box)

        apply_row = QHBoxLayout()
        self._apply_button = QPushButton("Apply")
        self._apply_button.clicked.connect(self._apply)
        apply_row.addWidget(self._apply_button)
        apply_row.addStretch(1)
        layout.addLayout(apply_row)

        pz_box = QGroupBox("Add Pole / Zero (forward root -> factor on the current P/Q)")
        pz_layout = QVBoxLayout(pz_box)

        pole_row = QHBoxLayout()
        self._pole_field = ComplexField("Pole at:", 0j)
        pole_row.addWidget(self._pole_field)
        add_pole_btn = QPushButton("Add Pole")
        add_pole_btn.clicked.connect(self._add_pole)
        pole_row.addWidget(add_pole_btn)
        pole_row.addStretch(1)
        pz_layout.addLayout(pole_row)

        zero_row = QHBoxLayout()
        self._zero_field = ComplexField("Zero at:", 0j)
        zero_row.addWidget(self._zero_field)
        add_zero_btn = QPushButton("Add Zero")
        add_zero_btn.clicked.connect(self._add_zero)
        zero_row.addWidget(add_zero_btn)
        zero_row.addStretch(1)
        pz_layout.addLayout(zero_row)

        layout.addWidget(pz_box)

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
        self._formula_edit.setStyleSheet("")

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
        while self._fixed_params_layout.count():
            item = self._fixed_params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._param_fields = {}
        active = self._active_combo.currentText() if self._current_params else ""
        for name in self._current_params:
            if name == active:
                continue
            field = ComplexField(f"{name} =", fixed.get(name, 0j))
            self._param_fields[name] = field
            self._fixed_params_layout.addWidget(field)

    # ---- live validation: syntax + parameter list, NOT applied to session.map ----------
    def _on_formula_text_changed(self, text: str) -> None:
        try:
            params = cdx.parse_rational_parameters(text)
        except ValueError as e:
            self._formula_edit.setStyleSheet(_INVALID_STYLE)
            self._error_label.setText(str(e))
            return
        self._formula_edit.setStyleSheet("")
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
