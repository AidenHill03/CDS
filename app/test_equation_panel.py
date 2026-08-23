"""test_equation_panel.py -- property-based checks for app/equation_panel.

Tests EquationPanel STANDALONE, constructed with its own Session and a spy
`on_edit` callback, rather than through a real SandboxWindow (same rationale
as app/test_settings_panel.py: this is what keeps these tests off the real
~/.complexdynamics/settings.json and off a real debounced render timer).

Replaces app/test_term_editor_panel.py (retired along with the two-table
term/pole editor it tested): parse/validate, parameter-select, Apply, and
pole/zero-addition checks stand in for the old poly/pole-table ones.

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_equation_panel

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

import cdx
from app.equation_panel import EquationPanel, _INVALID_STYLE
from app.session import Session

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.equation_panel tests ===")

    # ---- initial state: a term-based preset (mandelbrot) shown/parsed on construction --
    print("\ninitial state (term-based preset):")
    session = Session()   # starts on cdx.RationalMap.mandelbrot(), term-based, one parameter
    edits: list[None] = []
    panel = EquationPanel(session, lambda: edits.append(None))

    check(panel._formula_edit.text() == session.map.to_formula(),
          "formula field is pre-filled from the map's own to_formula() on construction")
    check(panel._active_combo.count() == 1 and panel._active_combo.currentText() == "a",
          "mandelbrot's single parameter 'a' is auto-selected as active")
    check(len(panel._param_fields) == 0,
          "no fixed-value fields are shown -- the only parameter is the active one")
    check(not session.map.is_pq_backed(),
          "construction alone does not convert a term-based preset to P/Q backing")
    check(not panel._undo_button.isEnabled() and not panel._redo_button.isEnabled(),
          "undo/redo both start disabled")

    # ---- live validation: invalid syntax marks the field, does not touch session.map ---
    print("\nlive validation (invalid syntax):")
    before = session.map
    panel._formula_edit.setText("z^2 + sin(a)")   # transcendental -- rejected by the parser
    check(panel._formula_edit.styleSheet() == _INVALID_STYLE,
          "a transcendental function marks the formula field invalid")
    check(panel._error_label.text() != "", "the parser's own error message is shown")
    check(session.map is before, "invalid live input never touches session.map")

    panel._formula_edit.setText("z^2 +")   # syntactically incomplete
    check(panel._formula_edit.styleSheet() == _INVALID_STYLE,
          "incomplete syntax also marks the field invalid")
    check(session.map is before, "still untouched")

    # ---- live validation: valid syntax clears the error and updates the param UI -------
    print("\nlive validation (valid syntax, new parameter list):")
    panel._formula_edit.setText("z^2 + b/z")
    check(panel._formula_edit.styleSheet() == "", "valid syntax clears the invalid styling")
    check(panel._error_label.text() == "", "valid syntax clears the error label")
    check(panel._current_params == ["b"], "the parsed parameter list updates live")
    check(panel._active_combo.count() == 1 and panel._active_combo.currentText() == "b",
          "the new single parameter is auto-selected active")
    check(session.map is before, "still untouched -- live validation never applies")

    # ---- Apply: builds the P/Q map, replaces session.map only on success ---------------
    print("\nApply (single parameter, auto-active):")
    edits.clear()
    panel._apply()
    check(session.map is not before, "Apply replaces session.map")
    check(session.map.is_pq_backed(), "the new map is P/Q-backed")
    check(session.map.pq_active_param == "b", "the active parameter round-trips through Apply")
    check(len(edits) == 1, "a successful Apply notifies on_edit exactly once")
    check(panel._undo_button.isEnabled(), "undo becomes available after a successful Apply")
    z = 1.5 + 0.3j
    check(abs(session.map.eval(z, 2 + 0j) - (z * z + (2 + 0j) / z)) < 1e-9,
          "the applied map evaluates as the authored formula, with the active parameter bound")

    # ---- Apply: invalid formula leaves the last valid map in place ---------------------
    print("\nApply (invalid formula, non-destructive):")
    before2 = session.map
    edits.clear()
    panel._formula_edit.setText("z^2 + log(b)")
    panel._apply()
    check(session.map is before2, "Apply on invalid syntax leaves session.map untouched")
    check(panel._error_label.text() != "", "the failure is reported in the error label")
    check(len(edits) == 0, "a failed Apply never notifies on_edit")

    # ---- multi-parameter: active selection + fixed-value substitution ------------------
    print("\nmulti-parameter Apply (explicit active + fixed values):")
    panel._formula_edit.setText("z^2 + p*z + q")
    check(panel._current_params == ["p", "q"], "both parameters are parsed")
    check(panel._active_combo.count() == 2, "the active combo lists both parameters")
    idx = panel._active_combo.findText("q")
    panel._active_combo.setCurrentIndex(idx)
    check(list(panel._param_fields.keys()) == ["p"],
          "only the NON-active parameter ('p') gets a fixed-value field")
    panel._param_fields["p"].set_value(3 + 0j)
    edits.clear()
    panel._apply()
    check(session.map.is_pq_backed(), "multi-parameter Apply succeeds")
    check(session.map.pq_active_param == "q", "the explicitly chosen parameter is active")
    check(session.map.pq_fixed_params.get("p") == 3 + 0j,
          "the other parameter is substituted as a constant at its entered value")
    z, q = 0.4 - 0.2j, -1 + 0.5j
    check(abs(session.map.eval(z, q) - (z * z + 3 * z + q)) < 1e-9,
          "the built map matches the fully-substituted expression (p=3, active=q)")

    # ---- add pole / add zero: forward root -> factor on the current P/Q ----------------
    print("\nadd pole / add zero:")
    before_eval = session.map.eval(z, q)
    pole_loc = 2 + 0j
    panel._pole_field.set_value(pole_loc)
    edits.clear()
    panel._add_pole()
    check(len(edits) == 1, "a successful Add Pole notifies on_edit")
    after_pole_eval = session.map.eval(z, q)
    check(abs(after_pole_eval - before_eval / (z - pole_loc)) < 1e-9,
          "Add Pole divides the map's value by (z - location), matching add_pole_at's contract")
    check(session.map.to_formula() == panel._formula_edit.text(),
          "the formula field shows the map's updated authored text after Add Pole")

    before_zero_eval = session.map.eval(z, q)
    zero_loc = -1 + 0j
    panel._zero_field.set_value(zero_loc)
    edits.clear()
    panel._add_zero()
    check(len(edits) == 1, "a successful Add Zero notifies on_edit")
    after_zero_eval = session.map.eval(z, q)
    check(abs(after_zero_eval - before_zero_eval * (z - zero_loc)) < 1e-9,
          "Add Zero multiplies the map's value by (z - location), matching add_zero_at's contract")

    # ---- add pole / add zero from a term-based map: implicit conversion first ----------
    print("\nadd pole from a term-based map (implicit P/Q conversion):")
    session2 = Session()   # mandelbrot again, term-based
    panel2 = EquationPanel(session2, lambda: None)
    check(not session2.map.is_pq_backed(), "starts term-based")
    panel2._pole_field.set_value(5 + 0j)
    panel2._add_pole()
    check(session2.map.is_pq_backed(), "Add Pole on a term-based map converts it to P/Q first")
    z2, a2 = 0.1 + 0.1j, 0.7 + 0j
    expected = cdx.RationalMap.mandelbrot().eval(z2, a2) / (z2 - (5 + 0j))
    check(abs(session2.map.eval(z2, a2) - expected) < 1e-9,
          "the resulting map matches the original formula divided by the new pole factor")

    # ---- undo / redo ---------------------------------------------------------------------
    print("\nundo/redo:")
    session3 = Session()
    edits3: list[None] = []
    panel3 = EquationPanel(session3, lambda: edits3.append(None))
    panel3._formula_edit.setText("z^2 + c")
    panel3._apply()
    # .serialize() STRINGS, not the map object itself -- add_pole_at below
    # mutates session3.map IN PLACE (see RationalMap.add_pole_at's own doc
    # comment), so a bare `map_after_apply = session3.map` would just alias
    # the same underlying object and "move" along with it, silently making
    # this comparison vacuous.
    text_after_apply = session3.map.serialize()
    panel3._pole_field.set_value(1 + 0j)
    panel3._add_pole()
    text_after_pole = session3.map.serialize()
    check(panel3._undo_button.isEnabled(), "undo is available after two mutations")

    panel3._undo()
    check(session3.map.serialize() == text_after_apply,
          "undo restores the map from immediately before Add Pole")
    check(panel3._redo_button.isEnabled(), "redo becomes available after an undo")

    panel3._redo()
    check(session3.map.serialize() == text_after_pole,
          "redo restores the map from after Add Pole")

    panel3._undo()
    panel3._undo()
    check(not panel3._undo_button.isEnabled(),
          "undo exhausts back to the map from construction and then disables itself")

    # ---- refresh_from_session: external map replacement resyncs the panel, clears history --
    print("\nrefresh_from_session:")
    session3.map = cdx.RationalMap.newton_cubic()
    panel3.refresh_from_session()
    check(panel3._formula_edit.text() == session3.map.to_formula(),
          "refresh_from_session re-pulls the formula field from the (externally replaced) map")
    check(not panel3._undo_button.isEnabled() and not panel3._redo_button.isEnabled(),
          "refresh_from_session clears undo/redo history -- it spans an unrelated map now")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"{failures} check(s) FAILED")
    else:
        print("all checks passed")


if __name__ == "__main__":
    main()
    import sys
    sys.exit(1 if failures else 0)
