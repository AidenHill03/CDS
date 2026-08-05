"""test_term_editor_panel.py -- property-based checks for app/term_editor_panel.

Tests TermEditorPanel STANDALONE, constructed with its own Session and a spy
`on_edit` callback, rather than through a real SandboxWindow (same rationale
as app/test_settings_panel.py: this is what keeps these tests off the real
~/.complexdynamics/settings.json and off a real debounced render timer).

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_term_editor_panel

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import cdx
from app.session import Session
from app.term_editor_panel import POLE_LOC, POLY_COEFF, TermEditorPanel

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.term_editor_panel tests ===")

    # ---- initial state: tables reflect session.map -----------------------------
    print("\ninitial state:")
    session = Session()
    session.map = cdx.RationalMap("scratch")
    session.add_poly_term(2 + 0j, 3, 0, "cubic")
    edits: list[None] = []
    panel = TermEditorPanel(session, lambda: edits.append(None))

    check(panel._poly_table.rowCount() == 1, "poly table starts with one row per existing term")
    check(panel._pole_table.rowCount() == 0, "pole table starts empty")
    check("z^3" in panel._formula_label.text() or "z^ 3" in panel._formula_label.text(),
          "formula label shows the map's to_formula() on construction")
    check(not panel._undo_button.isEnabled(), "undo starts disabled (nothing to undo yet)")
    check(not panel._redo_button.isEnabled(), "redo starts disabled")

    # ---- add / remove poly term --------------------------------------------------
    print("\nadd/remove poly term:")
    edits.clear()
    panel._add_poly_term()
    check(len(session.map.poly_terms()) == 2, "Add Term appends a new poly term to the map")
    check(panel._poly_table.rowCount() == 2, "poly table gains a row")
    check(len(edits) == 1, "a successful add calls on_edit exactly once")
    check(panel._undo_button.isEnabled(), "undo becomes enabled after a real edit")

    panel._poly_table.selectRow(1)
    edits.clear()
    panel._remove_selected_poly_term()
    check(len(session.map.poly_terms()) == 1, "Remove Selected removes the selected poly term")
    check(len(edits) == 1, "a successful remove calls on_edit")

    # ---- add / remove pole term, including the fallback-location search -----------
    print("\nadd/remove pole term:")
    edits.clear()
    panel._add_pole_term()
    check(len(session.map.pole_terms()) == 1, "Add Pole appends a new pole term")
    check(session.map.pole_terms()[0].location == 0j,
          "Add Pole tries the origin first when it's free")
    check(len(edits) == 1, "a successful pole add calls on_edit")

    panel._add_pole_term()
    check(len(session.map.pole_terms()) == 2, "a second Add Pole succeeds")
    check(session.map.pole_terms()[1].location == 1 + 0j,
          "Add Pole falls back to the next default location once the origin is taken")

    panel._pole_table.selectRow(1)
    edits.clear()
    panel._remove_selected_pole_term()
    check(len(session.map.pole_terms()) == 1, "Remove Selected removes the selected pole term")

    # ---- reorder: no phantom undo entry at a boundary ------------------------------
    print("\nreorder:")
    panel._add_poly_term()
    panel._add_poly_term()
    check(len(session.map.poly_terms()) == 3, "sanity: three poly terms for the reorder checks")
    undo_depth_before = len(panel._undo_stack)
    panel._poly_table.selectRow(0)
    panel._move_selected_poly_term(-1)   # already at index 0: a no-op
    check(len(panel._undo_stack) == undo_depth_before,
          "Move Up at row 0 is a no-op and pushes no undo entry (the phantom-undo bug this "
          "guards against)")

    panel._poly_table.selectRow(0)
    panel._move_selected_poly_term(1)
    check(len(panel._undo_stack) == undo_depth_before + 1,
          "Move Down at row 0 (a real move) pushes exactly one undo entry")
    check(panel._poly_table.currentRow() == 1, "the moved row's selection follows it")

    # ---- validation: rejected mutation leaves the map and undo stack untouched -----
    print("\nvalidation:")
    session2 = Session()
    session2.map = cdx.RationalMap("scratch2")
    session2.add_pole_term(1 + 0j, 1 + 0j, 1)
    edits2: list[None] = []
    panel2 = TermEditorPanel(session2, lambda: edits2.append(None))
    undo_depth = len(panel2._undo_stack)

    # setText() alone fires cellChanged (the table is only signal-blocked
    # DURING _refresh_pole_table's own rebuild, not on ordinary edits) --
    # so driving the cell through setText() is the real path, and calling
    # the handler again afterwards would double-invoke it against the
    # already-refreshed (and by then valid-looking) cell text.
    panel2._pole_table.item(0, POLE_LOC).setText("garbage")
    check(session2.map.pole_terms()[0].location == 1 + 0j,
          "an unparseable location leaves the term's actual location untouched")
    check(panel2._error_label.text() != "", "an unparseable location shows an inline error")
    check(len(panel2._undo_stack) == undo_depth, "a rejected edit pushes no undo entry")

    # add_pole_term's OWN uniqueness rejection (the thing the ADD path is
    # required to enforce) is already covered end-to-end by _add_pole_term's
    # fallback-search test above and by app/test_session.py's dedicated
    # validation section. edit_pole_term (moving an EXISTING pole's location
    # onto another existing pole's) is a documented, deliberate scope
    # boundary from that same milestone -- only the ADD path is gated, per
    # the task's literal "reject a second pole" wording -- so this is
    # expected to succeed silently, not raise; asserting otherwise would be
    # testing behavior nobody built.
    session2.add_pole_term(5 + 0j, 1 + 0j, 1)
    panel2._refresh_pole_table()
    edits2.clear()
    panel2._pole_table.item(1, POLE_LOC).setText("1+0j")
    check(session2.map.pole_terms()[1].location == 1 + 0j,
          "edit_pole_term's location field is not gated by the add-time uniqueness check "
          "(documented scope boundary, not a bug)")
    check(len(edits2) == 1, "an accepted edit calls on_edit")

    # ---- field edits: enabled checkbox, coefficient text ----------------------------
    print("\nfield edits:")
    session3 = Session()
    session3.map = cdx.RationalMap("scratch3")
    session3.add_poly_term(1 + 0j, 2, 0, "term")
    edits3: list[None] = []
    panel3 = TermEditorPanel(session3, lambda: edits3.append(None))

    panel3._on_poly_enabled_changed(0, Qt.CheckState.Unchecked.value)
    check(session3.map.poly_terms()[0].enabled is False,
          "unchecking the On checkbox disables the term")
    check(len(edits3) == 1, "toggling enabled calls on_edit")

    panel3._poly_table.item(0, POLY_COEFF).setText("3+4j")
    check(session3.map.poly_terms()[0].coeff == 3 + 4j,
          "editing the coefficient cell's text updates the term's coefficient")

    # ---- undo / redo round-trip ------------------------------------------------------
    print("\nundo/redo:")
    session4 = Session()
    session4.map = cdx.RationalMap("scratch4")
    session4.add_poly_term(1 + 0j, 2, 0, "only")
    panel4 = TermEditorPanel(session4, lambda: None)

    panel4._add_poly_term()
    check(len(session4.map.poly_terms()) == 2, "sanity: two terms before undo")
    panel4._undo()
    check(len(session4.map.poly_terms()) == 1, "Undo restores the map to before the last edit")
    check(panel4._redo_button.isEnabled(), "redo becomes available after an undo")

    panel4._redo()
    check(len(session4.map.poly_terms()) == 2, "Redo re-applies the undone edit")

    panel4._undo()
    edits4: list[None] = []
    panel4._on_edit = lambda: edits4.append(None)
    panel4._add_pole_term()
    check(not panel4._redo_button.isEnabled(),
          "a fresh edit after an undo clears the redo stack, not left stale")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
