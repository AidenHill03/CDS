"""test_library_panel.py -- property-based checks for app/library_panel.

Tests LibraryPanel STANDALONE, constructed with its own Session and spy
on_load/on_change callbacks (same rationale as test_settings_panel.py /
test_term_editor_panel.py / test_facts_panel.py: keeps these off a real
SandboxWindow, its render pipeline, and disk I/O -- this panel itself never
touches ~/.complexdynamics/library.txt, see its own module docstring).

Mutating actions that would pop a real QInputDialog (Save Current As,
Rename) are exercised through their `_do_*` methods directly rather than
by clicking the button -- see library_panel.py's own docstring for why.

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_library_panel

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

import cdx
from app.library_panel import LibraryPanel, default_view_for
from app.session import PRESET_FAMILY_NAMES, Session

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.library_panel tests ===")

    # ---- default-view lookup ------------------------------------------------------
    print("\ndefault view lookup:")
    center, scale = default_view_for("mandelbrot")
    check(center == complex(-0.5, 0.0) and scale == 1.5,
          "mandelbrot's default view matches the classic Mandelbrot framing")
    fallback_center, fallback_scale = default_view_for("some-user-family-not-in-the-table")
    check(fallback_center == 0j and fallback_scale > 0,
          "an unlisted name gets the generic fallback view, not a KeyError")

    # ---- initial state: list populated from session.library -----------------------
    print("\ninitial state:")
    session = Session()
    loads: list[None] = []
    changes: list[None] = []
    panel = LibraryPanel(session, lambda: loads.append(None), lambda: changes.append(None))

    check(panel._list.count() == len(session.library),
          "the list has one row per library entry (six presets, nothing saved yet)")
    check(panel._selected_name() is not None, "something is selected on construction")

    # ---- presets are read-only: every guarded action rejects a preset name --------
    print("\npresets are read-only:")
    panel._select_by_name("mandelbrot")
    check(panel._notes_edit.isReadOnly(), "a preset's notes editor is read-only")

    panel._do_save_as("mandelbrot")
    check(len(changes) == 0, "Save Current As over a preset name is rejected, on_change not called")
    check(panel._error_label.text() != "", "the rejection shows an inline error")

    panel._error_label.setText("")
    panel._do_rename("mandelbrot", "mandelbrot2")
    check(len(changes) == 0, "renaming a preset is rejected")
    check("mandelbrot" in session.library.names(), "the preset is still present under its own name")

    panel._error_label.setText("")
    panel._do_delete("mandelbrot")
    check(len(changes) == 0, "deleting a preset is rejected")
    check("mandelbrot" in session.library.names(), "the preset survives the rejected delete")

    # ---- save current map as a new entry -------------------------------------------
    print("\nsave current as:")
    session.map = cdx.RationalMap("scratch")
    session.add_poly_term(1 + 0j, 2, 0)
    changes.clear()
    panel._do_save_as("my-family")
    check("my-family" in session.library.names(), "the new entry is added to the library")
    check(session.library.find("my-family").to_formula() == session.map.to_formula(),
          "the saved entry's formula matches the map that was current at save time")
    check(len(changes) == 1, "a successful save calls on_change")
    check(panel._selected_name() == "my-family", "the list selects the newly-saved entry")

    # ---- notes: editable for a user entry, persisted through the library ----------
    print("\nnotes:")
    check(not panel._notes_edit.isReadOnly(), "a user entry's notes editor is NOT read-only")
    changes.clear()
    panel._notes_edit.setPlainText("a scratch quadratic")
    check(session.library.find("my-family").notes == "a scratch quadratic",
          "editing the notes box updates the library entry's notes field")
    check(len(changes) == 1, "a successful notes edit calls on_change")

    # ---- rename -----------------------------------------------------------------------
    print("\nrename:")
    changes.clear()
    panel._do_rename("my-family", "my-family-renamed")
    check("my-family-renamed" in session.library.names() and
         "my-family" not in session.library.names(),
          "rename moves the entry to the new name, removing the old one")
    check(session.library.find("my-family-renamed").notes == "a scratch quadratic",
          "the renamed entry keeps its notes")
    check(len(changes) == 1, "a successful rename calls on_change")
    check(panel._selected_name() == "my-family-renamed", "the list selects the renamed entry")

    changes.clear()
    panel._do_rename("my-family-renamed", "multibrot3")   # collides with a PRESET name
    check(len(changes) == 0, "renaming onto an existing preset's name is rejected")
    check("my-family-renamed" in session.library.names(),
          "the entry keeps its original name after the rejected rename")

    # ---- load: replaces session.map; on_load's own handler resets the viewport ------
    # LibraryPanel itself no longer touches any viewport (Session has none
    # of its own to touch -- see app.pane.Pane); that reset is on_load's
    # job now (app/sandbox.py's _on_family_loaded, covered in
    # app/test_sandbox.py against a real Pane) -- here, only what this
    # panel itself is actually responsible for.
    print("\nload:")
    loads.clear()
    panel._select_by_name("my-family-renamed")
    panel._load_selected()
    check(session.map.name == "my-family-renamed", "Load makes the selected entry the current map")
    check(len(loads) == 1, "a successful load calls on_load")

    # ---- delete -------------------------------------------------------------------------
    print("\ndelete:")
    changes.clear()
    panel._do_delete("my-family-renamed")
    check("my-family-renamed" not in session.library.names(), "delete removes the entry")
    check(len(changes) == 1, "a successful delete calls on_change")

    changes.clear()
    panel._do_delete("does-not-exist")
    check(len(changes) == 0, "deleting a nonexistent name is rejected, on_change not called")
    check(panel._error_label.text() != "", "the rejection shows an inline error")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
