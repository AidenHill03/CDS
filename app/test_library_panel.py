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

import math

from PySide6.QtWidgets import QApplication

import cdx
from app.library_panel import (LibraryPanel, _bounding_view, _DYNAMICAL_VIEW_FALLBACK,
                               _DYNAMICAL_VIEW_MIN_SCALE, _DYNAMICAL_VIEW_PADDING,
                               default_dynamical_view, default_view_for, default_view_for_mode)
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

    # ---- dynamical (Julia-set) default framing: mode-aware, Stage 2 ----------------
    # default_view_for's own table is a-SPACE (the parameter plane) --
    # default_dynamical_view frames z-space instead, and must NOT just
    # reuse the same table (that reuse was the Stage 2 bug: resetting a
    # Julia-set pane jumped it onto the parameter plane's own window).
    print("\ndynamical default framing (default_dynamical_view):")
    mandelbrot = cdx.RationalMap.mandelbrot()
    julia_param = complex(-0.7269, 0.1889)
    dyn_center, dyn_scale = default_dynamical_view(mandelbrot, julia_param)
    param_center, param_scale = default_view_for("mandelbrot")
    check((dyn_center, dyn_scale) != (param_center, param_scale),
          "the dynamical-plane default framing is NOT the same window as the "
          "parameter-plane's own table -- that reuse was the actual bug")
    check(dyn_scale > 0 and math.isfinite(dyn_center.real) and math.isfinite(dyn_center.imag),
          "the derived framing is a genuinely usable (finite center, positive scale) viewport")

    # newton_cubic(): `a` has NO effect on this map at all (confirmed
    # elsewhere, describe_parameter_role), but it still has real critical
    # and fixed points -- the dynamical framing must still derive something
    # usable from THOSE, independent of describe_parameter_role's own
    # "unused by this map" finding (a different, unrelated question).
    newton = cdx.RationalMap.newton_cubic()
    newton_center, newton_scale = default_dynamical_view(newton, 0j)
    check(newton_scale > 0 and math.isfinite(newton_center.real) and
          math.isfinite(newton_center.imag),
          "newton_cubic() -- whose `a` is unused -- still gets a real, finite dynamical "
          "default framing, derived from its critical/fixed points regardless")

    # ---- _bounding_view: the pure arithmetic, tested directly ----------------------
    # Coaxing a real RationalMap into producing a genuinely DEGENERATE
    # (all-excluded) critical/fixed-point set is impractical -- tested
    # directly against a plain point list instead (see _bounding_view's
    # own docstring for why it's factored out).
    print("\n_bounding_view (the padded-bounding-box arithmetic, pure):")
    check(_bounding_view([]) == _DYNAMICAL_VIEW_FALLBACK,
          "no points at all falls back cleanly")
    check(_bounding_view([complex(float("inf"), 0), complex(0, float("nan"))])
          == _DYNAMICAL_VIEW_FALLBACK,
          "only non-finite points (infinity, NaN) falls back the same way")
    check(_bounding_view([complex(1000, 0), complex(-1000, 0)]) == _DYNAMICAL_VIEW_FALLBACK,
          "points that are numerically finite but absurdly far out (beyond "
          "_DYNAMICAL_VIEW_MAGNITUDE_CAP) are excluded same as non-finite ones -- "
          "if that exclusion leaves nothing, it's a fallback too")

    box_center, box_scale = _bounding_view([complex(-1, 0), complex(1, 0)])
    check(box_center == 0j, "two points symmetric about the origin center on the origin")
    check(box_scale == max(1.0 * _DYNAMICAL_VIEW_PADDING, _DYNAMICAL_VIEW_MIN_SCALE),
          "the box is padded by exactly _DYNAMICAL_VIEW_PADDING over the tight half-width (1.0)")

    single_center, single_scale = _bounding_view([complex(3, 4)])
    check(single_center == complex(3, 4), "a single point centers exactly on itself")
    check(single_scale >= _DYNAMICAL_VIEW_MIN_SCALE,
          "a single point (zero tight half-width) still gets a USABLE window via the "
          "minimum-scale floor, not a zero-size/degenerate one")

    mixed_center, mixed_scale = _bounding_view([complex(0, 0), complex(1000, 1000)])
    check(mixed_center == 0j and mixed_scale == _DYNAMICAL_VIEW_MIN_SCALE,
          "an excluded far-out point does not drag the box out to include it -- with only "
          "the origin surviving the filter, this is exactly the single-point case above")

    # ---- mode-aware dispatch: default_view_for_mode routes on render_mode ----------
    print("\nmode-aware dispatch (default_view_for_mode):")
    check(default_view_for_mode(mandelbrot, julia_param, "parameter")
          == default_view_for("mandelbrot"),
          "a PARAMETER-plane mode routes to default_view_for's own a-space table, unchanged")
    check(default_view_for_mode(mandelbrot, julia_param, "parameter_greens")
          == default_view_for("mandelbrot"),
          "...the same for the other PARAMETER_PLANE_MODES entry")
    check(default_view_for_mode(mandelbrot, julia_param, "julia")
          == default_dynamical_view(mandelbrot, julia_param),
          "a DYNAMICAL mode routes to default_dynamical_view instead, using the actual "
          "map+param (not just the map's name)")

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
