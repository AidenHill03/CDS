"""test_library_panel.py -- property-based checks for app/library_panel.

Tests LibraryPanel STANDALONE, constructed with its own Session and spy
on_load/on_change callbacks (same rationale as test_settings_panel.py /
test_equation_panel.py / test_facts_panel.py: keeps these off a real
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
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

import app.settings as settings_module
import cdx
from app.library_panel import (LibraryPanel, _bounding_view, _derive_dynamical_view_from_facts,
                               _DYNAMICAL_VIEW_FALLBACK, _DYNAMICAL_VIEW_MIN_SCALE,
                               _DYNAMICAL_VIEW_PADDING, _LIST_ICON_SIZE, _NAME_ROLE,
                               _placeholder_icon, default_dynamical_view, default_view_for,
                               default_view_for_mode)
from app.family_views import (PRESET_PARAMETER_VIEWS, parameter_view_for, rename_family_meta,
                               save_parameter_view, stored_parameter_view)
from app.session import (DEFAULT_PARAMETER_VIEW_CENTER, DEFAULT_PARAMETER_VIEW_SCALE,
                         PRESET_FAMILY_NAMES, Session)
from app.settings import family_meta_path_for, preview_path_for

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    # LibraryPanel's _icon_for reads a preview sidecar path (see
    # app.settings.preview_path_for) on every list refresh -- rooted at
    # config_dir() (~/.complexdynamics by default), which would otherwise
    # create a real directory on the machine running this test just by
    # constructing a LibraryPanel. Redirected to a throwaway temp dir for
    # the whole module, same fix as app/test_sandbox.py uses for
    # library_path/load_settings/save_settings.
    _fake_config_dir = Path(tempfile.mkdtemp(suffix="-config"))
    settings_module.config_dir = lambda: _fake_config_dir

    print("=== app.library_panel tests ===")

    # ---- default-view lookup: each family's stored parameter-plane framing ---------
    # (app.family_views). The presets use the thumbnail table: the earlier
    # hand-picked table opened McMullen at +-2.5, where its structure is a
    # speck, and gave unlisted families an arbitrary (0, 2).
    print("\ndefault view lookup (stored per-family parameter framing):")
    expected_presets = {"mandelbrot": (complex(-0.75, 0), 1.5), "multibrot3": (0j, 1.35),
                        "multibrot5": (0j, 1.25), "mcmullen2": (0j, 0.3), "mcmullen3": (0j, 0.3)}
    check(all(default_view_for(n) == v for n, v in expected_presets.items()),
          "each preset opens at its table framing (McMullen at +-0.3, not a speck at +-2.5)")
    fallback = (DEFAULT_PARAMETER_VIEW_CENTER, DEFAULT_PARAMETER_VIEW_SCALE)
    check(default_view_for("some-user-family-not-in-the-table") == fallback,
          "a family with no stored view falls back to DEFAULT_PARAMETER_VIEW_CENTER/SCALE")
    check(default_view_for("newton3") == fallback,
          "...as does newton3 (a has no effect on it; no table entry)")

    # A saved family's view lives in a JSON sidecar beside its preview PNG.
    save_parameter_view("viewed-family", complex(0.1, -0.2), 0.05)
    check(stored_parameter_view("viewed-family") == (complex(0.1, -0.2), 0.05)
          and family_meta_path_for("viewed-family").exists(),
          "save_parameter_view stores a saved family's framing in its sidecar")
    rename_family_meta("viewed-family", "viewed-family-2")
    check(parameter_view_for("viewed-family-2") == (complex(0.1, -0.2), 0.05)
          and stored_parameter_view("viewed-family") is None,
          "rename moves the stored framing with the family")
    family_meta_path_for("garbled-family").write_text("{not json")
    check(parameter_view_for("garbled-family") == fallback,
          "an unreadable sidecar reads as 'nothing stored', never an exception")
    save_parameter_view("mandelbrot", 5j, 9.0)
    check(parameter_view_for("mandelbrot") == PRESET_PARAMETER_VIEWS["mandelbrot"],
          "a preset's framing is fixed: saving one for it is ignored")

    # ---- dynamical (Julia-set) default framing: FIXED, interim -----------------------
    # default_view_for's own table is a-SPACE (the parameter plane) --
    # default_dynamical_view frames z-space instead, and must NOT just
    # reuse the same table (that reuse was the Stage 2 bug: resetting a
    # Julia-set pane jumped it onto the parameter plane's own window).
    # INTERIM (this batch's Stage A): always the fixed fallback framing,
    # regardless of map/param -- per-instance derivation is deferred (see
    # _derive_dynamical_view_from_facts below, preserved but not called).
    print("\ndynamical default framing (default_dynamical_view, interim fixed value):")
    mandelbrot = cdx.RationalMap.mandelbrot()
    julia_param = complex(-0.7269, 0.1889)
    dyn_center, dyn_scale = default_dynamical_view(mandelbrot, julia_param)
    param_center, param_scale = default_view_for("mandelbrot")
    check((dyn_center, dyn_scale) != (param_center, param_scale),
          "the dynamical-plane default framing is NOT the same window as the "
          "parameter-plane's own table -- that reuse was the Stage 2 bug")
    check((dyn_center, dyn_scale) == _DYNAMICAL_VIEW_FALLBACK,
          "the dynamical framing is exactly the fixed fallback (center 0), not derived "
          "from mandelbrot's own critical/fixed points -- the interim direction")

    # newton_cubic(): `a` has NO effect on this map at all (confirmed
    # elsewhere, describe_parameter_role) -- a different, unrelated
    # question from what its dynamical default framing is, which (interim)
    # is the SAME fixed value regardless of map or param.
    newton = cdx.RationalMap.newton_cubic()
    newton_center, newton_scale = default_dynamical_view(newton, 0j)
    check((newton_center, newton_scale) == _DYNAMICAL_VIEW_FALLBACK,
          "newton_cubic() gets the SAME fixed dynamical framing as mandelbrot -- "
          "interim, center 0 regardless of map/param")

    # ---- _derive_dynamical_view_from_facts: preserved, NOT wired in -----------------
    # The per-instance (critical-/fixed-point-derived) logic default_
    # dynamical_view used to run directly -- still correct and callable,
    # just not reachable from the live reset path right now.
    print("\n_derive_dynamical_view_from_facts (deferred, preserved, still correct):")
    derived_center, derived_scale = _derive_dynamical_view_from_facts(mandelbrot, julia_param)
    check((derived_center, derived_scale) != _DYNAMICAL_VIEW_FALLBACK,
          "the DEFERRED function still genuinely derives a per-instance box (not just the "
          "fixed fallback) -- it's dormant, not broken")
    check(derived_scale > 0 and math.isfinite(derived_center.real) and
          math.isfinite(derived_center.imag),
          "...and the derived framing is still a genuinely usable viewport")

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

    check(len(panel.entry_names()) == len(session.library),
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

    # ---- preview thumbnails: list icons (Stage C) -----------------------------------
    # This panel never GENERATES a thumbnail itself (that's app.sandbox's
    # render pipeline, deliberately kept out of this standalone-widget test
    # -- see this module's own docstring) -- a raw PNG is written directly
    # to the sidecar path any real thumbnail would also land at, to
    # exercise _icon_for/refresh_previews's own READ-side logic in
    # isolation.
    print("\npreview thumbnails (list icons):")
    session.save_to_library("thumbed-family")
    session.save_to_library("bare-family")
    thumb_image = QImage(4, 4, QImage.Format.Format_RGB888)
    thumb_image.fill(0x336699)
    assert thumb_image.save(str(preview_path_for("thumbed-family")), "PNG")
    panel._refresh_list()

    placeholder_image = _placeholder_icon().pixmap(_LIST_ICON_SIZE, _LIST_ICON_SIZE).toImage()

    def icon_image(name: str):
        icon = panel.row_icon(name)
        if icon is None:
            raise AssertionError(f"{name!r} not found in the list")
        return icon.pixmap(_LIST_ICON_SIZE, _LIST_ICON_SIZE).toImage()

    check(icon_image("thumbed-family") != placeholder_image,
          "an entry WITH a sidecar preview shows that thumbnail as its list icon, not the "
          "placeholder")
    check(icon_image("bare-family") == placeholder_image,
          "an entry with NO sidecar preview yet falls back to the placeholder icon")
    check(icon_image("mandelbrot") == placeholder_image,
          "a preset (app.sandbox never generates one for a PRESET_FAMILY_NAMES entry) also "
          "falls back to the placeholder")

    # refresh_previews() re-reads icons WITHOUT rebuilding the list (unlike
    # _refresh_list, which also resets selection) -- the actual method
    # SandboxWindow._on_library_changed calls after it (re)writes sidecars.
    panel._select_by_name("bare-family")
    thumb_image.save(str(preview_path_for("bare-family")), "PNG")
    panel.refresh_previews()
    check(icon_image("bare-family") != placeholder_image,
          "refresh_previews() picks up a sidecar that appeared after the last _refresh_list, "
          "without a full list rebuild")
    check(panel._selected_name() == "bare-family",
          "...and does not disturb the current selection, unlike _refresh_list")

    # ---- double-click = Load (controls redesign, Step 3) -------------------------------
    print("\ndouble-click loads:")
    loads.clear()
    panel._select_by_name("mandelbrot")
    session.map = cdx.RationalMap("scratch")
    panel._list.itemDoubleClicked.emit(panel._list.currentItem())
    check(len(loads) == 1 and session.map.name == session.library.find("mandelbrot").name,
          "double-clicking a library entry loads it, exactly like the Load button")

    # ---- UI redesign: sections, search, loaded dot, row menus, confirm ------------------
    print("\nsidebar library (UI redesign):")
    from app.session import PRESET_FAMILY_NAMES
    names = panel.entry_names()
    presets = [n for n in names if n in PRESET_FAMILY_NAMES]
    check(names[:len(presets)] == presets and all(n not in PRESET_FAMILY_NAMES for n in names[len(presets):]),
          "two sections: Presets first, then Yours")
    headers = [panel._list.itemWidget(panel._list.item(r)).text()
               for r in range(panel._list.count()) if panel._list.item(r).data(_NAME_ROLE) is None]
    check(headers == ["PRESETS", "YOURS"], "the sections are headed PRESETS and YOURS")
    panel.search_field.setText("mcm")
    check(panel.visible_entry_names() == ["mcmullen2", "mcmullen3"], "search filters by name")
    panel.search_field.setText("z^5")
    check("multibrot5" in panel.visible_entry_names() and "mandelbrot" not in panel.visible_entry_names(),
          "search filters by formula too")
    panel.search_field.setText("")
    check(panel.visible_entry_names() == names, "clearing the search shows every entry again")
    check(panel.row_menu_texts("mandelbrot") == ["Load"],
          "a preset row's ⋯ menu offers Load only (no rename/delete)")
    check(panel.row_menu_texts("thumbed-family") == ["Load", "Save current map here", "Rename…", "Delete…"],
          "a user row's ⋯ menu offers Load, Save current map here, Rename, Delete")

    loads.clear()
    panel._select_by_name("multibrot3")
    panel._load_selected()
    check(panel.loaded_name() == "multibrot3" and panel._rows["multibrot3"].dot.text() == "●"
          and panel._rows["mandelbrot"].dot.text() == "", "the loaded dot follows a load")
    panel.set_loaded("mcmullen2")
    check(panel._selected_name() == "mcmullen2" and panel._rows["mcmullen2"].dot.text() == "●",
          "set_loaded moves the dot and selects that row")
    panel._select_by_name("newton3")
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent as _QEvent
    loads.clear()
    QApplication.sendEvent(panel._list, QKeyEvent(_QEvent.Type.KeyPress, Qt.Key.Key_Return,
                                                  Qt.KeyboardModifier.NoModifier))
    check(loads == [None] and session.map.name == "newton3", "Return loads the selected entry")

    asked: list[str] = []
    panel._confirm = lambda title, text: asked.append(text) or False
    panel._select_by_name("thumbed-family")
    panel._delete_selected()
    check(asked and "thumbed-family" in session.library.names(),
          "Delete asks first, and a No leaves the entry alone")
    panel._confirm = lambda title, text: True
    panel._delete_selected()
    check("thumbed-family" not in session.library.names(), "...and a Yes deletes it")

    # ---- legacy library file: no framing data anywhere -> defaults -------------------
    print("\nlegacy library file (written before per-family framings existed):")
    legacy_session = Session()
    legacy_session.map = cdx.RationalMap.from_expression("z^2 + a*z^3", "a", {}, "legacy-family")
    legacy_session.save_to_library("legacy-family")
    legacy_path = Path(tempfile.mkdtemp()) / "legacy-library.txt"
    legacy_session.save_library_file(str(legacy_path))
    check("parameter_view" not in legacy_path.read_text(),
          "sanity: the library file format itself carries no framing (it lives app-side)")
    fresh = Session()
    fresh.load_library_file(str(legacy_path))
    check("legacy-family" in fresh.library.names()
          and default_view_for("legacy-family") == fallback,
          "a library file with no view data loads, and its family opens at the defaults")

    # ---- callbacks: the owner persists the framing on save / rename -----------------
    saved_as: list[str] = []
    renamed: list[tuple[str, str]] = []
    cb_panel = LibraryPanel(Session(), lambda: None, lambda: None,
                            on_saved_as=saved_as.append,
                            on_renamed=lambda old, new: renamed.append((old, new)))
    cb_panel._do_save_as("cb-family")
    cb_panel._do_rename("cb-family", "cb-family-renamed")
    check(saved_as == ["cb-family"] and renamed == [("cb-family", "cb-family-renamed")],
          "Save current as and Rename tell the owner (it stores/moves the framing)")

    provided: list[str] = []
    thumbed = LibraryPanel(Session(), lambda: None, lambda: None,
                           thumbnail_provider=lambda name, entry: provided.append(name) or _placeholder_icon())
    check(sorted(provided) == sorted(PRESET_FAMILY_NAMES),
          "row thumbnails come from the thumbnail provider (one call per row; a fresh "
          "session's library is just the presets)")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
