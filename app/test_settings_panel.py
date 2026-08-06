"""test_settings_panel.py -- property-based checks for app/settings_panel.

Tests SettingsPanel STANDALONE, constructed with its own Session and a spy
`on_apply` callback, rather than through a real SandboxWindow -- this is
what keeps these tests from ever touching the real
~/.complexdynamics/settings.json (see app/test_sandbox.py's own note on the
same concern for its SandboxWindow-level tests): SettingsPanel itself never
imports app.settings' load_settings/save_settings at all, only the
generic FIELD_SPECS/validate_field/Settings/slow_render_warning helpers.

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_settings_panel

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QComboBox

from app.session import Session
from app.settings import FIELD_SPECS, Settings
from app.settings_panel import SettingsPanel

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.settings_panel tests ===")

    # ---- initial widget values match session.settings -----------------------------
    print("\ninitial values:")
    session = Session(settings=Settings(resolution=500, max_iter=300, escape_radius=3.0,
                                        tol=1e-5, threads=2, cache_budget_bytes=64 * 1024 * 1024))
    applied: list[Settings] = []
    panel = SettingsPanel(session, applied.append)

    check(panel._widgets["resolution"].value() == 500, "resolution widget starts at session.settings")
    check(panel._widgets["max_iter"].value() == 300, "max_iter widget starts at session.settings")
    check(abs(panel._widgets["escape_radius"].value() - 3.0) < 1e-9,
          "escape_radius widget starts at session.settings")
    check(panel._widgets["threads"].value() == 2, "threads widget starts at session.settings")
    check(panel._widgets["cache_budget_bytes"].value() == 64,
          "cache budget widget shows MB, not raw bytes (64 MB, not 67108864)")

    # ---- Apply: all-valid -----------------------------------------------------------
    print("\napply (all valid):")
    panel._widgets["resolution"].setValue(600)
    panel._widgets["threads"].setValue(3)
    panel._apply()

    check(len(applied) == 1, "a valid Apply calls on_apply exactly once")
    check(applied[-1].resolution == 600 and applied[-1].threads == 3,
          "the Settings passed to on_apply reflects the edited widget values")
    check(panel._error_label.text() == "", "no error text after a valid Apply")

    # ---- Apply: one invalid field reverts ALL fields, applies nothing -------------
    print("\napply (one field invalid):")
    applied.clear()
    panel._widgets["resolution"].setValue(700)      # a valid change, on its own
    # QDoubleSpinBox's own range floor is the FieldSpec's `minimum` even
    # when that minimum is EXCLUSIVE (see _spinbox_for's comment) -- so
    # escape_radius=0.0 is a value the WIDGET happily accepts but
    # validate_field rejects (must be > 0, not >= 0), the one invalid
    # case genuinely reachable by driving the widgets themselves rather
    # than constructing a bad Settings directly.
    panel._widgets["escape_radius"].setValue(0.0)
    panel._apply()

    check(len(applied) == 0, "an invalid field means on_apply is NOT called at all")
    check(panel._error_label.text() != "", "an error message is shown")
    check("Escape radius" in panel._error_label.text(),
          "the error message names the actual offending field")
    check(abs(panel._widgets["escape_radius"].value() - 3.0) < 1e-9,
          "the invalid field reverts to the LAST GOOD value (3.0), not left at the bad input")
    check(panel._widgets["resolution"].value() == 600,
          "a field that WAS valid (resolution=700) also reverts -- Apply is all-or-nothing, "
          "so it goes back to 600 (the last successful Apply), not the rejected 700")

    # ---- colouring fields: QComboBox for choices, generic apply round-trip ------------
    print("\ncolouring fields (choices widgets):")
    palette_widget = panel._widgets["colour_palette"]
    scaling_widget = panel._widgets["colour_scaling"]
    check(isinstance(palette_widget, QComboBox), "colour_palette gets a QComboBox, not a spinbox")
    check(isinstance(scaling_widget, QComboBox), "colour_scaling gets a QComboBox, not a spinbox")
    check([palette_widget.itemText(i) for i in range(palette_widget.count())]
          == list(FIELD_SPECS["colour_palette"].choices),
          "the combo box's items match FIELD_SPECS' choices, in order")
    check(palette_widget.currentText() == session.settings.colour_palette,
          "the combo box starts on session.settings' current palette")

    applied.clear()
    palette_widget.setCurrentText("viridis")
    scaling_widget.setCurrentText("histogram")
    panel._widgets["colour_period"].setValue(15.0)
    panel._apply()
    check(len(applied) == 1, "changing only combo-box/period fields still triggers a valid Apply")
    check(applied[-1].colour_palette == "viridis" and applied[-1].colour_scaling == "histogram"
         and applied[-1].colour_period == 15.0,
          "the applied Settings reflects the combo-box selections and the period spinbox")

    # ---- Clear Cache ------------------------------------------------------------------
    print("\nclear cache:")
    session.cache.put(("k",), __import__("numpy").zeros((4, 4)))
    check(session.cache.stats.entry_count == 1, "sanity: cache has one entry before clearing")
    panel._clear_cache()
    check(session.cache.stats.entry_count == 0, "Clear Cache empties the session's cache")

    # ---- cache readout label reflects live stats -----------------------------------
    print("\ncache readout:")
    session.cache.get(("nonexistent",))   # one miss
    panel._refresh_cache_readout()
    check("miss" in panel._cache_readout_label.text(), "the readout label mentions misses")
    check(str(session.cache.stats.misses) in panel._cache_readout_label.text(),
          "the readout label's miss count matches the cache's actual stats")

    # ---- slow-render hint updates as widgets change ---------------------------------
    print("\nslow-render hint:")
    panel._widgets["resolution"].setValue(FIELD_SPECS["resolution"].minimum)
    panel._widgets["max_iter"].setValue(10)
    panel._update_slow_render_hint()
    check(panel._slow_render_label.text() == "",
          "a small resolution/iteration combination shows no warning")

    panel._widgets["resolution"].setValue(FIELD_SPECS["resolution"].maximum)
    panel._widgets["max_iter"].setValue(FIELD_SPECS["max_iter"].maximum)
    panel._update_slow_render_hint()
    check(panel._slow_render_label.text() != "",
          "a large resolution/iteration combination shows a warning, live -- before Apply, "
          "not just after")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
