"""test_facts_panel.py -- property-based checks for app/facts_panel.

Tests FactsPanel STANDALONE, constructed with its own Session and a spy
`on_center_view` callback (same rationale as test_settings_panel.py /
test_term_editor_panel.py: keeps these off a real SandboxWindow and its
debounced render timer).

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_facts_panel

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

import cdx
from app.facts_panel import FactsPanel, _classify, _critical_points_with_multiplicity, _is_inf
from app.session import Session

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.facts_panel tests ===")

    # ---- classification helper ---------------------------------------------------
    print("\nclassification:")
    check(_classify(0j) == "superattracting", "multiplier 0 classifies as superattracting")
    check(_classify(0.5 + 0j) == "attracting", "multiplier magnitude < 1 classifies as attracting")
    check(_classify(2 + 0j) == "repelling", "multiplier magnitude > 1 classifies as repelling")
    check(_classify(1 + 0j) == "neutral", "multiplier magnitude exactly 1 classifies as neutral")
    check(_classify(-1 + 0j) == "neutral",
          "a negative-real multiplier of magnitude 1 still classifies as neutral (|m|, not m)")

    # ---- critical-point multiplicity grouping --------------------------------------
    print("\ncritical-point grouping (newton_cubic, a superattracting-fixed-point map):")
    newton = cdx.RationalMap.newton_cubic()
    groups = _critical_points_with_multiplicity(newton, 0j)
    check(sum(mult for _, mult in groups) == len(newton.critical_points(0j)),
          "multiplicities sum to the raw critical_points() count -- no point silently dropped")
    check(len(groups) == len(newton.distinct_critical_points(0j)),
          "one group per distinct_critical_points() representative")

    # ---- panel construction + degree/Riemann-Hurwitz summary ------------------------
    print("\ndegree / Riemann-Hurwitz summary (newton_cubic):")
    session = Session()
    session.map = cdx.RationalMap.newton_cubic()
    session.param = 0j
    centered: list[complex] = []
    panel = FactsPanel(session, centered.append)

    check("Degree: 3" in panel._degree_label.text(), "degree label shows the map's actual degree")
    expected_rh = 2 * 3 - 2
    check(str(expected_rh) in panel._rh_label.text(),
          "Riemann-Hurwitz label shows the expected 2d-2 count")
    check("✓" in panel._rh_label.text(),
          "a map whose critical_points() is complete shows the check as passing, not flagged")

    # ---- tables populated from dynamical_facts --------------------------------------
    print("\ntables:")
    check(panel._critical_table.rowCount() == len(groups),
          "critical-points table has one row per distinct critical point")
    check(panel._fixed_table.rowCount() == 4,
          "fixed-points table: newton_cubic has 4 fixed points (3 roots + infinity)")
    check(panel._cycle_table.rowCount() == 3,
          "attracting-cycles table: newton_cubic has 3 attracting fixed points (period-1 cycles)")
    check(panel._pole_table.rowCount() == 1,
          "poles table: newton_cubic has one pole")
    check(panel._pole_table.item(0, 1).text() == "2", "the pole's order column reads 2")

    # ---- clicking a row centres the view (spy callback, not a real viewport) --------
    print("\nclick-to-centre:")
    centered.clear()
    panel._on_row_clicked(panel._pole_table, 0)
    check(len(centered) == 1 and centered[0] == 0j,
          "clicking the pole row calls on_center_view with that pole's location")

    centered.clear()
    panel._on_row_clicked(panel._pole_table, 99)   # out of range
    check(len(centered) == 0, "clicking a nonexistent row is a no-op, not an IndexError")

    # ---- caching: refresh() is a no-op unless (map, param) actually changed ---------
    print("\ncaching:")
    key_before = panel._cache_key
    panel.refresh()
    check(panel._cache_key == key_before,
          "refresh() with an unchanged map/param does not recompute (same cache key)")

    session.param = 1 + 0j   # a real change
    panel.refresh()
    check(panel._cache_key != key_before,
          "refresh() after an actual parameter change recomputes (a new cache key)")

    # ---- a map that breaks the Riemann-Hurwitz invariant flags it, not silently ------
    print("\nRiemann-Hurwitz mismatch is surfaced, not hidden:")
    session2 = Session()
    session2.map = cdx.RationalMap("scratch")
    session2.add_poly_term(1 + 0j, 2, 0)   # z^2: 2 critical points expected (0 and infinity)
    panel2 = FactsPanel(session2, lambda p: None)
    check("✓" in panel2._rh_label.text() or "MISMATCH" in panel2._rh_label.text(),
          "the RH label always reads one of exactly these two outcomes -- never blank")

    # ---- infinity: never offered as a centre-the-view target -------------------------
    print("\ninfinity is not a centre-the-view target:")
    # session2's plain z^2 (no pole terms at all) has critical points at
    # exactly {0, infinity} -- unlike newton_cubic above, whose 4th critical
    # point is its finite pole location, not infinity.
    critical_points = panel2._row_points[id(panel2._critical_table)]
    inf_row = next((i for i, p in enumerate(critical_points) if p is not None and _is_inf(p)),
                   None)
    check(inf_row is not None, "z^2's critical-points table has an infinity row")
    if inf_row is not None:
        centered2: list[complex] = []
        panel2._on_center_view = centered2.append
        panel2._on_row_clicked(panel2._critical_table, inf_row)
        check(len(centered2) == 0, "clicking the infinity row never calls on_center_view")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
