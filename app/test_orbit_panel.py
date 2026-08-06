"""test_orbit_panel.py -- property-based checks for app/orbit_panel.

Tests OrbitPanel STANDALONE (its own Session + a real ImageView, since the
panel drives ImageView's own step_orbit/clear_orbit methods rather than
owning orbit state itself -- see its own module docstring), matching the
rest of this app's per-panel test files. app/test_orbit_tracker.py already
covers the underlying OrbitTracker's own state/classification logic in
depth, and app/test_sandbox.py's "orbit tracking" section already covers
ImageView's click-to-seed/staleness wiring -- this file's own job is the
one thing neither of those exercises: driving the actual Step/Run N/Clear
QPushButton/QSpinBox WIDGETS, not calling image_view's methods directly.

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_orbit_panel

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QPushButton

import cdx
from app.orbit_panel import OrbitPanel
from app.sandbox import ImageView
from app.session import Session

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def _button(panel: OrbitPanel, label: str) -> QPushButton:
    for child in panel.findChildren(QPushButton):
        if child.text() == label:
            return child
    raise AssertionError(f"no button labelled {label!r}")


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.orbit_panel tests ===")

    session = Session()
    session.map = cdx.RationalMap.mandelbrot()
    session.param = -1 + 0j   # the basilica
    session.set_render_mode("julia")
    view = ImageView(session)
    panel = OrbitPanel(session, view)

    check("Click the image" in panel._readout_label.text(),
          "the readout starts with the no-orbit prompt")

    view.orbit_tracker.seed(session.map, session.param, 0.1 + 0j)
    view.orbit_changed.emit()   # OrbitPanel only refreshes on the signal, not on direct mutation
    check("n = 0" in panel._readout_label.text(), "the readout updates once orbit_changed fires")

    # ---- Step button ------------------------------------------------------------------
    print("\nStep button:")
    _button(panel, "Step").click()
    check(view.orbit_tracker.state.n == 1, "clicking Step advances the orbit by exactly 1")
    check("n = 1" in panel._readout_label.text(), "the readout reflects the click")

    _button(panel, "Step").click()
    check(view.orbit_tracker.state.n == 2, "a second click advances it again")

    # ---- Run N button -------------------------------------------------------------------
    print("\nRun N button:")
    panel._run_n_spin.setValue(15)
    _button(panel, "Run N").click()
    check(view.orbit_tracker.state.n == 17, "Run N advances by exactly the spin box's value (15)")

    panel._run_n_spin.setValue(1)
    _button(panel, "Run N").click()
    check(view.orbit_tracker.state.n == 18, "Run N with N=1 behaves the same as a single Step")

    # ---- Clear button -------------------------------------------------------------------
    print("\nClear button:")
    _button(panel, "Clear").click()
    check(view.orbit_tracker.state is None, "clicking Clear removes the orbit")
    check("Click the image" in panel._readout_label.text(),
          "the readout reverts to the no-orbit prompt after Clear")

    # Step/Run N with no active orbit must not crash (OrbitTracker.step is
    # already a documented no-op with state=None; this confirms the BUTTON
    # path reaches that safely too, not just the underlying method).
    _button(panel, "Step").click()
    check(view.orbit_tracker.state is None, "clicking Step with no orbit active is a no-op")
    _button(panel, "Run N").click()
    check(view.orbit_tracker.state is None, "clicking Run N with no orbit active is also a no-op")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
