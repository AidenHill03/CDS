"""test_panel.py -- the Approximation tab, headless.

Run with (from the repository root):

    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m modules.approximation.test_panel
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

REF = os.path.join(os.path.dirname(__file__), "test_reference")
failures = 0


def check(cond, what):
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def settle(panel, timeout=300.0):
    """Process events until every job has finished and been delivered."""
    end = time.time() + timeout
    while time.time() < end:
        panel.wait(100)
        QApplication.processEvents()
        if not panel._jobs and not panel._render_timer.isActive():
            return True
    return False


def main():
    app = QApplication.instance() or QApplication([])
    from modules.approximation.panel import ApproximationPanel, make_panel
    print("=== modules.approximation panel tests ===")

    panel = ApproximationPanel()
    panel.resize(1100, 700)
    panel.show()
    check(not panel.build_button.isEnabled() and not panel.score_button.isEnabled(),
          "nothing to build or score before an image is loaded")

    print("\nbuild:")
    panel.load_image(os.path.join(REF, "sectors.png"))
    panel.size_spin.setValue(64)
    check(panel.build_button.isEnabled(), "Build is enabled once an image is loaded")
    panel.build()
    check(panel.cancel_button.isEnabled() and not panel.build_button.isEnabled(), "busy while building")
    check(settle(panel), "the build and the first basin render finish")
    check(panel.info is not None and panel.r is not None, "a map was built")
    dyn = panel.info["dynamics"]
    check(dyn["fatou_set_verified"], "its Fatou set is verified (report data)")
    check("Fatou set" in panel.report_text.toPlainText(), "the report tab shows the dynamical report")
    labels = getattr(panel, "_basin_labels", None)
    check(labels is not None and labels.shape == (panel.BASIN_RESOLUTION,) * 2, "the engine rendered the basins")
    # Image order on screen: the top row is the top of the plane. Compare with the
    # construction's own grid (image order) at a point in the top half.
    info = panel.info
    center, s = panel._view
    n = labels.shape[0]
    z = complex(0.1, 0.5)
    row = int((center.imag + s - z.imag) / (2 * s) * n)
    col = int((z.real - (center.real - s)) / (2 * s) * n)
    lab, _ = panel.r.classify(np.array([z]), info["trap_centers"], info["trap_radii"], 60)
    check(int(labels[row, col]) == int(lab[0]), "displayed basin at z = 0.1+0.5i matches r's own classification")
    check(panel.score_button.isEnabled() and panel.save_report_button.isEnabled(), "Score and Export enabled")

    print("\nzoom:")
    old = panel._view
    canvas = panel.basin_canvas
    pos = QPointF(canvas.width() / 2, canvas.height() / 2)
    ev = QWheelEvent(pos, canvas.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas, ev)
    check(abs(panel._view[1] - old[1] / 1.25) < 1e-12, "wheel up zooms in by 1.25 about the cursor")
    check(settle(panel), "the zoomed view re-renders")
    panel._on_reset_view()
    check(settle(panel) and panel._view is not None and abs(panel._view[1] - old[1]) < 1e-12,
          "double-click reset returns to the default view")

    print("\nscore and export:")
    panel.resolutions_edit.setText("128, 256")
    panel.run_score()
    check(settle(panel) and panel.score_result is not None, "scoring finishes")
    text = panel.report_text.toPlainText()
    check("Hausdorff" in text and "Wada" in text and "Per basin" in text, "the report shows Hausdorff and Wada")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "summary.json")
        panel.save_report(path)
        data = json.loads(open(path).read())
        check("scoring" in data and data["fatou_set_verified"] is True, "summary JSON includes the scoring")
        files = panel.save_images(tmp)
        check(len(files) == 3 and all(os.path.exists(os.path.join(tmp, f)) for f in files),
              "target, basin and certified images saved")

    print("\ncancel:")
    panel.load_image(os.path.join(REF, "spiral.png"))
    panel.size_spin.setValue(128)
    panel.build()
    time.sleep(0.3)
    panel.cancel()
    check(settle(panel), "a cancelled build stops")
    check(panel.status_label.text() == "Cancelled." and panel.build_button.isEnabled(),
          "the panel reports the cancellation and is usable again")
    panel.shutdown()

    print("\ncomposition root:")
    from app.sandbox import SandboxWindow
    window = SandboxWindow()
    widget, title = make_panel(window)
    index = window.add_panel(widget, title)
    check(window.tabs.tabText(index) == "Approximation" and window.tabs.widget(index) is widget,
          "make_panel + add_panel put the tab in the window")
    import phoenix
    check(len(phoenix._module_panels()) == 1, "phoenix.py finds the Approximation panel")
    window.close()

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
