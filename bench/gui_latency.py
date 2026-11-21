"""bench/gui_latency.py -- is a slow render blocking the GUI thread, or only slow?

Opens the real SandboxWindow offscreen, loads relaxed Newton (n = 5), puts a
pane in Parameter_basin (the slowest mode) and starts a render through the
window's own _start_render. While it runs, a 10 ms QTimer on the GUI thread
records the gaps between its own ticks: if the GUI thread were blocked by the
render, the gaps would be as long as the render. Reports the largest gap and
how long the preview and full stages took to arrive.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m bench.gui_latency
"""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def main():
    app = QApplication.instance() or QApplication([])
    from app.sandbox import SandboxWindow
    window = SandboxWindow()
    window.show()
    window.session.build_pq_map("z - a*(z^5-1)/(5*z^4)", "a", {}, "relaxed_newton5")
    pane = window.panes[0]
    pane.viewport.center, pane.viewport.scale, pane.viewport.resolution = 5 + 0j, 5.0, 200
    pane.set_render_mode("parameter_basin")
    for other in window.panes[1:]:                      # only one pane renders
        for task in other.pending_tasks.values():
            task.cancel.cancel()

    gaps, last = [], [time.perf_counter()]
    arrived = {}

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now
        if pane.request_id not in pane.pending_tasks and "full" not in arrived:
            arrived["full"] = now - t0
            QTimer.singleShot(200, app.quit)

    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(tick)
    t0 = time.perf_counter()
    window._start_render(pane)
    timer.start()
    QTimer.singleShot(180000, app.quit)
    app.exec()
    timer.stop()
    gaps.sort()
    print(f"render finished after {arrived.get('full', float('nan')):.2f} s "
          f"(buffer {round(200 * 1.3)}^2, Parameter_basin, relaxed Newton n=5)")
    print(f"GUI-thread timer (10 ms nominal): {len(gaps)} ticks, median gap {1e3 * gaps[len(gaps) // 2]:.1f} ms, "
          f"99th percentile {1e3 * gaps[int(0.99 * (len(gaps) - 1))]:.1f} ms, largest {1e3 * gaps[-1]:.1f} ms")
    window.close()


if __name__ == "__main__":
    main()
