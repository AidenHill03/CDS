"""test_sandbox.py -- property-based checks for app/sandbox.

Mirrors the rest of this project's test style (PASS/FAIL per check,
properties rather than eyeballing a window) applied to a GUI, using Qt's
offscreen platform so this runs without a real display and QTest.qWait to
pump the event loop while background renders complete.

Run with (note QT_QPA_PLATFORM=offscreen and, on this machine, an explicit
plugin path -- PySide6's wheel does not always end up on Qt's default
plugin search path):

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=../cdx/build python3 test_sandbox.py
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import cdx
from app.sandbox import ImageView, SandboxWindow
from app.session import Session

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def close(a: complex, b: complex, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def wait_for(predicate, timeout_ms: int = 5000, step_ms: int = 20) -> bool:
    """Pumps the event loop until predicate() is true or timeout_ms elapses.
    Needed because background-thread signals only get delivered while the
    event loop is running -- there is no app.exec() call in this script.
    """
    elapsed = 0
    while not predicate() and elapsed < timeout_ms:
        QTest.qWait(step_ms)
        elapsed += step_ms
    return predicate()


def main() -> None:
    global failures
    app = QApplication.instance() or QApplication([])

    print("=== app.sandbox tests ===")

    # ---- pixel <-> complex coordinate mapping -----------------------------------
    print("\npixel <-> complex mapping:")
    session = Session()
    session.viewport = cdx.Viewport(complex(0, 0), 1.5, 400)
    view = ImageView(session)
    view.resize(400, 400)

    center_pixel = QPoint(200, 200)
    center_complex = view._pixel_to_complex(center_pixel)
    check(close(center_complex, complex(0, 0), 1e-2),
          "the widget's center pixel maps to the viewport's center")

    top_left = view._pixel_to_complex(QPoint(0, 0))
    check(close(top_left, complex(-1.5, 1.5), 1e-2),
          "top-left pixel is (center.real-scale, center.imag+scale) -- "
          "screen top = largest imaginary part")

    bottom_right = view._pixel_to_complex(QPoint(400, 400))
    check(close(bottom_right, complex(1.5, -1.5), 1e-2),
          "bottom-right pixel is (center.real+scale, center.imag-scale)")

    # ---- cursor-anchored zoom formula --------------------------------------------
    print("\ncursor-anchored zoom:")
    session2 = Session()
    session2.viewport = cdx.Viewport(complex(0.2, -0.3), 1.0, 400)
    view2 = ImageView(session2)
    view2.resize(400, 400)

    cursor = QPoint(300, 120)   # off-center, deliberately
    w_before = view2._pixel_to_complex(cursor)

    vp = session2.viewport
    factor = 1.15 ** 3   # simulate 3 notches, matching wheelEvent's math directly
    new_center = w_before - (w_before - vp.center) / factor
    new_scale = vp.scale / factor
    session2.viewport = cdx.Viewport(new_center, new_scale, vp.resolution)

    w_after = view2._pixel_to_complex(cursor)
    check(close(w_before, w_after, 1e-9),
          "the complex point under the cursor is unchanged by a cursor-anchored zoom")
    check(session2.viewport.scale < 1.0, "zooming in (factor>1) shrinks scale")

    # ---- pan anchor invariant ------------------------------------------------------
    print("\npan anchor invariant:")
    session3 = Session()
    session3.viewport = cdx.Viewport(complex(0, 0), 1.5, 400)
    view3 = ImageView(session3)
    view3.resize(400, 400)

    anchor_pixel = QPoint(150, 100)
    anchor_complex = view3._pixel_to_complex(anchor_pixel)
    drag_to_pixel = QPoint(250, 260)   # dragged elsewhere

    vp3 = session3.viewport
    current_point = view3._pixel_to_complex(drag_to_pixel)
    new_center = vp3.center + (anchor_complex - current_point)
    session3.viewport = cdx.Viewport(new_center, vp3.scale, vp3.resolution)

    anchor_after = view3._pixel_to_complex(drag_to_pixel)
    check(close(anchor_complex, anchor_after, 1e-9),
          "after panning, the point anchored at drag-start is back under the cursor")

    # ---- window: initial render, threading, progressive display ------------------
    print("\nwindow: initial render happens and does not block:")
    window = SandboxWindow()
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "the initial render eventually produces a displayed image")

    # ---- render is off the GUI thread: a slow render must not block processEvents --
    # Fast built-in family (not a term-based custom map: slower per pixel,
    # and unnecessarily so for what this is testing), single render thread,
    # and a parameter/escape-radius combination with ordinary escape
    # dynamics -- calibrated by hand to ~2-3s, comfortably observable
    # without being so slow the test itself becomes the bottleneck. (An
    # earlier version of this test used newton_cubic() at a high iteration
    # count, which does not reliably escape a radius-2 test at all --
    # Newton's method orbits converge, they do not diverge -- so nearly
    # every pixel burned the full iteration budget and the "slow" render
    # actually took minutes, not seconds.)
    print("\nrendering off the GUI thread:")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.session.param = complex(-0.7269, 0.1889)
    window.session.viewport = cdx.Viewport(complex(0, 0), 1.5, 1800)
    window.session.render_settings = cdx.RenderSettings(400, 2.0, 1e-6, 1)   # single-threaded, slow

    t0 = time.perf_counter()
    window._start_render()
    dispatch_time = time.perf_counter() - t0
    check(dispatch_time < 0.2,
          f"starting a render returns immediately ({dispatch_time * 1000:.1f} ms), "
          "not after the render completes")

    # While that render is still in flight (it is deliberately slow), the
    # event loop must still be pumpable promptly -- a genuinely frozen GUI
    # thread would make this call itself block for the render's duration.
    t0 = time.perf_counter()
    QTest.qWait(50)
    pump_time = time.perf_counter() - t0
    check(pump_time < 1.0,
          f"the event loop stays responsive while a render is in flight ({pump_time * 1000:.1f} ms)")

    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=20000)
    check(ok, "the slow render eventually completes and is displayed")

    # ---- progressive rendering: preview arrives, then is replaced -----------------
    print("\nprogressive rendering:")
    seen_sizes = []
    original_set_image = window.image_view.set_image

    def tracking_set_image(array):
        seen_sizes.append(array.shape)
        original_set_image(array)

    window.image_view.set_image = tracking_set_image
    window.session.viewport = cdx.Viewport(complex(0.1, 0.1), 1.8, 800)
    window.session.render_settings = cdx.RenderSettings(500, 2.0, 1e-6, 1)
    window._start_render()
    ok = wait_for(lambda: len(seen_sizes) >= 2, timeout_ms=15000)
    window.image_view.set_image = original_set_image

    check(ok, "both a preview and a full-resolution image are displayed")
    if ok:
        preview_shape, full_shape = seen_sizes[0], seen_sizes[1]
        check(preview_shape[0] < full_shape[0],
              f"the preview ({preview_shape}) arrives before and is smaller than "
              f"the full render ({full_shape})")
        check(full_shape == (800, 800), "the full render matches the requested resolution")

    # ---- superseded requests are discarded ------------------------------------------
    print("\nsuperseded render requests:")
    window.session.viewport = cdx.Viewport(complex(0, 0), 1.5, 60)
    window._start_render()
    stale_id = window._request_id
    window._start_render()   # supersedes the previous request before it can complete
    current_id = window._request_id
    check(current_id != stale_id, "a new request gets a new id")

    import numpy as np
    window.image_view._pixmap = None
    window._on_full_ready(stale_id, np.zeros((10, 10)))
    check(window.image_view._pixmap is None,
          "a result tagged with a superseded request id is discarded, not displayed")

    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "the current (non-superseded) request still completes normally")

    # ---- Reset View --------------------------------------------------------------------
    print("\nReset View:")
    initial = window._initial_viewport
    window.session.viewport = cdx.Viewport(complex(5, 5), 0.001, 60)
    window._reset_view()
    vp = window.session.viewport
    check(close(vp.center, initial.center) and abs(vp.scale - initial.scale) < 1e-12,
          "Reset View restores exactly the viewport captured at startup")

    # ---- precision floor warning ---------------------------------------------------
    print("\nprecision floor warning:")
    renderer = cdx.Renderer(map=cdx.Map.custom(window.session.map, window.session.param),
                            viewport=window.session.viewport, settings=window.session.render_settings)
    tiny_scale = renderer.precision_floor / 10.0
    window.session.viewport = cdx.Viewport(complex(0, 0), tiny_scale, 60)
    window._update_status_bar()
    check("precision floor" in window.statusBar().currentMessage(),
          "the status bar warns when scale is at/below the precision floor")

    window.session.viewport = cdx.Viewport(complex(0, 0), 1.5, 60)
    window._update_status_bar()
    check("precision floor" not in window.statusBar().currentMessage(),
          "no warning at an ordinary scale")

    # Closes the window, which drains the thread pool (see
    # SandboxWindow.closeEvent) -- required here because the "superseded
    # render requests" section above deliberately leaves a stale task's
    # worker thread running in the background (discarding a result does not
    # cancel the render computing it); without draining before the process
    # exits, that thread can still be mid-render when Python starts tearing
    # down Qt objects, which crashes exactly the way a user closing the
    # real window mid-render would without the same closeEvent handling.
    window.close()

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
