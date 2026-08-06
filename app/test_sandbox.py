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

import math
import tempfile
import threading
import time

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import cdx
import app.sandbox as sandbox_module
from app.colour import PALETTES
from app.sandbox import ImageView, RenderTask, SandboxWindow, array_to_qimage
from app.session import Session
from app.settings import Settings

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

    # Every SandboxWindow() below must NOT touch the real
    # ~/.complexdynamics/settings.json -- load_settings() reading whatever a
    # previous real run of the app left there would make these tests
    # non-deterministic, and save_settings() writing to it would be a real
    # side effect on the machine running the test suite. Patched for the
    # whole module: load always returns plain defaults, save is a no-op.
    # (app/test_settings.py is what actually tests load_settings/
    # save_settings, against explicit temp paths.)
    sandbox_module.load_settings = lambda: Settings()
    sandbox_module.save_settings = lambda settings: None
    # Same concern, same fix, for ~/.complexdynamics/library.txt:
    # SandboxWindow.__init__ calls session.load_user_library(library_path())
    # unconditionally, and LibraryPanel's on_change callback calls
    # session.save_user_library(library_path()) on every successful
    # save/rename/delete. Pointing library_path() at a path that never
    # exists makes load a no-op (see Session.load_user_library's own
    # missing-file handling) and save write somewhere harmless instead of
    # the real file. (app/test_library_panel.py is what actually tests
    # save_user_library/load_user_library, against explicit temp paths.)
    _fake_library_path = tempfile.mktemp(suffix="-library.txt")
    sandbox_module.library_path = lambda: _fake_library_path

    print("=== app.sandbox tests ===")

    # ---- array_to_qimage: mode-aware colouring dispatch ---------------------------
    print("\narray_to_qimage (mode-aware colouring):")
    settings = Settings(colour_palette="viridis", colour_scaling="log1p", colour_period=0.0)
    # A tiny 1x2 array: row 0 (bottom, per cdx convention) never escapes,
    # row 1 (top) escapes near max_iter -- exercises both the never-escaped
    # flat colour AND a real palette lookup in one image.
    escape_array = np.array([[0.0], [199.0]])   # (height=2, width=1)

    img, normalized = array_to_qimage(escape_array, "julia", settings, max_iter=200)
    check(img.format() == QImage.Format.Format_RGB888,
          "julia mode produces an RGB image, not the old single-channel grayscale")
    check(normalized is None, "julia mode's normalized flag is None -- not a greens-family concept")
    check(img.width() == 1 and img.height() == 2, "output dimensions match the input array")
    # array_to_qimage flips vertically (row0-bottom -> row0-top for QImage,
    # same as the pre-P5c grayscale path) -- so QImage row 0 (top) shows the
    # ARRAY's row 1 (the escaped pixel, value 199.0).
    top_pixel = img.pixelColor(0, 0)
    check((top_pixel.red(), top_pixel.green(), top_pixel.blue())
          == tuple(int(c) for c in PALETTES["viridis"][255]),
          "the escaped pixel (smooth value near max_iter) lands at viridis' top colour, "
          "at the FLIPPED row (QImage top = array row 1)")
    bottom_pixel = img.pixelColor(0, 1)
    check((bottom_pixel.red(), bottom_pixel.green(), bottom_pixel.blue()) == (0, 0, 0),
          "the never-escaped pixel (array row 0, QImage bottom row) is flat black, not "
          "viridis' own index-0 colour")

    img_param, _n = array_to_qimage(escape_array, "parameter", settings, max_iter=200)
    check(img_param.format() == QImage.Format.Format_RGB888,
          "parameter mode gets the same escape-time RGB treatment as julia mode")

    # render_map's basin mode returns a STACKED (2, height, width) array --
    # index 0 labels, index 1 the per-pixel iteration count (see
    # session.render_map's own docstring) -- so array_to_qimage's basin
    # branch must be fed that same shape, not a plain 2D array.
    basin_labels = np.array([[0.0, 1.0, 1.0, 2.0]])
    basin_iters = np.array([[0.0, 3.0, 190.0, 3.0]])
    basin_array = np.stack([basin_labels, basin_iters])
    check(basin_array.shape == (2, 1, 4), "sanity: the stacked test array has the expected shape")

    basin_img, basin_normalized = array_to_qimage(basin_array, "basin", settings, max_iter=200)
    check(basin_img.format() == QImage.Format.Format_RGB888, "basin mode also produces RGB")
    check(basin_normalized is None, "basin mode's normalized flag is also None")
    unresolved = basin_img.pixelColor(0, 0)
    check((unresolved.red(), unresolved.green(), unresolved.blue()) == (0, 0, 0),
          "basin mode: an unresolved (label 0) pixel is flat black")
    b1, b3 = basin_img.pixelColor(1, 0), basin_img.pixelColor(3, 0)
    check((b1.red(), b1.green(), b1.blue()) != (b3.red(), b3.green(), b3.blue()),
          "basin mode: two different basin ids get visually distinct colours")
    fast, slow = basin_img.pixelColor(1, 0), basin_img.pixelColor(2, 0)   # same basin id (1)
    check(fast.red() + fast.green() + fast.blue() > slow.red() + slow.green() + slow.blue(),
          "basin mode: within the same basin, the FAST-converging pixel (iters=3) is brighter "
          "than the slow one (iters=190) -- shading actually reaches the displayed image")

    # render_map's "greens"/"parameter_greens" return (array, normalized)
    # TUPLES (see its own docstring), not a plain array -- array_to_qimage
    # must be fed that same shape.
    greens_settings = Settings(colour_palette="viridis", greens_band_width=1.0,
                               greens_period_bands=12.0, greens_contour=False)
    greens_payload = (np.array([[np.e ** 0, np.e ** 12]]), True)
    greens_img, greens_normalized = array_to_qimage(greens_payload, "greens", greens_settings,
                                                     max_iter=200)
    check(greens_img.format() == QImage.Format.Format_RGB888,
          "greens mode now gets a real RGB colour treatment (equipotential bands), not the "
          "old flat grayscale stretch")
    check(greens_normalized is True, "a normalized=True payload passes the flag straight through")
    g1, g2 = greens_img.pixelColor(0, 0), greens_img.pixelColor(1, 0)
    check((g1.red(), g1.green(), g1.blue()) == (g2.red(), g2.green(), g2.blue()),
          "e^0 and e^12 (12 bands apart at band_width=1, period_bands=12) land at the "
          "same colour -- greens_period_bands actually reaches the displayed pixels")

    unnorm_payload = (np.array([[1.0, 2.0]]), False)
    _img, unnorm_flag = array_to_qimage(unnorm_payload, "parameter_greens", greens_settings,
                                        max_iter=200)
    check(unnorm_flag is False,
          "parameter_greens propagates a normalized=False payload through unchanged")

    contour_settings = Settings(colour_palette="viridis", greens_band_width=1.0,
                                greens_period_bands=12.0, greens_contour=True)
    contour_payload = (np.array([[np.e ** 0, np.e ** 3]]), True)
    contour_img, _n2 = array_to_qimage(contour_payload, "greens", contour_settings, max_iter=200)
    check(contour_img.format() == QImage.Format.Format_RGB888,
          "greens_contour=True still produces a valid RGB image")

    period_settings = Settings(colour_palette="viridis", colour_scaling="log1p", colour_period=10.0)
    # Nonzero values one period apart, so both go through the real palette
    # lookup rather than one of them tripping the never-escaped (value==0)
    # short-circuit.
    wrap_array = np.array([[1.0, 11.0]])
    wrap_img, _n3 = array_to_qimage(wrap_array, "julia", period_settings, max_iter=200)
    p1, p2 = wrap_img.pixelColor(0, 0), wrap_img.pixelColor(1, 0)
    check((p1.red(), p1.green(), p1.blue()) == (p2.red(), p2.green(), p2.blue()),
          "with a colour_period of 10, values one period apart (1.0 and 11.0) land at the "
          "SAME colour -- the settings' period reaches the actual displayed pixels")

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

    for w in (complex(0.3, -0.2), complex(-1.0, 1.0), complex(0.0, 0.0)):
        px = view._complex_to_display_pixel(w)
        back = view._pixel_to_complex(px)
        check(abs(back - w) < 1e-9,
              f"_complex_to_display_pixel/_pixel_to_complex round-trip exactly for {w}")

    # ---- critical-point overlay: cache, orbit tracing, plane gating ----------------
    print("\ncritical-point overlay:")
    overlay_session = Session()
    overlay_session.map = cdx.RationalMap.mandelbrot()
    overlay_session.param = -1 + 0j   # the basilica: critical point 0, period-2 cycle 0 <-> -1
    overlay_session.set_render_mode("julia")
    overlay_view = ImageView(overlay_session)

    check(overlay_view._should_draw_critical_points() is False,
          "overlay is off by default even in a dynamical-plane mode")
    overlay_view.set_show_critical_points(True)
    check(overlay_view._should_draw_critical_points() is True,
          "enabling the checkbox turns it on in a dynamical-plane mode")

    overlay_session.set_render_mode("parameter")
    check(overlay_view._should_draw_critical_points() is False,
          "the overlay is suppressed on the PARAMETER plane -- critical points are dynamical-"
          "plane objects, per P5c's own spec")
    overlay_session.set_render_mode("parameter_greens")
    check(overlay_view._should_draw_critical_points() is False,
          "also suppressed on parameter_greens (the other parameter-plane mode)")
    overlay_session.set_render_mode("basin")
    check(overlay_view._should_draw_critical_points() is True,
          "shown again on basin mode -- basin pixels ARE dynamical-plane initial conditions")
    overlay_session.set_render_mode("julia")

    overlay_view.refresh_critical_points()
    check(0j in overlay_view._critical_points
         and any(math.isinf(z.real) for z in overlay_view._critical_points),
          "mandelbrot()'s critical points are 0 and infinity, matching cdx's own "
          "distinct_critical_points for this map")
    check(overlay_view._orbit_traces is None,
          "orbit traces are NOT computed until tracing is actually turned on")

    key_before = overlay_view._critical_points_key
    points_before = overlay_view._critical_points   # SAME list object if truly memoized
    overlay_session.viewport = cdx.Viewport(complex(0.4, 0.4), 0.2, 41)   # pan/zoom only
    overlay_view.refresh_critical_points()
    check(overlay_view._critical_points_key == key_before,
          "panning/zooming (viewport only) never changes the (map, param) cache key")
    check(overlay_view._critical_points is points_before,
          "-- and therefore never re-root-finds: the SAME list object survives a pan/zoom-"
          "triggered refresh call, not just an equal one")

    overlay_view.set_trace_orbits(True)
    check(overlay_view._orbit_traces is not None, "turning tracing on lazily fills in the traces")
    zero_trace = next(t for t in overlay_view._orbit_traces if t[0] == 0j)
    check(zero_trace[1] == -1 + 0j and zero_trace[2] == 0j,
          "the critical point 0's orbit traces the basilica's actual period-2 cycle: 0 -> -1 -> 0")

    inf_trace = next(t for t in overlay_view._orbit_traces
                     if math.isinf(t[0].real) or math.isinf(t[0].imag))
    check(len(inf_trace) == 1,
          "infinity's own orbit trace stops immediately (nothing further to evaluate at "
          "infinity) rather than crashing or looping")

    overlay_session.map = cdx.RationalMap.newton_cubic()
    overlay_view.refresh_critical_points()
    check(overlay_view._critical_points_key != key_before,
          "changing the MAP (not just the viewport) invalidates the cache key")
    check(overlay_view._critical_points is not points_before,
          "and genuinely recomputes -- a different list object, not the stale one")

    # ---- orbit tracking: click-to-seed, step/clear, painting, staleness -----------
    print("\norbit tracking:")
    from app.orbit_panel import OrbitPanel

    orbit_session = Session()
    orbit_session.map = cdx.RationalMap.mandelbrot()
    orbit_session.param = -1 + 0j   # the basilica again
    orbit_session.set_render_mode("julia")
    orbit_session.viewport = cdx.Viewport(complex(0, 0), 2.0, 400)
    orbit_view = ImageView(orbit_session)
    orbit_view.resize(400, 400)
    orbit_panel = OrbitPanel(orbit_session, orbit_view)

    check("Click the image" in orbit_panel._readout_label.text(),
          "the readout starts with no orbit seeded")
    check(orbit_view.orbit_tracker.state is None, "no orbit exists before any click")

    center_pixel = QPoint(200, 200)   # display-center -> complex 0, the basilica's own
                                      # critical point, which is ALREADY on the 2-cycle
    orbit_view._seed_orbit_at(center_pixel)
    check(orbit_view.orbit_tracker.state is not None, "_seed_orbit_at starts a real orbit")
    check(close(orbit_view.orbit_tracker.state.z0, complex(0, 0), 1e-2),
          "the orbit is seeded at the CLICKED point's complex coordinate")
    check("n = 0" in orbit_panel._readout_label.text(),
          "the panel's readout updates via the orbit_changed signal, not manual polling")

    orbit_view.step_orbit(3)
    check(orbit_view.orbit_tracker.state.n == 3, "step_orbit(3) advances the tracker by 3")
    check("n = 3" in orbit_panel._readout_label.text(), "the readout reflects the new n")
    check(len(orbit_view.orbit_tracker.state.history) == 4,
          "history has the seed plus 3 steps")

    orbit_view.clear_orbit()
    check(orbit_view.orbit_tracker.state is None, "clear_orbit removes the orbit")
    check("Click the image" in orbit_panel._readout_label.text(),
          "the readout reverts to the no-orbit prompt after Clear")

    # Parameter-plane modes: click-to-seed is a no-op (orbit tracking is a
    # dynamical-plane concept, same gating as the critical-point overlay).
    orbit_session.set_render_mode("parameter")
    orbit_view._seed_orbit_at(center_pixel)
    check(orbit_view.orbit_tracker.state is None,
          "clicking on the PARAMETER plane does not seed an orbit")
    orbit_session.set_render_mode("julia")

    # Staleness: survives a pan/zoom, clears on a map/param change -- the
    # SAME property already verified directly against OrbitTracker itself
    # in app/test_orbit_tracker.py; here we confirm ImageView's own
    # refresh_orbit_staleness wiring actually reaches it.
    orbit_view._seed_orbit_at(center_pixel)
    check(orbit_view.orbit_tracker.state is not None, "sanity: an orbit exists again")
    orbit_session.viewport = cdx.Viewport(complex(0.3, 0.3), 1.0, 400)   # pan/zoom only
    orbit_view.refresh_orbit_staleness()
    check(orbit_view.orbit_tracker.state is not None,
          "a viewport-only change survives refresh_orbit_staleness -- panning/zooming is the "
          "same dynamics, seen differently")

    orbit_session.param = -0.5 + 0j   # a genuine parameter change
    orbit_view.refresh_orbit_staleness()
    check(orbit_view.orbit_tracker.state is None,
          "a parameter change clears the orbit via refresh_orbit_staleness")

    # ---- orbit tracking: a real click (press+release, no drag) seeds; a real drag doesn't --
    print("\norbit tracking (mouse press/release gesture):")
    orbit_session2 = Session()
    orbit_session2.map = cdx.RationalMap.mandelbrot()
    orbit_session2.set_render_mode("julia")
    orbit_session2.viewport = cdx.Viewport(complex(0, 0), 2.0, 400)
    gesture_view = ImageView(orbit_session2)
    gesture_view.resize(400, 400)

    class _FakeMouseEvent:
        def __init__(self, pos: QPoint, button=Qt.MouseButton.LeftButton, modifiers=None):
            self._pos = pos
            self._button = button
            self._modifiers = modifiers or Qt.KeyboardModifier.NoModifier
        def position(self):
            return QPointF(self._pos)
        def button(self):
            return self._button
        def modifiers(self):
            return self._modifiers

    gesture_view.mousePressEvent(_FakeMouseEvent(QPoint(200, 200)))
    gesture_view.mouseReleaseEvent(_FakeMouseEvent(QPoint(200, 200)))   # no movement -> a click
    check(gesture_view.orbit_tracker.state is not None,
          "a press+release with no movement in between seeds an orbit -- a genuine click")

    gesture_view.clear_orbit()
    gesture_view.mousePressEvent(_FakeMouseEvent(QPoint(50, 50)))
    gesture_view.mouseMoveEvent(_FakeMouseEvent(QPoint(300, 300)))
    gesture_view.mouseReleaseEvent(_FakeMouseEvent(QPoint(300, 300)))   # a real drag
    check(gesture_view.orbit_tracker.state is None,
          "a real drag (rubber-band zoom) does NOT also seed an orbit at the release point")
    check(close(gesture_view.session.viewport.center, complex(0, 0), 0.7),
          "sanity: the drag actually did rubber-band zoom (viewport changed)")

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

    # ---- overscan: instant feedback draws real pixels from the buffer's margin ----
    print("\noverscan buffer mapping:")

    # set_image() is handed a buffer WIDER than the display viewport (same
    # center, same pixel density, scaled up by an overscan factor -- what
    # RenderTask.run() actually produces via _overscanned). The mapping
    # back onto the display must use the buffer's OWN stored viewport, not
    # assume the buffer matches the display exactly.
    session5 = Session()
    session5.viewport = cdx.Viewport(complex(0.3, -0.2), 1.0, 400)
    view5 = ImageView(session5)
    view5.resize(400, 400)

    overscan_factor = 1.3
    buffer_res = round(400 * overscan_factor)   # 520, exactly -- no rounding noise
    buffer_viewport = cdx.Viewport(session5.viewport.center,
                                   session5.viewport.scale * overscan_factor, buffer_res)
    view5.set_image(np.zeros((buffer_res, buffer_res)), buffer_viewport)

    # Cross-check against a fully independent hand computation -- not
    # _pixel_to_complex/_complex_to_buffer_pixel called and trusted, but the
    # same formulas worked out by hand here -- for an arbitrary INTERIOR
    # probe pixel (not a corner), confirming the two-corner source rect
    # really does determine every point, as the affine argument in
    # ImageView's docstring claims.
    probe_pixel = QPoint(310, 120)
    disp_vp = session5.viewport
    rel_x = probe_pixel.x() / 400.0
    rel_y = probe_pixel.y() / 400.0
    probe_complex = complex(disp_vp.center.real - disp_vp.scale + rel_x * 2.0 * disp_vp.scale,
                            disp_vp.center.imag + disp_vp.scale - rel_y * 2.0 * disp_vp.scale)
    buf_rel_x = ((probe_complex.real - (buffer_viewport.center.real - buffer_viewport.scale))
                / (2.0 * buffer_viewport.scale))
    buf_rel_y = ((buffer_viewport.center.imag + buffer_viewport.scale - probe_complex.imag)
                / (2.0 * buffer_viewport.scale))
    expected_buffer_pixel = QPointF(buf_rel_x * buffer_res, buf_rel_y * buffer_res)

    source_rect = view5._buffer_source_rect()
    actual_buffer_pixel = QPointF(source_rect.left() + rel_x * source_rect.width(),
                                  source_rect.top() + rel_y * source_rect.height())

    check(abs(actual_buffer_pixel.x() - expected_buffer_pixel.x()) < 1e-6 and
          abs(actual_buffer_pixel.y() - expected_buffer_pixel.y()) < 1e-6,
          "the buffer source rect maps a display pixel to exactly the complex point "
          "the display viewport says it shows, via the buffer's own stored viewport")

    # With no pan/zoom drift since the render (display viewport's center
    # equals the buffer's own center), the source rect is exactly the
    # buffer's centred 1/overscan_factor crop.
    expected_side = buffer_res / overscan_factor
    expected_margin = (buffer_res - expected_side) / 2.0
    check(abs(source_rect.width() - expected_side) < 1e-9 and
          abs(source_rect.left() - expected_margin) < 1e-9,
          "with no drift since the render, the source rect is the buffer's centred crop")

    # buffer_edge_fraction: exactly 1/overscan_factor right after a fresh
    # render (the display viewport's own half-width against the buffer's
    # larger one), growing as the viewport zooms out or pans away from the
    # buffer's center -- and an ImageView with no buffer yet is always due.
    check(abs(view5.buffer_edge_fraction() - 1.0 / overscan_factor) < 1e-9,
          "buffer_edge_fraction right after a render is exactly 1/overscan_factor")

    session5.viewport = cdx.Viewport(disp_vp.center, disp_vp.scale * 1.25, disp_vp.resolution)
    check(view5.buffer_edge_fraction() > 1.0 / overscan_factor,
          "zooming out grows the buffer edge fraction")

    fresh_view = ImageView(Session())
    check(fresh_view.buffer_edge_fraction() == 1.0,
          "an ImageView with no buffer yet is always due for a render")

    # ---- window: initial render, threading, progressive display ------------------
    print("\nwindow: initial render happens and does not block:")
    window = SandboxWindow()
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "the initial render eventually produces a displayed image")

    # Checked HERE, immediately after construction and before anything else
    # touches session.render_mode -- later sections poke session.render_mode
    # directly (bypassing the combo box, to set up unrelated scenarios),
    # which is not something the combo box is expected to track (nothing in
    # the real app ever changes render_mode except through the combo box
    # itself -- see the dedicated "mode selector" section further down for
    # that actual UI path).
    check(window.mode_combo.currentText() == window.session.render_mode,
          "the combo box starts on the session's actual startup render mode")

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
    window.session.render_mode = "julia"   # Session now starts in "parameter"; this test needs julia
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

    # ---- cancellation ------------------------------------------------------------
    print("\ncancellation:")

    # RenderTask.run()'s own cancellation handling, in isolation: called
    # synchronously here (not through the thread pool), with a background
    # thread cancelling shortly after it starts, so this measures ONLY
    # run()'s response to cancellation -- not conflated with QThreadPool
    # scheduling or _pending_tasks' sweep-on-next-start cleanup timing
    # (tested separately below), the way going through
    # SandboxWindow._start_render() would.
    slow_map = cdx.RationalMap.mandelbrot()
    slow_param = complex(-0.7269, 0.1889)
    slow_viewport = cdx.Viewport(complex(0.05, 0.05), 1.4, 1000)
    slow_settings = cdx.RenderSettings(250, 2.0, 1e-6, 1)   # single-threaded, slow but bounded

    t0 = time.perf_counter()
    RenderTask(1, slow_map, slow_param, slow_viewport, slow_settings,
              "julia", cdx.CancelToken()).run()
    uncancelled_time = time.perf_counter() - t0

    cancel_token = cdx.CancelToken()
    def cancel_soon():
        time.sleep(0.02)
        cancel_token.cancel()
    canceller = threading.Thread(target=cancel_soon)
    canceller.start()
    t0 = time.perf_counter()
    RenderTask(2, slow_map, slow_param, slow_viewport, slow_settings,
              "julia", cancel_token).run()
    cancelled_time = time.perf_counter() - t0
    canceller.join()

    check(cancelled_time < uncancelled_time * 0.5,
          f"a cancelled RenderTask.run() returns well under the uncancelled time "
          f"({cancelled_time:.3f}s vs {uncancelled_time:.3f}s)")

    # Now the same scenario through the real dispatch path: starting a new
    # render immediately cancels the superseded one's token.
    window.session.viewport = slow_viewport
    window._start_render()
    stale_task = window._pending_tasks[window._request_id]
    window._start_render()   # supersedes the task started above
    check(stale_task.cancel.is_cancelled,
          "starting a new render immediately cancels the superseded one's token")

    # Rapid-fire many supersessions (simulating a fast scroll burst, each
    # notch calling _start_render once the debounce settles) --
    # _pending_tasks must not accumulate one stale entry per call.
    window.session.viewport = cdx.Viewport(complex(0, 0), 1.5, 80)
    for _ in range(15):
        window._start_render()
    ok = wait_for(lambda: len(window._pending_tasks) <= 1, timeout_ms=10000)
    check(ok, "15 rapid supersessions leave at most the current task pending, not 15 stale ones")

    # closeEvent cancels and returns immediately rather than draining the
    # pool -- verified on a second window (the shared `window` above gets
    # closed at the very end of this script) so the rest of the test suite
    # can keep using it afterward.
    w2 = SandboxWindow()
    w2.session.map = cdx.RationalMap.mandelbrot()
    w2.session.param = complex(-0.7269, 0.1889)
    w2.session.viewport = cdx.Viewport(complex(0, 0), 1.5, 1800)
    w2.session.render_settings = cdx.RenderSettings(400, 2.0, 1e-6, 1)
    w2._start_render()
    pending_before_close = list(w2._pending_tasks.values())

    t0 = time.perf_counter()
    w2.close()
    close_time = time.perf_counter() - t0
    check(close_time < 0.2,
          f"closeEvent returns immediately ({close_time * 1000:.1f} ms), not after draining the pool")
    check(all(t.cancel.is_cancelled for t in pending_before_close),
          "closeEvent cancelled every task that was still pending")
    # Let the now-cancelled background render actually finish before this
    # script exits, for the same reason test_sandbox.py's very last line
    # closes `window`: an interpreter shutdown racing a still-running (even
    # if now brief) worker thread is exactly the crash this mechanism
    # exists to avoid.
    QTest.qWait(500)

    # ---- progressive rendering: preview arrives, then is replaced -----------------
    print("\nprogressive rendering:")
    seen_sizes = []
    original_set_image = window.image_view.set_image

    def tracking_set_image(array, buffer_viewport):
        seen_sizes.append(array.shape)
        original_set_image(array, buffer_viewport)

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
        # Overscanned by FULL_OVERSCAN_FACTOR (1.3), not the bare requested
        # resolution -- round(800*1.3) == 1040 exactly.
        expected_full = round(800 * 1.3)
        check(full_shape == (expected_full, expected_full),
              f"the full render is the requested resolution, overscanned ({expected_full})")

    # ---- render cache: RenderTask actually shares and hits session.cache ----------
    print("\nrender cache integration:")
    window.session.viewport = cdx.Viewport(complex(0.05, -0.05), 1.3, 60)
    window.session.render_settings = cdx.RenderSettings(60, 2.0, 1e-6, 1)

    window._start_render()
    ok = wait_for(lambda: window._request_id not in window._pending_tasks, timeout_ms=10000)
    check(ok, "first render (cache cold for this viewport/settings) completes")
    stats_after_first = window.session.cache.stats

    window._start_render()   # identical viewport/settings -- both stages should now hit
    ok = wait_for(lambda: window._request_id not in window._pending_tasks, timeout_ms=10000)
    check(ok, "second, identical render completes")
    stats_after_second = window.session.cache.stats
    check(stats_after_second.hits >= stats_after_first.hits + 2,
          "a repeat request at the same viewport/settings hits the cache for both the "
          "preview and full overscanned buffers, not just one")
    check(stats_after_second.misses == stats_after_first.misses,
          "the repeat request causes no new misses -- nothing new needed computing")

    # ---- superseded requests are discarded ------------------------------------------
    print("\nsuperseded render requests:")
    window.session.viewport = cdx.Viewport(complex(0, 0), 1.5, 60)
    window._start_render()
    stale_id = window._request_id
    window._start_render()   # supersedes the previous request before it can complete
    current_id = window._request_id
    check(current_id != stale_id, "a new request gets a new id")

    window.image_view._pixmap = None
    window._on_full_ready(stale_id, np.zeros((10, 10)), cdx.Viewport())
    check(window.image_view._pixmap is None,
          "a result tagged with a superseded request id is discarded, not displayed")

    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "the current (non-superseded) request still completes normally")

    # ---- Settings tab: Apply reaches the session and triggers a real re-render ------
    print("\nsettings tab:")
    check(window.tabs.count() == 5 and window.tabs.tabText(0) == "View"
          and window.tabs.tabText(1) == "Terms" and window.tabs.tabText(2) == "Facts"
          and window.tabs.tabText(3) == "Library" and window.tabs.tabText(4) == "Settings",
          "the window has View, Terms, Facts, Library, and Settings tabs, in that order")
    check(window.tabs.widget(1) is window.term_editor_panel,
          "the Terms tab holds the actual TermEditorPanel instance")
    check(window.tabs.widget(2) is window.facts_panel,
          "the Facts tab holds the actual FactsPanel instance")
    check(window.tabs.widget(3) is window.library_panel,
          "the Library tab holds the actual LibraryPanel instance")
    check(window.tabs.widget(4) is window.settings_panel,
          "the Settings tab holds the actual SettingsPanel instance")

    window.session.viewport = cdx.Viewport(complex(0, 0), 1.5, 60)
    window._start_render()
    wait_for(lambda: window._request_id not in window._pending_tasks, timeout_ms=10000)

    new_resolution = 250   # above the resolution widget's own 200 floor -- see FIELD_SPECS
    window.settings_panel._widgets["resolution"].setValue(new_resolution)
    window.settings_panel._widgets["threads"].setValue(1)
    window.settings_panel._apply()

    check(window.session.viewport.resolution == new_resolution,
          "Apply updates the session's viewport resolution")
    check(window.session.render_settings.threads == 1,
          "Apply updates the session's render_settings too, not just resolution")

    ok = wait_for(lambda: window.image_view._buffer_viewport is not None
                 and window.image_view._buffer_viewport.resolution == round(new_resolution * 1.3),
                 timeout_ms=10000)
    check(ok, "Apply triggers an immediate re-render at the NEW resolution -- not the debounce, "
          "and not the old one")

    # ---- mode selector: switching modes reaches the session and re-renders ------------
    print("\nmode selector:")
    check([window.mode_combo.itemText(i) for i in range(window.mode_combo.count())]
          == list(sandbox_module.RENDER_MODES),
          "the combo box offers exactly RENDER_MODES, in order")

    window.image_view._pixmap = None   # so wait_for below can't see a stale hit from above
    window.mode_combo.setCurrentText("julia")
    check(window.session.render_mode == "julia",
          "selecting a mode in the combo box updates session.render_mode")
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "switching modes triggers an immediate re-render -- not the debounce, and the "
          "new mode actually produces a displayed image")

    # Basin mode specifically, through the REAL end-to-end pipeline
    # (RenderTask -> session.render_map's stacked (2,H,W) array ->
    # array_to_qimage's basin branch), not just the pure-function tests
    # above -- this is the actual new code path the P5c basin-shading
    # milestone added, and nothing else exercises it live.
    window.image_view._pixmap = None
    window.mode_combo.setCurrentText("basin")
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "switching to basin mode renders and displays successfully end-to-end, with the "
          "new stacked labels+iterations array flowing all the way through")

    # Both Green's-function modes, likewise through the REAL pipeline --
    # session.render_map's (array, normalized) tuple flowing through
    # RenderCache/RenderTask's Qt signals/array_to_qimage, and the status
    # bar picking up image_view._last_normalized once the render lands.
    window.image_view._pixmap = None
    window.image_view._last_normalized = None
    window.mode_combo.setCurrentText("greens")
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "switching to greens mode renders and displays successfully end-to-end")
    check(window.image_view._last_normalized in (True, False),
          "the real render pipeline actually sets _last_normalized to a real bool, not left "
          "at its constructor default of None")
    check(window.image_view._last_normalized is True,
          "at this session's default max_iter, degree^max_iter does not overflow, so the "
          "normalization is genuinely fine -- no warning belongs in the status bar")
    check("overflowed" not in window.statusBar().currentMessage(),
          "no unnormalized warning in the status bar when normalization actually succeeded")

    window.image_view._pixmap = None
    window.image_view._last_normalized = None
    window.mode_combo.setCurrentText("parameter_greens")
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "switching to parameter_greens mode renders and displays successfully end-to-end")
    check(window.image_view._last_normalized is True, "parameter_greens also normalizes fine here")

    # Force the overflow path (huge max_iter -> degree^max_iter overflows
    # double) and confirm the status bar actually warns, not just that
    # the internal flag is set correctly (already checked in the pure
    # array_to_qimage tests above) -- this is the one thing only a live
    # window can confirm: the flag reaching the ACTUAL displayed message.
    window.session.render_settings = cdx.RenderSettings(2000, 2.0, 1e-6, 0)
    window.image_view._pixmap = None
    window.image_view._last_normalized = None
    window._start_render()
    ok = wait_for(lambda: window.image_view._last_normalized is not None, timeout_ms=10000)
    check(ok, "a render completes even with the overflow-inducing max_iter")
    check(window.image_view._last_normalized is False,
          "max_iter=2000 overflows degree^max_iter -- normalization genuinely failed here")
    window._update_status_bar()
    check("overflowed" in window.statusBar().currentMessage(),
          "the status bar surfaces the unnormalized warning once the flag is actually False")

    # ---- critical-point overlay toolbar checkboxes -------------------------------
    print("\ncritical-point overlay checkboxes:")
    check(window.image_view._show_critical_points is False,
          "the overlay starts off, matching the unchecked checkbox")
    window.critical_points_checkbox.setChecked(True)
    check(window.image_view._show_critical_points is True,
          "checking the toolbar checkbox actually reaches image_view's own flag")
    window.trace_orbits_checkbox.setChecked(True)
    check(window.image_view._trace_orbits is True,
          "checking the trace-orbits checkbox reaches image_view's own flag")
    check(window.image_view._orbit_traces is not None,
          "checking it also triggers the lazy orbit-trace computation via set_trace_orbits")
    window.critical_points_checkbox.setChecked(False)
    check(window.image_view._show_critical_points is False, "unchecking turns it back off")

    # ---- centre-view still switches to the View tab (regression: the View tab's ------
    # ---- widget became a container around image_view, not image_view itself, when ----
    # ---- the orbit-tracking strip was added below it) ---------------------------------
    print("\ncentre-view switches to the View tab:")
    check(window.tabs.widget(0) is window.view_container,
          "the View tab's actual page is the container (image_view + orbit strip), not "
          "image_view directly")
    window.tabs.setCurrentIndex(2)   # Facts tab -- anywhere but View
    window._on_center_view(complex(0.1, 0.1))
    check(window.tabs.currentWidget() is window.view_container,
          "_on_center_view actually switches to the View tab's container -- setCurrentWidget("
          "self.image_view) would silently no-op now that image_view isn't the tab's own "
          "page, which is exactly the regression this guards against")

    # ---- metadata header: reflects mode/map changes live ------------------------------
    print("\nmetadata header:")
    from app.metadata_header import format_metadata_text
    check(window.metadata_header._label.text() == format_metadata_text(window.session),
          "the header's displayed text matches the session's actual current state")

    window.mode_combo.setCurrentText("parameter")
    check("Parameter plane" in window.metadata_header._label.text(),
          "switching modes via the combo box refreshes the header's domain text")
    window.mode_combo.setCurrentText("julia")

    window.session.map.name = "renamed-live"
    window.term_editor_panel._notify_edit()   # the real path _on_term_edited fires from
    check("renamed-live" in window.metadata_header._label.text(),
          "a term edit refreshes the header's map name/formula")

    # ---- Reset View --------------------------------------------------------------------
    print("\nReset View:")
    initial = window._initial_viewport
    # Resolution stays at new_resolution (from the Settings Apply above) --
    # only center/scale are thrown away here, to isolate what Reset View
    # itself is being tested against.
    window.session.viewport = cdx.Viewport(complex(5, 5), 0.001, new_resolution)
    window._reset_view()
    vp = window.session.viewport
    check(close(vp.center, initial.center) and abs(vp.scale - initial.scale) < 1e-12,
          "Reset View restores exactly the viewport captured at startup")
    check(vp.resolution == new_resolution,
          "Reset View does NOT revert resolution -- that is a Settings concern (see the last "
          "Apply above), not part of 'the view' pan/zoom resets")

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
