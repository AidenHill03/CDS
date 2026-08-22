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

import base64
import dataclasses
import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QImage, QKeyEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import cdx
import app.sandbox as sandbox_module
import app.settings as settings_module
from app.colour import NEVER_ESCAPED_RGB, PALETTES
from app.library_panel import (_DYNAMICAL_VIEW_FALLBACK, _LIST_ICON_SIZE, _NAME_ROLE,
                               _placeholder_icon, default_dynamical_view, default_view_for)
from app.pane import Pane
from app.sandbox import (ATTRACTING_CYCLES_LAYER_KEY, CRITICAL_POINTS_LAYER_KEY, CYCLE_TRACE_KEY,
                         FIXED_POINTS_LAYER_KEY, ExportImageDialog, ImageView, RenderTask,
                         SandboxWindow, _attracting_cycles_provider,
                         _attracting_cycles_trace_provider, _CYCLE_STRONG_COLOUR,
                         _CYCLE_WEAK_COLOUR, _cycle_colour, _fixed_points_provider,
                         _FIXED_POINT_CLASSIFICATION_COLOURS, _OVERLAY_LAYERS_BY_KEY,
                         array_to_qimage, build_legend_entries, compose_export_image,
                         drawable_polyline_segments, paint_registry_layers)
from app.facts_panel import _classify, _is_inf
from app.session import Session, render_map
from app.settings import Settings, preview_path_for


def _pane_view(session: Session, center: complex = 0j, scale: float = 1.5,
              resolution: int = 400, render_mode: str = "julia"):
    """A Pane + its ImageView, wired together the same way SandboxWindow._build_ui
    does (pane constructed first with a real viewport/mode, then ImageView(pane,
    session) reads its initial state from it, then pane.image_view is set back)
    -- the standalone-widget-test equivalent of a real window's single pane,
    used throughout this file wherever a test needs an ImageView without a
    full SandboxWindow.
    """
    pane = Pane(cdx.Viewport(center, scale, resolution), render_mode)
    view = ImageView(pane, session)
    pane.image_view = view
    return pane, view

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


_PERF_ASSERT = os.environ.get("CDX_PERF_ASSERT") is not None


def check_perf(cond: bool, what: str) -> None:
    """Timing-ratio guard: reliable on dev hardware, too tight/noisy on shared
    CI runners. Reports always; only fails when explicitly enforced via
    CDX_PERF_ASSERT, matching the C++ perf gates in test_custom.cpp."""
    if _PERF_ASSERT:
        check(cond, what)
    else:
        print(f"  [{'ok' if cond else 'WARN'}] {what} "
              f"(perf; set CDX_PERF_ASSERT to enforce)")


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
    # Same concern, same fix, for ~/.complexdynamics/previews/ (Stage C):
    # _on_library_changed calls _regenerate_library_previews on every
    # successful save/rename/delete/notes-edit, which WRITES a sidecar PNG
    # per non-preset entry, and LibraryPanel._icon_for READS from the same
    # place on every list refresh. Both sandbox.py's and library_panel.py's
    # own preview_path_for/previews_dir imports resolve config_dir()
    # through app.settings's OWN globals internally (see
    # app.settings.previews_dir's body), so patching config_dir there ONCE
    # redirects every one of them consistently -- not two separate
    # per-module lambda rebinds that would need to be kept in sync.
    _fake_previews_config_dir = Path(tempfile.mkdtemp(suffix="-config"))
    settings_module.config_dir = lambda: _fake_previews_config_dir

    print("=== app.sandbox tests ===")

    # ---- array_to_qimage: mode-aware colouring dispatch ---------------------------
    print("\narray_to_qimage (mode-aware colouring):")
    settings = Settings(colour_palette="viridis", colour_scaling="log1p", colour_period=0.0)
    # A tiny 1x2 array: row 0 (bottom, per cdx convention) never escapes,
    # row 1 (top) escapes near max_iter -- exercises both the never-escaped
    # flat colour AND a real palette lookup in one image.
    escape_array = np.array([[0.0], [199.0]])   # (height=2, width=1)

    img = array_to_qimage(escape_array, "julia", settings, max_iter=200)
    check(img.format() == QImage.Format.Format_RGB888,
          "julia mode produces an RGB image, not the old single-channel grayscale")
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

    img_param = array_to_qimage(escape_array, "parameter", settings, max_iter=200)
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

    basin_img = array_to_qimage(basin_array, "basin", settings, max_iter=200)
    check(basin_img.format() == QImage.Format.Format_RGB888, "basin mode also produces RGB")
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

    # Stage 0: settings.colour_scaling/colour_period reach basin mode too, not just
    # escape-time/greens -- array_to_qimage's basin branch must actually forward them.
    basin_hist_settings = Settings(colour_palette="viridis", colour_scaling="histogram",
                                   colour_period=0.0)
    basin_hist_img = array_to_qimage(basin_array, "basin", basin_hist_settings, max_iter=200)
    log_pixels = [basin_img.pixelColor(x, 0) for x in range(4)]
    hist_pixels = [basin_hist_img.pixelColor(x, 0) for x in range(4)]
    check(any((lp.red(), lp.green(), lp.blue()) != (hp.red(), hp.green(), hp.blue())
             for lp, hp in zip(log_pixels, hist_pixels)),
          "settings.colour_scaling='histogram' actually reaches basin mode's rendered pixels, "
          "producing DIFFERENT bytes than the default 'log1p' -- not silently ignored")
    unresolved_hist = basin_hist_img.pixelColor(0, 0)
    check((unresolved_hist.red(), unresolved_hist.green(), unresolved_hist.blue()) == (0, 0, 0),
          "...and the unresolved pixel stays flat black under histogram scaling too")

    basin_period_settings = Settings(colour_palette="viridis", colour_scaling="log1p",
                                     colour_period=10.0)
    period_labels = np.array([[1.0, 1.0]])
    period_iters = np.array([[1.0, 11.0]])   # one period (10) apart
    period_array = np.stack([period_labels, period_iters])
    basin_period_img = array_to_qimage(period_array, "basin", basin_period_settings, max_iter=200)
    p0, p1 = basin_period_img.pixelColor(0, 0), basin_period_img.pixelColor(1, 0)
    check((p0.red(), p0.green(), p0.blue()) == (p1.red(), p1.green(), p1.blue()),
          "settings.colour_period reaches basin mode too -- values one period apart land at "
          "the same shading")

    # Stage 2 (corrected): a RATIONAL map's "julia" render is stacked (2,
    # height, width) -- values then labels -- distinguished from the
    # certified-polynomial plain-2D case by ndim alone (see
    # array_to_qimage's own docstring), not a separate flag. GOVERNING
    # PRINCIPLE: only the VALUES layer is coloured, through the exact same
    # colour_escape_time pipeline a plain escape-time array already uses --
    # labels never reach the colourer at all (that's basin mode's own job).
    # pixel 0: unresolved (value=0, label=0). pixels 1&2: the SAME value
    # (100) but DIFFERENT labels (1 vs 2) -- must land at the SAME colour,
    # proving colouring is label-blind. pixel 3: the SAME label as pixel 1
    # (1) but a DIFFERENT value (199) -- must land at a DIFFERENT colour,
    # proving colouring follows the smooth value like ordinary escape time.
    rational_julia_values = np.array([[0.0, 100.0, 100.0, 199.0]])
    rational_julia_labels = np.array([[0.0, 1.0, 2.0, 1.0]])
    rational_julia_array = np.stack([rational_julia_values, rational_julia_labels])
    check(rational_julia_array.shape == (2, 1, 4),
          "sanity: the stacked rational-julia test array has the expected shape")
    rational_julia_img = array_to_qimage(rational_julia_array, "julia", settings, max_iter=200)
    check(rational_julia_img.format() == QImage.Format.Format_RGB888,
          "rational julia mode also produces RGB, via the SAME colour_escape_time path a "
          "certified polynomial's plain array uses")
    check(rational_julia_img.width() == 4 and rational_julia_img.height() == 1,
          "output dimensions match the STACKED array's own (height, width), not (2, height, width)")
    unresolved_j = rational_julia_img.pixelColor(0, 0)
    check((unresolved_j.red(), unresolved_j.green(), unresolved_j.blue()) == NEVER_ESCAPED_RGB,
          "rational julia: an unresolved (value=0) pixel gets colour_escape_time's own "
          "NEVER_ESCAPED_RGB sentinel, the SAME flat colour a never-escaped polynomial pixel "
          "gets -- not a separate 'unresolved basin' treatment")
    j1, j2, j3 = (rational_julia_img.pixelColor(1, 0), rational_julia_img.pixelColor(2, 0),
                 rational_julia_img.pixelColor(3, 0))
    check((j1.red(), j1.green(), j1.blue()) == (j2.red(), j2.green(), j2.blue()),
          "rational julia: two pixels with the SAME smooth value but DIFFERENT basin labels "
          "get the IDENTICAL colour -- colouring genuinely ignores which attractor was reached")
    check((j1.red(), j1.green(), j1.blue()) != (j3.red(), j3.green(), j3.blue()),
          "...but two pixels with the SAME label and DIFFERENT smooth values get DIFFERENT "
          "colours -- colouring follows the value, exactly like ordinary escape time")

    # Palette-sensitivity: changing settings.colour_palette must change a
    # rational julia render's output, the same as it would for a certified
    # polynomial's -- proving this genuinely goes through colour_escape_time
    # (palette-aware) and not some palette-blind bespoke path.
    magma_settings = Settings(colour_palette="magma", colour_scaling="log1p", colour_period=0.0)
    rational_julia_magma = array_to_qimage(rational_julia_array, "julia", magma_settings,
                                           max_iter=200)
    check(rational_julia_magma.constBits().tobytes() != rational_julia_img.constBits().tobytes(),
          "rational julia: changing settings.colour_palette changes the rendered bytes -- "
          "settings.colour_palette/colour_scaling/colour_period are genuinely honoured, "
          "exactly as the governing principle requires")

    # render_map's "greens"/"parameter_greens" return a plain 2D array (see
    # its own docstring) -- array_to_qimage must be fed that same shape.
    greens_settings = Settings(colour_palette="viridis", greens_band_width=1.0,
                               greens_period_bands=12.0, greens_contour=False)
    greens_payload = np.array([[np.e ** 0, np.e ** 12]])
    greens_img = array_to_qimage(greens_payload, "greens", greens_settings, max_iter=200)
    check(greens_img.format() == QImage.Format.Format_RGB888,
          "greens mode now gets a real RGB colour treatment (equipotential bands), not the "
          "old flat grayscale stretch")
    g1, g2 = greens_img.pixelColor(0, 0), greens_img.pixelColor(1, 0)
    check((g1.red(), g1.green(), g1.blue()) == (g2.red(), g2.green(), g2.blue()),
          "e^0 and e^12 (12 bands apart at band_width=1, period_bands=12) land at the "
          "same colour -- greens_period_bands actually reaches the displayed pixels")

    pgreens_payload = np.array([[1.0, 2.0]])
    _img = array_to_qimage(pgreens_payload, "parameter_greens", greens_settings, max_iter=200)
    check(_img.format() == QImage.Format.Format_RGB888,
          "parameter_greens gets the same scalar-field RGB treatment as greens")

    contour_settings = Settings(colour_palette="viridis", greens_band_width=1.0,
                                greens_period_bands=12.0, greens_contour=True)
    contour_payload = np.array([[np.e ** 0, np.e ** 3]])
    contour_img = array_to_qimage(contour_payload, "greens", contour_settings, max_iter=200)
    check(contour_img.format() == QImage.Format.Format_RGB888,
          "greens_contour=True still produces a valid RGB image")

    period_settings = Settings(colour_palette="viridis", colour_scaling="log1p", colour_period=10.0)
    # Nonzero values one period apart, so both go through the real palette
    # lookup rather than one of them tripping the never-escaped (value==0)
    # short-circuit.
    wrap_array = np.array([[1.0, 11.0]])
    wrap_img = array_to_qimage(wrap_array, "julia", period_settings, max_iter=200)
    p1, p2 = wrap_img.pixelColor(0, 0), wrap_img.pixelColor(1, 0)
    check((p1.red(), p1.green(), p1.blue()) == (p2.red(), p2.green(), p2.blue()),
          "with a colour_period of 10, values one period apart (1.0 and 11.0) land at the "
          "SAME colour -- the settings' period reaches the actual displayed pixels")

    # ---- pixel <-> complex coordinate mapping -----------------------------------
    print("\npixel <-> complex mapping:")
    session = Session()
    pane, view = _pane_view(session, complex(0, 0), 1.5, 400)
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
    overlay_pane, overlay_view = _pane_view(overlay_session, render_mode="julia")

    check(overlay_view._should_draw_critical_points() is False,
          "overlay is off by default even in a dynamical-plane mode")
    overlay_view.set_show_critical_points(True)
    check(overlay_view._should_draw_critical_points() is True,
          "enabling the checkbox turns it on in a dynamical-plane mode")

    overlay_pane.set_render_mode("parameter")
    check(overlay_view._should_draw_critical_points() is False,
          "the overlay is suppressed on the PARAMETER plane -- critical points are dynamical-"
          "plane objects, per P5c's own spec")
    overlay_pane.set_render_mode("parameter_greens")
    check(overlay_view._should_draw_critical_points() is False,
          "also suppressed on parameter_greens (the other parameter-plane mode)")
    overlay_pane.set_render_mode("basin")
    check(overlay_view._should_draw_critical_points() is True,
          "shown again on basin mode -- basin pixels ARE dynamical-plane initial conditions")
    overlay_pane.set_render_mode("julia")

    overlay_view.refresh_critical_points()
    check(0j in overlay_view._critical_points
         and any(math.isinf(z.real) for z in overlay_view._critical_points),
          "mandelbrot()'s critical points are 0 and infinity, matching cdx's own "
          "distinct_critical_points for this map")
    check(overlay_view._orbit_traces is None,
          "orbit traces are NOT computed until tracing is actually turned on")

    key_before = overlay_view._critical_points_key
    points_before = overlay_view._critical_points   # SAME list object if truly memoized
    overlay_pane.viewport = cdx.Viewport(complex(0.4, 0.4), 0.2, 41)   # pan/zoom only
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

    # ---- fixed-point / attracting-cycle layers (Stage 2) ---------------------------
    print("\nfixed-point layer:")
    fp_session = Session()
    fp_session.map = cdx.RationalMap.mandelbrot()
    fp_session.param = -1 + 0j   # the basilica: 2 finite (repelling) fixed points + infinity
    fp_pane, fp_view = _pane_view(fp_session, render_mode="julia")
    fp_facts = fp_session.dynamical_facts()

    check(fp_view._layer_enabled[FIXED_POINTS_LAYER_KEY] is False,
          "the fixed-points layer is off by default")
    check(_OVERLAY_LAYERS_BY_KEY[FIXED_POINTS_LAYER_KEY].is_visible_for("julia") is True
          and _OVERLAY_LAYERS_BY_KEY[FIXED_POINTS_LAYER_KEY].is_visible_for("parameter") is False
          and _OVERLAY_LAYERS_BY_KEY[FIXED_POINTS_LAYER_KEY].is_visible_for("parameter_greens")
              is False,
          "fixed points are dynamical-plane gated, suppressed on both parameter-plane modes")

    fp_resolved = _fixed_points_provider(fp_session.map, fp_session.param, fp_facts)
    finite_fixed = [fp for fp in fp_facts.fixed_points if not _is_inf(fp.point)]
    check(len(fp_resolved) == len(finite_fixed) == 2,
          "plotted fixed points are exactly the FINITE entries of facts.fixed_points -- the "
          "same source the Facts tab's own table reads, infinity excluded")
    check({pt for pt, _colour in fp_resolved} == {fp.point for fp in finite_fixed},
          "the plotted point SET matches facts.fixed_points' own finite points exactly")
    check(not any(math.isinf(pt.real) or math.isinf(pt.imag) for pt, _colour in fp_resolved),
          "infinity itself never appears among the plotted points")
    for pt, colour in fp_resolved:
        fp = next(fp for fp in finite_fixed if fp.point == pt)
        check(colour == _FIXED_POINT_CLASSIFICATION_COLOURS[_classify(fp.multiplier)],
              f"fixed point at {pt} is coloured by _classify(multiplier) == "
              f"{_classify(fp.multiplier)!r}, matching the Facts tab's own classification")

    # Map-independent: every _classify output has a defined colour, and the
    # palette is genuinely 3 DISTINCT colours (attracting/repelling/neutral)
    # even though _classify itself returns 4 labels -- superattracting
    # shares attracting's colour (see _FIXED_POINT_CLASSIFICATION_COLOURS'
    # own module comment), not a silent KeyError for the common case of a
    # multiplier-0 fixed point (e.g. every root of Newton's method).
    check(set(_FIXED_POINT_CLASSIFICATION_COLOURS) == {"attracting", "superattracting",
                                                        "repelling", "neutral"},
          "every possible _classify(...) output has a defined overlay colour")
    check(_FIXED_POINT_CLASSIFICATION_COLOURS["attracting"]
          == _FIXED_POINT_CLASSIFICATION_COLOURS["superattracting"],
          "superattracting shares attracting's colour, keeping the palette to 3 visually "
          "distinct colours as specified, not 4")
    check(len({_FIXED_POINT_CLASSIFICATION_COLOURS["attracting"],
              _FIXED_POINT_CLASSIFICATION_COLOURS["repelling"],
              _FIXED_POINT_CLASSIFICATION_COLOURS["neutral"]}) == 3,
          "attracting/repelling/neutral are 3 genuinely distinct colours, not two of them "
          "coinciding by accident")

    print("\nattracting-cycle layer:")
    check(fp_view._layer_enabled[ATTRACTING_CYCLES_LAYER_KEY] is False,
          "the attracting-cycles layer is off by default too, independent of fixed points")
    check(_OVERLAY_LAYERS_BY_KEY[ATTRACTING_CYCLES_LAYER_KEY].is_visible_for("julia") is True
          and _OVERLAY_LAYERS_BY_KEY[ATTRACTING_CYCLES_LAYER_KEY].is_visible_for("parameter")
              is False,
          "attracting cycles are dynamical-plane gated too")

    cyc_resolved = _attracting_cycles_provider(fp_session.map, fp_session.param, fp_facts)
    period2 = next(c for c in fp_facts.attracting_cycles if c.period == 2)
    check({pt for pt, _colour in cyc_resolved} >= {0j, -1 + 0j},
          "plotted cycle points include the basilica's own period-2 cycle {0, -1}, the same "
          "source the Facts tab's own Attracting Cycles table reads")
    cyc_colours = {colour for pt, colour in cyc_resolved if pt in (0j, -1 + 0j)}
    check(len(cyc_colours) == 1,
          "every point belonging to the SAME cycle shares that cycle's one colour")

    # Map-independent: the strength gradient itself, at its two defined ends
    # and part-way between.
    check(_cycle_colour(0j) == _CYCLE_STRONG_COLOUR,
          "a superattracting cycle (|multiplier| == 0) gets the gradient's STRONG end exactly")
    check(_cycle_colour(complex(0.999, 0)) != _CYCLE_STRONG_COLOUR,
          "a barely-attracting cycle (|multiplier| near 1) is visually distinct from a "
          "superattracting one -- the gradient actually varies, not a flat colour")
    mid_colour = _cycle_colour(complex(0.5, 0))
    check(mid_colour != _CYCLE_STRONG_COLOUR and mid_colour != _CYCLE_WEAK_COLOUR,
          "a mid-strength cycle (|multiplier| == 0.5) lands strictly BETWEEN the two ends, "
          "not snapped to either one")
    check(_cycle_colour(complex(5, 0)) == _cycle_colour(complex(1, 0)),
          "|multiplier| is clamped at 1 -- an (unexpected, since attracting implies < 1) "
          "out-of-range value doesn't extrapolate past the gradient's own weak end")

    cyc_traces = _attracting_cycles_trace_provider(fp_session.map, fp_session.param, fp_facts)
    period2_trace = next((path, colour) for path, colour in cyc_traces
                         if set(path) == {0j, -1 + 0j})
    check(period2_trace[0] == list(period2.points) + [period2.points[0]],
          "a period-2 cycle's traced path is its 2 (ordered) points closed into a loop -- "
          "append points[0] -- not a re-sorted or independently-derived path")
    check(period2_trace[1] == cyc_colours.pop(),
          "...drawn in the SAME colour as that cycle's own points, not a separately computed one")

    print("\noverlay-layer registry: independence + trace gating:")
    fp_view.set_layer_enabled(FIXED_POINTS_LAYER_KEY, True)
    check(fp_view._layer_enabled[ATTRACTING_CYCLES_LAYER_KEY] is False,
          "enabling fixed points does not also enable attracting cycles -- independent flags")
    fp_view.set_trace_enabled(CYCLE_TRACE_KEY, True)
    fp_view.refresh_layers()
    check(fp_view._layer_traces[CYCLE_TRACE_KEY] is not None,
          "sanity: turning the cycle-trace toggle on does compute SOMETHING once refreshed")

    # Trace Cycle Paths must draw NOTHING while its own layer (Attracting
    # Cycles) is off -- paint_registry_layers' own layer_enabled gate, not
    # just the View-menu action's setEnabled (a UI convenience, not the
    # actual enforcement -- see SandboxWindow._build_view_menu_layer_actions).
    # Explicit layer_enabled/layer_trace_enabled dicts throughout (never
    # fp_view's own, already-mutated-above, live state) so each of the
    # three images below draws EXACTLY what its own name claims.
    def _paint_probe(layer_enabled: dict, layer_trace_enabled: dict) -> bytes:
        image = QImage(64, 64, QImage.Format.Format_RGB888)
        image.fill(0)
        painter = QPainter(image)
        paint_registry_layers(painter, fp_pane.viewport, 64, 64, "julia",
                              layer_enabled=layer_enabled, layer_trace_enabled=layer_trace_enabled,
                              layer_points=fp_view._layer_points, layer_traces=fp_view._layer_traces)
        painter.end()
        return image.constBits().tobytes()

    nothing_enabled_bytes = _paint_probe(
        {FIXED_POINTS_LAYER_KEY: False, ATTRACTING_CYCLES_LAYER_KEY: False}, {})
    fixed_only_bytes = _paint_probe(
        {FIXED_POINTS_LAYER_KEY: True, ATTRACTING_CYCLES_LAYER_KEY: False}, {})
    check(fixed_only_bytes != nothing_enabled_bytes,
          "sanity: enabling just fixed points DOES change the rendered bytes vs. nothing enabled")

    cycles_off_trace_on_bytes = _paint_probe(
        {FIXED_POINTS_LAYER_KEY: False, ATTRACTING_CYCLES_LAYER_KEY: False},
        {CYCLE_TRACE_KEY: True})   # the trace's OWN flag is on; its layer is not
    check(cycles_off_trace_on_bytes == nothing_enabled_bytes,
          "Trace Cycle Paths draws NOTHING while Attracting Cycles itself is off, even though "
          "the trace's own enabled flag is True -- it is genuinely gated behind the layer, not "
          "just disabled-looking in the menu")

    # ---- overlay legend (Stage 3): entry-list builder as a pure function -----------
    print("\noverlay legend:")
    check(fp_view._show_legend is False, "the legend is off by default")

    no_overlays = build_legend_entries(
        "julia", layer_enabled={}, layer_trace_enabled={},
        show_orbit=False, orbit_seeded=False, show_param_marker=False, param_set=False)
    check(no_overlays == [], "nothing enabled/visible -> an empty entry list, not a placeholder")

    fixed_and_cycles = build_legend_entries(
        "julia",
        layer_enabled={FIXED_POINTS_LAYER_KEY: True, ATTRACTING_CYCLES_LAYER_KEY: True},
        layer_trace_enabled={CYCLE_TRACE_KEY: True},
        show_orbit=False, orbit_seeded=False, show_param_marker=False, param_set=False)
    check([e.label for e in fixed_and_cycles] ==
          ["Fixed points: attracting / repelling / neutral",
           "Attracting cycles: coloured by strength", "Cycle paths"],
          "enabled+visible layers contribute their OWN legend_label, in registry order, "
          "with a trace's own sub-entry right after its layer's -- built from OVERLAY_LAYERS, "
          "not a hand-written per-overlay list")
    check(fixed_and_cycles[0].swatches ==
          (_FIXED_POINT_CLASSIFICATION_COLOURS["attracting"],
           _FIXED_POINT_CLASSIFICATION_COLOURS["repelling"],
           _FIXED_POINT_CLASSIFICATION_COLOURS["neutral"]),
          "fixed points' entry carries all three classification swatches")
    check(len(fixed_and_cycles[1].swatches) == 2,
          "attracting cycles' entry carries the strength gradient's two ends as swatches")
    check(fixed_and_cycles[2].swatches == (),
          "a trace's own entry (a line style, not a marker) carries no point swatches")

    cycles_no_trace = build_legend_entries(
        "julia", layer_enabled={ATTRACTING_CYCLES_LAYER_KEY: True},
        layer_trace_enabled={CYCLE_TRACE_KEY: False},   # layer on, its trace off
        show_orbit=False, orbit_seeded=False, show_param_marker=False, param_set=False)
    check([e.label for e in cycles_no_trace] == ["Attracting cycles: coloured by strength"],
          "a layer's trace sub-entry is independently gated -- present only when the trace "
          "itself is enabled too, same as what actually gets DRAWN")

    suppressed_on_parameter = build_legend_entries(
        "parameter",
        layer_enabled={FIXED_POINTS_LAYER_KEY: True, ATTRACTING_CYCLES_LAYER_KEY: True,
                      CRITICAL_POINTS_LAYER_KEY: True},
        layer_trace_enabled={},
        show_orbit=False, orbit_seeded=False, show_param_marker=False, param_set=False)
    check(suppressed_on_parameter == [],
          "every dynamical-plane-gated layer's entry disappears on the PARAMETER plane, even "
          "while enabled -- only entries for overlays actually shown in the CURRENT mode appear")

    orbit_entry = build_legend_entries(
        "julia", layer_enabled={}, layer_trace_enabled={},
        show_orbit=True, orbit_seeded=True, show_param_marker=False, param_set=False)
    check([e.label for e in orbit_entry] == ["Seeded orbit"],
          "the seeded orbit -- one of the two overlays OUTSIDE the registry -- gets its own "
          "entry when it's actually seeded and shown")
    check(build_legend_entries("julia", layer_enabled={}, layer_trace_enabled={},
                               show_orbit=True, orbit_seeded=False,
                               show_param_marker=False, param_set=False) == [],
          "...but not when show_orbit is on with nothing actually seeded yet")
    check(build_legend_entries("parameter", layer_enabled={}, layer_trace_enabled={},
                               show_orbit=True, orbit_seeded=True,
                               show_param_marker=False, param_set=False) == [],
          "...nor on the parameter plane, where orbit tracking has no meaning at all")

    marker_entry = build_legend_entries(
        "parameter", layer_enabled={}, layer_trace_enabled={},
        show_orbit=False, orbit_seeded=False, show_param_marker=True, param_set=True)
    check([e.label for e in marker_entry] == ["Parameter marker"],
          "the parameter marker -- the OTHER overlay outside the registry -- gets its own "
          "entry on the parameter plane when it's actually shown")
    check(build_legend_entries("julia", layer_enabled={}, layer_trace_enabled={},
                               show_orbit=False, orbit_seeded=False,
                               show_param_marker=True, param_set=True) == [],
          "...but not on a DYNAMICAL pane, where there is no parameter marker to explain")

    # ---- Legend menu action reaches the real per-pane flag -------------------------
    fp_view.set_show_legend(True)
    check(fp_view._show_legend is True, "set_show_legend reaches the flag directly")
    fp_view.set_show_legend(False)
    check(fp_view._show_legend is False, "...and back off")

    # ---- orbit tracking: click-to-seed, step/clear, painting, staleness -----------
    print("\norbit tracking:")
    from app.orbit_panel import OrbitPanel

    orbit_session = Session()
    orbit_session.map = cdx.RationalMap.mandelbrot()
    orbit_session.param = -1 + 0j   # the basilica again
    orbit_pane, orbit_view = _pane_view(orbit_session, complex(0, 0), 2.0, 400, "julia")
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
    orbit_pane.set_render_mode("parameter")
    orbit_view._seed_orbit_at(center_pixel)
    check(orbit_view.orbit_tracker.state is None,
          "clicking on the PARAMETER plane does not seed an orbit")
    orbit_pane.set_render_mode("julia")

    # Staleness: survives a pan/zoom, clears on a map/param change -- the
    # SAME property already verified directly against OrbitTracker itself
    # in app/test_orbit_tracker.py; here we confirm ImageView's own
    # refresh_orbit_staleness wiring actually reaches it.
    orbit_view._seed_orbit_at(center_pixel)
    check(orbit_view.orbit_tracker.state is not None, "sanity: an orbit exists again")
    orbit_pane.viewport = cdx.Viewport(complex(0.3, 0.3), 1.0, 400)   # pan/zoom only
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
    gesture_pane, gesture_view = _pane_view(orbit_session2, complex(0, 0), 2.0, 400, "julia")
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

    def _key_event(key, press: bool = True, autorepeat: bool = False) -> QKeyEvent:
        """A REAL QKeyEvent (not a hand-rolled stand-in like _FakeMouseEvent
        above): ImageView.keyPressEvent/keyReleaseEvent, unlike the mouse
        handlers, legitimately fall through to super().keyPressEvent(event)
        for an event they don't handle (e.g. an arrow key in a DYNAMICAL
        pane) -- PySide6's C++ binding rejects a duck-typed Python stand-in
        there (it type-checks the argument against QKeyEvent), so this
        needs to be constructed as one for real, still delivered directly
        (bypassing Qt's own event queue/focus system) the same way
        _FakeMouseEvent's events are, for the same offscreen-reliability
        reason.
        """
        kind = QEvent.Type.KeyPress if press else QEvent.Type.KeyRelease
        return QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier, autorep=autorepeat)

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
    check(close(gesture_view.pane.viewport.center, complex(0, 0), 0.7),
          "sanity: the drag actually did rubber-band zoom (viewport changed)")

    # ---- orbit trace: drawable_polyline_segments (pure geometry, no painter needed) --
    print("\ndrawable_polyline_segments (segment-break fix, pure geometry):")
    rect = QRectF(0, 0, 400, 400)   # already "inflated" for this test's own purposes
    p0, p1, p2, p3 = QPointF(10, 10), QPointF(20, 20), QPointF(30, 30), QPointF(40, 40)
    huge = QPointF(1e7, 1e7)   # a huge-but-finite point's mapped pixel -- far outside rect

    segs = drawable_polyline_segments([complex(0, 0), complex(1, 1), complex(2, 2)],
                                      [p0, p1, p2], rect)
    check(segs == [(p0, p1), (p1, p2)],
          "an all-finite, all-on-screen run draws every consecutive pair")

    # A non-finite point in the middle: must NOT bridge across it to connect its finite
    # neighbours -- the FILTER-AND-BRIDGE bug this whole fix exists to remove.
    points_nonfinite = [complex(0, 0), complex(math.inf, 0.0), complex(2, 2)]
    segs_nonfinite = drawable_polyline_segments(points_nonfinite, [p0, p1, p2], rect)
    check(segs_nonfinite == [],
          "a non-finite point breaks BOTH adjacent pairs -- neither survives, and the two "
          "finite neighbours either side of it are NOT bridged across the gap")

    # A huge-but-finite point (an escaping orbit before it overflows, e.g. |z|~1e300):
    # must not sweep a line across the whole pane to or from it -- the OFF-SCREEN SWEEP bug.
    points_huge = [complex(0, 0), complex(1e300, 1e300), complex(2, 2)]
    segs_huge = drawable_polyline_segments(points_huge, [p0, huge, p2], rect)
    check(segs_huge == [],
          "a huge-but-finite off-screen point draws no segment to OR from it -- no sweep "
          "across the pane, and its finite on-screen neighbours are not bridged across it")

    # Both defects in the SAME sequence at once, exactly the combined case the task asks
    # for: a NaN AND a huge-but-finite point.
    points_combined = [complex(0, 0), complex(1, 1), complex(math.nan, 0.0),
                       complex(1e300, 1e300), complex(3, 3)]
    pixels_combined = [p0, p1, p1, huge, p3]
    segs_combined = drawable_polyline_segments(points_combined, pixels_combined, rect)
    check(segs_combined == [(p0, p1)],
          "with a NaN AND a huge-but-finite point both present, only the one genuinely "
          "finite-and-on-screen consecutive pair survives -- neither defect bridges across "
          "the other, and no drawn segment leaves the inflated rect")
    check(all(rect.contains(a) and rect.contains(b) for a, b in segs_combined),
          "sanity: every segment this function returns stays fully within the given rect")

    # ---- orbit trace: connect-lines toggle, via the testable _orbit_line_segments ----
    print("\n_orbit_line_segments (connect-lines toggle):")
    toggle_session = Session()
    toggle_session.map = cdx.RationalMap.mandelbrot()
    toggle_session.param = -1 + 0j   # the basilica
    toggle_pane, toggle_view = _pane_view(toggle_session, complex(0, 0), 2.0, 400, "julia")
    toggle_view.resize(400, 400)
    toggle_view.orbit_tracker.seed(toggle_session.map, toggle_session.param, complex(0.3, 0.2))
    toggle_view.step_orbit(3)

    check(toggle_view._orbit_connect_lines is True, "sanity: the toggle defaults on")
    check(len(toggle_view._orbit_line_segments()) > 0,
          "sanity: with the toggle on, the seeded+stepped orbit has at least one drawable "
          "segment")
    history_before = list(toggle_view.orbit_tracker.state.history)

    toggle_view.set_orbit_connect_lines(False)
    check(toggle_view._orbit_line_segments() == [],
          "with the toggle off, _orbit_line_segments returns zero segments")
    check(list(toggle_view.orbit_tracker.state.history) == history_before,
          "...but the underlying orbit history -- what the dots and seed marker are drawn "
          "from -- is completely unaffected by the toggle")

    toggle_view.set_orbit_connect_lines(True)
    check(len(toggle_view._orbit_line_segments()) > 0, "re-enabling restores the segments")

    # Orbit-only: must never affect the SEPARATE post-critical-point traces.
    toggle_view._trace_orbits = True
    toggle_view.refresh_critical_points()
    toggle_view.set_orbit_connect_lines(False)
    check(toggle_view._trace_orbits is True,
          "the connect-lines toggle does not touch the independent trace-orbits flag -- "
          "it is orbit-only, per its own docstring")

    # ---- cursor readout: coordinate precision + mode-dependent sampled value ------
    print("\ncursor readout:")
    readout_session = Session()
    readout_session.map = cdx.RationalMap.mandelbrot()
    readout_session.param = -1 + 0j
    readout_pane, readout_view = _pane_view(readout_session, complex(0, 0), 1.5, 41, "julia")
    readout_view.resize(400, 400)

    check(readout_view.cursor_readout_text(QPoint(200, 200)) == "z = 0.0000+0.0000j",
          "with no buffer rendered yet, the readout is JUST the coordinate -- no sampled value")

    real_array = render_map(readout_session.map, readout_session.param, readout_pane.viewport,
                            readout_session.render_settings, "julia")
    readout_view.set_image(real_array, readout_pane.viewport)
    center_text = readout_view.cursor_readout_text(QPoint(200, 200))
    check("never escaped" in center_text,
          "the basilica's own critical point (screen center here) never escapes -- sampled "
          "correctly through the REAL render pipeline, not a synthetic array")
    corner_text = readout_view.cursor_readout_text(QPoint(2, 2))
    check("escape = " in corner_text,
          "a corner pixel (far from the filled Julia set) samples a real escape value")

    # Precision scales with zoom depth -- four decimals is useless at
    # scale 1e-9 (P5c's own wording), checked directly, not assumed from
    # reading the formula.
    readout_pane.viewport = cdx.Viewport(complex(0, 0), 1.5, 41)
    normal_text = readout_view.cursor_readout_text(QPoint(200, 200))
    normal_decimals = len(normal_text.split("z = ")[1].split("+")[0].split(".")[1])
    check(normal_decimals == 4, "at an ordinary zoom level, the coordinate shows 4 decimals")

    readout_pane.viewport = cdx.Viewport(complex(0, 0), 1e-9, 41)
    deep_text = readout_view.cursor_readout_text(QPoint(200, 200))
    deep_decimals = len(deep_text.split("z = ")[1].split("+")[0].split(".")[1])
    check(deep_decimals > normal_decimals,
          "at scale 1e-9, the coordinate shows MORE decimals than at ordinary zoom -- enough "
          "to actually distinguish adjacent pixels, not a fixed 4 regardless of depth")

    two_adjacent_pixels = [readout_view._pixel_to_complex(QPoint(200, 200)),
                          readout_view._pixel_to_complex(QPoint(201, 200))]
    check(two_adjacent_pixels[0] != two_adjacent_pixels[1],
          "sanity: two adjacent screen pixels really do map to different complex values here")

    # ---- cursor readout: mode-dependent second field (basin, greens) --------------
    print("\ncursor readout (basin / greens sampling):")
    readout_pane.viewport = cdx.Viewport(complex(0, 0), 2.0, 41)
    readout_pane.set_render_mode("basin")
    basin_array = render_map(readout_session.map, readout_session.param, readout_pane.viewport,
                             readout_session.render_settings, "basin")
    readout_view.set_image(basin_array, readout_pane.viewport)
    basin_text = readout_view.cursor_readout_text(QPoint(200, 200))
    check("basin = " in basin_text or "unresolved" in basin_text,
          "basin mode's cursor readout samples the label layer, not the escape-time format")

    readout_pane.set_render_mode("greens")
    greens_array = render_map(readout_session.map, readout_session.param, readout_pane.viewport,
                              readout_session.render_settings, "greens")
    readout_view.set_image(greens_array, readout_pane.viewport)
    greens_text = readout_view.cursor_readout_text(QPoint(200, 200))
    check("potential = " in greens_text,
          "greens mode's cursor readout samples the potential, not escape/basin formatting")

    readout_pane.set_render_mode("parameter_basin")
    pbasin_array = render_map(readout_session.map, readout_session.param, readout_pane.viewport,
                              readout_session.render_settings, "parameter_basin")
    readout_view.set_image(pbasin_array, readout_pane.viewport)
    pbasin_text = readout_view.cursor_readout_text(QPoint(200, 200))
    check("attracting cycle" in pbasin_text,
          "parameter_basin mode's cursor readout reports the attracting-cycle count, not "
          "escape/basin/potential formatting")

    # Stage 2: a RATIONAL map's "julia" render is stacked too -- the cursor
    # readout must sample it the SAME way basin's own stacked array is
    # sampled, not crash trying to unpack a 3D array's .shape into (h, w).
    readout_pane.set_render_mode("julia")
    readout_session.map = cdx.RationalMap.newton_cubic()   # has a pole -- takes the rational path
    readout_session.param = 0j
    rational_julia_array = render_map(readout_session.map, readout_session.param,
                                      readout_pane.viewport, readout_session.render_settings,
                                      "julia")
    check(rational_julia_array.ndim == 3,
          "sanity: newton_cubic's own julia render really is stacked (not certified)")
    readout_view.set_image(rational_julia_array, readout_pane.viewport)
    rational_text = readout_view.cursor_readout_text(QPoint(200, 200))
    check("basin = " in rational_text or "unresolved" in rational_text,
          "a rational map's julia cursor readout samples the LABEL layer, formatted like "
          "basin's own -- not the escape-time format, and no crash on the stacked shape")
    readout_session.map = cdx.RationalMap.mandelbrot()   # restore for later sections
    readout_session.param = -1 + 0j

    # ---- cursor readout: leaveEvent clears it --------------------------------------
    print("\ncursor readout (leaveEvent):")
    cleared = []
    readout_view.cursor_readout_changed.connect(cleared.append)
    readout_view.leaveEvent(None)
    check(cleared == [""], "the mouse leaving the widget emits an EMPTY readout, clearing it")

    # ---- cursor-anchored zoom formula --------------------------------------------
    print("\ncursor-anchored zoom:")
    session2 = Session()
    pane2, view2 = _pane_view(session2, complex(0.2, -0.3), 1.0, 400)
    view2.resize(400, 400)

    cursor = QPoint(300, 120)   # off-center, deliberately
    w_before = view2._pixel_to_complex(cursor)

    vp = pane2.viewport
    factor = 1.15 ** 3   # simulate 3 notches, matching wheelEvent's math directly
    new_center = w_before - (w_before - vp.center) / factor
    new_scale = vp.scale / factor
    pane2.viewport = cdx.Viewport(new_center, new_scale, vp.resolution)

    w_after = view2._pixel_to_complex(cursor)
    check(close(w_before, w_after, 1e-9),
          "the complex point under the cursor is unchanged by a cursor-anchored zoom")
    check(pane2.viewport.scale < 1.0, "zooming in (factor>1) shrinks scale")

    # ---- pan anchor invariant ------------------------------------------------------
    print("\npan anchor invariant:")
    session3 = Session()
    pane3, view3 = _pane_view(session3, complex(0, 0), 1.5, 400)
    view3.resize(400, 400)

    anchor_pixel = QPoint(150, 100)
    anchor_complex = view3._pixel_to_complex(anchor_pixel)
    drag_to_pixel = QPoint(250, 260)   # dragged elsewhere

    vp3 = pane3.viewport
    current_point = view3._pixel_to_complex(drag_to_pixel)
    new_center = vp3.center + (anchor_complex - current_point)
    pane3.viewport = cdx.Viewport(new_center, vp3.scale, vp3.resolution)

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
    pane5, view5 = _pane_view(session5, complex(0.3, -0.2), 1.0, 400)
    view5.resize(400, 400)

    overscan_factor = 1.3
    buffer_res = round(400 * overscan_factor)   # 520, exactly -- no rounding noise
    buffer_viewport = cdx.Viewport(pane5.viewport.center,
                                   pane5.viewport.scale * overscan_factor, buffer_res)
    view5.set_image(np.zeros((buffer_res, buffer_res)), buffer_viewport)

    # Cross-check against a fully independent hand computation -- not
    # _pixel_to_complex/_complex_to_buffer_pixel called and trusted, but the
    # same formulas worked out by hand here -- for an arbitrary INTERIOR
    # probe pixel (not a corner), confirming the two-corner source rect
    # really does determine every point, as the affine argument in
    # ImageView's docstring claims.
    probe_pixel = QPoint(310, 120)
    disp_vp = pane5.viewport
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

    pane5.viewport = cdx.Viewport(disp_vp.center, disp_vp.scale * 1.25, disp_vp.resolution)
    check(view5.buffer_edge_fraction() > 1.0 / overscan_factor,
          "zooming out grows the buffer edge fraction")

    _fresh_pane, fresh_view = _pane_view(Session())
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
    check(window.mode_combo.currentText() == window.pane.render_mode,
          "the combo box starts on the session's actual startup render mode")

    # ---- Stage 2: dual-pane splitter, focus, coupled toggle -----------------------
    print("\ndual-pane splitter / focus / coupled toggle:")
    ok2 = wait_for(lambda: window.image_view2._pixmap is not None, timeout_ms=10000)
    check(ok2, "the second pane's initial render also completes, independently of the first")

    check(window.pane.render_mode == "parameter" and window.pane2.render_mode == "julia",
          "the two panes start on different planes -- a natural parameter+dynamical pairing, "
          "not two copies of the same view")
    check(window.mode_combo2.currentText() == window.pane2.render_mode,
          "the second pane has its OWN mode combo, starting on ITS pane's mode")
    check(window.mode_combo is not window.mode_combo2,
          "the two mode combos are genuinely separate widgets, not one shared between panes")

    check(window._focused_pane is window.pane,
          "focus starts on pane A, matching today's single-view startup")
    check(window.coupled_view_action.isChecked(), "coupled view is the default -- both panes visible")

    window.show()   # the visibility checks below need a real, shown top-level window
    check(window.pane_column.isVisible() and window.pane_column2.isVisible(),
          "coupled: both pane columns are visible")

    # Clicking pane2's image -- not setting focus programmatically -- is
    # what actually moves focus in the real app (see ImageView.pane_activated,
    # emitted from mousePressEvent).
    window.image_view2.mousePressEvent(_FakeMouseEvent(QPoint(50, 50)))
    window.image_view2.mouseReleaseEvent(_FakeMouseEvent(QPoint(50, 50)))
    check(window._focused_pane is window.pane2,
          "pressing on pane2's image moves focus to pane2, via ImageView.pane_activated")
    check(window.pane_column2.styleSheet() != "" and window.pane_column.styleSheet() == "",
          "the focus-highlight border moves to the newly-focused pane's column")

    window.coupled_view_action.setChecked(False)
    check(window.pane_column2.isVisible() and not window.pane_column.isVisible(),
          "single-view collapses to just the FOCUSED pane -- pane2 here, not always pane A")
    window.coupled_view_action.setChecked(True)
    check(window.pane_column.isVisible() and window.pane_column2.isVisible(),
          "re-checking coupled shows both panes again")

    # Per-pane mode combo: switching pane2's mode does not touch pane A's.
    pane_mode_before = window.pane.render_mode
    window.mode_combo2.setCurrentText("basin")
    check(window.pane2.render_mode == "basin" and window.pane.render_mode == pane_mode_before,
          "switching pane2's OWN mode combo changes only pane2's render_mode")
    window.mode_combo2.setCurrentText("julia")   # restore for the orbit-strip check below

    # Orbit strip follows whichever pane is currently dynamical (focus
    # is still pane2 from the click above).
    check(window.orbit_panel.image_view is window.pane2.image_view,
          "the orbit strip follows pane2 -- the only dynamical pane right now (pane A is "
          "still on the parameter plane)")
    window.mode_combo.setCurrentText("julia")
    check(window.orbit_panel.isEnabled(), "with both panes dynamical, the orbit strip stays enabled")
    window.mode_combo2.setCurrentText("parameter")
    check(window.orbit_panel.image_view is window.pane.image_view,
          "with pane2 back on the parameter plane, the orbit strip follows pane A instead")
    window.mode_combo.setCurrentText("parameter")
    check(not window.orbit_panel.isEnabled(),
          "with NEITHER pane dynamical, the orbit strip is disabled rather than showing stale "
          "controls for a plane that has no orbit concept")

    # Per-pane debounce isolation: a viewport-changed event for one pane
    # must never touch the OTHER pane's request id / pending tasks.
    window.mode_combo.setCurrentText("julia")
    window.mode_combo2.setCurrentText("julia")
    wait_for(lambda: window.pane.request_id not in window.pane.pending_tasks, timeout_ms=10000)
    wait_for(lambda: window.pane2.request_id not in window.pane2.pending_tasks, timeout_ms=10000)
    pane2_request_id_before = window.pane2.request_id
    window._on_viewport_changed(window.pane)
    check(window.pane2.request_id == pane2_request_id_before,
          "a viewport-changed event for pane A never touches pane2's request id / debounce")

    # Restore both panes and focus to their startup defaults -- later
    # sections (e.g. "mode selector" below) assume pane A starts on
    # "parameter", the state this whole block found it in.
    window.mode_combo.setCurrentText("parameter")
    window.mode_combo2.setCurrentText("julia")
    window._set_focused_pane(window.pane)

    # ---- window title / About dialog -----------------------------------------------
    print("\nwindow title / About dialog:")
    check(window.windowTitle() == "ComplexDynamics",
          "the window title is the product name, not an internal milestone tag")
    menu_titles = [a.text() for a in window.menuBar().actions()]
    check("Help" in menu_titles, "a Help menu exists (macOS moves it into the app menu)")
    # window.help_menu directly, matching how SandboxWindow itself accesses
    # it (see _build_ui's own comment on why it's stored on self at all) --
    # NOT re-fetched via menuBar().actions()[i].menu(), which hits a real
    # PySide6/shiboken wrapper-identity quirk after the event loop has run
    # a few iterations (confirmed directly: menuBar().actions()[i] hands
    # back a wrapper whose .menu() call raises "already deleted" even
    # though the SAME menu is perfectly valid via a direct reference or
    # findChildren) -- a quirk of that particular lookup path, not evidence
    # the menu itself is actually gone.
    about_actions = [a.text() for a in window.help_menu.actions()]
    check("About ComplexDynamics" in about_actions,
          "the Help menu has an 'About ComplexDynamics' action")
    check(window.about_action.text() == "About ComplexDynamics",
          "window.about_action (what _build_ui actually wires the click handler to) matches too")
    from app.about_dialog import AboutDialog
    about = AboutDialog(window)
    check(about.windowTitle() == "About ComplexDynamics", "the dialog's own title matches too")
    about.close()

    # ---- View menu: mirrors the toolbar's own controls, bidirectionally (Stage 3) ---
    print("\nView menu:")
    check(menu_titles == ["File", "View", "Help"],
          "the menu bar is File, View, Help, in that order")
    view_action_texts = [a.text() for a in window.view_menu.actions() if a.text()]
    check(view_action_texts == ["Reset View", "Coupled View", "Critical Points", "Trace Orbits",
                                "Fixed Points", "Attracting Cycles", "Trace Cycle Paths",
                                "Connect Orbit Points", "Legend"],
          "the View menu has the toolbar's own view controls PLUS every registry layer's own "
          "action (and gated trace sub-action), in registry order, PLUS Legend last (blank "
          "entries are separators, filtered out above) -- Fixed Points/Attracting Cycles/"
          "Trace Cycle Paths/Legend are all menu-only (Stage 2/3), with no toolbar counterpart "
          "to mirror")

    check(window.reset_view_action.isCheckable() is False,
          "Reset View is a plain triggerable action, not a checkable toggle")
    coupled_before = window.coupled_view_action.isChecked()
    window.reset_view_action.trigger()
    check(window.coupled_view_action.isChecked() == coupled_before,
          "triggering Reset View from the menu doesn't touch unrelated state")

    # Stage 4: the View menu is the SOLE control for each of these -- no
    # toolbar checkbox exists any more to mirror against, so what's actually
    # tested here is that each action reaches its OWN real behavior
    # directly, both ways (on then off), not just that two widgets agree
    # with each other.
    window.critical_points_action.setChecked(True)
    check(window.image_view._show_critical_points is True and
          window.image_view2._show_critical_points is True,
          "toggling the 'Critical Points' menu action ON reaches BOTH panes' real flag directly")
    window.critical_points_action.setChecked(False)
    check(window.image_view._show_critical_points is False and
          window.image_view2._show_critical_points is False,
          "...and OFF reaches it too")

    window.trace_orbits_action.setChecked(True)
    check(window.image_view._trace_orbits is True and window.image_view2._trace_orbits is True,
          "toggling the 'Trace Orbits' menu action reaches BOTH panes' real flag directly")
    window.trace_orbits_action.setChecked(False)

    orbit_connect_before = window.orbit_connect_lines_action.isChecked()
    window.orbit_connect_lines_action.setChecked(not orbit_connect_before)
    check(window.image_view._orbit_connect_lines == (not orbit_connect_before) and
          window.image_view2._orbit_connect_lines == (not orbit_connect_before),
          "toggling the 'Connect Orbit Points' menu action reaches BOTH panes' real flag directly")
    window.orbit_connect_lines_action.setChecked(orbit_connect_before)   # restore

    # Legend (Stage 3) is menu-only -- no toolbar checkbox to mirror through,
    # so its action is wired DIRECTLY to both ImageViews' set_show_legend
    # (see _build_ui's own comment on why it isn't built via
    # _build_view_menu_layer_actions).
    check(window.legend_action.isChecked() is False, "the Legend menu action starts unchecked")
    window.legend_action.setChecked(True)
    check(window.image_view._show_legend is True and window.image_view2._show_legend is True,
          "toggling the 'Legend' menu action reaches BOTH panes' real flag directly")
    window.legend_action.setChecked(False)
    check(window.image_view._show_legend is False and window.image_view2._show_legend is False,
          "...and back off")

    # coupled_view_action's own real-behavior reach (pane visibility via
    # _relayout_panes) is already covered end-to-end by the "dual-pane
    # splitter / focus / coupled toggle" section above -- nothing further
    # to add here now that there's no separate checkbox left to mirror.

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
    window.pane.set_render_mode("julia")   # Session/pane now start in "parameter"; this test needs julia
    window.pane.viewport = cdx.Viewport(complex(0, 0), 1.5, 1800)
    window.session.render_settings = cdx.RenderSettings(400, 2.0, 1e-6, 1)   # single-threaded, slow

    t0 = time.perf_counter()
    window._start_render(window.pane)
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

    # Cancellation CORRECTNESS, proven structurally rather than by wall clock:
    # an already-cancelled token makes the per-column render loop
    # (cdx/src/renderer.cpp parallel_columns) bail at column 0, so its result
    # differs from a complete render. Deterministic -- no background thread, no
    # sleep, no timing threshold -- so it cannot flake on a shared CI runner the
    # way a timed mid-render cancel does.
    complete = render_map(slow_map, slow_param, slow_viewport, slow_settings,
                          "julia", cdx.CancelToken(), None)
    check(complete.min() != complete.max(),
          "sanity: the complete render is non-uniform, so a partial one is "
          "distinguishable from it")
    pre_cancelled = cdx.CancelToken()
    pre_cancelled.cancel()
    bailed = render_map(slow_map, slow_param, slow_viewport, slow_settings,
                        "julia", pre_cancelled, None)
    check(not np.array_equal(bailed, complete),
          "an already-cancelled render bails out of the per-column loop early, "
          "returning a partial result that differs from the complete render "
          "-- cooperative cancellation, proven without wall-clock timing")

    # The mid-render, cross-thread LATENCY is a real property but only reliably
    # measurable on dev hardware; on a shared runner the background canceller's
    # 0.02s wake isn't promptly scheduled, so this is observational unless
    # explicitly enforced. The timing was still measured above.
    check_perf(cancelled_time < uncancelled_time * 0.5,
               f"a cancelled RenderTask.run() returns well under the uncancelled "
               f"time ({cancelled_time:.3f}s vs {uncancelled_time:.3f}s)")

    # Now the same scenario through the real dispatch path: starting a new
    # render immediately cancels the superseded one's token.
    window.pane.viewport = slow_viewport
    window._start_render(window.pane)
    stale_task = window.pane.pending_tasks[window.pane.request_id]
    window._start_render(window.pane)   # supersedes the task started above
    check(stale_task.cancel.is_cancelled,
          "starting a new render immediately cancels the superseded one's token")

    # Rapid-fire many supersessions (simulating a fast scroll burst, each
    # notch calling _start_render once the debounce settles) --
    # pending_tasks must not accumulate one stale entry per call.
    window.pane.viewport = cdx.Viewport(complex(0, 0), 1.5, 80)
    for _ in range(15):
        window._start_render(window.pane)
    ok = wait_for(lambda: len(window.pane.pending_tasks) <= 1, timeout_ms=10000)
    check(ok, "15 rapid supersessions leave at most the current task pending, not 15 stale ones")

    # closeEvent cancels and returns immediately rather than draining the
    # pool -- verified on a second window (the shared `window` above gets
    # closed at the very end of this script) so the rest of the test suite
    # can keep using it afterward.
    w2 = SandboxWindow()
    w2.session.map = cdx.RationalMap.mandelbrot()
    w2.session.param = complex(-0.7269, 0.1889)
    w2.pane.viewport = cdx.Viewport(complex(0, 0), 1.5, 1800)
    w2.session.render_settings = cdx.RenderSettings(400, 2.0, 1e-6, 1)
    w2._start_render(w2.pane)
    pending_before_close = list(w2.pane.pending_tasks.values())

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
    window.pane.viewport = cdx.Viewport(complex(0.1, 0.1), 1.8, 800)
    window.session.render_settings = cdx.RenderSettings(500, 2.0, 1e-6, 1)
    window._start_render(window.pane)
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
    window.pane.viewport = cdx.Viewport(complex(0.05, -0.05), 1.3, 60)
    window.session.render_settings = cdx.RenderSettings(60, 2.0, 1e-6, 1)

    window._start_render(window.pane)
    ok = wait_for(lambda: window.pane.request_id not in window.pane.pending_tasks, timeout_ms=10000)
    check(ok, "first render (cache cold for this viewport/settings) completes")
    stats_after_first = window.session.cache.stats

    window._start_render(window.pane)   # identical viewport/settings -- both stages should now hit
    ok = wait_for(lambda: window.pane.request_id not in window.pane.pending_tasks, timeout_ms=10000)
    check(ok, "second, identical render completes")
    stats_after_second = window.session.cache.stats
    check(stats_after_second.hits >= stats_after_first.hits + 2,
          "a repeat request at the same viewport/settings hits the cache for both the "
          "preview and full overscanned buffers, not just one")
    check(stats_after_second.misses == stats_after_first.misses,
          "the repeat request causes no new misses -- nothing new needed computing")

    # ---- superseded requests are discarded ------------------------------------------
    print("\nsuperseded render requests:")
    window.pane.viewport = cdx.Viewport(complex(0, 0), 1.5, 60)
    window._start_render(window.pane)
    stale_id = window.pane.request_id
    window._start_render(window.pane)   # supersedes the previous request before it can complete
    current_id = window.pane.request_id
    check(current_id != stale_id, "a new request gets a new id")

    window.image_view._pixmap = None
    window._on_full_ready(stale_id, np.zeros((10, 10)), cdx.Viewport())
    check(window.image_view._pixmap is None,
          "a result tagged with a superseded request id is discarded, not displayed")

    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "the current (non-superseded) request still completes normally")

    # ---- RenderSignals lifetime: GUI-thread release + deleteLater teardown ----------
    # This offscreen suite calls every handler directly, on this one thread
    # -- it CANNOT reproduce the actual cross-thread race Stage 1 exists
    # for (see RenderSignals' own docstring for the crash: a context-less
    # lambda connection let a worker thread run these handlers inline,
    # including the pending_tasks.pop() that drops the object's own last
    # reference, mid-emit, off the GUI thread). Calling _on_full_ready/
    # _on_render_failed directly here IS, from the object's own
    # perspective, indistinguishable from a genuinely-queued cross-thread
    # delivery landing on the GUI thread -- which is exactly the property
    # under test (release + teardown happen wherever this call runs, and
    # the fix's whole point is that a real cross-thread emission is now
    # QUEUED to land HERE rather than running inline on the worker
    # thread). The actual race -- rapid pan/zoom/mode-switch while renders
    # churn on a low-core machine -- needs real on-machine stress testing;
    # this suite cannot exercise a genuine cross-thread emission at all.
    print("\nRenderSignals lifetime (GUI-thread release, deleteLater teardown):")
    probe_id = 10_000_001   # a request_id no real render has ever used
    check(window._pane_for_render_id(probe_id) is None,
          "_pane_for_render_id returns None for a request_id no pane is tracking -- the "
          "'unknown/already-cleaned-up request' no-op path")

    probe_task = RenderTask(probe_id, window.session.map, window.session.param,
                            window.pane.viewport, window.session.render_settings, "julia",
                            cdx.CancelToken())
    window.pane.pending_tasks[probe_id] = probe_task
    window.pane.request_id = probe_id
    check(window._pane_for_render_id(probe_id) is window.pane,
          "_pane_for_render_id resolves a pending request id to the pane whose pending_tasks "
          "actually holds it")

    signals_ref = probe_task.signals
    check(shiboken6.isValid(signals_ref), "sanity: the signals object starts out valid")
    window._on_full_ready(probe_id, np.zeros((4, 4)), cdx.Viewport())
    check(probe_id not in window.pane.pending_tasks,
          "_on_full_ready releases the GUI side's own reference (pops pending_tasks) when "
          "its handler runs")
    check(shiboken6.isValid(signals_ref),
          "deleteLater() only SCHEDULES teardown -- the object is still valid immediately "
          "after the handler returns, not synchronously destroyed")
    QTest.qWait(20)   # a bare processEvents() call doesn't reliably flush a DeferredDelete
    check(not shiboken6.isValid(signals_ref),
          "...and is actually torn down once the event loop processes the deferred delete -- "
          "on THIS (the GUI) thread, never the worker thread")

    # Same release+teardown path for the failure signal.
    probe_id2 = 10_000_002
    probe_task2 = RenderTask(probe_id2, window.session.map, window.session.param,
                             window.pane.viewport, window.session.render_settings, "julia",
                             cdx.CancelToken())
    window.pane.pending_tasks[probe_id2] = probe_task2
    signals_ref2 = probe_task2.signals
    window._on_render_failed(probe_id2, "synthetic failure for this test")
    check(probe_id2 not in window.pane.pending_tasks,
          "_on_render_failed releases pending_tasks' reference too, not just _on_full_ready's")
    QTest.qWait(20)   # a bare processEvents() call doesn't reliably flush a DeferredDelete
    check(not shiboken6.isValid(signals_ref2), "...and its RenderSignals is torn down the same way")

    # A stale request id resolves to the RIGHT pane even when it belongs to
    # pane2, not pane -- the global id space (self._next_render_id) is what
    # makes plain pending_tasks-membership scanning unambiguous across panes.
    window.pane2.viewport = cdx.Viewport(complex(0, 0), 1.5, 40)
    window._start_render(window.pane2)
    pane2_request_id = window.pane2.request_id
    check(window._pane_for_render_id(pane2_request_id) is window.pane2,
          "_pane_for_render_id resolves pane2's own request id to pane2, not pane -- ids are "
          "globally unique across both panes' independent per-pane counters")
    wait_for(lambda: pane2_request_id not in window.pane2.pending_tasks, timeout_ms=10000)

    # Torn-down-widget guard: a result delivered for a pane whose ImageView is
    # already gone (e.g. mid-shutdown) is a no-op, not a crash.
    guard_session = Session()
    guard_session.map = cdx.RationalMap.mandelbrot()
    guard_pane, guard_view = _pane_view(guard_session, complex(0, 0), 1.5, 40, "julia")
    guard_id = 10_000_003
    guard_task = RenderTask(guard_id, guard_session.map, guard_session.param,
                            guard_pane.viewport, guard_session.render_settings, "julia",
                            cdx.CancelToken())
    guard_pane.pending_tasks[guard_id] = guard_task
    guard_pane.request_id = guard_id
    guard_view.deleteLater()
    QTest.qWait(20)   # a bare processEvents() call doesn't reliably flush a DeferredDelete
    check(not shiboken6.isValid(guard_view), "sanity: the pane's ImageView is now really torn down")
    original_panes = window.panes
    window.panes = [guard_pane]   # so _pane_for_render_id (scans self.panes) can find it
    try:
        window._on_full_ready(guard_id, np.zeros((4, 4)), cdx.Viewport())   # must not raise
        check(True, "_on_full_ready no-ops (doesn't crash) when the pane's widget is already "
                   "torn down, instead of touching a dead C++ object")
    except Exception as exc:
        check(False, f"_on_full_ready raised on a torn-down pane's result: {exc!r}")
    finally:
        window.panes = original_panes

    # ---- Settings dialog (Stage 5): menu-opened, not a tab -------------------------
    print("\nsettings dialog:")
    check(window.tabs.count() == 4 and window.tabs.tabText(0) == "View"
          and window.tabs.tabText(1) == "Terms" and window.tabs.tabText(2) == "Facts"
          and window.tabs.tabText(3) == "Library",
          "the window has View, Terms, Facts, and Library tabs, in that order -- Settings is "
          "no longer one of them")
    check(window.tabs.widget(1) is window.term_editor_panel,
          "the Terms tab holds the actual TermEditorPanel instance")
    check(window.tabs.widget(2) is window.facts_panel,
          "the Facts tab holds the actual FactsPanel instance")
    check(window.tabs.widget(3) is window.library_panel,
          "the Library tab holds the actual LibraryPanel instance")

    check(window.settings_action.isCheckable() is False,
          "'Settings...' is a plain triggerable action, opening a dialog, not a toggle")
    check(window.settings_action.menuRole() == QAction.MenuRole.PreferencesRole,
          "its MenuRole is PreferencesRole, so macOS relocates it into the app menu (Cmd+,)")
    check(window.settings_dialog.isModal() is False,
          "the dialog is NON-modal -- Apply, see the re-render, and adjust again all without "
          "closing it")
    check(window.settings_panel.parent() is window.settings_dialog,
          "the Settings dialog holds the actual SettingsPanel instance")
    check(not window.settings_dialog.isVisible(), "the dialog starts hidden, not shown at launch")

    window._on_open_settings()
    check(window.settings_dialog.isVisible(), "triggering 'Settings...' shows the dialog")
    reopened_dialog = window.settings_dialog
    window._on_open_settings()
    check(window.settings_dialog is reopened_dialog,
          "triggering it again re-shows the SAME dialog instance, not a second stacked one")

    window.pane.viewport = cdx.Viewport(complex(0, 0), 1.5, 60)
    window._start_render(window.pane)
    wait_for(lambda: window.pane.request_id not in window.pane.pending_tasks, timeout_ms=10000)

    new_resolution = 250   # above the resolution widget's own 200 floor -- see FIELD_SPECS
    window.settings_panel._widgets["resolution"].setValue(new_resolution)
    window.settings_panel._widgets["threads"].setValue(1)
    window.settings_panel._apply()

    check(window.pane.viewport.resolution == new_resolution,
          "Apply THROUGH THE DIALOG still updates the pane's viewport resolution")
    check(window.session.render_settings.threads == 1,
          "Apply updates the session's render_settings too, not just resolution")

    ok = wait_for(lambda: window.image_view._buffer_viewport is not None
                 and window.image_view._buffer_viewport.resolution == round(new_resolution * 1.3),
                 timeout_ms=10000)
    check(ok, "Apply triggers an immediate re-render at the NEW resolution -- not the debounce, "
          "and not the old one -- exactly as it did as a tab")

    # ---- mode selector: switching modes reaches the session and re-renders ------------
    print("\nmode selector:")
    check([window.mode_combo.itemText(i) for i in range(window.mode_combo.count())]
          == list(sandbox_module.RENDER_MODES),
          "the combo box offers exactly RENDER_MODES, in order")

    window.image_view._pixmap = None   # so wait_for below can't see a stale hit from above
    window.mode_combo.setCurrentText("julia")
    check(window.pane.render_mode == "julia",
          "selecting a mode in the combo box updates the pane's render_mode")
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
    # session.render_map's plain array flowing through RenderCache/
    # RenderTask's Qt signals/array_to_qimage.
    window.image_view._pixmap = None
    window.mode_combo.setCurrentText("greens")
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "switching to greens mode renders and displays successfully end-to-end")
    check("overflow" not in window.statusBar().currentMessage(),
          "no overflow warning in the status bar -- per-pixel normalization at each pixel's "
          "own escape iteration has no overflow case left to warn about")

    window.image_view._pixmap = None
    window.mode_combo.setCurrentText("parameter_greens")
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "switching to parameter_greens mode renders and displays successfully end-to-end")

    window.image_view._pixmap = None
    window.mode_combo.setCurrentText("parameter_basin")
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "switching to parameter_basin mode renders and displays successfully end-to-end, "
         "with the new stacked counts+unresolved array flowing all the way through the "
         "categorical (non-palette) colourer")

    # A max_iter that would have overflowed the OLD accumulate/degree^
    # max_iter formula (degree^2000 is astronomically outside double
    # range) must still render successfully with no warning -- confirming
    # the overflow escape hatch this test used to force is genuinely gone,
    # not just untested.
    window.session.render_settings = cdx.RenderSettings(2000, 2.0, 1e-6, 0)
    window.image_view._pixmap = None
    window._start_render(window.pane)
    ok = wait_for(lambda: window.image_view._pixmap is not None, timeout_ms=10000)
    check(ok, "a render completes fine at a max_iter that used to force the overflow path")
    window._update_status_bar()
    check("overflow" not in window.statusBar().currentMessage(),
          "still no overflow warning at max_iter=2000 -- per-pixel normalization never "
          "comes near double's range regardless of max_iter")

    # ---- critical-point overlay View-menu actions (Stage 4: menu-only) -----------
    print("\ncritical-point overlay actions:")
    check(window.image_view._show_critical_points is False,
          "the overlay starts off, matching the unchecked menu action")
    window.critical_points_action.setChecked(True)
    check(window.image_view._show_critical_points is True,
          "checking the 'Critical Points' menu action actually reaches image_view's own flag")
    window.trace_orbits_action.setChecked(True)
    check(window.image_view._trace_orbits is True,
          "checking the 'Trace Orbits' menu action reaches image_view's own flag")
    check(window.image_view._orbit_traces is not None,
          "checking it also triggers the lazy orbit-trace computation via set_trace_orbits")
    window.critical_points_action.setChecked(False)
    check(window.image_view._show_critical_points is False, "unchecking turns it back off")

    check(window.image_view._orbit_connect_lines is True,
          "'Connect Orbit Points' starts checked, matching image_view's own default")
    window.orbit_connect_lines_action.setChecked(False)
    check(window.image_view._orbit_connect_lines is False,
          "unchecking 'Connect Orbit Points' reaches image_view's own flag")
    window.orbit_connect_lines_action.setChecked(True)
    check(window.image_view._orbit_connect_lines is True, "re-checking it turns it back on")

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
    check(window.metadata_header._label.text()
          == format_metadata_text(window.session, window.pane.render_mode),
          "the header's displayed text matches the session's actual current state")

    window.mode_combo.setCurrentText("parameter")
    check("Parameter plane" in window.metadata_header._label.text(),
          "switching modes via the combo box refreshes the header's domain text")
    window.mode_combo.setCurrentText("julia")

    window.session.map.name = "renamed-live"
    window.term_editor_panel._notify_edit()   # the real path _on_term_edited fires from
    check("renamed-live" in window.metadata_header._label.text(),
          "a term edit refreshes the header's map name/formula")
    check(window._debounce_timers[window.pane].isActive() and
          window._debounce_timers[window.pane2].isActive(),
          "cache asymmetry (Stage 3): session.map is shared, so a term edit debounces BOTH "
          "panes' renders, not just the focused one")

    # ---- cursor readout reaches the actual status bar ------------------------------
    print("\ncursor readout (live status bar):")
    window.image_view.mouseMoveEvent(_FakeMouseEvent(QPoint(200, 200)))
    check("z = " in window.statusBar().currentMessage(),
          "moving the mouse over the image updates the REAL status bar, not just the "
          "signal/label in isolation")
    check("scale = " in window.statusBar().currentMessage(),
          "the cursor readout is APPENDED to the existing scale/precision message, not "
          "replacing it")

    window.image_view.leaveEvent(None)
    check("z = " not in window.statusBar().currentMessage(),
          "the mouse leaving the image clears the cursor readout from the live status bar too")
    check("scale = " in window.statusBar().currentMessage(),
          "...while the scale/precision half of the message is untouched by that")

    # ---- Reset View --------------------------------------------------------------------
    print("\nReset View:")
    # window.session.map.name is "renamed-live" at this point (set just
    # above) -- not a name default_view_for recognizes, so it falls back to
    # the generic (a-space) view (0, 2.0) -- which, now that this batch's
    # Stage A made the dynamical default framing a FIXED constant too, is
    # the SAME (0, 2.0) value _DYNAMICAL_VIEW_FALLBACK already returns.
    # Reset to a RECOGNIZED name (mandelbrot, whose own parameter-table
    # entry (-0.5, 1.5) is provably different from the fixed dynamical
    # value) so the checks below can still distinguish "used the dynamical
    # table" from "used the parameter table by mistake" -- with
    # "renamed-live" still bound, the two would coincidentally agree and
    # a real routing bug could pass unnoticed.
    window.session.map = cdx.RationalMap.mandelbrot()
    expected_param_center, expected_param_scale = default_view_for(window.session.map.name)
    check((expected_param_center, expected_param_scale) == (complex(-0.5, 0.0), 1.5),
          "sanity: mandelbrot's own parameter-table entry, not the generic fallback")

    # window.pane is on "julia" (dynamical) here (last set by the mode
    # selector section above) -- Reset View on it must frame the FIXED
    # dynamical default (default_dynamical_view, this batch's Stage A),
    # never default_view_for's a-space table -- reusing that table for a
    # dynamical pane's reset was exactly the earlier Stage 2 bug (it
    # jumped a Julia-set pane onto whatever window makes the PARAMETER
    # plane legible instead).
    check(window.pane.render_mode == "julia", "sanity: pane A is dynamical here")
    expected_dyn_center, expected_dyn_scale = default_dynamical_view(window.session.map,
                                                                     window.session.param)
    check((expected_dyn_center, expected_dyn_scale) == _DYNAMICAL_VIEW_FALLBACK,
          "sanity: the dynamical default is exactly the fixed fallback (interim, Stage A)")
    check((expected_dyn_center, expected_dyn_scale)
          != (expected_param_center, expected_param_scale),
          "sanity: the dynamical and parameter-plane default framings are genuinely "
          "different windows here, so the two checks below can't pass by coincidence")

    # Resolution stays at new_resolution (from the Settings Apply above) --
    # only center/scale are thrown away here, to isolate what Reset View
    # itself is being tested against.
    window.pane.viewport = cdx.Viewport(complex(5, 5), 0.001, new_resolution)
    window._reset_view()
    vp = window.pane.viewport
    check(close(vp.center, expected_dyn_center) and abs(vp.scale - expected_dyn_scale) < 1e-9,
          "Reset View on a DYNAMICAL pane frames the Julia set (default_dynamical_view), "
          "NOT a frozen viewport captured at startup and NOT the parameter plane's own "
          "window -- the pre-Stage-2 and Stage 2 bugs, respectively")
    check(vp.resolution == new_resolution,
          "Reset View does NOT revert resolution -- that is a Settings concern (see the last "
          "Apply above), not part of 'the view' pan/zoom resets")

    # Switch that SAME pane to a parameter-plane mode and confirm Reset
    # View correctly switches which table it consults too.
    window.mode_combo.setCurrentText("parameter")
    window.pane.viewport = cdx.Viewport(complex(5, 5), 0.001, new_resolution)
    window._reset_view()
    vp_param = window.pane.viewport
    check(close(vp_param.center, expected_param_center) and
          abs(vp_param.scale - expected_param_scale) < 1e-9,
          "Reset View on a PARAMETER-plane pane still gives default_view_for's own framing")
    window.mode_combo.setCurrentText("julia")   # restore for the sections below

    # Reset View acts on the FOCUSED pane, not always pane A.
    window.pane2.viewport = cdx.Viewport(complex(7, 7), 0.002, window.pane2.viewport.resolution)
    window._set_focused_pane(window.pane2)
    pane_center_before = window.pane.viewport.center
    window._reset_view()
    check(close(window.pane2.viewport.center, expected_dyn_center) and
          abs(window.pane2.viewport.scale - expected_dyn_scale) < 1e-9,
          "Reset View acts on the FOCUSED pane -- here pane2 (also dynamical), once it is "
          "focused")
    check(window.pane.viewport.center == pane_center_before,
          "...and leaves the OTHER (unfocused) pane's viewport completely untouched")
    window._set_focused_pane(window.pane)   # restore focus to pane A for the sections below

    # ---- precision floor warning ---------------------------------------------------
    print("\nprecision floor warning:")
    renderer = cdx.Renderer(map=cdx.Map.custom(window.session.map, window.session.param),
                            viewport=window.pane.viewport, settings=window.session.render_settings)
    tiny_scale = renderer.precision_floor / 10.0
    window.pane.viewport = cdx.Viewport(complex(0, 0), tiny_scale, 60)
    window._update_status_bar()
    check("precision floor" in window.statusBar().currentMessage(),
          "the status bar warns when scale is at/below the precision floor")

    window.pane.viewport = cdx.Viewport(complex(0, 0), 1.5, 60)
    window._update_status_bar()
    check("precision floor" not in window.statusBar().currentMessage(),
          "no warning at an ordinary scale")

    # ---- P6: parameter `a` and orbit seed `z0`, symmetry of input -------------------
    print("\nparameter a field (typed):")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.mode_combo.setCurrentText("julia")
    param_committed = []
    window.image_view.param_changed.connect(param_committed.append)   # sanity: NOT this path

    window.param_field._line_edit.setText("2-3i")
    window.param_field._on_editing_finished()
    check(window.session.param == complex(2, -3),
          "committing the field is the SAME source of truth session.param reads")
    check(param_committed == [], "field commit does not go through the plane-click signal path")
    # The dynamical-plane's OWN default (default_dynamical_view -- interim,
    # a fixed constant; see this batch's Stage A), not default_view_for's
    # a-space table -- Stage 2's fix.
    expected_field_center, expected_field_scale = default_dynamical_view(window.session.map,
                                                                         window.session.param)
    check(close(window.pane.viewport.center, expected_field_center) and
          abs(window.pane.viewport.scale - expected_field_scale) < 1e-9,
          "the dynamical-plane viewport resets to this map's own DYNAMICAL-plane default on "
          "a param change, not the parameter plane's")
    check(window.pane.render_mode == "julia",
          "committing the FIELD (not a plane click) does not switch planes on its own")

    window.param_field._line_edit.setText("not a number")
    window.param_field._on_editing_finished()
    check(window.session.param == complex(2, -3),
          "invalid field input leaves session.param at its last valid value, never zeroed")
    check(window.param_field._line_edit.styleSheet() != "",
          "invalid input marks the field itself, rather than crashing or silently doing nothing")
    window.param_field._line_edit.setText("2-3i")   # restore a valid value for the next section
    window.param_field._on_editing_finished()

    print("\nparameter-plane click (coupled): drives the PARTNER pane, not the clicked one:")
    window.mode_combo.setCurrentText("parameter")
    window._set_focused_pane(window.pane)
    check(window.coupled_view_action.isChecked(), "sanity: still in coupled mode for this test")
    check(window.pane2.render_mode == "julia", "sanity: pane2 is the dynamical partner here")
    pane_a_viewport_before = window.pane.viewport
    # Deliberately perturbed to something the click's own reset couldn't
    # coincidentally already match, so the "genuinely changed" check below
    # actually proves the click drove it, not just that it was already there.
    window.pane2.viewport = cdx.Viewport(complex(9, 9), 0.001, window.pane2.viewport.resolution)
    pane2_viewport_before = window.pane2.viewport

    click_pixel = window.image_view._complex_to_display_pixel(complex(-0.1, 0.6)).toPoint()
    window.image_view._set_param_at(click_pixel)
    check(close(window.session.param, complex(-0.1, 0.6), 1e-2),
          "clicking the parameter plane sets session.param to the clicked coordinate")
    check(window.pane.render_mode == "parameter",
          "COUPLED: the CLICKED pane does not switch plane -- Stage 3's whole point is driving "
          "the partner instead of the clicked pane")
    check(window.pane.viewport.center == pane_a_viewport_before.center and
          window.pane.viewport.scale == pane_a_viewport_before.scale,
          "COUPLED: the clicked (parameter) pane's own viewport is left completely untouched")
    check(close(window.param_field.value, window.session.param, 1e-2),
          "the field is populated with the SAME value the click just set -- the two never diverge")

    # pane2 is dynamical ("julia"), so this MUST be default_dynamical_view's
    # own framing (via the mode-aware dispatcher _apply_param_change now
    # routes through -- interim, a fixed constant; see this batch's Stage
    # A), not default_view_for's a-space table -- Stage 2's fix, exercised
    # here through the Stage 3 coupling path specifically.
    expected_center, expected_scale = default_dynamical_view(window.session.map,
                                                              window.session.param)
    check(close(window.pane2.viewport.center, expected_center) and
          abs(window.pane2.viewport.scale - expected_scale) < 1e-9,
          "the PARTNER dynamical pane's viewport resets to this map's own DYNAMICAL-plane "
          "default framing, not the parameter plane's")
    check(window.pane2.viewport.center != pane2_viewport_before.center or
          abs(window.pane2.viewport.scale - pane2_viewport_before.scale) > 1e-15,
          "sanity: the partner's viewport genuinely changed, not coincidentally already there")

    print("\npersistent param marker: derived from session.param, parameter-plane panes only:")
    marker_a = window.image_view._param_marker_pixel()
    marker_2 = window.image_view2._param_marker_pixel()
    check(marker_a is not None,
          "the clicked (still parameter-plane) pane shows a marker at session.param")
    check(marker_2 is None,
          "the partner (dynamical-plane) pane shows no param marker -- that's a dynamical-plane "
          "concept it doesn't have")
    expected_pixel = window.image_view._complex_to_display_pixel(window.session.param)
    check(close(complex(marker_a.x(), marker_a.y()), complex(expected_pixel.x(), expected_pixel.y()),
               1e-6),
          "the marker sits exactly at session.param's own pixel position")

    # No separate marker state to fall out of sync -- a typed field commit
    # moves it exactly the same way a click does.
    window.param_field._line_edit.setText("0.3-0.2i")
    window.param_field._on_editing_finished()
    marker_after_field = window.image_view._param_marker_pixel()
    expected_pixel2 = window.image_view._complex_to_display_pixel(window.session.param)
    check(close(complex(marker_after_field.x(), marker_after_field.y()),
               complex(expected_pixel2.x(), expected_pixel2.y()), 1e-6),
          "the marker moves with a field commit too, not just a plane click")

    # ---- Stage 4: arrow-key parameter marker nudging ------------------------------
    print("\narrow-key parameter marker nudging: single press moves param by the expected delta:")
    window.coupled_view_action.setChecked(True)
    window._set_focused_pane(window.pane)
    window.mode_combo.setCurrentText("parameter")
    check(window.pane.render_mode == "parameter" and window._focused_pane is window.pane,
          "sanity: pane A is parameter-plane and focused")
    check(window.pane2.render_mode == "julia", "sanity: pane2 is still the dynamical partner")

    window.session.apply_settings(dataclasses.replace(
        window.session.settings, param_marker_step=6.0, param_marker_rate=25.0))

    view = window.image_view
    rect = view._display_rect()
    origin = QPointF(rect.center())

    def _expected_delta(dx: float, dy: float) -> complex:
        # Same computation ImageView._nudge_delta itself does -- see that
        # method's own docstring for why differencing two _pixel_to_complex
        # calls (an already-established, separately-tested primitive) is
        # the derivation, not a hand-rolled scale factor.
        moved = QPointF(origin.x() + dx, origin.y() + dy)
        return view._pixel_to_complex(moved) - view._pixel_to_complex(origin)

    param_before_left = window.session.param
    expected_left = _expected_delta(-6.0, 0.0)
    view.keyPressEvent(_key_event(Qt.Key.Key_Left))
    check(close(window.session.param, param_before_left + expected_left, 1e-9),
          "a single Left press moves session.param by exactly the expected complex delta -- "
          "real part decreases")
    view.keyReleaseEvent(_key_event(Qt.Key.Key_Left, press=False))
    check(not view._nudge_timer.isActive(),
          "releasing the only held arrow stops the repeat timer")

    param_before_up = window.session.param
    expected_up = _expected_delta(0.0, -6.0)   # screen-up = smaller pixel y
    view.keyPressEvent(_key_event(Qt.Key.Key_Up))
    check(close(window.session.param, param_before_up + expected_up, 1e-9),
          "a single Up press moves session.param by exactly the expected complex delta -- "
          "imaginary part increases (screen-up = +imag)")
    view.keyReleaseEvent(_key_event(Qt.Key.Key_Up, press=False))

    print("\nheld-key repeat: produces repeated moves at ~param_marker_rate, stops on release:")
    rate = 25.0
    window.session.apply_settings(dataclasses.replace(
        window.session.settings, param_marker_step=1.0, param_marker_rate=rate))
    nudges: list[complex] = []
    def _record_nudge(a: complex) -> None:
        nudges.append(a)
    view.param_nudged.connect(_record_nudge)

    check(not view._nudge_timer.isActive(), "sanity: the repeat timer is not running before any press")
    view.keyPressEvent(_key_event(Qt.Key.Key_Right))
    check(len(nudges) == 1, "the initial (non-autorepeat) press does exactly one immediate nudge")
    check(view._nudge_timer.isActive(), "the repeat timer starts running while the key is held")

    hold_ms = 600
    QTest.qWait(hold_ms)
    observed = len(nudges)
    expected = 1 + hold_ms / 1000.0 * rate
    # Generous tolerance -- QTimer/event-loop scheduling jitter under a
    # test harness is real, this is checking "roughly the configured rate
    # drove it," not a real-time guarantee.
    check(observed > 1, "the timer produced more nudges than just the initial press's own one")
    check(abs(observed - expected) < max(3, expected * 0.5),
          f"~{rate}/sec repeat rate: observed {observed} nudges in {hold_ms}ms "
          f"(expected ~{expected:.1f})")

    view.keyReleaseEvent(_key_event(Qt.Key.Key_Right, press=False))
    check(not view._nudge_timer.isActive(), "releasing the key stops the repeat timer")
    count_after_release = len(nudges)
    QTest.qWait(200)
    check(len(nudges) == count_after_release,
          "no further nudges happen after release, even after waiting")

    # Qt's own OS-level autorepeat must be ignored -- only OUR timer drives
    # cadence (see keyPressEvent's own isAutoRepeat() check).
    nudges.clear()
    view.keyPressEvent(_key_event(Qt.Key.Key_Right))
    check(len(nudges) == 1, "sanity: the initial non-autorepeat press nudges once")
    view.keyPressEvent(_key_event(Qt.Key.Key_Right, autorepeat=True))
    check(len(nudges) == 1,
          "a Qt autorepeat press (isAutoRepeat()==True) does NOT itself trigger another nudge")
    view.keyReleaseEvent(_key_event(Qt.Key.Key_Right, autorepeat=True, press=False))
    check(view._nudge_timer.isActive(),
          "an autorepeat RELEASE is also ignored -- the key is still physically held, the "
          "repeat timer keeps running")
    view.keyReleaseEvent(_key_event(Qt.Key.Key_Right, press=False))
    check(not view._nudge_timer.isActive(), "the genuine (non-autorepeat) release stops it")
    view.param_nudged.disconnect(_record_nudge)

    print("\narrow keys in a DYNAMICAL pane: unbound, param unchanged (future scope elsewhere):")
    window._set_focused_pane(window.pane2)
    param_before_dynamical = window.session.param
    window.image_view2.keyPressEvent(_key_event(Qt.Key.Key_Left))
    check(window.session.param == param_before_dynamical,
          "an arrow press on the DYNAMICAL pane leaves session.param completely unchanged")
    check(not window.image_view2._held_arrows,
          "...and isn't even tracked as held -- the press was never accepted/consumed there")
    window.image_view2.keyReleaseEvent(_key_event(Qt.Key.Key_Left, press=False))
    window._set_focused_pane(window.pane)

    print("\narrow-key nudge drives the PARTNER dynamical pane's re-render, "
         "WITHOUT resetting its viewport:")
    window.coupled_view_action.setChecked(True)
    wait_for(lambda: window.pane2.request_id not in window.pane2.pending_tasks, timeout_ms=10000)
    pane2_viewport_before_nudge = window.pane2.viewport
    pane2_request_id_before_nudge = window.pane2.request_id

    view.keyPressEvent(_key_event(Qt.Key.Key_Up))
    view.keyReleaseEvent(_key_event(Qt.Key.Key_Up, press=False))

    check(window.pane2.viewport.center == pane2_viewport_before_nudge.center and
         window.pane2.viewport.scale == pane2_viewport_before_nudge.scale,
          "RESOLVED PREMISE: unlike a parameter-plane CLICK, a nudge does NOT reset the "
          "partner dynamical pane's viewport -- it re-renders the SAME framing, so a held "
          "sweep can watch it evolve continuously instead of snapping back to default "
          "every tick")
    check(window.pane2.request_id != pane2_request_id_before_nudge,
          "...but a genuinely NEW render request was still dispatched for the partner pane")
    ok = wait_for(lambda: window.pane2.request_id not in window.pane2.pending_tasks,
                  timeout_ms=10000)
    check(ok, "the partner pane's nudge-triggered render actually completes")

    print("\nparam_marker_step/rate are read LIVE -- a Settings change mid-session takes "
         "effect on the very next nudge:")
    window.session.apply_settings(dataclasses.replace(
        window.session.settings, param_marker_step=3.0, param_marker_rate=25.0))
    origin_live = QPointF(view._display_rect().center())
    small_step_delta = (view._pixel_to_complex(QPointF(origin_live.x() + 3.0, origin_live.y())) -
                        view._pixel_to_complex(origin_live))
    param_before_small = window.session.param
    view.keyPressEvent(_key_event(Qt.Key.Key_Right))
    check(close(window.session.param, param_before_small + small_step_delta, 1e-9),
          "sanity: nudges with param_marker_step=3.0")
    view.keyReleaseEvent(_key_event(Qt.Key.Key_Right, press=False))

    window.session.apply_settings(dataclasses.replace(
        window.session.settings, param_marker_step=30.0, param_marker_rate=25.0))
    origin_live2 = QPointF(view._display_rect().center())
    large_step_delta = (view._pixel_to_complex(QPointF(origin_live2.x() + 30.0, origin_live2.y())) -
                        view._pixel_to_complex(origin_live2))
    param_before_large = window.session.param
    view.keyPressEvent(_key_event(Qt.Key.Key_Right))
    check(close(window.session.param, param_before_large + large_step_delta, 1e-9),
          "a Settings change made BETWEEN two presses (no fresh ImageView, no restart) is "
          "reflected immediately -- param_marker_step is read live, not cached at construction")
    check(abs(large_step_delta) > abs(small_step_delta) * 5,
          "sanity: the step really did change the move size, not coincidentally the same")
    view.keyReleaseEvent(_key_event(Qt.Key.Key_Right, press=False))

    print("\nsingle-view fallback: click still switches the one visible pane (legacy behavior):")
    window.coupled_view_action.setChecked(False)
    window._set_focused_pane(window.pane)
    check(window.pane_column.isVisible() and not window.pane_column2.isVisible(),
          "sanity: single-view, pane A is the one visible pane")
    single_click_point = complex(-0.6, 0.15)
    click_pixel_sv = window.image_view._complex_to_display_pixel(single_click_point).toPoint()
    window.image_view._set_param_at(click_pixel_sv)
    check(close(window.session.param, single_click_point, 1e-2),
          "single-view: clicking still sets session.param to the clicked coordinate")
    check(window.pane.render_mode == "julia",
          "single-view fallback: the one visible pane still switches straight to julia, matching "
          "today's pre-Stage-3 behavior -- there is no partner to drive instead")
    window.coupled_view_action.setChecked(True)   # restore coupled view for the rest of the suite

    print("\ndescribe_parameter_role reaches the live metadata header:")
    check("coefficient of z^0" in window.metadata_header._label.text(),
          "mandelbrot()'s role text (P6 section 2) actually reaches the displayed header, "
          "not just format_metadata_text in isolation")

    print("\norbit seed z0 field (typed) mirrors a dynamical-plane click exactly:")
    window.image_view.clear_orbit()
    window.z0_field._line_edit.setText("0.3+0.2i")
    window.z0_field._on_editing_finished()
    check(window.image_view.orbit_tracker.state is not None,
          "committing the z0 field SEEDS an orbit, exactly like a dynamical-plane click does")
    check(close(window.image_view.orbit_tracker.state.z0, complex(0.3, 0.2), 1e-9),
          "seeded at the typed value")

    print("\ndynamical-plane click populates the z0 field:")
    click_pixel2 = window.image_view._complex_to_display_pixel(complex(0.05, -0.05)).toPoint()
    window.image_view._seed_orbit_at(click_pixel2)
    check(close(window.z0_field.value, complex(0.05, -0.05), 1e-2),
          "clicking the dynamical plane populates the z0 field with the clicked point")

    print("\non a change to a: z0's orbit is RECOMPUTED, not blanked:")
    window.image_view.step_orbit(5)
    check(window.image_view.orbit_tracker.state.n == 5, "sanity: the orbit has advanced")
    seeded_z0 = window.image_view.orbit_tracker.state.z0
    window.param_field._line_edit.setText("-0.75+0i")
    window.param_field._on_editing_finished()
    check(window.image_view.orbit_tracker.state is not None,
          "the orbit overlay survives a param change -- z0 is a persistent, independently "
          "-chosen seed, not cleared the way a term edit clears it")
    check(window.image_view.orbit_tracker.state.z0 == seeded_z0,
          "the SAME z0 is kept across the param change")
    check(window.image_view.orbit_tracker.state.n == 0,
          "but its orbit restarts fresh (n=0) under the NEW map/param, not left stale at n=5")

    print("\nz0 field commit while on the parameter plane is a silent no-op:")
    window.mode_combo.setCurrentText("parameter")
    window.image_view.clear_orbit()
    window.z0_field._line_edit.setText("1+1i")
    window.z0_field._on_editing_finished()
    check(window.image_view.orbit_tracker.state is None,
          "orbit tracking stays a dynamical-plane concept -- same gate _seed_orbit_at itself uses")
    window.mode_combo.setCurrentText("julia")

    # ---- experiment snapshots: File > Save/Open Experiment, through the REAL window --
    # Session-level round-trip correctness is already covered directly in
    # app/test_session.py; this is the one thing only a live SandboxWindow
    # can confirm -- that _do_open_experiment's UI-facing glue (BOTH mode
    # combo boxes, the param field, layout/focus, and each pane's
    # ImageView.orbit_tracker actual seed()/step() reconstruction) all
    # genuinely happen, not just session.restore_from_snapshot's own
    # dict-level fields.
    print("\nexperiment snapshots (File > Save/Open Experiment, via the real window):")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.session.param = complex(0.111, -0.222)
    window.pane.viewport = cdx.Viewport(complex(0.01, 0.02), 0.05, 200)
    window.mode_combo.setCurrentText("julia")
    window.pane2.viewport = cdx.Viewport(complex(-0.3, 0.4), 0.8, 150)
    window.mode_combo2.setCurrentText("parameter")
    window.coupled_view_action.setChecked(False)
    window._set_focused_pane(window.pane2)
    saved_z0 = complex(0.05, -0.05)
    window.image_view.orbit_tracker.seed(window.session.map, window.session.param, saved_z0)
    window.image_view.step_orbit(3)
    saved_history = list(window.image_view.orbit_tracker.state.history)
    window.image_view2.clear_orbit()   # pane2 is parameter-plane anyway; sanity, not load-bearing

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roundtrip.cdsx")
        window._do_save_experiment(path)
        check(os.path.exists(path), "Save Experiment actually wrote a file")

        # Disturb EVERYTHING the snapshot should restore, so a pass here
        # can only mean the load genuinely put it back, not that it was
        # already correct by coincidence.
        window.session.map = cdx.RationalMap.newton_cubic()
        window.session.param = complex(9, 9)
        window.pane.viewport = cdx.Viewport(complex(9, 9), 9.0, 50)
        window.pane2.viewport = cdx.Viewport(complex(8, 8), 8.0, 50)
        window.mode_combo.setCurrentText("parameter")
        window.mode_combo2.setCurrentText("julia")
        window.coupled_view_action.setChecked(True)
        window._set_focused_pane(window.pane)
        window.image_view.clear_orbit()

        window._do_open_experiment(path)

        check(window.session.map.to_formula() == cdx.RationalMap.mandelbrot().to_formula(),
              "Open Experiment restores the map")
        check(window.session.param == complex(0.111, -0.222), "Open Experiment restores param")
        check(window.pane.viewport.center == complex(0.01, 0.02) and
              window.pane.viewport.scale == 0.05,
              "Open Experiment restores pane A's viewport")
        check(window.pane2.viewport.center == complex(-0.3, 0.4) and
              window.pane2.viewport.scale == 0.8,
              "Open Experiment restores pane2's viewport too, not just pane A's")
        check(window.mode_combo.currentText() == "julia",
              "Open Experiment syncs pane A's mode combo box to its restored render_mode, not "
              "just the pane's own render_mode underneath it")
        check(window.mode_combo2.currentText() == "parameter",
              "...and pane2's mode combo box to ITS restored render_mode")
        check(close(window.param_field.value, complex(0.111, -0.222), 1e-9),
              "Open Experiment syncs the 'Parameter a' field's displayed value too")
        check(not window.coupled_view_action.isChecked(),
              "Open Experiment restores the single/coupled layout flag")
        check(window._focused_pane is window.pane2,
              "Open Experiment restores which pane was focused")

        state = window.image_view.orbit_tracker.state
        check(state is not None and state.z0 == saved_z0,
              "Open Experiment reconstructs the orbit at the SAVED z0, on the SAME pane it "
              "was saved from")
        check(state.n == 3, "...stepped the saved number of times")
        check(list(state.history) == saved_history,
              "...through the tracker's real seed()/step() API, regenerating the exact same "
              "history a live orbit would have")
        check(window.image_view2.orbit_tracker.state is None,
              "the OTHER pane (parameter-plane, no saved orbit) gets no orbit reconstructed")

    # ---- malformed / wrong-version .cdsx: rejected, window left untouched ----------
    print("\nexperiment snapshots (Open Experiment rejects malformed/wrong-version files):")
    before_map_formula = window.session.map.to_formula()
    before_param = window.session.param
    before_pane_center = window.pane.viewport.center
    before_pane2_center = window.pane2.viewport.center
    before_coupled = window.coupled_view_action.isChecked()
    before_focused = window._focused_pane

    # QMessageBox.critical() is a REAL modal -- .exec() starts a nested
    # event loop and blocks waiting for a click that will never come
    # under QT_QPA_PLATFORM=offscreen, hanging the whole test process.
    # Stubbed out for just this one call, the same "can't touch a real
    # blocking dialog" constraint _do_save_experiment/_do_open_experiment's
    # own module docstring already documents for QFileDialog.
    critical_calls = []
    original_critical = sandbox_module.QMessageBox.critical
    sandbox_module.QMessageBox.critical = staticmethod(
        lambda *args, **kwargs: critical_calls.append(args))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "bad.cdsx")
            with open(bad_path, "w") as f:
                f.write('{"schema_version": 1, "not": "a real stage-4 snapshot"}')
            window._do_open_experiment(bad_path)
    finally:
        sandbox_module.QMessageBox.critical = original_critical
    check(len(critical_calls) == 1,
          "a malformed/wrong-version Open Experiment shows exactly one error dialog")

    check(window.session.map.to_formula() == before_map_formula and
          window.session.param == before_param,
          "a malformed/wrong-version file leaves the session's map/param completely untouched")
    check(window.pane.viewport.center == before_pane_center and
          window.pane2.viewport.center == before_pane2_center,
          "...and both panes' viewports too -- rejected before anything is mutated")
    check(window.coupled_view_action.isChecked() == before_coupled and
          window._focused_pane is before_focused,
          "...and the layout (coupled flag, focused pane) as well")

    # ---- no-effect-parameter guard: newton_cubic on the parameter plane -------------
    print("\nno-effect-parameter guard (a parameter-plane pane for a map `a` doesn't affect):")
    window.session.map = cdx.RationalMap.newton_cubic()
    window.mode_combo.setCurrentText("parameter")
    check(window.image_view.no_effect_parameter_message() is not None,
          "newton_cubic() on the parameter plane reports a no-effect message -- `a` genuinely "
          "has no term depending on it")
    check("no effect" in window.image_view.no_effect_parameter_message() or
          "has no" in window.image_view.no_effect_parameter_message(),
          "the message actually explains why, not just a blank/generic string")

    window.mode_combo.setCurrentText("parameter_basin")
    check(window.image_view.no_effect_parameter_message() is not None,
          "parameter_basin gets the SAME no-effect guard as plain parameter -- driven by "
          "PARAMETER_PLANE_MODES membership, not a hardcoded per-mode name list")
    window.mode_combo.setCurrentText("parameter")

    rid_before_guard = window.pane.request_id
    window._start_render(window.pane)
    check(window.pane.request_id == rid_before_guard,
          "_start_render does not dispatch a RenderTask for a no-effect parameter-plane pane "
          "-- there is nothing meaningful to compute, not just nothing meaningful to show")

    window.mode_combo.setCurrentText("julia")
    check(window.image_view.no_effect_parameter_message() is None,
          "the SAME map's dynamical plane has no such guard -- newton's iteration itself "
          "obviously does depend on z, just not on `a`")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.mode_combo.setCurrentText("parameter")
    check(window.image_view.no_effect_parameter_message() is None,
          "and mandelbrot()'s parameter plane (where `a` is the whole point) has no guard either")

    # ---- File > Export: PNG at a chosen resolution, and JSON facts (Stage 4) --------
    print("\nFile > Export (Image PNG / Facts JSON), via the real window:")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.mode_combo.setCurrentText("julia")
    window._set_focused_pane(window.pane)
    window.pane.viewport = cdx.Viewport(complex(-0.5, 0.0), 1.2, 100)

    with tempfile.TemporaryDirectory() as tmp:
        png_path = os.path.join(tmp, "export")   # deliberately no extension
        window._do_export_image(png_path, 64)
        actual_png_path = png_path + ".png"
        check(os.path.exists(actual_png_path),
              "_do_export_image appends .png when the chosen path doesn't already have one")
        exported_image = QImage(actual_png_path)
        check(exported_image.width() == 64 and exported_image.height() == 64,
              "the exported PNG is FRESH-rendered at the CHOSEN resolution -- not the pane's "
              "own display resolution (100) or a downsample of whatever's on screen")
        check(not exported_image.isNull(), "the exported file is a genuine, readable PNG")

        json_path = os.path.join(tmp, "facts")   # also no extension
        window._do_export_facts(json_path)
        actual_json_path = json_path + ".json"
        check(os.path.exists(actual_json_path),
              "_do_export_facts appends .json when the chosen path doesn't already have one")
        with open(actual_json_path) as f:
            exported_facts = json.load(f)
        check(exported_facts["map"]["formula"] == window.session.map.to_formula() and
              exported_facts["render_mode"] == "julia",
              "the exported facts describe the CURRENT map and the FOCUSED pane's own "
              "render_mode")
        check(exported_facts["viewport"]["center"] == [window.pane.viewport.center.real,
                                                        window.pane.viewport.center.imag],
              "...and the focused pane's own viewport too, for provenance")

    # ---- File > Export: the no-effect-parameter guard applies to image export too ---
    print("\nFile > Export: no-effect-parameter guard:")
    window.session.map = cdx.RationalMap.newton_cubic()
    window.mode_combo.setCurrentText("parameter")
    critical_calls_export = []
    original_critical_export = sandbox_module.QMessageBox.critical
    sandbox_module.QMessageBox.critical = staticmethod(
        lambda *args, **kwargs: critical_calls_export.append(args))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            guard_png_path = os.path.join(tmp, "guarded.png")
            window._do_export_image(guard_png_path, 50)
            check(not os.path.exists(guard_png_path),
                  "Export Image refuses to write a PNG for a no-effect parameter-plane pane "
                  "-- the same guard _start_render uses for the live view")
    finally:
        sandbox_module.QMessageBox.critical = original_critical_export
    check(len(critical_calls_export) == 1,
          "...and shows exactly one explanatory error dialog instead of a silent no-op")
    window.mode_combo.setCurrentText("julia")
    window.session.map = cdx.RationalMap.mandelbrot()

    # ---- File > Export: chosen overlays actually land in the exported bytes (Stage B) --
    # ExportImageDialog itself opens a real modal QDialog (.exec()) that can't be driven
    # offscreen -- so what's tested directly here is compose_export_image, the exact
    # function both the dialog's live preview and _do_export_image call, plus
    # ExportImageDialog's own checkbox-applicability logic (constructing it doesn't
    # require .exec()).
    print("\nFile > Export: overlays compose into the image, and the dialog only offers "
          "checkboxes that apply:")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.mode_combo.setCurrentText("julia")
    window._set_focused_pane(window.pane)
    window.pane.viewport = cdx.Viewport(complex(0.0, 0.0), 1.5, 100)

    clean = compose_export_image(window.session.map, window.session.param, window.pane.viewport,
                                 window.pane.render_mode, window.session.settings,
                                 window.session.render_settings, 48)
    with_critical = compose_export_image(
        window.session.map, window.session.param, window.pane.viewport, window.pane.render_mode,
        window.session.settings, window.session.render_settings, 48,
        show_critical_points=True, critical_points=[0j])
    check(clean.constBits().tobytes() != with_critical.constBits().tobytes(),
          "compose_export_image with the critical-point overlay enabled produces different "
          "pixel bytes than the same call with no overlays -- the overlay genuinely lands "
          "in the exported image, not just in an ignored kwarg")
    check(clean.width() == 48 and clean.height() == 48
          and with_critical.width() == 48 and with_critical.height() == 48,
          "compose_export_image renders at the CHOSEN resolution regardless of overlays, "
          "matching what File > Export's own resolution spinbox controls")

    window.pane.image_view.orbit_tracker.clear()   # earlier sections may have left one seeded

    dynamical_dialog = ExportImageDialog(window.session, window.pane, window)
    check(dynamical_dialog._critical_points_checkbox is not None
          and dynamical_dialog._trace_orbits_checkbox is not None,
          "a dynamical-plane pane's Export dialog offers critical-point and traced-orbit "
          "checkboxes")
    check(dynamical_dialog._param_marker_checkbox is None,
          "...but no parameter-marker checkbox -- there's no parameter marker on a "
          "dynamical plane")
    check(dynamical_dialog._orbit_checkbox is None and dynamical_dialog._connect_lines_checkbox is None,
          "and, with no orbit currently seeded on this pane, no orbit/connect-lines "
          "checkboxes either -- an empty toggle for a nonexistent orbit has nothing to "
          "control")
    dynamical_dialog.close()

    window.pane.image_view.orbit_tracker.seed(window.session.map, window.session.param,
                                               complex(0.1, 0.1))
    seeded_dialog = ExportImageDialog(window.session, window.pane, window)
    check(seeded_dialog._orbit_checkbox is not None
          and seeded_dialog._connect_lines_checkbox is not None,
          "...but once an orbit IS seeded, the same pane's Export dialog offers both the "
          "orbit and connect-lines checkboxes")
    seeded_dialog.close()
    window.pane.image_view.orbit_tracker.clear()

    window.mode_combo.setCurrentText("parameter")
    window._set_focused_pane(window.pane)
    param_dialog = ExportImageDialog(window.session, window.pane, window)
    check(param_dialog._param_marker_checkbox is not None,
          "a parameter-plane pane's Export dialog offers a parameter-marker checkbox")
    check(param_dialog._critical_points_checkbox is None
          and param_dialog._trace_orbits_checkbox is None,
          "...but no critical-point/traced-orbit checkboxes -- those are dynamical-plane-"
          "only concepts")
    param_dialog.close()
    window.mode_combo.setCurrentText("julia")
    window.session.map = cdx.RationalMap.mandelbrot()

    # ---- Facts tab: clicking a fixed/critical point seeds an orbit (Stage 5) --------
    print("\nFacts tab: clicking a fixed/critical point seeds an orbit on the dynamical plane:")
    window.session.map = cdx.RationalMap.newton_cubic()
    window.session.param = 0j
    window.mode_combo.setCurrentText("julia")
    window.mode_combo2.setCurrentText("parameter")
    window._set_focused_pane(window.pane2)   # focus the PARAMETER pane deliberately
    window.facts_panel.refresh(force=True)
    window.image_view.clear_orbit()
    window.image_view2.clear_orbit()

    fixed_points = window.facts_panel._row_points[id(window.facts_panel._fixed_table)]
    finite_fixed_row = next(i for i, p in enumerate(fixed_points)
                            if p is not None and not math.isinf(p.real) and not math.isinf(p.imag))
    target_point = fixed_points[finite_fixed_row]
    window.facts_panel._on_row_clicked(window.facts_panel._fixed_table, finite_fixed_row)

    check(window.image_view.orbit_tracker.state is not None and
          window.image_view.orbit_tracker.state.z0 == target_point,
          "clicking a fixed-point row seeds an orbit at that point on the DYNAMICAL pane, "
          "even though the PARAMETER pane (pane2) is the one currently focused")
    check(window.image_view2.orbit_tracker.state is None,
          "...and does NOT seed anything on the parameter-plane pane")
    check(window.tabs.currentWidget() is window.view_container,
          "seeding an orbit from the Facts tab switches to the View tab, the same as "
          "centring on a pole already does")

    # Critical-point rows do the same thing.
    window.tabs.setCurrentIndex(2)   # back to Facts
    window.image_view.clear_orbit()
    critical_points = window.facts_panel._row_points[id(window.facts_panel._critical_table)]
    finite_critical_row = next(i for i, p in enumerate(critical_points)
                               if p is not None and not math.isinf(p.real)
                               and not math.isinf(p.imag))
    target_critical_point = critical_points[finite_critical_row]
    window.facts_panel._on_row_clicked(window.facts_panel._critical_table, finite_critical_row)
    check(window.image_view.orbit_tracker.state is not None and
          window.image_view.orbit_tracker.state.z0 == target_critical_point,
          "clicking a CRITICAL-point row seeds an orbit too, the same as a fixed point")

    # Pole rows keep the EXISTING centre-the-view behavior, unchanged.
    window.tabs.setCurrentIndex(2)
    pane2_center_before = window.pane2.viewport.center
    poles = window.facts_panel._row_points[id(window.facts_panel._pole_table)]
    window.facts_panel._on_row_clicked(window.facts_panel._pole_table, 0)
    check(window.pane2.viewport.center == poles[0] and
          window.pane2.viewport.center != pane2_center_before,
          "a pole row still centres the FOCUSED pane's view (pane2, parameter) -- unchanged "
          "from before Stage 5")

    # No dynamical pane at all (both parameter-plane) -> a clean no-op.
    window.mode_combo.setCurrentText("parameter")   # now BOTH panes are parameter-plane
    window.image_view.clear_orbit()
    window.facts_panel._on_row_clicked(window.facts_panel._fixed_table, finite_fixed_row)
    check(window.image_view.orbit_tracker.state is None and
          window.image_view2.orbit_tracker.state is None,
          "with no pane in a dynamical mode, clicking a fixed/critical point row is a "
          "no-op -- orbit seeding is a dynamical-plane concept, same gate "
          "ImageView.seed_orbit itself uses")

    window.mode_combo.setCurrentText("julia")
    window.session.map = cdx.RationalMap.mandelbrot()
    window._set_focused_pane(window.pane)

    # ---- library: saving a family writes a sidecar thumbnail (Stage C) --------------
    print("\nLibrary: saving a family writes a sidecar preview thumbnail:")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.mode_combo.setCurrentText("julia")
    window.library_panel._do_save_as("stage-c-sidecar-family")
    check(preview_path_for("stage-c-sidecar-family").exists(),
          "Save Current As -> on_change -> _on_library_changed writes a real sidecar PNG, "
          "not just the library.txt entry")

    placeholder_image = _placeholder_icon().pixmap(_LIST_ICON_SIZE, _LIST_ICON_SIZE).toImage()
    saved_row = next(row for row in range(window.library_panel._list.count())
                     if window.library_panel._list.item(row).data(_NAME_ROLE)
                     == "stage-c-sidecar-family")
    saved_icon = (window.library_panel._list.item(saved_row).icon()
                 .pixmap(_LIST_ICON_SIZE, _LIST_ICON_SIZE).toImage())
    check(saved_icon != placeholder_image,
          "...and the live LibraryPanel's own list icon reflects it immediately, through the "
          "real _on_library_changed -> refresh_previews wiring, not just on the next full "
          "app restart")

    window.library_panel._do_delete("stage-c-sidecar-family")
    check(not preview_path_for("stage-c-sidecar-family").exists(),
          "deleting the entry removes its sidecar too, via the same regenerate-from-current-"
          "library-contents pass")

    # ---- File > Export/Open Experiment: embedded preview thumbnail (Stage C) --------
    print("\nFile > Save/Open Experiment: embedded preview thumbnail:")
    window.session.map = cdx.RationalMap.mandelbrot()
    window.mode_combo.setCurrentText("julia")
    window._set_focused_pane(window.pane)
    with tempfile.TemporaryDirectory() as tmp:
        cdsx_path = os.path.join(tmp, "with-preview.cdsx")
        window._do_save_experiment(cdsx_path)
        with open(cdsx_path) as f:
            saved_data = json.load(f)
        check(isinstance(saved_data.get("preview"), str) and len(saved_data["preview"]) > 0,
              "a saved .cdsx embeds a non-empty base64 'preview' string, not just the map/"
              "param/layout fields")
        check(base64.b64decode(saved_data["preview"])[:8] == b"\x89PNG\r\n\x1a\n",
              "...and it's a genuine PNG (correct magic bytes), not an arbitrary string")

        # Opening it must not raise on the new field, even though
        # _do_open_experiment intentionally never reads `preview` back out
        # (see its own comment) -- a round-trip that crashes on an unused
        # field would be its own bug.
        window._do_open_experiment(cdsx_path)
        check(window.session.map.to_formula() == cdx.RationalMap.mandelbrot().to_formula(),
              "Open Experiment still restores the map correctly with a preview-bearing file")

    window.mode_combo.setCurrentText("julia")
    window.session.map = cdx.RationalMap.mandelbrot()
    window._set_focused_pane(window.pane)

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
