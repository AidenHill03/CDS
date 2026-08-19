"""app/sandbox.py -- P5a/P5b: the interactive window.

Single window, driven by app.session.Session, with tabs: the image pane
(render pipeline: threaded, progressive, cursor-anchored zoom, overscan
buffer, correct orientation), a term editor (app/term_editor_panel.py), a
read-only dynamical-facts panel (app/facts_panel.py), a family library
(app/library_panel.py), and a Settings panel (app/settings_panel.py) for
render/cache configuration.

Requires the cdx extension module and PySide6 to be importable, e.g. from
the repository root:

    PYTHONPATH=cdx/build python -m app.sandbox

Entry point: `python -m app.sandbox`.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import shiboken6
from PySide6.QtCore import (QObject, QPoint, QPointF, QRect, QRectF, QRunnable, QSize, Qt,
                            QThreadPool, QTimer, Signal, Slot)
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter,
                               QTabWidget, QToolBar, QVBoxLayout, QWidget)

import cdx
from app.about_dialog import AboutDialog
from app.colour import colour_basin, colour_escape_time, colour_scalar_field
from app.complex_field import ComplexField
from app.facts_panel import FactsPanel
from app.library_panel import LibraryPanel, default_view_for_mode
from app.metadata_header import MetadataHeader, describe_parameter_role
from app.orbit_panel import OrbitPanel
from app.orbit_tracker import OrbitTracker
from app.pane import Pane
from app.render_cache import RenderCache
from app.session import (DEFAULT_PARAMETER_VIEW_CENTER, DEFAULT_PARAMETER_VIEW_SCALE,
                         PARAMETER_PLANE_MODES, RENDER_MODES, Session, render_map)
from app.settings import Settings, library_path, load_settings, save_settings
from app.term_editor_panel import TermEditorPanel
from app.settings_panel import SettingsPanel
from app.version import PRODUCT_NAME

# Per-notch scroll zoom factor (f > 1 zooms in). ~1.15 per the spec: a
# handful of notches gives a noticeable zoom without a single notch jumping
# too far to feel controllable.
ZOOM_FACTOR_PER_NOTCH = 1.15

# Restarted on every wheel/pan-move event; a render is only actually
# started once this settles, so a fast scroll or drag does not queue a
# backlog of renders.
RENDER_DEBOUNCE_MS = 80

# Quarter-resolution preview: resolution divided by 4 in each dimension
# (1/16 the pixel count), rendered first and replaced by the full-resolution
# image when it arrives.
PREVIEW_RESOLUTION_DIVISOR = 4

# Overscan: every render request asks for a region WIDER than the visible
# viewport (same pixel density, more area -- see _overscanned), so pan and
# zoom-out have real rendered pixels to reveal immediately instead of an
# empty border while the real re-render is still in flight. Squared cost (a
# factor-f overscan is f^2 the pixel count of the un-overscanned request):
# full uses a modest factor since it is already the most expensive render;
# preview affords a much larger one because it starts from 1/16 the pixel
# count (quartered per axis) of a full render, so even 4x its own cost is
# still cheap -- and it's exactly the buffer on screen right after a big
# zoom-out, where a wide real-pixel margin matters most.
FULL_OVERSCAN_FACTOR = 1.3
PREVIEW_OVERSCAN_FACTOR = 2.0

# Trigger a fresh render once the visible viewport's edge has drifted this
# far into the last-painted buffer's own half-width (as a fraction of it),
# rather than relying solely on the ordinary debounce. A long continuous
# drag or scroll restarts the debounce on every event and would otherwise
# never let it fire, panning or zooming the visible window clean off the
# edge of the buffer's real pixels before a fresh one ever lands.
BUFFER_EDGE_FRACTION = 0.85

# "Approaching" Renderer.precision_floor -- warn this many multiples out, so
# the user has advance notice before the image actually degenerates.
PRECISION_WARN_MULTIPLE = 100

# Post-critical orbit trace length: enough steps to show whether an orbit
# is heading toward a cycle/attractor without turning into unreadable
# clutter for a map with several critical points, and cheap regardless of
# map complexity -- a handful of RationalMap.eval() calls per critical
# point, computed once per (map, param) and cached (see ImageView.
# refresh_critical_points), not per paint.
CRITICAL_ORBIT_TRACE_STEPS = 60

CRITICAL_POINT_MARKER_RADIUS = 5.0

# How far outside the visible widget rect a mapped orbit/trace point may
# still fall and be treated as "leaving the view" rather than "off
# screen, don't draw toward it" -- see drawable_polyline_segments. A
# point escaping to infinity is finite right up until it isn't (a
# quadratic Julia orbit can sit at |z|~1e6 before overflowing), and that
# maps to a pixel far outside the widget; without this margin a single
# drawLine to it sweeps a line across the entire pane, and several such
# segments wash into a translucent "film" instead of reading as an orbit.
OFFSCREEN_MARGIN_PX = 300


def drawable_polyline_segments(points: list[complex], pixel_points: list,
                               rect) -> list[tuple]:
    """Which CONSECUTIVE pairs of an orbit/trace sequence are safe to draw
    as a line segment, given each point's own mapped screen position.

    Deliberately walks pairs of the ORIGINAL sequence rather than
    filtering out non-finite points first and connecting the survivors
    (the bug this replaces): a pair is drawable only when BOTH endpoints
    are finite complex numbers AND both mapped pixels fall inside `rect`
    (the caller passes an already-inflated rect -- see
    OFFSCREEN_MARGIN_PX). A pair failing either check is a BREAK, not
    bridged to the next drawable point -- a real gap in the orbit (an
    escape to infinity) stays a gap on screen instead of being painted
    over with a line to whatever finite point happens to come next.

    A point that's finite but off-screen is simply not drawn TO or FROM;
    the segment is skipped rather than clipped to the rect edge -- simpler,
    and sufficient to kill the sweep/film bug (a partial segment that
    genuinely exits the view just isn't drawn, rather than being drawn up
    to the boundary).

    Pure function of its arguments (no widget/painter access), so the
    geometry is unit-testable without constructing a QPainter.
    """
    def finite(w: complex) -> bool:
        return math.isfinite(w.real) and math.isfinite(w.imag)

    def drawable(i: int) -> bool:
        return finite(points[i]) and rect.contains(pixel_points[i])

    segments = []
    for i in range(len(points) - 1):
        if drawable(i) and drawable(i + 1):
            segments.append((pixel_points[i], pixel_points[i + 1]))
    return segments


def _overscanned(viewport: cdx.Viewport, factor: float) -> cdx.Viewport:
    """A viewport covering `factor` times the half-width of `viewport`, same
    center, at the SAME pixel density (resolution scaled by the same
    factor) -- so render cost scales as factor**2, not just factor. This is
    the overscan buffer request; see FULL_OVERSCAN_FACTOR/
    PREVIEW_OVERSCAN_FACTOR and ImageView's docstring for how the buffer's
    own viewport is kept alongside the rendered array afterwards so the
    display can map back onto it correctly.
    """
    resolution = max(1, round(viewport.resolution * factor))
    return cdx.Viewport(viewport.center, viewport.scale * factor, resolution)


def array_to_qimage(payload, mode: str, settings: Settings,
                    max_iter: int) -> QImage:
    """Converts a cdx render payload (see session.render_map's own
    docstring for what `payload` actually is per mode) into a displayable,
    COLOURED QImage.

    cdx render arrays have row 0 at the BOTTOM (matching Viewport::coord;
    see CLAUDE.md) but QImage has row 0 at the TOP -- flip vertically first,
    or preview and full renders would visibly disagree on orientation. This
    was a real bug in the MATLAB prototype (progressive rendering flipped
    the preview relative to the full render because the axis orientation
    was set once instead of after every draw call); flipping the array
    itself, once, here, is what rules that class of bug out entirely rather
    than relying on every caller remembering an origin='lower'-equivalent.

    Dispatches to app.colour by render mode: "julia"/"parameter" are
    escape-time (colour_escape_time, using `settings`' palette/scaling/
    period); "basin" is categorical + SHADED basin colouring (hue = basin
    id, brightness = convergence speed); "greens"/"parameter_greens" are
    scalar-field equipotential banding (colour_scalar_field, using
    `settings`' greens_band_width/greens_period_bands/greens_contour) --
    the SAME display treatment for both, since they're the same KIND of
    data (a non-negative potential) even though they live on different
    planes (see cdx::Renderer::render_parameter_greens' own header comment
    on why the two are computed by genuinely different kernels).

    "basin" mode's payload is STACKED (see render_map's own docstring):
    shape (2, height, width), index 0 = labels, index 1 = iterations --
    unpacked and flipped as two separate 2D layers BEFORE the generic
    np.flipud path below, since flipud always flips axis 0, which for a
    3D stacked array is the LAYER axis, not the row axis; flipping the
    unpacked 2D layers individually (axis 0 IS the row axis there) is what
    actually produces a correctly oriented image instead of silently
    swapping labels and iterations. "greens"/"parameter_greens" payloads
    are a plain 2D array like "julia"/"parameter", just with a different
    colour treatment (see below).

    Colouring is a pure DISPLAY-time transform, deliberately not baked into
    what RenderCache stores (raw float arrays) -- changing the palette must
    never be a cache key or trigger a re-render, only a re-paint.
    """
    if mode == "basin":
        labels = np.flipud(payload[0])
        iterations = np.flipud(payload[1])
        rgb = colour_basin(labels, iterations, max_iter=max_iter,
                           period=settings.colour_period or None)
        return _rgb_to_qimage(rgb)
    if mode in ("greens", "parameter_greens"):
        flipped = np.flipud(payload)
        rgb = colour_scalar_field(flipped, palette=settings.colour_palette,
                                  band_width=settings.greens_band_width,
                                  period_bands=settings.greens_period_bands,
                                  contour=settings.greens_contour)
        return _rgb_to_qimage(rgb)
    flipped = np.flipud(payload)
    if mode in ("julia", "parameter"):
        rgb = colour_escape_time(flipped, max_iter, palette=settings.colour_palette,
                                 scaling=settings.colour_scaling,
                                 period=settings.colour_period or None)
        return _rgb_to_qimage(rgb)
    raise AssertionError(f"unreachable: mode={mode!r}")


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    height, width, _channels = rgb.shape
    image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
    # .copy(): `rgb` is a local array pybind11/numpy does not keep alive on
    # QImage's behalf. Without this, the buffer QImage points at is freed as
    # soon as this function returns, and the image is garbage or a crash.
    return image.copy()


class RenderSignals(QObject):
    """QRunnable isn't itself a QObject and so cannot emit signals; this is
    the companion object RenderTask uses instead. Created on the GUI thread
    (in RenderTask.__init__, before the task is handed to the thread pool),
    emitted from the worker thread.

    Qt only queues a cross-thread emission to the GUI thread's event loop
    when it can determine the RECEIVER's thread affinity -- which requires
    an actual QObject receiver, not a bare function/lambda. A crash was
    once traced to exactly this: SandboxWindow._start_render used to
    connect these signals to context-less lambdas, so PySide had no
    receiver to check and fell back to a DIRECT (same-thread, i.e.
    worker-thread) call -- which ran widget-touching slot code off the GUI
    thread AND, worse, let a worker-thread slot invocation pop the task's
    own entry from Pane.pending_tasks (the last thing keeping this very
    RenderSignals object alive) from INSIDE its own emit() call, on the
    worker thread. Once that reference (and _Runnable's, autoDelete'd by
    QThreadPool right after run() returns, also worker-thread-side) both
    dropped, this QObject's C++ destructor could run mid-emit, on the
    worker thread, racing the GUI thread's event loop trying to deliver a
    still-pending posted event to it -- EXC_BAD_ACCESS in Shiboken's
    QObjectWrapper::event.

    The fix (see SandboxWindow._start_render/_on_partial_ready/
    _on_full_ready/_on_render_failed): connect these signals ONLY to
    genuine bound methods of a QObject living on the GUI thread (self, the
    window) -- PySide then correctly detects the cross-thread case and
    queues delivery, which both runs the slot body on the GUI thread AND
    means Pane.pending_tasks.pop() (the release of the GUI side's own
    reference) happens on the GUI thread too, so this object's last
    reference is never dropped from the worker thread.
    """
    partial_ready = Signal(int, object, object)   # request_id, render_map() payload, buffer viewport
    full_ready = Signal(int, object, object)       # request_id, render_map() payload, buffer viewport
    failed = Signal(int, str)                       # request_id, error message


class RenderTask:
    """Renders one request: quarter-resolution preview first, then full
    resolution, both on whichever thread pool thread runs this. Plain
    Python, not itself a QRunnable -- see the _Runnable adapter below --
    so it stays easy to construct and inspect directly (e.g. in a test).
    """

    def __init__(self, request_id: int, rational_map: cdx.RationalMap, param: complex,
                viewport: cdx.Viewport, settings: cdx.RenderSettings, mode: str,
                cancel: cdx.CancelToken, cache: RenderCache | None = None):
        self.request_id = request_id
        self.rational_map = rational_map
        self.param = param
        self.viewport = viewport
        self.settings = settings
        self.mode = mode
        self.cancel = cancel
        self.cache = cache
        self.signals = RenderSignals()

    def run(self) -> None:
        # Checked at every natural checkpoint, not just once at the top:
        # cancellation can arrive at any point while this runs, and a
        # cancelled task must not emit anything at all (see
        # SandboxWindow.closeEvent -- a cancelled-but-still-emitting task is
        # exactly the race that made draining the pool on close necessary
        # before cancellation existed; skipping emit() entirely here is what
        # lets closeEvent skip waiting instead).
        #
        # Both stages render an OVERSCANNED buffer (see _overscanned), wider
        # than self.viewport, and emit that buffer's own viewport alongside
        # the array -- ImageView needs it to map the buffer back onto the
        # display (see its docstring). self.viewport itself is never
        # rendered directly.
        try:
            if self.cancel.is_cancelled:
                return
            preview_res = max(1, self.viewport.resolution // PREVIEW_RESOLUTION_DIVISOR)
            preview_viewport = cdx.Viewport(self.viewport.center, self.viewport.scale, preview_res)
            preview_buffer_viewport = _overscanned(preview_viewport, PREVIEW_OVERSCAN_FACTOR)
            preview_array = render_map(self.rational_map, self.param, preview_buffer_viewport,
                                       self.settings, self.mode, self.cancel, self.cache)
            if self.cancel.is_cancelled:
                return
            self.signals.partial_ready.emit(self.request_id, preview_array, preview_buffer_viewport)

            full_buffer_viewport = _overscanned(self.viewport, FULL_OVERSCAN_FACTOR)
            full_array = render_map(self.rational_map, self.param, full_buffer_viewport,
                                    self.settings, self.mode, self.cancel, self.cache)
            if self.cancel.is_cancelled:
                return
            self.signals.full_ready.emit(self.request_id, full_array, full_buffer_viewport)
        except Exception as exc:   # report to the GUI thread rather than crashing the pool
            if not self.cancel.is_cancelled:
                self.signals.failed.emit(self.request_id, str(exc))


class _Runnable(QRunnable):
    """Wraps a RenderTask as an actual QRunnable for QThreadPool. A thin
    adapter rather than making RenderTask itself a QRunnable, so RenderTask
    stays plain Python -- easy to construct and inspect directly.
    """

    def __init__(self, task: RenderTask):
        super().__init__()
        self._task = task
        self.setAutoDelete(True)

    def run(self) -> None:
        self._task.run()


class ImageView(QWidget):
    """The single image pane: displays the current render, and owns every
    mouse/wheel interaction (scroll zoom, drag rubber-band zoom, pan).

    The displayed image is always drawn into a centered SQUARE sub-rect of
    the widget (see _display_rect), never stretched to the widget's full
    (possibly non-square) size. cdx.Viewport is inherently square (same
    half-width in both axes); stretching would give real and imaginary axes
    different pixel-to-plane scales, which would silently break both the
    "axes must match Viewport exactly" requirement and the cursor-anchored
    zoom math below (which assumes one uniform scale).

    INSTANT FEEDBACK. pane.viewport updates immediately on every wheel
    or pan event, but the real re-render is debounced (see SandboxWindow).
    In between, the widget keeps showing the last PAINTED buffer -- but
    that buffer is OVERSCANNED (rendered wider than the viewport it was
    requested for; see _overscanned in this module), so it usually still
    has real pixels covering the new, moved viewport. set_image() stores
    the buffer's own viewport alongside its pixmap (_buffer_viewport);
    paintEvent recomputes, on every paint, exactly which sub-rect of that
    buffer the CURRENT pane.viewport corresponds to (_buffer_source_rect)
    and draws only that sub-rect stretched to fill the display. Because
    this is recomputed fresh from the two known viewports every time rather
    than accumulated incrementally across events, five wheel notches or a
    long drag before the debounce fires need no special composition step --
    pane.viewport already reflects all of them, and the mapping from it
    back onto the buffer is exact by construction, not an approximation
    that could drift.

    The mapping composes two independently-derived steps -- screen pixel to
    complex value (_pixel_to_complex, under the CURRENT viewport) and
    complex value to buffer-local pixel (_complex_to_buffer_pixel, under
    the buffer's own stored viewport). Both are uniform scale + translation
    only (no rotation, since neither viewport nor buffer is ever rotated),
    so each is affine, and so is their composition -- meaning the two
    corners used to build the source rect (_buffer_source_rect) determine
    every interior point exactly, by linear interpolation, with no need to
    recompute per-pixel. test_sandbox.py checks an arbitrary interior probe
    pixel against a fully independent hand computation, not just the
    corners, to confirm that.

    Once the buffer's real margin runs out -- the visible viewport has
    drifted far enough that part of it falls outside the buffer entirely --
    see buffer_edge_fraction() for the check that asks SandboxWindow to
    render a fresh buffer before the ordinary debounce would.
    """

    viewport_changed = Signal()
    orbit_changed = Signal()
    cursor_readout_changed = Signal(str)
    # Emitted at the start of every mouse press -- Stage 2's only hook for
    # "which pane did the user just interact with." ImageView itself has
    # no notion of a window or of other panes; SandboxWindow is the one
    # that connects this (per instance, via a pane-capturing lambda -- see
    # _build_ui) to actually move focus. Unconnected in every standalone
    # test that builds an ImageView without a SandboxWindow, so it is a
    # pure no-op there.
    pane_activated = Signal()
    # Emitted with the clicked complex value on a PARAMETER-plane click --
    # deliberately does NOT set session.param itself (unlike _seed_orbit_at
    # setting the orbit tracker directly): SandboxWindow._apply_param_change
    # is the one place that assignment happens (via _on_param_changed_by_click
    # or _on_param_field_committed), so the plane-click and field-commit
    # paths funnel through EXACTLY the same underlying invalidation logic
    # rather than two divergent copies of it.
    param_changed = Signal(complex)

    def __init__(self, pane: Pane, session: Session, parent: QWidget | None = None):
        super().__init__(parent)
        # pane: viewport + render_mode, per-pane state (see app.pane.Pane).
        # session: map/param/render_settings/settings/cache -- SHARED across
        # every pane looking at the same experiment.
        self.pane = pane
        self.session = session
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        self.orbit_tracker = OrbitTracker()
        # Orbit-only -- deliberately independent of _trace_orbits below,
        # which governs the SEPARATE post-critical-point traces (see
        # set_orbit_connect_lines's own docstring).
        self._orbit_connect_lines: bool = True

        self._pixmap: QPixmap | None = None
        # The viewport _pixmap was actually rendered for (wider than
        # pane.viewport by the overscan factor -- see _overscanned).
        # None exactly when _pixmap is None; kept alongside it because
        # _buffer_source_rect needs both to map back onto the display.
        self._buffer_viewport: cdx.Viewport | None = None
        # See set_image's own comment -- the raw numeric payload + mode it
        # was rendered under, for the cursor readout's sampling.
        self._buffer_payload = None
        self._buffer_mode: str | None = None

        # ---- critical-point overlay (dynamical plane only) --------------------
        self._show_critical_points = False
        self._trace_orbits = False
        # Memoized on (map.serialize(), param) -- see refresh_critical_points --
        # so panning/zooming (which only ever changes pane.viewport, never
        # the map or param) never re-triggers root-finding. None until the
        # first refresh_critical_points() call (made once by SandboxWindow
        # right after construction, same as its other panels' initial state).
        self._critical_points_key: tuple[str, complex] | None = None
        self._critical_points: list[complex] = []
        # None means "not computed for the current key yet" (distinct from
        # an empty list, a genuinely orbit-free map) -- computed lazily,
        # only once _trace_orbits is actually turned on, since tracing costs
        # real RationalMap.eval() calls the marker-only display doesn't need.
        self._orbit_traces: list[list[complex]] | None = None

        self._rubber_band_origin: QPoint | None = None
        self._rubber_band_rect: QRect | None = None

        self._panning = False
        self._pan_anchor_complex: complex | None = None

    # ---- display geometry ----------------------------------------------------
    def _display_rect(self) -> QRect:
        side = min(self.width(), self.height())
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2
        return QRect(x, y, side, side)

    def _pixel_to_complex(self, pixel: QPoint | QPointF) -> complex:
        rect = self._display_rect()
        vp = self.pane.viewport
        if rect.width() <= 0 or rect.height() <= 0:
            return vp.center
        rel_x = (pixel.x() - rect.left()) / rect.width()
        # Screen y increases downward; the plane's imaginary part increases
        # upward. Top of the display rect (rel_y=0) is center.imag+scale,
        # matching array_to_qimage's flip (see its docstring).
        rel_y = (pixel.y() - rect.top()) / rect.height()
        re = vp.center.real - vp.scale + rel_x * 2.0 * vp.scale
        im = vp.center.imag + vp.scale - rel_y * 2.0 * vp.scale
        return complex(re, im)

    def _complex_to_display_pixel(self, w: complex) -> QPointF:
        """The exact inverse of _pixel_to_complex, against the CURRENT
        pane.viewport (not the buffer's own, unlike
        _complex_to_buffer_pixel below) -- what overlay drawing (critical
        points, orbit traces) uses to place a marker at a given complex
        point's actual on-screen position. Read-only: computing a screen
        position from pane.viewport is not the same as writing to it,
        and nothing here ever assigns pane.viewport -- see CLAUDE.md's
        note on the MATLAB prototype's overlay-expands-the-axes bug this
        is written to not repeat.
        """
        rect = self._display_rect()
        vp = self.pane.viewport
        rel_x = (w.real - (vp.center.real - vp.scale)) / (2.0 * vp.scale)
        rel_y = (vp.center.imag + vp.scale - w.imag) / (2.0 * vp.scale)
        return QPointF(rect.left() + rel_x * rect.width(), rect.top() + rel_y * rect.height())

    # ---- overscan buffer mapping: buffer's own viewport, not the display's ------
    def _complex_to_buffer_pixel(self, w: complex) -> QPointF:
        # The exact inverse of _pixel_to_complex, applied to the BUFFER's
        # stored viewport instead of the display's current one. Row 0 at
        # the top, matching the pixmap (array_to_qimage already flipped it
        # to screen convention), same as _pixel_to_complex assumes.
        vp = self._buffer_viewport
        width = self._pixmap.width()
        height = self._pixmap.height()
        rel_x = (w.real - (vp.center.real - vp.scale)) / (2.0 * vp.scale)
        rel_y = (vp.center.imag + vp.scale - w.imag) / (2.0 * vp.scale)
        return QPointF(rel_x * width, rel_y * height)

    # ---- cursor readout: live coordinate + mode-dependent sampled value ---------
    def cursor_readout_text(self, pixel: QPoint | QPointF) -> str:
        """z's coordinate, with PRECISION THAT SCALES WITH ZOOM DEPTH (four
        decimals is useless at scale 1e-9 -- P5c's own wording) -- enough
        decimal digits that two ADJACENT pixels' coordinates print as
        different strings, not just enough to show the value at all. Plus
        a mode-dependent second field sampled from the last-rendered
        buffer (see _sample_at_pixel): escape value for julia/parameter,
        basin index for basin, potential for greens/parameter_greens.
        """
        w = self._pixel_to_complex(pixel)
        vp = self.pane.viewport
        pixel_step = (2.0 * vp.scale) / max(vp.resolution, 1)
        if pixel_step > 0 and math.isfinite(pixel_step):
            decimals = max(4, int(math.ceil(-math.log10(pixel_step))) + 1)
        else:
            decimals = 4
        coord = f"z = {w.real:.{decimals}f}{'+' if w.imag >= 0 else '-'}{abs(w.imag):.{decimals}f}j"
        sample = self._sample_at_pixel(pixel)
        return f"{coord}   {sample}" if sample else coord

    def _sample_at_pixel(self, pixel: QPoint | QPointF) -> str | None:
        if self._buffer_payload is None or self._buffer_viewport is None or self._pixmap is None:
            return None
        buf_pixel = self._complex_to_buffer_pixel(self._pixel_to_complex(pixel))
        col = int(buf_pixel.x())
        row_top = int(buf_pixel.y())   # top-down, matching _complex_to_buffer_pixel's own convention

        mode = self._buffer_mode
        payload = self._buffer_payload
        if mode == "basin":
            height, width = payload[0].shape
        else:
            height, width = payload.shape

        if not (0 <= col < width and 0 <= row_top < height):
            return None   # cursor is over the display but off the (possibly overscanned-
                          # differently) buffer's own real pixels -- nothing to sample
        raw_row = height - 1 - row_top   # buffer is top-down; the array itself is row-0-bottom

        if mode in ("julia", "parameter"):
            value = payload[raw_row, col]
            return "never escaped" if value == 0.0 else f"escape = {value:.4g}"
        if mode == "basin":
            label = payload[0][raw_row, col]
            return "unresolved" if label == 0.0 else f"basin = {int(label)}"
        if mode in ("greens", "parameter_greens"):
            value = payload[raw_row, col]
            return f"potential = {value:.4g}"
        return None

    def _buffer_source_rect(self) -> QRectF | None:
        """The sub-rect of the last-painted buffer (in its own pixel space)
        that the CURRENT pane.viewport corresponds to -- computed fresh
        from the two known viewports every call, not accumulated across
        events, so it is always exact regardless of how many wheel/pan
        events have landed since the buffer was rendered. See the class
        docstring.
        """
        if self._pixmap is None or self._buffer_viewport is None:
            return None
        rect = self._display_rect()
        # Continuous corners (left+width, top+height), NOT QRect's own
        # bottomRight() -- that is INCLUSIVE (left+width-1, top+height-1),
        # a full pixel short of the edge _pixel_to_complex's rel_x/rel_y
        # already treat as continuous. Using it here would make this crop
        # off by roughly one buffer pixel in ~width, an approximation where
        # this mapping is meant to be exact.
        top_left = self._complex_to_buffer_pixel(
            self._pixel_to_complex(QPointF(rect.left(), rect.top())))
        bottom_right = self._complex_to_buffer_pixel(
            self._pixel_to_complex(QPointF(rect.left() + rect.width(), rect.top() + rect.height())))
        return QRectF(top_left, bottom_right)

    def buffer_edge_fraction(self) -> float:
        """How far the current session viewport's edge has drifted into
        the last-painted buffer's own half-width, as a fraction of it (the
        more extreme of the two axes) -- 1.0 means the visible viewport's
        edge exactly touches the buffer's edge; values above that mean part
        of the visible viewport is already outside the buffer's real
        pixels. No buffer yet (nothing painted) is reported as maximally
        due (1.0), same as an edge exactly reached -- both mean "render
        now, don't wait for the ordinary debounce."
        """
        if self._buffer_viewport is None or self._buffer_viewport.scale <= 0:
            return 1.0
        vp = self.pane.viewport
        buf = self._buffer_viewport
        d_re = abs(vp.center.real - buf.center.real) + vp.scale
        d_im = abs(vp.center.imag - buf.center.imag) + vp.scale
        return max(d_re, d_im) / buf.scale

    # ---- critical-point overlay: cached per (map, param), never re-root-found on
    # ---- pan/zoom -- see P5c's own "cache per (map, parameter); do not re-root-find
    # ---- on zoom or pan" requirement --------------------------------------------
    def refresh_critical_points(self) -> None:
        """Recomputes the critical-point set for the CURRENT session.map/
        param, memoized on (map.serialize(), param) so this is a no-op
        whenever neither has actually changed -- called by SandboxWindow
        after every term edit and family load (the same points that
        already refresh the Facts panel), NEVER from a viewport change.

        Uses distinct_critical_points (deduplicated within the engine's
        own tolerance), not the raw critical_points() -- this overlay
        draws one MARKER per distinct location; multiplicity (already
        shown properly in the Facts panel) has no separate visual meaning
        here, and drawing several exactly-overlapping markers at one
        multiplicity->2 point would add nothing but redundant painting.
        """
        key = (self.session.map.serialize(), self.session.param)
        if key != self._critical_points_key:
            self._critical_points_key = key
            self._critical_points = list(
                self.session.map.distinct_critical_points(self.session.param))
            self._orbit_traces = None   # stale -- lazily recomputed below if still wanted
        if self._trace_orbits and self._orbit_traces is None:
            self._orbit_traces = [self._trace_orbit(z0) for z0 in self._critical_points]

    def _trace_orbit(self, z0: complex) -> list[complex]:
        points = [z0]
        z = z0
        for _ in range(CRITICAL_ORBIT_TRACE_STEPS):
            if math.isinf(z.real) or math.isinf(z.imag) or math.isnan(z.real) or math.isnan(z.imag):
                break   # infinity (or a NaN escape) has no further finite orbit to draw
            z = self.session.map.eval(z, self.session.param)
            points.append(z)
        return points

    def set_show_critical_points(self, show: bool) -> None:
        self._show_critical_points = show
        self.update()

    def set_trace_orbits(self, trace: bool) -> None:
        self._trace_orbits = trace
        self.refresh_critical_points()   # lazily fills in traces if just turned on
        self.update()

    def set_orbit_connect_lines(self, on: bool) -> None:
        """Governs ONLY _paint_orbit's connecting segments -- the seeded
        orbit's own history, drawn in orange. Independent of
        set_trace_orbits: the post-critical-point traces (drawn in
        _paint_critical_point_overlay, in white) are a different overlay
        entirely and always connect when shown, regardless of this flag.
        """
        self._orbit_connect_lines = on
        self.update()

    # ---- orbit tracking: click-to-seed, Step/Run N/Clear -------------------------
    def refresh_orbit_staleness(self) -> None:
        """Called at the same map/param-change points refresh_critical_points
        is (term edits, family loads) -- clears the traced orbit the moment
        it stops describing THIS map/parameter's dynamics (see
        OrbitTracker.reset_if_stale's own docstring for why this is not
        optional). Deliberately never called from a viewport change --
        panning/zooming is the same dynamics, seen differently, and the
        orbit should stay exactly where it was traced.
        """
        self.orbit_tracker.reset_if_stale(self.session.map, self.session.param)
        self.orbit_changed.emit()

    def step_orbit(self, count: int = 1) -> None:
        self.orbit_tracker.step(self.session.map, self.session.param, count)
        self.orbit_changed.emit()
        self.update()

    def clear_orbit(self) -> None:
        self.orbit_tracker.clear()
        self.orbit_changed.emit()
        self.update()

    def refresh_orbit_for_new_param(self) -> None:
        """Called specifically when `a` changes via the parameter field or
        a parameter-plane click (SandboxWindow._apply_param_change) -- the
        counterpart to refresh_orbit_staleness's CLEAR, but recomputes
        instead: z0 is a persistent, independently-chosen seed (P6's own
        wording), so its orbit under the NEW map/param is exactly what
        should be shown, not a blanked overlay. A no-op if no orbit is
        currently seeded (see OrbitTracker.recompute_current).
        """
        self.orbit_tracker.recompute_current(self.session.map, self.session.param)
        self.orbit_changed.emit()
        self.update()

    def _seed_orbit_at(self, pixel: QPoint) -> None:
        if self.pane.render_mode in PARAMETER_PLANE_MODES:
            return   # orbit tracking is a dynamical-plane concept, see its own module docstring
        z0 = self._pixel_to_complex(pixel)
        self.orbit_tracker.seed(self.session.map, self.session.param, z0)
        self.orbit_changed.emit()
        self.update()

    def _set_param_at(self, pixel: QPoint) -> None:
        # Only emits -- see param_changed's own comment for why this
        # doesn't touch session.param itself.
        self.param_changed.emit(self._pixel_to_complex(pixel))

    # ---- painting --------------------------------------------------------------
    def set_image(self, payload, buffer_viewport: cdx.Viewport) -> None:
        # Reads CURRENT session state, not whatever was active when the
        # underlying render was dispatched -- correct for a Settings change
        # (a palette switch while a render is in flight should show up the
        # instant this frame is displayed) and safe for render_mode too:
        # a mode switch always starts a fresh render under a NEW request_id
        # (see SandboxWindow._start_render/_on_mode_changed), so any result
        # still arriving under the OLD request_id is already discarded by
        # the request_id check in _on_partial_ready/_on_full_ready before
        # set_image is ever called -- this can never run with a mode that
        # doesn't match what was actually rendered.
        image = array_to_qimage(payload, self.pane.render_mode, self.session.settings,
                                self.session.render_settings.max_iter)
        self._pixmap = QPixmap.fromImage(image)
        self._buffer_viewport = buffer_viewport
        # Raw payload + the mode it was rendered under, kept for the cursor
        # readout's mode-dependent second field (escape value / basin
        # index / potential) -- sampling needs the actual numbers, which
        # array_to_qimage's RGB output has already discarded. Capturing
        # pane.render_mode HERE (not re-reading it later, e.g. from
        # mouseMoveEvent) is what keeps this correct even mid-mode-switch:
        # see the comment above on why pane.render_mode is guaranteed
        # to match `payload` at this exact point.
        self._buffer_payload = payload
        self._buffer_mode = self.pane.render_mode
        self.update()

    def no_effect_parameter_message(self) -> str | None:
        """None ordinarily. On a PARAMETER-plane pane for a map where `a`
        has no effect on it AT ALL (describe_parameter_role reports
        "unused by this map", confirmed directly for e.g. newton_cubic() --
        no term has param_power != 0 or location_is_param) every pixel
        of a render here would be identical -- a uniform field that looks
        like a bug, not a genuine "nothing here." This is the message to
        show INSTEAD of that render: see SandboxWindow._start_render,
        which skips dispatching a RenderTask for a pane in this state
        entirely (there's nothing meaningful to compute, not just nothing
        meaningful to display), and paintEvent below, which shows this
        text instead of the (nonexistent) buffer.
        """
        if self.pane.render_mode not in PARAMETER_PLANE_MODES:
            return None
        if describe_parameter_role(self.session.map) != "unused by this map":
            return None
        return (f"'a' has no effect on {self.session.map.name} -- there is nothing to show "
                "on the parameter plane for this map")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        message = self.no_effect_parameter_message()
        if message is not None:
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(),
                             int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                             message)
            return
        source_rect = self._buffer_source_rect()
        if source_rect is not None:
            painter.drawPixmap(QRectF(self._display_rect()), self._pixmap, source_rect)
        if self._rubber_band_rect is not None:
            pen = QPen(QColor(255, 255, 255))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(self._rubber_band_rect)
        self._paint_critical_point_overlay(painter)
        self._paint_orbit(painter)
        self._paint_param_marker(painter)

    def _should_draw_critical_points(self) -> bool:
        # Critical points are objects of the DYNAMICAL plane -- a pixel on
        # the parameter plane is a PARAMETER, not a point the map acts on,
        # so "the map's critical point at this pixel" isn't a meaningful
        # thing to mark there (see P5c's own spec on this exact
        # distinction). False on those modes, not an error -- the
        # checkboxes stay checked/available for whenever the user switches
        # back to a dynamical-plane mode.
        return self._show_critical_points and self.pane.render_mode not in PARAMETER_PLANE_MODES

    def _paint_critical_point_overlay(self, painter: QPainter) -> None:
        if not self._should_draw_critical_points():
            return

        def finite(w: complex) -> bool:
            return math.isfinite(w.real) and math.isfinite(w.imag)

        if self._trace_orbits and self._orbit_traces:
            trace_pen = QPen(QColor(255, 255, 255, 160))
            trace_pen.setWidth(1)
            painter.setPen(trace_pen)
            inflated_rect = QRectF(self.rect()).adjusted(
                -OFFSCREEN_MARGIN_PX, -OFFSCREEN_MARGIN_PX,
                OFFSCREEN_MARGIN_PX, OFFSCREEN_MARGIN_PX)
            for trace in self._orbit_traces:
                pixel_points = [self._complex_to_display_pixel(w) for w in trace]
                for a, b in drawable_polyline_segments(trace, pixel_points, inflated_rect):
                    painter.drawLine(a, b)

        marker_pen = QPen(QColor(0, 0, 0))
        marker_pen.setWidth(2)
        painter.setPen(marker_pen)
        painter.setBrush(QColor(255, 255, 255))
        r = CRITICAL_POINT_MARKER_RADIUS
        for w in self._critical_points:
            if not finite(w):
                continue   # infinity is a valid critical point but has no finite screen position
            pixel = self._complex_to_display_pixel(w)
            painter.drawEllipse(pixel, r, r)

    def _orbit_line_segments(self):
        """The line segments _paint_orbit would draw for the CURRENT
        orbit, already respecting both the dynamical-plane-only gate and
        the connect-lines toggle -- factored out of _paint_orbit so this
        is directly testable without a real QPainter/paint event. Returns
        [] whenever _paint_orbit would draw no lines at all: no orbit,
        parameter-plane mode, or the toggle off.
        """
        state = self.orbit_tracker.state
        if (state is None or self.pane.render_mode in PARAMETER_PLANE_MODES
                or not self._orbit_connect_lines):
            return []
        inflated_rect = QRectF(self.rect()).adjusted(
            -OFFSCREEN_MARGIN_PX, -OFFSCREEN_MARGIN_PX,
            OFFSCREEN_MARGIN_PX, OFFSCREEN_MARGIN_PX)
        pixel_points = [self._complex_to_display_pixel(w) for w in state.history]
        return drawable_polyline_segments(state.history, pixel_points, inflated_rect)

    def _paint_orbit(self, painter: QPainter) -> None:
        state = self.orbit_tracker.state
        if state is None or self.pane.render_mode in PARAMETER_PLANE_MODES:
            return   # same dynamical-plane-only gating as critical points, see its own comment

        def finite(w: complex) -> bool:
            return math.isfinite(w.real) and math.isfinite(w.imag)

        inflated_rect = QRectF(self.rect()).adjusted(
            -OFFSCREEN_MARGIN_PX, -OFFSCREEN_MARGIN_PX,
            OFFSCREEN_MARGIN_PX, OFFSCREEN_MARGIN_PX)
        pixel_points = [self._complex_to_display_pixel(w) for w in state.history]

        # Lines are the only thing the toggle governs -- dots and the seed
        # marker below are cheap, never sweep, and stay on regardless (see
        # set_orbit_connect_lines's own docstring).
        line_pen = QPen(QColor(255, 120, 0, 200))   # orange -- distinct from the white/black
        line_pen.setWidth(1)                        # critical-point markers on any palette
        painter.setPen(line_pen)
        for a, b in self._orbit_line_segments():
            painter.drawLine(a, b)

        dot_pen = QPen(QColor(0, 0, 0))
        dot_pen.setWidth(1)
        painter.setPen(dot_pen)
        painter.setBrush(QColor(255, 120, 0))
        dot_r = CRITICAL_POINT_MARKER_RADIUS * 0.6   # smaller than a critical-point marker --
        for w, pixel in zip(state.history, pixel_points):   # visually subordinate, not competing
            if finite(w) and inflated_rect.contains(pixel):
                painter.drawEllipse(pixel, dot_r, dot_r)

        # The SEED (history[0]) gets its own larger, distinct marker -- the
        # one point in the trace the user actually chose, not one the
        # orbit passed through.
        seed_pixel = self._complex_to_display_pixel(state.z0)
        if finite(state.z0) and inflated_rect.contains(seed_pixel):
            seed_pen = QPen(QColor(0, 0, 0))
            seed_pen.setWidth(2)
            painter.setPen(seed_pen)
            painter.setBrush(QColor(255, 255, 0))
            painter.drawEllipse(seed_pixel, CRITICAL_POINT_MARKER_RADIUS,
                                CRITICAL_POINT_MARKER_RADIUS)

    def _param_marker_pixel(self) -> QPointF | None:
        """The screen pixel for session.param's crosshair marker on a
        PARAMETER-plane pane, or None if nothing should be drawn (a
        dynamical-plane mode, a non-finite param, or off the visible
        rect) -- factored out of _paint_param_marker so this is directly
        testable without a real QPainter/paint event, the same reason
        _orbit_line_segments exists.

        Deliberately derived from self.session.param at paint time, not
        separate marker state stored anywhere -- Stage 3's "keep the
        marker in sync with the typed field and with param set any other
        way" is then true BY CONSTRUCTION: there is nothing that could
        fall out of sync, because there is nothing else to keep in sync.
        """
        if self.pane.render_mode not in PARAMETER_PLANE_MODES:
            return None
        a = self.session.param
        if not (math.isfinite(a.real) and math.isfinite(a.imag)):
            return None
        pixel = self._complex_to_display_pixel(a)
        if not QRectF(self.rect()).contains(pixel):
            return None
        return pixel

    def _paint_param_marker(self, painter: QPainter) -> None:
        pixel = self._param_marker_pixel()
        if pixel is None:
            return
        r = CRITICAL_POINT_MARKER_RADIUS
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(QPointF(pixel.x() - r, pixel.y()), QPointF(pixel.x() + r, pixel.y()))
        painter.drawLine(QPointF(pixel.x(), pixel.y() - r), QPointF(pixel.x(), pixel.y() + r))

    # ---- scroll zoom, cursor-anchored -------------------------------------------
    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        notches = delta / 120.0
        factor = ZOOM_FACTOR_PER_NOTCH ** notches

        cursor_pos = event.position().toPoint()
        w = self._pixel_to_complex(cursor_pos)
        vp = self.pane.viewport
        new_center = w - (w - vp.center) / factor
        new_scale = vp.scale / factor
        self.pane.viewport = cdx.Viewport(new_center, new_scale, vp.resolution)

        # Instant feedback: repainting now re-derives the buffer source
        # rect from the viewport just set above -- see _buffer_source_rect.
        self.update()

        self.viewport_changed.emit()
        event.accept()

    # ---- drag: rubber-band zoom, or pan -----------------------------------------
    def mousePressEvent(self, event) -> None:
        self.pane_activated.emit()
        pos = event.position().toPoint()
        is_pan = (event.button() == Qt.MouseButton.MiddleButton or
                 (event.button() == Qt.MouseButton.LeftButton and
                  event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        if is_pan:
            self._panning = True
            self._pan_anchor_complex = self._pixel_to_complex(pos)
        elif event.button() == Qt.MouseButton.LeftButton:
            self._rubber_band_origin = pos
            self._rubber_band_rect = QRect(pos, QSize())
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._panning:
            vp = self.pane.viewport
            current_point = self._pixel_to_complex(pos)
            # Shift the center so the point that was under the cursor at
            # drag start is again under the cursor now -- see the module
            # test suite for the derivation this formula comes from.
            new_center = vp.center + (self._pan_anchor_complex - current_point)
            self.pane.viewport = cdx.Viewport(new_center, vp.scale, vp.resolution)
            self.update()

            self.viewport_changed.emit()
        elif self._rubber_band_origin is not None:
            self._rubber_band_rect = QRect(self._rubber_band_origin, pos).normalized()
            self.update()

        # Live, on every move regardless of panning/dragging -- the cursor
        # coordinate (and, mid-pan, what's still on screen at that point)
        # stays meaningful throughout the gesture, not just once it ends.
        self.cursor_readout_changed.emit(self.cursor_readout_text(pos))

    def leaveEvent(self, event) -> None:
        self.cursor_readout_changed.emit("")

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self._pan_anchor_complex = None
            self.viewport_changed.emit()
        elif self._rubber_band_origin is not None:
            rect = self._rubber_band_rect
            origin = self._rubber_band_origin
            self._rubber_band_origin = None
            self._rubber_band_rect = None
            self.update()
            if rect is not None and rect.width() > 4 and rect.height() > 4:
                self._apply_rubber_band_zoom(rect)
            else:
                # A genuine CLICK, not a drag -- P6's "symmetry of input":
                # a click on the DYNAMICAL plane seeds an orbit at z0
                # (P5c's original "click to seed an orbit"); a click on
                # the PARAMETER plane sets `a` instead, the gesture that
                # previously did nothing at all there (_seed_orbit_at's
                # own early return for PARAMETER_PLANE_MODES).
                if self.pane.render_mode in PARAMETER_PLANE_MODES:
                    self._set_param_at(origin)
                else:
                    self._seed_orbit_at(origin)

    def _apply_rubber_band_zoom(self, rect: QRect) -> None:
        top_left = self._pixel_to_complex(rect.topLeft())
        bottom_right = self._pixel_to_complex(rect.bottomRight())
        new_center = complex((top_left.real + bottom_right.real) / 2.0,
                             (top_left.imag + bottom_right.imag) / 2.0)
        half_re = abs(bottom_right.real - top_left.real) / 2.0
        half_im = abs(top_left.imag - bottom_right.imag) / 2.0
        new_scale = max(half_re, half_im)
        if new_scale <= 0:
            return
        vp = self.pane.viewport
        self.pane.viewport = cdx.Viewport(new_center, new_scale, vp.resolution)
        self.viewport_changed.emit()


class SandboxWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(PRODUCT_NAME)   # the product name, not an internal milestone tag
        self.resize(800, 860)
        self._status_base_message = ""
        self._status_cursor_text = ""

        # Persisted settings (app/settings.py's config file, next to where a
        # saved family library also lives -- see app.settings.config_dir)
        # survive between runs; load_settings() degrades gracefully to plain
        # defaults if that file is missing, malformed, or partially invalid.
        self.session = Session(settings=load_settings())
        # Merges any previously-saved user families into the library
        # Session() already populated with the six presets (see
        # load_user_library's own docstring for why this is a merge, not a
        # replace) -- degrades the same way load_settings() does if the
        # file is missing (nothing saved yet) or malformed.
        self.session.load_user_library(library_path())

        # Two panes: `self.pane` (left, defaults to the parameter plane)
        # and `self.pane2` (right, defaults to the dynamical plane at this
        # map's own default framing) -- see app.pane.Pane's own docstring
        # for what a pane owns (viewport + render_mode + its own
        # render-supersession state) versus what stays on Session
        # (map/param/render_settings/cache, shared across both). Neither
        # pane is privileged by role -- Stage 3's coupling keys off
        # whether a pane's OWN render_mode is in PARAMETER_PLANE_MODES,
        # never off "is this self.pane or self.pane2" -- these are just
        # the two panes' STARTING modes, matching today's single-view
        # startup (parameter plane) plus a natural dynamical partner.
        # Constructed BEFORE _build_ui() -- ImageView's own constructor
        # reads an initial viewport/mode from its pane, so both panes must
        # already exist with real values.
        resolution = self.session.settings.resolution
        self.pane = Pane(cdx.Viewport(DEFAULT_PARAMETER_VIEW_CENTER, DEFAULT_PARAMETER_VIEW_SCALE,
                                      resolution),
                         "parameter")
        # "julia" -- the dynamical plane -- so this MUST be
        # default_dynamical_view's own critical-/fixed-point-derived
        # framing (via the mode-aware dispatcher), not default_view_for's
        # a-space table -- using the parameter-plane's own window here was
        # Stage 2's bug: pane2 would start framed on (say) mandelbrot's
        # OWN (-0.5, 1.5) a-space window applied as though it were z-space.
        pane2_center, pane2_scale = default_view_for_mode(self.session.map, self.session.param,
                                                          "julia")
        self.pane2 = Pane(cdx.Viewport(pane2_center, pane2_scale, resolution), "julia")
        # Every pane this window owns. Iterated by closeEvent (cancel every
        # pane's in-flight renders) and _on_settings_applied (a resolution
        # change touches every pane's viewport, not just one).
        self.panes: list[Pane] = [self.pane, self.pane2]

        # The pane Reset View/the status bar/the metadata header act on --
        # starts on self.pane (matching today's single-view startup), moves
        # whenever either ImageView reports a mouse press (see
        # ImageView.pane_activated, wired in _build_ui). Not restricted to
        # "the pane currently marked dynamical" or any other role -- either
        # pane can be focused regardless of its mode.
        self._focused_pane = self.pane

        # A dedicated pool, not QThreadPool.globalInstance(): keeps this
        # app's renders independent of anything else in the process that
        # might use the global pool. SHARED across every pane -- panes
        # only need their OWN request-id/pending-task bookkeeping (see
        # app.pane.Pane) to keep results from crossing between them; the
        # pool executing the work is one app-wide resource regardless.
        self._thread_pool = QThreadPool()
        # Pane.pending_tasks (populated in _start_render) is the GUI-side
        # owner of every in-flight RenderTask -- NOT the _Runnable adapter
        # QThreadPool holds (that one is setAutoDelete(True) and gets torn
        # down from the WORKER thread the instant run() returns, which must
        # never be the thing that drops a RenderTask/RenderSignals' last
        # reference -- see RenderSignals' own docstring for the crash this
        # caused). pending_tasks keeps the task alive until a GUI-thread
        # handler (_on_full_ready/_on_render_failed, or the stale-sweep
        # below for a cancelled task that never emits a terminal signal at
        # all) explicitly releases it, which is also where
        # task.signals.deleteLater() posts the actual C++ teardown back to
        # the GUI thread's event loop rather than leaving it to whichever
        # thread's refcounting happens to hit zero last. Entries for a task
        # that completes (or fails) normally are removed when its final
        # signal is DELIVERED (queued to the GUI thread -- see
        # _on_full_ready/_on_render_failed). A CANCELLED task emits nothing
        # at all (see RenderTask.run), so its entry is instead swept on the
        # NEXT _start_render() call for THAT PANE, once cancellation (which
        # per-column checking bounds to roughly one column's worth of work,
        # not the full render) has had a full round to actually finish --
        # the same "good enough, not provably instant" tradeoff closeEvent
        # below makes explicitly for the same reason. Lives on each Pane
        # (pane.pending_tasks), not here -- results must never cross
        # between panes, and neither should their bookkeeping.
        #
        # request_id is GLOBAL (drawn from this one counter), not per-pane,
        # even though each pane only ever compares against its OWN latest
        # id (see Pane.request_id) -- a globally-unique id is what lets
        # _pane_for_render_id resolve which pane a signal's request_id
        # belongs to with a plain pending_tasks-membership scan, including
        # for a STALE id no pane currently considers current.
        self._next_render_id = 0

        # One debounce timer PER PANE -- an ambient edit (scroll/drag) on
        # one pane must not restart, or be satisfied by, a render of the
        # OTHER pane. Keyed by the Pane object itself (plain identity
        # hashing is fine -- Pane defines no __eq__/__hash__).
        self._debounce_timers: dict[Pane, QTimer] = {}
        for pane in self.panes:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(RENDER_DEBOUNCE_MS)
            timer.timeout.connect(lambda p=pane: self._start_render(p))
            self._debounce_timers[pane] = timer

        self._build_ui()
        for pane in self.panes:
            pane.image_view.refresh_critical_points()
        self._update_status_bar()
        for pane in self.panes:
            self._start_render(pane)

    def _build_ui(self) -> None:
        self.image_view = ImageView(self.pane, self.session, self)
        self.pane.image_view = self.image_view
        self.image_view2 = ImageView(self.pane2, self.session, self)
        self.pane2.image_view = self.image_view2

        # Every signal below is wired TWICE, once per pane's own ImageView,
        # each through a pane-capturing lambda -- the same pattern
        # _start_render(pane) already uses for render-result routing (see
        # its own comment on Pane.pending_tasks). Every handler now takes
        # the originating pane explicitly instead of assuming self.pane, so
        # a signal from EITHER pane reaches the same code and acts on the
        # right one.
        for pane, view in ((self.pane, self.image_view), (self.pane2, self.image_view2)):
            view.viewport_changed.connect(lambda p=pane: self._on_viewport_changed(p))
            view.cursor_readout_changed.connect(self._on_cursor_readout_changed)
            # No pane-capturing lambda needed here -- mousePressEvent
            # always emits pane_activated (moving focus) BEFORE a release
            # can ever emit param_changed, so by the time this fires,
            # self._focused_pane is already the clicked pane; see
            # _on_param_changed_by_click, which reads it as "the clicked pane."
            view.param_changed.connect(self._on_param_changed_by_click)
            # Keeps the "Orbit seed z0" field showing the CURRENT z0
            # whenever one is seeded via a dynamical-plane click (P6's
            # "symmetry of input": the click path must populate the
            # field, same as the field committing a value must seed an
            # orbit -- see _on_z0_field_committed). Also fires on
            # step/clear, both harmless no-ops here: z0 itself never
            # changes mid-orbit, and clear leaves state None, which this
            # deliberately does NOT react to -- z0 is a persistent,
            # independently-chosen seed, and Clear only stops DRAWING it,
            # it does not retract the choice (see
            # OrbitTracker.recompute_current's own docstring for the same
            # principle from the other direction).
            view.orbit_changed.connect(lambda p=pane: self._on_orbit_changed(p))
            # Moves focus to whichever pane the user actually pressed on
            # -- Reset View, the metadata header, and the status bar all
            # act on the focused pane (see _set_focused_pane).
            view.pane_activated.connect(lambda p=pane: self._set_focused_pane(p))

        self.orbit_panel = OrbitPanel(self.session, self.image_view, self)
        self.metadata_header = MetadataHeader(self.session, self.pane, self)

        # PER-PANE mode combo -- replaces the old single window-level
        # self.mode_combo. self.mode_combo is now specifically pane A's
        # (kept under that name for the many single-pane call sites/tests
        # that still mean "the" pane); self.mode_combo2 is pane B's, wired
        # identically. Neither one is more "correct" than the other --
        # Stage 3's coupling keys off render_mode, never off which combo
        # this is.
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(RENDER_MODES)
        self.mode_combo.setCurrentText(self.pane.render_mode)
        self.mode_combo.currentTextChanged.connect(lambda m: self._on_mode_changed(self.pane, m))

        self.mode_combo2 = QComboBox(self)
        self.mode_combo2.addItems(RENDER_MODES)
        self.mode_combo2.setCurrentText(self.pane2.render_mode)
        self.mode_combo2.currentTextChanged.connect(
            lambda m: self._on_mode_changed(self.pane2, m))

        self.pane_column = self._build_pane_column(self.mode_combo, self.image_view)
        self.pane_column2 = self._build_pane_column(self.mode_combo2, self.image_view2)
        self._pane_columns: dict[Pane, QWidget] = {
            self.pane: self.pane_column, self.pane2: self.pane_column2}
        self._mode_combos: dict[Pane, QComboBox] = {
            self.pane: self.mode_combo, self.pane2: self.mode_combo2}

        # Side by side in a splitter, not two fixed halves -- the user can
        # drag the divider to give one pane more room. Coupled (both
        # visible) is the natural instrument default (see
        # self.coupled_checkbox below); single-view collapses to just
        # whichever pane is currently focused (see _relayout_panes).
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.view_splitter.addWidget(self.pane_column)
        self.view_splitter.addWidget(self.pane_column2)

        # The View tab holds a small container -- the metadata header
        # above the splitter, the orbit-tracking strip below it -- not the
        # bare ImageView(s). The header sits directly above what it
        # describes so a screenshot of just this tab is self-explanatory
        # (P5c's own reason for building it at all); the orbit panel is
        # meaningless without the image it overlays and needs to stay
        # visible while the user clicks an image to seed a new orbit --
        # see _sync_orbit_panel for which pane it currently overlays.
        self.view_container = QWidget(self)
        view_layout = QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self.metadata_header)
        view_layout.addWidget(self.view_splitter, 1)
        view_layout.addWidget(self.orbit_panel)

        # A QTabWidget as the central widget, not the bare view container --
        # this is what makes room for the Settings tab (and whatever tab
        # comes after it) without redesigning the window. The toolbar stays
        # a QMainWindow-level toolbar, not per-tab: Reset View only means
        # anything on the View tab, but QMainWindow toolbars are independent
        # of which central-widget tab is showing either way.
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.view_container, "View")
        self.term_editor_panel = TermEditorPanel(self.session, self._on_term_edited, self)
        self.tabs.addTab(self.term_editor_panel, "Terms")
        self.facts_panel = FactsPanel(self.session, self._on_center_view, self)
        self.tabs.addTab(self.facts_panel, "Facts")
        self.library_panel = LibraryPanel(self.session, self._on_family_loaded,
                                          self._on_library_changed, self)
        self.tabs.addTab(self.library_panel, "Library")
        self.settings_panel = SettingsPanel(self.session, self._on_settings_applied, self)
        self.tabs.addTab(self.settings_panel, "Settings")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        toolbar = QToolBar("Controls", self)
        toolbar.setMovable(False)

        # Coupled (both panes visible) is the natural instrument default
        # -- see _relayout_panes for what unchecking this actually does
        # (collapse to just the focused pane, not always pane A).
        self.coupled_checkbox = QCheckBox("Coupled view", self)
        self.coupled_checkbox.setChecked(True)
        self.coupled_checkbox.toggled.connect(lambda _checked: self._relayout_panes())
        toolbar.addWidget(self.coupled_checkbox)

        # Two always-visible complex-number fields -- NOT buried in the
        # Terms tab -- the typed half of P6's "symmetry of input": a and
        # z0 are each settable by typing here OR by clicking a plane (see
        # ImageView.mouseReleaseEvent/param_changed and _on_orbit_changed),
        # and the two methods stay in sync through _apply_param_change and
        # _on_z0_field_committed respectively, never diverging.
        self.param_field = ComplexField("a =", self.session.param, self)
        self.param_field.committed.connect(self._on_param_field_committed)
        toolbar.addWidget(self.param_field)
        # 0+0j is just a neutral starting DISPLAY -- it does NOT mean an
        # orbit is seeded there; nothing is seeded until the user commits
        # this field or clicks the dynamical plane (see _on_orbit_changed,
        # which is what actually keeps this field truthful once a real
        # orbit exists).
        self.z0_field = ComplexField("z0 =", complex(0, 0), self)
        self.z0_field.committed.connect(self._on_z0_field_committed)
        toolbar.addWidget(self.z0_field)

        reset_button = QPushButton("Reset View", self)
        reset_button.clicked.connect(self._reset_view)
        toolbar.addWidget(reset_button)

        # GLOBAL, not per-pane -- acceptable for now (per-pane would be
        # more correct: e.g. tracing orbits only on the pane you're
        # actually looking at), but these three toggles are visual
        # preferences a user sets once and expects to stick regardless of
        # which pane they're looking at, and per-pane versions would mean
        # SIX checkboxes crowding the toolbar instead of three. Revisit if
        # that tradeoff stops holding.
        self.critical_points_checkbox = QCheckBox("Critical Points", self)
        self.critical_points_checkbox.toggled.connect(self.image_view.set_show_critical_points)
        self.critical_points_checkbox.toggled.connect(self.image_view2.set_show_critical_points)
        toolbar.addWidget(self.critical_points_checkbox)
        self.trace_orbits_checkbox = QCheckBox("Trace Orbits", self)
        self.trace_orbits_checkbox.toggled.connect(self.image_view.set_trace_orbits)
        self.trace_orbits_checkbox.toggled.connect(self.image_view2.set_trace_orbits)
        toolbar.addWidget(self.trace_orbits_checkbox)
        self.orbit_connect_lines_checkbox = QCheckBox("Connect orbit points", self)
        self.orbit_connect_lines_checkbox.setChecked(True)
        self.orbit_connect_lines_checkbox.toggled.connect(self.image_view.set_orbit_connect_lines)
        self.orbit_connect_lines_checkbox.toggled.connect(self.image_view2.set_orbit_connect_lines)
        toolbar.addWidget(self.orbit_connect_lines_checkbox)

        self.addToolBar(toolbar)

        # A full reproducible EXPERIMENT (map + parameter + view + mode +
        # render settings + orbit) -- separate from the Library tab, which
        # stores named MAPS only. See app.session.Session.snapshot_to_dict's
        # own docstring for why the two are not the same concept.
        #
        # Stored on self for the SAME reason help_menu/about_action are
        # below -- see that comment.
        self.file_menu = self.menuBar().addMenu("File")
        self.save_experiment_action = self.file_menu.addAction("Save Experiment...")
        self.save_experiment_action.triggered.connect(self._on_save_experiment)
        self.open_experiment_action = self.file_menu.addAction("Open Experiment...")
        self.open_experiment_action.triggered.connect(self._on_open_experiment)

        # On macOS, Qt moves a menu titled "Help" (and any action inside
        # named "About <AppName>") into the system application menu
        # automatically -- this is the standard, idiomatic way to get a
        # real "About ComplexDynamics" entry there, not a bespoke button.
        #
        # Stored on self DELIBERATELY, not left as a local -- addMenu()/
        # addAction() return a wrapper PySide6 can garbage-collect (deleting
        # the underlying C++ QMenu/QAction with it) once nothing in Python
        # still references it, even though the menu bar's own C++ parent-
        # child tree still lists it visually. Confirmed as a REAL bug, not
        # a hypothetical one: a local-only `help_menu` here made a later
        # `window.menuBar().actions()[...].menu()` raise "Internal C++
        # object (QMenu) already deleted" the moment this method returned.
        self.help_menu = self.menuBar().addMenu("Help")
        self.about_action = self.help_menu.addAction(f"About {PRODUCT_NAME}")
        self.about_action.triggered.connect(self._show_about_dialog)

        self.statusBar().showMessage("")

        self._update_pane_focus_styling()
        self._sync_orbit_panel()

    def _build_pane_column(self, mode_combo: QComboBox, image_view: "ImageView") -> QWidget:
        """One pane's own little control strip (mode combo) stacked above
        its ImageView -- what actually goes side by side in the splitter.
        """
        column = QWidget(self)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:", self))
        row.addWidget(mode_combo)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(image_view, 1)
        return column

    def _show_about_dialog(self) -> None:
        AboutDialog(self).exec()

    # ---- focus / layout: which pane Reset View, the header, and single- ---------
    # ---- view mode act on ---------------------------------------------------------
    def _set_focused_pane(self, pane: Pane) -> None:
        if pane is self._focused_pane:
            return
        self._focused_pane = pane
        self.metadata_header.pane = pane
        self.metadata_header.refresh()
        self._update_status_bar()
        self._update_pane_focus_styling()
        if not self.coupled_checkbox.isChecked():
            self._relayout_panes()

    def _update_pane_focus_styling(self) -> None:
        # Purely visual -- which column has a highlighted border -- so the
        # user can actually SEE which pane Reset View/typed-field actions
        # (still self.pane-directed until Stage 3) are about to act on.
        for pane, column in self._pane_columns.items():
            column.setStyleSheet(
                "border: 2px solid palette(highlight);" if pane is self._focused_pane else "")

    def _relayout_panes(self) -> None:
        # Coupled: both panes visible, side by side. Single: only the
        # FOCUSED pane's column is visible -- not always pane A -- the
        # other pane keeps rendering/tracking its own state in the
        # background (nothing about it is torn down), it's just hidden.
        coupled = self.coupled_checkbox.isChecked()
        for pane, column in self._pane_columns.items():
            column.setVisible(coupled or pane is self._focused_pane)

    # ---- orbit strip: follows whichever pane is currently dynamical -------------
    def _current_dynamical_pane(self) -> Pane | None:
        # Prefers the FOCUSED pane when it qualifies (most likely the one
        # the user is about to click to seed/step an orbit on); otherwise
        # falls back to scanning self.panes so a dynamical pane that isn't
        # focused is still found. None if neither pane is currently in a
        # dynamical mode.
        if self._focused_pane.render_mode not in PARAMETER_PLANE_MODES:
            return self._focused_pane
        for pane in self.panes:
            if pane.render_mode not in PARAMETER_PLANE_MODES:
                return pane
        return None

    def _sync_orbit_panel(self) -> None:
        dynamical_pane = self._current_dynamical_pane()
        self.orbit_panel.setEnabled(dynamical_pane is not None)
        if dynamical_pane is not None:
            self.orbit_panel.set_image_view(dynamical_pane.image_view)

    # ---- experiment snapshots: File > Save/Open Experiment... -------------------
    # MODAL DIALOGS AREN'T DIRECTLY TESTABLE (QFileDialog blocks on a real
    # event loop) -- split the same way app.library_panel's own save/load
    # buttons are: the button's own slot gathers a path via the dialog,
    # then hands off to a _do_* method containing the actual logic. Tests
    # call _do_save_experiment/_do_open_experiment directly with an
    # explicit path, the same way app/test_library_panel.py drives
    # LibraryPanel's mutating methods directly rather than through a
    # QPushButton click.
    def _on_save_experiment(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Experiment", "", "ComplexDynamics experiment (*.cdsx);;All files (*)")
        if path:
            self._do_save_experiment(path)

    def _do_save_experiment(self, path: str) -> None:
        # EVERY pane (Stage 4): layout (coupled flag, which pane is
        # focused) plus each pane's own (viewport, render_mode). The
        # orbit, if any, belongs to whichever pane orbit_panel currently
        # follows (_current_dynamical_pane) -- orbit tracking is
        # per-pane, so a plain (z0, n) alone would be ambiguous about
        # WHICH pane it came from; see Session.snapshot_to_dict's own
        # docstring for the (pane_index, z0, n) shape this becomes.
        if not path.endswith(".cdsx"):
            path += ".cdsx"
        panes = [(p.viewport, p.render_mode) for p in self.panes]
        focused_index = self.panes.index(self._focused_pane)
        coupled = self.coupled_checkbox.isChecked()
        orbit_pane = self._current_dynamical_pane()
        orbit = None
        if orbit_pane is not None:
            state = orbit_pane.image_view.orbit_tracker.state
            if state is not None:
                orbit = (self.panes.index(orbit_pane), state.z0, state.n)
        try:
            self.session.save_snapshot(path, panes, focused_index, coupled, orbit)
        except OSError as exc:
            QMessageBox.critical(self, "Save Experiment failed", str(exc))

    def _on_open_experiment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Experiment", "", "ComplexDynamics experiment (*.cdsx);;All files (*)")
        if path:
            self._do_open_experiment(path)

    def _do_open_experiment(self, path: str) -> None:
        # session.load_snapshot fully validates and parses BEFORE mutating
        # anything (see restore_from_snapshot's own docstring) -- on
        # failure the live session is untouched, so there is nothing here
        # to roll back, just an error to surface. Session has no panes,
        # viewport, or render_mode of its own, so the restored
        # (panes, focused_index, coupled, orbit) come back explicitly for
        # THIS window to apply to its own panes/layout/tracker.
        try:
            panes_data, focused_index, coupled, orbit = self.session.load_snapshot(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Open Experiment failed", str(exc))
            return

        # Apply viewport + mode to EVERY pane BEFORE touching any orbit, so
        # the dynamical-plane-only gate below sees each pane's RESTORED
        # mode, not whatever was showing before this load (the ordering
        # this whole method exists to get right: restore the modes, THEN
        # the orbit, never the other way around). A file with fewer panes
        # than this window has (can't happen from this app's own Save, but
        # a hand-edited file could) reuses that file's LAST pane entry for
        # any pane it doesn't cover, rather than crashing on a short list.
        for i, pane in enumerate(self.panes):
            viewport, mode = panes_data[min(i, len(panes_data) - 1)]
            pane.viewport = viewport
            pane.set_render_mode(mode)   # already validated by load_snapshot; re-validated here
                                         # for free, not re-checked for its own sake
            self._mode_combos[pane].setCurrentText(pane.render_mode)

        self.coupled_checkbox.setChecked(coupled)
        self._set_focused_pane(self.panes[min(focused_index, len(self.panes) - 1)])
        self.param_field.set_value(self.session.param)

        # Orbit reconstruction goes through the tracker's REAL seed()/step()
        # API, not a shortcut -- z and history regenerate through the exact
        # same code path a live, click-seeded orbit does. Every OTHER pane
        # (no saved orbit, or a parameter-plane mode after restore) gets
        # clear_orbit() -- the same gate _seed_orbit_at itself uses (orbit
        # tracking is a dynamical-plane concept).
        for i, pane in enumerate(self.panes):
            if (orbit is not None and orbit[0] == i
                    and pane.render_mode not in PARAMETER_PLANE_MODES):
                _, z0, n = orbit
                pane.image_view.orbit_tracker.seed(self.session.map, self.session.param, z0)
                pane.image_view.step_orbit(n)
            else:
                pane.image_view.clear_orbit()

        # The same full resync _on_family_loaded already does for a
        # wholesale map replacement -- a snapshot changes just as much at
        # once (map AND param AND every pane's viewport/mode AND settings),
        # so every panel that caches a view of the old state needs the
        # same treatment, not a partial refresh.
        self.term_editor_panel.refresh_from_session()
        self.facts_panel.refresh()
        self.metadata_header.refresh()
        for pane in self.panes:
            pane.image_view.refresh_critical_points()
            if pane is self._focused_pane:
                self._update_status_bar()
            self._debounce_timers[pane].stop()
            self._start_render(pane)
        self._sync_orbit_panel()

    # ---- parameter a: field commit and parameter-plane click, kept in sync ------
    def _on_param_field_committed(self, a: complex) -> None:
        # A typed field commit never switches or targets any ONE pane --
        # session.param is shared, so it invalidates every DYNAMICAL pane
        # (see the shared _apply_param_change below) and leaves every
        # parameter-plane pane's own plane/viewport alone, exactly the
        # field-commit behavior Stage 1/2 already had, just no longer
        # limited to the single (then-only) pane.
        self._apply_param_change(a)

    def _on_param_changed_by_click(self, a: complex) -> None:
        # mousePressEvent already emitted pane_activated before this can
        # ever fire (see _build_ui's own comment), so self._focused_pane
        # is the CLICKED pane.
        clicked_pane = self._focused_pane
        if not self.coupled_checkbox.isChecked():
            # SINGLE-VIEW fallback: only one pane is actually visible, so
            # there is no "partner" to drive -- keep today's pre-Stage-3
            # behavior exactly, the same as a P6 parameter-plane click
            # always had: set a AND switch the one visible pane straight
            # to the dynamical (julia) plane.
            self.session.param = a
            self.param_field.set_value(a)
            self.facts_panel.refresh()
            self.metadata_header.refresh()
            if clicked_pane.render_mode in PARAMETER_PLANE_MODES:
                # Triggers _on_mode_changed, which itself renders -- so
                # this does NOT also call _start_render below; doing both
                # would fire two render requests for one user action.
                self._mode_combos[clicked_pane].setCurrentText("julia")
            else:
                self._apply_param_change(a)   # already dynamical: just the ordinary invalidation
            return

        # COUPLED: this is what makes it an instrument rather than two
        # windows. The CLICKED pane does NOT switch plane and does NOT
        # reset its own viewport (its marker just moves, via the ordinary
        # repaint _apply_param_change already triggers for every
        # parameter-plane pane) -- only the PARTNER pane, if it's
        # currently dynamical, gets its viewport reset to this map's
        # dynamical-plane default (default_dynamical_view, Stage 2 --
        # never the parameter-plane's own table) and re-rendered. See
        # _apply_param_change's own docstring for why this is exactly the
        # same underlying invalidation a plain field commit also does.
        self._apply_param_change(a)

    def _apply_param_change(self, a: complex) -> None:
        """The ONE place session.param is ever assigned from user input --
        the field commit and every parameter-plane click (coupled or
        single-view-already-dynamical) funnel through here, so they
        cannot diverge (P6's own requirement, extended to N panes).

        CACHE ASYMMETRY (Stage 3): render_parameter ignores the bound
        param entirely -- every pixel there IS a parameter -- so a param
        change must invalidate/re-render ONLY panes currently in a
        DYNAMICAL mode: their viewport resets to this map's own
        DYNAMICAL-plane default (default_view_for_mode -> critical-/
        fixed-point-derived, see app.library_panel.default_dynamical_view;
        Stage 2 -- a deep zoom from the PREVIOUS parameter is almost never
        informative for a new one, and the parameter-plane's OWN table
        would frame the wrong space entirely) and they re-render. A
        parameter-plane pane's
        pixels don't need to change at all -- it only needs an .update()
        repaint so its marker (see ImageView._param_marker_pixel, derived
        from session.param at paint time) moves to the new value. Neither
        the clicked pane (if this came from a click) nor any other
        parameter-plane pane has its OWN viewport touched or plane
        switched here -- that is exactly Stage 3's "do NOT reset the
        clicked (parameter) pane's own viewport" requirement, and it
        falls out for free by just... not touching it.
        """
        self.session.param = a
        self.param_field.set_value(a)
        self.facts_panel.refresh()
        self.metadata_header.refresh()
        for pane in self.panes:
            if pane.render_mode in PARAMETER_PLANE_MODES:
                pane.image_view.update()   # move its marker; nothing else about it changed
                continue
            center, scale = default_view_for_mode(self.session.map, self.session.param,
                                                  pane.render_mode)
            vp = pane.viewport
            pane.viewport = cdx.Viewport(center, scale, vp.resolution)
            pane.image_view.refresh_critical_points()
            pane.image_view.refresh_orbit_for_new_param()
            if pane is self._focused_pane:
                self._update_status_bar()
            self._debounce_timers[pane].stop()
            self._start_render(pane)

    # ---- orbit seed z0: field commit mirrors a dynamical-plane click exactly ----
    def _on_z0_field_committed(self, z0: complex) -> None:
        # Acts on the FOCUSED pane -- same gate _seed_orbit_at itself uses
        # -- orbit tracking is a dynamical-plane concept; committing a z0
        # while the focused pane is looking at the parameter plane is
        # silently a no-op, matching what clicking the (nonexistent, on
        # that plane) orbit target would also do.
        pane = self._focused_pane
        if pane.render_mode in PARAMETER_PLANE_MODES:
            return
        pane.image_view.orbit_tracker.seed(self.session.map, self.session.param, z0)
        pane.image_view.orbit_changed.emit()
        pane.image_view.update()

    def _on_orbit_changed(self, pane: Pane) -> None:
        # Keeps the z0 FIELD showing the current z0 whenever a real orbit
        # exists -- fires on seed/step/clear alike; step leaves z0
        # unchanged (a harmless re-set to the same value) and clear
        # deliberately does nothing here (see this connection's own
        # comment in _build_ui for why the field outlives Clear). The z0
        # field describes the FOCUSED pane specifically -- an orbit change
        # on the OTHER, unfocused pane does not steal the field's display.
        if pane is not self._focused_pane:
            return
        state = pane.image_view.orbit_tracker.state
        if state is not None:
            self.z0_field.set_value(state.z0)

    # ---- mode: immediate re-render, same deliberate-action treatment as Apply ---
    def _on_mode_changed(self, pane: Pane, mode: str) -> None:
        pane.set_render_mode(mode)
        if pane is self._focused_pane:
            self.metadata_header.refresh()   # domain/mode text depends on the mode itself
            self._update_status_bar()
        self._sync_orbit_panel()   # this pane may have just become (or stopped being) dynamical
        self._debounce_timers[pane].stop()
        self._start_render(pane)

    # ---- settings: apply on demand, from the Settings tab -----------------------
    def _on_settings_applied(self, new_settings: Settings) -> None:
        # Already validated by SettingsPanel before this is ever called
        # (see its _apply) -- this just carries the effects: update the
        # live session (render_settings/cache budget -- see
        # Session.apply_settings), persist for next launch, and render
        # right away rather than waiting for the ordinary debounce. An
        # explicit Apply click is exactly the kind of deliberate action the
        # debounce (meant to coalesce rapid, ambient events like scrolling)
        # isn't for.
        #
        # Resolution is a per-pane viewport field now (session no longer
        # owns a viewport at all -- see app.pane.Pane), so THIS loop is
        # what actually applies a resolution change to every visible pane,
        # keeping each one's own center/scale untouched -- Settings does
        # not own WHERE the user is looking, only how it's rendered.
        self.session.apply_settings(new_settings)
        for pane in self.panes:
            vp = pane.viewport
            pane.viewport = cdx.Viewport(vp.center, vp.scale, new_settings.resolution)
        save_settings(new_settings)
        self._update_status_bar()
        for pane in self.panes:
            self._debounce_timers[pane].stop()
            self._start_render(pane)

    # ---- term edits: live but debounced, reusing the viewport-change path -------
    def _on_term_edited(self) -> None:
        # Deliberately the SAME debounced path viewport drag/zoom uses, not
        # an immediate render the way Settings' explicit Apply is: term
        # edits are meant to feel live (nudge a coefficient, watch the
        # picture update), and a burst of edits -- dragging a spinbox,
        # typing several digits into a coefficient cell -- should coalesce
        # into one render the same way a scroll burst does, not fire one
        # render per keystroke. TermEditorPanel has already validated and
        # applied the edit to self.session.map by the time this is called;
        # this only has to trigger the render side.
        #
        # session.map is SHARED and UNCONDITIONALLY affects every pane's
        # pixels regardless of mode (unlike a param change, which a
        # parameter-plane pane ignores entirely -- see
        # _apply_param_change's own cache-asymmetry docstring) -- so this
        # debounces BOTH panes, not just the focused one.
        self._update_status_bar()
        self.metadata_header.refresh()   # name/formula depend directly on the edited map
        # Cheap: refresh() only recomputes when (map, param) actually
        # changed since its last call, so this stays fresh for whenever the
        # user next looks at the Facts tab without paying for a recompute
        # on every keystroke's worth of edits.
        self.facts_panel.refresh()
        for pane in self.panes:
            self._debounce_timers[pane].start()
            # Same memoization property -- see ImageView.refresh_critical_points.
            pane.image_view.refresh_critical_points()
            # An orbit traced under the PRE-edit map no longer describes
            # this map's dynamics -- see OrbitTracker.reset_if_stale's own
            # docstring.
            pane.image_view.refresh_orbit_staleness()

    # ---- facts tab: recompute lazily, and re-check on becoming visible ----------
    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.facts_panel:
            self.facts_panel.refresh()

    # ---- facts tab: clicking a listed point centres the view on it --------------
    def _on_center_view(self, point: complex) -> None:
        pane = self._focused_pane
        vp = pane.viewport
        pane.viewport = cdx.Viewport(point, vp.scale, vp.resolution)
        self.tabs.setCurrentWidget(self.view_container)
        self._update_status_bar()
        self._debounce_timers[pane].stop()
        self._start_render(pane)

    # ---- library tab: loading a family replaces session.map wholesale -----------
    def _on_family_loaded(self) -> None:
        # LibraryPanel has already replaced self.session.map by the time
        # this fires (see its _load_selected) but no longer touches any
        # viewport itself (Session has none to touch -- see app.pane.Pane)
        # -- resetting EVERY pane's viewport to the new map's own
        # MODE-appropriate default (default_view_for_mode, Stage 2 -- a
        # parameter-plane pane gets a-space framing, a dynamical one gets
        # z-space framing derived from the new map's own critical/fixed
        # points) is this method's own job now, before resyncing every
        # OTHER panel
        # that caches a view of the old map, then rendering right away,
        # the same "deliberate action, not a debounce-worthy burst"
        # treatment Reset View and Settings' Apply already get. Every
        # pane, not just the focused one -- session.map is SHARED and
        # UNCONDITIONALLY affects every pane regardless of mode (see
        # _apply_param_change's own cache-asymmetry docstring for the
        # contrast with a plain param change).
        self.term_editor_panel.refresh_from_session()
        self.facts_panel.refresh()
        self.metadata_header.refresh()   # name/formula/param all just changed wholesale
        for pane in self.panes:
            center, scale = default_view_for_mode(self.session.map, self.session.param,
                                                  pane.render_mode)
            pane.viewport = cdx.Viewport(center, scale, pane.viewport.resolution)
            pane.image_view.refresh_critical_points()
            pane.image_view.refresh_orbit_staleness()
            if pane is self._focused_pane:
                self._update_status_bar()
            self._debounce_timers[pane].stop()
            self._start_render(pane)

    # ---- library tab: any successful save/rename/delete/notes edit persists -----
    def _on_library_changed(self) -> None:
        self.session.save_user_library(library_path())

    # ---- viewport change -> debounced render ------------------------------------
    def _on_viewport_changed(self, pane: Pane) -> None:
        if pane is self._focused_pane:
            self._update_status_bar()
        # A render is normally debounced -- but if the visible viewport has
        # drifted far enough into the last buffer's overscan margin, waiting
        # out the debounce risks running off the buffer's real pixels before
        # a fresh one lands (a long continuous drag/scroll restarts the
        # debounce on every event and might never let it fire on its own).
        # PER PANE: one pane drifting past its own buffer's margin starts
        # only ITS timer/render, never the other pane's.
        if pane.image_view.buffer_edge_fraction() > BUFFER_EDGE_FRACTION:
            self._debounce_timers[pane].stop()
            self._start_render(pane)
        else:
            self._debounce_timers[pane].start()   # restarts if already running

    def _reset_view(self) -> None:
        # Center/scale only -- NOT resolution. Resolution is a Settings
        # field now (see app/settings.py), independent of where the user
        # is looking; Reset View undoes pan/zoom, not an Apply from the
        # Settings tab. Using the pane's CURRENT resolution (whatever
        # Settings last applied) keeps that concern from fighting this one.
        #
        # Resets to this map's own MODE-appropriate default framing
        # (default_view_for_mode) -- the FOCUSED pane's own render_mode
        # decides which table applies: default_view_for's a-space table
        # (DEFAULT_PARAMETER_VIEW_CENTER/SCALE is exactly
        # default_view_for("mandelbrot")) for a parameter-plane pane,
        # default_dynamical_view's critical-/fixed-point-derived z-space
        # framing for a dynamical one (_apply_param_change routes through
        # the identical dispatcher for exactly that purpose). Earlier this
        # used ONE table for both planes regardless of mode -- resetting a
        # Julia-set pane jumped it onto the PARAMETER plane's own window,
        # not a Julia-appropriate one; that was Stage 2's bug to fix. NOT a
        # frozen startup snapshot either, which is what this used to
        # restore before that and was a separate, earlier bug (a deep zoom
        # from a PREVIOUS session/param is never what "reset" should mean).
        pane = self._focused_pane
        center, scale = default_view_for_mode(self.session.map, self.session.param,
                                              pane.render_mode)
        pane.viewport = cdx.Viewport(center, scale, pane.viewport.resolution)
        self._update_status_bar()
        self._debounce_timers[pane].stop()
        self._start_render(pane)

    # ---- render dispatch -----------------------------------------------------------
    # PER-PANE: request-id/pending-tasks bookkeeping lives on the pane
    # itself (see app.pane.Pane), not here -- a stale/superseded result
    # for one pane must never paint a different one, and this is what
    # makes that true even once a second pane exists (Stage 2). The
    # thread pool stays shared (one app-wide resource); each RenderTask's
    # signals are connected to plain BOUND METHODS of this window (a
    # QObject living on the GUI thread), never to a context-less lambda --
    # see RenderSignals' own docstring for the crash that caused. The
    # handlers below resolve which pane a signal belongs to via
    # _pane_for_render_id (request_id is GLOBALLY unique -- see
    # self._next_render_id) rather than a lambda closing over `pane`.
    def _pane_for_render_id(self, request_id: int) -> Pane | None:
        for pane in self.panes:
            if request_id in pane.pending_tasks:
                return pane
        return None   # already cleaned up (or, in a test, never dispatched by this window at all)

    def _start_render(self, pane: Pane) -> None:
        # Every still-pending task FOR THIS PANE is now stale -- cancel it
        # immediately rather than letting it run to completion only to be
        # discarded by the request_id check. Also sweep away entries left
        # over from an EARLIER round that were cancelled then: see the
        # comment on Pane.pending_tasks for why cleanup happens here
        # instead of via a signal from the (silent, on cancellation) task
        # itself. A different pane's pending tasks are untouched --
        # starting a render for one pane must never cancel another's.
        for stale_id, stale_task in list(pane.pending_tasks.items()):
            if stale_task.cancel.is_cancelled:
                del pane.pending_tasks[stale_id]
                # This IS the GUI thread (every _start_render caller is a
                # GUI-thread handler), so this deletion already satisfies
                # RenderSignals' "destroyed on the GUI thread" requirement
                # on its own -- deleteLater() is still used, not a bare
                # `del`, purely for the same defense-in-depth reason
                # _on_full_ready/_on_render_failed use it below: it posts
                # the actual C++ teardown through Qt's own event queue
                # instead of depending on Python refcount timing.
                stale_task.signals.deleteLater()
            else:
                stale_task.cancel.cancel()

        if pane.image_view.no_effect_parameter_message() is not None:
            # A parameter-plane pane for a map where `a` has no effect at
            # all -- every pixel would render identically, so there is
            # nothing meaningful to COMPUTE, not just nothing meaningful
            # to display. Skip dispatching a RenderTask entirely; just
            # repaint the in-pane message (see ImageView.paintEvent).
            pane.image_view.update()
            return

        self._next_render_id += 1
        request_id = self._next_render_id
        pane.request_id = request_id

        vp = pane.viewport
        viewport_snapshot = cdx.Viewport(vp.center, vp.scale, vp.resolution)
        rs = self.session.render_settings
        settings_snapshot = cdx.RenderSettings(rs.max_iter, rs.escape_radius, rs.tol, rs.threads)

        task = RenderTask(request_id, self.session.map, self.session.param,
                          viewport_snapshot, settings_snapshot, pane.render_mode,
                          cdx.CancelToken(), self.session.cache)
        # Bound methods, not lambdas: PySide resolves `self` (this window,
        # a QObject on the GUI thread) as the receiver and correctly
        # queues delivery across threads -- see RenderSignals' docstring.
        task.signals.partial_ready.connect(self._on_partial_ready)
        task.signals.full_ready.connect(self._on_full_ready)
        task.signals.failed.connect(self._on_render_failed)
        pane.pending_tasks[request_id] = task
        self._thread_pool.start(_Runnable(task))

    def _on_partial_ready(self, request_id: int, payload, buffer_viewport: cdx.Viewport) -> None:
        pane = self._pane_for_render_id(request_id)
        if pane is None:
            return   # unknown/already-cleaned-up request -- nothing to do
        if request_id != pane.request_id:
            return   # superseded by a newer request for THIS pane; discard
        if not shiboken6.isValid(pane.image_view):
            return   # this pane's widget is already torn down (e.g. window closing)
        pane.image_view.set_image(payload, buffer_viewport)
        self._update_status_bar()

    def _on_full_ready(self, request_id: int, payload, buffer_viewport: cdx.Viewport) -> None:
        pane = self._pane_for_render_id(request_id)
        if pane is None:
            return
        # This IS the GUI thread (bound-method connection -- see
        # _start_render), so popping here (releasing the GUI side's own
        # reference) and the deleteLater() below both genuinely happen on
        # the GUI thread, closing the race RenderSignals' docstring
        # describes.
        task = pane.pending_tasks.pop(request_id, None)
        if task is not None:
            task.signals.deleteLater()
        if request_id != pane.request_id:
            return   # superseded by a newer request for THIS pane; discard
        if not shiboken6.isValid(pane.image_view):
            return
        pane.image_view.set_image(payload, buffer_viewport)
        self._update_status_bar()

    def _on_render_failed(self, request_id: int, message: str) -> None:
        pane = self._pane_for_render_id(request_id)
        if pane is None:
            return
        task = pane.pending_tasks.pop(request_id, None)
        if task is not None:
            task.signals.deleteLater()
        if request_id != pane.request_id:
            return
        if not shiboken6.isValid(self):
            return   # the window itself is already torn down
        self.statusBar().showMessage(f"render failed: {message}")

    # ---- status bar: scale + precision-floor warning ----------------------------
    def _update_status_bar(self) -> None:
        # Deliberately the EXPENSIVE half (constructs a cdx.Renderer just
        # to read precision_floor) -- called only at discrete state-change
        # points (viewport/settings/mode changes, a completed render),
        # never per mouse-move event. The cheap, per-mouse-move half lives
        # in _on_cursor_readout_changed below; both write into
        # _status_base_message/_status_cursor_text and _render_status_bar
        # combines them, so neither overwrites the other's half of the
        # displayed message. One shared status bar describes the FOCUSED
        # pane specifically (see _set_focused_pane, _on_viewport_changed,
        # _on_mode_changed) -- not a per-pane status bar of its own.
        vp = self._focused_pane.viewport
        renderer = cdx.Renderer(map=cdx.Map.custom(self.session.map, self.session.param),
                                viewport=vp, settings=self.session.render_settings)
        floor = renderer.precision_floor

        message = f"scale = {vp.scale:.6e}"
        if vp.scale <= floor:
            message += (f"   ⚠ AT the precision floor ({floor:.3e}) -- "
                        "neighbouring pixels are rounding to the same double; "
                        "this image is degenerate")
        elif vp.scale <= floor * PRECISION_WARN_MULTIPLE:
            message += f"   ⚠ approaching the precision floor ({floor:.3e})"
        self._status_base_message = message
        self._render_status_bar()

    def _on_cursor_readout_changed(self, text: str) -> None:
        self._status_cursor_text = text
        self._render_status_bar()

    def _render_status_bar(self) -> None:
        text = f"{self._status_base_message}   {self._status_cursor_text}" \
            if self._status_cursor_text else self._status_base_message
        self.statusBar().showMessage(text)

    # ---- shutdown ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        # Cancel everything still running and close right away, rather than
        # calling self._thread_pool.waitForDone() -- with cancellation,
        # per-column checking bounds a worker thread's remaining work to
        # roughly one column, not the rest of a possibly multi-second
        # render, so the residual window where a task could still be
        # touching Qt objects as they get torn down is milliseconds, not
        # seconds. RenderTask.run() emits nothing at all once cancelled (see
        # its own comment), which is what keeps that residual window from
        # being a real risk rather than just a shorter one.
        for pane in self.panes:
            self._debounce_timers[pane].stop()
            for task in pane.pending_tasks.values():
                task.cancel.cancel()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = SandboxWindow()
    window.show()
    if os.environ.get("CDX_VERIFY_LAUNCH_SELFTEST"):
        # A packaged build "launching" only means something if a real
        # window came up -- process survival alone doesn't prove Qt's
        # platform plugin loaded. Used by CI (see .github/workflows) as a
        # smoke test on every packaged artifact, not just at dev time.
        def _check() -> None:
            ok = window.isVisible() and window.windowTitle() == PRODUCT_NAME
            print("LAUNCH_SELFTEST_OK" if ok else "LAUNCH_SELFTEST_FAIL", flush=True)
            app.exit(0 if ok else 1)

        QTimer.singleShot(2000, _check)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
