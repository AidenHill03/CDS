"""app/sandbox.py -- P5a/P5b: the interactive window.

Single window, driven by app.session.Session, with tabs: the image pane
(render pipeline: threaded, progressive, cursor-anchored zoom, overscan
buffer, correct orientation), a term editor (app/term_editor_panel.py), a
read-only dynamical-facts panel (app/facts_panel.py), and a Settings panel
(app/settings_panel.py) for render/cache configuration.

Requires the cdx extension module and PySide6 to be importable, e.g. from
the repository root:

    PYTHONPATH=cdx/build python -m app.sandbox

Entry point: `python -m app.sandbox`.
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import (QObject, QPoint, QPointF, QRect, QRectF, QRunnable, QSize, Qt,
                            QThreadPool, QTimer, Signal, Slot)
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTabWidget, QToolBar,
                               QWidget)

import cdx
from app.facts_panel import FactsPanel
from app.render_cache import RenderCache
from app.session import Session, render_map
from app.settings import Settings, load_settings, save_settings
from app.term_editor_panel import TermEditorPanel
from app.settings_panel import SettingsPanel

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


def array_to_qimage(array: np.ndarray) -> QImage:
    """Converts a cdx render array into a displayable QImage.

    cdx render arrays have row 0 at the BOTTOM (matching Viewport::coord;
    see CLAUDE.md) but QImage has row 0 at the TOP -- flip vertically first,
    or preview and full renders would visibly disagree on orientation. This
    was a real bug in the MATLAB prototype (progressive rendering flipped
    the preview relative to the full render because the axis orientation
    was set once instead of after every draw call); flipping the array
    itself, once, here, is what rules that class of bug out entirely rather
    than relying on every caller remembering an origin='lower'-equivalent.

    Grayscale, min-max normalized. Deliberately simple -- proving the
    pipeline does not need a colormap.
    """
    flipped = np.flipud(array)
    lo = float(flipped.min())
    hi = float(flipped.max())
    if hi > lo:
        normalized = (flipped - lo) * (255.0 / (hi - lo))
    else:
        normalized = np.zeros_like(flipped)
    gray = np.ascontiguousarray(normalized.astype(np.uint8))
    height, width = gray.shape
    image = QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8)
    # .copy(): `gray` is a local array pybind11/numpy does not keep alive on
    # QImage's behalf. Without this, the buffer QImage points at is freed as
    # soon as this function returns, and the image is garbage or a crash.
    return image.copy()


class RenderSignals(QObject):
    """QRunnable isn't itself a QObject and so cannot emit signals; this is
    the companion object RenderTask uses instead. Created on the GUI thread
    (in RenderTask.__init__, before the task is handed to the thread pool),
    emitted from the worker thread -- Qt detects the cross-thread emission
    automatically and queues delivery to the GUI thread's event loop, which
    is what makes it safe to update widgets from the connected slots.
    """
    partial_ready = Signal(int, object, object)   # request_id, ndarray, buffer viewport
    full_ready = Signal(int, object, object)       # request_id, ndarray, buffer viewport
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

    INSTANT FEEDBACK. session.viewport updates immediately on every wheel
    or pan event, but the real re-render is debounced (see SandboxWindow).
    In between, the widget keeps showing the last PAINTED buffer -- but
    that buffer is OVERSCANNED (rendered wider than the viewport it was
    requested for; see _overscanned in this module), so it usually still
    has real pixels covering the new, moved viewport. set_image() stores
    the buffer's own viewport alongside its pixmap (_buffer_viewport);
    paintEvent recomputes, on every paint, exactly which sub-rect of that
    buffer the CURRENT session.viewport corresponds to (_buffer_source_rect)
    and draws only that sub-rect stretched to fill the display. Because
    this is recomputed fresh from the two known viewports every time rather
    than accumulated incrementally across events, five wheel notches or a
    long drag before the debounce fires need no special composition step --
    session.viewport already reflects all of them, and the mapping from it
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

    def __init__(self, session: Session, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

        self._pixmap: QPixmap | None = None
        # The viewport _pixmap was actually rendered for (wider than
        # session.viewport by the overscan factor -- see _overscanned).
        # None exactly when _pixmap is None; kept alongside it because
        # _buffer_source_rect needs both to map back onto the display.
        self._buffer_viewport: cdx.Viewport | None = None

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
        vp = self.session.viewport
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

    def _buffer_source_rect(self) -> QRectF | None:
        """The sub-rect of the last-painted buffer (in its own pixel space)
        that the CURRENT session.viewport corresponds to -- computed fresh
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
        vp = self.session.viewport
        buf = self._buffer_viewport
        d_re = abs(vp.center.real - buf.center.real) + vp.scale
        d_im = abs(vp.center.imag - buf.center.imag) + vp.scale
        return max(d_re, d_im) / buf.scale

    # ---- painting --------------------------------------------------------------
    def set_image(self, array: np.ndarray, buffer_viewport: cdx.Viewport) -> None:
        self._pixmap = QPixmap.fromImage(array_to_qimage(array))
        self._buffer_viewport = buffer_viewport
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        source_rect = self._buffer_source_rect()
        if source_rect is not None:
            painter.drawPixmap(QRectF(self._display_rect()), self._pixmap, source_rect)
        if self._rubber_band_rect is not None:
            pen = QPen(QColor(255, 255, 255))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(self._rubber_band_rect)

    # ---- scroll zoom, cursor-anchored -------------------------------------------
    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        notches = delta / 120.0
        factor = ZOOM_FACTOR_PER_NOTCH ** notches

        cursor_pos = event.position().toPoint()
        w = self._pixel_to_complex(cursor_pos)
        vp = self.session.viewport
        new_center = w - (w - vp.center) / factor
        new_scale = vp.scale / factor
        self.session.viewport = cdx.Viewport(new_center, new_scale, vp.resolution)

        # Instant feedback: repainting now re-derives the buffer source
        # rect from the viewport just set above -- see _buffer_source_rect.
        self.update()

        self.viewport_changed.emit()
        event.accept()

    # ---- drag: rubber-band zoom, or pan -----------------------------------------
    def mousePressEvent(self, event) -> None:
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
            vp = self.session.viewport
            current_point = self._pixel_to_complex(pos)
            # Shift the center so the point that was under the cursor at
            # drag start is again under the cursor now -- see the module
            # test suite for the derivation this formula comes from.
            new_center = vp.center + (self._pan_anchor_complex - current_point)
            self.session.viewport = cdx.Viewport(new_center, vp.scale, vp.resolution)
            self.update()

            self.viewport_changed.emit()
        elif self._rubber_band_origin is not None:
            self._rubber_band_rect = QRect(self._rubber_band_origin, pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self._pan_anchor_complex = None
            self.viewport_changed.emit()
        elif self._rubber_band_origin is not None:
            rect = self._rubber_band_rect
            self._rubber_band_origin = None
            self._rubber_band_rect = None
            self.update()
            if rect is not None and rect.width() > 4 and rect.height() > 4:
                self._apply_rubber_band_zoom(rect)

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
        vp = self.session.viewport
        self.session.viewport = cdx.Viewport(new_center, new_scale, vp.resolution)
        self.viewport_changed.emit()


class SandboxWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ComplexDynamics sandbox (P5a)")
        self.resize(800, 860)

        # Persisted settings (app/settings.py's config file, next to where a
        # saved family library would also live -- see app.settings.config_dir)
        # survive between runs; load_settings() degrades gracefully to plain
        # defaults if that file is missing, malformed, or partially invalid.
        self.session = Session(settings=load_settings())
        # Captured once, independent of session.viewport (a fresh Viewport,
        # not a reference to it) -- Reset View must restore exactly this
        # regardless of anything that has happened since, undo history or
        # not (there is no undo yet, but the point is this does not need one).
        vp0 = self.session.viewport
        self._initial_viewport = cdx.Viewport(vp0.center, vp0.scale, vp0.resolution)

        self._request_id = 0
        # A dedicated pool, not QThreadPool.globalInstance(): keeps this
        # app's renders independent of anything else in the process that
        # might use the global pool.
        self._thread_pool = QThreadPool()
        # RenderTask (and the RenderSignals QObject it owns) has no other
        # Python-side strong reference once _start_render() returns -- the
        # QRunnable adapter is only reachable from QThreadPool's C++ side.
        # Without this, Python's GC can (and does: this was a real crash,
        # not a theoretical one -- "RuntimeError: Signal source has been
        # deleted") collect the task while its worker thread is still
        # running and about to emit through it. Entries for a task that
        # completes (or fails) normally are removed when its final signal
        # fires. A CANCELLED task emits nothing at all (see RenderTask.run),
        # so its entry is instead swept on the NEXT _start_render() call,
        # once cancellation (which per-column checking bounds to roughly one
        # column's worth of work, not the full render) has had a full round
        # to actually finish -- the same "good enough, not provably instant"
        # tradeoff closeEvent below makes explicitly for the same reason.
        self._pending_tasks: dict[int, RenderTask] = {}

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(RENDER_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._start_render)

        self._build_ui()
        self._update_status_bar()
        self._start_render()

    def _build_ui(self) -> None:
        self.image_view = ImageView(self.session, self)
        self.image_view.viewport_changed.connect(self._on_viewport_changed)

        # A QTabWidget as the central widget, not the bare ImageView --
        # this is what makes room for the Settings tab (and whatever tab
        # comes after it) without redesigning the window. The toolbar stays
        # a QMainWindow-level toolbar, not per-tab: Reset View only means
        # anything on the View tab, but QMainWindow toolbars are independent
        # of which central-widget tab is showing either way.
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.image_view, "View")
        self.term_editor_panel = TermEditorPanel(self.session, self._on_term_edited, self)
        self.tabs.addTab(self.term_editor_panel, "Terms")
        self.facts_panel = FactsPanel(self.session, self._on_center_view, self)
        self.tabs.addTab(self.facts_panel, "Facts")
        self.settings_panel = SettingsPanel(self.session, self._on_settings_applied, self)
        self.tabs.addTab(self.settings_panel, "Settings")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        toolbar = QToolBar("Controls", self)
        toolbar.setMovable(False)
        reset_button = QPushButton("Reset View", self)
        reset_button.clicked.connect(self._reset_view)
        toolbar.addWidget(reset_button)
        self.addToolBar(toolbar)

        self.statusBar().showMessage("")

    # ---- settings: apply on demand, from the Settings tab -----------------------
    def _on_settings_applied(self, new_settings: Settings) -> None:
        # Already validated by SettingsPanel before this is ever called
        # (see its _apply) -- this just carries the effects: update the
        # live session (viewport.resolution/render_settings/cache budget --
        # see Session.apply_settings), persist for next launch, and render
        # right away rather than waiting for the ordinary debounce. An
        # explicit Apply click is exactly the kind of deliberate action the
        # debounce (meant to coalesce rapid, ambient events like scrolling)
        # isn't for.
        self.session.apply_settings(new_settings)
        save_settings(new_settings)
        self._update_status_bar()
        self._debounce_timer.stop()
        self._start_render()

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
        self._update_status_bar()
        self._debounce_timer.start()
        # Cheap: refresh() only recomputes when (map, param) actually
        # changed since its last call, so this stays fresh for whenever the
        # user next looks at the Facts tab without paying for a recompute
        # on every keystroke's worth of edits.
        self.facts_panel.refresh()

    # ---- facts tab: recompute lazily, and re-check on becoming visible ----------
    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.facts_panel:
            self.facts_panel.refresh()

    # ---- facts tab: clicking a listed point centres the view on it --------------
    def _on_center_view(self, point: complex) -> None:
        vp = self.session.viewport
        self.session.viewport = cdx.Viewport(point, vp.scale, vp.resolution)
        self.tabs.setCurrentWidget(self.image_view)
        self._update_status_bar()
        self._debounce_timer.stop()
        self._start_render()

    # ---- viewport change -> debounced render ------------------------------------
    @Slot()
    def _on_viewport_changed(self) -> None:
        self._update_status_bar()
        # A render is normally debounced -- but if the visible viewport has
        # drifted far enough into the last buffer's overscan margin, waiting
        # out the debounce risks running off the buffer's real pixels before
        # a fresh one lands (a long continuous drag/scroll restarts the
        # debounce on every event and might never let it fire on its own).
        if self.image_view.buffer_edge_fraction() > BUFFER_EDGE_FRACTION:
            self._debounce_timer.stop()
            self._start_render()
        else:
            self._debounce_timer.start()   # restarts if already running

    def _reset_view(self) -> None:
        # Center/scale only -- NOT resolution. Resolution is a Settings
        # field now (see app/settings.py), independent of where the user
        # is looking; Reset View undoes pan/zoom, not an Apply from the
        # Settings tab. Using the session's CURRENT resolution (whatever
        # Settings last applied), not _initial_viewport's, is what keeps
        # those two concerns from fighting each other.
        iv = self._initial_viewport
        self.session.viewport = cdx.Viewport(iv.center, iv.scale, self.session.viewport.resolution)
        self._update_status_bar()
        self._debounce_timer.stop()
        self._start_render()

    # ---- render dispatch ---------------------------------------------------------
    def _start_render(self) -> None:
        # Every still-pending task is now stale -- cancel it immediately
        # rather than letting it run to completion only to be discarded by
        # the request_id check. Also sweep away entries left over from an
        # EARLIER round that were cancelled then: see the comment on
        # _pending_tasks's declaration for why cleanup happens here instead
        # of via a signal from the (silent, on cancellation) task itself.
        for stale_id, stale_task in list(self._pending_tasks.items()):
            if stale_task.cancel.is_cancelled:
                del self._pending_tasks[stale_id]
            else:
                stale_task.cancel.cancel()

        self._request_id += 1
        request_id = self._request_id

        vp = self.session.viewport
        viewport_snapshot = cdx.Viewport(vp.center, vp.scale, vp.resolution)
        rs = self.session.render_settings
        settings_snapshot = cdx.RenderSettings(rs.max_iter, rs.escape_radius, rs.tol, rs.threads)

        task = RenderTask(request_id, self.session.map, self.session.param,
                          viewport_snapshot, settings_snapshot, self.session.render_mode,
                          cdx.CancelToken(), self.session.cache)
        task.signals.partial_ready.connect(self._on_partial_ready)
        task.signals.full_ready.connect(self._on_full_ready)
        task.signals.failed.connect(self._on_render_failed)
        self._pending_tasks[request_id] = task
        self._thread_pool.start(_Runnable(task))

    @Slot(int, object, object)
    def _on_partial_ready(self, request_id: int, array: np.ndarray,
                          buffer_viewport: cdx.Viewport) -> None:
        if request_id != self._request_id:
            return   # superseded by a newer request; discard
        self.image_view.set_image(array, buffer_viewport)

    @Slot(int, object, object)
    def _on_full_ready(self, request_id: int, array: np.ndarray,
                       buffer_viewport: cdx.Viewport) -> None:
        self._pending_tasks.pop(request_id, None)   # this task is done; safe to release
        if request_id != self._request_id:
            return
        self.image_view.set_image(array, buffer_viewport)

    @Slot(int, str)
    def _on_render_failed(self, request_id: int, message: str) -> None:
        self._pending_tasks.pop(request_id, None)
        if request_id != self._request_id:
            return
        self.statusBar().showMessage(f"render failed: {message}")

    # ---- status bar: scale + precision-floor warning ----------------------------
    def _update_status_bar(self) -> None:
        vp = self.session.viewport
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
        self.statusBar().showMessage(message)

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
        self._debounce_timer.stop()
        for task in self._pending_tasks.values():
            task.cancel.cancel()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = SandboxWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
