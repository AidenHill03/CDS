"""app/sandbox.py -- P5a: first interactive window.

Single window, single image pane, driven by app.session.Session. Scope is
deliberately minimal -- proves the render pipeline (threaded, progressive,
cursor-anchored zoom, correct orientation), not the full sandbox UI. No term
editor, mode selector, or facts panel yet.

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
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QToolBar, QWidget

import cdx
from app.session import Session, render_map

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

# "Approaching" Renderer.precision_floor -- warn this many multiples out, so
# the user has advance notice before the image actually degenerates.
PRECISION_WARN_MULTIPLE = 100


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
    partial_ready = Signal(int, object)   # request_id, ndarray
    full_ready = Signal(int, object)       # request_id, ndarray
    failed = Signal(int, str)              # request_id, error message


class RenderTask:
    """Renders one request: quarter-resolution preview first, then full
    resolution, both on whichever thread pool thread runs this. Plain
    Python, not itself a QRunnable -- see the _Runnable adapter below --
    so it stays easy to construct and inspect directly (e.g. in a test).
    """

    def __init__(self, request_id: int, rational_map: cdx.RationalMap, param: complex,
                viewport: cdx.Viewport, settings: cdx.RenderSettings, mode: str,
                cancel: cdx.CancelToken):
        self.request_id = request_id
        self.rational_map = rational_map
        self.param = param
        self.viewport = viewport
        self.settings = settings
        self.mode = mode
        self.cancel = cancel
        self.signals = RenderSignals()

    def run(self) -> None:
        # Checked at every natural checkpoint, not just once at the top:
        # cancellation can arrive at any point while this runs, and a
        # cancelled task must not emit anything at all (see
        # SandboxWindow.closeEvent -- a cancelled-but-still-emitting task is
        # exactly the race that made draining the pool on close necessary
        # before cancellation existed; skipping emit() entirely here is what
        # lets closeEvent skip waiting instead).
        try:
            if self.cancel.is_cancelled:
                return
            preview_res = max(1, self.viewport.resolution // PREVIEW_RESOLUTION_DIVISOR)
            preview_viewport = cdx.Viewport(self.viewport.center, self.viewport.scale, preview_res)
            preview_array = render_map(self.rational_map, self.param, preview_viewport,
                                       self.settings, self.mode, self.cancel)
            if self.cancel.is_cancelled:
                return
            self.signals.partial_ready.emit(self.request_id, preview_array)

            full_array = render_map(self.rational_map, self.param, self.viewport,
                                    self.settings, self.mode, self.cancel)
            if self.cancel.is_cancelled:
                return
            self.signals.full_ready.emit(self.request_id, full_array)
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
    In between, the widget shows an OPTIMISTIC placeholder: the last
    PAINTED pixmap, transformed by _preview_scale/_preview_translation, an
    affine map (uniform scale + translation, no rotation) accumulated since
    that pixmap was painted. Each wheel event composes a "scale by f about
    the cursor pixel" step; each pan move event composes a pure translation
    by the on-screen pixel delta since the last move event. Composition
    (not recomputing from scratch against the original pixmap each time)
    is what makes five wheel notches before the debounce fires show f^5,
    not f. set_image() resets the accumulator to identity -- a freshly
    painted pixmap has nothing left to approximate.

    The zoom step is the EXACT inverse of the viewport's cursor-anchored
    zoom (center' = w - (w-center)/f, scale' = scale/f), not an
    approximation: pixel_to_complex is affine in viewport.center, so for
    any point p_old on the OLD pixmap and the corresponding screen position
    p_screen showing the same complex value under the NEW viewport,
    p_screen = cursor_pixel + f*(p_old - cursor_pixel) -- a pure "scale by
    f about cursor_pixel" -- follows directly by substituting the zoom
    formula into pixel_to_complex and solving. If this and the real
    viewport update ever disagree, the image visibly jumps the instant the
    real render lands; test_sandbox.py checks the two match exactly, not
    just approximately.
    """

    viewport_changed = Signal()

    def __init__(self, session: Session, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

        self._pixmap: QPixmap | None = None
        self._preview_scale = 1.0
        self._preview_translation = QPointF(0.0, 0.0)

        self._rubber_band_origin: QPoint | None = None
        self._rubber_band_rect: QRect | None = None

        self._panning = False
        self._pan_anchor_complex: complex | None = None
        self._last_pan_pixel: QPoint | None = None

    # ---- display geometry ----------------------------------------------------
    def _display_rect(self) -> QRect:
        side = min(self.width(), self.height())
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2
        return QRect(x, y, side, side)

    def _pixel_to_complex(self, pixel: QPoint) -> complex:
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

    # ---- instant-feedback transform: composed since the last painted render ----
    def _compose_zoom(self, anchor: QPointF, factor: float) -> None:
        # k_new = f*k_old; t_new = A + f*(t_old - A). See the class
        # docstring for the derivation; test_sandbox.py checks this
        # composed transform matches a direct recomputation, multi-step.
        self._preview_translation = anchor + factor * (self._preview_translation - anchor)
        self._preview_scale *= factor

    def _compose_pan(self, delta: QPointF) -> None:
        self._preview_translation = self._preview_translation + delta

    def _transformed_rect(self, rect: QRect) -> QRectF:
        k = self._preview_scale
        t = self._preview_translation
        return QRectF(k * rect.left() + t.x(), k * rect.top() + t.y(),
                      k * rect.width(), k * rect.height())

    # ---- painting --------------------------------------------------------------
    def set_image(self, array: np.ndarray) -> None:
        self._pixmap = QPixmap.fromImage(array_to_qimage(array))
        self._preview_scale = 1.0
        self._preview_translation = QPointF(0.0, 0.0)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self._pixmap is not None:
            painter.drawPixmap(self._transformed_rect(self._display_rect()), self._pixmap,
                               QRectF(self._pixmap.rect()))
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

        self._compose_zoom(QPointF(cursor_pos), factor)
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
            self._last_pan_pixel = pos
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

            # Instant feedback: the raw on-screen pixel delta since the last
            # move event is EXACTLY the translation needed (a pure center
            # shift in the complex plane is exactly a pure pixel
            # translation on screen; no scale change during a pan, so no
            # anchor point is needed the way zoom's step needs one).
            self._compose_pan(QPointF(pos) - QPointF(self._last_pan_pixel))
            self._last_pan_pixel = pos
            self.update()

            self.viewport_changed.emit()
        elif self._rubber_band_origin is not None:
            self._rubber_band_rect = QRect(self._rubber_band_origin, pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self._pan_anchor_complex = None
            self._last_pan_pixel = None
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

        self.session = Session()
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
        self.setCentralWidget(self.image_view)

        toolbar = QToolBar("Controls", self)
        toolbar.setMovable(False)
        reset_button = QPushButton("Reset View", self)
        reset_button.clicked.connect(self._reset_view)
        toolbar.addWidget(reset_button)
        self.addToolBar(toolbar)

        self.statusBar().showMessage("")

    # ---- viewport change -> debounced render ------------------------------------
    @Slot()
    def _on_viewport_changed(self) -> None:
        self._update_status_bar()
        self._debounce_timer.start()   # restarts if already running

    def _reset_view(self) -> None:
        iv = self._initial_viewport
        self.session.viewport = cdx.Viewport(iv.center, iv.scale, iv.resolution)
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
                          cdx.CancelToken())
        task.signals.partial_ready.connect(self._on_partial_ready)
        task.signals.full_ready.connect(self._on_full_ready)
        task.signals.failed.connect(self._on_render_failed)
        self._pending_tasks[request_id] = task
        self._thread_pool.start(_Runnable(task))

    @Slot(int, object)
    def _on_partial_ready(self, request_id: int, array: np.ndarray) -> None:
        if request_id != self._request_id:
            return   # superseded by a newer request; discard
        self.image_view.set_image(array)

    @Slot(int, object)
    def _on_full_ready(self, request_id: int, array: np.ndarray) -> None:
        self._pending_tasks.pop(request_id, None)   # this task is done; safe to release
        if request_id != self._request_id:
            return
        self.image_view.set_image(array)

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
