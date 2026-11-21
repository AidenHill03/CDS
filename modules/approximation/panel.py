"""modules.approximation.panel -- the "Approximation" tab of Project Phoenix.

Load an image of colored regions, build the Fisher-Hill-Lazebnik-Thompson map
r = M o F^(n) o S for it (construction.image_to_basins), and look at:

  Target     the input raster as the construction reads it
  Basins     the basins of r, rendered by the ENGINE (cdx.Map.composed +
             cdx.Renderer.render_basin) -- wheel to zoom about the cursor,
             double-click to reset; it re-renders at every zoom, a coarse
             pass first and then the full resolution
  Certified  the cells proved to lie in each basin; black = undecided layer
  Report     the dynamical data (construction.report) and, after "Score",
             the Wada statistics and Hausdorff distances (scoring.score)

Threading follows app/sandbox.py's rules (see RenderSignals there): work runs
on a QThreadPool; its signals live on an object created on the GUI thread,
connect only to bound methods of this widget, and are released on the GUI
thread. Long jobs are cancellable; the pool is drained when the app quits.

Layering: this module imports app/ nothing it does not need and app/ never
imports this module. The panel is attached by the composition root
(phoenix.py) through SandboxWindow.add_panel.
"""

from __future__ import annotations

import itertools
import json
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from . import construction, scoring
from app import theme
from app.controls import PANELS, ControlEntry, MouseBinding

_TAB = "Approximation tab"

# This tab's controls, for the Info tab (SandboxWindow.add_panel reads the
# panel's `controls` attribute). app/ never imports this module.
APPROXIMATION_CONTROLS = (
    ControlEntry("approx.note", PANELS,
                 "Open image… picks a target; Build constructs a map (Cancel stops it); Score "
                 "compares it with the target; Save report / Save images export the results. "
                 "The Target, Basins, Certified and Report sub-tabs show each stage.", _TAB),
    ControlEntry("approx.wheel_zoom", PANELS, "Basins view: zoom in/out at the cursor (fixed steps)",
                 _TAB, MouseBinding("wheel")),
    ControlEntry("approx.double_click_reset", PANELS, "Basins view: reset to the full view", _TAB,
                 MouseBinding("double-click")),
    ControlEntry("approx.hover_readout", PANELS,
                 "Target / Basins / Certified views: show the point under the cursor below the "
                 "view", _TAB, MouseBinding("hover")),
)

UNRESOLVED_RGB = (0.30, 0.30, 0.30)
UNDECIDED_RGB = (0.0, 0.0, 0.0)


def rgb_image(labels, palette, zero_rgb=UNRESOLVED_RGB):
    """QImage of an integer label raster in IMAGE order (row 0 = top); 0 -> zero_rgb."""
    pal = np.vstack([np.asarray(zero_rgb, float), np.asarray(palette, float)])
    rgb = np.floor(np.clip(pal[np.asarray(labels, int)], 0, 1) * 255 + 0.5).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# A canvas that draws one QImage, fitted, with the plane extent it covers.

EMPTY_TARGET = "Open an image to begin"
EMPTY_BASINS = "Build a map to see its basins"
EMPTY_CERTIFIED = "Build a map to see its certified region"


class ImageCanvas(QWidget):
    """Draws a QImage scaled to fit (aspect kept) and maps widget points to the plane.

    `extent` is (xmin, xmax, ymin, ymax) of the image in the plane, with the
    image's row 0 at ymax (image order). Emits zoom_requested(z, factor) on
    the wheel, reset_requested on double-click, hovered(z) on mouse move.
    """

    zoom_requested = Signal(object, float)   # (complex point, zoom factor)
    reset_requested = Signal()
    hovered = Signal(object)                 # complex point

    def __init__(self, parent=None, interactive=False):
        super().__init__(parent)
        self._image = None
        self._extent = None
        self._message = ""
        self._interactive = interactive
        self.setMouseTracking(True)
        self.setMinimumSize(240, 240)

    def set_image(self, image, extent):
        self._image, self._extent, self._message = image, tuple(float(v) for v in extent), ""
        self.update()

    def set_message(self, text):
        self._message = text
        self.update()

    def _target_rect(self):
        if self._image is None:
            return None
        w, h = self.width(), self.height()
        iw, ih = self._image.width(), self._image.height()
        s = min(w / iw, h / ih)
        return QRectF((w - iw * s) / 2, (h - ih * s) / 2, iw * s, ih * s)

    def plane_point(self, pos):
        rect = self._target_rect()
        if rect is None or self._extent is None or not rect.contains(QPointF(pos)):
            return None
        x0, x1, y0, y1 = self._extent
        u = (pos.x() - rect.left()) / rect.width()
        v = (pos.y() - rect.top()) / rect.height()
        return complex(x0 + u * (x1 - x0), y1 - v * (y1 - y0))

    def paintEvent(self, event):   # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.fillRect(self.rect(), theme.color("bg"))
        rect = self._target_rect()
        if rect is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(rect, self._image)
        if self._message:
            # Empty states and progress notes: one centered line, muted.
            p.setPen(theme.color("muted"))
            p.setFont(theme.ui_font())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
        p.end()

    def wheelEvent(self, event):   # noqa: N802
        if not self._interactive:
            return
        z = self.plane_point(event.position().toPoint())
        if z is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.zoom_requested.emit(z, factor)

    def mouseDoubleClickEvent(self, event):   # noqa: N802
        if self._interactive:
            self.reset_requested.emit()

    def mouseMoveEvent(self, event):   # noqa: N802
        z = self.plane_point(event.position().toPoint())
        if z is not None:
            self.hovered.emit(z)


# ---------------------------------------------------------------------------
# Background jobs

class JobSignals(QObject):
    """Created on the GUI thread; emitted from the worker; connected to bound methods only."""
    progress = Signal(int, str)
    partial = Signal(int, object)
    finished = Signal(int, object)
    failed = Signal(int, str)


class _Job(QRunnable):
    def __init__(self, job_id, fn, signals):
        super().__init__()
        self.job_id, self.fn, self.signals = job_id, fn, signals

    def run(self):
        try:
            result = self.fn(lambda text: self.signals.progress.emit(self.job_id, text),
                             lambda payload: self.signals.partial.emit(self.job_id, payload))
            self.signals.finished.emit(self.job_id, result)
        except construction.Cancelled:
            self.signals.failed.emit(self.job_id, "Cancelled.")
        except Exception as exc:   # surfaced in the panel, never swallowed
            self.signals.failed.emit(self.job_id, f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------

class ApproximationPanel(QWidget):
    """The Approximation tab."""

    controls = APPROXIMATION_CONTROLS
    # A full-width workspace next to Explore (see SandboxWindow.add_panel).
    placement = "workspace"

    BASIN_RESOLUTION = 512
    # Coarse-first: every basin render shows a pass at 1/4 the resolution per
    # side (1/16 of the pixels, so about 6% extra work) before the full one.
    # MEASURED (bench/bench_approx.py, spiral.png at 128, deg S = 4916): the
    # full 512^2 render takes 2.3 s on a quiet 2-core machine (10 s when it is
    # busy), the coarse pass 0.15 s.
    BASIN_PREVIEW_DIVISOR = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)      # one build/score + one basin render at a time
        self._ids = itertools.count(1)
        self._jobs = {}                      # job_id -> (kind, JobSignals); released on the GUI thread
        self._cancel_build = False
        self._render_token = None
        self._image_path = None
        self.r = None
        self.info = None
        self.score_result = None
        self._view = None                    # (center, scale) of the basin view
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)
        self._render_timer.timeout.connect(self._start_basin_render)
        self._build_ui()
        self._set_busy(False)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    # ---- layout -------------------------------------------------------------
    def _build_ui(self):
        controls = QWidget(self)
        col = QVBoxLayout(controls)

        src = QGroupBox("Target image", controls)
        f = QFormLayout(src)
        self.open_button = QPushButton("Open image...", src)
        self.open_button.clicked.connect(self._on_open)
        self.path_label = QLabel("(none)", src)
        self.path_label.setWordWrap(True)
        f.addRow(self.open_button)
        f.addRow(self.path_label)
        col.addWidget(src)

        opts = QGroupBox("Construction", controls)
        f = QFormLayout(opts)
        self.basins_combo = QComboBox(opts)
        self.basins_combo.addItems(["auto"] + [str(k) for k in range(2, 9)])
        self.exterior_combo = QComboBox(opts)
        self.exterior_combo.addItems(["auto"] + [str(k) for k in range(1, 9)])
        self.boundary_edit = QLineEdit(opts)
        self.boundary_edit.setPlaceholderText("e.g. 255,255,255  (optional)")
        self.size_spin = QSpinBox(opts)
        self.size_spin.setRange(16, 1024)
        self.size_spin.setValue(128)
        self.order_spin = QSpinBox(opts)
        self.order_spin.setRange(1, 16)
        self.order_spin.setValue(4)
        self.certify_check = QCheckBox("Certify cells and critical points", opts)
        self.certify_check.setChecked(True)
        f.addRow("Basins (d)", self.basins_combo)
        f.addRow("Exterior label", self.exterior_combo)
        f.addRow("Boundary color", self.boundary_edit)
        f.addRow("Image size (px)", self.size_spin)
        f.addRow("Quadrature order", self.order_spin)
        f.addRow(self.certify_check)
        buttons = QHBoxLayout()
        self.build_button = QPushButton("Build", opts)
        self.build_button.clicked.connect(self.build)
        self.cancel_button = QPushButton("Cancel", opts)
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.build_button)
        buttons.addWidget(self.cancel_button)
        f.addRow(buttons)
        col.addWidget(opts)

        sc = QGroupBox("Score against the target", controls)
        f = QFormLayout(sc)
        self.resolutions_edit = QLineEdit("256, 512", sc)
        self.score_button = QPushButton("Score (Wada + Hausdorff)", sc)
        self.score_button.clicked.connect(self.run_score)
        self.score_button.setEnabled(False)
        f.addRow("Resolutions", self.resolutions_edit)
        f.addRow(self.score_button)
        col.addWidget(sc)

        ex = QGroupBox("Export", controls)
        v = QVBoxLayout(ex)
        self.save_report_button = QPushButton("Save report and summary...", ex)
        self.save_report_button.clicked.connect(self._on_save_report)
        self.save_images_button = QPushButton("Save images...", ex)
        self.save_images_button.clicked.connect(self._on_save_images)
        for b in (self.save_report_button, self.save_images_button):
            b.setEnabled(False)
            v.addWidget(b)
        col.addWidget(ex)

        self.status_label = QLabel("Open an image to begin.", controls)
        self.status_label.setWordWrap(True)
        col.addWidget(self.status_label)
        col.addStretch(1)

        self.views = QTabWidget(self)
        self.target_canvas = ImageCanvas(self)
        self.basin_canvas = ImageCanvas(self, interactive=True)
        self.certified_canvas = ImageCanvas(self)
        self.report_text = QPlainTextEdit(self)
        self.report_text.setReadOnly(True)
        self.report_text.setFont(theme.mono_font())
        self.report_text.setPlaceholderText("Score a built map to see its report")
        self.report_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.views.addTab(self.target_canvas, "Target")
        self.views.addTab(self.basin_canvas, "Basins")
        self.views.addTab(self.certified_canvas, "Certified")
        self.views.addTab(self.report_text, "Report")
        self.basin_canvas.zoom_requested.connect(self._on_zoom)
        self.basin_canvas.reset_requested.connect(self._on_reset_view)
        self.basin_canvas.hovered.connect(self._on_hover)
        # Empty states: one centered line each (the Info tab lists the
        # canvas gestures -- see APPROXIMATION_CONTROLS).
        self.target_canvas.set_message(EMPTY_TARGET)
        self.basin_canvas.set_message(EMPTY_BASINS)
        self.certified_canvas.set_message(EMPTY_CERTIFIED)
        self.readout = QLabel("", self)

        right = QWidget(self)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.views, 1)
        rv.addWidget(self.readout)

        split = QSplitter(self)
        split.addWidget(controls)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        outer = QHBoxLayout(self)
        outer.addWidget(split)

    # ---- jobs ---------------------------------------------------------------
    def _submit(self, kind, fn):
        job_id = next(self._ids)
        signals = JobSignals()
        signals.progress.connect(self._on_job_progress)
        signals.partial.connect(self._on_job_partial)
        signals.finished.connect(self._on_job_finished)
        signals.failed.connect(self._on_job_failed)
        self._jobs[job_id] = (kind, signals)
        job = _Job(job_id, fn, signals)
        job.setAutoDelete(True)
        self._pool.start(job)
        return job_id

    def _on_job_progress(self, job_id, text):
        if job_id in self._jobs and self._jobs[job_id][0] != "render":
            self.status_label.setText(text)

    def _on_job_partial(self, job_id, payload):
        if job_id in self._jobs and self._jobs[job_id][0] == "render":
            self._show_basins(payload)

    def _on_job_finished(self, job_id, result):
        kind, _signals = self._jobs.pop(job_id, (None, None))
        if kind == "build":
            self._build_done(result)
        elif kind == "score":
            self._score_done(result)
        elif kind == "render":
            self._render_done(result)

    def _on_job_failed(self, job_id, message):
        kind, _signals = self._jobs.pop(job_id, (None, None))
        if kind == "render":
            if message != "Cancelled.":
                self.basin_canvas.set_message(message.splitlines()[0])
            return
        self._set_busy(False)
        self.status_label.setText(message.splitlines()[0] if message else "Failed.")
        if message and message != "Cancelled.":
            self.report_text.setPlainText(message)
            self.views.setCurrentWidget(self.report_text)

    def busy(self):
        return any(kind in ("build", "score") for kind, _ in self._jobs.values())

    def wait(self, msecs=-1):
        """Block until every job has finished and its result has been delivered (tests, shutdown)."""
        self._pool.waitForDone(msecs)
        QApplication.processEvents()

    def shutdown(self):
        """Cancel everything and drain the pool -- called when the app quits."""
        self.cancel()
        self._render_timer.stop()
        self._pool.waitForDone(60000)

    def cancel(self):
        """Stop the running build or score (and any basin render)."""
        self._cancel_build = True
        for token in (self._render_token, getattr(self, "_score_token", None)):
            if token is not None:
                token.cancel()

    def _set_busy(self, busy):
        have_map = self.info is not None
        self.open_button.setEnabled(not busy)
        self.build_button.setEnabled(not busy and self._image_path is not None)
        self.cancel_button.setEnabled(busy)
        for w in (self.score_button, self.save_report_button, self.save_images_button):
            w.setEnabled(not busy and have_map)

    # ---- image ----------------------------------------------------------------
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open target image", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        self._image_path = str(path)
        self.path_label.setText(Path(path).name)
        image = QImage(self._image_path)
        if image.isNull():
            self.status_label.setText("Could not read that image.")
            self._image_path = None
        else:
            w, h = image.width(), image.height()
            e = max(w, h)
            self.target_canvas.set_image(image, (-w / e, w / e, -h / e, h / e))
            self.views.setCurrentWidget(self.target_canvas)
            self.status_label.setText(f"{w}x{h} image loaded. Build when ready.")
        self._set_busy(False)

    # ---- build ------------------------------------------------------------------
    def options(self):
        """The construction options the controls currently describe."""
        opts = dict(image_size=self.size_spin.value(), quadrature_order=self.order_spin.value(),
                    certify=self.certify_check.isChecked(), grid_size=256)
        if self.basins_combo.currentText() != "auto":
            opts["num_basins"] = int(self.basins_combo.currentText())
        if self.exterior_combo.currentText() != "auto":
            opts["outside_label"] = int(self.exterior_combo.currentText())
        text = self.boundary_edit.text().strip()
        if text:
            vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
            if len(vals) % 3:
                raise ValueError("Boundary color needs R,G,B triples, e.g. 255,255,255.")
            opts["boundary_colors"] = np.array(vals, float).reshape(-1, 3)
        return opts

    def build(self):
        if self._image_path is None or self.busy():
            return
        try:
            opts = self.options()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._cancel_build = False
        self._set_busy(True)
        path = self._image_path

        def work(progress, _partial):
            return construction.image_to_basins(path, progress=progress,
                                                 should_cancel=lambda: self._cancel_build, **opts)
        self._submit("build", work)

    def _build_done(self, result):
        self.r, self.info = result
        self.score_result = None
        self._set_busy(False)
        info = self.info
        b = info["image_bounds"]
        self.target_canvas.set_image(rgb_image(info["target_labels"], info["palette"]), b)
        cert = info["dynamics"]["certificate"]
        if cert is not None:
            self.certified_canvas.set_image(rgb_image(cert["cell_labels"], info["palette"], UNDECIDED_RGB),
                                            cert["window"])
        else:
            self.certified_canvas.set_message("Certification was off for this build.")
        self.report_text.setPlainText(construction.report(info))
        dyn = info["dynamics"]
        self.status_label.setText(f"deg r = {dyn['degree']['r']:.4g}. Fatou set = the "
                                  f"{dyn['degree']['F']} basins: {dyn['fatou_verdict']}.")
        self._view = None
        self._start_basin_render()
        self.views.setCurrentWidget(self.basin_canvas)

    # ---- basin view ---------------------------------------------------------------
    def _current_viewport(self):
        import cdx
        if self._view is None:
            vp = scoring.default_viewport(self.info, self.BASIN_RESOLUTION)
            self._view = (complex(vp.center), float(vp.scale))
        center, s = self._view
        return cdx.Viewport(center, s, self.BASIN_RESOLUTION)

    def _start_basin_render(self):
        if self.r is None:
            return
        import cdx
        if self._render_token is not None:
            self._render_token.cancel()          # supersede the previous render
        token = cdx.CancelToken()
        self._render_token = token
        vp = self._current_viewport()
        r, info = self.r, self.info
        self.basin_canvas.set_message("Rendering...")

        coarse = cdx.Viewport(vp.center, vp.scale, max(16, vp.resolution // self.BASIN_PREVIEW_DIVISOR))

        def work(_progress, partial):
            labels, _ = scoring.render_basins(r, info, coarse, cancel=token)
            if token.is_cancelled:
                raise construction.Cancelled()
            partial((labels, vp.center, vp.scale))
            labels, _ = scoring.render_basins(r, info, vp, cancel=token)
            if token.is_cancelled:
                raise construction.Cancelled()
            return labels, vp.center, vp.scale
        self._submit("render", work)

    def _render_done(self, result):
        self._show_basins(result)

    def _show_basins(self, result):
        """Display a basin render (the coarse pass or the full one) unless the view has moved on."""
        labels, center, s = result
        if self._view is None or abs(center - self._view[0]) > 0 or s != self._view[1]:
            return                                # a stale render; a newer one is on its way
        img_order = construction.to_cdx_orientation(labels).astype(int)   # cdx -> image order, once
        self._basin_labels = img_order
        extent = (center.real - s, center.real + s, center.imag - s, center.imag + s)
        self.basin_canvas.set_image(rgb_image(img_order, self.info["palette"]), extent)

    def _on_zoom(self, z, factor):
        if self._view is None:
            return
        center, s = self._view
        new_s = s / factor
        self._view = (z + (center - z) / factor, new_s)
        self._render_timer.start()

    def _on_reset_view(self):
        self._view = None
        self._render_timer.start()

    def _on_hover(self, z):
        if self.info is None or not hasattr(self, "_basin_labels") or self._view is None:
            return
        center, s = self._view
        n = self._basin_labels.shape[0]
        col = int((z.real - (center.real - s)) / (2 * s) * n)
        row = int(((center.imag + s) - z.imag) / (2 * s) * n)
        if 0 <= row < n and 0 <= col < n:
            lab = int(self._basin_labels[row, col])
            what = "unresolved" if lab == 0 else f"basin {lab}" + (
                " (exterior)" if lab == self.info["outside_label"] else "")
            self.readout.setText(f"z = {z.real:+.6f} {z.imag:+.6f}i    {what}")

    # ---- scoring ------------------------------------------------------------------
    def run_score(self):
        if self.r is None or self.busy():
            return
        try:
            res = [int(v) for v in self.resolutions_edit.text().replace(";", ",").split(",") if v.strip()]
            if not res or min(res) < 16:
                raise ValueError
        except ValueError:
            self.status_label.setText("Resolutions: a comma-separated list of integers >= 16.")
            return
        import cdx
        token = cdx.CancelToken()
        self._score_token = token
        self._set_busy(True)
        r, info = self.r, self.info

        def work(progress, _partial):
            out = scoring.score(r, info, resolutions=res, cancel=token, progress=progress)
            if out is None:
                raise construction.Cancelled()
            return out
        self._submit("score", work)

    def _score_done(self, sc):
        self.score_result = sc
        self._set_busy(False)
        self.report_text.setPlainText(construction.report(self.info) + "\n\n" + scoring.score_report(sc))
        self.status_label.setText("Scored. See the Report tab.")
        self.views.setCurrentWidget(self.report_text)

    # ---- export -------------------------------------------------------------------
    def _on_save_report(self):
        if self.info is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save summary", "basins_summary.json", "JSON (*.json)")
        if path:
            self.save_report(path)

    def save_report(self, path):
        """JSON summary (construction.summary + scoring numbers) and a .txt report beside it."""
        data = construction.summary(self.info)
        text = construction.report(self.info)
        if self.score_result is not None:
            sc = {k: v for k, v in self.score_result.items() if k not in ("target_labels", "basin_labels")}
            sc["viewport"] = dict(center=[sc["viewport"][0].real, sc["viewport"][0].imag], scale=sc["viewport"][1])
            data["scoring"] = json.loads(json.dumps(sc, default=lambda v: str(v)))
            text += "\n\n" + scoring.score_report(self.score_result)
        Path(path).write_text(json.dumps(data, indent=2, default=lambda v: str(v)))
        Path(str(path) + ".txt").write_text(text + "\n")
        self.status_label.setText(f"Saved {Path(path).name} and {Path(path).name}.txt")

    def _on_save_images(self):
        if self.info is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the images")
        if folder:
            self.save_images(folder)

    def save_images(self, folder):
        stem = Path(self._image_path).stem if self._image_path else "basins"
        info = self.info
        out = []
        rgb_image(info["target_labels"], info["palette"]).save(str(Path(folder) / f"{stem}_target.png"))
        out.append(f"{stem}_target.png")
        if getattr(self, "_basin_labels", None) is not None:
            rgb_image(self._basin_labels, info["palette"]).save(str(Path(folder) / f"{stem}_basins.png"))
            out.append(f"{stem}_basins.png")
        cert = info["dynamics"]["certificate"]
        if cert is not None:
            rgb_image(cert["cell_labels"], info["palette"], UNDECIDED_RGB).save(
                str(Path(folder) / f"{stem}_certified.png"))
            out.append(f"{stem}_certified.png")
        self.status_label.setText("Saved " + ", ".join(out))
        return out


def make_panel(window):
    """Factory for SandboxWindow.add_panel: (widget, tab title)."""
    return ApproximationPanel(window), "Approximation"
