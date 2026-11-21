"""app/sandbox.py -- P5a/P5b: the interactive window.

Single window, driven by app.session.Session (UI redesign layout): a top
bar (map chip, workspace switch, Coupled | Single, render status, Reset
view, Export); the Explore workspace -- the parameter and dynamical planes
side by side (render pipeline: threaded, progressive, cursor-anchored zoom,
overscan buffer, correct orientation); a right sidebar dock (Info,
Library -- app/library_panel.py, Facts -- app/facts_panel.py); and a bottom
"Equations & Settings" dock (Equation -- app/equation_panel.py, Rendering
and Coloring -- app/settings_panel.py, applied live, and Orbit). Research
modules add workspaces or tabs through SandboxWindow.add_panel.

Requires the cdx extension module and PySide6 to be importable, e.g. from
the repository root:

    PYTHONPATH=cdx/build python -m app.sandbox

Entry point: `python -m app.sandbox`.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import os
import sys
import time
from typing import Callable

import numpy as np
import shiboken6
from PySide6.QtCore import (QBuffer, QEvent, QIODevice, QObject, QPoint, QPointF, QRect,
                            QRectF, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, Signal,
                            Slot)
from PySide6.QtGui import (QAction, QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QFrame,
                               QScrollArea,
                               QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QSplitter, QStackedWidget, QTabWidget, QTextEdit, QToolBar,
                               QVBoxLayout, QWidget)

import cdx
from app import controls, theme
from app.about_dialog import AboutDialog
from app.color import (color_basin, color_escape_time, color_parameter_basin,
                        color_parameter_period, color_scalar_field)
from app.complex_field import ComplexField
from app.equation_panel import EquationPanel
from app.facts_panel import (FactsPanel, _classify, _is_inf, critical_orbit_fates, facts_to_dict,
                             format_short)
from app.info_panel import InfoPanel
from app.library_panel import LibraryPanel, default_dynamical_view, default_view_for_mode
from app.metadata_header import MetadataHeader, describe_parameter_role
from app.orbit_panel import OrbitPanel
from app.orbit_tracker import OrbitTracker
from app.pane import Pane
from app.render_cache import RenderCache
from app.session import (DEFAULT_PARAMETER_VIEW_CENTER, DEFAULT_PARAMETER_VIEW_SCALE,
                         PARAMETER_PLANE_MODES, PRESET_FAMILY_NAMES, RENDER_MODES, Session,
                         render_map)
from app.settings import (Settings, library_path, load_settings, preview_path_for,
                          previews_dir, save_settings)
from app.settings_panel import (COLORING_GROUPS, RENDERING_GROUPS, SETTINGS_APPLY_DEBOUNCE_MS,
                                SettingsPanel)
from app.version import PRODUCT_NAME
from app.plane_frame import PlaneFrame
from app.widgets import Segment, TabbedDock, TopBar

# Per-notch scroll zoom factor (f > 1 zooms in). ~1.15 per the spec: a
# handful of notches gives a noticeable zoom without a single notch jumping
# too far to feel controllable.
ZOOM_FACTOR_PER_NOTCH = 1.15

# Wheel events whose angleDelta is 0 but pixelDelta is not (some high-
# resolution trackpad drivers) are converted to notches at this many
# logical pixels per notch. 120 matches the angleDelta convention (one
# classic wheel notch = 120 units) on the common X11/Wayland drivers that
# report a 15-degree notch as roughly that many pixels; it only needs to
# be in the right ballpark, since these events arrive in long smooth
# streams and the sum is what the user sees.
PIXEL_DELTA_PER_NOTCH = 120.0

# Keyboard navigation in a pane (controls redesign, Step 2).
# +/- (and View > Zoom In/Out) zoom about the pane's center by this factor.
KEY_ZOOM_FACTOR = ZOOM_FACTOR_PER_NOTCH ** 2
# An arrow press pans by this fraction of the view's half-width...
KEY_PAN_FRACTION = 0.1
# ...and, held, repeats after PAN_KEY_REPEAT_DELAY_MS every PAN_KEY_REPEAT_MS
# (deliberately independent of Settings.param_marker_rate, which paces
# parameter-marker nudging). The initial delay lets a normal tap pan once.
PAN_KEY_REPEAT_DELAY_MS = 300
PAN_KEY_REPEAT_MS = 60

# How long the status bar's per-pane controls hint stays after the cursor
# enters a pane (it also clears when the cursor leaves).
PANE_HINT_MS = 6000
STATUS_FLASH_MS = 4000
# Undo history of the parameter a (plane clicks, nudges, typed values).
PARAM_HISTORY_LIMIT = 200
# A held/repeated arrow nudge is ONE undo step, not one per tick.
NUDGE_HISTORY_COALESCE_S = 0.7

# Preset thumbnails (render_thumbnail), shared by every window of the process.
_PRESET_THUMBNAIL_CACHE: dict[tuple, "QIcon"] = {}

# ---- render modes as the UI shows them ------------------------------------------------------
# RENDER_MODES ids stay data; the segmented controls show these labels, and
# the tooltips give the full names.
MODE_LABELS: dict[str, tuple[str, str]] = {
    "parameter": ("Escape", "Escape time of the critical orbit, for each parameter a"),
    "parameter_greens": ("Green's", "Green's function (escape-rate potential) of the critical "
                                    "orbit, for each parameter a"),
    "parameter_basin": ("Cycles", "Number of attracting cycles, for each parameter a"),
    "parameter_period": ("Period", "Period of the attracting cycle, for each parameter a"),
    "julia": ("Escape", "Escape time of each starting point z"),
    "basin": ("Basins", "Basins of attraction, colored by attractor (infinity included)"),
    "greens": ("Green's", "Green's function (escape-rate potential) of each starting point z"),
}
PARAMETER_MODE_ORDER = ["parameter", "parameter_greens", "parameter_basin", "parameter_period"]
DYNAMICAL_MODE_ORDER = ["julia", "basin", "greens"]
assert set(MODE_LABELS) == set(RENDER_MODES) == set(PARAMETER_MODE_ORDER + DYNAMICAL_MODE_ORDER)
assert set(PARAMETER_MODE_ORDER) == set(PARAMETER_PLANE_MODES)


def _mode_segment(mode: str) -> Segment:
    label, tooltip = MODE_LABELS[mode]
    return Segment(mode, label, tooltip)


class ModeSelector:
    """Compatibility shim with the old per-pane mode QComboBox's API
    (setCurrentText/currentText/count/itemText) over the segmented control:
    setCurrentText(mode) switches the pane exactly as clicking that segment
    does. Kept so existing call sites and tests still read naturally."""

    def __init__(self, window: "SandboxWindow", pane: Pane):
        self._window = window
        self._pane = pane

    def setCurrentText(self, mode: str) -> None:
        self._window._on_mode_changed(self._pane, mode)

    def currentText(self) -> str:
        return self._pane.render_mode

    def count(self) -> int:
        return len(RENDER_MODES)

    def itemText(self, index: int) -> str:
        return RENDER_MODES[index]


# ---- window layout (UI redesign) -------------------------------------------------------
MIN_WINDOW_SIZE = QSize(1100, 720)
DEFAULT_WINDOW_SIZE = QSize(1389, 868)    # a 13" MacBook's usable screen, in logical px
SIDEBAR_MIN_WIDTH = 260
SIDEBAR_DEFAULT_WIDTH = 300
SETTINGS_DOCK_DEFAULT_HEIGHT = 220
# Below this window width the sidebar collapses to its tab strip (and
# re-expands above it, unless the user collapsed it by hand).
SIDEBAR_AUTO_COLLAPSE_WIDTH = 1250
# Bumped whenever the dock/toolbar arrangement changes shape, so an old
# saved state is ignored instead of restoring a layout that no longer exists.
LAYOUT_STATE_VERSION = 1


def default_layout_settings() -> QSettings:
    """Where window geometry and dock state persist: an INI file next to
    settings.json (see app.settings.config_dir)."""
    from app.settings import config_dir
    return QSettings(str(config_dir() / "layout.ini"), QSettings.Format.IniFormat)

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
# refresh_layers), not per paint.
CRITICAL_ORBIT_TRACE_STEPS = 60


def trace_critical_orbit(rational_map: cdx.RationalMap, param: complex,
                         z0: complex) -> list[complex]:
    """The forward orbit of `z0` under (rational_map, param), up to
    CRITICAL_ORBIT_TRACE_STEPS or an infinite/NaN escape, whichever comes
    first -- a module-level pure function (ImageView._trace_orbit is now a
    thin wrapper) so File > Export's preview dialog can compute a traced-
    orbit overlay for its OWN chosen critical points without needing a
    live ImageView instance's cached state.
    """
    points = [z0]
    z = z0
    for _ in range(CRITICAL_ORBIT_TRACE_STEPS):
        if math.isinf(z.real) or math.isinf(z.imag) or math.isnan(z.real) or math.isnan(z.imag):
            break   # infinity (or a NaN escape) has no further finite orbit to draw
        z = rational_map.eval(z, param)
        points.append(z)
    return points


# ---- overlay-layer registry (Stage 1) ----------------------------------------
# A DECLARED layer, not a one-off flag: every dynamical/parameter-plane overlay
# that plots points from the map's own algebraic/dynamical data (as opposed to
# the seeded orbit or the parameter marker, which are their own separate
# concepts -- see paint_overlays' own docstring) is meant to become one entry
# here instead of its own hand-written flag/method quartet. Adding a layer
# should mean adding one OverlayLayer to OVERLAY_LAYERS below, not touching
# ImageView's painting code or SandboxWindow's View-menu construction (see
# ImageView.refresh_layers/_build_view_menu_layer_actions).
#
# `facts` is threaded through every provider's signature uniformly (a single
# cdx.DynamicalFacts computed once per (map, param), shared across every
# layer -- see ImageView.refresh_layers) even though the CRITICAL-points
# layer below does not actually use it: DynamicalFacts.critical_points is
# documented WITH MULTIPLICITY (see cdx/include/cdx/analysis.hpp), the same
# set RationalMap.critical_points() returns, not distinct_critical_points().
# This overlay has always deliberately drawn DISTINCT points only ("one
# marker per location, multiplicity has no separate visual meaning here" --
# see the pre-Stage-1 refresh_critical_points' own docstring) -- sourcing it
# from `facts` as-is would silently draw duplicate/near-duplicate markers at
# any multiplicity>1 critical point, a real regression. Its provider and
# trace therefore keep calling distinct_critical_points() directly (exactly
# as before this stage) and simply ignore `facts`; `facts` remains part of
# the signature purely for uniformity across layers, and IS what Stage 2's
# new layers (fixed points, attracting cycles) actually read.
OverlayColor = tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class OverlayTrace:
    """A layer's OPTIONAL sub-overlay: a path per (a subset of) the layer's
    own points, gated behind the layer itself being enabled (see
    ImageView.set_trace_enabled) -- mirrors how the critical-points overlay's
    own "Trace Orbits" toggle has always been a sub-feature of "Critical
    Points", never independently drawable (paint_overlays' own docstring
    calls this out explicitly; preserved, not a new coupling).

    path_provider returns (polyline, color) pairs -- paired color, unlike
    the spec prose's own "list of polylines" phrasing, because Stage 2's
    attracting-cycle trace needs each cycle's path drawn in that SAME
    cycle's own (strength-dependent) color; pairing it inline here is the
    simplest way to carry that without a second, order-dependent list the
    compositor would have to zip against this one.
    """
    key: str
    label: str
    path_provider: Callable[[cdx.RationalMap, complex, object], list[tuple[list[complex], OverlayColor]]]
    # Shown in the legend (Stage 3) instead of `label` when non-empty --
    # e.g. "Traced orbits" reads better there than the menu action's own
    # "Trace Orbits". Empty means: use `label` as-is (see build_legend_entries).
    legend_label: str = ""
    # The overlay chip's (shorter, sentence-case) text; empty means `label`.
    chip_label: str = ""


@dataclasses.dataclass(frozen=True)
class OverlayLayer:
    """One declared overlay: a marker style/radius, a gate (which render
    modes it's even meaningful on), and a points_provider returning
    (point, color) pairs -- color may already vary per point (e.g. Stage
    2's fixed-point classification coloring) since it's computed INSIDE
    the provider, not applied uniformly afterward by some separate field.
    """
    key: str
    label: str
    gate: str   # "dynamical" | "parameter" | "always"
    marker_radius: float
    points_provider: Callable[[cdx.RationalMap, complex, object], list[tuple[complex, OverlayColor]]]
    marker_style: str = "circle"   # "circle" | "square" | "diamond" -- see _draw_marker
    # Legend (Stage 3) text/swatches -- legend_label empty means "use
    # `label` as-is" (matches critical points, whose legend entry needs no
    # extra text); legend_swatches is the small set of representative
    # colors build_legend_entries/draw_legend show beside that text (e.g.
    # fixed points' three classification colors, or a cycle's two gradient
    # ends) -- kept as their OWN fields rather than derived from
    # points_provider's live output, since the legend describes what a
    # color MEANS in general, not whatever happens to be on screen right now.
    legend_label: str = ""
    legend_swatches: tuple[OverlayColor, ...] = ()
    trace: OverlayTrace | None = None
    chip_label: str = ""      # the overlay chip's text; empty means `label`
    tooltip: str = ""

    def is_visible_for(self, render_mode: str) -> bool:
        if self.gate == "always":
            return True
        is_dynamical = render_mode not in PARAMETER_PLANE_MODES
        return is_dynamical if self.gate == "dynamical" else not is_dynamical


CRITICAL_POINTS_LAYER_KEY = "critical_points"
CRITICAL_TRACE_KEY = "trace_orbits"
CRITICAL_POINT_MARKER_RADIUS = 5.0
CRITICAL_POINT_FILL_COLOR: OverlayColor = theme.rgba("mark_point")
CRITICAL_TRACE_COLOR: OverlayColor = theme.rgba("mark_trace")


def _critical_points_provider(rational_map: cdx.RationalMap, param: complex,
                              facts: object) -> list[tuple[complex, OverlayColor]]:
    del facts   # see OVERLAY_LAYERS' own module comment on why this layer ignores it
    return [(pt, CRITICAL_POINT_FILL_COLOR)
           for pt in rational_map.distinct_critical_points(param)]


def _critical_trace_provider(rational_map: cdx.RationalMap, param: complex,
                             facts: object) -> list[tuple[list[complex], OverlayColor]]:
    del facts
    points = rational_map.distinct_critical_points(param)
    return [(trace_critical_orbit(rational_map, param, z0), CRITICAL_TRACE_COLOR) for z0 in points]


# ---- fixed points (Stage 2) ---------------------------------------------------
# Color BY CLASSIFICATION -- app.facts_panel._classify(fp.multiplier), the
# SAME classification the Facts tab's own Fixed Points table already shows,
# reused rather than reimplemented here (so this overlay and that table can
# never disagree about what a given multiplier counts as). _classify returns
# FOUR labels ("superattracting"/"attracting"/"neutral"/"repelling"), not the
# three named colors a fixed 3-color palette implies -- "superattracting"
# (|multiplier| exactly 0, the common case for e.g. every one of Newton's
# method's own root fixed points) is the strongest member of the SAME
# qualitative family as "attracting" (points nearby converge either way), so
# it shares that family's color here rather than getting a 4th color.
FIXED_POINTS_LAYER_KEY = "fixed_points"
FIXED_POINT_MARKER_RADIUS = 5.0
# green / red / yellow -- clear, distinguishable, and conventional for
# attracting/repelling/neutral without relying on a legend to be readable.
_FIXED_POINT_CLASSIFICATION_COLORS: dict[str, OverlayColor] = {
    "attracting": theme.rgba("fixed_attracting"),
    "superattracting": theme.rgba("fixed_attracting"),
    "repelling": theme.rgba("fixed_repelling"),
    "neutral": theme.rgba("fixed_neutral"),
}


def _fixed_points_provider(rational_map: cdx.RationalMap, param: complex,
                           facts: object) -> list[tuple[complex, OverlayColor]]:
    if facts is None:
        return []
    return [(fp.point, _FIXED_POINT_CLASSIFICATION_COLORS[_classify(fp.multiplier)])
           for fp in facts.fixed_points if not _is_inf(fp.point)]


# ---- attracting cycles (Stage 2) -----------------------------------------------
# Color BY ATTRACTING STRENGTH -- |cyc.multiplier| ranges over [0, 1) for any
# genuinely attracting cycle (0 = superattracting, approaching 1 = barely
# attracting), a small two-color gradient over that range so every point AND
# the traced path of a given cycle -- both computed from the SAME
# _cycle_color(cyc.multiplier) call -- share one color, making which points
# belong to which cycle (and how strongly it attracts) visually legible
# without a separate legend entry per cycle.
ATTRACTING_CYCLES_LAYER_KEY = "attracting_cycles"
CYCLE_TRACE_KEY = "trace_cycle_paths"
CYCLE_MARKER_RADIUS = 5.0
_CYCLE_STRONG_COLOR: OverlayColor = theme.rgba("cycle_strong")   # superattracting (|multiplier| = 0)
_CYCLE_WEAK_COLOR: OverlayColor = theme.rgba("cycle_weak")       # |multiplier| approaching 1


def _cycle_color(multiplier: complex) -> OverlayColor:
    t = min(1.0, abs(multiplier))   # clamp: an attracting cycle's |multiplier| is always < 1, but guard anyway
    return tuple(int(round(a + (b - a) * t))
                for a, b in zip(_CYCLE_STRONG_COLOR, _CYCLE_WEAK_COLOR))


def _attracting_cycles_provider(rational_map: cdx.RationalMap, param: complex,
                                facts: object) -> list[tuple[complex, OverlayColor]]:
    if facts is None:
        return []
    color_points = []
    for cyc in facts.attracting_cycles:
        color = _cycle_color(cyc.multiplier)
        color_points.extend((pt, color) for pt in cyc.points if not _is_inf(pt))
    return color_points


def _attracting_cycles_trace_provider(rational_map: cdx.RationalMap, param: complex,
                                      facts: object) -> list[tuple[list[complex], OverlayColor]]:
    if facts is None:
        return []
    loops = []
    for cyc in facts.attracting_cycles:
        if not cyc.points:
            continue
        # Closes the cycle's own ORDERED points into a loop (append
        # points[0]) -- deliberately NOT pre-filtered for infinity the way
        # the points_provider above is: drawable_polyline_segments already
        # breaks (not bridges) any segment touching a non-finite point, the
        # same handling a critical-point trace ending at infinity already
        # relies on, so a cycle that happens to include infinity still
        # draws its finite legs correctly instead of needing a second,
        # divergent filtering rule here.
        loops.append((list(cyc.points) + [cyc.points[0]], _cycle_color(cyc.multiplier)))
    return loops


OVERLAY_LAYERS: list[OverlayLayer] = [
    OverlayLayer(
        key=CRITICAL_POINTS_LAYER_KEY, label="Critical Points", gate="dynamical",
        marker_radius=CRITICAL_POINT_MARKER_RADIUS, points_provider=_critical_points_provider,
        marker_style="circle", legend_swatches=(CRITICAL_POINT_FILL_COLOR,),
        trace=OverlayTrace(key=CRITICAL_TRACE_KEY, label="Trace Orbits",
                           path_provider=_critical_trace_provider,
                           legend_label="Traced orbits", chip_label="Trace orbits"),
        chip_label="Critical points",
        tooltip="Mark the critical points of f (their orbits decide the dynamics)",
    ),
    OverlayLayer(
        key=FIXED_POINTS_LAYER_KEY, label="Fixed Points", gate="dynamical",
        marker_radius=FIXED_POINT_MARKER_RADIUS, points_provider=_fixed_points_provider,
        marker_style="square",
        legend_label="Fixed points: attracting / repelling / neutral",
        legend_swatches=(_FIXED_POINT_CLASSIFICATION_COLORS["attracting"],
                         _FIXED_POINT_CLASSIFICATION_COLORS["repelling"],
                         _FIXED_POINT_CLASSIFICATION_COLORS["neutral"]),
        chip_label="Fixed points",
        tooltip="Mark the fixed points: green attracting, red repelling, yellow neutral",
    ),
    OverlayLayer(
        key=ATTRACTING_CYCLES_LAYER_KEY, label="Attracting Cycles", gate="dynamical",
        marker_radius=CYCLE_MARKER_RADIUS, points_provider=_attracting_cycles_provider,
        marker_style="diamond",
        legend_label="Attracting cycles: colored by strength",
        legend_swatches=(_CYCLE_STRONG_COLOR, _CYCLE_WEAK_COLOR),
        trace=OverlayTrace(key=CYCLE_TRACE_KEY, label="Trace Cycle Paths",
                           path_provider=_attracting_cycles_trace_provider,
                           legend_label="Cycle paths", chip_label="Trace cycles"),
        chip_label="Cycles",
        tooltip="Mark the attracting cycles, colored by how strongly they attract",
    ),
]

_OVERLAY_LAYERS_BY_KEY: dict[str, OverlayLayer] = {layer.key: layer for layer in OVERLAY_LAYERS}


# ---- overlay toggles: the on/off switches the chips and View menu share ----------
# The registry layers above, plus three overlays that are drawn by
# paint_overlays itself rather than from a points provider: the seeded orbit,
# the parameter marker, and its crosshair. overlay_toggles() is the ONE list
# both the overlay chips (under each plane) and the View menu are built from.
ORBIT_TOGGLE_KEY = "seeded_orbit"
PARAM_MARKER_TOGGLE_KEY = "param_marker"
CROSSHAIR_TOGGLE_KEY = "crosshair"


@dataclasses.dataclass(frozen=True)
class OverlayToggle:
    key: str
    menu_label: str
    chip_label: str
    gate: str                        # "dynamical" | "parameter"
    swatch: OverlayColor | None      # the chip's legend swatch (None: no dot)
    default: bool = False
    parent: str | None = None        # a trace: only meaningful while its layer is on
    tooltip: str = ""


@dataclasses.dataclass(frozen=True)
class BuiltinOverlay:
    toggle: OverlayToggle
    after: str | None                # placed right after this toggle key (None: first)


BUILTIN_OVERLAYS: list[BuiltinOverlay] = [
    BuiltinOverlay(OverlayToggle(ORBIT_TOGGLE_KEY, "Seeded Orbit", "Orbit", "dynamical",
                                 theme.rgba("mark_orbit"), default=True,
                                 tooltip="Show the orbit seeded by clicking the dynamical plane"),
                   after=CRITICAL_TRACE_KEY),
    BuiltinOverlay(OverlayToggle(PARAM_MARKER_TOGGLE_KEY, "Parameter Marker", "Marker", "parameter",
                                 theme.rgba("mark_param"), default=True,
                                 tooltip="Mark the current parameter a"),
                   after=CYCLE_TRACE_KEY),
    BuiltinOverlay(OverlayToggle(CROSSHAIR_TOGGLE_KEY, "Crosshair", "Crosshair", "parameter",
                                 None, default=False,
                                 tooltip="Thin lines through the parameter marker"),
                   after=PARAM_MARKER_TOGGLE_KEY),
]


def overlay_toggles() -> list[OverlayToggle]:
    """Every overlay switch, in display order: the registry layers (each
    followed by its trace) with the built-in overlays slotted in."""
    toggles: list[OverlayToggle] = []
    for layer in OVERLAY_LAYERS:
        toggles.append(OverlayToggle(
            layer.key, layer.label, layer.chip_label or layer.label, layer.gate,
            layer.legend_swatches[0] if layer.legend_swatches else None, tooltip=layer.tooltip))
        if layer.trace is not None:
            toggles.append(OverlayToggle(
                layer.trace.key, layer.trace.label, layer.trace.chip_label or layer.trace.label,
                layer.gate, None, parent=layer.key,
                tooltip=f"{layer.trace.legend_label or layer.trace.label} "
                        f"(while {layer.chip_label or layer.label} is on)"))
    for builtin in BUILTIN_OVERLAYS:
        if builtin.after is None:
            index = 0
        else:
            keys = [t.key for t in toggles]
            index = keys.index(builtin.after) + 1 if builtin.after in keys else len(toggles)
        toggles.insert(index, builtin.toggle)
    return toggles


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


def format_readout_complex(z: complex, digits: int = 4) -> str:
    """'−0.1226 + 0.7449i' with real minus signs, for the readout chips."""
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        return "∞"
    re_part = f"{z.real:.{digits}f}".replace("-", "−")
    sign = "−" if z.imag < 0 else "+"
    return f"{re_part} {sign} {abs(z.imag):.{digits}f}i"


def complex_to_pixel(w: complex, viewport: cdx.Viewport, width: int, height: int) -> QPointF:
    """The exact inverse of ImageView._pixel_to_complex, generalized to an
    ARBITRARY (viewport, width, height) instead of reading a live widget's
    pane.viewport/self.width()/self.height() -- what makes overlay
    compositing (see paint_overlays below) usable for an export image at
    its own chosen resolution, not just the on-screen widget. Same square-
    centered ("letterboxed" inside a non-square width/height) convention
    ImageView._display_rect uses; for a SQUARE width==height (every export
    -- see SandboxWindow._do_export_image) there is no letterboxing at all.

    ImageView._complex_to_display_pixel is now a thin wrapper around this,
    passing its own pane.viewport/self.width()/self.height() -- kept as an
    instance method (not deleted) because it's directly exercised by
    existing tests and by every other overlay/click-handling call site
    that reads a point's screen position off THIS pane specifically.
    """
    side = min(width, height)
    x = (width - side) // 2
    y = (height - side) // 2
    rel_x = (w.real - (viewport.center.real - viewport.scale)) / (2.0 * viewport.scale)
    rel_y = (viewport.center.imag + viewport.scale - w.imag) / (2.0 * viewport.scale)
    return QPointF(x + rel_x * side, y + rel_y * side)


# ---- overlay legend (Stage 3) ---------------------------------------------------
# Representative colors for the two overlays that live OUTSIDE the registry
# (the seeded orbit, the parameter marker) -- matching (not necessarily byte-
# identical to every shade paint_overlays' own orbit/marker branches use, just
# representative of them) so the legend's swatch reads as the same color a
# user sees drawn elsewhere on the SAME pane.
PARAM_MARKER_RADIUS = 6.0
PARAM_CROSSHAIR_ALPHA = 140
ORBIT_LEGEND_COLOR: OverlayColor = theme.rgba("mark_orbit")      # matches the orbit dots/lines
PARAM_MARKER_LEGEND_COLOR: OverlayColor = theme.rgba("mark_param")   # matches the marker


@dataclasses.dataclass(frozen=True)
class LegendEntry:
    label: str
    swatches: tuple[OverlayColor, ...] = ()


def build_legend_entries(render_mode: str, *, layer_enabled: dict, layer_trace_enabled: dict,
                         show_orbit: bool, orbit_seeded: bool,
                         show_param_marker: bool, param_set: bool) -> list[LegendEntry]:
    """Which legend entries should appear for the CURRENT render_mode +
    overlay state -- a PURE function of exactly the same inputs paint_overlays/
    paint_registry_layers already gate their own drawing on (layer_enabled/
    layer_trace_enabled + OverlayLayer.is_visible_for for registry layers;
    show_orbit/orbit_seeded and show_param_marker/param_set for the two
    overlays outside the registry), so the legend can never claim to explain
    something that isn't actually being drawn right now, or omit something
    that is.

    BUILT FROM THE REGISTRY: no per-layer branching here -- each
    OverlayLayer/OverlayTrace already carries its own legend_label/
    legend_swatches (see their own docstrings); adding a layer to
    OVERLAY_LAYERS is enough to give it a legend entry too.
    """
    is_dynamical = render_mode not in PARAMETER_PLANE_MODES
    entries: list[LegendEntry] = []
    for layer in OVERLAY_LAYERS:
        if not layer_enabled.get(layer.key, False) or not layer.is_visible_for(render_mode):
            continue
        entries.append(LegendEntry(layer.legend_label or layer.label, layer.legend_swatches))
        if layer.trace is not None and layer_trace_enabled.get(layer.trace.key, False):
            entries.append(LegendEntry(layer.trace.legend_label or layer.trace.label))
    if show_orbit and is_dynamical and orbit_seeded:
        entries.append(LegendEntry("Seeded orbit", (ORBIT_LEGEND_COLOR,)))
    if show_param_marker and not is_dynamical and param_set:
        entries.append(LegendEntry("Parameter marker", (PARAM_MARKER_LEGEND_COLOR,)))
    return entries


LEGEND_MARGIN = 8.0
LEGEND_LINE_HEIGHT = 16.0
LEGEND_SWATCH_SIZE = 10.0
LEGEND_BOX_WIDTH = 260.0


def draw_legend(painter: QPainter, width: int, height: int, entries: list[LegendEntry]) -> None:
    """A compact, semi-transparent corner box, one line per entry -- a small
    colored swatch per meaning (fixed points draw three, side by side, one
    per classification; most other entries draw one; a trace sub-entry
    draws none, just its own text) followed by that entry's label. Top-
    right corner, clear of the cursor-readout text paintEvent's caller
    shows in the status bar instead of over the image itself.
    """
    if not entries:
        return
    box_height = LEGEND_MARGIN * 2 + LEGEND_LINE_HEIGHT * len(entries)
    box_rect = QRectF(width - LEGEND_BOX_WIDTH - LEGEND_MARGIN, LEGEND_MARGIN,
                      LEGEND_BOX_WIDTH, box_height)
    painter.fillRect(box_rect, theme.color("legend_bg"))
    painter.setPen(theme.color("legend_text"))
    for i, entry in enumerate(entries):
        line_top = box_rect.top() + LEGEND_MARGIN + i * LEGEND_LINE_HEIGHT
        x = box_rect.left() + LEGEND_MARGIN
        for j, swatch in enumerate(entry.swatches):
            swatch_rect = QRectF(x + j * (LEGEND_SWATCH_SIZE + 3), line_top,
                                 LEGEND_SWATCH_SIZE, LEGEND_SWATCH_SIZE)
            painter.fillRect(swatch_rect, QColor(*swatch))
        text_x = x + (len(entry.swatches) * (LEGEND_SWATCH_SIZE + 3) + 4 if entry.swatches else 0)
        text_rect = QRectF(text_x, line_top, box_rect.right() - text_x, LEGEND_LINE_HEIGHT)
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter), entry.label)


def paint_overlays(painter: QPainter, viewport: cdx.Viewport, width: int, height: int,
                   render_mode: str, *,
                   show_critical_points: bool = False,
                   critical_points: "list[complex]" = (),
                   show_traced_orbits: bool = False,
                   orbit_traces: "list[list[complex]] | None" = None,
                   show_orbit: bool = False,
                   orbit_connect_lines: bool = True,
                   orbit_state=None,
                   show_param_marker: bool = False,
                   show_crosshair: bool = False,
                   param: "complex | None" = None,
                   show_legend: bool = False,
                   layer_enabled: "dict | None" = None,
                   layer_trace_enabled: "dict | None" = None) -> None:
    """Draws every overlay CURRENTLY requested (the show_* flags) that is
    also VALID for `render_mode` -- critical points, traced (post-critical-
    point) orbits, and the seeded orbit itself are all dynamical-plane-only
    (PARAMETER_PLANE_MODES); the param crosshair marker is parameter-plane-
    only -- exactly the SAME mode-gating ImageView's own overlay painters
    already enforced (_should_draw_critical_points, _orbit_line_segments,
    _param_marker_pixel), just no longer coupled to a live widget:
    `viewport`/`width`/`height` are the caller's OWN (ImageView.paintEvent
    passes its pane.viewport/self.width()/self.height(), unchanged
    behavior; SandboxWindow._do_export_image/the export preview dialog
    pass the EXPORT image's own viewport/resolution instead).

    show_orbit is a genuinely NEW top-level toggle export introduces: the
    live view has no "hide the seeded orbit entirely" checkbox (an orbit,
    once seeded, is always drawn) -- ImageView.paintEvent always passes
    show_orbit=True to preserve that, while the export dialog offers it as
    a real, uncheckable-until-an-orbit-exists choice. orbit_connect_lines
    is the SAME sub-toggle ImageView._orbit_connect_lines already is --
    governs the connecting LINES only; the dots and seed marker are drawn
    whenever show_orbit is on, regardless (see ImageView.
    set_orbit_connect_lines's own docstring for why).

    show_traced_orbits is gated behind show_critical_points, exactly as
    the original code's own nesting had it (traced orbits are a critical-
    point-overlay sub-feature, not independently drawable) -- not a new
    coupling, preserved intentionally.

    show_legend (Stage 3) draws build_legend_entries' output via
    draw_legend, LAST, over everything else painted above -- layer_enabled/
    layer_trace_enabled are ImageView's own per-layer state dicts (see
    ImageView.paintEvent), needed here ONLY to decide which registry
    layers' entries to show; they don't affect what the layers THEMSELVES
    draw (that stays paint_registry_layers' job, a separate call ImageView.
    paintEvent makes). Both default to None (treated as {}, i.e. no
    registry-layer entries) so File > Export's existing calls -- which
    never pass show_legend at all -- are completely unaffected; the
    moment a caller DOES pass show_legend=True (live view now, Export
    whenever it chooses to), this same code already knows how to draw one.
    """
    is_dynamical = render_mode not in PARAMETER_PLANE_MODES

    def finite(w: complex) -> bool:
        return math.isfinite(w.real) and math.isfinite(w.imag)

    bounds = QRectF(0, 0, width, height)
    inflated_rect = bounds.adjusted(-OFFSCREEN_MARGIN_PX, -OFFSCREEN_MARGIN_PX,
                                    OFFSCREEN_MARGIN_PX, OFFSCREEN_MARGIN_PX)

    if show_critical_points and is_dynamical:
        if show_traced_orbits and orbit_traces:
            trace_pen = QPen(QColor(*CRITICAL_TRACE_COLOR))
            trace_pen.setWidth(1)
            painter.setPen(trace_pen)
            for trace in orbit_traces:
                pixel_points = [complex_to_pixel(w, viewport, width, height) for w in trace]
                for a, b in drawable_polyline_segments(trace, pixel_points, inflated_rect):
                    painter.drawLine(a, b)

        marker_pen = QPen(theme.color("mark_outline"))
        marker_pen.setWidth(2)
        painter.setPen(marker_pen)
        painter.setBrush(QColor(*CRITICAL_POINT_FILL_COLOR))
        r = CRITICAL_POINT_MARKER_RADIUS
        for w in critical_points:
            if finite(w):   # infinity is a valid critical point but has no finite screen position
                painter.drawEllipse(complex_to_pixel(w, viewport, width, height), r, r)

    if show_orbit and is_dynamical and orbit_state is not None:
        pixel_points = [complex_to_pixel(w, viewport, width, height) for w in orbit_state.history]
        # Lines are the only thing orbit_connect_lines governs -- dots and
        # the seed marker below are cheap, never sweep, and stay on
        # regardless (see ImageView.set_orbit_connect_lines's own
        # docstring).
        if orbit_connect_lines:
            line_pen = QPen(QColor(*theme.rgba("mark_orbit", alpha=220)))
            line_pen.setWidth(1)
            painter.setPen(line_pen)
            for a, b in drawable_polyline_segments(orbit_state.history, pixel_points,
                                                   inflated_rect):
                painter.drawLine(a, b)

        dot_pen = QPen(theme.color("mark_outline"))
        dot_pen.setWidth(1)
        painter.setPen(dot_pen)
        painter.setBrush(theme.color("mark_orbit"))
        dot_r = CRITICAL_POINT_MARKER_RADIUS * 0.6   # smaller than a critical-point marker --
        for w, pixel in zip(orbit_state.history, pixel_points):   # visually subordinate
            if finite(w) and inflated_rect.contains(pixel):
                painter.drawEllipse(pixel, dot_r, dot_r)

        # The SEED (history[0]) gets its own larger, distinct marker -- the
        # one point in the trace the user actually chose, not one the
        # orbit passed through.
        seed_pixel = complex_to_pixel(orbit_state.z0, viewport, width, height)
        if finite(orbit_state.z0) and inflated_rect.contains(seed_pixel):
            seed_pen = QPen(theme.color("mark_outline"))
            seed_pen.setWidth(2)
            painter.setPen(seed_pen)
            painter.setBrush(theme.color("mark_orbit"))
            painter.drawEllipse(seed_pixel, CRITICAL_POINT_MARKER_RADIUS,
                                CRITICAL_POINT_MARKER_RADIUS)

    if not is_dynamical and param is not None and finite(param):
        pixel = complex_to_pixel(param, viewport, width, height)
        side = min(width, height)
        image = QRectF((width - side) // 2, (height - side) // 2, side, side)
        if show_crosshair and image.contains(pixel):
            # Thin lines through the marker, across the whole image.
            pen = QPen(QColor(*theme.rgba("mark_param", alpha=PARAM_CROSSHAIR_ALPHA)))
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.drawLine(QPointF(image.left(), pixel.y()), QPointF(image.right(), pixel.y()))
            painter.drawLine(QPointF(pixel.x(), image.top()), QPointF(pixel.x(), image.bottom()))
        if show_param_marker and bounds.contains(pixel):
            # Mockup B's marker: an amber dot with a dark ring.
            pen = QPen(theme.color("bg"))
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.setBrush(theme.color("mark_param"))
            painter.drawEllipse(pixel, PARAM_MARKER_RADIUS, PARAM_MARKER_RADIUS)

    if show_legend:
        entries = build_legend_entries(
            render_mode, layer_enabled=layer_enabled or {}, layer_trace_enabled=layer_trace_enabled or {},
            show_orbit=show_orbit, orbit_seeded=orbit_state is not None,
            show_param_marker=show_param_marker, param_set=param is not None and finite(param))
        draw_legend(painter, width, height, entries)


def _draw_marker(painter: QPainter, pixel: QPointF, radius: float, style: str,
                 color: OverlayColor) -> None:
    """One overlay-layer marker -- a BLACK 2px outline (matching critical
    points' own outline treatment, so every layer's markers read as the
    same visual FAMILY) filled with `color`, in one of a small set of
    shapes distinct enough to tell layers apart without reading a legend:
    circle (critical points), square (fixed points), diamond (attracting
    cycles).
    """
    pen = QPen(theme.color("mark_outline"))
    pen.setWidth(2)
    painter.setPen(pen)
    painter.setBrush(QColor(*color))
    if style == "circle":
        painter.drawEllipse(pixel, radius, radius)
    elif style == "square":
        painter.drawRect(QRectF(pixel.x() - radius, pixel.y() - radius, radius * 2.0, radius * 2.0))
    elif style == "diamond":
        painter.drawPolygon(QPolygonF([
            QPointF(pixel.x(), pixel.y() - radius), QPointF(pixel.x() + radius, pixel.y()),
            QPointF(pixel.x(), pixel.y() + radius), QPointF(pixel.x() - radius, pixel.y())]))
    else:
        raise AssertionError(f"unreachable: marker style={style!r}")


def paint_registry_layers(painter: QPainter, viewport: cdx.Viewport, width: int, height: int,
                          render_mode: str, *,
                          layer_enabled: dict,
                          layer_trace_enabled: dict,
                          layer_points: dict,
                          layer_traces: dict) -> None:
    """Draws every OVERLAY_LAYERS entry EXCEPT critical_points -- that one
    stays on paint_overlays' own dedicated show_critical_points/
    critical_points/show_traced_orbits/orbit_traces kwargs (unchanged),
    because File > Export's ExportImageDialog/compose_export_image depend
    on those specific kwarg names and this stage doesn't touch Export.
    ImageView.paintEvent calls this SEPARATELY, right alongside its
    existing paint_overlays call, for the layers Stage 2 actually adds.

    Pure function of its arguments (no ImageView access) -- the four
    `layer_*` dicts are exactly ImageView's own _layer_enabled/
    _layer_trace_enabled/_layer_points/_layer_traces, read by the caller
    and handed in explicitly, the same "resolved data in, painter
    strokes out" shape paint_overlays itself already has.
    """
    bounds = QRectF(0, 0, width, height)
    inflated_rect = bounds.adjusted(-OFFSCREEN_MARGIN_PX, -OFFSCREEN_MARGIN_PX,
                                    OFFSCREEN_MARGIN_PX, OFFSCREEN_MARGIN_PX)

    def finite(w: complex) -> bool:
        return math.isfinite(w.real) and math.isfinite(w.imag)

    for layer in OVERLAY_LAYERS:
        if layer.key == CRITICAL_POINTS_LAYER_KEY:
            continue
        if not layer_enabled.get(layer.key, False) or not layer.is_visible_for(render_mode):
            continue

        if layer.trace is not None and layer_trace_enabled.get(layer.trace.key, False):
            resolved_trace = layer_traces.get(layer.trace.key)
            if resolved_trace:
                for path, color in resolved_trace:
                    pen = QPen(QColor(*color))
                    pen.setWidth(1)
                    painter.setPen(pen)
                    pixel_points = [complex_to_pixel(w, viewport, width, height) for w in path]
                    for a, b in drawable_polyline_segments(path, pixel_points, inflated_rect):
                        painter.drawLine(a, b)

        for point, color in layer_points.get(layer.key, []):
            if finite(point):
                _draw_marker(painter, complex_to_pixel(point, viewport, width, height),
                            layer.marker_radius, layer.marker_style, color)


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
    COLORED QImage.

    cdx render arrays have row 0 at the BOTTOM (matching Viewport::coord;
    see CLAUDE.md) but QImage has row 0 at the TOP -- flip vertically first,
    or preview and full renders would visibly disagree on orientation. This
    was a real bug in the MATLAB prototype (progressive rendering flipped
    the preview relative to the full render because the axis orientation
    was set once instead of after every draw call); flipping the array
    itself, once, here, is what rules that class of bug out entirely rather
    than relying on every caller remembering an origin='lower'-equivalent.

    Dispatches to app.color by render mode: "parameter" is always escape-
    time (color_escape_time, using `settings`' palette/scaling/period);
    "julia" is escape-time too, ALWAYS -- see below for what changes (and
    what deliberately does not) for a RATIONAL map's render (Stage 2);
    "basin" is categorical + SHADED basin coloring (hue = basin id,
    brightness = convergence speed); "greens"/"parameter_greens" are
    scalar-field equipotential banding (color_scalar_field, using
    `settings`' greens_band_width/greens_period_bands/greens_contour) --
    the SAME display treatment for both, since they're the same KIND of
    data (a potential -- Stage 3's CONFORMAL/Koenigs variant is not always
    non-negative, but color_scalar_field's own scale_scalar_field already
    clips below 0 before taking a log, so this needs no special-casing
    either) even though they live on different planes (see cdx::Renderer::
    render_parameter_greens' own header comment on why the two are
    computed by genuinely different kernels).

    GOVERNING PRINCIPLE: the sphere-aware milestone changes CLASSIFICATION
    and the SCALAR a pixel produces; it must never change a mode's
    COLORING PHILOSOPHY. A rational map's "julia" render must look like
    the SAME KIND of image a polynomial's does -- palette + scaling +
    period, exactly like color_escape_time already gives every other
    escape-time-shaped render -- not a bespoke scheme. Concretely: the
    smooth chordal approach-rate (index 0 of the stacked payload) is the
    direct analog of smooth escape time (fast approach = low value = the
    SAME palette end fast escape already lands at), and an unresolved
    pixel (0, per Renderer::render_julia_rational's own "0 = unresolved"
    convention) flows into color_escape_time's existing "0 = never
    escaped" branch unmodified -- no special-casing needed at all, since
    both conventions already agree. Labels (index 1) are extracted and
    kept available to the caller (see ImageView._sample_at_pixel) for the
    cursor readout to still report which basin a pixel is in; only the
    COLORING ignores them -- which basin was reached is BASIN mode's own
    job, not Julia's.

    "basin" mode's payload is STACKED (see render_map's own docstring):
    shape (2, height, width), index 0 = labels, index 1 = iterations --
    unpacked and flipped as two separate 2D layers BEFORE the generic
    np.flipud path below, since flipud always flips axis 0, which for a
    3D stacked array is the LAYER axis, not the row axis; flipping the
    unpacked 2D layers individually (axis 0 IS the row axis there) is what
    actually produces a correctly oriented image instead of silently
    swapping labels and iterations. "julia" mode's payload is STACKED the
    SAME way, but only for a RATIONAL map (see render_map's own docstring)
    -- distinguished here by `payload.ndim` (3 = stacked/rational, 2 =
    plain/certified-polynomial) rather than a separate flag, since that IS
    exactly what render_map's own return shape already encodes; only the
    VALUES layer (index 0) is actually colored, exactly like a plain 2D
    escape-time array would be. "greens"/"parameter_greens" payloads follow
    the SAME ndim==3-means-stacked/rational convention (Stage 3): index 0
    = potential value (colored), index 1 = `exact` (1.0 where CONFORMAL
    was genuinely computed, 0.0 where it fell back to Pragmatic -- kept
    available for the cursor readout, per this function's own GOVERNING
    PRINCIPLE, never the colorer); a certified polynomial's payload stays
    a plain 2D array, same as "parameter".

    "parameter_basin" (Stage 2 of the P/Q-representation-turned-parameter-
    plane batch) is the ONE deliberate exception to the governing
    principle above: its payload is a COUNT (number of distinct attracting
    cycles), not a potential or an escape value, so it gets its own
    categorical colorer (color_parameter_basin) rather than being forced
    through color_escape_time/color_scalar_field -- see that function's
    own docstring for why a count is not a gradient. ALWAYS stacked
    (counts, unresolved, certain), never a plain 2D array (there is no
    certified-polynomial fast path here that would make it one).

    "parameter_period" is the SAME kind of exception, for the SAME reason:
    its payload is a PERIOD (also a category, not a gradient -- see
    color_parameter_period's own docstring), so it gets its own colorer
    too. ALWAYS stacked (periods, undetermined, is_infinity).

    Coloring is a pure DISPLAY-time transform, deliberately not baked into
    what RenderCache stores (raw float arrays) -- changing the palette must
    never be a cache key or trigger a re-render, only a re-paint.
    """
    if mode == "basin":
        labels = np.flipud(payload[0])
        iterations = np.flipud(payload[1])
        rgb = color_basin(labels, iterations, max_iter=max_iter,
                           period=settings.color_period or None,
                           scaling=settings.color_scaling)
        return _rgb_to_qimage(rgb)
    if mode in ("greens", "parameter_greens"):
        values = payload[0] if np.asarray(payload).ndim == 3 else payload
        flipped = np.flipud(values)
        rgb = color_scalar_field(flipped, palette=settings.color_palette,
                                  band_width=settings.greens_band_width,
                                  period_bands=settings.greens_period_bands,
                                  contour=settings.greens_contour)
        return _rgb_to_qimage(rgb)
    if mode == "parameter_basin":
        # Always stacked (counts, unresolved, certain) -- see render_map's
        # own docstring; categorical coloring (color_parameter_basin), NOT
        # the escape-time palette/scaling pipeline every other mode here
        # goes through -- see that function's own docstring for why a
        # COUNT is deliberately not treated as a gradient. `certain` feeds
        # color_parameter_basin's own POLICY (a certain/injected attractor
        # is shown as confirmed even when unresolved > counts overall --
        # see that function's own docstring).
        counts = np.flipud(payload[0])
        unresolved = np.flipud(payload[1])
        certain = np.flipud(payload[2])
        rgb = color_parameter_basin(counts, unresolved, certain)
        return _rgb_to_qimage(rgb)
    if mode == "parameter_period":
        # Always stacked (periods, undetermined, is_infinity) -- see
        # render_map's own docstring; categorical coloring
        # (color_parameter_period), the period-coloring counterpart of
        # "parameter_basin" just above.
        periods = np.flipud(payload[0])
        undetermined = np.flipud(payload[1])
        is_infinity = np.flipud(payload[2])
        rgb = color_parameter_period(periods, undetermined, is_infinity)
        return _rgb_to_qimage(rgb)
    # A RATIONAL "julia" payload is stacked (values, labels) -- color ONLY
    # the values layer, through the exact SAME color_escape_time call a
    # plain (certified-polynomial) payload already goes through below; see
    # this function's own "GOVERNING PRINCIPLE" paragraph for why labels
    # never reach the colorer.
    if mode == "julia" and np.asarray(payload).ndim == 3:
        payload = payload[0]
    flipped = np.flipud(payload)
    if mode in ("julia", "parameter"):
        rgb = color_escape_time(flipped, max_iter, palette=settings.color_palette,
                                 scaling=settings.color_scaling,
                                 period=settings.color_period or None)
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
                cancel: cdx.CancelToken, cache: RenderCache | None = None,
                potential: str = "pragmatic",
                strategy: str = "slowest", critical_index: int = 0):
        self.request_id = request_id
        self.rational_map = rational_map
        self.param = param
        self.viewport = viewport
        self.settings = settings
        self.mode = mode
        self.cancel = cancel
        self.cache = cache
        # Stage 3: which of the two rational Green's potentials -- see
        # app.session.render_map's own `potential` doc comment. Threaded
        # through separately from `settings` (cdx.RenderSettings) because
        # it changes the COMPUTED values, unlike palette/scaling/period,
        # which stay pure display-time and never reach render_map at all
        # (see array_to_qimage's own docstring).
        self.potential = potential
        # Stage 1: which ParameterStrategy "parameter" mode's rational path
        # combines free critical orbits with -- see app.session.render_map's
        # own `strategy`/`critical_index` doc comment. Same reasoning as
        # `potential` above: threaded through explicitly rather than read
        # off `settings`, since it changes computed values.
        self.strategy = strategy
        self.critical_index = critical_index
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
                                       self.settings, self.mode, self.cancel, self.cache,
                                       self.potential, self.strategy, self.critical_index)
            if self.cancel.is_cancelled:
                return
            self.signals.partial_ready.emit(self.request_id, preview_array, preview_buffer_viewport)

            full_buffer_viewport = _overscanned(self.viewport, FULL_OVERSCAN_FACTOR)
            full_array = render_map(self.rational_map, self.param, full_buffer_viewport,
                                    self.settings, self.mode, self.cancel, self.cache,
                                    self.potential, self.strategy, self.critical_index)
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


# Stage 4: arrow-key parameter-marker nudging. Screen convention (Up =
# +imag, matching every other screen<->complex mapping in this module --
# see ImageView._pixel_to_complex's own comment on why screen-down is
# -imag) -- Left/Right move the real axis, Up/Down the imaginary axis.
# Values are UNNORMALIZED unit-ish directions, summed directly when
# multiple arrows are held (see ImageView._do_nudge): a diagonal (e.g.
# Up+Right) moves sqrt(2) times as far per tick as a single cardinal
# direction, not renormalized to match cardinal speed -- simplest thing
# that supports diagonal movement at all, and "if cheap" (per this
# feature's own spec) didn't ask for more than that.
_ARROW_DIRECTIONS: dict[Qt.Key, complex] = {
    Qt.Key.Key_Left:  complex(-1.0, 0.0),
    Qt.Key.Key_Right: complex(1.0, 0.0),
    Qt.Key.Key_Up:    complex(0.0, 1.0),
    Qt.Key.Key_Down:  complex(0.0, -1.0),
}

# Plot-scoped key sets, read from the controls registry (app/controls.py) so
# the Info tab documents exactly what the handler does.
_CANCEL_KEYS = controls.plot_keys("plot.esc_cancel")
_ZOOM_IN_KEYS = controls.plot_keys("plot.key_zoom_in")
_ZOOM_OUT_KEYS = controls.plot_keys("plot.key_zoom_out")
_ORBIT_STEP_KEYS = controls.plot_keys("plot.space_step")
_RUN_N_MODIFIERS = controls.plot_modifiers("plot.shift_space_run")
_ALT_PAN_MODIFIERS = controls.plot_modifiers("plot.alt_arrows_pan")
assert set(_ARROW_DIRECTIONS) == controls.plot_keys("plot.arrows_pan")


def preferences_shortcuts(platform_bindings: list[QKeySequence] | None = None) -> list[QKeySequence]:
    """Settings: StandardKey.Preferences (Cmd+, on macOS); that binding is
    EMPTY on Windows, so the registry's fallback (Ctrl+,) is used there.
    `platform_bindings` injects the StandardKey result for tests."""
    if platform_bindings is None:
        return controls.key_sequences("app.settings")
    fallback = controls.entry("app.settings").binding.fallback
    return list(platform_bindings) or [QKeySequence(text) for text in fallback]


def text_undo_target(widget: QWidget | None):
    """The editable text widget whose OWN undo/redo should handle Edit >
    Undo while `widget` has focus, or None (-> the Equation panel's stack)."""
    if isinstance(widget, QAbstractSpinBox):
        widget = widget.findChild(QLineEdit)
    if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)) and not widget.isReadOnly():
        return widget
    return None


def pane_hint(render_mode: str, coupled: bool, platform: str | None = None) -> str:
    """The status-bar hint shown when the cursor enters a pane: the
    registry's hinted controls for that plane (app/controls.py), labelled
    for `platform` (default: the running one)."""
    if render_mode in PARAMETER_PLANE_MODES:
        parts = controls.hint_parts((controls.ANY_PANE, controls.PARAMETER), platform)
        if not coupled:
            parts.append("Coupled View off: a click also switches this pane to julia")
    else:
        parts = controls.hint_parts((controls.ANY_PANE, controls.DYNAMICAL), platform)
    return " · ".join(parts)


class ImageView(QWidget):
    """The single image pane: displays the current render, and owns every
    mouse/wheel/pinch interaction (drag pan, Shift-drag box zoom, scroll/pinch zoom) and,
    in a PARAMETER-plane pane, arrow-key marker nudging (Stage 4 -- see
    keyPressEvent/keyReleaseEvent).

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
    # True when the cursor enters this pane, False when it leaves -- the
    # window shows a transient controls hint for the pane (pane_hint).
    hover_changed = Signal(bool)
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
    # Emitted with the NUDGED complex value (session.param + one step) from
    # a PARAMETER-plane arrow-key press/repeat -- deliberately a SEPARATE
    # signal from param_changed above, not a reuse of it: param_changed's
    # existing single-view-mode handler switches the clicked pane straight
    # to "julia" (the right thing for a deliberate click, wrong for a
    # nudge -- the user is presumably WATCHING the parameter plane while
    # nudging, not asking to leave it), and its coupled-mode handler resets
    # the partner dynamical pane's viewport to the map's default framing on
    # every call (right for a deliberate jump to a new parameter, wrong for
    # continuous nudging -- see SandboxWindow._apply_param_change's own
    # `reset_dynamical_viewport` parameter). Like param_changed, this does
    # NOT touch session.param itself -- SandboxWindow._on_param_nudged is
    # the one place that assignment happens, via the same
    # _apply_param_change every other param-setting path already funnels
    # through.
    param_nudged = Signal(complex)
    # Shift+Space in a dynamical pane: "Run N" with the orbit panel's N, which
    # only the window knows (see SandboxWindow._on_run_n_requested).
    run_n_requested = Signal()

    def __init__(self, pane: Pane, session: Session, parent: QWidget | None = None):
        super().__init__(parent)
        # pane: viewport + render_mode, per-pane state (see app.pane.Pane).
        # session: map/param/render_settings/settings/cache -- SHARED across
        # every pane looking at the same experiment.
        self.pane = pane
        self.session = session
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        # StrongFocus (tab- and click-focusable) so this pane can receive
        # arrow-key events at all -- a QWidget's default policy is
        # Qt.NoFocus, which silently swallows key events rather than
        # delivering them to keyPressEvent below. mousePressEvent also
        # calls setFocus() explicitly (belt-and-suspenders alongside Qt's
        # own click-to-focus, which some offscreen/test environments don't
        # reliably grant on a simulated click).
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Held-key repeat state (Stage 4) -- see keyPressEvent/
        # keyReleaseEvent/_on_nudge_timer_tick. A set, not a single value,
        # so multiple simultaneously-held arrows (e.g. Up+Right) combine
        # into a diagonal nudge rather than the second press silently
        # replacing the first.
        self._held_arrows: set[complex] = set()
        self._nudge_timer = QTimer(self)
        self._nudge_timer.timeout.connect(self._on_nudge_timer_tick)
        # Arrow-key PANNING (dynamical plane, or Alt+arrows anywhere) keeps
        # its own held set and timer -- a different cadence from marker
        # nudging (PAN_KEY_REPEAT_*), and never driven by param_marker_rate.
        self._held_pan_arrows: set[complex] = set()
        self._pan_timer = QTimer(self)
        self._pan_timer.timeout.connect(self._on_pan_timer_tick)
        self.orbit_tracker = OrbitTracker()
        # Orbit-only -- deliberately independent of _trace_orbits below,
        # which governs the SEPARATE post-critical-point traces (see
        # set_orbit_connect_lines's own docstring).
        self._orbit_connect_lines: bool = True
        # Legend (Stage 3) -- off by default, matching every other overlay's
        # own starting state.
        self._show_legend: bool = False
        # Built-in overlay switches (see BUILTIN_OVERLAYS); registry layers
        # keep their own dicts below.
        self._show_orbit: bool = True
        self._show_param_marker: bool = True
        self._show_crosshair: bool = False
        self._reference_scale_key = None
        self._reference_scale_value: float | None = None

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

        # ---- overlay-layer registry state (Stage 1) ----------------------------
        # Generic per-layer storage backing EVERY registered OverlayLayer (see
        # OVERLAY_LAYERS), not just critical points -- adding a layer means
        # adding one OverlayLayer entry, never a new quartet of fields here.
        # _show_critical_points/_trace_orbits/_critical_points/_orbit_traces
        # below are now thin COMPATIBILITY properties over this same storage
        # (see their own definitions) -- kept because they're depended on
        # directly, by name, by existing tests and by ExportImageDialog.
        #
        # Memoized on (map.serialize(), param) -- see refresh_layers -- so
        # panning/zooming (which only ever changes pane.viewport, never the
        # map or param) never re-triggers root-finding. _layer_facts_key is
        # None until the first refresh_layers() call (made once by
        # SandboxWindow right after construction, same as its other panels'
        # initial state).
        self._layer_facts_key: tuple[str, complex] | None = None
        self._cached_facts = None   # cdx.DynamicalFacts | None -- see refresh_layers' own comment
        self._layer_enabled: dict[str, bool] = {layer.key: False for layer in OVERLAY_LAYERS}
        self._layer_trace_enabled: dict[str, bool] = {
            layer.trace.key: False for layer in OVERLAY_LAYERS if layer.trace is not None}
        # Resolved (point, color) / (path, color) pairs, keyed by layer/
        # trace key -- what a future generic paint/legend step iterates.
        self._layer_points: dict[str, list[tuple[complex, OverlayColor]]] = {}
        self._layer_traces: dict[str, list[tuple[list[complex], OverlayColor]] | None] = {}
        # Plain-point/plain-path PROJECTIONS of the two dicts above, built
        # ONCE per refresh (inside the same memoization gate), not per
        # access -- what _critical_points/_orbit_traces actually read, so
        # repeated access between refreshes returns the SAME list object
        # (existing tests assert this identity to confirm memoization is
        # real, not just cheap-looking).
        self._layer_point_values: dict[str, list[complex]] = {}
        self._layer_trace_values: dict[str, list[list[complex]] | None] = {}

        self._rubber_band_rect: QRect | None = None
        self._pan_anchor_complex: complex | None = None

        # Mouse-gesture state machine (controls redesign, Step 1):
        #   None        -- no button held
        #   "pending"   -- left press, not yet moved past the drag threshold;
        #                  the viewport is NOT touched; a release here is a click
        #   "pan"       -- left drag past the threshold, or any middle drag
        #   "box"       -- Shift+left press (Shift is read at PRESS time only)
        #   "cancelled" -- Esc pressed mid-gesture; everything is ignored
        #                  until the button that started it is released
        self._gesture: str | None = None
        self._gesture_button = Qt.MouseButton.NoButton
        self._press_pos: QPointF | None = None
        self._pre_drag_viewport: cdx.Viewport | None = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

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

        A thin wrapper around the module-level complex_to_pixel (see its
        own docstring) -- kept as an instance method since it's directly
        exercised by existing tests and by every click-handling call site
        that means "this pane's own current screen position," not because
        the actual math lives here.
        """
        return complex_to_pixel(w, self.pane.viewport, self.width(), self.height())

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
        # On the parameter plane a pixel IS a value of a, not a point z.
        name = "a" if self.pane.render_mode in PARAMETER_PLANE_MODES else "z"
        coord = f"{name} = {w.real:.{decimals}f}{'+' if w.imag >= 0 else '-'}{abs(w.imag):.{decimals}f}j"
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
        # "basin", "parameter_basin", and "parameter_period" are ALWAYS
        # stacked; "julia" and "greens"/"parameter_greens" are stacked only
        # for a RATIONAL map's render (Stage 2/Stage 3 -- see render_map/
        # array_to_qimage's own docstrings, distinguished the SAME way
        # there: ndim, not a second flag threaded through).
        stacked = (mode in ("basin", "parameter_basin", "parameter_period") or
                  (mode in ("julia", "greens", "parameter_greens") and payload.ndim == 3))
        if stacked:
            height, width = payload[0].shape
        else:
            height, width = payload.shape

        if not (0 <= col < width and 0 <= row_top < height):
            return None   # cursor is over the display but off the (possibly overscanned-
                          # differently) buffer's own real pixels -- nothing to sample
        raw_row = height - 1 - row_top   # buffer is top-down; the array itself is row-0-bottom

        if mode == "julia" and stacked:
            value = payload[0][raw_row, col]
            label = payload[1][raw_row, col]
            return ("unresolved" if label == 0.0
                    else f"basin = {int(label)}, approach rate = {value:.4g}")
        if mode in ("julia", "parameter"):
            value = payload[raw_row, col]
            return "never escaped" if value == 0.0 else f"escape = {value:.4g}"
        if mode == "basin":
            label = payload[0][raw_row, col]
            return "unresolved" if label == 0.0 else f"basin = {int(label)}"
        if mode in ("greens", "parameter_greens") and stacked:
            value = payload[0][raw_row, col]
            exact = payload[1][raw_row, col]
            return (f"potential = {value:.4g}" if exact != 0.0
                    else f"potential = {value:.4g} (approx.)")
        if mode in ("greens", "parameter_greens"):
            value = payload[raw_row, col]
            return f"potential = {value:.4g}"
        if mode == "parameter_basin":
            count = payload[0][raw_row, col]
            unresolved = payload[1][raw_row, col]
            base = f"{int(count)} attracting cycle{'s' if count != 1 else ''}"
            return base if unresolved == 0.0 else f"{base}, {int(unresolved)} unresolved"
        if mode == "parameter_period":
            period = payload[0][raw_row, col]
            undetermined = payload[1][raw_row, col]
            is_infinity = payload[2][raw_row, col]
            if undetermined != 0.0:
                return "undetermined"
            if is_infinity != 0.0:
                return "converges to infinity"
            return f"period = {int(period)}"
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

    # ---- overlay layers: cached per (map, param), never re-root-found on pan/
    # ---- zoom -- see P5c's own "cache per (map, parameter); do not re-root-find
    # ---- on zoom or pan" requirement --------------------------------------------
    def refresh_layers(self) -> None:
        """Recomputes every registered OverlayLayer's resolved points (and,
        for any layer whose trace is currently enabled, its resolved trace
        paths too) for the CURRENT session.map/param -- memoized on
        (map.serialize(), param) so this is a no-op whenever neither has
        actually changed -- called by SandboxWindow after every term edit
        and family load (the same points that already refresh the Facts
        panel), NEVER from a viewport change. `refresh_critical_points` is
        kept as a same-behavior alias (see its own docstring) for existing
        callers/tests that still name it directly.

        A single cdx.DynamicalFacts (session.dynamical_facts()) is computed
        once per key and shared across every layer's provider (see
        OVERLAY_LAYERS' own module comment on why the critical-points layer
        specifically still ignores it) -- one root-find/attracting-cycle
        search serving every layer, not one per layer.
        """
        key = (self.session.map.serialize(), self.session.param)
        if key != self._layer_facts_key:
            self._layer_facts_key = key
            self._cached_facts = self.session.dynamical_facts()
            for layer in OVERLAY_LAYERS:
                resolved = list(layer.points_provider(self.session.map, self.session.param,
                                                       self._cached_facts))
                self._layer_points[layer.key] = resolved
                self._layer_point_values[layer.key] = [pt for pt, _color in resolved]
                if layer.trace is not None:
                    self._layer_traces[layer.trace.key] = None   # stale -- lazily recomputed below
                    self._layer_trace_values[layer.trace.key] = None
        for layer in OVERLAY_LAYERS:
            if (layer.trace is not None and self._layer_trace_enabled.get(layer.trace.key, False)
                    and self._layer_traces.get(layer.trace.key) is None):
                resolved_trace = list(layer.trace.path_provider(
                    self.session.map, self.session.param, self._cached_facts))
                self._layer_traces[layer.trace.key] = resolved_trace
                self._layer_trace_values[layer.trace.key] = [path for path, _color in resolved_trace]

    def refresh_critical_points(self) -> None:
        """Same-behavior alias for refresh_layers -- kept under its old name
        because it's called directly, by name, by existing tests and by
        every SandboxWindow call site that predates the layer registry.
        """
        self.refresh_layers()

    def set_layer_enabled(self, key: str, enabled: bool) -> None:
        self._layer_enabled[key] = enabled
        self.update()

    def set_trace_enabled(self, trace_key: str, enabled: bool) -> None:
        self._layer_trace_enabled[trace_key] = enabled
        self.refresh_layers()   # lazily fills in the trace if just turned on
        self.update()

    def set_show_critical_points(self, show: bool) -> None:
        self.set_layer_enabled(CRITICAL_POINTS_LAYER_KEY, show)

    def set_trace_orbits(self, trace: bool) -> None:
        self.set_trace_enabled(CRITICAL_TRACE_KEY, trace)

    # ---- compatibility properties: the pre-Stage-1 critical-point overlay's
    # own field names, now thin views over the generic per-layer storage
    # above -- kept because ExportImageDialog and existing tests read/write
    # them directly, by name (see this class's own __init__ comment).
    @property
    def _show_critical_points(self) -> bool:
        return self._layer_enabled[CRITICAL_POINTS_LAYER_KEY]

    @_show_critical_points.setter
    def _show_critical_points(self, value: bool) -> None:
        self._layer_enabled[CRITICAL_POINTS_LAYER_KEY] = value

    @property
    def _trace_orbits(self) -> bool:
        return self._layer_trace_enabled[CRITICAL_TRACE_KEY]

    @_trace_orbits.setter
    def _trace_orbits(self, value: bool) -> None:
        self._layer_trace_enabled[CRITICAL_TRACE_KEY] = value

    @property
    def _critical_points(self) -> list[complex]:
        return self._layer_point_values.get(CRITICAL_POINTS_LAYER_KEY, [])

    @property
    def _orbit_traces(self) -> list[list[complex]] | None:
        return self._layer_trace_values.get(CRITICAL_TRACE_KEY)

    @property
    def _critical_points_key(self) -> tuple[str, complex] | None:
        # The SAME key every layer shares (see refresh_layers) -- exposed
        # under its pre-Stage-1 name because existing tests compare it
        # directly to confirm panning/zooming never invalidates it.
        return self._layer_facts_key

    def set_orbit_connect_lines(self, on: bool) -> None:
        """Governs ONLY the seeded orbit's connecting segments (drawn in
        orange by paint_overlays' own show_orbit branch) -- the seeded
        orbit's own history. Independent of set_trace_orbits: the critical-
        points layer's own post-critical-point traces (drawn in white, via
        the SAME paint_overlays call) are a different overlay entirely and
        always connect when shown, regardless of this flag.
        """
        self._orbit_connect_lines = on
        self.update()

    def set_show_legend(self, show: bool) -> None:
        self._show_legend = show
        self.update()

    def set_overlay(self, key: str, on: bool) -> None:
        """One entry point for every overlay switch (see overlay_toggles):
        what the overlay chips and the View menu both call."""
        if key == ORBIT_TOGGLE_KEY:
            self._show_orbit = on
        elif key == PARAM_MARKER_TOGGLE_KEY:
            self._show_param_marker = on
        elif key == CROSSHAIR_TOGGLE_KEY:
            self._show_crosshair = on
        elif key in self._layer_enabled:
            self.set_layer_enabled(key, on)
            return
        elif key in self._layer_trace_enabled:
            self.set_trace_enabled(key, on)
            return
        else:
            raise KeyError(key)
        self.update()

    def overlay_on(self, key: str) -> bool:
        return {ORBIT_TOGGLE_KEY: self._show_orbit, PARAM_MARKER_TOGGLE_KEY: self._show_param_marker,
                CROSSHAIR_TOGGLE_KEY: self._show_crosshair}.get(
            key, self._layer_enabled.get(key, self._layer_trace_enabled.get(key, False)))

    # ---- readout chips (painted on the image, bottom-left) -------------------------
    def zoom_factor(self) -> float:
        """How far in from this map's default framing for this mode (x1 = default)."""
        m = self.session.map
        key = (m.name, m.to_formula(), self.pane.render_mode,
               None if self.pane.render_mode in PARAMETER_PLANE_MODES else self.session.param)
        if key != self._reference_scale_key:
            self._reference_scale_key = key
            try:
                self._reference_scale_value = default_view_for_mode(
                    m, self.session.param, self.pane.render_mode)[1]
            except Exception:   # a degenerate map: fall back to "no zoom shown"
                self._reference_scale_value = None
        ref = self._reference_scale_value
        scale = self.pane.viewport.scale
        return ref / scale if ref and scale > 0 else 1.0

    def readout_chips(self) -> list[tuple[str, str]]:
        """(accent label, value) pairs for the chips painted at the image's
        bottom-left: `a` and the zoom on the parameter plane; z0, the step
        count and the orbit's fate on the dynamical plane (once seeded)."""
        if self.pane.render_mode in PARAMETER_PLANE_MODES:
            zoom = self.zoom_factor()
            zoom_text = f"×{zoom:.2f}" if zoom < 100 else f"×{zoom:.3g}"
            return [("a", format_readout_complex(self.session.param)), ("", zoom_text)]
        state = self.orbit_tracker.state
        if state is None:
            return []
        rs = self.session.render_settings
        c = self.orbit_tracker.classify(self.session.map, self.session.param, rs.max_iter, rs.tol)
        fate = {"converged_fixed": "fixed point",
                "converged_cycle": f"{c.period}-cycle" if c.period else "cycle",
                "undetermined": "undetermined", "running": "running"}.get(c.kind, "")
        return [("z₀", format_readout_complex(state.z0)),
                ("", f"n {state.n}" + (f" · {fate}" if fate else ""))]

    def _paint_readout_chips(self, painter: QPainter) -> None:
        chips = self.readout_chips()
        if not chips:
            return
        rect = QRectF(self._display_rect())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(theme.mono_font(theme.FONT_SIZE_MONO))
        metrics = painter.fontMetrics()
        accent = theme.color("amber" if self.pane.render_mode in PARAMETER_PLANE_MODES else "teal")
        pad, gap, h = 8.0, 6.0, 24.0
        x, y = rect.left() + 10.0, rect.bottom() - 10.0 - h
        for label, value in chips:
            label_w = metrics.horizontalAdvance(label + " ") if label else 0
            w = pad * 2 + label_w + metrics.horizontalAdvance(value)
            if x + w > rect.right() - 10.0 and x > rect.left() + 10.0:
                break   # no room for this chip on a small image
            box = QRectF(x, y, w, h)
            painter.setPen(QPen(theme.color("border_strong"), 1))
            painter.setBrush(theme.color("overlay"))
            painter.drawRoundedRect(box, 6, 6)
            text_box = box.adjusted(pad, 0, -pad, 0)
            align = int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            if label:
                painter.setPen(accent)
                painter.drawText(text_box, align, label)
                text_box = text_box.adjusted(label_w, 0, 0, 0)
                painter.setPen(theme.color("text"))
            else:
                painter.setPen(theme.color("text_soft"))
            painter.drawText(text_box, align, value)
            x += w + gap
        painter.restore()

    # ---- off-screen parameter marker: an edge arrow that re-centers on it -------------
    OFFSCREEN_ARROW_INSET = 18.0
    OFFSCREEN_ARROW_SIZE = 11.0

    def offscreen_marker_arrow(self) -> tuple[QPointF, float] | None:
        """(tip position, angle in radians) of the edge arrow pointing toward
        the parameter marker when it lies outside the visible image, else None."""
        if self.pane.render_mode not in PARAMETER_PLANE_MODES or not self._show_param_marker:
            return None
        a = self.session.param
        if not (math.isfinite(a.real) and math.isfinite(a.imag)):
            return None
        rect = QRectF(self._display_rect())
        pixel = self._complex_to_display_pixel(a)
        if rect.contains(pixel):
            return None
        c = rect.center()
        dx, dy = pixel.x() - c.x(), pixel.y() - c.y()
        half_w = rect.width() / 2 - self.OFFSCREEN_ARROW_INSET
        half_h = rect.height() / 2 - self.OFFSCREEN_ARROW_INSET
        if half_w <= 0 or half_h <= 0:
            return None
        t = min(half_w / abs(dx) if dx else math.inf, half_h / abs(dy) if dy else math.inf)
        if not math.isfinite(t):
            return None
        return QPointF(c.x() + dx * t, c.y() + dy * t), math.atan2(dy, dx)

    def _offscreen_arrow_hit(self, pos: QPointF) -> bool:
        arrow = self.offscreen_marker_arrow()
        if arrow is None:
            return False
        tip, _angle = arrow
        r = self.OFFSCREEN_ARROW_SIZE + 8
        return QRectF(tip.x() - r, tip.y() - r, 2 * r, 2 * r).contains(pos)

    def recenter_on_param(self) -> None:
        vp = self.pane.viewport
        self.pane.viewport = cdx.Viewport(self.session.param, vp.scale, vp.resolution)
        self.update()
        self.viewport_changed.emit()

    def _paint_offscreen_arrow(self, painter: QPainter) -> None:
        arrow = self.offscreen_marker_arrow()
        if arrow is None:
            return
        tip, angle = arrow
        s = self.OFFSCREEN_ARROW_SIZE
        back = QPointF(tip.x() - s * 1.6 * math.cos(angle), tip.y() - s * 1.6 * math.sin(angle))
        side = QPointF(-math.sin(angle) * s, math.cos(angle) * s)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(theme.color("bg"), 2))
        painter.setBrush(theme.color("mark_param"))
        painter.drawPolygon(QPolygonF([tip, back + side, back - side]))
        painter.restore()

    # ---- orbit tracking: click-to-seed, Step/Run N/Clear -------------------------
    def refresh_orbit_staleness(self) -> None:
        """Called at the same map/param-change points refresh_layers is
        (term edits, family loads) -- clears the traced orbit the moment
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
        self.seed_orbit(self._pixel_to_complex(pixel))

    def seed_orbit(self, z0: complex) -> None:
        """The actual seeding logic -- factored out from _seed_orbit_at so
        a caller that already HAS an exact complex point (a fixed/critical
        point from FactsPanel, Stage 5 -- see
        SandboxWindow._on_seed_orbit_from_facts) can seed it directly,
        without a lossy pixel round-trip through whatever this pane's
        CURRENT viewport happens to be (the point may not even be on
        screen right now).
        """
        if self.pane.render_mode in PARAMETER_PLANE_MODES:
            return   # orbit tracking is a dynamical-plane concept, see its own module docstring
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
            painter.fillRect(self.rect(), theme.color("bg"))
            painter.setPen(theme.color("muted"))
            painter.drawText(self.rect(),
                             int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                             message)
            return
        source_rect = self._buffer_source_rect()
        if source_rect is not None:
            painter.drawPixmap(QRectF(self._display_rect()), self._pixmap, source_rect)
        if self._rubber_band_rect is not None:
            pen = QPen(theme.color("text"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(self._rubber_band_rect)
        # The actual overlay drawing is the module-level paint_overlays
        # (Stage B) -- this pane's own state, translated into its flags/
        # data, exactly reproducing what _paint_critical_point_overlay/
        # _paint_orbit/_paint_param_marker used to do directly (now
        # removed; see paint_overlays' own docstring). show_orbit/
        # show_param_marker are always True here -- the live view has no
        # "hide the orbit/marker entirely" toggle of its own (that choice
        # only exists for File > Export's preview dialog); paint_overlays'
        # own mode-gating (PARAMETER_PLANE_MODES) still governs which of
        # the two actually draws anything for THIS pane's render_mode.
        paint_overlays(painter, self.pane.viewport, self.width(), self.height(),
                       self.pane.render_mode,
                       show_critical_points=self._show_critical_points,
                       critical_points=self._critical_points,
                       show_traced_orbits=self._trace_orbits,
                       orbit_traces=self._orbit_traces,
                       show_orbit=self._show_orbit,
                       orbit_connect_lines=self._orbit_connect_lines,
                       orbit_state=self.orbit_tracker.state,
                       show_param_marker=self._show_param_marker,
                       show_crosshair=self._show_crosshair,
                       param=self.session.param,
                       show_legend=self._show_legend,
                       layer_enabled=self._layer_enabled,
                       layer_trace_enabled=self._layer_trace_enabled)
        # The registry's OWN layers (Stage 2 onward: fixed points,
        # attracting cycles, ...) -- a SEPARATE call, not folded into
        # paint_overlays above, so Export's existing kwargs stay untouched
        # (see paint_registry_layers' own docstring).
        paint_registry_layers(painter, self.pane.viewport, self.width(), self.height(),
                              self.pane.render_mode,
                              layer_enabled=self._layer_enabled,
                              layer_trace_enabled=self._layer_trace_enabled,
                              layer_points=self._layer_points,
                              layer_traces=self._layer_traces)
        self._paint_offscreen_arrow(painter)
        self._paint_readout_chips(painter)

    def _should_draw_critical_points(self) -> bool:
        # Critical points are objects of the DYNAMICAL plane -- a pixel on
        # the parameter plane is a PARAMETER, not a point the map acts on,
        # so "the map's critical point at this pixel" isn't a meaningful
        # thing to mark there (see P5c's own spec on this exact
        # distinction). False on those modes, not an error -- the
        # checkboxes stay checked/available for whenever the user switches
        # back to a dynamical-plane mode. Routed through the registry's own
        # gate (OverlayLayer.is_visible_for) rather than re-checking
        # PARAMETER_PLANE_MODES inline, so this and the registry can never
        # silently disagree about what "dynamical-plane gated" means.
        return (self._show_critical_points
               and _OVERLAY_LAYERS_BY_KEY[CRITICAL_POINTS_LAYER_KEY]
                   .is_visible_for(self.pane.render_mode))

    def _orbit_line_segments(self):
        """The line segments paint_overlays would draw for the CURRENT
        orbit, already respecting both the dynamical-plane-only gate and
        the connect-lines toggle -- factored out so this is directly
        testable without a real QPainter/paint event. Returns [] whenever
        paint_overlays would draw no lines at all: no orbit, parameter-
        plane mode, or the toggle off.
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

    def _param_marker_pixel(self) -> QPointF | None:
        """The screen pixel for session.param's crosshair marker on a
        PARAMETER-plane pane, or None if nothing should be drawn (a
        dynamical-plane mode, a non-finite param, or off the visible
        rect) -- factored out so this is directly testable without a real
        QPainter/paint event, the same reason _orbit_line_segments exists.

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

    # ---- zoom: one path for wheel, pinch and keyboard --------------------------
    def _zoom_at(self, pixel: QPoint | QPointF, factor: float) -> None:
        """Zooms by `factor` (> 1 zooms in) keeping the plane point under
        `pixel` fixed on screen. The single zoom path: wheel, trackpad
        scroll, Ctrl+wheel (Windows precision-touchpad pinch), the macOS
        native pinch gesture and keyboard zoom all come through here."""
        if not (factor > 0 and math.isfinite(factor)) or factor == 1.0:
            return
        w = self._pixel_to_complex(pixel)
        vp = self.pane.viewport
        new_center = w - (w - vp.center) / factor
        new_scale = vp.scale / factor
        self.pane.viewport = cdx.Viewport(new_center, new_scale, vp.resolution)

        # Instant feedback: repainting now re-derives the buffer source
        # rect from the viewport just set above -- see _buffer_source_rect.
        self.update()

        self.viewport_changed.emit()

    # ---- scroll zoom, cursor-anchored -------------------------------------------
    def wheelEvent(self, event) -> None:
        # Modifiers are deliberately NOT inspected: a Windows precision-
        # touchpad pinch arrives as wheel events with ControlModifier set,
        # and must zoom exactly like a plain wheel/two-finger scroll.
        # angleDelta is already fractional on smooth trackpads (notches need
        # not be integers); pixelDelta is only a fallback for drivers that
        # report nothing else -- see PIXEL_DELTA_PER_NOTCH.
        delta = event.angleDelta().y()
        if delta != 0:
            notches = delta / 120.0
        else:
            pixel_delta = event.pixelDelta().y()
            if pixel_delta == 0:
                event.accept()
                return
            notches = pixel_delta / PIXEL_DELTA_PER_NOTCH
        self._zoom_at(event.position(), ZOOM_FACTOR_PER_NOTCH ** notches)
        event.accept()

    # ---- macOS trackpad pinch -------------------------------------------------
    def event(self, event) -> bool:
        if (event.type() == QEvent.Type.NativeGesture and
                event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture):
            # value() is the INCREMENTAL scale change since the previous
            # gesture event (e.g. 0.02 = 2% bigger), not a cumulative scale.
            self._zoom_at(event.position(), 1.0 + event.value())
            event.accept()
            return True
        return super().event(event)

    # ---- drag: pan (plain), box zoom (Shift), click = seed / set a --------------
    @staticmethod
    def drag_threshold() -> int:
        """Screen-pixel distance a left press must travel before it stops
        being a click -- the platform's own drag distance (CLAUDE.md:
        click-vs-drag thresholds belong in screen pixels). Also the minimum
        box-zoom side."""
        return QApplication.startDragDistance()

    def _update_cursor(self, modifiers=None) -> None:
        if self._gesture == "pan":
            shape = Qt.CursorShape.ClosedHandCursor
        elif self._gesture == "box":
            shape = Qt.CursorShape.CrossCursor
        elif self._gesture is None and modifiers is not None and \
                modifiers & Qt.KeyboardModifier.ShiftModifier:
            shape = Qt.CursorShape.CrossCursor
        else:
            shape = Qt.CursorShape.OpenHandCursor
        if self.cursor().shape() != shape:
            self.setCursor(shape)

    def _begin_pan(self) -> None:
        self._gesture = "pan"
        # Anchored at the PRESS position under the PRE-drag viewport: the
        # first visible move then already includes the threshold distance,
        # so the image never jumps when the threshold is crossed.
        self._pan_anchor_complex = self._pixel_to_complex(self._press_pos)
        self._update_cursor()

    def _end_gesture(self) -> None:
        self._gesture = None
        self._gesture_button = Qt.MouseButton.NoButton
        self._press_pos = None
        self._pre_drag_viewport = None
        self._pan_anchor_complex = None
        self._rubber_band_rect = None

    def gesture_active(self) -> bool:
        return self._gesture is not None and self._gesture != "cancelled"

    def cancel_gesture(self) -> bool:
        """Esc mid-gesture: a pan restores the pre-drag viewport, a box
        zoom discards its rectangle, a pending click is dropped. Returns
        whether anything was cancelled. The gesture stays 'cancelled' (all
        moves ignored) until its button is released."""
        if not self.gesture_active():
            return False
        was_pan = self._gesture == "pan"
        pre = self._pre_drag_viewport
        self._rubber_band_rect = None
        self._pan_anchor_complex = None
        self._gesture = "cancelled"
        if was_pan and pre is not None:
            self.pane.viewport = cdx.Viewport(pre.center, pre.scale, pre.resolution)
            self.viewport_changed.emit()
        self._update_cursor()
        self.update()
        return True

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.pane_activated.emit()
        if self._gesture is not None:
            return   # a second button mid-gesture is ignored
        button = event.button()
        if button not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            return
        if button == Qt.MouseButton.LeftButton and self._offscreen_arrow_hit(QPointF(event.position())):
            self.recenter_on_param()   # the edge arrow: jump back to the marker
            return
        self._gesture_button = button
        self._press_pos = QPointF(event.position())
        vp = self.pane.viewport
        self._pre_drag_viewport = cdx.Viewport(vp.center, vp.scale, vp.resolution)
        if button == Qt.MouseButton.MiddleButton:
            self._begin_pan()
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._gesture = "box"
            self._update_cursor()
        else:
            self._gesture = "pending"
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos_f = QPointF(event.position())
        pos = pos_f.toPoint()
        if self._gesture == "pending":
            if (pos_f - self._press_pos).manhattanLength() > self.drag_threshold():
                self._begin_pan()
        if self._gesture == "pan":
            vp = self.pane.viewport
            current_point = self._pixel_to_complex(pos_f)
            # Shift the center so the point that was under the cursor at
            # press time is again under the cursor now -- see the module
            # test suite for the derivation this formula comes from.
            new_center = vp.center + (self._pan_anchor_complex - current_point)
            self.pane.viewport = cdx.Viewport(new_center, vp.scale, vp.resolution)
            self.update()
            self.viewport_changed.emit()
        elif self._gesture == "box":
            self._rubber_band_rect = QRect(self._press_pos.toPoint(), pos).normalized()
            self.update()
        elif self._gesture is None:
            if self._offscreen_arrow_hit(pos_f):
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.setToolTip("Re-center on the parameter marker")
            else:
                self.setToolTip("")
                self._update_cursor(event.modifiers())

        # Live, on every move regardless of panning/dragging -- the cursor
        # coordinate (and, mid-pan, what's still on screen at that point)
        # stays meaningful throughout the gesture, not just once it ends.
        self.cursor_readout_changed.emit(self.cursor_readout_text(pos))

    def enterEvent(self, event) -> None:
        self._update_cursor(QApplication.queryKeyboardModifiers())
        self.hover_changed.emit(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._gesture is None:
            self._update_cursor()
        self.cursor_readout_changed.emit("")
        self.hover_changed.emit(False)

    def mouseReleaseEvent(self, event) -> None:
        if self._gesture is None or event.button() != self._gesture_button:
            return
        gesture = self._gesture
        press_pos = self._press_pos
        rect = self._rubber_band_rect
        self._end_gesture()
        self._update_cursor(event.modifiers())
        self.update()
        if gesture == "pan":
            self.viewport_changed.emit()
        elif gesture == "box":
            # QRect(p1, p2) is inclusive (width = |dx| + 1); compare the
            # DRAGGED distance itself against the threshold.
            threshold = self.drag_threshold()
            if rect is not None and rect.width() - 1 > threshold and rect.height() - 1 > threshold:
                self._apply_rubber_band_zoom(rect)
            # else: too small to be a meaningful box -- discarded, NOT a click
        elif gesture == "pending":
            # A genuine CLICK, not a drag -- P6's "symmetry of input":
            # a click on the DYNAMICAL plane seeds an orbit at z0
            # (P5c's original "click to seed an orbit"); a click on
            # the PARAMETER plane sets `a` instead, the gesture that
            # previously did nothing at all there (_seed_orbit_at's
            # own early return for PARAMETER_PLANE_MODES).
            origin = press_pos.toPoint()
            if self.pane.render_mode in PARAMETER_PLANE_MODES:
                self._set_param_at(origin)
            else:
                self._seed_orbit_at(origin)
        # "cancelled": nothing

    # ---- keyboard: plot-scoped keys --------------------------------------------
    # Arrows: pan (dynamical plane) / nudge the parameter marker (parameter
    # plane, Stage 4 -- see param_nudged's own comment); Alt+arrows pan on
    # either. +/-: zoom at center. Space / Shift+Space: step / run the orbit
    # (dynamical plane). Esc: cancel a drag.
    @staticmethod
    def _plain_modifiers(event) -> Qt.KeyboardModifier:
        # macOS tags arrow keys with KeypadModifier, and the numpad +/- carry
        # it everywhere -- it says which physical key, not a chord.
        return event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier

    def keyPressEvent(self, event) -> None:
        """Plot-scoped keys (controls redesign, Step 2). Handled HERE, with
        focus scoping, rather than as application shortcuts -- so none of
        them fire while typing in a text field or activate a focused button.
        Any chord with Ctrl/Cmd or Meta falls through to Qt (and so to the
        window's QAction shortcuts)."""
        key = event.key()
        mods = self._plain_modifiers(event)
        if key in _CANCEL_KEYS and self.cancel_gesture():
            event.accept()
            return
        if key == Qt.Key.Key_Shift:
            self._update_cursor(mods | Qt.KeyboardModifier.ShiftModifier)
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            super().keyPressEvent(event)
            return

        direction = _ARROW_DIRECTIONS.get(key)
        if direction is not None:
            event.accept()   # so an arrow never also triggers focus navigation
            if event.isAutoRepeat():
                return   # our own timers drive held-key cadence, not Qt's OS-level repeat
            # Dynamical plane: arrows pan. Parameter plane: arrows nudge the
            # marker, Alt/Option+arrows pan. Decided per key press.
            pans = (self.pane.render_mode not in PARAMETER_PLANE_MODES or
                    bool(mods & _ALT_PAN_MODIFIERS))
            if pans:
                self._held_pan_arrows.add(direction)
                self._do_key_pan()
                self._pan_timer.start(PAN_KEY_REPEAT_DELAY_MS)
            else:
                self._held_arrows.add(direction)
                self._do_nudge()
                self._nudge_timer.start(self._nudge_interval_ms())
            return

        if mods & Qt.KeyboardModifier.AltModifier:
            super().keyPressEvent(event)
            return
        if key in _ZOOM_IN_KEYS:
            self.zoom_at_center(KEY_ZOOM_FACTOR)
            event.accept()
            return
        if key in _ZOOM_OUT_KEYS:
            self.zoom_at_center(1.0 / KEY_ZOOM_FACTOR)
            event.accept()
            return
        if key in _ORBIT_STEP_KEYS and self.pane.render_mode not in PARAMETER_PLANE_MODES:
            if (mods & _RUN_N_MODIFIERS) == _RUN_N_MODIFIERS:
                self.run_n_requested.emit()
            else:
                self.step_orbit(1)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Shift:
            self._update_cursor(self._plain_modifiers(event) & ~Qt.KeyboardModifier.ShiftModifier)
        direction = _ARROW_DIRECTIONS.get(event.key())
        # Only act if this direction is actually tracked as held -- keeps
        # this symmetric with an unconsumed press and side-steps re-checking
        # render_mode here. Checked BEFORE isAutoRepeat() -- an untracked
        # direction must fall through to super() regardless of autorepeat,
        # and a TRACKED one must never reach super() at all once accepted
        # (Qt's C++-typed base implementation rejects a non-QKeyEvent, which
        # an accepted-then-forwarded autorepeat release would otherwise hit).
        tracked = direction is not None and (direction in self._held_arrows or
                                             direction in self._held_pan_arrows)
        if not tracked:
            super().keyReleaseEvent(event)
            return
        event.accept()
        if event.isAutoRepeat():
            return   # the key is still physically held; only a genuine release stops the timer
        self._held_arrows.discard(direction)
        self._held_pan_arrows.discard(direction)
        if not self._held_arrows:
            self._nudge_timer.stop()
        if not self._held_pan_arrows:
            self._pan_timer.stop()

    def focusOutEvent(self, event) -> None:
        # A key can be physically released after focus has already moved
        # elsewhere (or the window itself lost focus) -- Qt does not
        # synthesize a keyReleaseEvent for that, so without this a held
        # arrow could leave a repeat timer running forever.
        self._held_arrows.clear()
        self._nudge_timer.stop()
        self._held_pan_arrows.clear()
        self._pan_timer.stop()
        super().focusOutEvent(event)

    # ---- keyboard pan / zoom -------------------------------------------------------
    def zoom_at_center(self, factor: float) -> None:
        """Keyboard and View-menu zoom: anchored at the pane's center."""
        self._zoom_at(QRectF(self._display_rect()).center(), factor)

    def pan_by_fraction(self, direction: complex, fraction: float = KEY_PAN_FRACTION) -> None:
        """Moves the view by `fraction` of its half-width (viewport.scale) in
        `direction` (an _ARROW_DIRECTIONS sum: Right = +real, Up = +imag)."""
        vp = self.pane.viewport
        new_center = vp.center + direction * fraction * vp.scale
        self.pane.viewport = cdx.Viewport(new_center, vp.scale, vp.resolution)
        self.update()
        self.viewport_changed.emit()

    def _do_key_pan(self) -> None:
        direction = sum(self._held_pan_arrows, complex(0.0, 0.0))
        if direction != 0:   # e.g. Left+Right both held cancel out
            self.pan_by_fraction(direction)

    def _on_pan_timer_tick(self) -> None:
        self._do_key_pan()
        self._pan_timer.setInterval(PAN_KEY_REPEAT_MS)

    def _nudge_interval_ms(self) -> int:
        # Read LIVE every time (called on every timer restart AND every
        # tick -- see _on_nudge_timer_tick) so a Settings change to
        # param_marker_rate takes effect on a held key immediately, not
        # only on the next fresh press.
        rate = max(self.session.settings.param_marker_rate, 1e-6)
        return max(1, round(1000.0 / rate))

    def _on_nudge_timer_tick(self) -> None:
        self._do_nudge()
        self._nudge_timer.setInterval(self._nudge_interval_ms())

    def _do_nudge(self) -> None:
        if not self._held_arrows:
            return
        direction = sum(self._held_arrows, complex(0.0, 0.0))
        if direction == 0:
            return   # e.g. Left+Right both held -- they cancel, nothing to emit
        self.param_nudged.emit(self.session.param + self._nudge_delta(direction))

    def _nudge_delta(self, direction: complex) -> complex:
        """A `param_marker_step`-pixel screen-space move in `direction`
        (see _ARROW_DIRECTIONS), converted to a complex delta through THIS
        pane's own current viewport -- differences two _pixel_to_complex
        calls rather than re-deriving its affine scale factor by hand, so
        this can never drift out of sync with the actual pixel<->complex
        mapping every other interaction in this view already uses.
        """
        step = self.session.settings.param_marker_step
        rect = self._display_rect()
        origin = QPointF(rect.center())
        moved = QPointF(origin.x() + direction.real * step, origin.y() - direction.imag * step)
        return self._pixel_to_complex(moved) - self._pixel_to_complex(origin)

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


def compose_export_image(rational_map: cdx.RationalMap, param: complex, viewport: cdx.Viewport,
                         render_mode: str, color_settings: Settings,
                         render_settings: cdx.RenderSettings, resolution: int,
                         **overlay_flags) -> QImage:
    """Renders (rational_map, param) at `viewport`'s own center/scale but a
    CHOSEN `resolution` -- via render_map, the SAME free function every
    on-screen render already goes through, not a downsample/upscale of
    whatever happens to be on screen -- colored through the identical
    array_to_qimage pipeline the display itself uses (same palette/
    scaling/period, so an export always matches what is actually shown),
    then composites whichever overlays are requested on top (paint_overlays
    -- **overlay_flags forwards show_critical_points/critical_points/
    show_traced_orbits/orbit_traces/show_orbit/orbit_connect_lines/
    orbit_state/show_param_marker straight through to it, so this
    function's own signature never has to be kept in sync with that one's).
    The param MARKER always draws at this SAME `param` -- the one thing
    the overlay compositing genuinely cannot diverge from what was just
    rendered -- so `overlay_flags` should never include its own "param"
    key; this function supplies it.

    ONE function, called by BOTH the export preview dialog (a live, cheap-
    resolution preview) and SandboxWindow._do_export_image (the actual
    save) -- so the saved PNG can never show something different from what
    the preview showed; there is no second copy of "render, color,
    composite" logic that could drift out of sync with this one.
    """
    export_viewport = cdx.Viewport(viewport.center, viewport.scale, resolution)
    array = render_map(rational_map, param, export_viewport, render_settings, render_mode,
                       potential=color_settings.greens_potential,
                       strategy=color_settings.parameter_strategy,
                       critical_index=color_settings.parameter_critical_index)
    image = array_to_qimage(array, render_mode, color_settings, render_settings.max_iter)
    painter = QPainter(image)
    paint_overlays(painter, export_viewport, resolution, resolution, render_mode,
                   param=param, **overlay_flags)
    painter.end()
    return image


# A live preview stays fast (and doesn't need to match the final export's
# own resolution pixel-for-pixel -- only the composition/overlays it shows
# need to match) at a small, fixed size, recomposed on every checkbox
# toggle.
EXPORT_PREVIEW_RESOLUTION = 240


# List-icon-sized (Stage C) -- a saved family/experiment browsed at a
# glance, not a real export.
THUMBNAIL_RESOLUTION = 64


def render_thumbnail(rational_map: cdx.RationalMap, param: complex, color_settings: Settings,
                     render_settings: cdx.RenderSettings,
                     resolution: int = THUMBNAIL_RESOLUTION) -> QImage:
    """A small CLEAN dynamical-plane render (Julia set, no overlays) of
    (rational_map, param) -- the same render_map + array_to_qimage pipeline
    every other render goes through, so a thumbnail always matches what the
    map actually looks like, never a stale/approximated stand-in.

    Always the DYNAMICAL plane (never parameter): a family's own critical/
    fixed-point structure lives there regardless of whether `a` affects
    anything at all, whereas a parameter-plane render can be entirely
    degenerate for a map like newton_cubic() where `a` has no effect (see
    ImageView.no_effect_parameter_message) -- there is no single parameter-
    plane framing that is meaningful for every possible family, but
    default_dynamical_view's fixed center-0 framing (Stage A) is.
    """
    center, scale = default_dynamical_view(rational_map, param)
    viewport = cdx.Viewport(center, scale, resolution)
    array = render_map(rational_map, param, viewport, render_settings, "julia")
    return array_to_qimage(array, "julia", color_settings, render_settings.max_iter)


def qimage_to_base64_png(image: QImage) -> str:
    """PNG-encodes `image` in memory (QBuffer, never touches disk) and
    base64-encodes the bytes -- how a thumbnail is embedded inline in a
    .cdsx experiment snapshot (a plain JSON file) rather than written as a
    sidecar the way a saved family's preview is (see preview_path_for):
    an experiment is already a single self-contained file, and it should
    stay one file, not gain a sidecar of its own.
    """
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.data().data()).decode("ascii")


def _regenerate_library_previews(library: cdx.FamilyLibrary, color_settings: Settings,
                                 render_settings: cdx.RenderSettings) -> None:
    """(Re)writes a sidecar thumbnail (see app.settings.preview_path_for) for
    every non-preset entry currently in `library`, and removes any leftover
    sidecar for a name no longer present -- so previews/ never drifts out of
    sync with what save_user_library just wrote to library.txt. Called after
    EVERY persisted library change (SandboxWindow._on_library_changed): a
    rename or delete is handled for free by recomputing from the library's
    current contents rather than special-casing each action, since a rename
    changes an entry's hashed filename and a delete just drops it from the
    "keep" set below.

    Rendered at a=0 -- RationalMap itself carries no stored parameter (see
    Session.param, which belongs to the session, not to any one library
    entry, and may currently be showing a completely different family), so
    there is no "the" parameter a saved entry was last viewed at to reuse;
    0 is a plain, always-valid evaluation point, not a per-entry guess.
    """
    keep_paths = set()
    for name in library.names():
        if name in PRESET_FAMILY_NAMES:
            continue
        entry = library.find(name)
        if entry is None:
            continue
        path = preview_path_for(name)
        keep_paths.add(path)
        image = render_thumbnail(entry, 0j, color_settings, render_settings)
        image.save(str(path), "PNG")
    for existing in previews_dir().glob("*.png"):
        if existing not in keep_paths:
            existing.unlink()


class ExportImageDialog(QDialog):
    """File > Export Image's preview dialog (Stage B): a resolution input,
    a checkbox per overlay APPLICABLE to the pane's current mode/state
    (never one for an overlay that couldn't draw anything -- e.g. no
    orbit/connect-lines checkboxes at all when no orbit is seeded), and a
    live preview of the exact composition compose_export_image would
    produce, updated on every toggle. Save hands the chosen (resolution,
    overlay flags) back to the caller (SandboxWindow._on_export_image),
    which still owns the actual QFileDialog + _do_export_image call --
    this dialog never touches disk itself, matching every other modal-
    dialog split in this file (see Save/Open Experiment's own comment).
    """

    def __init__(self, session: Session, pane: Pane, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        self._session = session
        self._pane = pane
        self._is_dynamical = pane.render_mode not in PARAMETER_PLANE_MODES
        self._has_orbit = pane.image_view.orbit_tracker.state is not None

        # None for any overlay NOT applicable to this pane's current
        # mode/state -- checked for None, never assumed present, by
        # overlay_flags() below.
        self._critical_points_checkbox: QCheckBox | None = None
        self._trace_orbits_checkbox: QCheckBox | None = None
        self._orbit_checkbox: QCheckBox | None = None
        self._connect_lines_checkbox: QCheckBox | None = None
        self._param_marker_checkbox: QCheckBox | None = None

        self._build_ui()
        self._update_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._preview_label = QLabel(self)
        self._preview_label.setFixedSize(EXPORT_PREVIEW_RESOLUTION, EXPORT_PREVIEW_RESOLUTION)
        layout.addWidget(self._preview_label)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution (pixels, square):", self))
        self._resolution_spin = QSpinBox(self)
        self._resolution_spin.setRange(1, 20000)
        self._resolution_spin.setValue(self._pane.viewport.resolution)
        res_row.addWidget(self._resolution_spin)
        res_row.addStretch(1)
        layout.addLayout(res_row)

        # Critical points + traced orbits: dynamical-plane-only concepts
        # (see PARAMETER_PLANE_MODES) -- not offered at all on a
        # parameter-plane pane, the same "only offer what's applicable"
        # this whole dialog follows throughout.
        if self._is_dynamical:
            view = self._pane.image_view
            self._critical_points_checkbox = QCheckBox("Critical points", self)
            self._critical_points_checkbox.setChecked(view._show_critical_points)
            self._critical_points_checkbox.toggled.connect(self._on_critical_points_toggled)
            layout.addWidget(self._critical_points_checkbox)

            self._trace_orbits_checkbox = QCheckBox("Traced (post-critical-point) orbits", self)
            self._trace_orbits_checkbox.setChecked(view._trace_orbits)
            # A sub-feature of critical points (see paint_overlays' own
            # docstring on this nesting, preserved from the original
            # code) -- disabled, not just silently ineffective, whenever
            # critical points itself is off.
            self._trace_orbits_checkbox.setEnabled(self._critical_points_checkbox.isChecked())
            self._trace_orbits_checkbox.toggled.connect(self._update_preview)
            layout.addWidget(self._trace_orbits_checkbox)

            # Orbit + connect-lines: only offered if an orbit is actually
            # SEEDED right now -- an empty toggle for a nonexistent orbit
            # has nothing to control.
            if self._has_orbit:
                self._orbit_checkbox = QCheckBox("Seeded orbit", self)
                self._orbit_checkbox.setChecked(True)   # matches what the pane is CURRENTLY showing
                self._orbit_checkbox.toggled.connect(self._on_orbit_toggled)
                layout.addWidget(self._orbit_checkbox)

                self._connect_lines_checkbox = QCheckBox("Connect orbit points", self)
                self._connect_lines_checkbox.setChecked(view._orbit_connect_lines)
                self._connect_lines_checkbox.toggled.connect(self._update_preview)
                layout.addWidget(self._connect_lines_checkbox)
        else:
            self._param_marker_checkbox = QCheckBox("Parameter marker", self)
            self._param_marker_checkbox.setChecked(True)   # matches what the pane is CURRENTLY showing
            self._param_marker_checkbox.toggled.connect(self._update_preview)
            layout.addWidget(self._param_marker_checkbox)

        # The painted legend is off on screen by default (the overlay chips
        # carry the swatches now), but an exported image has no chips: offer it.
        self._legend_checkbox = QCheckBox("Legend", self)
        self._legend_checkbox.setChecked(self._pane.image_view._show_legend)
        self._legend_checkbox.toggled.connect(self._update_preview)
        layout.addWidget(self._legend_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_critical_points_toggled(self, checked: bool) -> None:
        if self._trace_orbits_checkbox is not None:
            self._trace_orbits_checkbox.setEnabled(checked)
        self._update_preview()

    def _on_orbit_toggled(self, checked: bool) -> None:
        if self._connect_lines_checkbox is not None:
            self._connect_lines_checkbox.setEnabled(checked)
        self._update_preview()

    def overlay_flags(self) -> dict:
        """The paint_overlays-shaped kwargs for whatever is CURRENTLY
        checked -- shared by the live preview and, via
        SandboxWindow._on_export_image, the actual saved export, so the
        two can never show something different from each other. Traced-
        orbit data is computed HERE, on demand, only when that checkbox is
        actually checked (see trace_critical_orbit) -- not read off the
        live pane's own _orbit_traces cache, which may not exist at all if
        the live "Trace Orbits" toolbar checkbox was never turned on.

        Deliberately never includes a "param" key -- see
        compose_export_image's own docstring for why that one specific
        flag is ALWAYS supplied by the caller (the rendered param and the
        marker's own param can never be allowed to diverge), not by this
        dict.
        """
        flags: dict = {}
        view = self._pane.image_view
        if self._critical_points_checkbox is not None:
            show_critical = self._critical_points_checkbox.isChecked()
            flags["show_critical_points"] = show_critical
            flags["critical_points"] = view._critical_points
            show_traced = (self._trace_orbits_checkbox is not None
                          and self._trace_orbits_checkbox.isChecked())
            flags["show_traced_orbits"] = show_traced
            if show_traced:
                flags["orbit_traces"] = [
                    trace_critical_orbit(self._session.map, self._session.param, z0)
                    for z0 in view._critical_points]
        if self._orbit_checkbox is not None:
            flags["show_orbit"] = self._orbit_checkbox.isChecked()
            flags["orbit_connect_lines"] = (self._connect_lines_checkbox is not None
                                            and self._connect_lines_checkbox.isChecked())
            flags["orbit_state"] = view.orbit_tracker.state
        if self._param_marker_checkbox is not None:
            flags["show_param_marker"] = self._param_marker_checkbox.isChecked()
            flags["show_crosshair"] = view.overlay_on(CROSSHAIR_TOGGLE_KEY)
        if self._legend_checkbox.isChecked():
            flags["show_legend"] = True
            # Only what the export itself draws gets a legend line.
            flags["layer_enabled"] = {CRITICAL_POINTS_LAYER_KEY: flags.get("show_critical_points", False)}
            flags["layer_trace_enabled"] = {CRITICAL_TRACE_KEY: flags.get("show_traced_orbits", False)}
        return flags

    def resolution(self) -> int:
        return self._resolution_spin.value()

    def _update_preview(self) -> None:
        image = compose_export_image(
            self._session.map, self._session.param, self._pane.viewport, self._pane.render_mode,
            self._session.settings, self._session.render_settings, EXPORT_PREVIEW_RESOLUTION,
            **self.overlay_flags())
        self._preview_label.setPixmap(QPixmap.fromImage(image))


class SandboxWindow(QMainWindow):
    def __init__(self, layout_settings: QSettings | None = None):
        """`layout_settings`: where window geometry and dock state are restored
        from and saved to on close (main() passes default_layout_settings();
        None -- tests, tools -- means start from the default layout and save
        nothing)."""
        super().__init__()
        self.setWindowTitle(PRODUCT_NAME)   # the product name, not an internal milestone tag
        self.setMinimumSize(MIN_WINDOW_SIZE)
        self.resize(DEFAULT_WINDOW_SIZE)
        self._layout_settings = layout_settings
        self._param_undo: list[complex] = []
        self._param_redo: list[complex] = []
        self._last_nudge_time: float | None = None
        self._sidebar_auto_collapsed = False
        self._was_narrow = False
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
        self._apply_default_layout()
        self._default_layout_state = self.saveState(LAYOUT_STATE_VERSION)
        if layout_settings is not None:
            self.restore_layout(layout_settings)
        for pane in self.panes:
            pane.image_view.refresh_layers()
        self._update_status_bar()
        self._refresh_map_chip()
        self._refresh_render_status()
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
            view.hover_changed.connect(lambda inside, p=pane: self._on_pane_hover(p, inside))
            # No pane-capturing lambda needed here -- mousePressEvent
            # always emits pane_activated (moving focus) BEFORE a release
            # can ever emit param_changed, so by the time this fires,
            # self._focused_pane is already the clicked pane; see
            # _on_param_changed_by_click, which reads it as "the clicked pane."
            view.param_changed.connect(self._on_param_changed_by_click)
            # Same "no pane-capturing lambda needed" reasoning as
            # param_changed above: keyPressEvent only fires on whichever
            # ImageView actually has Qt keyboard focus, which mousePressEvent's
            # own setFocus() call (plus StrongFocus's click-to-focus) keeps
            # in sync with self._focused_pane -- see ImageView.param_nudged's
            # own comment for why this is a SEPARATE signal/handler from
            # param_changed, not a reuse of it.
            view.param_nudged.connect(self._on_param_nudged)
            view.run_n_requested.connect(lambda v=view: v.step_orbit(self.orbit_panel.run_n()))
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

        # Each pane is a PlaneFrame: caps title + segmented mode control,
        # the framed image, and a row of overlay chips (added in
        # _build_overlay_chips, once the View-menu actions they mirror exist).
        # mode_combo/mode_combo2 are compatibility shims over the segmented
        # controls (setCurrentText(mode) still switches that pane's mode).
        self.plane_frame = PlaneFrame(self.image_view, self)
        self.plane_frame2 = PlaneFrame(self.image_view2, self)
        self._plane_frames: dict[Pane, PlaneFrame] = {
            self.pane: self.plane_frame, self.pane2: self.plane_frame2}
        for pane, frame in self._plane_frames.items():
            frame.mode_control.selected.connect(
                lambda mode, p=pane: self._on_mode_changed(p, mode))
            frame.resized.connect(self._equalize_chip_rows)
        self.mode_combo = ModeSelector(self, self.pane)
        self.mode_combo2 = ModeSelector(self, self.pane2)
        self._mode_combos: dict[Pane, ModeSelector] = {
            self.pane: self.mode_combo, self.pane2: self.mode_combo2}
        self.pane_column, self.pane_column2 = self.plane_frame, self.plane_frame2   # old names
        self._pane_columns: dict[Pane, QWidget] = dict(self._plane_frames)

        # Side by side in a splitter, not two fixed halves -- the user can
        # drag the divider to give one pane more room. Coupled (both
        # visible) is the natural instrument default (see
        # self.coupled_view_action below); single-view collapses to just
        # whichever pane is currently focused (see _relayout_panes).
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.view_splitter.addWidget(self.pane_column)
        self.view_splitter.addWidget(self.pane_column2)

        # ---- Explore workspace: the two planes, side by side ----------------------
        # The metadata header's job moved to the top bar's map chip (and, per
        # pane, the readout chips): it is kept as an object for its text
        # (format_metadata_text) but never shown.
        self.metadata_header.hide()
        self.explore_workspace = QWidget(self)
        explore_layout = QVBoxLayout(self.explore_workspace)
        explore_layout.setContentsMargins(theme.GUTTER, theme.GUTTER, theme.GUTTER, theme.GUTTER)
        explore_layout.setSpacing(0)
        self.view_splitter.setHandleWidth(theme.GUTTER)
        self.view_splitter.setChildrenCollapsible(False)
        explore_layout.addWidget(self.view_splitter, 1)
        self.view_container = self.explore_workspace   # compatibility name

        # Central widget: one page per workspace. Explore is always first;
        # add_panel(placement="workspace") adds more and shows the switch.
        self.workspaces = QStackedWidget(self)
        self.workspaces.addWidget(self.explore_workspace)
        self._workspace_keys: dict[str, QWidget] = {"explore": self.explore_workspace}
        self._workspace_titles: dict[str, str] = {"explore": "Explore"}
        self.setCentralWidget(self.workspaces)
        self._extra_panels: list[QWidget] = []   # see add_panel

        # ---- the two complex fields that left the toolbar ---------------------------
        # `a` sits on the Equation tab (it is the active parameter's value) and
        # z0 on the Orbit tab; each stays in sync with clicks in the planes
        # (see _apply_param_change and _on_orbit_changed).
        self.param_field = ComplexField("", self.session.param, self)
        self.param_field.setAccessibleName("Value of the active parameter")
        self.param_field.committed.connect(self._on_param_field_committed)
        self.z0_field = ComplexField("", complex(0, 0), self)
        self.z0_field.setAccessibleName("Orbit seed z0")
        self.z0_field.committed.connect(self._on_z0_field_committed)

        # ---- sidebar (right dock): Info | Library | Facts ----------------------------
        self.sidebar = TabbedDock("Sidebar", "SidebarDock", "right", self)
        self.info_panel = InfoPanel(self)
        self.library_panel = LibraryPanel(self.session, self._on_family_loaded,
                                          self._on_library_changed, self,
                                          thumbnail_provider=self._preset_thumbnail)
        self.facts_panel = FactsPanel(self.session, self._on_center_view,
                                      self._on_seed_orbit_from_facts, self)
        # Info scrolls itself (a QTextBrowser); the others get a scroll area.
        self.sidebar.add_tab(self.info_panel, "Info", "Controls and this map", scroll=False)
        self.sidebar.add_tab(self.library_panel, "Library", "Families: presets and yours")
        self.sidebar.add_tab(self.facts_panel, "Facts", "Critical points, fixed points, cycles, poles")
        self.sidebar.set_current_widget(self.library_panel)
        self.sidebar.currentChanged.connect(self._on_sidebar_tab_changed)
        self.sidebar.collapsedChanged.connect(lambda _c: self._refresh_this_map())
        # Expanding the sidebar by hand makes it the user's choice again.
        self.sidebar.collapsedChanged.connect(
            lambda collapsed: None if collapsed else setattr(self, "_sidebar_auto_collapsed", False))
        self.sidebar.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sidebar)

        # ---- bottom dock: Equations & Settings ----------------------------------------
        self.settings_dock = TabbedDock("Equations & Settings", "SettingsDock", "bottom", self,
                                        heading="Equations & Settings")
        self.equation_panel = EquationPanel(self.session, self._on_term_edited, self)
        # The active parameter's value field IS the old toolbar `a =` field.
        self.equation_panel.set_active_field(self.param_field)
        self.equation_tab = QWidget(self)
        eq_layout = QVBoxLayout(self.equation_tab)
        eq_layout.setContentsMargins(18, 12, 18, 12)
        eq_layout.addWidget(self.equation_panel, 1)
        # Rendering and Coloring apply live (debounced); there is no dialog.
        # One shared pause-then-apply for both tabs, and one coalesced
        # re-render for whatever they applied together.
        self._settings_apply_timer = QTimer(self)
        self._settings_apply_timer.setSingleShot(True)
        self._settings_apply_timer.setInterval(SETTINGS_APPLY_DEBOUNCE_MS)
        self._settings_render_timer = QTimer(self)
        self._settings_render_timer.setSingleShot(True)
        self._settings_render_timer.setInterval(0)
        self._settings_render_timer.timeout.connect(self._render_after_settings)
        self.rendering_panel = SettingsPanel(self.session, self._on_settings_applied, self,
                                             groups=RENDERING_GROUPS,
                                             schedule_apply=self._settings_apply_timer.start)
        self.coloring_panel = SettingsPanel(self.session, self._on_settings_applied, self,
                                            groups=COLORING_GROUPS,
                                            schedule_apply=self._settings_apply_timer.start)
        self._settings_apply_timer.timeout.connect(self.rendering_panel.apply_pending)
        self._settings_apply_timer.timeout.connect(self.coloring_panel.apply_pending)
        self.settings_panel = self.rendering_panel   # old name
        self.rendering_tab, self.coloring_tab = self.rendering_panel, self.coloring_panel
        self.orbit_tab = QWidget(self)
        orbit_layout = QVBoxLayout(self.orbit_tab)
        orbit_layout.setContentsMargins(18, 12, 18, 12)
        orbit_layout.setSpacing(8)
        z0_row = QHBoxLayout()
        z0_label = QLabel("z₀ =")
        z0_label.setProperty("mono", True)
        z0_row.addWidget(z0_label)
        z0_row.addWidget(self.z0_field, 1)
        z0_row.addStretch(2)
        orbit_layout.addLayout(z0_row)
        orbit_layout.addWidget(self.orbit_panel)
        self.connect_lines_checkbox = QCheckBox("Connect orbit points", self)
        self.connect_lines_checkbox.setChecked(True)
        orbit_layout.addWidget(self.connect_lines_checkbox)
        orbit_layout.addStretch(1)
        self.settings_dock.add_tab(self.equation_tab, "Equation", "The map f(z) and its parameters")
        self.settings_dock.add_tab(self.rendering_tab, "Rendering", "Resolution, iterations, threads, cache")
        self.settings_dock.add_tab(self.coloring_tab, "Coloring", "Palette, scaling, Green's function")
        self.settings_dock.add_tab(self.orbit_tab, "Orbit", "Seed, step and run an orbit")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.settings_dock)
        # Bottom dock spans the full width under the planes AND the sidebar.
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)

        # ---- top bar (replaces the old toolbar) ---------------------------------------
        self.top_bar = TopBar(self)
        self.top_bar_toolbar = QToolBar("Top bar", self)
        self.top_bar_toolbar.setObjectName("TopBarToolbar")
        self.top_bar_toolbar.setMovable(False)
        self.top_bar_toolbar.setFloatable(False)
        self.top_bar_toolbar.toggleViewAction().setVisible(False)
        self.top_bar_toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.top_bar_toolbar.addWidget(self.top_bar)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.top_bar_toolbar)
        self.top_bar.map_chip.clicked.connect(lambda: self.sidebar.reveal(self.library_panel))
        self.top_bar.reset_button.clicked.connect(self._reset_view)
        self.top_bar.export_button.clicked.connect(self._on_export_image)
        self.top_bar.workspace_switch.selected.connect(self.show_workspace)
        self._render_stage: dict[Pane, str | None] = {pane: None for pane in self.panes}

        # Coupled view / Critical Points / Trace Orbits / Connect Orbit
        # Points used to be toolbar checkboxes here too (Stage 1-3 kept them
        # as a second control surface, mirroring the View-menu action
        # bidirectionally). Stage 4 removes them: the View menu is now the
        # SOLE control for all four -- see its own construction below, where
        # each action is wired DIRECTLY to the real behavior a checkbox
        # would have driven (coupled_view_action -> _relayout_panes,
        # orbit_connect_lines_action -> set_orbit_connect_lines; the two
        # registry-layer actions were ALREADY wired directly since Stage 1,
        # so removing their mirror checkboxes needed no rewiring at all).

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
        # Shortcuts: StandardKey wherever one exists, so Qt renders and binds
        # the native sequence per platform ("Ctrl" in a QKeySequence IS
        # Command on macOS -- never hard-code Cmd).
        self.save_experiment_action.setShortcuts(controls.key_sequences("app.save"))
        self.open_experiment_action.setShortcuts(controls.key_sequences("app.open"))

        # File > Export (Stage 4): a rendered PNG at a CHOSEN resolution, or
        # a JSON fact sheet -- both of the FOCUSED pane specifically (see
        # _do_export_image/_do_export_facts's own docstrings). Same
        # store-on-self reason as everything else on this menu bar.
        self.export_menu = self.file_menu.addMenu("Export")
        self.export_image_action = self.export_menu.addAction("Export Image (PNG)...")
        self.export_image_action.triggered.connect(self._on_export_image)
        self.export_image_action.setShortcuts(controls.key_sequences("app.export_image"))
        self.export_facts_action = self.export_menu.addAction("Export Facts (JSON)...")
        self.export_facts_action.triggered.connect(self._on_export_facts)
        self.file_menu.addSeparator()

        # Settings: shows the dock's Rendering tab (_on_open_settings).
        # PreferencesRole is what makes
        # Qt relocate this out of File and into the app menu on macOS
        # (Cmd+,) regardless of where it's added in code; File is just
        # where it lives on platforms that don't do that relocation.
        self.settings_action = self.file_menu.addAction("Settings...")
        self.settings_action.setMenuRole(QAction.MenuRole.PreferencesRole)
        self.settings_action.triggered.connect(self._on_open_settings)
        self.settings_action.setShortcuts(preferences_shortcuts())

        # ---- Edit menu: Undo/Redo of Equation edits -------------------------------
        # StandardKey.Undo/Redo: Cmd+Z / Shift+Cmd+Z on macOS; Ctrl+Z and
        # both Ctrl+Y and Ctrl+Shift+Z on Windows. A focused text field's
        # own undo wins: QLineEdit/QPlainTextEdit claim these sequences via
        # ShortcutOverride before the action sees them, and _route_undo
        # forwards to the field too when the action is chosen from the menu.
        self.edit_menu = self.menuBar().addMenu("Edit")
        self.undo_action = self.edit_menu.addAction("Undo")
        self.undo_action.setShortcuts(controls.key_sequences("app.undo"))
        self.undo_action.triggered.connect(lambda: self._route_undo(redo=False))
        self.redo_action = self.edit_menu.addAction("Redo")
        self.redo_action.setShortcuts(controls.key_sequences("app.redo"))
        self.redo_action.triggered.connect(lambda: self._route_undo(redo=True))

        # ---- View menu: the SOLE control for every toggle below (Stage 4) -------
        # Same store-on-self reason as file_menu/help_menu -- see help_menu's own
        # comment below. Through Stage 3 this menu MIRRORED a parallel set of
        # toolbar checkboxes (each one's toggled signal set the other's
        # checked state); Stage 4 removed those checkboxes entirely, so
        # every checkable action here is now wired DIRECTLY to the real
        # behavior a checkbox used to drive -- no more mirror, one owner.
        #
        # No Experiment/Map menu for mode selection: with TWO independent
        # per-pane mode combos (Stage 2) rather than one window-level mode,
        # "the map's mode" isn't a single menu-shaped concept any more --
        # which pane would a single menu control? Left on the per-pane
        # combos, as the spec's own "otherwise skip" allows.
        self.view_menu = self.menuBar().addMenu("View")
        self.reset_view_action = self.view_menu.addAction("Reset View")
        self.reset_view_action.triggered.connect(self._reset_view)
        self.reset_view_action.setShortcuts(controls.key_sequences("app.reset_view"))
        # Zoom the FOCUSED pane about its center (the plot's own +/- keys do
        # the same, focus-scoped; these work from anywhere in the window).
        self.zoom_in_action = self.view_menu.addAction("Zoom In")
        self.zoom_in_action.setShortcuts(controls.key_sequences("app.zoom_in"))
        self.zoom_in_action.triggered.connect(
            lambda: self._focused_pane.image_view.zoom_at_center(KEY_ZOOM_FACTOR))
        self.zoom_out_action = self.view_menu.addAction("Zoom Out")
        self.zoom_out_action.setShortcuts(controls.key_sequences("app.zoom_out"))
        self.zoom_out_action.triggered.connect(
            lambda: self._focused_pane.image_view.zoom_at_center(1.0 / KEY_ZOOM_FACTOR))
        self.view_menu.addSeparator()

        # Coupled (both panes visible) is the natural instrument default --
        # see _relayout_panes for what unchecking this actually does
        # (collapse to just the focused pane, not always pane A).
        self.coupled_view_action = self.view_menu.addAction("Coupled View")
        self.coupled_view_action.setCheckable(True)
        self.coupled_view_action.setChecked(True)
        self.coupled_view_action.toggled.connect(lambda _checked: self._relayout_panes())
        # The top bar's Coupled | Single control and this menu item are one
        # setting: each drives the other.
        self.coupled_view_action.toggled.connect(self.top_bar.view_switch.set_current)
        self.top_bar.view_switch.selected.connect(
            lambda coupled: self.coupled_view_action.setChecked(bool(coupled)))
        self.view_menu.addSeparator()

        # ---- overlay-layer actions: built FROM the registry (Stage 1), not
        # hand-written per overlay -- adding a layer to OVERLAY_LAYERS is
        # enough to give it a View-menu toggle (and its trace a gated sub-
        # toggle), with NO changes needed here. Each action is wired
        # DIRECTLY to image_view[2].set_layer_enabled/set_trace_enabled --
        # always has been, since Stage 1, which is why removing critical_
        # points_checkbox/trace_orbits_checkbox in Stage 4 needed no
        # rewiring here at all (see _build_view_menu_layer_actions's own
        # docstring).
        self._build_view_menu_layer_actions()

        self.orbit_connect_lines_action = self.view_menu.addAction("Connect Orbit Points")
        self.orbit_connect_lines_action.setCheckable(True)
        self.orbit_connect_lines_action.setChecked(True)
        self.orbit_connect_lines_action.toggled.connect(self.image_view.set_orbit_connect_lines)
        self.orbit_connect_lines_action.toggled.connect(self.image_view2.set_orbit_connect_lines)
        # The Orbit tab's checkbox and this menu item are one setting.
        self.orbit_connect_lines_action.toggled.connect(self.connect_lines_checkbox.setChecked)
        self.connect_lines_checkbox.toggled.connect(self.orbit_connect_lines_action.setChecked)
        self.view_menu.addSeparator()

        # Legend (Stage 3) -- menu-only, like Fixed Points/Attracting Cycles/
        # Trace Cycle Paths: no toolbar checkbox exists or is planned for it,
        # so it's wired directly to both ImageViews' set_show_legend rather
        # than going through _build_view_menu_layer_actions (which is for
        # OVERLAY_LAYERS entries specifically -- the legend isn't a data
        # layer with its own points/gate, just a compositor-level toggle).
        self.legend_action = self.view_menu.addAction("Legend")
        self.legend_action.setCheckable(True)
        self.legend_action.toggled.connect(self.image_view.set_show_legend)
        self.legend_action.toggled.connect(self.image_view2.set_show_legend)
        self.view_menu.addSeparator()

        # Ctrl+1..Ctrl+9 jump to the tabs in order -- rebuilt whenever a tab
        # is added (add_panel), so module tabs get numbers too.
        # Docks: toggleViewAction() reopens a closed dock; Reset Layout puts
        # every dock back where a fresh install has it.
        self.sidebar_action = self.sidebar.toggleViewAction()
        self.sidebar_action.setText("Sidebar")
        self.view_menu.addAction(self.sidebar_action)
        self.settings_dock_action = self.settings_dock.toggleViewAction()
        self.settings_dock_action.setText("Equations && Settings")
        self.view_menu.addAction(self.settings_dock_action)
        self.reset_layout_action = self.view_menu.addAction("Reset Layout")
        self.reset_layout_action.triggered.connect(self.reset_layout)
        self.view_menu.addSeparator()

        self.go_to_tab_menu = self.view_menu.addMenu("Go to Tab")
        self.tab_actions: list[QAction] = []
        self._rebuild_tab_actions()

        # On macOS, Qt moves a menu titled "Help" (and any action inside
        # named "About <AppName>") into the system application menu
        # automatically -- this is the standard, idiomatic way to get a
        # real "About Project Phoenix" entry there, not a bespoke button.
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
        # StandardKey.HelpContents (F1 on Windows, Cmd+? on macOS) plus F1
        # everywhere -- see the registry's app.info entry.
        self.info_action = self.help_menu.addAction("Controls && Shortcuts")
        self.info_action.setShortcuts(controls.key_sequences("app.info"))
        self.info_action.triggered.connect(lambda: self.sidebar.reveal(self.info_panel))

        self.statusBar().showMessage("")
        # Transient controls hint for the pane under the cursor: its OWN
        # label (right side), so it never overwrites the scale/coordinate
        # readout in the message area; cleared on leave or after
        # PANE_HINT_MS.
        self.pane_hint_label = QLabel(self)
        self.statusBar().addPermanentWidget(self.pane_hint_label)
        self._pane_hint_timer = QTimer(self)
        self._pane_hint_timer.setSingleShot(True)
        self._pane_hint_timer.setInterval(PANE_HINT_MS)
        self._pane_hint_timer.timeout.connect(self.pane_hint_label.clear)
        self._status_flash = ""
        self._status_flash_timer = QTimer(self)
        self._status_flash_timer.setSingleShot(True)
        self._status_flash_timer.timeout.connect(self._clear_status_flash)

        self._build_overlay_chips()
        self._update_pane_focus_styling()
        self._sync_mode_controls()
        self._sync_orbit_panel()

    def _on_open_settings(self) -> None:
        """File > Settings... / Cmd+, / Ctrl+,: the Rendering tab of the dock
        (there is no Settings dialog any more)."""
        self.settings_dock.reveal(self.rendering_tab)

    def _build_view_menu_layer_actions(self) -> None:
        """One checkable QAction per overlay switch (overlay_toggles(): every
        OVERLAY_LAYERS entry and its gated trace, plus the seeded orbit, the
        parameter marker and its crosshair) -- called once from _build_ui.
        The overlay chips under each plane are built from the SAME list
        (_build_overlay_chips) and mirror these actions, so the menu and the
        chips can never disagree.

        Each action drives BOTH ImageViews (one visual preference, not one
        per pane). A trace's action is enabled only while its layer is on.
        """
        self.overlay_actions: dict[str, QAction] = {}
        previous_gate = None
        for toggle in overlay_toggles():
            if toggle.parent is None and self.overlay_actions and toggle.gate != previous_gate:
                self.view_menu.addSeparator()
            previous_gate = toggle.gate
            action = self.view_menu.addAction(toggle.menu_label)
            action.setCheckable(True)
            action.setChecked(toggle.default)
            action.setToolTip(toggle.tooltip)
            for view in (self.image_view, self.image_view2):
                view.set_overlay(toggle.key, toggle.default)
                action.toggled.connect(lambda checked, v=view, k=toggle.key: v.set_overlay(k, checked))
            if toggle.parent is not None:
                parent_action = self.overlay_actions[toggle.parent]
                action.setEnabled(parent_action.isChecked())
                parent_action.toggled.connect(action.setEnabled)
            self.overlay_actions[toggle.key] = action
            setattr(self, f"{toggle.key}_action", action)
        self.view_menu.addSeparator()

    def _build_overlay_chips(self) -> None:
        """The chip row under each plane, one chip per overlay switch whose
        gate matches the plane; traces are small secondary chips."""
        for frame in self._plane_frames.values():
            for toggle in overlay_toggles():
                frame.add_chip(toggle.key, self.overlay_actions[toggle.key], toggle.chip_label,
                               toggle.gate, toggle.swatch, toggle.tooltip,
                               secondary=toggle.parent is not None)

    def _equalize_chip_rows(self) -> None:
        """Side by side, both planes' chip rows take the taller one's height,
        so the two images stay the same size and line up."""
        frames = [f for f in self._plane_frames.values() if f.isVisible()]
        height = max((f.chip_rows_height() for f in frames), default=0)
        for f in self._plane_frames.values():
            if f.chip_row.minimumHeight() != height:
                f.chip_row.setMinimumHeight(height)

    def _sync_mode_controls(self) -> None:
        """Each plane's header: title/accent from its current mode's plane,
        and the modes its segmented control offers -- in Coupled view only
        that plane's own family (parameter or dynamical); in Single view every
        mode, grouped "Parameter ▸ ... | Dynamical ▸ ...". Ids stay data; the
        buttons show MODE_LABELS."""
        if not hasattr(self, "coupled_view_action"):
            return
        coupled = self.coupled_view_action.isChecked()
        explore = not hasattr(self, "workspaces") or self.current_workspace() == "explore"
        for pane, frame in self._plane_frames.items():
            plane = "parameter" if pane.render_mode in PARAMETER_PLANE_MODES else "dynamical"
            frame.set_plane(plane)
            frame.set_chips_visible(explore)
            if coupled:
                family = PARAMETER_MODE_ORDER if plane == "parameter" else DYNAMICAL_MODE_ORDER
                groups, titles = [[_mode_segment(m) for m in family]], None
            else:
                groups = [[_mode_segment(m) for m in PARAMETER_MODE_ORDER],
                          [_mode_segment(m) for m in DYNAMICAL_MODE_ORDER]]
                titles = ["Parameter ▸", "Dynamical ▸"]
            key = (coupled, plane)
            if getattr(frame, "_mode_key", None) != key:
                frame._mode_key = key
                frame.mode_control.set_segments(groups, titles)
            frame.mode_control.set_current(pane.render_mode)
            frame.set_focused(pane is self._focused_pane)
        self._equalize_chip_rows()

    def _show_about_dialog(self) -> None:
        AboutDialog(self).exec()

    # ---- focus / layout: which pane Reset View, the header, and single- ---------
    # ---- view mode act on ---------------------------------------------------------
    def _set_focused_pane(self, pane: Pane) -> None:
        if pane is self._focused_pane:
            return
        self._focused_pane = pane
        self.metadata_header.pane = pane
        self._refresh_header()
        self._update_status_bar()
        self._update_pane_focus_styling()
        if not self.coupled_view_action.isChecked():
            self._relayout_panes()

    def _update_pane_focus_styling(self) -> None:
        # Purely visual -- which column has a highlighted border -- so the
        # user can actually SEE which pane Reset View/typed-field actions
        # (still self.pane-directed until Stage 3) are about to act on.
        for pane, frame in self._plane_frames.items():
            frame.set_focused(pane is self._focused_pane)

    def _relayout_panes(self) -> None:
        # Coupled: both panes visible, side by side. Single: only the
        # FOCUSED pane's column is visible -- not always pane A -- the
        # other pane keeps rendering/tracking its own state in the
        # background (nothing about it is torn down), it's just hidden.
        coupled = self.coupled_view_action.isChecked()
        for pane, column in self._pane_columns.items():
            column.setVisible(coupled or pane is self._focused_pane)
        self._sync_mode_controls()
        self._refresh_render_status()

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
            self, "Save Experiment", "", f"{PRODUCT_NAME} experiment (*.cdsx);;All files (*)")
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
        coupled = self.coupled_view_action.isChecked()
        orbit_pane = self._current_dynamical_pane()
        orbit = None
        if orbit_pane is not None:
            state = orbit_pane.image_view.orbit_tracker.state
            if state is not None:
                orbit = (self.panes.index(orbit_pane), state.z0, state.n)
        # Clean thumbnail (Stage C), same as a library entry's own -- see
        # render_thumbnail's own docstring for why it's always the
        # dynamical plane, never parameter.
        preview_image = render_thumbnail(self.session.map, self.session.param,
                                         self.session.settings, self.session.render_settings)
        preview = qimage_to_base64_png(preview_image)
        try:
            self.session.save_snapshot(path, panes, focused_index, coupled, orbit, preview)
        except OSError as exc:
            QMessageBox.critical(self, "Save Experiment failed", str(exc))

    def _on_open_experiment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Experiment", "", f"{PRODUCT_NAME} experiment (*.cdsx);;All files (*)")
        if path:
            self._do_open_experiment(path)

    def _do_open_experiment(self, path: str) -> None:
        # session.load_snapshot fully validates and parses BEFORE mutating
        # anything (see restore_from_snapshot's own docstring) -- on
        # failure the live session is untouched, so there is nothing here
        # to roll back, just an error to surface. Session has no panes,
        # viewport, or render_mode of its own, so the restored
        # (panes, focused_index, coupled, orbit, preview) come back
        # explicitly for THIS window to apply to its own panes/layout/
        # tracker. `preview` (Stage C) is intentionally unused here -- this
        # app has no experiment BROWSER yet for it to appear in (see
        # render_thumbnail's own module-level note); it round-trips through
        # save/load correctly (see app/test_sandbox.py) and is available
        # for a future browser to display without another format change.
        try:
            panes_data, focused_index, coupled, orbit, _preview = self.session.load_snapshot(path)
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
        self._sync_mode_controls()

        self.coupled_view_action.setChecked(coupled)
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
        self.equation_panel.refresh_from_session()
        self.facts_panel.refresh()
        self._refresh_header()
        for pane in self.panes:
            pane.image_view.refresh_layers()
            if pane is self._focused_pane:
                self._update_status_bar()
            self._debounce_timers[pane].stop()
            self._start_render(pane)
        self._sync_orbit_panel()

    # ---- File > Export: PNG (any combination of overlays) or a JSON fact sheet --
    # Same _on_*/_do_* split as Save/Open Experiment above (see that
    # section's own comment): the ExportImageDialog itself (Stage B) and
    # QFileDialog both block on a real event loop, so _on_export_image
    # drives both, then hands off to _do_export_image for the actual
    # (directly testable) save logic.
    def _on_export_image(self) -> None:
        pane = self._focused_pane
        message = pane.image_view.no_effect_parameter_message()
        if message is not None:
            # Same guard _start_render uses for the live view -- no point
            # opening a dialog to preview/export a meaningless uniform
            # field.
            QMessageBox.critical(self, "Export Image failed", message)
            return
        dialog = ExportImageDialog(self.session, pane, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG image (*.png);;All files (*)")
        if path:
            self._do_export_image(path, dialog.resolution(), dialog.overlay_flags())

    def _do_export_image(self, path: str, resolution: int,
                         overlay_flags: dict | None = None) -> None:
        """Renders the FOCUSED pane's own (map, param, viewport center/
        scale, mode, settings) FRESH at `resolution` via compose_export_image
        -- the SAME function ExportImageDialog's own live preview calls, so
        the saved file can never show something different from what was
        previewed -- then saves the result as a plain PNG.

        `overlay_flags` (paint_overlays-shaped kwargs -- see
        ExportImageDialog.overlay_flags) defaults to None/{} -- a clean
        field with no overlays at all, the original (pre-Stage-B) export
        behavior -- so a caller that doesn't care about overlays (a test,
        or any future direct call) still gets exactly that without having
        to spell out every flag as False.
        """
        if not path.lower().endswith(".png"):
            path += ".png"
        pane = self._focused_pane
        message = pane.image_view.no_effect_parameter_message()
        if message is not None:
            # Same guard _start_render uses for the live view (see
            # ImageView.no_effect_parameter_message's own docstring) --
            # exporting would otherwise silently write a meaningless
            # uniform-color PNG with no indication anything was wrong.
            QMessageBox.critical(self, "Export Image failed", message)
            return
        image = compose_export_image(self.session.map, self.session.param, pane.viewport,
                                     pane.render_mode, self.session.settings,
                                     self.session.render_settings, resolution,
                                     **(overlay_flags or {}))
        if not image.save(path, "PNG"):
            QMessageBox.critical(self, "Export Image failed", f"could not write {path!r}")

    def _on_export_facts(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Facts", "", "JSON (*.json);;All files (*)")
        if path:
            self._do_export_facts(path)

    def _do_export_facts(self, path: str) -> None:
        """facts_to_dict(session, mode, viewport) -> JSON, for the FOCUSED
        pane's own render_mode/viewport (provenance only -- the facts
        themselves are of session.map/session.param, shared by every pane;
        see facts_to_dict's own docstring). Reuses FactsPanel's own
        classification/grouping helpers, so this always matches the Facts
        tab exactly, not a second copy of that logic that could drift.
        """
        if not path.lower().endswith(".json"):
            path += ".json"
        pane = self._focused_pane
        data = facts_to_dict(self.session, pane.render_mode, pane.viewport)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        except OSError as exc:
            QMessageBox.critical(self, "Export Facts failed", str(exc))

    # ---- parameter a: field commit and parameter-plane click, kept in sync ------
    def _on_param_field_committed(self, a: complex) -> None:
        # A typed field commit never switches or targets any ONE pane --
        # session.param is shared, so it invalidates every DYNAMICAL pane
        # (see the shared _apply_param_change below) and leaves every
        # parameter-plane pane's own plane/viewport alone, exactly the
        # field-commit behavior Stage 1/2 already had, just no longer
        # limited to the single (then-only) pane.
        self._record_param_history()
        self._apply_param_change(a)

    # ---- undo history of a: Cmd/Ctrl+Z in a plane --------------------------------
    def _record_param_history(self, coalesce: bool = False) -> None:
        """Remembers the CURRENT a before a user change. `coalesce`: a nudge
        within NUDGE_HISTORY_COALESCE_S of the last one joins its step."""
        now = time.monotonic()
        if coalesce and self._last_nudge_time is not None and \
                now - self._last_nudge_time < NUDGE_HISTORY_COALESCE_S:
            self._last_nudge_time = now
            return
        self._param_undo.append(self.session.param)
        del self._param_undo[:-PARAM_HISTORY_LIMIT]
        self._param_redo.clear()
        self._last_nudge_time = now if coalesce else None

    def undo_param(self) -> bool:
        """Back to the previous a; False if there is none."""
        if not self._param_undo:
            return False
        self._param_redo.append(self.session.param)
        self._last_nudge_time = None
        self._apply_param_change(self._param_undo.pop())
        return True

    def redo_param(self) -> bool:
        if not self._param_redo:
            return False
        self._param_undo.append(self.session.param)
        self._last_nudge_time = None
        self._apply_param_change(self._param_redo.pop())
        return True

    def _on_param_changed_by_click(self, a: complex) -> None:
        self._record_param_history()
        self._on_param_changed_by_click_inner(a)

    def _on_param_changed_by_click_inner(self, a: complex) -> None:
        # mousePressEvent already emitted pane_activated before this can
        # ever fire (see _build_ui's own comment), so self._focused_pane
        # is the CLICKED pane.
        clicked_pane = self._focused_pane
        if not self.coupled_view_action.isChecked():
            # SINGLE-VIEW fallback: only one pane is actually visible, so
            # there is no "partner" to drive -- keep today's pre-Stage-3
            # behavior exactly, the same as a P6 parameter-plane click
            # always had: set a AND switch the one visible pane straight
            # to the dynamical (julia) plane.
            self.session.param = a
            self.param_field.set_value(a)
            self.facts_panel.refresh()
            self._refresh_header()
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

    def _apply_param_change(self, a: complex, *, reset_dynamical_viewport: bool = True) -> None:
        """The ONE place session.param is ever assigned from user input --
        the field commit, every parameter-plane click (coupled or
        single-view-already-dynamical), AND arrow-key marker nudging
        (Stage 4, via _on_param_nudged) funnel through here, so they
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

        `reset_dynamical_viewport` (Stage 4): the ONE exception to "every
        dynamical pane's viewport resets to the default framing on every
        param change." A held arrow can call this many times a second;
        resetting the partner pane's viewport on every one of those ticks
        would snap it back to the default framing every tick instead of
        letting the user watch the SAME framing evolve continuously as the
        parameter moves -- exactly backwards from what "continuous
        re-render" / "smooth held-arrow sweeps" are supposed to mean.
        False ONLY from _on_param_nudged below; every other caller keeps
        today's reset-to-default behavior, unchanged.
        """
        self.session.param = a
        self.param_field.set_value(a)
        self.facts_panel.refresh()
        self._refresh_header()
        for pane in self.panes:
            if pane.render_mode in PARAMETER_PLANE_MODES:
                pane.image_view.update()   # move its marker; nothing else about it changed
                continue
            if reset_dynamical_viewport:
                center, scale = default_view_for_mode(self.session.map, self.session.param,
                                                      pane.render_mode)
                vp = pane.viewport
                pane.viewport = cdx.Viewport(center, scale, vp.resolution)
            pane.image_view.refresh_layers()
            pane.image_view.refresh_orbit_for_new_param()
            if pane is self._focused_pane:
                self._update_status_bar()
            self._debounce_timers[pane].stop()
            self._start_render(pane)

    def _on_param_nudged(self, a: complex) -> None:
        """Arrow-key marker nudging (Stage 4, ImageView.param_nudged): the
        SAME underlying invalidation _apply_param_change already gives
        every other param-setting path, just without resetting a dynamical
        partner pane's viewport on every tick -- see
        _apply_param_change's own `reset_dynamical_viewport` doc comment
        for why. Unlike _on_param_changed_by_click, this never switches
        any pane's plane/mode either, coupled or single-view: nudging the
        marker is not a request to stop looking at the parameter plane.
        """
        self._record_param_history(coalesce=True)
        self._apply_param_change(a, reset_dynamical_viewport=False)

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
        self._sync_mode_controls()
        if pane is self._focused_pane:
            self._refresh_header()   # domain/mode text depends on the mode itself
            self._update_status_bar()
        self._sync_orbit_panel()   # this pane may have just become (or stopped being) dynamical
        self._debounce_timers[pane].stop()
        self._start_render(pane)

    # ---- settings: apply on demand, from the Settings dialog (Stage 5) ----------
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
        # Coalesced: Rendering and Coloring applying in the same pause
        # produce ONE re-render per pane, not one each.
        self._settings_render_timer.start()

    def _render_after_settings(self) -> None:
        for pane in self.panes:
            self._debounce_timers[pane].stop()
            self._start_render(pane)

    # ---- map edits: live but debounced, reusing the viewport-change path -------
    def _on_term_edited(self) -> None:
        # Deliberately the SAME debounced path viewport drag/zoom uses, not
        # an immediate render the way Settings' explicit Apply is: an
        # Apply/Add Pole/Add Zero click on the Equation tab is meant to
        # feel live (click Apply, watch the picture update), and several
        # edits in quick succession should coalesce into one render the
        # same way a scroll burst does, not fire one render per click.
        # EquationPanel has already validated and applied the edit to
        # self.session.map by the time this is called; this only has to
        # trigger the render side.
        #
        # session.map is SHARED and UNCONDITIONALLY affects every pane's
        # pixels regardless of mode (unlike a param change, which a
        # parameter-plane pane ignores entirely -- see
        # _apply_param_change's own cache-asymmetry docstring) -- so this
        # debounces BOTH panes, not just the focused one.
        self._update_status_bar()
        self._refresh_header()   # name/formula depend directly on the edited map
        # Cheap: refresh() only recomputes when (map, param) actually
        # changed since its last call, so this stays fresh for whenever the
        # user next looks at the Facts tab without paying for a recompute
        # on every keystroke's worth of edits.
        self.facts_panel.refresh()
        for pane in self.panes:
            self._debounce_timers[pane].start()
            # Same memoization property -- see ImageView.refresh_layers.
            pane.image_view.refresh_layers()
            # An orbit traced under the PRE-edit map no longer describes
            # this map's dynamics -- see OrbitTracker.reset_if_stale's own
            # docstring.
            pane.image_view.refresh_orbit_staleness()

    # ---- facts: recompute lazily, and re-check on becoming visible -----------------
    def _on_sidebar_tab_changed(self, index: int) -> None:
        if self.sidebar.widget(index) is self.facts_panel:
            self.facts_panel.refresh()
        self._refresh_this_map()

    def _placeholder_tab(self, text: str) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        label = QLabel(text, page)
        label.setProperty("role", "empty")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return page

    # ---- workspaces -------------------------------------------------------------------
    def show_workspace(self, key: str) -> None:
        widget = self._workspace_keys.get(key)
        if widget is None:
            return
        self.workspaces.setCurrentWidget(widget)
        self.top_bar.workspace_switch.set_current(key)
        self._on_workspace_changed(key)

    def workspace_widget(self, key: str) -> QWidget | None:
        """The widget a workspace shows (a module's own panel, not its scroll area)."""
        page = self._workspace_keys.get(key)
        return page.widget() if isinstance(page, QScrollArea) else page

    def current_workspace(self) -> str:
        current = self.workspaces.currentWidget()
        return next(k for k, w in self._workspace_keys.items() if w is current)

    def _on_workspace_changed(self, key: str) -> None:
        """Explore-only controls hide while another workspace shows: Coupled |
        Single, the overlay chips, and the plane-specific dock tabs (the
        Equation tab stays: a module may use the map)."""
        explore = key == "explore"
        self.top_bar.view_switch.setVisible(explore)
        plane_tabs = (self.rendering_tab, self.coloring_tab, self.orbit_tab)
        if not explore and self.settings_dock.current_widget() in plane_tabs:
            self.settings_dock.set_current_widget(self.equation_tab)
        for widget in plane_tabs:
            self.settings_dock.set_tab_visible(self.settings_dock.index_of(widget), explore)
        self._sync_mode_controls()

    # ---- extra panels (research modules) ----------------------------------------
    def add_panel(self, widget: QWidget, title: str, placement: str | None = None) -> int:
        """Adds a research module's panel and returns its index within the
        place it went:

          "workspace"  a page of the central stack, next to Explore; the
                       top bar's Explore | ... switch appears
          "sidebar"    a sidebar tab, after the built-in ones (the default)
          "dock"       a tab of the Equations & Settings dock

        `placement` defaults to the widget's own `placement` attribute, then
        "sidebar". This is how a module's UI (modules/<name>/panel.py)
        appears WITHOUT app/ importing anything from modules/: the
        composition root (phoenix.py) builds the panel and hands it over.
        If the widget has a shutdown() method it is called from closeEvent,
        so a panel with background work of its own can cancel and drain it.
        """
        placement = placement or getattr(widget, "placement", None) or "sidebar"
        if placement not in ("workspace", "sidebar", "dock"):
            raise ValueError(f"unknown placement {placement!r}")
        self._extra_panels.append(widget)
        if placement == "workspace":
            key = title.lower()
            # In a scroll area: a module's own minimum size must not squeeze
            # the docks (a QStackedWidget is as tall as its tallest page).
            page = QScrollArea(self)
            page.setWidgetResizable(True)
            page.setFrameShape(QFrame.Shape.NoFrame)
            page.setWidget(widget)
            self._workspace_keys[key] = page
            self._workspace_titles[key] = title
            index = self.workspaces.addWidget(page)
            switch = self.top_bar.workspace_switch
            switch.set_segments([[Segment(k, self._workspace_titles[k])
                                  for k in self._workspace_keys]])
            switch.set_current(self.current_workspace())
            switch.show()
        elif placement == "sidebar":
            index = self.sidebar.add_tab(widget, title)
        else:
            index = self.settings_dock.add_tab(widget, title)
        # A panel documents its own controls by carrying a `controls`
        # sequence of app.controls.ControlEntry (see modules/approximation).
        self.info_panel.add_entries(getattr(widget, "controls", ()))
        self._rebuild_tab_actions()
        return index

    def go_to_destinations(self) -> list[tuple[str, Callable[[], None]]]:
        """What Ctrl+1..9 jump to, in order: the sidebar's tabs, then the
        dock's tabs, then (when a module registered one) the next workspace."""
        out: list[tuple[str, Callable[[], None]]] = []
        for dock in (self.sidebar, self.settings_dock):
            for i in range(dock.count()):
                widget = dock.widget(i)
                out.append((dock.tab_text(i), lambda d=dock, w=widget: d.reveal(w)))
        if len(self._workspace_keys) > 1:
            out.append(("Switch Workspace", self.cycle_workspace))
        return out

    def cycle_workspace(self) -> None:
        keys = list(self._workspace_keys)
        self.show_workspace(keys[(keys.index(self.current_workspace()) + 1) % len(keys)])

    def _rebuild_tab_actions(self) -> None:
        self.go_to_tab_menu.clear()
        self.tab_actions = []
        first, last = controls.entry("app.go_to_tab").binding.sequences   # "Ctrl+1", "Ctrl+9"
        prefix, start, stop = first[:-1], int(first[-1]), int(last[-1])
        for index, (title, go) in enumerate(self.go_to_destinations()[:stop - start + 1]):
            action = self.go_to_tab_menu.addAction(title)
            action.setShortcut(QKeySequence(f"{prefix}{start + index}"))
            action.triggered.connect(lambda _checked=False, g=go: g())
            self.tab_actions.append(action)

    # ---- Edit > Undo/Redo ---------------------------------------------------------
    def _route_undo(self, redo: bool, focus: QWidget | None = None) -> None:
        """Undo/Redo: a focused, editable text widget's own text undo wins;
        otherwise the Equation panel's map-edit stack. `focus` is injectable
        for tests; it defaults to the application's focus widget."""
        widget = QApplication.focusWidget() if focus is None else focus
        target = text_undo_target(widget)
        if target is not None:
            target.redo() if redo else target.undo()
        elif isinstance(widget, ImageView) and (self.redo_param() if redo else self.undo_param()):
            pass   # in a plane: the last change to a (when there is one)
        elif redo:
            self.equation_panel.redo()
        else:
            self.equation_panel.undo()

    # ---- facts tab: clicking a listed point centres the view on it --------------
    def _on_center_view(self, point: complex) -> None:
        pane = self._focused_pane
        vp = pane.viewport
        pane.viewport = cdx.Viewport(point, vp.scale, vp.resolution)
        self.show_workspace("explore")
        self._update_status_bar()
        self._debounce_timers[pane].stop()
        self._start_render(pane)

    # ---- facts tab: clicking a fixed/critical point seeds an orbit there (Stage 5) --
    def _on_seed_orbit_from_facts(self, point: complex) -> None:
        # NOT self._focused_pane -- orbit tracking is a dynamical-plane
        # concept, and the focused pane could easily be a parameter-plane
        # one (e.g. the pane the user just clicked to set `a`). Routes
        # through the SAME "which pane counts as THE dynamical one right
        # now" helper the orbit strip itself follows (_current_dynamical_
        # pane -- see _sync_orbit_panel) and Save Experiment's own orbit
        # provenance, rather than a second, independently-drifting notion
        # of it: prefers the FOCUSED pane if it qualifies, else the first
        # dynamical pane found among self.panes (not restricted to
        # whichever pane is currently VISIBLE in single-view mode -- same
        # tradeoff those two other call sites already make).
        dynamical_pane = self._current_dynamical_pane()
        if dynamical_pane is None:
            return   # no pane is dynamical (both parameter-plane) -- orbit seeding is a no-op
        dynamical_pane.image_view.seed_orbit(point)
        self.show_workspace("explore")

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
        self.equation_panel.refresh_from_session()
        self.facts_panel.refresh()
        self._refresh_header()   # name/formula/param all just changed wholesale
        self.flash_status(f"Loaded {self.session.map.name}")
        # Back to the planes: loading is done from the sidebar, exploring isn't.
        self.show_workspace("explore")
        self._focused_pane.image_view.setFocus(Qt.FocusReason.OtherFocusReason)
        for pane in self.panes:
            center, scale = default_view_for_mode(self.session.map, self.session.param,
                                                  pane.render_mode)
            pane.viewport = cdx.Viewport(center, scale, pane.viewport.resolution)
            pane.image_view.refresh_layers()
            pane.image_view.refresh_orbit_staleness()
            if pane is self._focused_pane:
                self._update_status_bar()
            self._debounce_timers[pane].stop()
            self._start_render(pane)

    # ---- library tab: any successful save/rename/delete/notes edit persists -----
    def _on_library_changed(self) -> None:
        self.session.save_user_library(library_path())
        _regenerate_library_previews(self.session.library, self.session.settings,
                                     self.session.render_settings)
        self.library_panel.refresh_previews()

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
                          cdx.CancelToken(), self.session.cache,
                          self.session.settings.greens_potential,
                          self.session.settings.parameter_strategy,
                          self.session.settings.parameter_critical_index)
        # Bound methods, not lambdas: PySide resolves `self` (this window,
        # a QObject on the GUI thread) as the receiver and correctly
        # queues delivery across threads -- see RenderSignals' docstring.
        task.signals.partial_ready.connect(self._on_partial_ready)
        task.signals.full_ready.connect(self._on_full_ready)
        task.signals.failed.connect(self._on_render_failed)
        pane.pending_tasks[request_id] = task
        self._thread_pool.start(_Runnable(task))
        self._set_render_stage(pane, "preview")

    def _on_partial_ready(self, request_id: int, payload, buffer_viewport: cdx.Viewport) -> None:
        pane = self._pane_for_render_id(request_id)
        if pane is None:
            return   # unknown/already-cleaned-up request -- nothing to do
        if request_id != pane.request_id:
            return   # superseded by a newer request for THIS pane; discard
        if not shiboken6.isValid(pane.image_view):
            return   # this pane's widget is already torn down (e.g. window closing)
        pane.image_view.set_image(payload, buffer_viewport)
        self._set_render_stage(pane, "full")
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
        self._set_render_stage(pane, None)
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
        self._set_render_stage(pane, None)
        self.statusBar().showMessage(f"render failed: {message}")

    # ---- top bar: map chip and render status ----------------------------------------
    def _refresh_header(self) -> None:
        self.metadata_header.refresh()
        self._refresh_map_chip()
        self._this_map_dirty = True
        self._refresh_this_map()

    def _preset_thumbnail(self, name: str, entry: cdx.RationalMap) -> QIcon:
        """A preset's real thumbnail (render_thumbnail), rendered once per
        (formula, coloring) and cached for the process -- presets never change."""
        st = self.session.settings
        key = (name, entry.to_formula(), st.color_palette, st.color_scaling, st.color_period)
        icon = _PRESET_THUMBNAIL_CACHE.get(key)
        if icon is None:
            image = render_thumbnail(entry, 0j, st, self.session.render_settings)
            icon = QIcon(QPixmap.fromImage(image))
            _PRESET_THUMBNAIL_CACHE[key] = icon
        return icon

    def _refresh_this_map(self) -> None:
        """Info's "This map" section -- computed only while Info is showing
        (it needs the dynamical facts), otherwise on the next reveal."""
        if not getattr(self, "_this_map_dirty", True) or not hasattr(self, "sidebar"):
            return
        if self.sidebar.current_widget() is not self.info_panel or self.sidebar.is_collapsed():
            return
        self._this_map_dirty = False
        self.info_panel.set_this_map(self.this_map_rows())

    def this_map_rows(self) -> list[tuple[str, str]]:
        m, a = self.session.map, self.session.param
        facts = self.session.dynamical_facts()
        rows = [("Map", f"{self.map_display_name()}: {m.to_formula() or '(empty map)'}"),
                ("Degree", str(facts.degree)),
                ("Critical points", ", ".join(format_short(c, 4) for c in m.distinct_critical_points(a))
                 or "none")]
        rs = self.session.render_settings
        for c, fate in critical_orbit_fates(m, a, facts, rs.max_iter, max(rs.tol, 1e-6)):
            rows.append((f"Orbit of {format_short(c, 4)}", fate))
        rows.append(("a", format_readout_complex(a)))
        return rows

    def flash_status(self, text: str, ms: int = STATUS_FLASH_MS) -> None:
        """A transient status-bar message ("Loaded newton3") shown ahead of
        the scale/cursor readout, which it never replaces."""
        self._status_flash = text
        self._status_flash_timer.start(ms)
        self._render_status_bar()

    def _clear_status_flash(self) -> None:
        self._status_flash = ""
        self._render_status_bar()

    def map_display_name(self) -> str:
        """The family name while the map is still exactly a library entry,
        "custom" once its formula has been edited away from it."""
        m = self.session.map
        entry = self.session.library.find(m.name) if m.name else None
        if entry is not None and entry.to_formula() == m.to_formula():
            return m.name
        return "custom"

    def _refresh_map_chip(self) -> None:
        if not hasattr(self, "top_bar"):
            return
        chip = self.top_bar.map_chip
        display = self.map_display_name()
        chip.set_map(display, self.session.map.to_formula())
        loaded = None if display == "custom" else display
        if loaded != self.library_panel.loaded_name():
            self.library_panel.set_loaded(loaded)   # the dot (and selection) follow the map
        chip.setToolTip(f"{self.session.map.name}: {self.session.map.to_formula()}\n"
                        f"a: {describe_parameter_role(self.session.map)}\n"
                        "Click to open the Library")

    def _set_render_stage(self, pane: Pane, stage: str | None) -> None:
        if not hasattr(self, "_render_stage"):
            return
        self._render_stage[pane] = stage
        self._refresh_render_status()

    def render_status(self) -> tuple[str, bool]:
        """(text, busy) for the top bar: which plane(s) are rendering and at
        which stage (preview -> full). There are no percentages to show: cdx's
        render calls report no fractional progress."""
        busy = [(p, st) for p, st in self._render_stage.items()
                if st is not None and (self.coupled_view_action.isChecked() or p is self._focused_pane)]
        if not busy:
            return "Ready", False
        names = {("parameter plane" if p.render_mode in PARAMETER_PLANE_MODES else "dynamical plane")
                 for p, _ in busy}
        what = names.pop() if len(names) == 1 else "both planes"
        stage = "preview" if any(st == "preview" for _, st in busy) else "full"
        return f"Rendering {what} · {stage}", True

    def _refresh_render_status(self) -> None:
        if hasattr(self, "top_bar"):
            self.top_bar.set_render_status(*self.render_status())

    # ---- window layout: docks, collapse, save/restore ---------------------------------
    def _apply_default_layout(self) -> None:
        self.sidebar.show()
        self.settings_dock.show()
        self.sidebar.set_collapsed(False)
        self.settings_dock.set_collapsed(False)
        self._apply_default_dock_sizes()
        # Dock sizes only stick once the window has been laid out; a window
        # not shown yet gets them again from its first showEvent.
        self._default_sizes_pending = not self.isVisible()

    def _apply_default_dock_sizes(self) -> None:
        self.resizeDocks([self.sidebar], [SIDEBAR_DEFAULT_WIDTH], Qt.Orientation.Horizontal)
        self.resizeDocks([self.settings_dock], [SETTINGS_DOCK_DEFAULT_HEIGHT], Qt.Orientation.Vertical)
        half = max(1, self.view_splitter.width() // 2)
        self.view_splitter.setSizes([half, half])

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(self, "_default_sizes_pending", False):
            self._default_sizes_pending = False
            self._apply_default_dock_sizes()

    def reset_layout(self) -> None:
        """View > Reset Layout: docks back in place, open, expanded, default sizes."""
        self.restoreState(self._default_layout_state, LAYOUT_STATE_VERSION)
        self._sidebar_auto_collapsed = False
        self._was_narrow = False
        self._apply_default_layout()
        self._auto_collapse_sidebar()

    def save_layout(self, settings: QSettings) -> None:
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState(LAYOUT_STATE_VERSION))
        settings.setValue("sidebar/collapsed", self.sidebar.is_collapsed()
                          and not self._sidebar_auto_collapsed)
        settings.setValue("sidebar/tab", self.sidebar.tab_text(self.sidebar.current_index()))
        settings.setValue("dock/collapsed", self.settings_dock.is_collapsed())
        settings.setValue("dock/tab", self.settings_dock.tab_text(self.settings_dock.current_index()))
        settings.setValue("splitter", self.view_splitter.saveState())
        settings.sync()

    def restore_layout(self, settings: QSettings) -> None:
        geometry = settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = settings.value("window/state")
        if state is not None and self.restoreState(state, LAYOUT_STATE_VERSION):
            self._default_sizes_pending = False   # the saved sizes win
        splitter = settings.value("splitter")
        if splitter is not None:
            self.view_splitter.restoreState(splitter)
        for dock, key in ((self.sidebar, "sidebar"), (self.settings_dock, "dock")):
            tab = settings.value(f"{key}/tab")
            for i in range(dock.count()):
                if dock.tab_text(i) == tab:
                    dock.set_current_index(i)
            dock.set_collapsed(str(settings.value(f"{key}/collapsed", "false")).lower() == "true")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._auto_collapse_sidebar()

    def _auto_collapse_sidebar(self) -> None:
        """Narrow window: the sidebar becomes its tab strip so both planes
        keep their room; widening again restores it (only if it was this,
        not the user, that collapsed it)."""
        if not hasattr(self, "sidebar"):
            return
        narrow = self.width() < SIDEBAR_AUTO_COLLAPSE_WIDTH
        if narrow == self._was_narrow:
            return   # only crossing the threshold acts; a hand-made choice in between stands
        self._was_narrow = narrow
        if narrow and not self.sidebar.is_collapsed():
            self.sidebar.set_collapsed(True)
            self._sidebar_auto_collapsed = True
        elif not narrow and self._sidebar_auto_collapsed:
            self._sidebar_auto_collapsed = False
            self.sidebar.set_collapsed(False)

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

    def _on_pane_hover(self, pane: Pane, inside: bool) -> None:
        if inside:
            self.pane_hint_label.setText(
                pane_hint(pane.render_mode, coupled=self.coupled_view_action.isChecked()))
            self._pane_hint_timer.start()
        else:
            self.pane_hint_label.clear()
            self._pane_hint_timer.stop()

    def _on_cursor_readout_changed(self, text: str) -> None:
        self._status_cursor_text = text
        self._render_status_bar()

    def _render_status_bar(self) -> None:
        text = f"{self._status_base_message}   {self._status_cursor_text}" \
            if self._status_cursor_text else self._status_base_message
        flash = getattr(self, "_status_flash", "")
        self.statusBar().showMessage(f"{flash}   ·   {text}" if flash else text)

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
        for panel in self._extra_panels:
            shutdown = getattr(panel, "shutdown", None)
            if callable(shutdown):
                shutdown()
        if self._layout_settings is not None:
            self.save_layout(self._layout_settings)
        super().closeEvent(event)


def main(extra_panels=()) -> None:
    """Runs the app. `extra_panels` is a sequence of factories, each called with
    the window and returning (widget, tab title) -- how the composition root
    (phoenix.py) attaches research-module panels; app/ itself imports none."""
    app = QApplication(sys.argv)
    theme.apply(app)
    window = SandboxWindow(layout_settings=default_layout_settings())
    for factory in extra_panels:
        widget, title = factory(window)
        window.add_panel(widget, title)   # placement: the widget's own `placement`, else sidebar
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
