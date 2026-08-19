"""app/library_panel.py -- the Library tab: browse/load/save/rename/delete
saved RationalMap families, plus their notes, backed by FamilyLibrary's own
serialize round-trip.

READ-ONLY PRESETS. The six built-in families (app.session.PRESET_FAMILY_NAMES)
can be loaded and viewed but never saved-over, renamed, deleted, or have
their notes edited. session.save_to_library/rename_in_library/
delete_from_library/set_library_notes all enforce this themselves (raising
ValueError) -- the disabled buttons and read-only notes editor here are UI
convenience layered on top of a real invariant enforced one level down, the
same _try_mutate-adjacent shape app.term_editor_panel uses for its own
validation.

MODAL DIALOGS AREN'T DIRECTLY TESTABLE (QInputDialog.getText blocks on a
real event loop), so every button handler that pops one is split: the
button's own slot gathers the text via the dialog, then hands off to a
`_do_*` method containing the actual mutation logic. Tests call the `_do_*`
methods directly, the same way app/test_term_editor_panel.py drives
TermEditorPanel's mutating methods directly rather than through QPushButton
clicks.

PERSISTENCE. This panel never WRITES to disk itself -- every successful
mutation (save/rename/delete/notes edit) calls on_change(), and every
successful load calls on_load(); SandboxWindow wires the former to
session.save_user_library(app.settings.library_path()) (and, since Stage C,
regenerating each entry's sidecar preview thumbnail) and the latter to a
viewport reset + immediate render. Keeping WRITE I/O out of this class is
what keeps it constructible standalone in tests without ever touching a
real ~/.complexdynamics/library.txt (see app/test_settings_panel.py's own
note on the identical concern for Settings). It does READ one thing from
disk directly, though: _icon_for checks whether a given entry's sidecar
preview (app.settings.preview_path_for) exists, to show it as a list icon
-- see app/test_library_panel.py's own note on redirecting config_dir() for
the whole test module so this stays off the real filesystem there too.

LOADING resets each visible pane's VIEWPORT to a per-family, MODE-aware
default (default_view_for_mode below) -- hand-picked a-space framings
(_DEFAULT_VIEWS, via default_view_for) for the six built-in shapes' most
legible parameter-plane region on a parameter-plane pane, a FIXED center-0
z-space framing (default_dynamical_view) on a dynamical-plane one --
using the PARAMETER-plane table for a dynamical pane's reset was a real
bug (Stage 2): it jumped every Julia-set pane onto whatever window makes
the parameter plane legible, not the Julia set. default_dynamical_view's
own per-instance (critical-/fixed-point-derived) framing is DEFERRED for
now -- see its own docstring and _derive_dynamical_view_from_facts, kept
but not called. That reset itself now happens in on_load's own handler
(app/sandbox.py's
_on_family_loaded), not here: this panel no longer touches any pane's
viewport directly (see app.pane.Pane -- Session itself has no single
viewport to reset anymore). Loading a family does not touch render_mode or
the parameter either; loading has no informed opinion about either one.
"""

from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

import cdx
from app.session import PARAMETER_PLANE_MODES, PRESET_FAMILY_NAMES
from app.settings import preview_path_for

# Matches self._list.setIconSize below -- a saved family's sidecar preview
# (see app.sandbox.THUMBNAIL_RESOLUTION/preview_path_for) is rendered
# larger than this and scaled down, the same "render bigger, display
# smaller" every other thumbnail in this app already does.
_LIST_ICON_SIZE = 32

# Hand-picked (center, scale) for each built-in preset's most legible
# region -- not derived from any formula, just where each shape's
# interesting structure actually sits. A name not listed here (a
# user-saved family, or a future preset this table hasn't caught up
# with) falls back to _FALLBACK_VIEW.
_DEFAULT_VIEWS: dict[str, tuple[complex, float]] = {
    "mandelbrot": (complex(-0.5, 0.0), 1.5),
    "multibrot3": (complex(0.0, 0.0), 1.5),
    "multibrot5": (complex(0.0, 0.0), 1.3),
    "mcmullen2": (complex(0.0, 0.0), 2.5),
    "mcmullen3": (complex(0.0, 0.0), 2.5),
    "newton3": (complex(0.0, 0.0), 2.0),
}
_FALLBACK_VIEW: tuple[complex, float] = (complex(0.0, 0.0), 2.0)

_NAME_ROLE = Qt.ItemDataRole.UserRole


def default_view_for(name: str) -> tuple[complex, float]:
    return _DEFAULT_VIEWS.get(name, _FALLBACK_VIEW)


# KNOWN OPEN PROBLEM, not addressed here: _DEFAULT_VIEWS is a hand-picked
# table for exactly six built-in shapes, and _FALLBACK_VIEW is a single
# generic guess for everything else (any user-saved family). There is no
# per-instance parameter-plane framing at all -- a custom map's a-space
# window is just whatever the generic fallback happens to be, legible or
# not. Left exactly as it is; only the DYNAMICAL side changes below.


# The dynamical (Julia-set) plane's OWN default framing has nothing to do
# with the table above: _DEFAULT_VIEWS/default_view_for frames a-SPACE (the
# parameter plane), and using it for z-space too was a real bug -- resetting
# or loading a family while looking at a Julia set jumped to whatever window
# happens to make the PARAMETER plane legible (e.g. mandelbrot's own
# (-0.5, 1.5)), not the dynamical plane's. See default_view_for_mode below
# for the actual per-pane routing.
_DYNAMICAL_VIEW_FALLBACK: tuple[complex, float] = (complex(0.0, 0.0), 2.0)
# Points farther out than this are treated as "may as well be infinity" for
# framing purposes -- a legitimate critical/fixed point this far from the
# origin would zoom the default view out past anywhere the actual filled
# Julia set structure is likely to be legible, which defeats the point of a
# *default* framing (a numerically-finite but dynamically-irrelevant outlier
# should not dominate the box).
_DYNAMICAL_VIEW_MAGNITUDE_CAP = 50.0
_DYNAMICAL_VIEW_PADDING = 1.5   # multiplies the tight half-width so points sit inside the frame, not on its edge
_DYNAMICAL_VIEW_MIN_SCALE = 0.5   # a degenerate (single-point, or nearly so) box still gets a usable window


def default_dynamical_view(rational_map: cdx.RationalMap, param: complex) -> tuple[complex, float]:
    """The dynamical plane's own default framing -- INTERIM: always
    _DYNAMICAL_VIEW_FALLBACK (center 0, scale 2), regardless of map/param.

    This used to derive a padded bounding box around the map's own finite
    critical/fixed points (see _derive_dynamical_view_from_facts below,
    PRESERVED but no longer called from here) -- reverted to a fixed
    framing per direction: per-instance tailoring needs a proper
    algorithmic solution, not that heuristic, and this also drops a real
    root-find (cdx.dynamical_facts) off every reset/family-load/param-
    change in the meantime. Swap the body back to
    `return _derive_dynamical_view_from_facts(rational_map, param)` once
    that solution exists -- `rational_map`/`param` are kept as this
    function's own parameters (unused for now) specifically so every
    existing call site (default_view_for_mode, and everything in
    app/sandbox.py that calls THAT) needs no changes when it does.
    """
    del rational_map, param   # interim: genuinely unused -- see docstring
    return _DYNAMICAL_VIEW_FALLBACK


# ---- DEFERRED: future per-instance dynamical framing -- NOT WIRED IN -----------
# default_dynamical_view above used to call this directly. Preserved,
# callable, tested (see app/test_library_panel.py) -- just not reachable
# from the real per-pane reset path right now (see that function's own
# docstring for why). Do not delete: this is where a real algorithmic
# solution should resume from, not a start-from-scratch redesign.
def _derive_dynamical_view_from_facts(rational_map: cdx.RationalMap,
                                      param: complex) -> tuple[complex, float]:
    """A padded bounding box around the map's finite critical and fixed
    points at THIS parameter (Fatou: every attracting cycle attracts a
    critical point, so critical points are where the interesting structure
    clusters; fixed points mark where a filled Julia set's boundary
    characteristically threads through), excluding poles/infinity/
    absurdly-far-out points (see _DYNAMICAL_VIEW_MAGNITUDE_CAP) and falling
    back to _DYNAMICAL_VIEW_FALLBACK when nothing usable survives that
    filter (a degenerate map, or one whose only critical/fixed points are
    excluded).

    Genuinely depends on `param`, not just the map's NAME -- unlike
    default_view_for above, this can't be a name-keyed table: which points
    are critical/fixed (and therefore where the box goes) is a real function
    of the currently-bound parameter for families where `a` affects a pole's
    location or a term's exponent-bearing coefficient.
    """
    facts = cdx.dynamical_facts(rational_map, param)
    points = list(facts.critical_points) + [fp.point for fp in facts.fixed_points]
    return _bounding_view(points)


def _bounding_view(points: list[complex]) -> tuple[complex, float]:
    """The actual padded-bounding-box arithmetic
    _derive_dynamical_view_from_facts uses, factored out as a pure function
    of a plain point list so it's directly testable without needing a real
    cdx.RationalMap to coax into producing a specific (in particular, a
    genuinely DEGENERATE/all-excluded) set of critical/fixed points -- the
    same reason app.sandbox._orbit_line_segments/drawable_polyline_segments
    exist as their own testable functions rather than being buried inside a
    paint method. Not currently called from the live reset path (see
    _derive_dynamical_view_from_facts' own docstring) -- preserved for the
    same reason.
    """
    finite_points = [p for p in points
                     if math.isfinite(p.real) and math.isfinite(p.imag)
                     and abs(p) < _DYNAMICAL_VIEW_MAGNITUDE_CAP]
    if not finite_points:
        return _DYNAMICAL_VIEW_FALLBACK

    min_re = min(p.real for p in finite_points)
    max_re = max(p.real for p in finite_points)
    min_im = min(p.imag for p in finite_points)
    max_im = max(p.imag for p in finite_points)
    center = complex((min_re + max_re) / 2.0, (min_im + max_im) / 2.0)
    half_width = max(max_re - min_re, max_im - min_im) / 2.0
    scale = max(half_width * _DYNAMICAL_VIEW_PADDING, _DYNAMICAL_VIEW_MIN_SCALE)
    return center, scale


def default_view_for_mode(rational_map: cdx.RationalMap, param: complex,
                          render_mode: str) -> tuple[complex, float]:
    """The single entry point every pane-viewport reset should go through
    (Stage 2): a-space framing (default_view_for, keyed on the map's own
    NAME) for a PARAMETER_PLANE_MODE, z-space framing
    (default_dynamical_view, which needs the actual map+param) for a
    dynamical one -- routes on render_mode so call sites never have to
    remember which table applies to which plane themselves.
    """
    if render_mode in PARAMETER_PLANE_MODES:
        return default_view_for(rational_map.name)
    return default_dynamical_view(rational_map, param)


# A flat mid-grey square -- built once, lazily, and reused for every entry
# without its own sidecar (every preset, always, since _regenerate_library_
# previews in app/sandbox.py only ever writes one for a non-preset entry;
# also any brand-new user entry from before its first _on_library_changed
# has run). Module-level rather than per-QApplication-instance state
# because QPixmap itself can't be constructed before a QApplication exists
# -- built lazily on first use, not at import time, for the same reason
# every other QPixmap/QIcon in this codebase is.
_placeholder_icon_cache: QIcon | None = None


def _placeholder_icon() -> QIcon:
    global _placeholder_icon_cache
    if _placeholder_icon_cache is None:
        pixmap = QPixmap(_LIST_ICON_SIZE, _LIST_ICON_SIZE)
        pixmap.fill(Qt.GlobalColor.lightGray)
        _placeholder_icon_cache = QIcon(pixmap)
    return _placeholder_icon_cache


def _icon_for(name: str) -> QIcon:
    """A saved family's own sidecar thumbnail (see
    app.sandbox._regenerate_library_previews/app.settings.preview_path_for)
    if one has been written for `name`, else a plain placeholder -- never
    an empty/missing icon, so every row in the list looks consistent.
    """
    path = preview_path_for(name)
    if path.exists():
        return QIcon(str(path))
    return _placeholder_icon()


class LibraryPanel(QWidget):
    def __init__(self, session, on_load: Callable[[], None], on_change: Callable[[], None],
                parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self._on_load = on_load
        self._on_change = on_change

        self._build_ui()
        self._refresh_list()

    # ---- construction ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        split = QHBoxLayout()
        self._list = QListWidget()
        self._list.setIconSize(QSize(_LIST_ICON_SIZE, _LIST_ICON_SIZE))
        self._list.currentItemChanged.connect(lambda *_: self._refresh_detail())
        split.addWidget(self._list, 1)

        detail = QVBoxLayout()
        self._formula_label = QLabel()
        self._formula_label.setWordWrap(True)
        self._formula_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        detail.addWidget(self._formula_label)
        detail.addWidget(QLabel("Notes:"))
        self._notes_edit = QPlainTextEdit()
        self._notes_edit.textChanged.connect(self._on_notes_edited)
        detail.addWidget(self._notes_edit)
        split.addLayout(detail, 1)
        layout.addLayout(split)

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #cc4444;")
        layout.addWidget(self._error_label)

        row = QHBoxLayout()
        self._load_button = QPushButton("Load")
        self._load_button.clicked.connect(self._load_selected)
        self._save_button = QPushButton("Save Current As...")
        self._save_button.clicked.connect(self._save_current_as)
        self._rename_button = QPushButton("Rename...")
        self._rename_button.clicked.connect(self._rename_selected)
        self._delete_button = QPushButton("Delete")
        self._delete_button.clicked.connect(self._delete_selected)
        for b in (self._load_button, self._save_button, self._rename_button,
                 self._delete_button):
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

    # ---- refresh: rebuild the list (and detail pane) from session.library --------------
    def _refresh_list(self, select: str | None = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for entry_name in self.session.library.names():
            label = f"{entry_name}  (preset)" if entry_name in PRESET_FAMILY_NAMES else entry_name
            item = QListWidgetItem(_icon_for(entry_name), label)
            item.setData(_NAME_ROLE, entry_name)
            self._list.addItem(item)
        self._list.blockSignals(False)
        target = select if select is not None else self.session.map.name
        self._select_by_name(target)
        self._refresh_detail()

    def refresh_previews(self) -> None:
        """Re-reads each item's icon from disk without touching selection or
        the detail pane -- called by SandboxWindow._on_library_changed after
        it (re)writes the sidecar thumbnails themselves (this panel never
        touches disk directly -- see this module's own docstring), so an
        icon that just got (re)generated actually shows up without a full
        _refresh_list rebuild (which would also reset the current
        selection).
        """
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setIcon(_icon_for(item.data(_NAME_ROLE)))

    def _select_by_name(self, name: str) -> None:
        for row in range(self._list.count()):
            if self._list.item(row).data(_NAME_ROLE) == name:
                self._list.setCurrentRow(row)
                return
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _selected_name(self) -> str | None:
        item = self._list.currentItem()
        return item.data(_NAME_ROLE) if item is not None else None

    def _refresh_detail(self) -> None:
        name = self._selected_name()
        self._notes_edit.blockSignals(True)
        if name is None:
            self._formula_label.setText("")
            self._notes_edit.setPlainText("")
            self._notes_edit.setReadOnly(True)
        else:
            entry = self.session.library.find(name)
            self._formula_label.setText((entry.to_formula() if entry is not None else "")
                                        or "(empty map)")
            self._notes_edit.setPlainText(entry.notes if entry is not None else "")
            self._notes_edit.setReadOnly(name in PRESET_FAMILY_NAMES)
        self._notes_edit.blockSignals(False)

    # ---- notes: live edit, same "apply on every edit" spirit as the term editor --------
    def _on_notes_edited(self) -> None:
        name = self._selected_name()
        if name is None or name in PRESET_FAMILY_NAMES:
            return   # the box is read-only for a preset; this can't actually fire for one
        try:
            self.session.set_library_notes(name, self._notes_edit.toPlainText())
        except (KeyError, ValueError) as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        self._on_change()

    # ---- load: replaces session.map; the CALLER resets its own pane's viewport --------
    def _load_selected(self) -> None:
        # Viewport reset to this family's default used to happen HERE, but
        # session no longer owns a viewport (see app.pane.Pane) -- on_load
        # (app/sandbox.py's _on_family_loaded) does it now, for whichever
        # pane(s) it owns, via default_view_for_mode(self.session.map, ...)
        # -- MODE-aware per pane (Stage 2), not just name-keyed.
        name = self._selected_name()
        if name is None:
            return
        try:
            self.session.load_from_library(name)
        except KeyError as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        self._on_load()

    # ---- save current map as a new (or updated) library entry --------------------------
    def _save_current_as(self) -> None:
        text, ok = QInputDialog.getText(self, "Save Current As", "Family name:",
                                        text=self.session.map.name)
        if not ok or not text.strip():
            return
        self._do_save_as(text.strip())

    def _do_save_as(self, name: str) -> None:
        try:
            self.session.save_to_library(name)
        except ValueError as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        self._refresh_list(select=name)
        self._on_change()

    # ---- rename a library entry ---------------------------------------------------------
    def _rename_selected(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        text, ok = QInputDialog.getText(self, "Rename", "New name:", text=name)
        if not ok or not text.strip() or text.strip() == name:
            return
        self._do_rename(name, text.strip())

    def _do_rename(self, old_name: str, new_name: str) -> None:
        try:
            self.session.rename_in_library(old_name, new_name)
        except (KeyError, ValueError) as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        self._refresh_list(select=new_name)
        self._on_change()

    # ---- delete a library entry ----------------------------------------------------------
    def _delete_selected(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        self._do_delete(name)

    def _do_delete(self, name: str) -> None:
        try:
            self.session.delete_from_library(name)
        except (KeyError, ValueError) as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        self._refresh_list()
        self._on_change()
