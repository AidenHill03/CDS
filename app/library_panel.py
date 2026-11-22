"""app/library_panel.py -- the Library tab: browse/load/save/rename/delete
saved RationalMap families, plus their notes, backed by FamilyLibrary's own
serialize round-trip.

READ-ONLY PRESETS. The six built-in families (app.session.PRESET_FAMILY_NAMES)
can be loaded and viewed but never saved-over, renamed, deleted, or have
their notes edited. session.save_to_library/rename_in_library/
delete_from_library/set_library_notes all enforce this themselves (raising
ValueError) -- the disabled buttons and read-only notes editor here are UI
convenience layered on top of a real invariant enforced one level down, the
same _try_mutate-adjacent shape app.equation_panel uses for its own
validation.

MODAL DIALOGS AREN'T DIRECTLY TESTABLE (QInputDialog.getText blocks on a
real event loop), so every button handler that pops one is split: the
button's own slot gathers the text via the dialog, then hands off to a
`_do_*` method containing the actual mutation logic. Tests call the `_do_*`
methods directly, the same way app/test_equation_panel.py drives
EquationPanel's mutating methods directly rather than through QPushButton
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
default (default_view_for_mode below) -- the family's own STORED parameter-
plane framing (app.family_views: a table for the presets, the parameter
pane's viewport at "Save current as..." for a saved family, else
DEFAULT_PARAMETER_VIEW_CENTER/SCALE) on a parameter-plane pane, a FIXED center-0
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

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit,
                               QPushButton, QToolButton, QVBoxLayout, QWidget)

import cdx
from app import theme
from app.family_views import parameter_view_for
from app.session import PARAMETER_PLANE_MODES, PRESET_FAMILY_NAMES
from app.settings import preview_path_for
from app.mathdisplay import DisplayParseError, FormulaView, parse_display, plain_text
from app.widgets import more_icon

# Matches self._list.setIconSize below -- a saved family's sidecar preview
# (see app.sandbox.THUMBNAIL_RESOLUTION/preview_path_for) is rendered
# larger than this and scaled down, the same "render bigger, display
# smaller" every other thumbnail in this app already does.
_LIST_ICON_SIZE = 32

_NAME_ROLE = Qt.ItemDataRole.UserRole
_ICON_ROLE = Qt.ItemDataRole.UserRole + 1     # the row's thumbnail (drawn by its row widget)
_ROW_HEIGHT = 48                               # >= 44 px: sidebar touch-target minimum


def default_view_for(name: str) -> tuple[complex, float]:
    """The family's parameter-plane framing -- see app.family_views."""
    return parameter_view_for(name)


# The dynamical (Julia-set) plane's OWN default framing has nothing to do
# with the parameter view above: default_view_for frames a-SPACE (the
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


# A flat placeholder square -- built once, lazily, and reused for every
# entry without a thumbnail (a user entry before its first sidecar is
# written; a preset when no thumbnail provider was given, e.g. in tests).
# Module-level because QPixmap can't be constructed before a QApplication
# exists -- built on first use, not at import time.
_placeholder_icon_cache: QIcon | None = None


def _placeholder_icon() -> QIcon:
    global _placeholder_icon_cache
    if _placeholder_icon_cache is None:
        pixmap = QPixmap(_LIST_ICON_SIZE, _LIST_ICON_SIZE)
        pixmap.fill(theme.color("raised"))
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


def _plain_formula(formula: str) -> str:
    """The formula as linear Unicode (H₅(z), z²) for search -- so typing what
    the typeset display shows finds it."""
    try:
        return plain_text(parse_display(formula))
    except DisplayParseError:
        return formula


class LibraryRow(QWidget):
    """One list row: loaded dot, thumbnail, name, formula, and a ⋯ menu."""

    def __init__(self, name: str, formula: str, icon: QIcon, is_preset: bool,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(8)
        self.dot = QLabel("●", self)
        self.dot.setProperty("role", "loadedDot")
        self.dot.setToolTip("Currently loaded")
        self.dot.setFixedWidth(10)
        layout.addWidget(self.dot)
        self.thumbnail = QLabel(self)
        self.thumbnail.setFixedSize(_LIST_ICON_SIZE, _LIST_ICON_SIZE)
        layout.addWidget(self.thumbnail)
        self.set_icon(icon)
        text = QVBoxLayout()
        text.setSpacing(0)
        self.name_label = QLabel(name, self)
        # Typeset, compact (fitted to the row, middle-elided; full formula
        # in a tooltip).
        self.formula_view = FormulaView(self, compact=True, px=12, height=20, color_role="muted")
        self.formula_view.set_empty_text("(empty map)")
        self.formula_view.set_source(formula)
        text.addWidget(self.name_label)
        text.addWidget(self.formula_view)
        layout.addLayout(text, 1)
        self.menu_button = QToolButton(self)
        self.menu_button.setProperty("role", "icon")
        self.menu_button.setIcon(more_icon(theme.color("muted")))
        self.menu_button.setToolTip(f"Actions for {name}")
        self.menu_button.setAccessibleName(f"Actions for {name}")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        layout.addWidget(self.menu_button)
        for label in (self.name_label, self.thumbnail, self.dot):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.set_loaded(False)

    def set_icon(self, icon: QIcon) -> None:
        self.thumbnail.setPixmap(icon.pixmap(_LIST_ICON_SIZE, _LIST_ICON_SIZE))

    def set_loaded(self, loaded: bool) -> None:
        # Keep the space (so names line up), only show/hide the dot itself.
        self.dot.setText("●" if loaded else "")


class _ReturnLoads(QObject):
    """Return/Enter on the list loads the current entry."""

    def __init__(self, panel: "LibraryPanel"):
        super().__init__(panel)
        self._panel = panel

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return,
                                                                     Qt.Key.Key_Enter):
            self._panel._load_selected()
            return True
        return False


class LibraryPanel(QWidget):
    """Search, then two sections (Presets, Yours), each row with a loaded
    dot, a thumbnail and a ⋯ menu; the selected entry's formula and notes
    below; "Save current as..." at the bottom.

    `thumbnail_provider(name, map) -> QIcon | None` supplies every row's
    thumbnail, preset or saved (SandboxWindow's FamilyThumbnailer renders
    them off the GUI thread and returns a placeholder until each arrives;
    refresh_preview(name) then swaps it in). Without one, a saved family
    shows its sidecar PNG and a preset a placeholder. `confirm(title, text)
    -> bool` asks before a delete (a QMessageBox by default; tests inject
    their own). `on_saved_as(name)` and `on_renamed(old, new)` let the
    owner persist per-family data that lives outside the library (the
    parameter-plane framing, app.family_views) -- this panel itself still
    never writes to disk.
    """

    def __init__(self, session, on_load: Callable[[], None], on_change: Callable[[], None],
                parent: QWidget | None = None,
                thumbnail_provider: Callable[[str, cdx.RationalMap], QIcon | None] | None = None,
                confirm: Callable[[str, str], bool] | None = None,
                on_saved_as: Callable[[str], None] | None = None,
                on_renamed: Callable[[str, str], None] | None = None):
        super().__init__(parent)
        self.session = session
        self._on_load = on_load
        self._on_change = on_change
        self._thumbnail_provider = thumbnail_provider
        self._confirm = confirm or self._confirm_with_dialog
        self._on_saved_as = on_saved_as
        self._on_renamed = on_renamed
        self._loaded_name: str | None = None
        self._rows: dict[str, LibraryRow] = {}

        self._build_ui()
        self._refresh_list()

    # ---- construction ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.search_field = QLineEdit(self)
        self.search_field.setPlaceholderText("Search families")
        self.search_field.setAccessibleName("Search families")
        self.search_field.setClearButtonEnabled(True)
        self.search_field.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_field)

        self._list = QListWidget(self)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._list.setMinimumHeight(_ROW_HEIGHT * 4)
        self._list.currentItemChanged.connect(lambda *_: self._refresh_detail())
        # Double-click or Return = Load (the press already made it current).
        self._list.itemDoubleClicked.connect(lambda _item: self._load_selected())
        self._list.installEventFilter(_ReturnLoads(self))
        layout.addWidget(self._list, 1)

        # Detail header: the family's name, then its typeset formula (compact).
        header = QHBoxLayout()
        header.setSpacing(8)
        self._formula_label = QLabel(self)
        self._formula_label.setProperty("mono", True)
        self._formula_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self._formula_label)
        self._formula_view = FormulaView(self, compact=True, px=15, height=34)
        self._formula_view.set_empty_text("(empty map)")
        header.addWidget(self._formula_view, 1)
        layout.addLayout(header)
        notes_caption = QLabel("NOTES", self)
        notes_caption.setProperty("role", "section")
        layout.addWidget(notes_caption)
        self._notes_edit = QPlainTextEdit(self)
        self._notes_edit.setAccessibleName("Notes")
        self._notes_edit.setMaximumHeight(110)
        self._notes_edit.textChanged.connect(self._on_notes_edited)
        layout.addWidget(self._notes_edit)

        self._error_label = QLabel(self)
        self._error_label.setWordWrap(True)
        self._error_label.setProperty("role", "error")
        layout.addWidget(self._error_label)

        self._save_button = QPushButton("Save current as…", self)
        self._save_button.setToolTip("Add the current map to Yours under a new name")
        self._save_button.clicked.connect(self._save_current_as)
        layout.addWidget(self._save_button)

    # ---- refresh: rebuild the list (and detail pane) from session.library --------------
    def _add_section(self, title: str) -> None:
        item = QListWidgetItem()
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QSize(0, 28))
        self._list.addItem(item)
        label = QLabel(title.upper())
        label.setProperty("role", "section")
        label.setContentsMargins(4, 6, 0, 0)
        self._list.setItemWidget(item, label)

    def _row_icon(self, name: str, entry) -> QIcon:
        icon = self._thumbnail_provider(name, entry) if self._thumbnail_provider else None
        if icon is not None:
            return icon
        return _placeholder_icon() if name in PRESET_FAMILY_NAMES else _icon_for(name)

    def _refresh_list(self, select: str | None = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        self._rows.clear()
        names = list(self.session.library.names())
        presets = [n for n in names if n in PRESET_FAMILY_NAMES]
        yours = [n for n in names if n not in PRESET_FAMILY_NAMES]
        for title, group in (("Presets", presets), ("Yours", yours)):
            if not group:
                continue
            self._add_section(title)
            for name in group:
                entry = self.session.library.find(name)
                icon = self._row_icon(name, entry)
                item = QListWidgetItem()
                item.setData(_NAME_ROLE, name)
                item.setData(_ICON_ROLE, icon)
                item.setSizeHint(QSize(0, _ROW_HEIGHT))
                item.setToolTip("Double-click or press Return to load")
                self._list.addItem(item)
                row = LibraryRow(name, entry.to_formula() if entry is not None else "", icon,
                                 name in PRESET_FAMILY_NAMES)
                row.menu_button.setMenu(self._row_menu(name))
                row.set_loaded(name == self._loaded_name)
                self._list.setItemWidget(item, row)
                self._rows[name] = row
        self._list.blockSignals(False)
        target = select if select is not None else (self._loaded_name or self.session.map.name)
        self._select_by_name(target)
        self._apply_filter(self.search_field.text())
        self._refresh_detail()

    def _row_menu(self, name: str) -> QMenu:
        menu = QMenu(self)
        load = menu.addAction("Load")
        load.triggered.connect(lambda: (self._select_by_name(name), self._load_selected()))
        if name not in PRESET_FAMILY_NAMES:   # presets are read-only: no rename/delete
            menu.addSeparator()
            overwrite = menu.addAction("Save current map here")
            overwrite.triggered.connect(lambda: self._do_save_as(name))
            rename = menu.addAction("Rename…")
            rename.triggered.connect(lambda: (self._select_by_name(name), self._rename_selected()))
            delete = menu.addAction("Delete…")
            delete.triggered.connect(lambda: (self._select_by_name(name), self._delete_selected()))
        return menu

    def row_menu_texts(self, name: str) -> list[str]:
        row = self._rows.get(name)
        if row is None:
            return []
        return [a.text() for a in row.menu_button.menu().actions() if a.text()]

    def row_icon(self, name: str) -> QIcon | None:
        item = self._item_for(name)
        return item.data(_ICON_ROLE) if item is not None else None

    def entry_names(self) -> list[str]:
        """The entries in list order (section headers excluded)."""
        return [self._list.item(r).data(_NAME_ROLE) for r in range(self._list.count())
                if self._list.item(r).data(_NAME_ROLE) is not None]

    def visible_entry_names(self) -> list[str]:
        return [n for n in self.entry_names() if not self._item_for(n).isHidden()]

    def _item_for(self, name: str) -> QListWidgetItem | None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_NAME_ROLE) == name:
                return item
        return None

    def refresh_previews(self) -> None:
        """Re-reads each row's thumbnail without touching selection or the
        detail pane -- after a library change, or a palette/strategy change
        (the thumbnail provider re-renders what is out of date)."""
        for name in self.entry_names():
            self.refresh_preview(name)

    def refresh_preview(self, name: str) -> None:
        """Re-reads ONE row's thumbnail -- e.g. when its render arrives."""
        item = self._item_for(name)
        if item is None:
            return
        icon = self._row_icon(name, self.session.library.find(name))
        item.setData(_ICON_ROLE, icon)
        if name in self._rows:
            self._rows[name].set_icon(icon)

    # ---- search ----------------------------------------------------------------------------
    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        section_item = None
        section_has_visible = False
        for row in range(self._list.count()):
            item = self._list.item(row)
            name = item.data(_NAME_ROLE)
            if name is None:   # a section header
                if section_item is not None:
                    section_item.setHidden(not section_has_visible)
                section_item, section_has_visible = item, False
                continue
            entry = self.session.library.find(name)
            formula = entry.to_formula() if entry is not None else ""
            haystack = f"{name} {formula} {_plain_formula(formula)}".lower()
            hidden = bool(needle) and needle not in haystack
            item.setHidden(hidden)
            section_has_visible = section_has_visible or not hidden
        if section_item is not None:
            section_item.setHidden(not section_has_visible)

    # ---- loaded entry: a dot, and the selection follows it ------------------------------
    def set_loaded(self, name: str | None) -> None:
        """Marks `name` as the loaded family (None: the map is custom) and
        selects its row."""
        self._loaded_name = name
        for row_name, row in self._rows.items():
            row.set_loaded(row_name == name)
        if name is not None and name in self._rows:
            self._select_by_name(name)

    def loaded_name(self) -> str | None:
        return self._loaded_name

    def _select_by_name(self, name: str) -> None:
        item = self._item_for(name)
        if item is not None:
            self._list.setCurrentItem(item)
            return
        names = self.entry_names()
        if names:
            self._list.setCurrentItem(self._item_for(names[0]))

    def _selected_name(self) -> str | None:
        item = self._list.currentItem()
        return item.data(_NAME_ROLE) if item is not None else None

    def _refresh_detail(self) -> None:
        name = self._selected_name()
        self._notes_edit.blockSignals(True)
        if name is None:
            self._formula_label.setText("")
            self._formula_view.set_source("")
            self._notes_edit.setPlainText("")
            self._notes_edit.setReadOnly(True)
        else:
            entry = self.session.library.find(name)
            formula = entry.to_formula() if entry is not None else ""
            self._formula_label.setText(f"{name}:")
            self._formula_view.set_source(formula)
            self._formula_view.setToolTip(formula)
            self._notes_edit.setPlainText(entry.notes if entry is not None else "")
            self._notes_edit.setReadOnly(name in PRESET_FAMILY_NAMES)
            self._notes_edit.setPlaceholderText("" if name in PRESET_FAMILY_NAMES
                                                else "Notes about this family")
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
    def load_family(self, name: str) -> bool:
        """Selects `name` and loads it, exactly as double-clicking its row
        would. False if there is no such entry."""
        item = self._item_for(name)
        if item is None:
            return False
        self._list.setCurrentItem(item)
        self._load_selected()
        return True

    def _load_selected(self) -> None:
        # The viewport reset (mode-aware, per pane) is on_load's job
        # (app/sandbox.py's _on_family_loaded): this panel touches no viewport.
        name = self._selected_name()
        if name is None:
            return
        try:
            self.session.load_from_library(name)
        except KeyError as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        self.set_loaded(name)
        self._on_load()

    # ---- save current map as a new (or updated) library entry --------------------------
    def _save_current_as(self) -> None:
        text, ok = QInputDialog.getText(self, "Save Current As", "Family name:",
                                        text=self.session.map.name)
        if not ok or not text.strip():
            return
        self._do_save_as(text.strip())

    def save_as(self, name: str) -> str | None:
        """Saves the current map under `name` (made unique with " (2)", ...
        if taken) without asking -- the family builder's "Save as family".
        Returns the name used, or None on failure (message in the panel)."""
        from app.family_builder import unique_name
        name = unique_name(name, self.session.library.names())
        self._do_save_as(name)
        return name if self.session.library.find(name) is not None else None

    def _do_save_as(self, name: str) -> None:
        try:
            self.session.save_to_library(name)
        except ValueError as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        self._loaded_name = name   # the current map now IS this entry
        if self._on_saved_as is not None:
            self._on_saved_as(name)   # e.g. record its parameter-plane framing
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
        if self._loaded_name == old_name:
            self._loaded_name = new_name
        if self._on_renamed is not None:
            self._on_renamed(old_name, new_name)
        self._refresh_list(select=new_name)
        self._on_change()

    # ---- delete a library entry (asks first) -----------------------------------------------
    @staticmethod
    def _confirm_with_dialog(title: str, text: str) -> bool:
        answer = QMessageBox.question(None, title, text,
                                      QMessageBox.StandardButton.Delete
                                      | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Delete

    def _delete_selected(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        if not self._confirm("Delete family",
                             f"Delete \"{name}\" from your library? This can't be undone."):
            return
        self._do_delete(name)

    def _do_delete(self, name: str) -> None:
        try:
            self.session.delete_from_library(name)
        except (KeyError, ValueError) as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        if self._loaded_name == name:
            self._loaded_name = None
        self._refresh_list()
        self._on_change()
