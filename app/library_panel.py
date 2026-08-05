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

PERSISTENCE. This panel never touches disk itself -- every successful
mutation (save/rename/delete/notes edit) calls on_change(), and every
successful load calls on_load(); SandboxWindow wires the former to
session.save_user_library(app.settings.library_path()) and the latter to a
viewport reset + immediate render. Keeping disk I/O out of this class is
what keeps it constructible standalone in tests without ever touching a
real ~/.complexdynamics/library.txt (see app/test_settings_panel.py's own
note on the identical concern for Settings).

LOADING RESETS THE VIEWPORT to a per-family default (_DEFAULT_VIEWS below)
-- hand-picked framings for the six built-in shapes' most legible region,
falling back to a generic default for anything else (a user-saved family).
It does not touch render_mode or the parameter; loading a family has no
informed opinion about either.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

import cdx
from app.session import PRESET_FAMILY_NAMES

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
            item = QListWidgetItem(label)
            item.setData(_NAME_ROLE, entry_name)
            self._list.addItem(item)
        self._list.blockSignals(False)
        target = select if select is not None else self.session.map.name
        self._select_by_name(target)
        self._refresh_detail()

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

    # ---- load: replaces session.map, resets the viewport to a sensible default ---------
    def _load_selected(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        try:
            self.session.load_from_library(name)
        except KeyError as e:
            self._error_label.setText(str(e))
            return
        self._error_label.setText("")
        center, scale = default_view_for(name)
        vp = self.session.viewport
        self.session.viewport = cdx.Viewport(center, scale, vp.resolution)
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
