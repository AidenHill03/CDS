"""info_panel.py -- the Info tab: every control, generated from app/controls.py.

Read-only and scrollable. The document is rebuilt from the registry (plus any
entries research-module panels contribute through SandboxWindow.add_panel),
never written by hand, so it cannot disagree with what the controls do.
Styling is deliberately plain (phase 2 of the controls redesign restyles it).
"""

from __future__ import annotations

import html
from typing import Iterable

from PySide6.QtCore import QUrl
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QTextBrowser, QWidget

from app import controls, theme

_FORMULA_URL = "formula:this-map"


class InfoPanel(QTextBrowser):
    def __init__(self, parent: QWidget | None = None, platform: str | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenExternalLinks(False)
        self._platform = platform
        self._extra: list[controls.ControlEntry] = []
        self._this_map: list[tuple[str, str]] = []
        self._map_formula = ""
        self._formula_image = None
        self.setContentsMargins(0, 0, 0, 0)
        self.document().setDocumentMargin(14)
        self.refresh()

    def add_entries(self, entries: Iterable[controls.ControlEntry]) -> None:
        self._extra.extend(entries)
        self.refresh()

    def set_this_map(self, rows: Iterable[tuple[str, str]]) -> None:
        """The "This map" section after the controls: (label, value) rows
        (degree, critical points, where each critical orbit goes)."""
        rows = list(rows)
        if rows != self._this_map:
            self._this_map = rows
            self.refresh()

    def set_map_formula(self, formula: str) -> None:
        """The current map's formula, typeset (compact) at the top of "This
        map"; the text row stays in the table for copying."""
        if formula != self._map_formula:
            self._map_formula = formula
            self._formula_image = None
            self.refresh()

    def _render_formula(self):
        from app.mathdisplay import DisplayParseError, render_image
        if self._formula_image is None and self._map_formula:
            try:
                self._formula_image = render_image(
                    self._map_formula, px=15, compact_size=(max(200.0, self.viewport().width() - 40.0), 34.0),
                    color=theme.color("text"), dpr=max(1.0, self.devicePixelRatioF()))
            except DisplayParseError:
                self._formula_image = None
        return self._formula_image

    def loadResource(self, rtype: int, url: QUrl):   # noqa: N802 (Qt naming)
        if url.toString() == _FORMULA_URL:
            img = self._render_formula()
            if img is not None:
                return img
        return super().loadResource(rtype, url)

    def this_map_rows(self) -> list[tuple[str, str]]:
        return list(self._this_map)

    def refresh(self) -> None:
        scroll = self.verticalScrollBar().value()
        doc = controls.info_html(self._extra, self._platform)
        if self._this_map:
            cells = "".join(f"<tr><td>{html.escape(k)}</td><td>&nbsp;&nbsp;{html.escape(v)}</td></tr>"
                            for k, v in self._this_map)
            formula = ""
            img = self._render_formula()
            if img is not None:
                dpr = img.devicePixelRatio()
                formula = (f'<p><img src="{_FORMULA_URL}" width="{round(img.width() / dpr)}" '
                           f'height="{round(img.height() / dpr)}"></p>')
            doc += f"\n<h3>This map</h3>{formula}<table>{cells}</table>"
        self.setHtml(doc)
        self.verticalScrollBar().setValue(scroll)
