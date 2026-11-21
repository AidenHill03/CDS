"""info_panel.py -- the Info tab: every control, generated from app/controls.py.

Read-only and scrollable. The document is rebuilt from the registry (plus any
entries research-module panels contribute through SandboxWindow.add_panel),
never written by hand, so it cannot disagree with what the controls do.
Styling is deliberately plain (phase 2 of the controls redesign restyles it).
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtWidgets import QTextBrowser, QWidget

from app import controls


class InfoPanel(QTextBrowser):
    def __init__(self, parent: QWidget | None = None, platform: str | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenExternalLinks(False)
        self._platform = platform
        self._extra: list[controls.ControlEntry] = []
        self.refresh()

    def add_entries(self, entries: Iterable[controls.ControlEntry]) -> None:
        self._extra.extend(entries)
        self.refresh()

    def refresh(self) -> None:
        self.setHtml(controls.info_html(self._extra, self._platform))
