"""widgets.py -- small reusable pieces of the redesigned window.

  SegmentedControl  exclusive, checkable QToolButtons in a QButtonGroup; the
                    items carry data (e.g. RENDER_MODES ids), the buttons
                    only ever show friendly labels.
  TabbedDock        a QDockWidget holding an underline tab strip + stacked
                    pages, collapsible to just its tab strip (a vertical
                    strip when docked at the side).
  MapChip           the top bar's "family  formula" pill.
  TopBar            wordmark, map chip, workspace switch, Coupled/Single,
                    render status, Reset view, Export.

Looks come from app/theme.py through dynamic properties; nothing here
hard-codes a color.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QDockWidget, QFrame, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
                               QTabBar,
                               QToolButton, QVBoxLayout, QWidget)

from app import theme

QWIDGETSIZE_MAX = (1 << 24) - 1


# ---- segmented control ----------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    data: Any
    label: str
    tooltip: str = ""


class SegmentedControl(QFrame):
    """`selected(data)` fires on a user click only; set_current() is silent.

    `groups` lets one control hold several runs of segments separated by a
    thin divider (Single view's "Parameter ▸ … | Dynamical ▸ …")."""

    selected = Signal(object)

    def __init__(self, segments: Iterable[Segment] = (), parent: QWidget | None = None,
                 accessible_name: str = ""):
        super().__init__(parent)
        self.setProperty("role", "segmented")
        self.setAccessibleName(accessible_name)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonClicked.connect(self._on_clicked)
        self._buttons: dict[Any, QToolButton] = {}
        self._last_selected: Any = None
        self.set_segments([list(segments)])
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def set_segments(self, groups: list[list[Segment]], titles: list[str] | None = None) -> None:
        """`titles`: an optional small caption before each group (Single
        view's "Parameter ▸" / "Dynamical ▸")."""
        current = self.current()
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()          # deleteLater alone leaves it painted until
                item.widget().deleteLater()   # the event loop next runs deferred deletes
        for button in list(self._group.buttons()):
            self._group.removeButton(button)
        self._buttons.clear()
        for gi, segments in enumerate(groups):
            if gi > 0 and segments:
                divider = QFrame(self)
                divider.setProperty("role", "segmentDivider")
                divider.setFixedWidth(1)
                self._layout.addWidget(divider)
            if titles and gi < len(titles) and titles[gi]:
                caption = QLabel(titles[gi], self)
                caption.setProperty("role", "segmentGroup")
                self._layout.addWidget(caption)
            for seg in segments:
                button = QToolButton(self)
                button.setProperty("role", "segment")
                button.setText(seg.label)
                button.setToolTip(seg.tooltip or seg.label)
                button.setAccessibleName(seg.tooltip or seg.label)
                button.setCheckable(True)
                button.setAutoRaise(False)
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
                button.setProperty("segmentData", seg.data)
                button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
                self._group.addButton(button)
                self._layout.addWidget(button)
                self._buttons[seg.data] = button
        if current in self._buttons:
            self._buttons[current].setChecked(True)

    def _on_clicked(self, button: QToolButton) -> None:
        data = button.property("segmentData")
        if data == self._last_selected:
            return   # re-clicking the selected segment changes nothing
        self._last_selected = data
        self.selected.emit(data)

    def set_current(self, data: Any) -> None:
        button = self._buttons.get(data)
        self._last_selected = data
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def current(self) -> Any:
        button = self._group.checkedButton() if hasattr(self, "_group") else None
        return None if button is None else button.property("segmentData")

    def button(self, data: Any) -> QToolButton | None:
        return self._buttons.get(data)

    def items(self) -> list[Any]:
        return list(self._buttons)

    def labels(self) -> list[str]:
        return [b.text() for b in self._buttons.values()]


# ---- tabbed, collapsible dock -------------------------------------------------------------
class TabbedDock(QDockWidget):
    """A dock whose content is an underline tab strip over stacked pages.

    side="right": the strip sits at the top of the dock (tabs equal width);
    collapsed, the dock shrinks to a vertical strip of the same tabs.
    side="bottom": the strip (with an optional caps heading) is the dock's
    title bar; collapsed, only the strip remains.
    """

    currentChanged = Signal(int)
    collapsedChanged = Signal(bool)

    COLLAPSED_SIDE_WIDTH = 48

    def __init__(self, title: str, object_name: str, side: str, parent: QWidget | None = None,
                 heading: str | None = None):
        super().__init__(title, parent)
        self.setObjectName(object_name)
        self._side = side
        self._collapsed = False
        self._expanded_extent: int | None = None
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea if side == "right"
                             else Qt.DockWidgetArea.BottomDockWidgetArea)

        self.tab_bar = QTabBar()
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setExpanding(side == "right")
        self.tab_bar.setUsesScrollButtons(False)
        self.tab_bar.setElideMode(Qt.TextElideMode.ElideNone)
        if side == "right":
            self.tab_bar.setProperty("role", "sidebarTabs")
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        self.tab_bar.tabBarClicked.connect(self._on_tab_clicked)

        self.stack = QStackedWidget()
        self._contents: list[QWidget] = []   # the tab widgets (pages may wrap them in a scroll area)

        self.collapse_button = QToolButton()
        self.collapse_button.setProperty("role", "icon")
        self.collapse_button.setAutoRaise(True)
        self.collapse_button.clicked.connect(lambda: self.set_collapsed(not self._collapsed))

        header = QWidget()
        header.setProperty("role", "surface")
        header_layout = QHBoxLayout(header)
        header_layout.setSpacing(4)
        if heading:
            self.heading_label = QLabel(heading.upper())
            self.heading_label.setProperty("role", "dockTitle")
            header_layout.addWidget(self.heading_label)
            header_layout.addSpacing(8)
        else:
            self.heading_label = None
        header_layout.addWidget(self.tab_bar, 1 if side == "right" else 0)
        if side != "right":
            header_layout.addStretch(1)
        header_layout.addWidget(self.collapse_button)
        self.header = header

        if side == "right":
            header_layout.setContentsMargins(0, 0, 4, 0)
            self.setTitleBarWidget(QWidget())   # the tab strip is the header
            self.side_bar = QTabBar()
            self.side_bar.setShape(QTabBar.Shape.RoundedEast)
            self.side_bar.setDrawBase(False)
            self.side_bar.setProperty("role", "sidebarTabs")
            self.side_bar.tabBarClicked.connect(self._on_side_tab_clicked)
            self.expand_button = QToolButton()
            self.expand_button.setProperty("role", "icon")
            self.expand_button.setAutoRaise(True)
            self.expand_button.clicked.connect(lambda: self.set_collapsed(False))

            expanded = QWidget()
            ev = QVBoxLayout(expanded)
            ev.setContentsMargins(0, 0, 0, 0)
            ev.setSpacing(0)
            ev.addWidget(header)
            ev.addWidget(self.stack, 1)
            collapsed = QWidget()
            collapsed.setProperty("role", "surface")
            cv = QVBoxLayout(collapsed)
            cv.setContentsMargins(4, 4, 4, 4)
            cv.setSpacing(4)
            cv.addWidget(self.expand_button, 0, Qt.AlignmentFlag.AlignHCenter)
            cv.addWidget(self.side_bar, 0, Qt.AlignmentFlag.AlignHCenter)
            cv.addStretch(1)
            self._pages = QStackedWidget()
            self._pages.addWidget(expanded)
            self._pages.addWidget(collapsed)
            self.setWidget(self._pages)
        else:
            header_layout.setContentsMargins(14, 0, 8, 0)
            self.side_bar = None
            self.setTitleBarWidget(header)
            self.setWidget(self.stack)
        self._update_collapse_icons()

    # ---- tabs ----
    def add_tab(self, widget: QWidget, title: str, tooltip: str = "", scroll: bool = True) -> int:
        return self.insert_tab(self.count(), widget, title, tooltip, scroll)

    def insert_tab(self, index: int, widget: QWidget, title: str, tooltip: str = "",
                   scroll: bool = True) -> int:
        """`scroll`: wrap the page in a scroll area, so a dock smaller than
        the page's own minimum size scrolls instead of refusing to shrink
        (everything stays reachable on a small screen)."""
        page = widget
        if scroll:
            page = QScrollArea()
            page.setWidgetResizable(True)
            page.setFrameShape(QFrame.Shape.NoFrame)
            page.setWidget(widget)
        self._contents.insert(index, widget)
        self.stack.insertWidget(index, page)
        index = self.tab_bar.insertTab(index, title)
        if tooltip:
            self.tab_bar.setTabToolTip(index, tooltip)
        if self.side_bar is not None:
            self.side_bar.insertTab(index, title)
        self.stack.setCurrentIndex(self.tab_bar.currentIndex())
        return index

    def set_tab_visible(self, index: int, visible: bool) -> None:
        self.tab_bar.setTabVisible(index, visible)
        if self.side_bar is not None:
            self.side_bar.setTabVisible(index, visible)

    def count(self) -> int:
        return self.tab_bar.count()

    def widget(self, index: int) -> QWidget | None:
        return self._contents[index] if 0 <= index < len(self._contents) else None

    def index_of(self, widget: QWidget) -> int:
        return self._contents.index(widget) if widget in self._contents else -1

    def tab_text(self, index: int) -> str:
        return self.tab_bar.tabText(index)

    def current_index(self) -> int:
        return self.tab_bar.currentIndex()

    def current_widget(self) -> QWidget | None:
        return self.widget(self.current_index())

    def set_current_index(self, index: int) -> None:
        self.tab_bar.setCurrentIndex(index)

    def set_current_widget(self, widget: QWidget) -> None:
        index = self.index_of(widget)
        if index >= 0:
            self.tab_bar.setCurrentIndex(index)

    def reveal(self, widget: QWidget | None = None) -> None:
        """Show the dock (reopening it if closed), expand it, and select
        `widget`'s tab -- what a menu item or shortcut that "goes to" a tab does."""
        if widget is not None:
            self.set_current_widget(widget)
        self.show()
        self.raise_()
        self.set_collapsed(False)

    def _on_tab_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if self.side_bar is not None and self.side_bar.currentIndex() != index:
            self.side_bar.setCurrentIndex(index)
        self.currentChanged.emit(index)

    def _on_tab_clicked(self, index: int) -> None:
        # Bottom dock: clicking a tab of a collapsed dock opens it.
        if self._collapsed and index >= 0:
            self.set_collapsed(False)

    def _on_side_tab_clicked(self, index: int) -> None:
        if index >= 0:
            self.tab_bar.setCurrentIndex(index)
        self.set_collapsed(False)

    # ---- collapse ----
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._collapsed:
            return
        main = self.parentWidget()
        if self._side == "right":
            if collapsed:
                self._expanded_extent = self.width()
                self._pages.setCurrentIndex(1)
                self.setMinimumWidth(self.COLLAPSED_SIDE_WIDTH)
                self.setMaximumWidth(self.COLLAPSED_SIDE_WIDTH)
            else:
                self._pages.setCurrentIndex(0)
                self.setMinimumWidth(0)
                self.setMaximumWidth(QWIDGETSIZE_MAX)
                if main is not None and self._expanded_extent:
                    main.resizeDocks([self], [self._expanded_extent], Qt.Orientation.Horizontal)
        else:
            if collapsed:
                self._expanded_extent = self.height()
                self.stack.hide()
                self.setMaximumHeight(self.header.sizeHint().height())
            else:
                self.stack.show()
                self.setMaximumHeight(QWIDGETSIZE_MAX)
                if main is not None and self._expanded_extent:
                    main.resizeDocks([self], [self._expanded_extent], Qt.Orientation.Vertical)
        self._collapsed = collapsed
        self._update_collapse_icons()
        self.collapsedChanged.emit(collapsed)

    def _update_collapse_icons(self) -> None:
        color = theme.color("muted")
        if self._side == "right":
            self.collapse_button.setIcon(chevron_icon("right", color))
            self.collapse_button.setToolTip("Collapse the sidebar to its tabs")
            self.expand_button.setIcon(chevron_icon("left", color))
            self.expand_button.setToolTip("Expand the sidebar")
        else:
            self.collapse_button.setIcon(chevron_icon("up" if self._collapsed else "down", color))
            self.collapse_button.setToolTip("Expand" if self._collapsed else "Collapse to the tab strip")
        self.collapse_button.setAccessibleName(self.collapse_button.toolTip())


# ---- icons (painted, so they follow the theme's colors) --------------------------------------
def _icon(paint, size: int = 16) -> QIcon:
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    paint(painter, size)
    painter.end()
    return QIcon(pixmap)


def reset_icon(color) -> QIcon:
    """A counter-clockwise arrow (mockup B's reset glyph)."""
    def paint(p: QPainter, s: int) -> None:
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        r = QRectF(2.5, 2.5, s - 5, s - 5)
        path = QPainterPath()
        path.arcMoveTo(r, 160)
        path.arcTo(r, 160, 290)
        p.drawPath(path)
        start = path.pointAtPercent(0)
        p.drawLine(start, start + QPointF(0, -3.6))
        p.drawLine(start, start + QPointF(3.6, 0.4))
    return _icon(paint)


def chevron_icon(direction: str, color) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        c, d = s / 2, 3.0
        pts = {"right": [(c - d / 2, c - d), (c + d / 2, c), (c - d / 2, c + d)],
               "left": [(c + d / 2, c - d), (c - d / 2, c), (c + d / 2, c + d)],
               "down": [(c - d, c - d / 2), (c, c + d / 2), (c + d, c - d / 2)],
               "up": [(c - d, c + d / 2), (c, c - d / 2), (c + d, c + d / 2)]}[direction]
        p.drawPolyline([QPointF(x, y) for x, y in pts])
    return _icon(paint)


def more_icon(color) -> QIcon:
    """⋯ (row menus)."""
    def paint(p: QPainter, s: int) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        for x in (s / 2 - 5, s / 2, s / 2 + 5):
            p.drawEllipse(QPointF(x, s / 2), 1.4, 1.4)
    return _icon(paint)


# ---- a label that elides instead of clipping ----------------------------------------------
class ElidedLabel(QLabel):
    """Shows "…" when its text doesn't fit; the full text is in the tooltip."""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:   # noqa: N802 (Qt naming)
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = max(0, self.width())
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width) \
            if width > 0 else self._full
        super().setText(shown)


# ---- formula pretty-printing ----------------------------------------------------------------
_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def pretty_formula(formula: str) -> str:
    """'z^2 + a*z - 1' -> 'z² + a·z − 1' (display only; never parsed back)."""
    text = re.sub(r"\^\(?(-?\d+)\)?", lambda m: m.group(1).translate(_SUPERSCRIPTS), formula)
    text = text.replace("*", "·")
    text = re.sub(r"(?<=\s)-(?=\s)", "−", text)
    return text


# ---- map chip ---------------------------------------------------------------------------
class MapChip(QPushButton):
    """'family  formula' in a pill; clicking it opens the Library."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "mapChip")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(8)
        self.name_label = QLabel(self)
        self.name_label.setProperty("role", "muted")
        self.name_label.setProperty("mono", True)
        self.formula_label = QLabel(self)
        self.formula_label.setProperty("mono", True)
        for label in (self.name_label, self.formula_label):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(label)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def set_map(self, name: str, formula: str) -> None:
        self.name_label.setText(name)
        self.formula_label.setText(pretty_formula(formula) if formula else "(empty map)")
        self.setAccessibleName(f"{name}: {formula}. Open the library")
        self.updateGeometry()

    def text_shown(self) -> str:
        return f"{self.name_label.text()} {self.formula_label.text()}"

    def sizeHint(self) -> QSize:
        layout = self.layout()
        hint = layout.sizeHint()
        return QSize(hint.width(), max(theme.CONTROL_HEIGHT, hint.height()))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


# ---- top bar -----------------------------------------------------------------------------
class TopBar(QWidget):
    """Layout only: SandboxWindow wires every control."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(14)

        self.wordmark = QLabel("PHOENIX", self)
        self.wordmark.setProperty("role", "wordmark")
        layout.addWidget(self.wordmark)

        self.map_chip = MapChip(self)
        layout.addWidget(self.map_chip)

        self.workspace_switch = SegmentedControl(
            [Segment("explore", "Explore", "The parameter and dynamical planes")], self,
            accessible_name="Workspace")
        self.workspace_switch.set_current("explore")
        self.workspace_switch.hide()   # shown once a module registers a workspace
        layout.addWidget(self.workspace_switch)

        self.view_switch = SegmentedControl(
            [Segment(True, "Coupled", "Coupled view: parameter plane and dynamical plane side by side"),
             Segment(False, "Single", "Single view: one plane, any mode")], self,
            accessible_name="Coupled or single view")
        self.view_switch.set_current(True)
        layout.addWidget(self.view_switch)

        layout.addStretch(1)

        self.status_label = QLabel("", self)
        self.status_label.setProperty("role", "status")
        layout.addWidget(self.status_label)
        self.progress = QProgressBar(self)
        self.progress.setFixedWidth(120)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setAccessibleName("Render progress")
        layout.addWidget(self.progress)

        self.reset_button = QToolButton(self)
        self.reset_button.setIcon(reset_icon(theme.color("text")))
        self.reset_button.setIconSize(QSize(16, 16))
        self.reset_button.setAccessibleName("Reset view")
        layout.addWidget(self.reset_button)

        self.export_button = QPushButton("Export", self)
        self.export_button.setProperty("role", "primary")
        layout.addWidget(self.export_button)

    def set_render_status(self, text: str, busy: bool) -> None:
        self.status_label.setText(text)
        # No fractional progress exists to report (cdx renders have no
        # progress callback): an indeterminate bar while busy, full when idle.
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
        # Idle: no bar at all (a full bar next to "Ready" reads as noise).
        self.progress.setVisible(busy)
