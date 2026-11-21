"""plane_frame.py -- one plane of the Explore workspace.

    PARAMETER PLANE                 [Escape|Green's|Cycles|Period]
    ┌──────────────────────────────────────────────────────────┐
    │ image (ImageView; readout chips are painted inside it)   │
    └──────────────────────────────────────────────────────────┘
    (Marker) (Crosshair)                  <- overlay chips, wrapping

Layout and look only: SandboxWindow decides which modes the segmented
control offers and wires every chip to its View-menu action.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget, QWidgetItem)

from app import theme
from app.widgets import SegmentedControl


class FlowLayout(QLayout):
    """Left-to-right, wrapping to a new row when out of width (Qt's flow-layout example)."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _do_layout(self, rect: QRect, apply: bool) -> int:
        x, y, line_h = rect.x(), rect.y(), 0
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and line_h > 0:
                x = rect.x()
                y += line_h + self._spacing
                line_h = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


def swatch_icon(rgba: tuple[int, int, int, int] | None, size: int = 8) -> QIcon:
    """The chip's legend dot (a transparent placeholder keeps chips aligned)."""
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    if rgba is not None:
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*rgba))
        painter.drawEllipse(0, 0, size, size)
        painter.end()
    return QIcon(pixmap)


class OverlayChip(QToolButton):
    """A checkable pill mirroring one View-menu overlay action."""

    def __init__(self, action: QAction, label: str, swatch, tooltip: str, secondary: bool,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.action = action
        self.setProperty("role", "chip")
        self.setProperty("secondary", secondary)
        self.setText(label)
        self.setToolTip(tooltip or label)
        self.setAccessibleName(label)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        if swatch is not None:
            self.setIcon(swatch_icon(swatch))
            self.setIconSize(QSize(8, 8))
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        else:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setChecked(action.isChecked())
        self.setEnabled(action.isEnabled())
        # The action owns the state; the chip only mirrors and forwards it.
        self.toggled.connect(self._forward)
        action.toggled.connect(self._mirror)
        action.changed.connect(lambda: self.setEnabled(action.isEnabled()))

    def _forward(self, checked: bool) -> None:
        if self.action.isChecked() != checked:
            self.action.setChecked(checked)

    def _mirror(self, checked: bool) -> None:
        if self.isChecked() != checked:
            self.blockSignals(True)
            self.setChecked(checked)
            self.blockSignals(False)


class PlaneFrame(QWidget):
    resized = Signal()

    PLANE_TITLES = {"parameter": "PARAMETER PLANE", "dynamical": "DYNAMICAL PLANE"}

    def __init__(self, image_view: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.image_view = image_view
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(10)
        self.title_label = QLabel(self)
        self.title_label.setProperty("role", "planeTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.mode_control = SegmentedControl(parent=self, accessible_name="Render mode")
        header.addWidget(self.mode_control)
        outer.addLayout(header)

        self.image_frame = QFrame(self)
        self.image_frame.setProperty("role", "planeFrame")
        frame_layout = QVBoxLayout(self.image_frame)
        frame_layout.setContentsMargins(1, 1, 1, 1)
        frame_layout.addWidget(image_view)
        outer.addWidget(self.image_frame, 1)

        self.chip_row = QWidget(self)
        self.chip_layout = FlowLayout(self.chip_row, spacing=6)
        self.chip_row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        outer.addWidget(self.chip_row)
        self.chips: dict[str, tuple[OverlayChip, str]] = {}   # key -> (chip, gate)
        self._plane = ""

    def add_chip(self, key: str, action: QAction, label: str, gate: str, swatch=None,
                 tooltip: str = "", secondary: bool = False) -> OverlayChip:
        chip = OverlayChip(action, label, swatch, tooltip, secondary, self.chip_row)
        self.chip_layout.addWidget(chip)
        self.chips[key] = (chip, gate)
        return chip

    def set_plane(self, plane: str) -> None:
        """'parameter' or 'dynamical': the caps title, its accent, and which chips show."""
        self._plane = plane
        self.title_label.setText(self.PLANE_TITLES[plane])
        theme.set_property(self.title_label, "plane", plane)
        self.set_chips_visible(True)

    def set_chips_visible(self, visible: bool) -> None:
        for chip, gate in self.chips.values():
            chip.setVisible(visible and gate == self._plane)
        self.chip_row.setVisible(visible)
        self.chip_layout.invalidate()

    def chip_rows_height(self) -> int:
        """Height the chip row needs at the current width (0 if hidden)."""
        if self.chip_row.isHidden():
            return 0
        return self.chip_layout.heightForWidth(max(1, self.chip_row.width() or self.width()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    def plane(self) -> str:
        return self._plane

    def set_focused(self, focused: bool) -> None:
        theme.set_property(self.image_frame, "focused", focused)

    def chip(self, key: str) -> OverlayChip | None:
        entry = self.chips.get(key)
        return entry[0] if entry else None

    def visible_chip_keys(self) -> list[str]:
        return [k for k, (chip, _g) in self.chips.items() if not chip.isHidden()]
