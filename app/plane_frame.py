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

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from app import theme
from app.widgets import FlowLayout, SegmentedControl


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
        # The parameter strategy, whenever it is not the default
        # ("Escape · per-critical #1") -- never a hidden setting. A caption
        # line under the title: the row is as tall as the mode control
        # anyway, so it costs no height and takes no width from the modes.
        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        title_box.addStretch(1)
        title_box.addWidget(self.title_label)
        self.strategy_label = QLabel(self)
        self.strategy_label.setProperty("role", "caption")
        self.strategy_label.setAccessibleName("Parameter strategy")
        self.strategy_label.hide()
        title_box.addWidget(self.strategy_label)
        title_box.addStretch(1)
        header.addLayout(title_box)
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

    def set_strategy_text(self, text: str, tooltip: str = "") -> None:
        """The header's strategy note; empty hides it."""
        self.strategy_label.setText(text)
        self.strategy_label.setToolTip(tooltip)
        self.strategy_label.setVisible(bool(text))

    def strategy_text(self) -> str:
        return self.strategy_label.text() if not self.strategy_label.isHidden() else ""

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
