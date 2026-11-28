"""app/popups.py -- screen-aware placement for every popup the app opens.

One rule for the Build popover, the Library row ⋯ menus and the formula
tooltip (and any future popup): anchor to the widget (or point) that opened
it; open BELOW when the whole popup fits there, otherwise ABOVE when it fits
there, otherwise on whichever side has more room with the height capped to
that room (the popup's contents then scroll); clamp horizontally; and keep
the whole popup inside QScreen.availableGeometry() of the screen the anchor
is on -- never under the macOS menu bar or Dock, the Windows taskbar, or off
the edge of one monitor onto another.

popup_geometry() is pure geometry (testable without showing anything);
place() applies it to a widget; popup_menu() shows a QMenu through it.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtGui import QGuiApplication, QScreen
from PySide6.QtWidgets import QMenu, QWidget

GAP = 4   # px between the anchor and the popup


def anchor_rect(anchor: QWidget | QRect | QPoint) -> QRect:
    """The anchor in global coordinates."""
    if isinstance(anchor, QWidget):
        return QRect(anchor.mapToGlobal(QPoint(0, 0)), anchor.size())
    if isinstance(anchor, QPoint):
        return QRect(anchor, QSize(1, 1))
    return QRect(anchor)


def screen_for(anchor: QWidget | QRect | QPoint) -> QScreen:
    """The screen the anchor is on: the one under its centre, else the
    widget's own screen, else the primary."""
    r = anchor_rect(anchor)
    for p in (r.center(), r.topLeft()):
        screen = QGuiApplication.screenAt(p)
        if screen is not None:
            return screen
    # An anchor hanging past a screen edge (a pointer in the last pixels of
    # a monitor, a widget partly off-screen): the screen nearest to it, not
    # the primary.
    screens = QGuiApplication.screens()
    if screens:
        c = r.center()

        def distance(sc: QScreen) -> int:
            g = sc.geometry()
            dx = max(g.left() - c.x(), 0, c.x() - (g.left() + g.width() - 1))
            dy = max(g.top() - c.y(), 0, c.y() - (g.top() + g.height() - 1))
            return dx * dx + dy * dy
        return min(screens, key=distance)
    if isinstance(anchor, QWidget) and anchor.screen() is not None:
        return anchor.screen()
    return QGuiApplication.primaryScreen()


def available_rect(anchor: QWidget | QRect | QPoint) -> QRect:
    return screen_for(anchor).availableGeometry()


def popup_geometry(anchor: QRect, size: QSize, avail: QRect, min_height: int = 0,
                   gap: int = GAP, align_right: bool = False) -> QRect:
    """Where a popup of `size` goes for a global `anchor` rect inside
    `avail` (see the module docstring). The result always lies inside
    `avail`; its height may be less than size.height() (never less than
    min_height unless `avail` itself is shorter)."""
    w = min(size.width(), avail.width())
    h = size.height()
    a_top, a_bottom = anchor.top(), anchor.top() + anchor.height()      # bottom exclusive
    v_top, v_bottom = avail.top(), avail.top() + avail.height()
    below = v_bottom - (a_bottom + gap)
    above = (a_top - gap) - v_top
    if h <= below:
        y = a_bottom + gap
    elif h <= above:
        y = a_top - gap - h
    else:
        room = max(below, above)
        if room >= max(min_height, 1):
            h = room
            y = a_bottom + gap if below >= above else a_top - gap - h
        else:   # neither side has even the minimum: overlap the anchor, inside the screen
            h = min(h, avail.height())
            y = min(max(a_bottom + gap, v_top), v_bottom - h)
    x = (anchor.left() + anchor.width() - w) if align_right else anchor.left()
    x = min(max(x, avail.left()), avail.left() + avail.width() - w)
    return QRect(x, y, w, h)


def place(popup: QWidget, anchor: QWidget | QRect | QPoint, size: QSize | None = None,
          min_height: int = 0, align_right: bool = False) -> QRect:
    """Moves/resizes `popup` (a top-level window) by popup_geometry, on the
    anchor's screen. `size` defaults to the popup's sizeHint()."""
    want = size if size is not None else popup.sizeHint()
    screen = screen_for(anchor)
    geo = popup_geometry(anchor_rect(anchor), want, screen.availableGeometry(), min_height,
                         align_right=align_right)
    if popup.isWindow() and popup.screen() is not screen:
        popup.setScreen(screen)   # else Qt may clamp it back onto its old screen
    popup.setGeometry(geo)
    return geo


def popup_menu(menu: QMenu, anchor: QWidget | QRect | QPoint, align_right: bool = False) -> QRect:
    """Shows `menu` through popup_geometry (a long menu is capped to the room
    and scrolls -- Qt's own menu scrolling)."""
    menu.ensurePolished()
    hint = menu.sizeHint()
    screen = screen_for(anchor)
    geo = popup_geometry(anchor_rect(anchor), hint, screen.availableGeometry(), align_right=align_right)
    if menu.screen() is not screen:
        menu.setScreen(screen)
    menu.setMaximumHeight(geo.height() if geo.height() < hint.height() else 16777215)
    menu.popup(geo.topLeft())
    return geo
