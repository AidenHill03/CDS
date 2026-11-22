"""test_popups.py -- screen-aware popups (app/popups.py) and the layout fixes
that came with them.

Runs on Qt's offscreen platform with a CONFIGURED screen layout of known
size -- a 1100x720 screen and an 800x600 one to its right -- so every
placement is checked against real QScreen.availableGeometry() values:

  - popup_geometry(): below when it fits, above otherwise, capped (and
    scrolling) when neither side fits; clamped horizontally; always inside
    the available rect.
  - The Build popover, a Library row's ⋯ menu and the formula tooltip, each
    opened from an anchor near the bottom edge, the right edge, the top and
    the middle of the screen, and on the second monitor: the popup lies
    inside that screen's availableGeometry, and Apply / Save as family are
    visible.
  - The Equation tab's preview is never clipped by the dock; the Library's
    section header comes into view with its first row.

Run (from the repository root; it sets its own QT_QPA_PLATFORM):

    PYTHONPATH=cdx/build python -m app.test_popups
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SCREENS = [("A", 0, 0, 1100, 720), ("B", 1100, 0, 800, 600)]


def _configure_offscreen() -> None:
    cfg = {"synchronousWindowSystemEvents": True, "windowFrameMargins": False,
           "screens": [{"name": n, "x": x, "y": y, "width": w, "height": h, "logicalDpi": 96,
                        "logicalBaseDpi": 96, "dpr": 1} for n, x, y, w, h in SCREENS]}
    path = Path(tempfile.mkdtemp()) / "screens.json"
    path.write_text(json.dumps(cfg))
    os.environ["QT_QPA_PLATFORM"] = f"offscreen:configfile={path}"
    if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
        import PySide6
        plugins = Path(PySide6.__file__).parent / "Qt" / "plugins" / "platforms"
        if plugins.is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins)


_configure_offscreen()

from PySide6.QtCore import QPoint, QRect, QSize, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def inside(inner: QRect, outer: QRect) -> bool:
    return (inner.left() >= outer.left() and inner.top() >= outer.top()
            and inner.left() + inner.width() <= outer.left() + outer.width()
            and inner.top() + inner.height() <= outer.top() + outer.height())


def fmt(r: QRect) -> str:
    return f"({r.x()},{r.y()} {r.width()}x{r.height()})"


def settle(app) -> None:
    for _ in range(6):
        app.processEvents()


def main() -> None:
    app = QApplication.instance() or QApplication([])
    from app import popups, theme
    theme.apply(app)

    print("=== app.popups tests ===")
    names = [s.name() for s in app.screens()]
    check(names == ["A", "B"] and app.screens()[0].availableGeometry() == QRect(0, 0, 1100, 720),
          f"offscreen screens configured: {[(s.name(), fmt(s.availableGeometry())) for s in app.screens()]}")

    # ---- pure geometry ---------------------------------------------------------------------------
    print("\npopup_geometry:")
    avail = QRect(0, 25, 1100, 640)            # e.g. under a menu bar, above a Dock
    size = QSize(400, 300)
    g = popup_geometry = popups.popup_geometry
    r = g(QRect(100, 100, 80, 30), size, avail)
    check(r.top() == 134 and r.height() == 300 and inside(r, avail), f"room below: opens below {fmt(r)}")
    r = g(QRect(100, 600, 80, 30), size, avail)
    check(r.top() + r.height() == 596 and inside(r, avail), f"no room below: opens above {fmt(r)}")
    r = g(QRect(1050, 100, 40, 30), size, avail)
    check(r.left() == 700 and inside(r, avail), f"right edge: clamped horizontally {fmt(r)}")
    r = g(QRect(100, 30, 80, 30), size, avail)
    check(r.top() >= avail.top() and inside(r, avail), f"top edge: below, never under the menu bar {fmt(r)}")
    r = g(QRect(100, 300, 80, 30), QSize(400, 900), avail, min_height=200)
    check(inside(r, avail) and r.height() < 900 and r.height() >= 200,
          f"taller than either side: capped to the larger side {fmt(r)}")
    r = g(QRect(100, 300, 80, 30), QSize(400, 900), QRect(0, 0, 1100, 320), min_height=600)
    check(inside(r, QRect(0, 0, 1100, 320)), f"taller than the screen: fits the screen {fmt(r)}")
    check(popups.screen_for(QRect(1895, 304, 16, 16)).name() == "B",
          "an anchor hanging past a monitor's edge belongs to that monitor, not the primary")
    r = g(QRect(1500, 550, 80, 30), size, QRect(1100, 0, 800, 600))
    check(inside(r, QRect(1100, 0, 800, 600)), f"second monitor: stays on it {fmt(r)}")
    del popup_geometry

    # ---- the Build popover ---------------------------------------------------------------------
    print("\nBuild popover (anchored to Build ▾):")
    from app.equation_panel import EquationPanel
    from app.session import Session
    host = QWidget(None, Qt.WindowType.Window)
    host.setWindowTitle("host")
    lay = QVBoxLayout(host)
    lay.setContentsMargins(8, 8, 8, 8)
    panel = EquationPanel(Session(), lambda: None)
    panel.set_save_family_handler(lambda name: name)
    lay.addWidget(panel)
    host.resize(760, 150)
    host.show()
    settle(app)
    pop = panel.builder()
    button = panel.build_button()

    placements = {
        "top-left": QPoint(0, 0),
        "bottom edge": QPoint(100, 720 - 150),
        "right edge": QPoint(1100 - 760, 300),
        "bottom-right corner": QPoint(1100 - 760, 720 - 150),
        "middle": QPoint(150, 280),
        "second monitor, bottom": QPoint(1100 + 20, 600 - 150),
    }
    for label, pos in placements.items():
        host.move(pos)
        settle(app)
        panel.open_builder()
        settle(app)
        screen = popups.screen_for(button)
        av = screen.availableGeometry()
        geo = pop.geometry()
        apply_rect = QRect(pop.apply_button.mapToGlobal(QPoint(0, 0)), pop.apply_button.size())
        save_rect = QRect(pop.save_button.mapToGlobal(QPoint(0, 0)), pop.save_button.size())
        btn = popups.anchor_rect(button)
        above = geo.top() + geo.height() <= btn.top()
        below = geo.top() >= btn.top() + btn.height()
        check(inside(geo, av) and pop.apply_button.isVisible() and inside(apply_rect, geo)
              and inside(save_rect, geo) and (above or below),
              f"{label}: {fmt(geo)} inside {screen.name()} {fmt(av)}, "
              f"{'above' if above else 'below' if below else 'OVERLAPS'} the button, Apply/Save visible")
        if geo.height() < pop.natural_size().height():
            bar = pop.scroll.verticalScrollBar()
            check(bar.maximum() > 0, f"{label}: capped height -> the form scrolls (range {bar.maximum()})")
        pop.hide()
    host.hide()

    # ---- a Library row's ⋯ menu ----------------------------------------------------------------
    print("\nLibrary row ⋯ menu:")
    from app.library_panel import LibraryPanel
    lib_host = QWidget(None, Qt.WindowType.Window)
    ll = QVBoxLayout(lib_host)
    lib_panel = LibraryPanel(Session(), lambda: None, lambda: None, lib_host)
    ll.addWidget(lib_panel)
    lib_host.resize(320, 700)
    lib_host.show()
    settle(app)
    row = lib_panel._rows["mandelbrot"]
    btn_off = row.menu_button.mapTo(lib_host, QPoint(0, 0))
    btn_h = row.menu_button.height()
    # host positions that put the ⋯ button itself at each edge
    for label, pos, upward in [
            ("top", QPoint(400, 0), False),
            ("bottom edge", QPoint(300, 716 - btn_off.y() - btn_h), True),
            ("right edge", QPoint(1100 - btn_off.x() - row.menu_button.width() - 2, 200), False),
            ("second monitor, bottom", QPoint(1300, 596 - btn_off.y() - btn_h), True)]:
        lib_host.move(pos)
        settle(app)
        geo = popups.popup_menu(row.menu, row.menu_button, align_right=True)
        settle(app)
        av = popups.available_rect(row.menu_button)
        shown = row.menu.geometry()
        btn = popups.anchor_rect(row.menu_button)
        went_up = shown.top() + shown.height() <= btn.top()
        check(inside(geo, av) and inside(shown, av) and went_up == upward,
              f"{label}: ⋯ at {fmt(btn)}, menu {fmt(shown)} {'above' if went_up else 'below'}, "
              f"inside {fmt(av)}")
        row.menu.hide()
    # A row near the very bottom of the screen opens its menu upward.
    btn_rect = QRect(QPoint(1000, 700), QSize(24, 20))
    geo = popups.popup_geometry(btn_rect, row.menu.sizeHint(), QRect(0, 0, 1100, 720), align_right=True)
    check(geo.top() + geo.height() <= 700 and inside(geo, QRect(0, 0, 1100, 720)),
          f"a ⋯ button at the bottom-right opens its menu upward, inside the screen {fmt(geo)}")
    lib_host.hide()

    # ---- the formula tooltip -------------------------------------------------------------------
    print("\nformula tooltip:")
    from app.mathdisplay import FormulaToolTip
    kim = "z^3*(2*(a-1)+a^2 +4*(a-1)*z+2*(a-1)*z^2)/((2*(a-1) + 4*(a-1)*z + (a^2+2*a-2)*z^2)^2)"
    owner = QWidget()
    for label, pt in {"bottom-right corner": QPoint(1095, 715), "top-left": QPoint(2, 2),
                      "second monitor, right edge": QPoint(1895, 300)}.items():
        FormulaToolTip.show_for(owner, pt, kim, "custom: Kim")
        settle(app)
        tip = FormulaToolTip.instance()
        av = popups.available_rect(pt)
        check(inside(tip.geometry(), av), f"{label}: {fmt(tip.geometry())} inside {fmt(av)}")
    FormulaToolTip.hide_for(owner)

    # ---- the Equation dock and the Library header, in the real window -------------------------
    print("\nEquation preview and Library header (1100x720 window):")
    from app import sandbox as sandbox_module
    from app import settings as settings_module
    sandbox_module.save_settings = lambda settings: None
    fake_library = tempfile.mktemp(suffix="-library.txt")
    sandbox_module.library_path = lambda: fake_library
    fake_config = Path(tempfile.mkdtemp(suffix="-config"))
    settings_module.config_dir = lambda: fake_config
    window = sandbox_module.SandboxWindow()
    window.reset_layout()
    window.move(0, 0)
    window.resize(1100, 720)
    window.show()
    settle(app)
    ep = window.equation_panel
    dock = window.settings_dock
    dock.reveal(ep)
    ep.apply_expression("(z^3 + a*z^2 + 2*z - 1)/(3*z^2 - a + (z-1)^2/(z+2))")
    ep.flush_preview()
    settle(app)
    check(dock.minimumHeight() > 0 and dock.minimumHeight() <= 200,
          f"the expanded dock has a minimum height ({dock.minimumHeight()} px)")
    for want in (400, 220, 150, 40):
        window.resizeDocks([dock], [want], Qt.Orientation.Vertical)
        settle(app)
        pv = ep.preview()
        page = dock.stack.currentWidget()
        top = pv.mapTo(page, QPoint(0, 0)).y()
        box = pv.painted_box()
        visible = top >= 0 and top + pv.height() <= page.height() and box.h <= pv.height() + 0.5
        check(visible, f"dock {dock.height()} px: preview {pv.height()} px fully visible "
                       f"({'fitted' if pv.is_fitted() else 'full size'})")
    window.resizeDocks([dock], [260], Qt.Orientation.Vertical)
    settle(app)
    check(not ep.preview().is_fitted(), "with room, the preview is full size")
    dock.set_collapsed(True)
    settle(app)
    check(dock.height() <= dock.header.sizeHint().height() + 2, "the dock still collapses to its tab strip")
    dock.set_collapsed(False)
    settle(app)
    check(dock.height() >= dock.minimumHeight(), "...and expands back to at least its minimum")

    # Library: selecting the first "Yours" row brings the YOURS header with it.
    ep.apply_expression("z^3 + a/z")
    window.library_panel.save_as("Mine one")
    ep.apply_expression("z^4 + a")
    window.library_panel.save_as("Mine two")
    window.sidebar.reveal(window.library_panel)
    settle(app)
    lp = window.library_panel
    lp._list.scrollToTop()
    lp._select_by_name("Mine one")
    settle(app)
    header = lp.section_header_rect("Yours")
    vp = lp._list.viewport()
    check(header is not None and header.top() >= 0 and header.bottom() <= vp.height(),
          f"YOURS header fully visible when its first row is selected ({header})")
    field_bottom = lp.search_field.mapTo(window, QPoint(0, lp.search_field.height())).y()
    vp_top = vp.mapTo(window, QPoint(0, 0)).y()
    check(vp_top - field_bottom >= 6, f"the list's rows start {vp_top - field_bottom} px under the search field")
    window.close()

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
