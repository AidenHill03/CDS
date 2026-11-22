"""snapshot_ui.py -- dev tool: grab PNGs of the window for visual review.

Builds the full app (with any research-module panels that are built)
offscreen at 1389x868 (a 13" MacBook's usable screen) and 1100x720 (the
minimum window size), waits for the renders to finish, and saves
QWidget.grab() images of: both workspaces, each sidebar tab, each dock tab,
Single view, and the collapsed sidebar.

    PYTHONPATH=cdx/build python tools/snapshot_ui.py [--out DIR]

The offscreen platform draws the menu bar inside the window (as Windows
does); on macOS the real menu bar is the system one, outside the window.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SIZES = [(1389, 868), (1100, 720)]


def _prepare_qt_env() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
        import PySide6
        plugins = Path(PySide6.__file__).parent / "Qt" / "plugins" / "platforms"
        if plugins.is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins)


def _settle(app, window, timeout_s: float = 60.0) -> None:
    """Process events until no pane render and no Library thumbnail is in
    flight (or timeout)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        busy = any(stage is not None for stage in window._render_stage.values())
        pending = any(p.pending_tasks for p in window.panes) or window.thumbnailer.pending() > 0
        debouncing = any(t.isActive() for t in window._debounce_timers.values())
        if not busy and not pending and not debouncing:
            break
        time.sleep(0.02)
    for _ in range(5):
        app.processEvents()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO / "phoenix-ui-redesign" / "snapshots"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _prepare_qt_env()
    from PySide6.QtWidgets import QApplication

    from app import theme
    from app.sandbox import SandboxWindow

    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply(app)
    window = SandboxWindow()
    try:
        from phoenix import _module_panels
        for factory in _module_panels():
            widget, title = factory(window)
            window.add_panel(widget, title)
    except Exception as exc:   # a dev tool: carry on with the plain sandbox
        print(f"snapshot_ui: module panels unavailable ({exc})", file=sys.stderr)
    window.show()
    _settle(app, window)

    saved: list[Path] = []

    def grab(name: str) -> None:
        _settle(app, window)
        path = out / f"{name}.png"
        window.grab().save(str(path))
        saved.append(path)

    for w, h in SIZES:
        tag = f"{w}x{h}"
        window.reset_layout()
        window.resize(w, h)
        window.show_workspace("explore")
        _settle(app, window)
        if window.sidebar.is_collapsed():
            grab(f"{tag}-explore-sidebar-collapsed")
        default_dock = window.settings_dock.current_widget()
        for i in range(window.sidebar.count()):
            window.sidebar.reveal(window.sidebar.widget(i))
            slug = window.sidebar.tab_text(i).lower().replace(" ", "-")
            grab(f"{tag}-explore-sidebar-{slug}")
        window.sidebar.reveal(window.library_panel)
        for i in range(window.settings_dock.count()):
            if not window.settings_dock.tab_bar.isTabVisible(i):
                continue
            window.settings_dock.reveal(window.settings_dock.widget(i))
            slug = window.settings_dock.tab_text(i).lower().replace(" ", "-")
            grab(f"{tag}-explore-dock-{slug}")
        window.settings_dock.set_current_widget(default_dock)
        window.coupled_view_action.setChecked(False)
        grab(f"{tag}-explore-single")
        window.coupled_view_action.setChecked(True)
        for key in list(window._workspace_keys):
            if key != "explore":
                window.show_workspace(key)
                grab(f"{tag}-workspace-{key}")
        window.show_workspace("explore")

    for path in saved:
        print(path)
    window.close()


if __name__ == "__main__":
    main()
