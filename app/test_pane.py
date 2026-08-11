"""test_pane.py -- property-based checks for app/pane.Pane.

Plain Python + one cdx type, no Qt -- offscreen-safe without any display
plugin at all. Run with:

    PYTHONPATH=cdx/build python -m app.test_pane

(from the repository root).
"""

from __future__ import annotations

import cdx
from app.pane import Pane

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    print("=== app.pane tests ===")

    print("\nconstruction:")
    vp = cdx.Viewport(complex(-0.5, 0.0), 1.5, 400)
    p = Pane(vp, "parameter")
    check(p.viewport is vp, "stores the given viewport")
    check(p.render_mode == "parameter", "stores the given render_mode")
    check(p.image_view is None, "image_view starts unset -- assigned by the caller after "
                                "constructing ImageView(pane, session, ...)")
    check(p.request_id == 0, "request_id starts at 0")
    check(p.pending_tasks == {}, "pending_tasks starts empty")

    print("\nset_render_mode:")
    p.set_render_mode("julia")
    check(p.render_mode == "julia", "a valid mode is accepted and stored")
    try:
        p.set_render_mode("not_a_real_mode")
        check(False, "an invalid mode should have raised")
    except ValueError:
        check(True, "an invalid mode raises ValueError, the same guard Session.set_render_mode "
                    "used to provide before render_mode moved here")
    check(p.render_mode == "julia",
          "a REJECTED set_render_mode call leaves the pane's mode exactly where it was")

    print("\nindependence between two panes:")
    p2 = Pane(cdx.Viewport(complex(0, 0), 2.0, 200), "greens")
    p.viewport = cdx.Viewport(complex(1, 1), 0.5, 100)
    check(p2.viewport.center == complex(0, 0) and p2.viewport.scale == 2.0,
          "mutating one pane's viewport never touches a second, independently-constructed pane")
    p.request_id = 5
    check(p2.request_id == 0, "request_id bookkeeping is per-instance, not shared")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
