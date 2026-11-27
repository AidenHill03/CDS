"""test_label_cuts.py -- the Per-critical strategy's UX.

  - app.label_cuts finds the collision parameters: at each one two tracked
    free critical points really do meet (z^3 + a z + 1: exactly a = 0;
    z^4 + a z^2 + z: the three cube roots of -27/8).
  - The labels really are re-assigned across each ray and nowhere else
    nearby: just either side of the ray, critical_point_labels swaps; just
    either side of the segment from the collision back toward a*, it does not.
  - In the window: the plane header shows a non-default strategy
    ("Escape · per-critical #1") and nothing for the default; the overlay's
    segments lie on the rays through the computed collisions, pointing away
    from a*; hovering a cut finds it (the tooltip), hovering elsewhere
    doesn't; the Rendering tab's note says it's a labelling convention.
  - Loading a preset resets the strategy to the default; a family saved
    with a non-default strategy restores it on load.

Run (from the repository root):
    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m app.test_label_cuts
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

import cdx

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def cross(u: complex, v: complex) -> float:
    return u.real * v.imag - u.imag * v.real


def labels(rmap, a):
    return [complex(z) for z in rmap.critical_point_labels(a)]


def same_order(p, q, tol=1e-3) -> bool:
    return len(p) == len(q) and all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in zip(p, q))


def main() -> None:
    app = QApplication.instance() or QApplication([])
    from app import label_cuts as lc
    from app import theme
    theme.apply(app)

    print("=== Per-critical labelling cuts ===")

    # ---- the collisions ------------------------------------------------------------------
    print("\ncollision parameters:")
    cubic = cdx.RationalMap.from_expression("z^3 + a*z + 1", "a", {}, "cubic")
    cuts = lc.compute_label_cuts(cubic)
    check(cuts is not None and len(cuts.collisions) == 1 and abs(cuts.collisions[0]) < 1e-9,
          f"z^3 + a z + 1: one collision, at a = 0 ({cuts and cuts.collisions})")
    check(cuts is not None and abs(cuts.a_star - complex(0.618, 0.382)) < 1e-12,
          "a* is the convention's base parameter")

    quartic = cdx.RationalMap.from_expression("z^4 + a*z^2 + z", "a", {}, "quartic")
    qcuts = lc.compute_label_cuts(quartic)
    # f' = 4z^3 + 2az + 1 has a double root iff its discriminant
    # -4*4*(2a)^3 - 27*4^2 = -128 a^3 - 432 vanishes: a^3 = -27/8.
    want = (complex(-1.5, 0), 1.5 * complex(0.5, 3 ** 0.5 / 2), 1.5 * complex(0.5, -(3 ** 0.5) / 2))
    got = list(qcuts.collisions) if qcuts else []
    check(len(got) == 3 and all(min(abs(g - w) for g in got) < 1e-8 for w in want),
          f"z^4 + a z^2 + z: the three roots of a^3 = -27/8 ({got})")
    for c in got:
        pts = lc._tracked_points(quartic, c)
        sep = min((abs(p - q) for i, p in enumerate(pts) for q in pts[i + 1:]), default=0.0)
        check(len(pts) < 3 or sep < 1e-3, f"  two critical points meet at a = {c:.4f}")

    check(lc.compute_label_cuts(cdx.RationalMap.mandelbrot()) is None,
          "z^2 + a has nothing to cut (one critical point, constant)")

    # ---- the labels swap across each ray, and only there ----------------------------------
    print("\nlabels across the rays:")
    for name, rmap, cc in (("cubic", cubic, cuts), ("quartic", quartic, qcuts)):
        for c, d in zip(cc.collisions, cc.directions()):
            n = 1j * d                                  # normal to the ray
            span = abs(c - cc.a_star)
            on_ray = c + 0.4 * span * d                 # beyond the collision
            off_ray = c - 0.4 * span * d                # between a* and it
            eps = 1e-4 * span
            swapped = not same_order(labels(rmap, on_ray + eps * n), labels(rmap, on_ray - eps * n))
            kept = same_order(labels(rmap, off_ray + eps * n), labels(rmap, off_ray - eps * n))
            check(swapped and kept,
                  f"{name}, collision {c:.3f}: labels swap across the ray (and not before it)")

    # ---- the window --------------------------------------------------------------------------
    from app import sandbox as sandbox_module
    from app import settings as settings_module
    sandbox_module.save_settings = lambda settings: None
    fake_library = tempfile.mktemp(suffix="-library.txt")
    sandbox_module.library_path = lambda: fake_library
    fake_config = Path(tempfile.mkdtemp(suffix="-config"))
    settings_module.config_dir = lambda: fake_config
    window = sandbox_module.SandboxWindow()
    window.resize(1200, 800)
    window.show()
    app.processEvents()

    param_pane = next(p for p in window.panes if p.render_mode == "parameter")
    frame = window._plane_frames[param_pane]
    view = param_pane.image_view

    print("\nheader:")
    check(frame.strategy_text() == "", "the default strategy shows nothing in the header")
    window.equation_panel.apply_expression("z^3 + a*z + 1")
    app.processEvents()
    pc = dataclasses.replace(window.session.settings, parameter_strategy="per_critical",
                             parameter_critical_index=1)
    window._on_settings_applied(pc)
    app.processEvents()
    check(frame.strategy_text() == "Escape · per-critical #1",
          f"Per-critical #1 shows in the parameter header: {frame.strategy_text()!r}")
    check("labelling convention" in frame.strategy_label.toolTip(),
          "...with a tooltip explaining the cuts")
    window._on_settings_applied(dataclasses.replace(pc, parameter_strategy="fastest"))
    app.processEvents()
    check(frame.strategy_text() == "Escape · fastest", f"Fastest shows too: {frame.strategy_text()!r}")
    check(not view.label_cut_segments(), "...and no cuts are drawn for it")
    window._on_settings_applied(pc)
    app.processEvents()
    window._on_mode_changed(param_pane, "parameter_greens")
    app.processEvents()
    check(frame.strategy_text() == "", "a mode the strategy doesn't affect (Green's) shows nothing")
    check(not view.label_cut_segments(), "...and draws no cuts")
    window._on_mode_changed(param_pane, "parameter")
    app.processEvents()
    check(frame.strategy_text() == "Escape · per-critical #1", "back on Escape it shows again")

    print("\noverlay:")
    param_pane.viewport = cdx.Viewport(0j, 1.5, param_pane.viewport.resolution)
    view.update()
    app.processEvents()
    segs = view.label_cut_segments()
    a_star = lc.label_cuts(window.session.map).a_star
    check(len(segs) == 1, f"one cut crosses the view ({len(segs)})")
    for c, s0, s1 in segs:
        d = (c - a_star) / abs(c - a_star)
        on_line = abs(cross(s0 - c, d)) < 1e-9 and abs(cross(s1 - c, d)) < 1e-9
        outward = ((s0 - c) * d.conjugate()).real >= -1e-12 and ((s1 - c) * d.conjugate()).real > 0
        check(on_line and outward,
              f"the drawn segment lies on the ray through the collision {c:.2g}, away from a*")
        check(abs(s0 - c) < 1e-12, "...and starts at the collision (it's inside the view)")
        mid = view._complex_to_display_pixel((s0 + s1) / 2)
        check(view.label_cut_at(mid), "hovering the cut finds it (the tooltip shows)")
        away = view._complex_to_display_pixel(c - 0.5 * d)
        check(not view.label_cut_at(away), "hovering between a* and the collision does not")
    check(lc.TOOLTIP.endswith("not dynamics."), "the tooltip calls it a labelling convention")
    grab = view.grab().toImage()
    check(not grab.isNull(), "the pane paints with the overlay on")

    print("\nRendering note:")
    note = window.rendering_panel.field_note("parameter_strategy")
    check("labelling convention" in note and "not dynamics" in note,
          f"a note under the strategy setting: {note!r}")

    print("\nfamily loads:")
    window.library_panel.load_family("mandelbrot")
    app.processEvents()
    s = window.session.settings
    check((s.parameter_strategy, s.parameter_critical_index) == ("slowest", 0),
          f"loading a preset resets the strategy to the default ({s.parameter_strategy}, "
          f"{s.parameter_critical_index})")
    check(frame.strategy_text() == "", "...and the header clears")
    check(window.rendering_panel._widgets["parameter_strategy"].currentText() == "slowest",
          "...and the Rendering tab shows it")

    window.equation_panel.apply_expression("z^3 + a*z + 1")
    window._on_settings_applied(pc)
    app.processEvents()
    saved = window.library_panel.save_as("cubic per-critical")
    window.library_panel.load_family("Relaxed Newton · Hermite H₄")
    app.processEvents()
    check(window.session.settings.parameter_strategy == "slowest",
          "loading another family after a Per-critical session resets it")
    window.library_panel.load_family(saved)
    app.processEvents()
    s = window.session.settings
    check((s.parameter_strategy, s.parameter_critical_index) == ("per_critical", 1),
          "a family saved with Per-critical #1 restores it on load")
    check(frame.strategy_text() == "Escape · per-critical #1", "...and the header says so")

    window.close()
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
