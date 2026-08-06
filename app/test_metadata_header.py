"""test_metadata_header.py -- property-based checks for app/metadata_header.

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_metadata_header

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

import cdx
from app.metadata_header import MetadataHeader, format_metadata_text
from app.session import Session

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.metadata_header tests ===")

    # ---- format_metadata_text: dynamical plane -------------------------------------
    print("\nformat_metadata_text (dynamical plane):")
    session = Session()
    session.map = cdx.RationalMap.mandelbrot()
    session.param = complex(-0.7269, 0.1889)
    session.set_render_mode("julia")
    text = format_metadata_text(session)

    check("mandelbrot" in text, "the map's own name appears")
    check("z^2 + a" in text, "the map's to_formula() output appears")
    check("Dynamical plane" in text, "julia mode is reported as the dynamical plane")
    check("Parameter plane" not in text, "and NOT also as the parameter plane")
    check("-0.7269" in text and "0.1889" in text,
          "the actual bound parameter value appears on a dynamical-plane mode")
    check("mode: julia" in text, "the render mode itself is named")

    # ---- format_metadata_text: parameter plane --------------------------------------
    print("\nformat_metadata_text (parameter plane):")
    session.set_render_mode("parameter")
    text_param = format_metadata_text(session)
    check("Parameter plane" in text_param, "parameter mode is reported as the parameter plane")
    check("Dynamical plane" not in text_param, "and NOT also as the dynamical plane")
    check("-0.7269" not in text_param,
          "the bound parameter's VALUE is not shown as if it mattered -- render_parameter "
          "ignores it entirely (every pixel IS a parameter), so showing it would be misleading")
    check("no effect" in text_param,
          "the text explains explicitly that the bound parameter has no effect here, "
          "not just silently omitting it without explanation")

    session.set_render_mode("parameter_greens")
    check("Parameter plane" in format_metadata_text(session),
          "parameter_greens is also reported as the parameter plane (the other of the two "
          "PARAMETER_PLANE_MODES)")

    # ---- format_metadata_text: name/formula track the CURRENT map, not a stale one --
    print("\nformat_metadata_text tracks the current map:")
    session.map = cdx.RationalMap.newton_cubic()
    text_newton = format_metadata_text(session)
    check("newton3" in text_newton, "renaming/replacing the map is reflected immediately")
    check("mandelbrot" not in text_newton, "the OLD map's name is gone, not left stale")

    # A map with no terms at all: to_formula() itself already returns the
    # string "0" (a genuinely correct formula for a map that always
    # evaluates to zero -- confirmed directly, not assumed), never an
    # empty string -- so the "(empty map)" fallback (matching
    # term_editor_panel.py's own identical pattern) is defensive but never
    # actually reachable through to_formula() as it exists today. Checked
    # for what ACTUALLY happens, not for the unreachable fallback text.
    empty = Session()
    empty.map = cdx.RationalMap("scratch")
    check("scratch: 0" in format_metadata_text(empty),
          "a map with no terms shows its own genuinely correct formula, '0' -- not a blank "
          "or broken string")

    # ---- MetadataHeader widget: refresh() actually updates the displayed label ------
    print("\nMetadataHeader widget:")
    session2 = Session()
    session2.map = cdx.RationalMap.mandelbrot()
    session2.set_render_mode("julia")
    header = MetadataHeader(session2)
    check(header._label.text() == format_metadata_text(session2),
          "the widget's initial text matches format_metadata_text on construction")

    session2.map = cdx.RationalMap.newton_cubic()
    check("mandelbrot" in header._label.text(),
          "sanity: the widget does NOT auto-update just because session.map changed -- "
          "refresh() must be called explicitly (same pattern every other panel in this app "
          "already uses)")
    header.refresh()
    check("newton3" in header._label.text() and "mandelbrot" not in header._label.text(),
          "calling refresh() picks up the new map")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
