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
from app.metadata_header import MetadataHeader, describe_parameter_role, format_metadata_text
from app.pane import Pane
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
    # render_mode is the CALLER's own now (see app.pane.Pane) -- passed to
    # format_metadata_text directly as a plain string; MetadataHeader
    # itself (below) takes a real Pane, the same as app/sandbox.py does.
    print("\nformat_metadata_text (dynamical plane):")
    session = Session()
    session.map = cdx.RationalMap.mandelbrot()
    session.param = complex(-0.7269, 0.1889)
    text = format_metadata_text(session, "julia")

    check("mandelbrot" in text, "the map's own name appears")
    check("z^2 + a" in text, "the map's to_formula() output appears")
    check("Dynamical plane" in text, "julia mode is reported as the dynamical plane")
    check("Parameter plane" not in text, "and NOT also as the parameter plane")
    check("-0.7269" in text and "0.1889" in text,
          "the actual bound parameter value appears on a dynamical-plane mode")
    check("mode: julia" in text, "the render mode itself is named")

    # ---- format_metadata_text: parameter plane --------------------------------------
    print("\nformat_metadata_text (parameter plane):")
    text_param = format_metadata_text(session, "parameter")
    check("Parameter plane" in text_param, "parameter mode is reported as the parameter plane")
    check("Dynamical plane" not in text_param, "and NOT also as the dynamical plane")
    check("-0.7269" not in text_param,
          "the bound parameter's VALUE is not shown as if it mattered -- render_parameter "
          "ignores it entirely (every pixel IS a parameter), so showing it would be misleading")
    check("no effect" in text_param,
          "the text explains explicitly that the bound parameter has no effect here, "
          "not just silently omitting it without explanation")

    check("Parameter plane" in format_metadata_text(session, "parameter_greens"),
          "parameter_greens is also reported as the parameter plane (the other of the two "
          "PARAMETER_PLANE_MODES)")

    # ---- format_metadata_text: name/formula track the CURRENT map, not a stale one --
    print("\nformat_metadata_text tracks the current map:")
    session.map = cdx.RationalMap.newton_cubic()
    text_newton = format_metadata_text(session, "julia")
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
    check("scratch: 0" in format_metadata_text(empty, "julia"),
          "a map with no terms shows its own genuinely correct formula, '0' -- not a blank "
          "or broken string")

    # ---- describe_parameter_role: read from the terms, never hardcoded --------------
    print("\ndescribe_parameter_role (P6 section 2 -- `a` is not assumed to be additive):")
    check(describe_parameter_role(cdx.RationalMap.mandelbrot()) == "coefficient of z^0",
          "mandelbrot(): a is the coefficient of z^0 (param_power=1, exponent=0), matching "
          "the classic z^2+a reading exactly")
    check(describe_parameter_role(cdx.RationalMap.multibrot(5)) == "coefficient of z^0",
          "multibrot(5): same additive-constant role, different exponent on the OTHER term")

    mcmullen_role = describe_parameter_role(cdx.RationalMap.mcmullen(2))
    check("strength" in mcmullen_role and "location" not in mcmullen_role,
          "mcmullen(2): a multiplies the pole's STRENGTH (param_power=1), never its location "
          "-- confirmed via the actual term flags, not assumed from the family name")

    check(describe_parameter_role(cdx.RationalMap.newton_cubic()) == "unused by this map",
          "newton_cubic(): a genuinely has NO effect on this map (no term has param_power != 0 "
          "or location_is_param) -- confirmed directly against the preset's own C++ "
          "construction, not assumed just because it's an unusual case")

    # A hand-built map exercising the ONE combination none of the built-in
    # presets do: a pole whose LOCATION tracks `a` directly.
    tracking = cdx.RationalMap("tracking")
    tracking.add_pole(complex(0, 0), complex(1, 0), 1, 0, "tracker")
    tracking.pole_terms()[0].location_is_param = True
    check(describe_parameter_role(tracking) == "location of tracker",
          "location_is_param produces 'location of <label>' -- the one PoleTerm combination "
          "no built-in preset exercises, checked directly rather than left implicit")

    # A pole with BOTH location_is_param AND a nonzero param_power at once --
    # both roles must be reported, not just one silently winning.
    both = cdx.RationalMap("both")
    both.add_pole(complex(0, 0), complex(1, 0), 1, 2, "p")
    both.pole_terms()[0].location_is_param = True
    both_role = describe_parameter_role(both)
    check("location of p" in both_role and "strength of p" in both_role,
          "a pole term can depend on `a` in BOTH ways at once (location AND strength) -- "
          "both are reported, neither shadows the other")

    disabled = cdx.RationalMap("disabled")
    disabled.add_poly(complex(1, 0), 0, 1, "a")
    disabled.poly_terms()[0].enabled = False
    check(describe_parameter_role(disabled) == "unused by this map",
          "a DISABLED term's param dependence doesn't count -- it has no effect on the map "
          "as actually evaluated, matching effective_coeff/RationalMap::eval's own enabled check")

    # ---- MetadataHeader widget: refresh() actually updates the displayed label ------
    print("\nMetadataHeader widget:")
    session2 = Session()
    session2.map = cdx.RationalMap.mandelbrot()
    pane2 = Pane(cdx.Viewport(complex(0, 0), 1.5, 100), "julia")
    header = MetadataHeader(session2, pane2)
    check(header._label.text() == format_metadata_text(session2, pane2.render_mode),
          "the widget's initial text matches format_metadata_text on construction")

    session2.map = cdx.RationalMap.newton_cubic()
    check("mandelbrot" in header._label.text(),
          "sanity: the widget does NOT auto-update just because session.map changed -- "
          "refresh() must be called explicitly (same pattern every other panel in this app "
          "already uses)")
    header.refresh()
    check("newton3" in header._label.text() and "mandelbrot" not in header._label.text(),
          "calling refresh() picks up the new map")

    pane2.render_mode = "parameter"
    header.refresh()
    check("Parameter plane" in header._label.text(),
          "refresh() also picks up a render_mode change on the SAME pane object -- the "
          "header reads pane.render_mode live, not a value captured once at construction")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
