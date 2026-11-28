"""test_family_builder.py -- app/family_builder.py and the Build popover.

  - Every builder combination (every polynomial type x every map, with
    numeric and symbolic alpha/beta/lambda) produces an expression cdx parses.
  - Degrees match theory: Newton on a degree-n polynomial with simple roots
    has degree n; Halley has degree 2n - 1 (generically) -- for Hermite,
    Legendre, Chebyshev T and Laguerre.
  - The four presets are in the Library, load, are the builder's own
    expressions, and their thumbnails are non-uniform.
  - The popover: preview, expression text, stats, Apply, Save as family.

Run (from the repository root):
    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m app.test_family_builder
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtWidgets import QApplication

import cdx

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])
    from app import family_builder as fb
    from app import theme
    theme.apply(app)

    print("=== app.family_builder tests ===")

    # ---- every combination parses ---------------------------------------------------------
    print("\nevery builder combination parses in cdx:")
    bad = []
    count = 0
    symbol_sets = [("a", "0", "1"), ("0.5", "-0.25", "a"), ("2", "a", "0.75")]
    for t in fb.POLY_TYPES:
        for m in fb.MAP_KINDS:
            for n in (1, 3, 8):
                for alpha, beta, lam in (symbol_sets if t.symbols else symbol_sets[:1]):
                    c = fb.BuilderChoice(poly=t.key, degree=n, alpha=alpha, beta=beta, lam=lam,
                                         custom="z^3 - 2*z + 2", map_kind=m.key)
                    expr = fb.map_expression(c)
                    count += 1
                    try:
                        params = cdx.parse_rational_parameters(expr)
                        if any(p != "a" for p in params):
                            bad.append((expr, f"parameters {params}"))
                    except ValueError as e:
                        bad.append((expr, str(e)))
    check(not bad, f"{count} combinations, all accepted" + (f": {bad[:3]}" if bad else ""))
    c = fb.BuilderChoice(poly="hermite", degree=5, map_kind="relaxed_newton")
    check(fb.map_expression(c) == "z - a*hermite(5,z)/d(hermite(5,z))",
          "relaxed Newton on H5 is z - a*hermite(5,z)/d(hermite(5,z))")
    check(fb.family_name(c) == "Relaxed Newton · Hermite H₅", f"generated name: {fb.family_name(c)}")
    check(fb.family_name(fb.BuilderChoice(poly="gen_laguerre", degree=4, alpha="a", map_kind="halley"))
          == "Halley · Laguerre L₄^(a)", "generalized Laguerre's name carries alpha")
    try:
        fb.build(fb.BuilderChoice(poly="custom", custom="z^3 + b", map_kind="newton"))
        check(False, "a custom p with a parameter other than a is refused")
    except fb.BuilderError as e:
        check("only the parameter a" in str(e), f"a custom p with another parameter is refused ({e})")
    try:
        fb.build(fb.BuilderChoice(poly="gen_laguerre", alpha="z", map_kind="newton"))
        check(False, "alpha = z is refused")
    except fb.BuilderError as e:
        check("must not depend on z" in str(e), f"alpha = z is refused with cdx's message ({e})")
    check(fb.unique_name("X", ["X", "X (2)"]) == "X (3)", "generated names are made unique")

    # ---- degrees match theory ---------------------------------------------------------------
    print("\ndegrees match theory:")
    for poly in ("hermite", "legendre", "chebyshev_t", "laguerre"):
        newton_ok, halley_ok, details = True, True, []
        for n in range(2, 9):
            dn = fb.build(fb.BuilderChoice(poly=poly, degree=n, map_kind="newton")).degree
            dh = fb.build(fb.BuilderChoice(poly=poly, degree=n, map_kind="halley")).degree
            newton_ok &= dn == n
            halley_ok &= dh == 2 * n - 1
            if dn != n or dh != 2 * n - 1:
                details.append((n, dn, dh))
        check(newton_ok, f"Newton on {poly}(n), n = 2..8, has degree n" + (f" {details}" if details else ""))
        check(halley_ok, f"Halley on {poly}(n), n = 2..8, has degree 2n - 1" + (f" {details}" if details else ""))
    b = fb.build(fb.BuilderChoice(poly="hermite", degree=4, map_kind="relaxed_newton"))
    check(b.degree == 4 and b.free_critical_points == 6,
          f"relaxed Newton on H4: degree 4, 2n - 2 = 6 free critical points ({b.degree}, {b.free_critical_points})")

    # ---- presets -----------------------------------------------------------------------------
    print("\npresets:")
    from app.session import PRESET_FAMILY_NAMES, Session
    from app.settings import Settings
    session = Session()
    names = session.library.names()
    want = ["Newton · Legendre P₅", "Relaxed Newton · Hermite H₄", "Newton · Chebyshev T₆",
            "Halley · Laguerre L₄"]
    check(fb.preset_names() == want, f"the four presets: {fb.preset_names()}")
    check(all(n in names and n in PRESET_FAMILY_NAMES for n in want),
          "they are read-only presets in the Library")
    from app.family_views import parameter_view_for
    check(parameter_view_for("Relaxed Newton · Hermite H₄") == (complex(1, 0), 1.25),
          "relaxed Newton · H4 has its own parameter framing (around a = 1)")
    from app.sandbox import render_family_thumbnail
    for p, name in zip(fb.PRESETS, want):
        session.load_from_library(name)
        check(session.map.name == name and session.map.to_formula() == fb.map_expression(p.choice),
              f"{name} loads, and is the builder's expression")
        img, source = render_family_thumbnail(name, session.map, Settings(), session.render_settings,
                                              resolution=48)
        arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
        check(int(arr.max()) - int(arr.min()) > 0 and len(np.unique(arr.reshape(-1, 4)[:, :3], axis=0)) > 4,
              f"{name}: thumbnail ({source} plane) is non-uniform")

    # ---- the popover -------------------------------------------------------------------------
    print("\nBuild popover:")
    from app.equation_panel import EquationPanel
    session = Session()
    saved: list[str] = []

    def save_family(name: str) -> str | None:
        session.save_to_library(name)
        saved.append(name)
        return name

    panel = EquationPanel(session, lambda: None)
    panel.set_save_family_handler(save_family)
    pop = panel.builder()
    pop.set_choice(fb.BuilderChoice(poly="legendre", degree=5, map_kind="halley"))
    expr = fb.map_expression(fb.BuilderChoice(poly="legendre", degree=5, map_kind="halley"))
    check(pop.expression() == expr and pop.expression_label.text() == expr,
          "the popover shows the expression it will put in the field")
    check(pop.preview.source() == expr and not pop.preview.is_greyed(), "...and the typeset map")
    check("degree 9" in pop.stats_label.text() and "free critical" in pop.stats_label.text(),
          f"...and degree / free critical points: {pop.stats_label.text()}")
    pop.type_combo.setCurrentIndex(pop.type_combo.findData("jacobi"))
    check(pop.symbol_fields["alpha"].isVisibleTo(pop) and pop.symbol_fields["beta"].isVisibleTo(pop)
          and not pop.symbol_fields["lam"].isVisibleTo(pop), "Jacobi shows the α and β fields only")
    pop.symbol_fields["alpha"].setText("z")
    pop.recompute()
    check(not pop.apply_button.isEnabled() and "depend on z" in pop.error_label.text(),
          "an invalid α disables Apply and shows why")
    pop.set_choice(fb.BuilderChoice(poly="hermite", degree=30, map_kind="newton"))
    check("degree above 25" in pop.error_label.text() and pop.apply_button.isEnabled(),
          "degree 30 shows the parser's warning but can still be applied")
    pop.set_choice(fb.BuilderChoice(poly="hermite", degree=5, map_kind="relaxed_newton"))
    check(pop.apply() and session.map.to_formula() == "z - a*hermite(5,z)/d(hermite(5,z))"
          and panel._formula_edit.text() == session.map.to_formula(),
          "Apply fills the f(z) field and applies it")
    check(session.map.pq_active_param == "a", "...with a as the parameter-plane parameter")
    pop.set_choice(fb.BuilderChoice(poly="chebyshev_t", degree=4, map_kind="halley"))
    check(pop.save() and saved == ["Halley · Chebyshev T₄"]
          and session.library.find("Halley · Chebyshev T₄") is not None,
          "Save as family applies and adds it to the Library under the generated name")

    # In the real window, Save as family goes through the Library panel (which
    # stores the parameter-plane framing and makes the name unique).
    # Kept off the real ~/.complexdynamics exactly as app/test_sandbox.py does.
    import tempfile
    from pathlib import Path
    from app import sandbox as sandbox_module
    from app import settings as settings_module
    sandbox_module.save_settings = lambda settings: None
    fake_library = tempfile.mktemp(suffix="-library.txt")
    sandbox_module.library_path = lambda: fake_library
    fake_config = Path(tempfile.mkdtemp(suffix="-config"))
    settings_module.config_dir = lambda: fake_config
    window = sandbox_module.SandboxWindow()
    before = set(window.session.library.names())
    pop = window.equation_panel.builder()
    pop.set_choice(fb.BuilderChoice(poly="legendre", degree=3, map_kind="relaxed_newton"))
    ok1 = pop.save()
    ok2 = pop.save()
    added = sorted(set(window.session.library.names()) - before)
    check(ok1 and ok2 and added == ["Relaxed Newton · Legendre P₃", "Relaxed Newton · Legendre P₃ (2)"],
          f"in the window, saving twice gives unique names: {added}")
    from app.family_views import stored_parameter_view
    check(stored_parameter_view("Relaxed Newton · Legendre P₃") is not None,
          "...and the saved family keeps its parameter framing")
    check(window.top_bar.map_chip.name_label.text() == "Relaxed Newton · Legendre P₃ (2)",
          "the chip names the saved family (not \"custom\")")
    window.close()

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
