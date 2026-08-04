"""test_session.py -- property-based checks for app/session.Session.

Mirrors the C++ suite's style (PASS/FAIL per check, properties rather than
golden output) rather than pulling in a Python test framework this project
does not otherwise use. Run with:

    PYTHONPATH=../cdx/build python test_session.py

(from this directory).
"""

from __future__ import annotations

import math
import os
import tempfile

import cdx
from session import Session

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    print("=== app.session tests ===")

    # ---- defaults --------------------------------------------------------------
    print("\ndefaults:")
    s = Session()
    check(s.map.name == "mandelbrot", "starts on the mandelbrot preset")
    check(s.param == complex(-0.7269, 0.1889),
          "starts with the dendritic Julia parameter, for whenever the user switches to it")
    check(s.render_mode == "parameter", "starts in parameter-plane render mode (the Mandelbrot set)")
    check(len(s.library) >= 6, "library starts populated (with_defaults)")

    # ---- render mode ------------------------------------------------------------
    print("\nrender mode:")
    try:
        s.set_render_mode("nonsense")
        check(False, "invalid render mode raises")
    except ValueError:
        check(True, "invalid render mode raises ValueError")

    s.viewport = cdx.Viewport(complex(0, 0), 1.5, 41)
    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)

    s.set_render_mode("julia")
    julia = s.render()
    check(julia.shape == (41, 41), "julia render matches viewport resolution")

    s.param = -1 + 0j   # the basilica
    s.set_render_mode("basin")
    basin = s.render()
    labels = set(basin.flatten().tolist())
    check(len(labels) >= 2, "basin render for the basilica finds more than one region")

    s.set_render_mode("parameter")
    param_plane = s.render()
    check(param_plane.shape == (41, 41), "parameter-plane render matches viewport resolution")

    s.set_render_mode("greens")
    greens = s.render()
    check(greens.shape == (41, 41), "greens render matches viewport resolution")

    # ---- term editing ----------------------------------------------------------
    print("\nterm editing:")
    s2 = Session()
    s2.map = cdx.RationalMap("scratch")
    idx = s2.add_poly_term(2 + 0j, 3)
    check(idx == 0, "add_poly_term returns the new term's index")
    check(s2.map.eval(2 + 0j, 0j) == 16 + 0j, "added term (2z^3) evaluates correctly: 2*2^3=16")

    s2.edit_poly_term(0, coeff=3 + 0j)
    check(s2.map.eval(2 + 0j, 0j) == 24 + 0j, "edit_poly_term mutated the live term (3*2^3=24)")

    try:
        s2.edit_poly_term(5, coeff=1 + 0j)
        check(False, "out-of-range edit_poly_term raises")
    except IndexError:
        check(True, "out-of-range edit_poly_term raises IndexError")

    pole_idx = s2.add_pole_term(1 + 0j, 1 + 0j, 1)
    check(pole_idx == 0, "add_pole_term returns the new term's index")
    s2.remove_poly_term(0)
    check(len(s2.map.poly_terms()) == 0, "remove_poly_term removes the poly term")
    s2.remove_pole_term(0)
    check(len(s2.map.pole_terms()) == 0, "remove_pole_term removes the pole term")

    # ---- library -----------------------------------------------------------------
    print("\nlibrary:")
    s3 = Session()
    s3.map = cdx.RationalMap.mandelbrot()
    s3.save_to_library("my-mandelbrot")
    check(s3.library.find("my-mandelbrot") is not None, "save_to_library adds under the given name")

    s3.map = cdx.RationalMap.newton_cubic()
    s3.load_from_library("my-mandelbrot")
    check(s3.map.name == "my-mandelbrot" and s3.map.to_formula() == "z^2 + a",
          "load_from_library restores the saved map")

    try:
        s3.load_from_library("does-not-exist")
        check(False, "loading a missing name raises")
    except KeyError:
        check(True, "loading a missing name raises KeyError")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "library.txt")
        s3.save_library_file(path)
        names_before = set(s3.library.names())
        s4 = Session()
        s4.load_library_file(path)
        check(set(s4.library.names()) == names_before,
              "save_library_file/load_library_file round-trips the same names")

    # ---- dynamical_facts -----------------------------------------------------------
    print("\ndynamical_facts:")
    s5 = Session()
    s5.map = cdx.RationalMap.newton_cubic()
    s5.param = 0j
    facts = s5.dynamical_facts()
    check(facts.degree == 3, "newton_cubic: degree 3")
    check(len(facts.critical_points) == 4, "newton_cubic: 4 critical points")
    check(len(facts.attracting_cycles) == 3, "newton_cubic: 3 attracting cycles")
    check(all(ac.period == 1 for ac in facts.attracting_cycles),
          "newton_cubic: all 3 attracting cycles are fixed points")
    check(len(facts.fixed_points) == 4, "newton_cubic: 4 fixed points (3 roots + infinity)")
    check(facts.pole_locations == [0j] and facts.pole_orders == [2],
          "newton_cubic: one pole at the origin, order 2")

    opts = cdx.FindAttractorsOptions()
    opts.burn_in = 5   # deliberately too small to reliably converge
    facts_custom = s5.dynamical_facts(opts)
    check(facts_custom.degree == facts.degree,
          "dynamical_facts(opts): facts unrelated to discovery are unaffected")

    s6 = Session()
    s6.map = cdx.RationalMap.mandelbrot()
    s6.param = -1 + 0j
    facts_basilica = s6.dynamical_facts()
    n_2cycle = sum(1 for ac in facts_basilica.attracting_cycles if ac.period == 2)
    n_finite_fixed_attracting = sum(
        1 for fp in facts_basilica.fixed_points
        if not math.isinf(fp.point.real) and abs(fp.multiplier) < 1.0
    )
    check(n_2cycle == 1, "basilica: one attracting 2-cycle")
    check(n_finite_fixed_attracting == 0, "basilica: zero finite attracting fixed points")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
