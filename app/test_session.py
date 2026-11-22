"""test_session.py -- property-based checks for app/session.Session.

Mirrors the C++ suite's style (PASS/FAIL per check, properties rather than
golden output) rather than pulling in a Python test framework this project
does not otherwise use. Run with:

    PYTHONPATH=cdx/build python -m app.test_session

(from the repository root -- app.session now imports app.render_cache, so
this must run package-qualified with the repo root on sys.path, not
standalone from inside this directory).
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

import cdx
from app.orbit_tracker import OrbitTracker
from app.session import PARAMETER_PLANE_MODES, PRESET_FAMILY_NAMES, Session, render_map

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def lib_c_preset(name: str):
    return cdx.FamilyLibrary.with_defaults().find(name)


def main() -> None:
    print("=== app.session tests ===")

    # ---- defaults --------------------------------------------------------------
    print("\ndefaults:")
    s = Session()
    check(s.map.name == "mandelbrot", "starts on the mandelbrot preset")
    check(s.param == complex(-0.7269, 0.1889),
          "starts with the dendritic Julia parameter, for whenever the user switches to it")
    check(len(s.library) >= 6, "library starts populated (with_defaults)")
    # Session no longer owns a render_mode or a viewport at all (see
    # app.pane.Pane -- both moved there in the coupled-viewer refactor);
    # RENDER_MODES/PARAMETER_PLANE_MODES/mode-string validation are now
    # exercised against Pane directly, in app/test_pane.py.

    # ---- render mode dispatch: render() takes viewport/mode explicitly ----------
    print("\nrender mode (Session.render(viewport, mode) dispatch):")
    vp = cdx.Viewport(complex(0, 0), 1.5, 41)
    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)

    julia = s.render(vp, "julia")
    check(julia.shape == (3, 41, 41),
          "julia render is STACKED (values, escaped, labels) at the viewport resolution")

    s.param = -1 + 0j   # the basilica
    basin = s.render(vp, "basin")
    check(basin.shape == (2, 41, 41),
          "basin render is STACKED: (2, height, width) -- labels, then iterations")
    labels, iterations = basin[0], basin[1]
    check(len(set(labels.flatten().tolist())) >= 2,
          "basin render for the basilica finds more than one region")
    check(iterations.min() >= 0, "the iterations layer is always non-negative")
    check(iterations[labels > 0].min() >= 1,
          "every RESOLVED pixel took at least one iteration to get there")

    param_plane = s.render(vp, "parameter")
    check(param_plane.shape == (2, 41, 41),
          "parameter-plane render is STACKED (values, escaped) at the viewport resolution")

    greens_array = s.render(vp, "greens")
    check(greens_array.shape == (41, 41), "greens render matches viewport resolution")

    pgreens_array = s.render(vp, "parameter_greens")
    check(pgreens_array.shape == (41, 41),
          "parameter_greens render also matches viewport resolution")

    pbasin_array = s.render(vp, "parameter_basin")
    check(pbasin_array.shape == (3, 41, 41),
          "parameter_basin render is ALWAYS STACKED: (3, height, width) -- counts, "
          "unresolved, then certain -- there is no certified-polynomial plain-2D fast "
          "path here")

    # ---- Stage 2: rational Julia is STACKED, escape_radius-invariant ------------
    print("\nrender mode dispatch: rational Julia (Stage 2 sphere-aware classification):")
    newton4 = cdx.RationalMap("newton4")   # (3/4)z + (1/4)z^-3 -- Newton's method for z^4-1
    newton4.add_poly(complex(0.75, 0.0), 1, 0, "(3/4)z")
    newton4.add_pole(complex(0.0, 0.0), complex(0.25, 0.0), 3, 0, "1/(4z^3)")
    check(cdx.polynomial_escape_certified(newton4) is False,
          "sanity: Newton z^4-1 genuinely takes the rational path")

    s_rational = Session()
    s_rational.map = newton4
    s_rational.param = 0j
    rational_vp = cdx.Viewport(complex(0, 0), 2.0, 41)
    s_rational.render_settings = cdx.RenderSettings(80, 2.0, 1e-6, 1)
    rational_julia = s_rational.render(rational_vp, "julia")
    check(rational_julia.shape == (3, 41, 41),
          "a RATIONAL map's julia render is STACKED: (3, height, width) -- smooth values, "
          "escaped (resolved) mask, then basin labels")
    values, labels = rational_julia[0], rational_julia[2]
    check(np.array_equal(rational_julia[1] != 0, labels > 0),
          "for a rational map the escaped mask is exactly 'reached an attractor' (label > 0)")
    check(len(np.unique(labels[labels > 0])) >= 2,
          "Newton z^4-1's Julia render finds more than one basin (more than one root "
          "captures pixels)")

    # ACCEPTANCE TEST at the full Session.render level: escape_radius must
    # not affect a rational map's classification AT ALL.
    s_rational.render_settings = cdx.RenderSettings(80, 10.0, 1e-6, 1)
    rational_julia_r10 = s_rational.render(rational_vp, "julia")
    check(np.array_equal(rational_julia, rational_julia_r10),
          "ACCEPTANCE: Session.render(..., 'julia') for a rational map is BYTE-IDENTICAL "
          "at escape_radius=2 vs escape_radius=10 -- through the FULL app dispatch, not "
          "just the bare cdx.Renderer call")

    # A certified polynomial's Julia render has the SAME stacked shape (the
    # bubble fix made every escape-time payload (values, escaped, ...); the
    # old plain-2D shape let the value double as the escape flag). Its
    # labels layer is all 0: there is no basin concept on that path.
    s_rational.map = cdx.RationalMap.mandelbrot()
    s_rational.param = complex(-0.7269, 0.1889)
    s_rational.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)
    poly_julia = s_rational.render(rational_vp, "julia")
    check(poly_julia.shape == (3, 41, 41) and not np.any(poly_julia[2]),
          "a certified polynomial's julia render is stacked the same way, labels all 0")
    check(np.array_equal(poly_julia[1] == 0, poly_julia[0] == 0),
          "...and at this ordinary view its mask and its nonzero values agree exactly")

    # ---- Parameter_basin: counts distinct attracting cycles per parameter ----------
    print("\nrender mode dispatch: Parameter_basin (number of attracting cycles):")
    # z^2 + a: at c=0 (deep in the cardioid) BOTH the origin's own fixed
    # point AND infinity are independently confirmed attracting -- count=2.
    # Far outside (c=5+5i), both critical-point seeds (0 and infinity)
    # converge to the SAME attractor and dedupe to count=1. Cross-checked
    # against cdx.find_attractors directly in cdx/test/test_analysis.cpp;
    # this is the app-dispatch-level counterpart, not a re-derivation.
    quad = cdx.RationalMap("quad")
    quad.add_poly(1 + 0j, 2, 0, "z^2")
    quad.add_poly(1 + 0j, 0, 1, "a")

    s_pb = Session()
    s_pb.map = quad
    s_pb.render_settings = cdx.RenderSettings(100, 2.0, 1e-6, 1)

    tiny_vp_inside = cdx.Viewport(complex(0, 0), 0.001, 3)
    counts_inside, unresolved_inside, certain_inside = s_pb.render(tiny_vp_inside, "parameter_basin")
    check(counts_inside[1, 1] == 2.0 and unresolved_inside[1, 1] == 0.0,
          "c=0: count=2 (origin's own fixed point + infinity), unresolved=0")
    check(certain_inside[1, 1] == 0.0,
          "c=0: neither attractor is CERTAIN (algebraically injected) here -- both z=0 "
          "and infinity are themselves critical points, found directly by the ordinary "
          "critical-seeded search, never needing the fixed-point union at all")

    tiny_vp_far = cdx.Viewport(complex(5, 5), 0.001, 3)
    counts_far, unresolved_far, certain_far = s_pb.render(tiny_vp_far, "parameter_basin")
    check(counts_far[1, 1] == 1.0 and unresolved_far[1, 1] == 0.0,
          "c=5+5i: count=1 (both seeds dedupe onto infinity), unresolved=0")
    check(certain_far[1, 1] == 0.0, "c=5+5i: still not certain -- also found by critical seeding")

    # A real render spanning the cusp shows count genuinely CHANGING across
    # the plane -- this is what makes color_parameter_basin's sharp
    # color edges meaningful at all, not a uniform/degenerate image.
    wide_vp = cdx.Viewport(complex(-0.5, 0), 1.5, 61)
    counts_wide, _unresolved_wide, _certain_wide = s_pb.render(wide_vp, "parameter_basin")
    unique_counts = set(counts_wide.flatten().tolist())
    check(len(unique_counts & {1.0, 2.0}) == 2,
          "a real render shows BOTH count=1 and count=2 pixels -- a genuine "
          "classification, not a flat plane")

    # No-effect-parameter guard (dynamical-plane concept generalized):
    # newton_cubic() has no term depending on `a` at all, so ImageView's
    # own no_effect_parameter_message (app/sandbox.py) should treat
    # "parameter_basin" the same as "parameter"/"parameter_greens" --
    # verified here at the describe_parameter_role level this session
    # already tests elsewhere, confirming PARAMETER_PLANE_MODES
    # membership (not a hardcoded per-mode name list) is what drives it.
    check("parameter_basin" in PARAMETER_PLANE_MODES,
          "parameter_basin is a genuine PARAMETER-plane mode -- gets the marker/"
          "arrow-nudging/no-effect-guard machinery every other one already has, "
          "purely through set membership")

    # ---- Parameter_period: period of the tracked critical orbit's attractor --------
    print("\nrender mode dispatch: Parameter_period (period coloring, KLS comparison):")
    check("parameter_period" in PARAMETER_PLANE_MODES,
          "parameter_period is a genuine PARAMETER-plane mode too, same free machinery")

    # Relaxed Newton of z^n-1, n=6: N_a(z) = ((n-a)z^n + a) / (n z^{n-1}).
    # map.degree(a) == n exactly for this family (numerator degree n,
    # denominator degree n-1) -- render_map's own "parameter_period" branch
    # relies on this to recover n without a separate piece of session state.
    n = 6
    src = f"(({n}-a)*z^{n} + a) / ({n}*z^{n - 1})"
    relaxed_newton = cdx.RationalMap.from_expression(src, "a", {}, "relaxed_newton")

    s_pp = Session()
    s_pp.map = relaxed_newton
    s_pp.render_settings = cdx.RenderSettings(200, 2.0, 1e-6, 1)
    check(round(relaxed_newton.degree(0j)) == n,
          "sanity: map.degree(a) == n for this family -- what render_map's own "
          "parameter_period dispatch relies on")

    # Superattracting center a_1 = n - (n-1)*omega_1 -- expected period n=6
    # (gcd(1,6)=1) -- the SAME validated anchor cdx/test/test_analysis.cpp
    # checks at the engine level, exercised here through the FULL app
    # dispatch (Session.render, not the bare cdx.Renderer call).
    omega_1 = complex(math.cos(2 * math.pi / n), math.sin(2 * math.pi / n))
    a_1 = n - (n - 1) * omega_1
    tiny_vp = cdx.Viewport(a_1, 0.001, 3)
    periods, undetermined, is_infinity, _mixed = s_pp.render(tiny_vp, "parameter_period")
    check(periods[1, 1] == 6.0 and undetermined[1, 1] == 0.0 and is_infinity[1, 1] == 0.0,
          "a_1 = n-(n-1)*omega_1 reports period 6 through the full Session.render "
          "dispatch, matching the engine-level validation anchor exactly")

    # Exterior |a-n| > n -- the infinity class, not undetermined.
    exterior_vp = cdx.Viewport(complex(3 * n, 0.0), 0.001, 3)
    periods_ext, undetermined_ext, is_infinity_ext, _ = s_pp.render(exterior_vp,
                                                                      "parameter_period")
    check(is_infinity_ext[1, 1] == 1.0 and undetermined_ext[1, 1] == 0.0,
          "well outside |a-n|=n reports the infinity class")

    # A real render spanning several superattracting centers shows period
    # genuinely CHANGING across the plane -- the period-coloring analogue of
    # Parameter_basin's own "not a uniform plane" check above.
    wide_pp_vp = cdx.Viewport(complex(n, 0.0), n + 1.0, 61)
    periods_wide, undetermined_wide, _is_inf_wide, _ = s_pp.render(wide_pp_vp,
                                                                   "parameter_period")
    resolved_periods = set(periods_wide[undetermined_wide == 0.0].flatten().tolist())
    check(len(resolved_periods - {0.0}) >= 2,
          "a real render shows more than one DISTINCT period value -- a genuine "
          "classification, not a flat plane")

    check(cdx.relaxed_newton_power_n(relaxed_newton) == n,
          "relaxed_newton_power_n recognises the family -- the closed-form seeds are used")

    # ---- Parameter_period on maps that are NOT relaxed Newton (Bug A) -----------
    print("\nParameter_period: every other map tracks its OWN free critical points:")
    mandel = cdx.RationalMap.mandelbrot()
    check(cdx.relaxed_newton_power_n(mandel) is None and
          cdx.relaxed_newton_power_n(cdx.RationalMap.from_expression("z^2+a", "a", {})) is None,
          "z^2 + a is not mistaken for relaxed Newton (term-based or typed)")
    pp_settings = cdx.RenderSettings(200, 2.0, 1e-6, 0)

    def mandel_period_at(a):
        st = render_map(mandel, 0j, cdx.Viewport(a, 1e-9, 3), pp_settings, "parameter_period")
        if st[2][1, 1] == 1.0:
            return "escapes"
        if st[1][1, 1] == 1.0:
            return "undetermined"
        return int(st[0][1, 1])

    # 0.25+0.5i is the ROOT of the period-4 bulb (c = mu/2 - mu^2/4 at mu = i):
    # its fixed point i/2 has multiplier exactly i, so the critical orbit
    # creeps parabolically toward it and never resolves -- "undetermined" is
    # the honest answer, not "escapes". 0.5+0.5i genuinely escapes.
    for a, want in [(0j, 1), (-1 + 0j, 2), (-1.31 + 0j, 4), (-1.7549 + 0j, 3),
                    (-1.1 + 0.2j, 2), (-0.1226 + 0.7449j, 3), (0.25 + 0.5j, "undetermined"),
                    (0.5 + 0.5j, "escapes")]:
        got = mandel_period_at(a)
        check(got == want, f"Mandelbrot parameter_period at a={a}: {got} (want {want})")

    # Default view vs an independent plain-numpy reference: iterate z = 0 for
    # 1500 steps, then look for a return within 1e-7, up to period 40.
    res = 360
    vp_m = cdx.Viewport(complex(-0.75, 0.0), 1.5, res)
    st = render_map(mandel, 0j, vp_m, pp_settings, "parameter_period")
    s_px = 2 * 1.5 / (res - 1)
    cols = -0.75 - 1.5 + s_px * np.arange(res)
    rows = -1.5 + s_px * np.arange(res)
    c = cols[None, :] + 1j * rows[:, None]
    z = np.zeros_like(c)
    alive = np.ones(c.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(1500):
            z = np.where(alive, z * z + c, z)
            alive &= np.abs(z) <= 2.0
        ref_period = np.zeros(c.shape, dtype=int)
        w = z.copy()
        for p in range(1, 41):
            w = w * w + c
            hit = alive & (ref_period == 0) & (np.abs(w - z) < 1e-7)
            ref_period[hit] = p
    interior = ref_period > 0
    agree = np.mean(st[0][interior] == ref_period[interior])
    check(interior.sum() > 1000 and agree >= 0.99,
          f"default Mandelbrot view, 360 px: {agree:.4f} of the {interior.sum()} reference-"
          "interior pixels get the reference period (>= 0.99)")
    check(float(np.mean(st[2][interior] == 1.0)) < 0.001,
          "...and essentially none of the true interior is reported 'escapes' (was 20.6%)")

    # Relaxed Newton n = 5: FromMap (the map's own free critical points, the
    # one critical_index selects) agrees with the closed-form seeds.
    rn5 = cdx.RationalMap.from_expression("z - a*(z^5-1)/(5*z^4)", "a", {}, "rn5")
    check(cdx.relaxed_newton_power_n(rn5) == 5, "the other written form of relaxed Newton is "
          "recognised too (evaluation-based, not a pattern match)")
    agree_all = True
    for a in (1.0 + 0.3j, 5 - 4 * complex(math.cos(2 * math.pi / 5), math.sin(2 * math.pi / 5)),
              5 - 4 * complex(math.cos(4 * math.pi / 5), math.sin(4 * math.pi / 5)), 18 + 0j):
        r = cdx.Renderer(map=cdx.Map.custom(rn5, 0j), viewport=cdx.Viewport(a, 1e-6, 3),
                         settings=pp_settings)
        closed = r.render_parameter_period(cdx.CriticalPointFamily.RelaxedNewtonPower, 5)
        for idx in range(5):
            frm = r.render_parameter_period(cdx.CriticalPointFamily.FromMap, 0, None, idx)
            for k in range(3):
                if not np.array_equal(closed[k], frm[k]):
                    agree_all = False
    check(agree_all, "relaxed Newton n=5 at four parameters (periods 1, 5, 5, infinity): "
          "FromMap with every critical_index gives exactly the closed-form result")

    # z^3 + a z + 1, two free critical points. In this window no pixel has
    # both orbits bounded, so "both on the same cycle" means both escape --
    # an independent numpy escape test per critical point decides it.
    cubic = cdx.RationalMap.from_expression("z^3 + a*z + 1", "a", {}, "cubic")
    res_c = 61
    vp_c = cdx.Viewport(0j, 1.5, res_c)
    st_c = render_map(cubic, 0j, vp_c, pp_settings, "parameter_period")
    s_c = 2 * 1.5 / (res_c - 1)
    ac = (-1.5 + s_c * np.arange(res_c))[None, :] + 1j * (-1.5 + s_c * np.arange(res_c))[:, None]
    escaped = []
    with np.errstate(over="ignore", invalid="ignore"):
        for sign in (1, -1):
            zc = sign * np.sqrt(-ac / 3 + 0j)
            esc = np.zeros(ac.shape, dtype=bool)
            for _ in range(500):
                zc = np.where(esc, zc, zc ** 3 + ac * zc + 1)
                esc |= np.abs(zc) > 1e6
            escaped.append(esc)
    both_escape = escaped[0] & escaped[1]
    one_escapes = escaped[0] ^ escaped[1]
    check(both_escape.sum() > 100 and np.all(st_c[3][both_escape] == 0.0),
          f"z^3+az+1: mixed is 0 on all {both_escape.sum()} pixels where both free orbits "
          "reach the same cycle (infinity)")
    check(one_escapes.sum() > 20 and np.mean(st_c[3][one_escapes] == 1.0) > 0.95,
          "...and 1 where one escapes and the other is captured by a finite cycle")
    st_c1 = render_map(cubic, 0j, vp_c, pp_settings, "parameter_period", critical_index=1)
    check(not np.array_equal(st_c[2], st_c1[2]),
          "critical_index selects which free orbit parameter_period reports")

    # ---- Parameter: multi-critical polynomials (Bug B, part 1) --------------------
    print("\nParameter: z^3 + a z + 1 iterates BOTH free critical orbits, per strategy:")
    cubic_b = cdx.RationalMap.from_expression("z^3 + a*z + 1", "a", {}, "cubic_b")
    res_b, scale_b = 120, 1.5
    vp_b = cdx.Viewport(0j, scale_b, res_b)
    set_b = cdx.RenderSettings(200, 2.0, 1e-6, 0)
    imgs_b = {name: render_map(cubic_b, 0j, vp_b, set_b, "parameter", strategy=name)
              for name in ("slowest", "fastest", "per_critical")}
    imgs_b["per_critical_1"] = render_map(cubic_b, 0j, vp_b, set_b, "parameter",
                                          strategy="per_critical", critical_index=1)
    s_b = 2 * scale_b / (res_b - 1)
    a_b = (-scale_b + s_b * np.arange(res_b))[None, :] + 1j * (-scale_b + s_b * np.arange(res_b))[:, None]
    # Independent reference: does each of +-sqrt(-a/3) escape |z| > R within
    # max_iter? (The R used here is the render's own; whether R is itself a
    # valid bound is part 2's concern, not this check's.)
    esc_b = []
    with np.errstate(over="ignore", invalid="ignore"):
        for sign in (1, -1):
            zb = sign * np.sqrt(-a_b / 3 + 0j)
            e = np.zeros(a_b.shape, dtype=bool)
            for _ in range(200):
                zb = np.where(e, zb, zb ** 3 + a_b * zb + 1)
                e |= np.abs(zb) > 2.0
            esc_b.append(e)
    some_bounded = ~(esc_b[0] & esc_b[1])
    none_escape = ~(esc_b[0] | esc_b[1])
    # Escape status is the MASK (layer 1), never value > 0 -- see render_map.
    esc_of = {k: v[1] != 0 for k, v in imgs_b.items()}
    val_of = {k: v[0] for k, v in imgs_b.items()}
    check(np.mean(~esc_of["slowest"] == some_bounded) > 0.995,
          "Slowest is not-escaped exactly where SOME free orbit stays bounded")
    check(np.mean(~esc_of["fastest"] == none_escape) > 0.995,
          "Fastest is not-escaped exactly where NO free orbit escapes (the connectedness locus)")
    check(np.all(val_of["slowest"][esc_of["slowest"]] >= val_of["fastest"][esc_of["slowest"]]),
          "where both escape, Slowest >= Fastest pixel-for-pixel")
    pc_pair = np.stack([esc_of["per_critical"], esc_of["per_critical_1"]])
    ref_pair = np.stack([esc_b[0], esc_b[1]])
    same_set = ((pc_pair[0] == ref_pair[0]) & (pc_pair[1] == ref_pair[1])) | \
               ((pc_pair[0] == ref_pair[1]) & (pc_pair[1] == ref_pair[0]))
    check(np.mean(same_set) > 0.995,
          "PerCritical 0 and 1 are the two individual free orbits (as an unordered pair)")
    check(not np.array_equal(val_of["slowest"], val_of["fastest"]) and
          not np.array_equal(val_of["slowest"], val_of["per_critical_1"]),
          "the strategies genuinely differ (they were byte-identical before)")
    for name in ("slowest", "fastest"):
        m_esc = esc_of[name]
        check(np.array_equal(m_esc, np.flipud(m_esc)),
              f"{name}: symmetric under a -> conj(a) (real coefficients) -- no seam on Im a = 0")

    # ---- The "bubble": escape status is the MASK, never value > 0 -----------------
    # The smooth value is legitimately <= 0 for fast escapes; when the value
    # doubled as the "escaped?" flag, z^5 + a at half-width 105 was 99.97%
    # black and the Mandelbrot plane at half-width 50 was black beyond
    # |a| = e^4 ~ 54.6 (12%). Both checked through the app's own colouring.
    print("\nThe bubble: fast escapes are exterior (explicit escaped mask):")
    from app.color import NEVER_ESCAPED_RGB, color_escape_time
    a5 = complex(-0.1747, 1.0721)
    set_bb = cdx.RenderSettings(200, 2.0, 1e-6, 0)
    vp5 = cdx.Viewport(0j, 105.0, 200)
    st5 = render_map(cdx.RationalMap.multibrot(5), a5, vp5, set_bb, "julia")
    t5 = np.linspace(-105.0, 105.0, 200)
    z5 = np.abs(t5[None, :] + 1j * t5[:, None])
    R5 = max(2.0, cdx.polynomial_escape_radius([a5, 0, 0, 0, 0, 1]))
    outside = z5 > R5
    check(bool(np.all(st5[1][outside] == 1.0)),
          f"z^5 + a at half-width 105: all {int(outside.sum())} pixels with |z0| > R(a) = "
          f"{R5:.3g} are masked escaped")
    check(float(st5[0][outside].min()) < 0.0,
          f"...including the fast escapes whose smooth value is negative (min "
          f"{st5[0][outside].min():.3f}) -- the value is a shade, not a flag")
    rgb5 = color_escape_time(st5[0], 200, palette="classic", escaped=st5[1])
    painted_interior = np.all(rgb5 == np.array(NEVER_ESCAPED_RGB, dtype=np.uint8), axis=-1)
    check(not np.any(painted_interior[outside]),
          "...and none of them is coloured as interior (was 99.97% black)")
    check(bool(np.all(np.isfinite(st5[0]))), "...and every smooth value is finite")

    # Far beyond double range in one step (z^5 at |z0| ~ 1e62 overflows): still
    # escaped, finite, and ordered -- not -inf, not NaN, not "interior".
    far = render_map(cdx.RationalMap.multibrot(5), a5, cdx.Viewport(0j, 1e200, 9), set_bb, "julia")
    far_g = render_map(cdx.RationalMap.multibrot(5), a5, cdx.Viewport(0j, 1e200, 9), set_bb, "greens")
    check(bool(np.all(far[1] == 1.0) and np.all(np.isfinite(far[0])) and np.all(far_g > 0)
               and np.all(np.isfinite(far_g))),
          "|z0| up to 1e200 (first step overflows): escaped, finite smooth value, finite G > 0")

    mb = cdx.RationalMap.mandelbrot()
    res50 = 400
    vp50 = cdx.Viewport(0j, 50.0, res50)
    st50 = render_map(mb, 0j, vp50, set_bb, "parameter")
    t50 = np.linspace(-50.0, 50.0, res50)
    c50 = t50[None, :] + 1j * t50[:, None]
    zz = np.zeros_like(c50)
    filled = np.ones(c50.shape, dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(200):
            zz = np.where(filled, zz * zz + c50, zz)
            filled &= np.abs(zz) <= 2.0

    def dilate(mask):
        out = mask.copy()
        out[1:, :] |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    black50 = st50[1] == 0.0
    check(bool(np.all(dilate(filled)[black50]) and np.all(dilate(black50)[filled])),
          f"Mandelbrot parameter plane, half-width 50: the black set ({int(black50.sum())} px) "
          f"is the filled set ({int(filled.sum())} px) to within one pixel ring, not "
          f"everything beyond |a| = e^4")
    pg50 = render_map(mb, 0j, vp50, set_bb, "parameter_greens")
    check(bool(np.all(pg50[st50[1] == 1.0] > 0.0)),
          "parameter_greens: G > 0 on every escaped pixel (no clamp or sentinel reads as 0)")

    # Ordinary views: the mask is exactly the old "value > 0" classification,
    # so interior/exterior is unchanged wherever no smooth value is negative.
    ordinary = [("mandelbrot", cdx.RationalMap.mandelbrot(), -0.7269 + 0.1889j, "julia", 0j, 1.6),
                ("mandelbrot", cdx.RationalMap.mandelbrot(), 0j, "parameter", -0.5 + 0j, 1.5),
                ("multibrot5", cdx.RationalMap.multibrot(5), 0.3j, "julia", 0j, 1.3),
                ("z^3+az+1", cdx.RationalMap.from_expression("z^3+a*z+1", "a", {}, "c3"), 0j,
                 "parameter", 0j, 1.5),
                ("mcmullen2", lib_c_preset("mcmullen2"), 0j, "parameter", 0j, 0.3),
                ("newton3", lib_c_preset("newton3"), 0j, "julia", 0j, 1.5)]
    for name, m_o, a_o, mode_o, c_o, s_o in ordinary:
        st_o = render_map(m_o, a_o, cdx.Viewport(c_o, s_o, 96), set_bb, mode_o)
        check(np.array_equal(st_o[1] != 0.0, st_o[0] > 0.0),
              f"{name} {mode_o} at an ordinary view: escaped mask == old 'value > 0' "
              f"classification on every pixel")
        legacy = color_escape_time(st_o[0], 200, palette="classic")
        masked = color_escape_time(st_o[0], 200, palette="classic", escaped=st_o[1])
        check(np.array_equal(legacy, masked), f"...and its colours are byte-identical")

    # ---- Smooth escape value: base d, continuous across escape bands ---------------
    # nu = n + 1 - log_d(log|z_n| / log R). With the old base 2 every band
    # boundary of z^3 + a jumped by log2(3) - 1 = 0.58 and of z^5 + a by 1.32.
    # "Away from the boundary of the set": only band crossings where the
    # field itself is smooth -- the same-band neighbours on BOTH sides change
    # by < 0.02 per pixel. (Near the Julia set the field is legitimately
    # rough at any resolution, so a jump there says nothing about seams.)
    # The integer escape iteration of each pixel comes from numpy.
    print("\nSmooth escape value is continuous across escape-iteration boundaries (base d):")
    res_s, sc_s = 400, 3.0
    t_s = np.linspace(-sc_s, sc_s, res_s)
    g_s = t_s[None, :] + 1j * t_s[:, None]
    set_s = cdx.RenderSettings(200, 2.0, 1e-6, 0)

    def escape_iteration(d, a, z0):
        z = z0.copy()
        n_esc = np.zeros(z.shape, dtype=int)
        alive = np.ones(z.shape, dtype=bool)
        with np.errstate(all="ignore"):
            for k in range(1, 201):
                z = np.where(alive, z ** d + a, z)
                out = alive & (np.abs(z) > 2.0)
                n_esc[out] = k
                alive &= ~out
        return n_esc

    def worst_band_jump(img, n_esc):
        worst, crossings = 0.0, 0
        for X, N in ((img, n_esc), (img.T, n_esc.T)):
            dx = np.diff(X, axis=1)
            ok = ((N[:, 2:-1] != N[:, 1:-2]) & (N[:, :-3] == N[:, 1:-2]) & (N[:, 3:] == N[:, 2:-1])
                  & (N[:, 1:-2] > 0) & (N[:, 2:-1] > 0)
                  & (np.abs(dx[:, :-2]) < 0.02) & (np.abs(dx[:, 2:]) < 0.02))
            if ok.any():
                worst = max(worst, float(np.abs(dx[:, 1:-1])[ok].max()))
                crossings += int(ok.sum())
        return worst, crossings

    for d in (3, 5):
        fam_s = cdx.RationalMap.multibrot(d)
        jul = render_map(fam_s, 0.3j, cdx.Viewport(0j, sc_s, res_s), set_s, "julia")[0]
        par = render_map(fam_s, 0j, cdx.Viewport(0j, sc_s, res_s), set_s, "parameter")[0]
        for plane, img, n_esc in (("julia a=0.3i", jul, escape_iteration(d, 0.3j, g_s)),
                                  ("parameter", par, escape_iteration(d, g_s, np.zeros_like(g_s)))):
            worst, crossings = worst_band_jump(img, n_esc)
            check(crossings > 20 and worst <= 0.05,
                  f"z^{d} + a {plane}: largest jump across an escape band {worst:.4f} "
                  f"(<= 0.05; {crossings} crossings checked)")

    # ---- Parameter: only FREE critical points drive the strategies (Bug C) -------
    # z^n + a/z^n has critical points 0 (a double pole, f(0) = inf) and inf (a
    # superattracting fixed point) besides the 2n free ones z^(2n) = a. Their
    # orbits are the same for every a; classifying them made Slowest 100% black
    # and Fastest constant. Reference: numpy iterates the free point a^(1/(2n))
    # for 200 steps, escape at |z| > 50, "black" = never escaped.
    #
    # The rational path is sphere-aware: `parameter` is 0 only where an orbit
    # never settles on ANY attractor, so a Mandelbrot copy (free orbit captured
    # by a finite attracting cycle) is shaded, not black. The like-for-like
    # quantity -- "the free orbit does not reach infinity" -- is
    # parameter_period's is_infinity == 0; that is what is compared with the
    # numpy black fraction (+-3 points), and every black `parameter` pixel
    # must lie inside the numpy black set.
    print("\nParameter: McMullen z^n + a/z^n uses only its free critical points (Bug C):")
    lib_c = cdx.FamilyLibrary.with_defaults()
    res_m, scale_m = 128, 0.3
    vp_m3 = cdx.Viewport(0j, scale_m, res_m)
    set_m = cdx.RenderSettings(200, 2.0, 1e-6, 0)
    t_m = np.linspace(-scale_m, scale_m, res_m)
    a_m = t_m[None, :] + 1j * t_m[:, None]
    for name, n_m in (("mcmullen2", 2), ("mcmullen3", 3)):
        fam = lib_c.find(name)
        free_m = fam.free_critical_points(0.01 + 0.02j)
        check(len(free_m) == 2 * n_m and all(math.isfinite(abs(z)) and abs(z) > 1e-3 for z in free_m),
              f"{name}: exactly the {2 * n_m} roots of z^{2 * n_m} = a are free (0 and inf are not)")
        zr = a_m ** (1.0 / (2 * n_m))
        alive_m = np.ones(a_m.shape, dtype=bool)
        with np.errstate(all="ignore"):
            for _ in range(200):
                zr = np.where(alive_m, zr ** n_m + a_m / zr ** n_m, zr)
                alive_m &= np.isfinite(zr) & (np.abs(zr) <= 50.0)
        ref_black = float(alive_m.mean())
        slow_st = render_map(fam, 0j, vp_m3, set_m, "parameter", strategy="slowest")
        slow, black = slow_st[0], slow_st[1] == 0
        check(np.ptp(slow) > 0 and 0.0 < float(np.mean(black)) < 0.5,
              f"{name} Slowest is not uniform (black {np.mean(black):.3f}; was 1.000)")
        check(bool(np.all(alive_m[black])),
              f"{name}: every black Slowest pixel is black in the numpy reference too")
        st_m = render_map(fam, 0j, vp_m3, set_m, "parameter_period")
        bounded = float(np.mean(st_m[2] == 0.0))
        check(abs(bounded - ref_black) <= 0.03,
              f"{name}: free orbit misses infinity on {bounded:.4f} of pixels, numpy "
              f"reference black {ref_black:.4f} (within 3 points)")
    fast3 = render_map(lib_c.find("mcmullen3"), 0j, vp_m3, set_m, "parameter", strategy="fastest")[0]
    check(np.ptp(fast3) > 1.0, f"mcmullen3 Fastest is not constant (ptp {np.ptp(fast3):.3g}; was 0)")
    pb_m = render_map(lib_c.find("mcmullen2"), 0j, cdx.Viewport(0j, scale_m, 33), set_m,
                      "parameter_basin")
    check(bool(np.all(pb_m[0] >= 1.0)),
          "mcmullen2 parameter_basin still counts infinity (it keeps ALL critical points)")
    # Zero free critical points: Newton's map for z^2 - 1, (z^2 + 1)/(2z), has
    # only +-1, both superattracting fixed points, and no parameter at all.
    newton2 = cdx.RationalMap.from_expression("(z^2+1)/(2*z)", "", {}, "newton2")
    check(newton2.free_critical_points(0.3j) == [], "Newton for z^2 - 1 has no free critical point")
    nz = render_map(newton2, 0j, cdx.Viewport(0j, 1.0, 21), set_m, "parameter")
    nz_p = render_map(newton2, 0j, cdx.Viewport(0j, 1.0, 21), set_m, "parameter_period")
    check(np.all(nz == 0.0) and np.all(nz_p[1] == 1.0),   # values and mask all 0
          "...so its parameter plane is honestly uniform (0; period undetermined), not a crash")

    # ---- Parameter planes from the free critical orbits vs the reconciled oracle ---
    # The default path (critical_orbits.hpp) against the old per-pixel
    # discovery + fixed-point union + reconciliation path, kept behind
    # Renderer._reconciled_parameter_path as the oracle. Compared quantity:
    # `parameter` -- the escaped/resolved mask; `parameter_basin` -- the count;
    # `parameter_period` -- period / undetermined / infinity.
    print("\nParameter planes: critical-orbit path agrees with the reconciled oracle (128 px):")
    set_o = cdx.RenderSettings(200, 2.0, 1e-6, 0)
    lib_o = cdx.FamilyLibrary.with_defaults()
    rn5_o = cdx.RationalMap.from_expression("z - a*(z^5-1)/(5*z^4)", "a", {}, "rn5")
    fams_o = [("mandelbrot", lib_o.find("mandelbrot"), complex(-0.75, 0), 1.5),
              ("multibrot3", lib_o.find("multibrot3"), 0j, 1.35),
              ("multibrot5", lib_o.find("multibrot5"), 0j, 1.25),
              ("mcmullen2", lib_o.find("mcmullen2"), 0j, 0.3),
              ("mcmullen3", lib_o.find("mcmullen3"), 0j, 0.3),
              ("z^3+az+1", cdx.RationalMap.from_expression("z^3+a*z+1", "a", {}, "c3o"), 0j, 1.5),
              ("relaxed Newton n=5", rn5_o, complex(5, 0), 6.0)]

    def oracle_render(m, c, sc, mode, reconciled):
        r = cdx.Renderer(map=cdx.Map.custom(m, 0j), viewport=cdx.Viewport(c, sc, 128),
                         settings=set_o)
        r._reconciled_parameter_path = reconciled
        if mode == "parameter":
            return r.render_parameter(None, cdx.ParameterStrategy.Slowest, 0)[1] != 0
        if mode == "parameter_basin":
            return r.render_parameter_basin(None)[0]
        nn = cdx.relaxed_newton_power_n(m)
        fam_k = (cdx.CriticalPointFamily.RelaxedNewtonPower if nn is not None
                 else cdx.CriticalPointFamily.FromMap)
        per, und, inf_, _mix = r.render_parameter_period(fam_k, nn or 0, None, 0)
        return np.where(und == 1, -1, np.where(inf_ == 1, -2, per))

    for name, m, c, sc in fams_o:
        for mode in ("parameter", "parameter_basin", "parameter_period"):
            new = oracle_render(m, c, sc, mode, False)
            old = oracle_render(m, c, sc, mode, True)
            differ = new != old
            note = ""
            if name.startswith("relaxed") and mode == "parameter_period":
                # The reconciled path called infinity only once |z| passed 1e12
                # inside its 500-step burn-in, i.e. only for |lambda_inf| below
                # 1e-12^(1/500) = 0.947. Where infinity attracts more slowly it
                # reported "undetermined": wrong, since lambda_inf = 5/(5 - a) is
                # closed-form. Those pixels are checked against that formula,
                # then excluded.
                t_o = np.linspace(-sc, sc, 128)
                a_o = (c.real + t_o[None, :]) + 1j * (c.imag + t_o[:, None])
                lam_inf = np.abs(5.0 / (5.0 - a_o))
                missed = differ & (old == -1) & (new == -2)
                check(missed.sum() > 100 and bool(np.all(lam_inf[missed] < 1.0)),
                      f"relaxed Newton period: all {int(missed.sum())} pixels the old path left "
                      f"undetermined and the new one sends to infinity have |5/(5-a)| < 1 "
                      f"(max {lam_inf[missed].max():.4f}): infinity IS attracting there")
                differ &= ~missed
                note = " (excluding the old path's infinity misses)"
            agree = 1.0 - float(differ.mean())
            check(agree >= 0.995, f"{name} {mode}: {100 * agree:.2f}% of pixels agree{note}")

    # ---- critical_index labelling convention (Bug B, part 3) ----------------------
    print("\ncritical_index labelling: continued from a* along a straight segment:")
    cub = cdx.RationalMap.from_expression("z^3 + a*z + 1", "a", {}, "cubic_l")
    a_star = 0.618 + 0.382j

    # The old seam: Im a = 0, Re a in [-1.5, -0.45]. Labels must not jump there.
    eps = 1e-7
    worst = 0.0
    for x in np.linspace(-1.5, -0.05, 60):
        up = cub.critical_point_labels(complex(x, eps))
        dn = cub.critical_point_labels(complex(x, -eps))
        worst = max(worst, max(abs(u - d) for u, d in zip(up, dn)))
    check(worst < 1e-5, f"labels are continuous across Im a = 0 for Re a < 0 "
                        f"(largest change across the axis {worst:.2g})")

    # The ONLY discontinuity: the ray from the collision at a = 0 pointing
    # directly away from a*. Walk a circle around 0; the labels may swap
    # exactly once, at angle arg(-a*).
    n_samp = 720
    ang = 2 * np.pi * np.arange(n_samp) / n_samp
    labs = [cub.critical_point_labels(0.5 * complex(math.cos(t), math.sin(t))) for t in ang]
    jumps = []
    for i in range(n_samp):
        l0, l1 = labs[i], labs[(i + 1) % n_samp]
        if abs(l0[0] - l1[0]) > abs(l0[0] - l1[1]):     # label 0 landed nearer old label 1
            jumps.append(ang[i] + np.pi / n_samp)        # midpoint of the sample interval
    predicted = (math.atan2(-a_star.imag, -a_star.real)) % (2 * np.pi)
    check(len(jumps) == 1 and abs(((jumps[0] - predicted + np.pi) % (2 * np.pi)) - np.pi)
          <= 2 * np.pi / n_samp,
          f"around |a| = 0.5 the labels swap exactly once ({len(jumps)} found), on the ray "
          f"from 0 directly away from a* (angle {math.degrees(predicted):.1f} deg)")
    check(all(len(l) == 2 for l in labs), "two free critical points at every sample")

    # Viewport independence: the same parameter gets the same PerCritical
    # value in two different viewports (pixels whose coordinates coincide
    # exactly, computed with the engine's own Viewport.coord formula).
    set_l = cdx.RenderSettings(150, 2.0, 1e-6, 0)
    vpA = cdx.Viewport(0j, 1.5, 61)          # pixel size 0.05
    vpB = cdx.Viewport(0.25 - 0.5j, 1.0, 41)  # pixel size 0.05, shifted
    imA = render_map(cub, 0j, vpA, set_l, "parameter", strategy="per_critical", critical_index=1)[0]
    imB = render_map(cub, 0j, vpB, set_l, "parameter", strategy="per_critical", critical_index=1)[0]

    def coords(vp):
        s_ = 2.0 * vp.scale / (vp.resolution - 1)
        idx = np.arange(vp.resolution)
        return vp.center.real - vp.scale + s_ * idx, vp.center.imag - vp.scale + s_ * idx

    xa, ya = coords(vpA)
    xb, yb = coords(vpB)
    pairs = 0
    same = True
    for cb, xv in enumerate(xb):
        ca = np.nonzero(xa == xv)[0]
        if not len(ca):
            continue
        for rb, yv in enumerate(yb):
            ra = np.nonzero(ya == yv)[0]
            if not len(ra):
                continue
            pairs += 1
            same &= bool(imA[ra[0], ca[0]] == imB[rb, cb])
    check(pairs >= 100 and same, f"PerCritical is identical at the {pairs} parameters the two "
                                 "viewports share -- labels do not depend on the viewport")

    # critical_index selects the same orbit in "parameter" and "parameter_period".
    vp_x = cdx.Viewport(-0.6 + 0.2j, 0.6, 31)
    agree = True
    for k in (0, 1):
        pc = render_map(cub, 0j, vp_x, set_l, "parameter", strategy="per_critical", critical_index=k)
        pp = render_map(cub, 0j, vp_x, set_l, "parameter_period", critical_index=k)
        escaped_pc = pc[1] != 0
        escaped_pp = pp[2] == 1.0
        decided = (pp[1] == 0.0)                      # period mode resolved this orbit
        agree &= np.mean(escaped_pc[decided] == escaped_pp[decided]) > 0.99
    check(agree, "critical_index k is the same free critical point in parameter (PerCritical) "
                 "and parameter_period")

    # ---- render cache -----------------------------------------------------------
    print("\nrender cache:")
    s.map = cdx.RationalMap.mandelbrot()
    s.param = 0j
    vp = cdx.Viewport(complex(0, 0), 1.5, 41)
    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)
    s.cache.clear()

    before = s.cache.stats
    first = s.render(vp, "julia")
    after_first = s.cache.stats
    check(after_first.misses == before.misses + 1 and after_first.entry_count == 1,
          "Session.render() populates the cache on a miss")

    second = s.render(vp, "julia")
    after_second = s.cache.stats
    check(after_second.hits == after_first.hits + 1 and after_second.misses == after_first.misses,
          "an identical Session.render() call is a cache hit, not a second render")
    check(np.array_equal(first, second), "a cache hit returns the same pixels as the original render")

    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 2)   # only threads differs
    after_threads = s.cache.stats
    s.render(vp, "julia")
    check(s.cache.stats.hits == after_threads.hits + 1,
          "threads is not part of the cache key -- changing only thread count still hits")

    vp = cdx.Viewport(complex(0, 0), 1.5, 43)   # a real, output-affecting change
    misses_before_res_change = s.cache.stats.misses
    s.render(vp, "julia")
    check(s.cache.stats.misses == misses_before_res_change + 1,
          "changing resolution is a cache miss (a different key), not a stale hit")

    # P5b integration requirement: term edits and parameter changes must
    # invalidate the right cache keys and nothing else. make_key() folds in
    # map.serialize() (the FULL term content, not just object identity) and
    # param as separate components (see render_cache.py's own docstring),
    # so this should already hold "by construction" -- confirmed directly
    # here rather than just asserted, since a cache key silently missing a
    # real input is exactly the kind of bug that only shows up as a stale
    # image, not an exception.
    s.render(vp, "julia")   # re-establish a hit baseline at the CURRENT (43-res) key
    hits_before_edit = s.cache.stats.hits
    misses_before_edit = s.cache.stats.misses
    entries_before_edit = s.cache.stats.entry_count
    s.map.poly_terms()[0].coeff = 3 + 0j   # mandelbrot's z^2 term, live-mutated in place
    s.render(vp, "julia")
    check(s.cache.stats.misses == misses_before_edit + 1 and s.cache.stats.hits == hits_before_edit,
          "editing a term is a cache miss, not a stale hit -- even though viewport/settings/mode "
          "are all unchanged and s.map is the SAME Python object (mutated in place, not reassigned)")
    check(s.cache.stats.entry_count == entries_before_edit + 1,
          "the edit adds exactly one new entry -- the old (now-unreachable) one isn't touched, "
          "not silently overwritten or duplicated")

    misses_before_param = s.cache.stats.misses
    s.param = s.param + 0.01
    s.render(vp, "julia")
    check(s.cache.stats.misses == misses_before_param + 1,
          "changing the parameter is also a cache miss, not a stale hit -- param is part of "
          "the key even though it isn't part of map.serialize() at all")

    # A cancelled render must not poison the cache with a partial result --
    # cancelled BEFORE the render call (not raced against a slow render) so
    # this is deterministic: render_map still checks `cancel` after calling
    # into the engine, so a token that was already cancelled on entry never
    # gets to populate the cache at all.
    cancel_key_viewport = cdx.Viewport(complex(0.4, 0.4), 1.2, 41)
    cancel_token = cdx.CancelToken()
    cancel_token.cancel()
    entries_before_cancel = s.cache.stats.entry_count
    render_map(s.map, s.param, cancel_key_viewport, s.render_settings, "julia",
              cancel_token, s.cache)
    check(s.cache.stats.entry_count == entries_before_cancel,
          "a cancelled render's partial result is never stored in the cache")

    misses_before_retry = s.cache.stats.misses
    render_map(s.map, s.param, cancel_key_viewport, s.render_settings, "julia", None, s.cache)
    check(s.cache.stats.misses == misses_before_retry + 1,
          "a subsequent lookup for that same key still misses -- nothing was cached under it")

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

    # ---- term editing: reorder --------------------------------------------------
    print("\nterm editing (reorder):")
    s2r = Session()
    s2r.map = cdx.RationalMap("scratch-reorder")
    s2r.add_poly_term(1 + 0j, 2, 0, "first")
    s2r.add_poly_term(2 + 0j, 3, 0, "second")
    s2r.add_poly_term(3 + 0j, 4, 0, "third")

    ok = s2r.move_poly_term(0, +1)
    labels = [t.label for t in s2r.map.poly_terms()]
    check(ok and labels == ["second", "first", "third"],
          "move_poly_term(0, +1) swaps with its neighbor, not a silent duplicate "
          "(the bind_vector aliasing trap this is written to avoid)")

    ok = s2r.move_poly_term(0, -1)
    check(not ok and [t.label for t in s2r.map.poly_terms()] == ["second", "first", "third"],
          "move_poly_term at index 0 with direction -1 is a no-op (no neighbor there)")

    ok = s2r.move_poly_term(2, +1)
    check(not ok, "move_poly_term at the last index with direction +1 is a no-op")

    s2r.add_pole_term(0j, 1 + 0j, 1, 0, "polefirst")
    s2r.add_pole_term(5 + 0j, 2 + 0j, 1, 0, "polesecond")
    before_eval = s2r.map.eval(3 + 0j, 0j)
    ok = s2r.move_pole_term(0, +1)
    after_eval = s2r.map.eval(3 + 0j, 0j)
    check(ok and [t.label for t in s2r.map.pole_terms()] == ["polesecond", "polefirst"],
          "move_pole_term swaps pole terms the same way")
    check(after_eval == before_eval,
          "reordering terms never changes what the map evaluates to -- it's a sum either way")

    # ---- term editing: pole-location uniqueness / negative-exponent rejection ----
    print("\nterm editing (validation):")
    s2b = Session()
    s2b.map = cdx.RationalMap("scratch2")
    try:
        s2b.add_poly_term(1 + 0j, -2)
        check(False, "add_poly_term with a negative exponent raises")
    except ValueError as e:
        check("negative exponent" in str(e), "add_poly_term raises ValueError, message explains why")
    check(len(s2b.map.poly_terms()) == 0, "the rejected add_poly_term added nothing")

    s2b.add_pole_term(1 + 0j, 1 + 0j, 1)
    try:
        s2b.add_pole_term(1 + 0j, 2 + 0j, 3)
        check(False, "add_pole_term at an already-occupied location raises")
    except ValueError as e:
        check("already exists" in str(e), "add_pole_term raises ValueError, message names the conflict")
    check(len(s2b.map.pole_terms()) == 1, "the rejected add_pole_term added nothing")

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

    # ---- library: presets are read-only --------------------------------------------
    print("\nlibrary (preset read-only guards):")
    s3b = Session()
    try:
        s3b.save_to_library("mandelbrot")
        check(False, "save_to_library over a preset name raises")
    except ValueError as e:
        check("preset" in str(e), "save_to_library raises ValueError naming the preset")
    s3b.map.name = "mandelbrot"
    try:
        s3b.save_to_library()   # name=None -- uses self.map.name, still a preset
        check(False, "save_to_library() with self.map.name already a preset also raises")
    except ValueError:
        check(True, "the guard applies whether the name comes from the argument or self.map.name")

    try:
        s3b.rename_in_library("mandelbrot", "not-mandelbrot")
        check(False, "renaming a preset raises")
    except ValueError:
        check(True, "rename_in_library raises ValueError for a preset source name")
    try:
        s3b.save_to_library("scratch-for-rename")
        s3b.rename_in_library("scratch-for-rename", "mandelbrot")
        check(False, "renaming ONTO a preset name raises")
    except ValueError:
        check(True, "rename_in_library raises ValueError for a preset target name")

    try:
        s3b.delete_from_library("mandelbrot")
        check(False, "deleting a preset raises")
    except ValueError:
        check(True, "delete_from_library raises ValueError for a preset name")

    try:
        s3b.set_library_notes("mandelbrot", "nope")
        check(False, "editing a preset's notes raises")
    except ValueError:
        check(True, "set_library_notes raises ValueError for a preset name")

    # ---- library: rename / delete / notes on a real user entry ---------------------
    print("\nlibrary (rename/delete/notes):")
    s3c = Session()
    s3c.map = cdx.RationalMap("scratch")
    s3c.save_to_library("fam-a")
    s3c.set_library_notes("fam-a", "first attempt")
    check(s3c.library.find("fam-a").notes == "first attempt", "set_library_notes updates the entry")

    try:
        s3c.set_library_notes("does-not-exist", "x")
        check(False, "editing notes on a missing name raises")
    except KeyError:
        check(True, "set_library_notes raises KeyError for a missing name")

    s3c.rename_in_library("fam-a", "fam-b")
    check("fam-b" in s3c.library.names() and "fam-a" not in s3c.library.names(),
          "rename_in_library moves the entry to the new name")
    check(s3c.library.find("fam-b").notes == "first attempt", "the renamed entry keeps its notes")

    s3c.save_to_library("fam-c")
    try:
        s3c.rename_in_library("fam-b", "fam-c")
        check(False, "renaming onto an existing (non-preset) name raises")
    except ValueError as e:
        check("already exists" in str(e), "rename_in_library names the collision")

    try:
        s3c.rename_in_library("does-not-exist", "whatever")
        check(False, "renaming a missing source name raises")
    except KeyError:
        check(True, "rename_in_library raises KeyError for a missing source name")

    check(s3c.delete_from_library("fam-b") is None, "delete_from_library succeeds silently")
    check("fam-b" not in s3c.library.names(), "the deleted entry is gone")
    try:
        s3c.delete_from_library("fam-b")
        check(False, "deleting an already-gone name raises")
    except KeyError:
        check(True, "delete_from_library raises KeyError for a missing name")

    # ---- library: user-only persistence (save_user_library/load_user_library) ------
    print("\nlibrary (user-only persistence):")
    with tempfile.TemporaryDirectory() as tmp:
        lib_path = os.path.join(tmp, "library.txt")

        s5a = Session()
        s5a.map = cdx.RationalMap("scratch")
        s5a.save_to_library("only-user-family")
        s5a.save_user_library(lib_path)
        saved_text = open(lib_path).read()
        check("only-user-family" in saved_text and "mandelbrot" not in saved_text,
              "save_user_library writes the user entry but none of the six presets")

        s5b = Session()
        check("only-user-family" not in s5b.library.names(),
              "sanity: a fresh Session doesn't have the user family yet")
        s5b.load_user_library(lib_path)
        check("only-user-family" in s5b.library.names(),
              "load_user_library merges the saved user family in")
        check(set(PRESET_FAMILY_NAMES) <= set(s5b.library.names()),
              "load_user_library never removes any of the six presets")

        s5c = Session()
        s5c.load_user_library(os.path.join(tmp, "does-not-exist.txt"))
        check(set(s5c.library.names()) == PRESET_FAMILY_NAMES,
              "load_user_library on a missing file is a silent no-op, not an error")

        malformed_path = os.path.join(tmp, "malformed.txt")
        with open(malformed_path, "w") as f:
            f.write("this is not a valid library file\n")
        s5d = Session()
        s5d.load_user_library(malformed_path)   # must not raise
        check(set(s5d.library.names()) == PRESET_FAMILY_NAMES,
              "load_user_library on a malformed file is also a silent no-op, not an error")

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

    # ---- attracting fixed points: complete_attractors' union (cosmetic batch) ------
    # Nova, a=(-0.345714,-0.100000): cdx.find_attractors mislabels the
    # genuinely attracting fixed point (0.74392,-0.0516) as a spurious
    # period-9 cycle (see cdx/test/test_analysis.cpp's own "complete_
    # attractors" section for how this real case was found and root-
    # caused) -- dynamical_facts() must show it correctly as period 1.
    print("\nattracting fixed points: complete_attractors' union (real Nova reproduction):")
    nova = cdx.RationalMap("nova")
    nova.add_poly(2.0 / 3.0 + 0j, 1, 0, "(2/3)z")
    nova.add_pole(0j, 1.0 / 3.0 + 0j, 2, 0, "(1/3)z^-2")
    nova.add_poly(1 + 0j, 0, 1, "a")
    a_repro = -0.345714 - 0.1j
    target = 0.74392 - 0.0516219j

    s7 = Session()
    s7.map = nova
    s7.param = a_repro
    facts_nova = s7.dynamical_facts()

    target_fp = next((fp for fp in facts_nova.fixed_points if abs(fp.point - target) < 1e-3), None)
    check(target_fp is not None and 0.9 < abs(target_fp.multiplier) < 0.9701,
          "sanity: fixed_points() reports this point attracting, |mult|~0.965")

    broken_cycles = cdx.find_attractors(nova, a_repro)
    period9_present = any(len(c.points) == 9 and abs(c.points[0] - target) < 1e-3
                          for c in broken_cycles)
    # Used to assert the mislabel was still present; find_attractors now
    # reduces closures to their minimal period, so it is gone at the source.
    check(not period9_present,
          "cdx.find_attractors no longer mislabels this point as a period-9 cycle "
          "(minimal-period reduction in find_attractors)")

    target_ac = next((ac for ac in facts_nova.attracting_cycles
                      if len(ac.points) == 1 and abs(ac.points[0] - target) < 1e-3), None)
    check(target_ac is not None and target_ac.period == 1,
          "FIX: session.dynamical_facts()'s attracting_cycles table shows this point "
          "correctly as its own period-1 entry -- the fact sheet the user actually sees "
          "is correct even though find_attractors' own raw output is not")

    complete_cycles = cdx.complete_attractors(nova, a_repro)
    check(any(len(c.points) == 1 and abs(c.points[0] - target) < 1e-3 for c in complete_cycles),
          "cdx.complete_attractors (exposed to Python, what session.py's basin mode now "
          "uses) recovers the same correct period-1 entry directly")

    # basin-mode render: a pixel AT this fixed point must resolve to a real
    # basin, not stay unresolved or only match the spurious period-9 cycle.
    tiny_vp = cdx.Viewport(target, 0.01, 3)
    basin_repro = s7.render(tiny_vp, "basin")
    center_label = basin_repro[0][1, 1]
    check(center_label != 0.0,
          "session.render(..., 'basin') resolves the pixel at this fixed point to a real "
          "attractor id -- this is the basin-mode region the fix claims, through the "
          "actual render path the app uses")

    # ---- experiment snapshots: round-trip (map/param/layout/settings/orbit) --------
    print("\nexperiment snapshots (snapshot_to_dict/restore_from_snapshot):")

    # A non-trivial, custom (not-a-preset) map -- to_formula() is the
    # cleanest thing to compare on, since RationalMap has no __eq__.
    # Neither a viewport/render_mode NOR a notion of "panes" is Session's
    # own (see app.pane.Pane) -- snapshot_to_dict/restore_from_snapshot
    # now take/return a full LAYOUT (one (viewport, render_mode) pair per
    # pane, which pane is focused, coupled/single) explicitly, the same
    # way a real SandboxWindow would supply/receive it (Stage 4).
    src = Session()
    custom_map = cdx.RationalMap("scratch")
    custom_map.add_poly(complex(1, 0), 3, 0, "z^3")
    custom_map.add_pole(complex(0.5, -0.5), complex(1, 0), 2, 1, "pole")
    src.map = custom_map
    src.param = complex(0.123, -0.456)
    src_pane_a = (cdx.Viewport(complex(0.01, -0.02), 1e-4, 333), "basin")   # a genuinely zoomed view
    src_pane_b = (cdx.Viewport(complex(-0.5, 0.0), 1.5, 400), "julia")
    src_panes = [src_pane_a, src_pane_b]
    src_focused_index = 1
    src_coupled = False
    src.render_settings = cdx.RenderSettings(321, 3.5, 1e-8, 2)

    z0 = complex(0.3, -0.1)
    d = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled, orbit=(1, z0, 4))
    dst = Session()
    dst_panes, dst_focused_index, dst_coupled, orbit, _preview = dst.restore_from_snapshot(d)

    check(dst.map.to_formula() == src.map.to_formula(), "round-trip: map formula matches")
    check(dst.param == src.param, "round-trip: param matches")
    check(len(dst_panes) == 2, "round-trip: both panes survive, not just one")
    for i, (src_vp, src_mode) in enumerate(src_panes):
        dst_vp, dst_mode = dst_panes[i]
        check(dst_vp.center == src_vp.center and dst_vp.scale == src_vp.scale and
              dst_vp.resolution == src_vp.resolution,
              f"round-trip: pane {i}'s viewport (center/scale/resolution) matches")
        check(dst_mode == src_mode, f"round-trip: pane {i}'s render_mode matches")
    check(dst_focused_index == src_focused_index, "round-trip: which pane is focused matches")
    check(dst_coupled == src_coupled, "round-trip: coupled/single layout flag matches")
    check(dst.render_settings.max_iter == src.render_settings.max_iter and
          dst.render_settings.escape_radius == src.render_settings.escape_radius and
          dst.render_settings.tol == src.render_settings.tol and
          dst.render_settings.threads == src.render_settings.threads,
          "round-trip: render_settings (all four fields) matches")
    check(orbit is not None and orbit[0] == 1 and orbit[1] == z0 and orbit[2] == 4,
          "round-trip: orbit's owning pane index, z0, and n all survive")

    # z/history are DERIVED, not stored -- reconstruct via the tracker's
    # REAL API (exactly what app/sandbox.py's Open Experiment does) and
    # confirm the regenerated orbit matches what stepping the ORIGINAL
    # map/param the same way would have produced.
    restored_tracker = OrbitTracker()
    restored_tracker.seed(dst.map, dst.param, orbit[1])
    restored_tracker.step(dst.map, dst.param, orbit[2])
    reference_tracker = OrbitTracker()
    reference_tracker.seed(src.map, src.param, z0)
    reference_tracker.step(src.map, src.param, 4)
    check(restored_tracker.state.z == reference_tracker.state.z,
          "round-trip: the regenerated orbit's current z matches stepping the ORIGINAL "
          "map/param the same number of times from the same z0")
    check(restored_tracker.state.history == reference_tracker.state.history,
          "round-trip: the regenerated orbit's full history matches too, not just the "
          "final point")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.cdsx")
        src.save_snapshot(path, src_panes, src_focused_index, src_coupled, orbit=(1, z0, 4))
        dst_file = Session()
        (panes_file, focused_index_file, coupled_file, orbit_file,
         _preview_file) = dst_file.load_snapshot(path)
        check(dst_file.map.to_formula() == src.map.to_formula() and dst_file.param == src.param
              and focused_index_file == src_focused_index and coupled_file == src_coupled
              and panes_file[0][1] == src_pane_a[1] and panes_file[1][1] == src_pane_b[1]
              and orbit_file == (1, z0, 4),
              "save_snapshot/load_snapshot round-trips through an ACTUAL JSON file, not just "
              "the in-memory dict -- the float encode/decode step is lossless too")

    # ---- orbit-null round-trip -------------------------------------------------------
    d_null = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled, orbit=None)
    dst_null = Session()
    (_panes_null, _focus_null, _coupled_null,
     orbit_null, _preview_null) = dst_null.restore_from_snapshot(d_null)
    check(orbit_null is None, "orbit-null round-trip: no active orbit survives as no orbit")

    # ---- parameter-plane mode with no orbit -------------------------------------------
    param_plane_session = Session()
    param_plane_session.map = cdx.RationalMap.mandelbrot()
    d_pp = param_plane_session.snapshot_to_dict(
        [(src_pane_a[0], "parameter"), src_pane_b], 0, True, orbit=None)
    dst_pp = Session()
    panes_pp, _focus_pp, _coupled_pp, orbit_pp, _preview_pp = dst_pp.restore_from_snapshot(d_pp)
    check(orbit_pp is None and panes_pp[0][1] == "parameter",
          "a snapshot taken with a parameter-plane pane and no orbit restores cleanly, with "
          "nothing to seed")

    # ---- embedded preview (Stage C): round-trips, and is optional on read ------------
    print("\nexperiment snapshots (embedded preview thumbnail):")
    d_preview = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled,
                                     orbit=None, preview_png_base64="not-really-a-png-just-a-marker")
    dst_preview = Session()
    (_panes_pv, _focus_pv, _coupled_pv,
     _orbit_pv, preview_pv) = dst_preview.restore_from_snapshot(d_preview)
    check(preview_pv == "not-really-a-png-just-a-marker",
          "a snapshot saved WITH a preview restores that exact preview string")

    d_no_preview = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled, orbit=None)
    dst_no_preview = Session()
    (_panes_np, _focus_np, _coupled_np,
     _orbit_np, preview_np) = dst_no_preview.restore_from_snapshot(d_no_preview)
    check(preview_np is None,
          "a snapshot saved WITHOUT a preview restores preview as None, not an error")

    # A hand-built version-2 dict -- the actual pre-Stage-C shape, with NO
    # "preview" key at all (not even present, unlike d_no_preview above
    # which is a version-3 dict whose preview happens to be None) -- must
    # still load cleanly: see MIN_SNAPSHOT_SCHEMA_VERSION's own comment for
    # why this, unlike the version-1 case below, is accepted rather than
    # rejected.
    v2_snapshot = {
        "schema_version": 2, "app_version": "0.0.0-test",
        "map": src.map.serialize(), "param": [src.param.real, src.param.imag],
        "render_settings": {"max_iter": src.render_settings.max_iter,
                            "escape_radius": src.render_settings.escape_radius,
                            "tol": src.render_settings.tol,
                            "threads": src.render_settings.threads},
        "layout": {"coupled": src_coupled, "focused_index": src_focused_index,
                   "panes": [{"render_mode": mode,
                              "viewport": {"center": [vp.center.real, vp.center.imag],
                                          "scale": vp.scale, "resolution": vp.resolution}}
                             for vp, mode in src_panes]},
        "orbit": None,
    }
    dst_v2 = Session()
    (panes_v2, focus_v2, coupled_v2,
     orbit_v2, preview_v2) = dst_v2.restore_from_snapshot(v2_snapshot)
    check(dst_v2.map.to_formula() == src.map.to_formula() and focus_v2 == src_focused_index
          and coupled_v2 == src_coupled and len(panes_v2) == 2 and orbit_v2 is None,
          "an older (version-2, pre-Stage-C) snapshot with no 'preview' key at all still "
          "loads fine -- not just rejected the way a version-1 snapshot is")
    check(preview_v2 is None,
          "...with preview correctly reported as None, not a KeyError")

    # ---- malformed / wrong-version input: raises, changes NOTHING --------------------
    print("\nexperiment snapshots (validation -- malformed input never mutates state):")
    guard_session = Session()
    guard_session.map = cdx.RationalMap.mandelbrot()
    guard_session.param = complex(-0.7269, 0.1889)
    before_map = guard_session.map.serialize()   # pre-captured, compared byte-for-byte after
    before_param = guard_session.param
    before_max_iter = guard_session.render_settings.max_iter

    valid_map_text = guard_session.map.serialize()
    valid_layout = {"coupled": True, "focused_index": 0,
                    "panes": [{"render_mode": "julia",
                               "viewport": {"center": [0, 0], "scale": 1, "resolution": 10}}]}
    bad_snapshots = [
        {},                                   # missing schema_version entirely
        {"schema_version": 999},              # wrong version
        {"schema_version": 1, "map": valid_map_text, "param": [0, 0],   # the OLD (Stage 1) shape,
         "viewport": {"center": [0, 0], "scale": 1, "resolution": 10},  # rejected by version, not
         "render_mode": "julia",                                        # migrated -- see
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0}},  # SNAPSHOT_SCHEMA_VERSION's own comment
        "not even a dict",
        None,
        {"schema_version": 2, "map": "not a real serialized map", "param": [0, 0],
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": valid_layout},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": {"coupled": True, "focused_index": 0,
                    "panes": [{"render_mode": "not_a_real_mode",
                               "viewport": {"center": [0, 0], "scale": 1, "resolution": 10}}]}},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],   # focused_index out of range
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": {"coupled": True, "focused_index": 5,
                    "panes": [{"render_mode": "julia",
                               "viewport": {"center": [0, 0], "scale": 1, "resolution": 10}}]}},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],   # layout.panes empty
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": {"coupled": True, "focused_index": 0, "panes": []}},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],   # orbit pane_index out of range
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": valid_layout,
         "orbit": {"pane_index": 9, "z0": [0, 0], "n": 0}},
    ]
    for bad in bad_snapshots:
        try:
            guard_session.restore_from_snapshot(bad)
            check(False, f"malformed input {bad!r} should have raised ValueError")
        except ValueError:
            check(True, f"malformed/wrong-version input ({type(bad).__name__}) raises ValueError")

    check(guard_session.map.serialize() == before_map,
          "the live session's map is BYTE-FOR-BYTE unchanged after every rejected load")
    check(guard_session.param == before_param and
          guard_session.render_settings.max_iter == before_max_iter,
          "...and so is everything else Session still owns -- no half-applied load")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
