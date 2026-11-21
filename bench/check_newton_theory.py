"""bench/check_newton_theory.py -- the engine against the established results for
relaxed Newton N_a(z) = z - a (z^n - 1)/(n z^(n-1)), through the Python API.

  T2  |a - 1| < 1   the n roots of unity are attracting fixed points
                    (multiplier 1 - a); outside the closed disc they are not
  T3  |a - n| > n   infinity is an attracting fixed point (multiplier
                    n/(n - a)); inside it is not
  T4  a_k = n - (n-1) w_k, w_k = exp(2 pi i k/n): the free critical points
      z_c w^j, z_c^n = a(n-1)/(n-a), satisfy N(z_c) = w_k z_c -- a
      superattracting star cycle of period n/gcd(k, n)
  T5  at and near a_k the n free critical points fall into gcd(k, n) distinct
      cycles, so that many attracting cycles coexist

Deterministic output: run it before and after an engine change and diff the
text. Exit status 1 on any failed check.

    PYTHONPATH=cdx/build python -m bench.check_newton_theory
"""

from __future__ import annotations

import cmath
import math

import numpy as np

import cdx

failures = 0


def check(cond, what):
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def relaxed_newton(n):
    return cdx.RationalMap.from_expression(f"z - a*(z^{n}-1)/({n}*z^{n - 1})", "a", {}, f"relaxed_newton{n}")


def is_inf(z):
    return not (math.isfinite(z.real) and math.isfinite(z.imag))


def fixed_attractors(cycles):
    return [c.points[0] for c in cycles if len(c.points) == 1]


def roots_attracting(m, n, a):
    pts = [p for p in fixed_attractors(cdx.complete_attractors(m, a)) if not is_inf(p)]
    return all(any(abs(p - cmath.exp(2j * math.pi * j / n)) < 1e-6 for p in pts) for j in range(n))


def infinity_attracting(m, a):
    return any(is_inf(p) for p in fixed_attractors(cdx.complete_attractors(m, a)))


def free_critical_points(n, a):
    base = (a * (n - 1) / (n - a)) ** (1.0 / n)
    return [base * cmath.exp(2j * math.pi * j / n) for j in range(n)]


def main():
    rng = np.random.default_rng(20261113)
    for n in (5, 8):
        m = relaxed_newton(n)
        print(f"\n=== n = {n} ===")

        inside = [1 + r * cmath.exp(1j * t) for r, t in zip(rng.uniform(0.05, 0.97, 40), rng.uniform(0, 2 * math.pi, 40))]
        outside = [1 + r * cmath.exp(1j * t) for r, t in zip(rng.uniform(1.03, 1.6, 40), rng.uniform(0, 2 * math.pi, 40))]
        check(all(roots_attracting(m, n, a) for a in inside), "T2: all n roots attracting at 40 sampled a with |a-1| < 1")
        check(not any(roots_attracting(m, n, a) for a in outside), "T2: roots not attracting at 40 sampled a with |a-1| > 1")

        far = [n + r * cmath.exp(1j * t) for r, t in zip(rng.uniform(1.03 * n, 2.0 * n, 40), rng.uniform(0, 2 * math.pi, 40))]
        near = [n + r * cmath.exp(1j * t) for r, t in zip(rng.uniform(0.2 * n, 0.97 * n, 40), rng.uniform(0, 2 * math.pi, 40))]
        check(all(infinity_attracting(m, a) for a in far), "T3: infinity attracting at 40 sampled a with |a-n| > n")
        check(not any(infinity_attracting(m, a) for a in near), "T3: infinity not attracting at 40 sampled a with |a-n| < n")

        for k in range(1, n):
            w = cmath.exp(2j * math.pi * k / n)
            a = n - (n - 1) * w
            g = math.gcd(k, n)
            seeds = free_critical_points(n, a)
            outcomes = cdx.per_seed_outcomes(seeds, m, a)
            periods = sorted({o.period for o in outcomes if o.resolved and not o.is_infinity})
            mult = 1.0 + 0j
            z = seeds[0]
            for _ in range(n // g):
                mult *= m.deriv(z, a)
                z = m.eval(z, a)
            check(all(o.resolved for o in outcomes) and periods == [n // g] and abs(mult) < 1e-6
                  and abs(z - seeds[0]) < 1e-9,
                  f"T4: k={k}: centre a={a:.6f} has a superattracting cycle of period n/gcd = {n // g} "
                  f"(periods found {periods}, |multiplier| = {abs(mult):.1e})")
            counts = []
            for da in (0j, 1e-3, -1e-3j, 7e-4 + 7e-4j):
                cyc = cdx.find_attractors_from_seeds(free_critical_points(n, a + da), m, a + da) \
                    if hasattr(cdx, "find_attractors_from_seeds") else cdx.complete_attractors(m, a + da)
                counts.append((len(cyc), sorted(len(c.points) for c in cyc)))
            check(all(c == (g, [n // g] * g) for c in counts),
                  f"T5: k={k}: gcd(k,n) = {g} coexisting cycles of period {n // g} at and near the centre "
                  f"(found {counts[0][0]} at the centre; nearby {[c[0] for c in counts[1:]]})")

        # the same facts as the parameter-plane renders see them, on a coarse grid
        res = 41
        vp = cdx.Viewport(complex(n), float(n) * 1.3, res)
        r = cdx.Renderer(map=cdx.Map.custom(m, 1 + 0j), viewport=vp, settings=cdx.RenderSettings(500, 2.0, 1e-8, 0))
        counts, unresolved, certain = (np.asarray(v) for v in r.render_parameter_basin(None))
        s = 2 * vp.scale / (res - 1)
        A = (vp.center.real - vp.scale + s * np.arange(res))[None, :] + 1j * (vp.center.imag - vp.scale + s * np.arange(res))[:, None]
        in_t2 = np.abs(A - 1) < 0.97
        in_t3 = np.abs(A - n) > 1.03 * n
        check(bool(np.all(counts[in_t2] >= n) and np.all(certain[in_t2] + counts[in_t2] >= n)),
              f"T2 on the Parameter_basin grid: count >= n at all {int(in_t2.sum())} pixels with |a-1| < 0.97")
        check(bool(np.all(counts[in_t3] >= 1)),
              f"T3 on the Parameter_basin grid: count >= 1 at all {int(in_t3.sum())} pixels with |a-n| > 1.03 n")
        print(f"  grid digest: sum(counts) = {int(counts.sum())}, sum(unresolved) = {int(unresolved.sum())}, "
              f"sum(certain) = {int(certain.sum())}")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} ({failures} failures)")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
