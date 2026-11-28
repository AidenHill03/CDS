"""test_special_polys.py -- the parser's special polynomials against sympy.

sympy is a TEST dependency only (requirements-dev.txt), never bundled. It is
an independent implementation: its tables come from its own formulas, not
from cdx's recurrences. Bessel polynomials (absent from sympy) are checked
against the explicit sum y_n(x) = sum_k (n+k)!/((n-k)! k!) (x/2)^k in exact
rationals.

Run from the repository root:
    PYTHONPATH=cdx/build python cdx/test/test_special_polys.py
"""

from __future__ import annotations

import sys

import sympy as sp

import cdx

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


X = sp.Symbol("x")


def sympy_coeffs(expr, subs: dict | None = None) -> list[complex]:
    expr = sp.expand(expr.subs(subs or {}))
    poly = sp.Poly(expr, X)
    out = [complex(sp.N(c, 30)) for c in reversed(poly.all_coeffs())]
    while out and out[-1] == 0:
        out.pop()
    return out


def cdx_coeffs(src: str, params: dict | None = None) -> list[complex]:
    P, Q = cdx.parse_rational_coefficients(src, params or {})
    assert len(Q) == 1, f"{src}: expected a polynomial"
    out = [c / Q[0] for c in P]
    while out and out[-1] == 0:
        out.pop()
    return out


def same(a: list[complex], b: list[complex], rel: float) -> bool:
    if len(a) != len(b):
        return False
    scale = max([abs(c) for c in b] + [1.0])
    # Relative to the largest coefficient (cancellation in a single small
    # coefficient is judged against the polynomial's own scale) AND to the
    # coefficient itself where it is not tiny.
    return all(abs(x - y) <= rel * max(abs(y), 1e-3 * scale) for x, y in zip(a, b))


def bessel_sympy(n: int):
    return sum(sp.factorial(n + k) / (sp.factorial(n - k) * sp.factorial(k)) * (X / 2) ** k
               for k in range(n + 1))


FAMILIES = [
    # (cdx call template, sympy expression builder)
    ("hermite({n}, z)", lambda n: sp.hermite(n, X)),
    ("hermite_e({n}, z)", lambda n: sp.hermite_prob(n, X)),
    ("legendre({n}, z)", lambda n: sp.legendre(n, X)),
    ("laguerre({n}, z)", lambda n: sp.laguerre(n, X)),
    ("laguerre({n}, 0.5, z)", lambda n: sp.assoc_laguerre(n, sp.Rational(1, 2), X)),
    ("laguerre({n}, 3, z)", lambda n: sp.assoc_laguerre(n, 3, X)),
    ("chebyshev_t({n}, z)", lambda n: sp.chebyshevt(n, X)),
    ("chebyshev_u({n}, z)", lambda n: sp.chebyshevu(n, X)),
    ("gegenbauer({n}, 1.5, z)", lambda n: sp.gegenbauer(n, sp.Rational(3, 2), X)),
    ("gegenbauer({n}, 0.25, z)", lambda n: sp.gegenbauer(n, sp.Rational(1, 4), X)),
    ("jacobi({n}, 1, 2, z)", lambda n: sp.jacobi_poly(n, 1, 2, X)),
    ("jacobi({n}, 0.5, -0.5, z)", lambda n: sp.jacobi_poly(n, sp.Rational(1, 2), sp.Rational(-1, 2), X)),
    ("bessel_poly({n}, z)", bessel_sympy),
    ("bernoulli({n}, z)", lambda n: sp.bernoulli(n, X)),
    ("euler_poly({n}, z)", lambda n: sp.euler(n, X)),
]


def main() -> int:
    print("every family, n = 0..8, against sympy:")
    for template, build in FAMILIES:
        bad = [n for n in range(9)
               if not same(cdx_coeffs(template.format(n=n)), sympy_coeffs(build(n)), 1e-12)]
        check(not bad, f"{template.format(n='n')}: n = 0..8" + (f" (mismatch at n = {bad})" if bad else ""))
    bad = [n for n in range(1, 9)
           if not same(cdx_coeffs(f"cyclotomic({n}, z)"), sympy_coeffs(sp.cyclotomic_poly(n, X)), 0)]
    check(not bad, "cyclotomic(n, z): n = 1..8, exact" + (f" (mismatch at n = {bad})" if bad else ""))
    bad = [n for n in (12, 30, 36, 40)
           if not same(cdx_coeffs(f"cyclotomic({n}, z)"), sympy_coeffs(sp.cyclotomic_poly(n, X)), 0)]
    check(not bad, "cyclotomic(n, z): n = 12, 30, 36, 40, exact")

    print("\nsymbolic family parameters, n = 0..8 (evaluated at several values):")
    a = sp.Symbol("a")
    for template, build, values in [
        ("laguerre({n}, a, z)", lambda n: sp.assoc_laguerre(n, a, X), [0.37 + 0.2j, -0.8, 4.0]),
        ("gegenbauer({n}, a, z)", lambda n: sp.gegenbauer(n, a, X), [0.37 + 0.2j, 1.5, 3.0]),
        ("jacobi({n}, a, 1.5, z)", lambda n: sp.jacobi_poly(n, a, sp.Rational(3, 2), X), [0.37 + 0.2j, -0.5, 2.0]),
    ]:
        bad = []
        for n in range(9):
            for v in values:
                vs = sp.nsimplify(v) if isinstance(v, float) else sp.Float(v.real, 30) + sp.I * sp.Float(v.imag, 30)
                if not same(cdx_coeffs(template.format(n=n), {"a": complex(v)}),
                            sympy_coeffs(build(n), {a: vs}), 1e-11):
                    bad.append((n, v))
        check(not bad, f"{template.format(n='n')} with symbolic a" + (f" (mismatch: {bad[:3]})" if bad else ""))
    for n in range(9):
        info = cdx.parse_rational_info(f"jacobi({n}, a, b, z)")
        if n and info["parameters"] != ["a", "b"]:
            check(False, f"jacobi({n}, a, b, z) parameters {info['parameters']}")
            break
    else:
        check(True, "jacobi(n, a, b, z) keeps both a and b symbolic")
    bad = []
    for n in range(9):
        for av, bv in [(0.3, 1.7), (-0.4 + 0.5j, 2.0)]:
            ref = sp.jacobi_poly(n, a, sp.Symbol("b"), X).subs({a: sp.nsimplify(av) if isinstance(av, float) else
                                                           sp.Float(av.real, 30) + sp.I * sp.Float(av.imag, 30),
                                                           sp.Symbol("b"): sp.nsimplify(bv)})
            if not same(cdx_coeffs(f"jacobi({n}, a, b, z)", {"a": complex(av), "b": complex(bv)}),
                        sympy_coeffs(ref), 1e-11):
                bad.append((n, av, bv))
    check(not bad, "jacobi(n, a, b, z) matches sympy at several (a, b)" + (f" {bad[:3]}" if bad else ""))

    print("\nhigher degree:")
    check(same(cdx_coeffs("hermite(20, z)"), sympy_coeffs(sp.hermite(20, X)), 1e-15),
          "hermite(20, z) == sympy H20 (coefficients ~1e13) to 1e-15")
    check(same(cdx_coeffs("legendre(20, z)"), sympy_coeffs(sp.legendre(20, X)), 1e-13),
          "legendre(20, z) == sympy P20 to 1e-13")
    check(same(cdx_coeffs("hermite(40, z)"), sympy_coeffs(sp.hermite(40, X)), 1e-14),
          "hermite(40, z) == sympy H40 to 1e-14")
    check(same(cdx_coeffs("bernoulli(40, z)"), sympy_coeffs(sp.bernoulli(40, X)), 1e-14),
          "bernoulli(40, z) == sympy B40(x) to 1e-14")
    check(same(cdx_coeffs("euler_poly(40, z)"), sympy_coeffs(sp.euler(40, X)), 1e-14),
          "euler_poly(40, z) == sympy E40(x) to 1e-14")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
