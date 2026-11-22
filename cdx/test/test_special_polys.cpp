// =============================================================================
// test_special_polys.cpp -- the rational-expression grammar's built-in
// functions: special polynomials and d(...).
//
// Coefficient tables for EVERY family at n = 0..8, and Hermite/Legendre at
// n = 20, are checked against sympy in cdx/test/test_special_polys.py (an
// independent implementation). This file covers what needs no oracle
// library: hard-coded textbook polynomials, exact Bernoulli numbers,
// derivative identities checked by evaluation, symbolic-parameter
// coefficients against the closed form, the error and warning messages, and
// the Newton-Legendre round trip through RationalMap.
// =============================================================================
#include "cdx/rational.hpp"
#include "cdx/rational_parser.hpp"
#include "cdx/special_polys.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const std::string& what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
    if (!cond) ++failures;
}

static bool near(Cplx a, Cplx b, double tol = 1e-9) {
    return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
}

static CanonicalRational parse_ok(const std::string& src) {
    CanonicalRational cr;
    std::string err;
    if (!parse_rational(src, cr, err)) {
        std::printf("  unexpected parse failure for '%s': %s\n", src.c_str(), err.c_str());
        ++failures;
    }
    return cr;
}

static std::vector<Cplx> coeffs(const PolyZ& p, const std::map<std::string, Cplx>& params = {}) {
    std::vector<Cplx> out;
    for (const auto& c : p.coeffs) out.push_back(c->eval(params));
    return out;
}

// P has exactly these real coefficients (ascending) and Q is the constant 1.
static bool poly_is(const std::string& src, const std::vector<double>& expect, double tol = 1e-12) {
    CanonicalRational cr = parse_ok(src);
    auto P = coeffs(cr.P), Q = coeffs(cr.Q);
    if (Q.size() != 1 || !near(Q[0], 1.0)) return false;
    if (P.size() != expect.size()) return false;
    for (std::size_t k = 0; k < P.size(); ++k)
        if (!near(P[k], expect[k], tol)) return false;
    return true;
}

static std::string parse_error(const std::string& src) {
    CanonicalRational cr;
    std::string err;
    if (parse_rational(src, cr, err)) return "(accepted)";
    return err;
}

static bool starts_with(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

int main() {
    // ---- textbook polynomials --------------------------------------------------------
    std::printf("textbook polynomials (hard-coded):\n");
    check(poly_is("hermite(5, z)", {0, 120, 0, -160, 0, 32}), "H5 = 32z^5 - 160z^3 + 120z");
    check(poly_is("H(5, z)", {0, 120, 0, -160, 0, 32}), "alias H(5, z) == hermite(5, z)");
    check(poly_is("He(4, z)", {3, 0, -6, 0, 1}), "He4 = z^4 - 6z^2 + 3");
    check(poly_is("P(4, z)", {3.0 / 8, 0, -30.0 / 8, 0, 35.0 / 8}), "P4 = (35z^4 - 30z^2 + 3)/8");
    check(poly_is("T(6, z)", {-1, 0, 18, 0, -48, 0, 32}), "T6 = 32z^6 - 48z^4 + 18z^2 - 1");
    check(poly_is("U(3, z)", {0, -4, 0, 8}), "U3 = 8z^3 - 4z");
    check(poly_is("L(3, z)", {1, -3, 1.5, -1.0 / 6}), "L3 = 1 - 3z + 3z^2/2 - z^3/6");
    check(poly_is("bessel_poly(3, z)", {1, 6, 15, 15}), "y3 = 15z^3 + 15z^2 + 6z + 1");
    check(poly_is("cyclotomic(15, z)", {1, -1, 0, 1, -1, 1, 0, -1, 1}),
          "Phi15 = z^8 - z^7 + z^5 - z^4 + z^3 - z + 1");
    check(poly_is("cyclotomic(1, z)", {-1, 1}), "Phi1 = z - 1");
    check(poly_is("bernoulli(4, z)", {-1.0 / 30, 0, 1, -2, 1}), "B4(z) = z^4 - 2z^3 + z^2 - 1/30");
    check(poly_is("euler_poly(4, z)", {0, 1, 0, -2, 1}), "E4(z) = z^4 - 2z^3 + z");
    check(poly_is("hermite(0, z)", {1}), "H0 = 1");
    check(poly_is("gegenbauer(2, 1, z)", {-1, 0, 4}), "C2^(1) = U2 = 4z^2 - 1");
    check(poly_is("jacobi(1, 1, 2, z)", {-0.5, 2.5}), "P1^(1,2) = (a-b)/2 + (a+b+2)z/2");

    std::printf("\nexact Bernoulli numbers:\n");
    // Relative 1e-15, not exact: long double is plain double on MSVC, and
    // the exact rational is rounded twice there (numerator, then quotient).
    check(std::abs(special::bernoulli_number(1) / -0.5L - 1.0L) < 1e-15L, "B1 = -1/2");
    check(std::abs(special::bernoulli_number(12) / (-691.0L / 2730.0L) - 1.0L) < 1e-15L, "B12 = -691/2730");
    check(special::bernoulli_number(39) == 0.0L, "B39 = 0 exactly");
    {
        // B40 = -261082718496449122051 / 13530 (numerator exceeds 64 bits).
        const long double b40 = -261082718496449122051.0L / 13530.0L;
        check(std::abs(special::bernoulli_number(40) / b40 - 1.0L) < 1e-15L,
              "B40 = -261082718496449122051/13530");
    }

    // ---- arguments that are expressions ---------------------------------------------
    std::printf("\nthe argument x can be any expression:\n");
    {
        CanonicalRational cr = parse_ok("hermite(2, 2*z - 1)");
        bool ok = true;
        for (Cplx z : {Cplx(0.3, 0.1), Cplx(-1.2, 2.0), Cplx(5, 0)}) {
            const Cplx x = 2.0 * z - 1.0;
            ok = ok && near(cr.eval(z, {}), 4.0 * x * x - 2.0);
        }
        check(ok, "hermite(2, 2*z-1) == 4(2z-1)^2 - 2");

        cr = parse_ok("legendre(3, a*z)");
        ok = cr.parameters == std::vector<std::string>{"a"};
        for (Cplx z : {Cplx(0.3, 0.1), Cplx(-1.2, 2.0)}) {
            const Cplx a(0.7, -0.4), x = a * z;
            ok = ok && near(cr.eval(z, {{"a", a}}), (5.0 * x * x * x - 3.0 * x) / 2.0);
        }
        check(ok, "legendre(3, a*z) has parameter a and the right values");

        cr = parse_ok("chebyshev_t(4, 1/z)");
        ok = cr.Q.coeffs.size() == 5;
        for (Cplx z : {Cplx(0.3, 0.1), Cplx(-1.2, 2.0)}) {
            const Cplx x = 1.0 / z;
            ok = ok && near(cr.eval(z, {}), 8.0 * std::pow(x, 4) - 8.0 * x * x + 1.0);
        }
        check(ok, "chebyshev_t(4, 1/z): a rational argument gives T4(1/z) = (...)/z^4");
    }

    // ---- symbolic family parameters ---------------------------------------------------
    std::printf("\nsymbolic alpha / lam / beta:\n");
    {
        CanonicalRational cr = parse_ok("laguerre(3, a, z)");
        bool ok = cr.parameters == std::vector<std::string>{"a"} && cr.P.coeffs.size() == 4;
        for (Cplx a : {Cplx(0, 0), Cplx(2, 0), Cplx(-0.5, 1.5)}) {
            // Closed form: L3^(a)(x) = (a+1)(a+2)(a+3)/6 - (a+2)(a+3)/2 x + (a+3)/2 x^2 - x^3/6
            const std::vector<Cplx> expect = {(a + 1.0) * (a + 2.0) * (a + 3.0) / 6.0,
                                              -(a + 2.0) * (a + 3.0) / 2.0, (a + 3.0) / 2.0,
                                              Cplx(-1.0 / 6.0, 0)};
            auto got = coeffs(cr.P, {{"a", a}});
            for (int k = 0; k < 4; ++k) ok = ok && near(got[k], expect[k]);
        }
        check(ok, "laguerre(3, a, z) coefficients match the closed form for symbolic a");
        check(!references_param(parse_ok("laguerre(3, 2, z)").P.coeffs[0], "a"),
              "numeric alpha folds to plain constants");

        cr = parse_ok("L(2, b + 1, z) + a");
        check(cr.parameters == std::vector<std::string>{"a", "b"},
              "alpha can be an expression in parameters (b + 1)");

        cr = parse_ok("gegenbauer(2, lam, z)");
        ok = cr.parameters == std::vector<std::string>{"lam"};
        for (Cplx l : {Cplx(0.5, 0), Cplx(1, 0), Cplx(0.3, 0.2)}) {
            auto got = coeffs(cr.P, {{"lam", l}});   // 2 l (1+l) z^2 - l
            ok = ok && near(got[0], -l) && near(got[2], 2.0 * l * (1.0 + l));
        }
        check(ok, "gegenbauer(2, lam, z) = 2 lam (1+lam) z^2 - lam");

        cr = parse_ok("jacobi(2, a, b, z)");
        ok = cr.parameters == std::vector<std::string>{"a", "b"};
        for (auto ab : {std::pair<double, double>{0, 0}, {1, 2}, {-0.3, 0.7}}) {
            const double a = ab.first, b = ab.second;
            // At a = b = 0 Jacobi is Legendre; in general compare against the
            // binomial sum evaluated directly at a few z.
            for (Cplx z : {Cplx(0.2, 0.1), Cplx(-1.3, 0.4)}) {
                const Cplx u = (z - 1.0) / 2.0, v = (z + 1.0) / 2.0;
                auto C = [](double top, int k) { double r = 1; for (int i = 1; i <= k; ++i) r *= (top - k + i) / i; return r; };
                const Cplx direct = C(2 + a, 2) * v * v + C(2 + a, 1) * C(2 + b, 1) * u * v + C(2 + b, 2) * u * u;
                ok = ok && near(cr.eval(z, {{"a", a}, {"b", b}}), direct);
            }
        }
        check(ok, "jacobi(2, a, b, z) matches the binomial sum for several (a, b)");
        check(poly_is("P(3, 0, 0, z)", {0, -1.5, 0, 2.5}), "P(3, 0, 0, z) (Jacobi) == Legendre P3");
    }

    // ---- d(...) ----------------------------------------------------------------------
    std::printf("\nd(...): derivative in z:\n");
    {
        check(poly_is("d(z^3)", {0, 0, 3}), "d(z^3) = 3z^2");
        check(poly_is("d(d(z^4))", {0, 0, 12}), "d(d(z^4)) = 12z^2");
        check(poly_is("d(d(d(z^4)))", {0, 24}), "d(d(d(z^4))) = 24z");
        check(poly_is("d(7)", {}), "d(7) = 0");
        check(poly_is("d(hermite(3, z))", {-12, 0, 24}), "d(H3) = 24z^2 - 12");

        CanonicalRational cr = parse_ok("d(a*z^2 + z)");
        auto P = coeffs(cr.P, {{"a", 3.0}});
        check(P.size() == 2 && near(P[0], 1.0) && near(P[1], 6.0), "d(a*z^2 + z) = 2az + 1 (a = 3)");

        cr = parse_ok("d((z^2 + 1)/(z - 1))");
        bool ok = true;
        for (Cplx z : {Cplx(0.3, 0.1), Cplx(-1.2, 2.0), Cplx(4, -1)}) {
            const Cplx expect = (z * z - 2.0 * z - 1.0) / ((z - 1.0) * (z - 1.0));
            ok = ok && near(cr.eval(z, {}), expect);
        }
        check(ok, "quotient: d((z^2+1)/(z-1)) = (z^2 - 2z - 1)/(z-1)^2");

        cr = parse_ok("d(1/z)");
        check(near(cr.eval(Cplx(2, 0), {}), -0.25), "d(1/z) = -1/z^2");

        cr = parse_ok("d(d(1/(z^2 + a)))");
        ok = true;
        for (Cplx z : {Cplx(0.3, 0.1), Cplx(-1.2, 2.0)}) {
            const Cplx a(0.5, 0.25), w = z * z + a;
            const Cplx expect = (6.0 * z * z - 2.0 * a) / (w * w * w);   // (1/w)'' = (6z^2 - 2a)/w^3
            ok = ok && near(cr.eval(z, {{"a", a}}), expect);
        }
        check(ok, "quotient twice: d(d(1/(z^2+a))) = (6z^2 - 2a)/(z^2+a)^3");

        cr = parse_ok("d(hermite(6, z)) - 12*hermite(5, z)");
        check(coeffs(cr.P).empty() || [&] {
            for (auto c : coeffs(cr.P)) if (std::abs(c) > 1e-9) return false;
            return true; }(), "H6' = 12 H5 (Hermite identity)");
    }

    // ---- errors -----------------------------------------------------------------------
    std::printf("\nerror messages:\n");
    check(starts_with(parse_error("hermite(n, z)"), "hermite: degree must be an integer literal"),
          "hermite(n, z): " + parse_error("hermite(n, z)"));
    check(starts_with(parse_error("hermite(2.5, z)"), "hermite: degree must be an integer literal"),
          "hermite(2.5, z): " + parse_error("hermite(2.5, z)"));
    check(starts_with(parse_error("hermite(-1, z)"), "hermite: degree must be an integer literal"),
          "hermite(-1, z): " + parse_error("hermite(-1, z)"));
    check(starts_with(parse_error("jacobi(3, 1, z)"), "jacobi expects 4 arguments (n, alpha, beta, x)"),
          "jacobi(3, 1, z): " + parse_error("jacobi(3, 1, z)"));
    check(starts_with(parse_error("hermite(3)"), "hermite expects 2 arguments (n, x)"),
          "hermite(3): " + parse_error("hermite(3)"));
    check(starts_with(parse_error("P(3, 1, z)"), "P expects 2 arguments (n, x) or 4 (n, alpha, beta, x)"),
          "P(3, 1, z): " + parse_error("P(3, 1, z)"));
    check(starts_with(parse_error("laguerre(3, 1, 2, z)"), "laguerre expects 2 or 3 arguments"),
          "laguerre(3, 1, 2, z): " + parse_error("laguerre(3, 1, 2, z)"));
    check(starts_with(parse_error("hermite(41, z)"), "hermite: degree must be at most 40"),
          "hermite(41, z): " + parse_error("hermite(41, z)"));
    check(starts_with(parse_error("laguerre(3, z, z)"), "laguerre: alpha must not depend on z"),
          "laguerre(3, z, z): " + parse_error("laguerre(3, z, z)"));
    check(starts_with(parse_error("laguerre(3, 1/a, z)"), "laguerre: alpha must be a number or a polynomial"),
          "laguerre(3, 1/a, z): " + parse_error("laguerre(3, 1/a, z)"));
    check(starts_with(parse_error("cyclotomic(0, z)"), "cyclotomic: index must be at least 1"),
          "cyclotomic(0, z): " + parse_error("cyclotomic(0, z)"));
    check(starts_with(parse_error("d(z, 1)"), "d expects 1 argument"), "d(z, 1): " + parse_error("d(z, 1)"));
    check(starts_with(parse_error("a_1 + z"), "unexpected character '_'"),
          "an underscore in a PARAMETER name is still rejected");
    check(parse_error("sin(z)").find("transcendental") != std::string::npos,
          "transcendental functions are still rejected as such");

    // ---- aliases vs parameters with the same letter ---------------------------------
    std::printf("\nalias letters stay usable as parameters:\n");
    {
        CanonicalRational cr = parse_ok("P*z + L + T + U + H + d");
        check(cr.parameters == std::vector<std::string>{"H", "L", "P", "T", "U", "d"},
              "P, L, T, U, H, d without '(' are parameters");
        cr = parse_ok("P(2, z) + P");
        check(cr.parameters == std::vector<std::string>{"P"}, "P(2, z) + P: a call and a parameter");
    }

    // ---- warnings ----------------------------------------------------------------------
    std::printf("\nwarnings above degree 25:\n");
    check(parse_ok("hermite(25, z)").warnings.empty(), "n = 25: no warning");
    {
        auto w = parse_ok("hermite(26, z)").warnings;
        check(w.size() == 1 && w[0].find("degree above 25") != std::string::npos,
              "n = 26: one warning (" + (w.empty() ? std::string("none") : w[0]) + ")");
    }
    check(parse_ok("hermite(40, z)").P.coeffs.size() == 41, "n = 40 is accepted (degree 40)");
    check(parse_ok("cyclotomic(37, z)").warnings.empty(),
          "cyclotomic: no warning (its coefficients stay tiny)");

    // ---- n = 40 accuracy spot check: H40 leading and constant terms --------------------
    {
        auto P = coeffs(parse_ok("hermite(40, z)").P);
        // H_n leading coefficient 2^n; H_(2m)(0) = (-1)^m (2m)!/m!
        long double h0 = 1.0L;
        for (int k = 21; k <= 40; ++k) h0 *= k;   // 40!/20!
        check(near(P[40], std::ldexp(1.0, 40), 1e-15) && near(P[0], static_cast<double>(h0), 1e-13),
              "H40: leading 2^40 and H40(0) = 40!/20! to double precision");
    }

    // ---- round trip: Newton on Legendre P5 ---------------------------------------------
    std::printf("\nround trip: z - p/d(p), p = legendre(5, z):\n");
    {
        RationalMap m = RationalMap::from_expression("z - legendre(5, z)/d(legendre(5, z))", "", {}, "nl5");
        const double roots[5] = {0.0, 0.5384693101056831, -0.5384693101056831,
                                 0.9061798459386640, -0.9061798459386640};
        bool fixed_ok = true, super_ok = true;
        for (double r : roots) {
            fixed_ok = fixed_ok && std::abs(m.eval(Cplx(r, 0), 0.0) - r) < 1e-12;
            super_ok = super_ok && std::abs(m.deriv(Cplx(r, 0), 0.0)) < 1e-9;
        }
        check(fixed_ok, "each Legendre root is fixed");
        check(super_ok, "each is superattracting (multiplier 0)");
        auto fps = m.fixed_points(0.0);
        int found = 0;
        for (double r : roots)
            for (const auto& fp : fps)
                if (std::abs(fp.point - r) < 1e-9 && std::abs(fp.multiplier) < 1e-7) { ++found; break; }
        check(found == 5, "fixed_points() finds all 5 roots as superattracting (" + std::to_string(found) + ")");
        check(m.degree(0.0) == 5, "the Newton map has degree 5");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
