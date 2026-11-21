// =============================================================================
// test_composed.cpp -- ComposedMap / Map::composed, tested by property:
// composition identities against directly-built RationalMaps, the chain
// rule, degree multiplicativity, both charts of the sphere, and rendering
// through the ordinary Renderer.
// =============================================================================
#include "cdx/composed.hpp"
#include "cdx/renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static bool close(Cplx a, Cplx b, double tol = 1e-10) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) < tol * scale;
}

static bool is_inf(Cplx z) { return std::isinf(z.real()) || std::isinf(z.imag()); }

static RationalMap expr(const std::string& s) {
    return RationalMap::from_expression(s, "", {});
}

int main() {
    std::printf("=== cdx ComposedMap tests ===\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.5, 1.5);
    const double inf = std::numeric_limits<double>::infinity();

    // ---- single stage equals the RationalMap itself ---------------------------
    {
        const Cplx c(-0.12, 0.74);
        const RationalMap q = RationalMap::mandelbrot();
        ComposedMap f({ComposedStage{q, c, 1}});
        bool ok = true;
        for (int k = 0; k < 200; ++k) {
            const Cplx z(U(rng), U(rng));
            ok = ok && close(f.eval(z), q.eval(z, c));
        }
        check(ok, "one stage z^2+c agrees with RationalMap::eval");
        check(f.degree() == 2.0 && f.applications() == 1, "degree 2, one application");
    }

    // ---- repeat n equals n-fold iteration, and composition order --------------
    {
        const Cplx c(0.3, -0.2);
        const RationalMap q = RationalMap::mandelbrot();
        ComposedMap f3({ComposedStage{q, c, 3}});
        const RationalMap m = expr("(z^2 + 1)/(2*z - 1)");
        ComposedMap gm({ComposedStage{q, c, 1}, ComposedStage{m, 0.0, 1}});   // m o q
        bool ok_iter = true, ok_order = true;
        for (int k = 0; k < 200; ++k) {
            const Cplx z(U(rng), U(rng));
            Cplx w = z;
            for (int j = 0; j < 3; ++j) w = q.eval(w, c);
            ok_iter = ok_iter && close(f3.eval(z), w, 1e-9);
            ok_order = ok_order && close(gm.eval(z), m.eval(q.eval(z, c), 0.0), 1e-9);
        }
        check(ok_iter, "repeat 3 equals three iterations of z^2+c");
        check(ok_order, "stages apply in order: stages[1] o stages[0]");
        check(f3.degree() == 8.0, "degree multiplies: deg (z^2+c)^3 = 8");
        check(gm.degree() == 4.0, "degree multiplies: deg(m) * deg(q) = 2 * 2");
    }

    // ---- chain rule ------------------------------------------------------------
    {
        const RationalMap m = expr("(z^3 - 2)/(3*z^2 + 0.5)");
        const RationalMap q = RationalMap::mandelbrot();
        ComposedMap f({ComposedStage{q, Cplx(-0.4, 0.1), 2}, ComposedStage{m, 0.0, 1}});
        bool ok = true;
        for (int k = 0; k < 100; ++k) {
            const Cplx z(U(rng) * 0.6, U(rng) * 0.6);
            const double h = 1e-6;
            const Cplx fd = (f.eval(z + h) - f.eval(z - h)) / (2.0 * h);
            ok = ok && close(f.deriv(z), fd, 1e-5);
        }
        check(ok, "deriv matches a centered finite difference");
        check(std::isnan(f.deriv(Cplx(inf, 0.0)).real()), "deriv at infinity is NaN (plane chart only)");
    }

    // ---- both charts of a P/Q stage ---------------------------------------------
    {
        // F(u) = u^3 / (1 + 1.5 u^2): the model map in reciprocal coordinates (d = 3).
        const RationalMap F = expr("z^3/(1 + 1.5*z^2)");
        ComposedMap f({ComposedStage{F, 0.0, 1}});
        bool ok = true;
        for (int k = 0; k < 200; ++k) {
            const Cplx u(U(rng) * 3.0, U(rng) * 3.0);
            ok = ok && close(f.eval(u), u * u * u / (1.0 + 1.5 * u * u), 1e-12);
        }
        check(ok, "P/Q stage agrees with the formula inside and outside the unit disk");
        // |u|^3 ~ 3e420 would overflow a one-chart Horner evaluation; |u|^2 < 1e300
        // keeps the point finite under the chordal_distance convention.
        const Cplx big(3e140, -1e140);
        const Cplx v = f.eval(big);
        check(!is_inf(v) && close(v, big / 1.5, 1e-12), "no overflow at |u| ~ 3e140: F(u) ~ u/1.5 there (reciprocal chart)");
        check(is_inf(f.eval(Cplx(inf, 0.0))), "F(inf) = inf (deg P > deg Q)");
        ComposedMap mob({ComposedStage{expr("(2*z + 1)/(z - 3)"), 0.0, 1}});
        check(close(mob.eval(Cplx(inf, 0.0)), Cplx(2.0, 0.0)), "Mobius value at infinity = ratio of leading coefficients");
        check(is_inf(mob.eval(Cplx(3.0, 0.0))), "Mobius pole maps to infinity");
        const Cplx t = Cplx(0.0, 1.0) / std::sqrt(1.5);   // a root of 1 + 1.5 u^2: a pole of F
        check(is_inf(f.eval(t)) || std::abs(f.eval(t)) > 1e12, "F at its pole is infinite (or huge after rounding of the root)");
    }

    // ---- term-backed stage: partial fractions with many simple poles ------------
    {
        RationalMap S;
        S.add_poly(Cplx(0.25, -0.5), 0);
        std::vector<Cplx> poles, res;
        for (int k = 0; k < 400; ++k) {
            const double th = 2.0 * M_PI * k / 400.0;
            poles.emplace_back(std::cos(th), std::sin(th));
            res.emplace_back(0.01 * std::cos(3 * th), 0.02 * std::sin(th));
            PoleTerm p; p.location = poles.back(); p.strength = res.back();
            S.pole_terms().push_back(p);   // direct push: add_pole's O(m) collision check per pole is not needed here
        }
        ComposedMap f({ComposedStage{S, 0.0, 1}});
        bool ok = true;
        for (int k = 0; k < 100; ++k) {
            const Cplx z(U(rng), U(rng));
            Cplx direct(0.25, -0.5);
            for (std::size_t j = 0; j < poles.size(); ++j) direct += res[j] / (z - poles[j]);
            ok = ok && close(f.eval(z), direct, 1e-11);
        }
        check(ok, "400 simple poles agree with the direct partial-fraction sum");
        check(f.eval(Cplx(inf, 0.0)) == Cplx(0.25, -0.5), "value at infinity = the constant term, exactly");
        check(is_inf(f.eval(poles[17])), "an exact pole hit is infinity, not NaN");
        check(close(f.eval(Cplx(1e200, 1e200)), Cplx(0.25, -0.5), 1e-12), "huge |z| counts as infinity");
        check(f.degree() == 400.0, "degree = number of simple poles");
    }

    // ---- NaN, validation, describe --------------------------------------------
    {
        ComposedMap f({ComposedStage{RationalMap::mandelbrot(), 0.0, 2}});
        check(std::isnan(f.eval(Cplx(std::nan(""), 0.0)).real()), "NaN in, NaN out");
        bool threw = false;
        try { ComposedMap g(std::vector<ComposedStage>{}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "empty stage list is rejected");
        threw = false;
        try { ComposedMap g({ComposedStage{RationalMap::mandelbrot(), 0.0, 0}}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "repeat 0 is rejected");
        check(!f.describe().empty(), "describe() is non-empty");
    }

    // ---- rendering through Map::composed ------------------------------------------
    {
        // (z^2)^2 = z^4: two superattracting basins, 0 and infinity, split by |z| = 1.
        RationalMap sq; sq.add_poly(Cplx(1.0, 0.0), 2);
        ComposedMap f({ComposedStage{sq, 0.0, 2}});
        const Map m = Map::composed(f);
        check(m.family() == Family::Composed && m.degree() == 4, "Map::composed: family and degree");
        check(!m.escape_certified(), "composed maps take the sphere-aware rational paths");
        Viewport v; v.center = 0.0; v.scale = 2.0; v.resolution = 81;
        RenderSettings s; s.max_iter = 50; s.tol = 1e-9;
        Renderer r(m, v, s);
        std::vector<Cycle> cycles;
        Cycle zero; zero.points = {Cplx(0.0, 0.0)}; zero.id = 1; cycles.push_back(zero);
        Cycle at_inf; at_inf.points = {Cplx(inf, 0.0)}; at_inf.id = 2; cycles.push_back(at_inf);
        const Image img = r.render_basin(cycles);
        int wrong = 0, total = 0;
        for (int row = 0; row < v.resolution; ++row) {
            for (int col = 0; col < v.resolution; ++col) {
                const double a = std::abs(v.coord(col, row));
                if (std::fabs(a - 1.0) < 0.05) continue;          // skip the Julia set itself
                ++total;
                const int want = a < 1.0 ? 1 : 2;
                if (static_cast<int>(img.at(col, row)) != want) ++wrong;
            }
        }
        check(total > 5000 && wrong == 0, "render_basin on z^4 (as z^2 composed twice): |z|<1 -> 0, |z|>1 -> inf");

        const Image par = r.render_parameter();
        double mx = 0.0;
        for (double x : par.data) mx = std::max(mx, std::fabs(x));
        check(mx == 0.0, "parameter plane is honestly all-zero for a composed map");

        Image labels;
        const Image jul = r.render_julia(nullptr, cycles, &labels);
        check(labels.width == v.resolution, "render_julia runs on a composed map (rational path)");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
