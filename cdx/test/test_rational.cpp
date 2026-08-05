// =============================================================================
// test_rational.cpp -- property-based checks for the term-based RationalMap.
//
// Mirrors test_renderer.cpp's style: presets are checked against their
// documented closed forms, and serialization is checked as a round-trip
// property, rather than comparing against golden output.
// =============================================================================
#include "cdx/rational.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static bool close(Cplx a, Cplx b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// add_pole() now rejects a second pole at a location that already has one
// (see its own doc comment) -- correct for the normal editing path, but
// several tests below deliberately construct a REDUNDANT pole (two term
// objects at the same location) to check that pole_locations()/
// pole_orders()/critical_points()/fixed_points() still combine such a
// state correctly, since it CAN still arise from deserialize() or direct
// pole_terms() mutation (both of which bypass add_pole entirely) even
// though the validated add_pole path no longer creates one fresh. This
// goes straight through pole_terms(), the same live-mutable-vector access
// those two bypasses use, rather than pretending add_pole would allow it.
static void push_pole_direct(RationalMap& m, Cplx location, Cplx strength, int order) {
    PoleTerm t;
    t.location = location;
    t.strength = strength;
    t.order = order;
    m.pole_terms().push_back(t);
}

// Reference integer power for computing expected values in tests. Separate
// from (and simpler than) the library's own ipow, so a bug shared between
// the two wouldn't cancel out.
static Cplx ipow_ref(Cplx b, int n) {
    const bool inv = n < 0;
    const int m = inv ? -n : n;
    Cplx acc(1.0, 0.0);
    for (int i = 0; i < m; ++i) acc *= b;
    return inv ? Cplx(1.0, 0.0) / acc : acc;
}

int main() {
    std::printf("=== cdx rational tests ===\n");

    const Cplx zs[] = {{0.7, 0.3}, {-1.1, 0.4}, {2.0, -0.5}, {0.0, 1.0}};
    const Cplx as[] = {{0.0, 0.0}, {-1.0, 0.0}, {0.3, 0.2}, {1.5, -0.5}};

    // ---- term-level parameter binding -------------------------------------
    std::printf("\nterm-level parameter binding:\n");
    {
        PolyTerm t;
        t.coeff = {2.0, 0.0};
        t.param_power = 3;
        bool ok = true;
        for (Cplx a : as)
            if (!close(t.effective_coeff(a), t.coeff * ipow_ref(a, 3))) ok = false;
        check(ok, "PolyTerm::effective_coeff == coeff * a^param_power");
    }
    {
        PoleTerm t;
        t.strength = {1.0, -1.0};
        t.param_power = -2;
        bool ok = true;
        for (Cplx a : as) {
            if (a == Cplx(0, 0)) continue;   // a^-2 undefined at a=0
            if (!close(t.effective_strength(a), t.strength * ipow_ref(a, -2))) ok = false;
        }
        check(ok, "PoleTerm::effective_strength == strength * a^param_power");
    }
    {
        PoleTerm t;
        t.location = {5.0, 5.0};
        t.location_is_param = true;
        bool ok = true;
        for (Cplx a : as)
            if (t.effective_location(a) != a) ok = false;
        check(ok, "location_is_param makes effective_location track a, ignoring location");
    }

    // ---- presets match their documented closed forms -----------------------
    std::printf("\npresets match documented closed forms:\n");
    {
        RationalMap m = RationalMap::mandelbrot();
        bool eval_ok = true, deriv_ok = true;
        for (Cplx z : zs) for (Cplx a : as) {
            if (!close(m.eval(z, a), z * z + a)) eval_ok = false;
            if (!close(m.deriv(z, a), Cplx(2.0, 0.0) * z)) deriv_ok = false;
        }
        check(eval_ok, "mandelbrot: eval == z^2+a");
        check(deriv_ok, "mandelbrot: deriv == 2z");
        check(m.degree({0, 0}) == 2, "mandelbrot: degree == 2");
    }
    for (int n : {3, 5}) {
        RationalMap m = RationalMap::multibrot(n);
        bool eval_ok = true, deriv_ok = true;
        for (Cplx z : zs) for (Cplx a : as) {
            if (!close(m.eval(z, a), ipow_ref(z, n) + a)) eval_ok = false;
            const Cplx expect_deriv = static_cast<double>(n) * ipow_ref(z, n - 1);
            if (!close(m.deriv(z, a), expect_deriv)) deriv_ok = false;
        }
        char buf[80];
        std::snprintf(buf, sizeof buf, "multibrot(%d): eval == z^%d+a", n, n);
        check(eval_ok, buf);
        std::snprintf(buf, sizeof buf, "multibrot(%d): deriv == %d*z^%d", n, n, n - 1);
        check(deriv_ok, buf);
        check(m.degree({0, 0}) == n, "multibrot: degree == n");
    }
    for (int n : {2, 3}) {
        RationalMap m = RationalMap::mcmullen(n);
        bool eval_ok = true, deriv_ok = true;
        for (Cplx z : zs) for (Cplx a : as) {
            const Cplx expect_eval = ipow_ref(z, n) + a * ipow_ref(z, -n);
            if (!close(m.eval(z, a), expect_eval)) eval_ok = false;
            const Cplx expect_deriv =
                static_cast<double>(n) * ipow_ref(z, n - 1) -
                static_cast<double>(n) * a * ipow_ref(z, -(n + 1));
            if (!close(m.deriv(z, a), expect_deriv)) deriv_ok = false;
        }
        char buf[80];
        std::snprintf(buf, sizeof buf, "mcmullen(%d): eval == z^%d + a/z^%d", n, n, n);
        check(eval_ok, buf);
        std::snprintf(buf, sizeof buf, "mcmullen(%d): deriv matches closed form", n);
        check(deriv_ok, buf);
        // degree = 2n, matching Family::McMullen2/3's hardcoded fast-path degree
        check(m.degree({0, 0}) == 2 * n, "mcmullen: degree == 2n");
    }
    {
        RationalMap m = RationalMap::newton_cubic();
        bool eval_ok = true, deriv_ok = true;
        for (Cplx z : zs) {
            const Cplx expect_eval = (2.0 / 3.0) * z + (1.0 / 3.0) * ipow_ref(z, -2);
            if (!close(m.eval(z, {0, 0}), expect_eval)) eval_ok = false;
            const Cplx expect_deriv = Cplx(2.0 / 3.0, 0) - (2.0 / 3.0) * ipow_ref(z, -3);
            if (!close(m.deriv(z, {0, 0}), expect_deriv)) deriv_ok = false;
        }
        check(eval_ok, "newton_cubic: eval == (2/3)z + (1/3)z^-2");
        check(deriv_ok, "newton_cubic: deriv == 2/3 - (2/3)z^-3");
        check(m.degree({0, 0}) == 3, "newton_cubic: degree == 3 (matches Family::Newton3)");
    }

    // ---- general poles: location, strength, singular sentinel --------------
    std::printf("\ngeneral poles:\n");
    {
        RationalMap m("test");
        m.add_pole({1.0, 0.0}, {1.0, 0.0}, 1);
        check(close(m.eval({2.0, 0.0}, {0, 0}), Cplx(1.0, 0.0)), "1/(z-1) at z=2 == 1");
        check(close(m.deriv({2.0, 0.0}, {0, 0}), Cplx(-1.0, 0.0)), "d/dz 1/(z-1) at z=2 == -1");
        check(m.eval({1.0, 0.0}, {0, 0}).real() > 1e299,
              "evaluating exactly at a pole returns the huge escape sentinel");
    }

    // ---- add_poly/add_pole validation: one representation per pole -----------
    std::printf("\nadd_poly/add_pole validation:\n");
    {
        RationalMap m("test");
        bool threw = false;
        try { m.add_poly({1, 0}, -1); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "add_poly with a negative exponent throws std::invalid_argument");
        check(m.poly_terms().empty(),
              "the rejected call added nothing -- not a partial/rolled-back add");
    }
    {
        RationalMap m("test");
        // Exponent 0 (a bare constant/parameter term) and positive exponents
        // are unaffected -- only negative ones represent a pole.
        m.add_poly({1, 0}, 0);
        m.add_poly({1, 0}, 5);
        check(m.poly_terms().size() == 2, "exponent 0 and positive exponents are unaffected");
    }
    {
        RationalMap m("test");
        m.add_pole({1.0, 0.0}, {1.0, 0.0}, 1);
        bool threw = false;
        std::string what;
        try { m.add_pole({1.0, 0.0}, {2.0, 0.0}, 3); }
        catch (const std::invalid_argument& e) { threw = true; what = e.what(); }
        check(threw, "add_pole at an already-occupied location throws std::invalid_argument");
        check(m.pole_terms().size() == 1, "the rejected call added nothing");
        check(what.find("1") != std::string::npos,
              "the exception message identifies the colliding location, not just 'rejected'");
    }
    {
        // Backward compatibility: a map that already has a negative-
        // exponent PolyTerm at the origin (as any RationalMap built before
        // this restriction existed could, e.g. one loaded via deserialize()
        // -- add_poly's own restriction is not retroactive) must still be
        // caught by add_pole's collision check, even though add_poly can no
        // longer CREATE one fresh -- both mechanisms mean the same pole.
        RationalMap m("legacy");
        PolyTerm legacy_pole_term;
        legacy_pole_term.coeff = {1.0, 0.0};
        legacy_pole_term.exponent = -2;
        m.poly_terms().push_back(legacy_pole_term);   // simulates pre-restriction / deserialized data
        bool threw = false;
        try { m.add_pole({0.0, 0.0}, {1.0, 0.0}, 1); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "add_pole at the origin throws when a legacy negative-exponent "
                     "PolyTerm already implies a pole there");
    }
    {
        // A DISABLED existing pole does not block a new one at the same
        // spot -- disabled means invisible to eval() and everything else,
        // so it should not be invisible-except-for-blocking-a-re-add.
        RationalMap m("test");
        m.add_pole({1.0, 0.0}, {1.0, 0.0}, 1);
        m.pole_terms()[0].enabled = false;
        bool threw = false;
        try { m.add_pole({1.0, 0.0}, {2.0, 0.0}, 1); }
        catch (const std::invalid_argument&) { threw = true; }
        check(!threw, "a disabled pole at the same location does not block adding a new one");
    }
    {
        // A pole whose location tracks the parameter (location_is_param)
        // is skipped by the collision check -- see add_pole's own comment
        // for why a fixed numeric collision can't be decided without `a`.
        RationalMap m("test");
        m.add_pole({0.0, 0.0}, {1.0, 0.0}, 1);
        m.pole_terms()[0].location_is_param = true;
        bool threw = false;
        try { m.add_pole({0.0, 0.0}, {2.0, 0.0}, 1); }
        catch (const std::invalid_argument&) { threw = true; }
        check(!threw, "a location_is_param pole is skipped by the collision check");
    }
    {
        // deserialize() bypasses add_poly/add_pole entirely (pushes
        // directly onto the term vectors -- see rational.cpp), so an OLD
        // saved file with a negative-exponent PolyTerm (e.g. a
        // pre-restriction newton_cubic() someone saved to their library)
        // must still round-trip without throwing.
        RationalMap legacy("legacy");
        PolyTerm legacy_term;
        legacy_term.coeff = {1.0, 0.0};
        legacy_term.exponent = -2;
        legacy_term.label = "legacy pole";
        legacy.poly_terms().push_back(legacy_term);
        const std::string text = legacy.serialize();
        RationalMap round;
        std::string err;
        check(RationalMap::deserialize(text, round, err),
              "a serialized negative-exponent poly term still deserializes without throwing");
        check(round.poly_terms().size() == 1 && round.poly_terms()[0].exponent == -2,
              "...and keeps its exponent exactly, not silently converted");
    }
    {
        // newton_cubic() itself must still construct without throwing --
        // it now calls add_pole for its origin term, not a negative-
        // exponent add_poly, and there's nothing for that add_pole call to
        // collide with (its only poly term is the unrelated linear one).
        bool threw = false;
        try { RationalMap m = RationalMap::newton_cubic(); (void)m; }
        catch (const std::invalid_argument&) { threw = true; }
        check(!threw, "newton_cubic() constructs without throwing");
        RationalMap m = RationalMap::newton_cubic();
        check(m.poly_terms().size() == 1 && m.pole_terms().size() == 1,
              "newton_cubic() now has exactly one poly term (linear) and one pole term "
              "(the origin), not two poly terms");
    }

    // ---- pole_locations: dedup and negative-exponent implication -----------
    std::printf("\npole_locations:\n");
    {
        RationalMap m("test");
        m.add_pole({1.0, 0.0}, {1.0, 0.0}, 1);
        push_pole_direct(m, {1.0, 0.0}, {2.0, 0.0}, 2);   // same location, different term
        m.add_pole({-1.0, 0.0}, {1.0, 0.0}, 1);
        auto locs = m.pole_locations({0, 0});
        check(locs.size() == 2, "duplicate pole locations are deduplicated");
    }
    {
        // Its pole at the origin is now an explicit PoleTerm (add_poly
        // rejects negative exponents -- see newton_cubic's own comment),
        // not a negative polynomial exponent, but pole_locations() doesn't
        // care which representation produced it.
        RationalMap m = RationalMap::newton_cubic();
        auto locs = m.pole_locations({0, 0});
        check(locs.size() == 1 && close(locs[0], Cplx(0, 0)),
              "a negative polynomial exponent implies a pole at the origin");
    }

    // ---- critical_points: matches closed forms for the built-in shapes -------
    // Total count includes ALL THREE sources -- ordinary derivative zeros,
    // poles (multiplicity order-1), and infinity (multiplicity |p-q|-1 when
    // |p-q| >= 2) -- so every case below is cross-checked against
    // Riemann-Hurwitz (total == 2*degree-2) as well as its specific shape.
    std::printf("\ncritical_points:\n");
    {
        RationalMap m = RationalMap::mandelbrot();
        const Cplx a{0.3, -0.7};
        auto cp = m.critical_points(a);
        // deriv = 2z: ordinary critical point at 0. No poles. |p-q| = |2-0|
        // = 2, so infinity is critical too (multiplicity 1).
        check(cp.size() == 2, "mandelbrot: 2 critical points (0 and infinity)");
        check(static_cast<int>(cp.size()) == 2 * m.degree(a) - 2,
              "mandelbrot: total multiplicity == 2d-2");
        int n_zero = 0, n_inf = 0;
        for (Cplx z : cp) {
            if (std::isinf(z.real())) ++n_inf;
            else if (close(z, Cplx(0, 0), 1e-9)) ++n_zero;
        }
        check(n_zero == 1 && n_inf == 1, "mandelbrot: one at 0, one at infinity");
    }
    for (int n : {3, 5}) {
        RationalMap m = RationalMap::multibrot(n);
        const Cplx a{0.3, -0.7};
        auto cp = m.critical_points(a);
        // deriv = n*z^(n-1): a root at 0 of multiplicity n-1 (multiple-root
        // convergence is looser than a simple root, same caveat as
        // test_roots.cpp's triple-root case). |p-q| = n, so infinity
        // contributes another n-1.
        int n_zero = 0, n_inf = 0;
        for (Cplx z : cp) {
            if (std::isinf(z.real())) ++n_inf;
            else if (std::abs(z) < 1e-4) ++n_zero;
        }
        char buf[100];
        std::snprintf(buf, sizeof buf,
                      "multibrot(%d): %d at 0, %d at infinity (both multiplicity %d)",
                      n, n - 1, n - 1, n - 1);
        check(n_zero == n - 1 && n_inf == n - 1, buf);
        std::snprintf(buf, sizeof buf, "multibrot(%d): total multiplicity == 2d-2", n);
        check(static_cast<int>(cp.size()) == 2 * m.degree(a) - 2, buf);
    }
    {
        RationalMap m = RationalMap::mcmullen(2);
        const Cplx a{2.0, 1.0};
        auto cp = m.critical_points(a);
        // Ordinary: z^4 = a (4 points). Pole at 0 has true order 2 (from
        // add_pole's order=n=2), contributing multiplicity 1. |p-q| =
        // |4-2| = 2, so infinity contributes multiplicity 1 too. Total 6.
        check(static_cast<int>(cp.size()) == 2 * m.degree(a) - 2,
              "mcmullen(2): total multiplicity == 2d-2 (6)");
        int n_fourth_power = 0, n_zero = 0, n_inf = 0;
        for (Cplx z : cp) {
            if (std::isinf(z.real())) ++n_inf;
            else if (std::abs(z) < 1e-6) ++n_zero;
            else if (std::abs(z * z * z * z - a) < 1e-6) ++n_fourth_power;
        }
        check(n_fourth_power == 4, "mcmullen(2): 4 ordinary points with z^4 == a");
        check(n_zero == 1, "mcmullen(2): the pole at the origin now IS included (multiplicity 1)");
        check(n_inf == 1, "mcmullen(2): infinity is critical too (multiplicity 1)");
    }
    {
        RationalMap m = RationalMap::mcmullen(3);
        const Cplx a{2.0, 1.0};
        auto cp = m.critical_points(a);
        // Ordinary: z^6 = a (6). Pole at 0 order 3 -> multiplicity 2.
        // |p-q| = |6-3| = 3 -> infinity multiplicity 2. Total 10.
        check(static_cast<int>(cp.size()) == 2 * m.degree(a) - 2,
              "mcmullen(3): total multiplicity == 2d-2 (10)");
        int n_zero = 0, n_inf = 0;
        for (Cplx z : cp) {
            if (std::isinf(z.real())) ++n_inf;
            else if (std::abs(z) < 1e-6) ++n_zero;
        }
        check(n_zero == 2, "mcmullen(3): pole at the origin has multiplicity 2");
        check(n_inf == 2, "mcmullen(3): infinity has multiplicity 2");
    }
    {
        // R(z) = 2/z, built from two redundant order-1 pole terms at the same
        // location so clear_denominators() has to use an unreduced common
        // denominator (z-0)^1 * (z-0)^1 rather than the minimal one.
        // Ordinary: derivative -2/z^2 has no finite zero. Pole: TRUE local
        // order at 0 is 1 (a simple pole, despite being built from two
        // order-1 terms -- vanishing_order() finds the actual combined
        // behaviour of 2z/z^2 = 2/z, not the sum 1+1), so multiplicity 0.
        // Infinity: this map's degree() is 2 (it does not reduce the
        // redundant terms away), but critical_points()'s |p-q| is computed
        // from the SAME unreduced construction on both sides, so the
        // redundancy cancels out of the difference regardless: |1-2| = 1,
        // not critical. So the fully correct answer is still zero critical
        // points -- this is NOT checked against 2*degree()-2, since
        // degree() itself doesn't reduce the redundancy (would claim d=2,
        // 2d-2=2, which is simply the wrong degree for this construction).
        RationalMap m("mobius");
        push_pole_direct(m, {0, 0}, {1, 0}, 1);
        push_pole_direct(m, {0, 0}, {1, 0}, 1);
        auto cp = m.critical_points({0, 0});
        check(cp.empty(),
              "R(z)=2/z: correctly zero critical points despite the redundant construction");
    }
    {
        RationalMap m = RationalMap::newton_cubic();
        const Cplx a{0, 0};
        auto cp = m.critical_points(a);
        // Ordinary: z^3=1, the three roots Newton's method converges to
        // (superattracting fixed points). Pole at 0: true order 2 (the
        // 1/(3z^2) term, an explicit PoleTerm), multiplicity 1 -- this is
        // the dynamically informative
        // one (Map::critical_point returns {0,0} for the built-in Newton3
        // for exactly this reason). |p-q| = |3-2| = 1, so infinity is NOT
        // critical here, matching the CLAUDE.md-adjacent fact that this is
        // the case a naive "always count infinity" rule would get wrong.
        check(static_cast<int>(cp.size()) == 2 * m.degree(a) - 2,
              "newton_cubic: total multiplicity == 2d-2 (4)");
        const std::vector<Cplx> cube_roots_of_unity = {
            {1.0, 0.0}, {-0.5, 0.8660254037844386}, {-0.5, -0.8660254037844386}};
        int n_cube_root = 0, n_zero = 0, n_inf = 0;
        for (Cplx z : cp) {
            if (std::isinf(z.real())) { ++n_inf; continue; }
            if (std::abs(z) < 1e-6) { ++n_zero; continue; }
            for (Cplx e : cube_roots_of_unity)
                if (close(z, e, 1e-6)) ++n_cube_root;
        }
        check(n_cube_root == 3, "newton_cubic: the 3 trivial fixed points are still found");
        check(n_zero == 1, "newton_cubic: the pole at the origin is now included");
        check(n_inf == 0, "newton_cubic: infinity is correctly NOT critical (|p-q|=1)");
    }

    // ---- critical_points: Riemann-Hurwitz on randomly generated sandbox maps --
    std::printf("\ncritical_points: Riemann-Hurwitz invariant on random sandbox maps:\n");
    {
        std::mt19937 rng(20260802);   // fixed seed: deterministic test
        std::uniform_real_distribution<double> mag_dist(0.3, 2.0);
        std::uniform_real_distribution<double> angle_dist(0.0, 6.28318530718);
        std::uniform_int_distribution<int> poly_count_dist(1, 3);
        std::uniform_int_distribution<int> pole_count_dist(1, 3);
        std::uniform_int_distribution<int> exponent_dist(1, 5);
        std::uniform_int_distribution<int> order_dist(1, 3);
        auto random_cplx = [&]() {
            const double r = mag_dist(rng), t = angle_dist(rng);
            return Cplx(r * std::cos(t), r * std::sin(t));
        };

        bool all_ok = true;
        for (int trial = 0; trial < 20; ++trial) {
            // Positive, distinct exponents and mutually-distant, distinct
            // pole locations only: overlapping locations make degree()
            // itself under-report (see the Mobius case above), which would
            // make this invariant check meaningless rather than wrong.
            RationalMap m("random" + std::to_string(trial));
            std::vector<int> used_exponents;
            for (int i = 0, n = poly_count_dist(rng); i < n; ++i) {
                int e;
                do { e = exponent_dist(rng); }
                while (std::find(used_exponents.begin(), used_exponents.end(), e) !=
                       used_exponents.end());
                used_exponents.push_back(e);
                m.add_poly(random_cplx(), e);
            }
            std::vector<Cplx> used_locations;
            for (int i = 0, n = pole_count_dist(rng); i < n; ++i) {
                Cplx loc;
                bool far_enough;
                do {
                    const double r = 1.0 + mag_dist(rng), t = angle_dist(rng);
                    loc = Cplx(r * std::cos(t), r * std::sin(t));
                    far_enough = true;
                    for (Cplx u : used_locations)
                        if (std::abs(loc - u) < 0.5) far_enough = false;
                } while (!far_enough);
                used_locations.push_back(loc);
                m.add_pole(loc, random_cplx(), order_dist(rng));
            }

            const Cplx a{0.0, 0.0};   // random maps never reference `a` (param_power == 0)
            const int d = m.degree(a);
            const auto cp = m.critical_points(a);
            if (static_cast<int>(cp.size()) != 2 * d - 2) {
                all_ok = false;
                std::printf("  MISMATCH %s: formula=%s degree=%d found=%zu expected=%d\n",
                            m.name().c_str(), m.to_formula().c_str(), d, cp.size(), 2 * d - 2);
            }
        }
        check(all_ok, "20 random sandbox maps all satisfy total multiplicity == 2d-2");
    }

    // ---- critical_points_constant: structural parameter-independence flag ----
    std::printf("\ncritical_points_constant:\n");
    {
        // Every z^n + a preset: the "+a" term has exponent 0, so it never
        // reaches the derivative -- the motivating case this exists for.
        check(RationalMap::mandelbrot().critical_points_constant(),
              "mandelbrot: critical points are constant (0 and infinity, always)");
        check(RationalMap::multibrot(5).critical_points_constant(),
              "multibrot(5): critical points are constant");
        check(RationalMap::newton_cubic().critical_points_constant(),
              "newton_cubic: no parameter dependence anywhere, so trivially constant");
    }
    {
        // mcmullen's pole strength is a*1, so its own location's TRUE order
        // -- and hence critical_points()'s pole/infinity multiplicities --
        // genuinely depends on `a` (see the mcmullen critical_points block
        // above: the pole itself becomes critical only because of that
        // dependence). Must NOT be flagged constant.
        check(!RationalMap::mcmullen(2).critical_points_constant(),
              "mcmullen(2): pole strength depends on a, so NOT constant");
        check(!RationalMap::mcmullen(3).critical_points_constant(),
              "mcmullen(3): same reasoning");
    }
    {
        RationalMap m("moving-pole");
        m.add_poly({1, 0}, 2, 0);
        m.add_pole({0, 0}, {1, 0}, 1, 0, "fixed strength");
        m.pole_terms()[0].location_is_param = true;   // pole tracks a
        check(!m.critical_points_constant(),
              "a pole whose location tracks the parameter is NOT constant, "
              "even with param_power == 0 everywhere");
    }
    {
        // Cross-check against the actual set: for every one of a spread of
        // `a` values, a map flagged constant must produce the exact same
        // critical_points() result -- this is the property the flag exists
        // to guarantee, not just a hardcoded expectation on presets.
        RationalMap m = RationalMap::mandelbrot();
        check(m.critical_points_constant(), "mandelbrot flagged constant (precondition)");
        auto sorted_by_finite = [](std::vector<Cplx> v) {
            std::sort(v.begin(), v.end(), [](Cplx x, Cplx y) {
                if (std::isinf(x.real()) != std::isinf(y.real()))
                    return std::isinf(y.real());   // finite first
                return x.real() < y.real();
            });
            return v;
        };
        const auto reference = sorted_by_finite(m.critical_points(Cplx(1.0, 0.0)));
        bool all_match = true;
        for (Cplx a : as) {
            const auto cp = sorted_by_finite(m.critical_points(a));
            if (cp.size() != reference.size()) { all_match = false; continue; }
            for (std::size_t i = 0; i < cp.size(); ++i) {
                const bool both_inf =
                    std::isinf(cp[i].real()) && std::isinf(reference[i].real());
                if (!both_inf && !close(cp[i], reference[i], 1e-6)) all_match = false;
            }
        }
        check(all_match,
              "mandelbrot: critical_points(a) is bit-for-bit the same set across varied a, "
              "as critical_points_constant() promises");
    }

    // ---- compile: CompiledMap::step matches eval exactly -----------------------
    std::printf("\ncompile (CompiledMap):\n");
    {
        bool eval_ok = true;
        for (RationalMap m : {RationalMap::mandelbrot(), RationalMap::mcmullen(3),
                              RationalMap::newton_cubic()}) {
            for (Cplx a : as) {
                const CompiledMap c = m.compile(a);
                for (Cplx z : zs) {
                    double zr = z.real(), zi = z.imag();
                    c.step(zr, zi);
                    if (!close(Cplx(zr, zi), m.eval(z, a))) eval_ok = false;
                }
            }
        }
        check(eval_ok, "CompiledMap::step matches RationalMap::eval for every (z, a)");
    }
    {
        // A disabled term must vanish from the compiled evaluator too, not
        // just from eval() -- compile() has its own `enabled` filter to
        // keep in sync.
        RationalMap m = RationalMap::mandelbrot();
        m.poly_terms()[1].enabled = false;
        const CompiledMap c = m.compile({5.0, 0.0});
        double zr = 2.0, zi = 0.0;
        c.step(zr, zi);
        check(close(Cplx(zr, zi), Cplx(4.0, 0.0)),
              "compile() respects disabled terms, matching eval()'s behaviour");
    }
    {
        // Evaluating exactly at a pole returns the same huge sentinel as
        // eval(), not NaN.
        RationalMap m("test");
        m.add_pole({1.0, 0.0}, {1.0, 0.0}, 1);
        const CompiledMap c = m.compile({0, 0});
        double zr = 1.0, zi = 0.0;
        c.step(zr, zi);
        check(zr > 1e299,
              "the compiled evaluator returns the escape sentinel at a pole, matching eval()");
    }
    {
        // Exponents outside CompiledMap::step's unrolled |e| <= 4 special
        // cases (see cdx::detail::cipow) must still agree with the
        // std::complex-based ipow() eval() uses -- both positive (z^7) and
        // negative (z^-5, a pole of an unusually high order).
        RationalMap m("high-exponent");
        m.add_poly({1.0, 0.0}, 7, 0, "z^7");
        m.add_pole({0.3, -0.2}, {2.0, 1.0}, 5, 0, "order-5 pole");
        const CompiledMap c = m.compile({0, 0});
        bool ok = true;
        for (Cplx z : zs) {
            double zr = z.real(), zi = z.imag();
            c.step(zr, zi);
            if (!close(Cplx(zr, zi), m.eval(z, {0, 0}), 1e-6)) ok = false;
        }
        check(ok, "CompiledMap matches eval() for exponents beyond the |e|<=4 unrolled cases");
    }

    // ---- distinct_critical_points: clusters multiplicity to one point each ---
    std::printf("\ndistinct_critical_points:\n");
    {
        RationalMap m = RationalMap::multibrot(3);
        const Cplx a{0.3, -0.7};
        const auto cp = m.critical_points(a);
        const auto dcp = m.distinct_critical_points(a);
        check(cp.size() == 4, "multibrot(3): critical_points has multiplicity (4 total)");
        check(dcp.size() == 2,
              "multibrot(3): distinct_critical_points collapses to 2 (0 and infinity)");
        int n_zero = 0, n_inf = 0;
        for (Cplx z : dcp) {
            if (std::isinf(z.real())) ++n_inf;
            else if (std::abs(z) < 1e-4) ++n_zero;
        }
        check(n_zero == 1 && n_inf == 1, "multibrot(3): exactly one representative each");
    }
    {
        // No multiplicity to collapse: distinct == critical_points here.
        RationalMap m = RationalMap::mandelbrot();
        const Cplx a{0.3, -0.7};
        check(m.critical_points(a).size() == m.distinct_critical_points(a).size(),
              "mandelbrot: nothing to collapse when every critical point is already simple");
    }
    {
        // Explicit tolerance override still behaves sanely.
        RationalMap m = RationalMap::multibrot(5);
        const Cplx a{0.3, -0.7};
        const auto dcp = m.distinct_critical_points(a, 1e-3);
        check(dcp.size() == 2, "multibrot(5): distinct points with an explicit tolerance");
    }

    // ---- pole_orders: true local order, not naive per-term sums --------------
    std::printf("\npole_orders:\n");
    {
        RationalMap m = RationalMap::mcmullen(2);
        const auto locs = m.pole_locations({0.3, -0.7});
        const auto orders = m.pole_orders({0.3, -0.7});
        check(locs.size() == 1 && orders.size() == 1 && orders[0] == 2,
              "mcmullen(2): the pole at the origin has order 2 (matches add_pole's order=n)");
    }
    {
        RationalMap m = RationalMap::newton_cubic();
        const auto orders = m.pole_orders({0, 0});
        check(orders.size() == 1 && orders[0] == 2,
              "newton_cubic: the pole at the origin has order 2");
    }
    {
        // R(z) = 2/z from two redundant order-1 pole terms: true order is 1
        // (a simple pole), not the naive sum 1+1=2.
        RationalMap m("mobius");
        push_pole_direct(m, {0, 0}, {1, 0}, 1);
        push_pole_direct(m, {0, 0}, {1, 0}, 1);
        const auto orders = m.pole_orders({0, 0});
        check(orders.size() == 1 && orders[0] == 1,
              "R(z)=2/z: true order is 1, not the naive per-term sum of 2");
    }
    {
        RationalMap m = RationalMap::mandelbrot();
        check(m.pole_orders({0.3, -0.7}).empty(), "mandelbrot: no poles, so no orders");
    }

    // ---- fixed_points ----------------------------------------------------------
    std::printf("\nfixed_points:\n");
    {
        // The basilica: z^2-1 = z  =>  z^2-z-1=0, the golden ratio and its
        // conjugate, both REPELLING (|multiplier|>1) -- cross-checks
        // test_analysis.cpp's find_attractors result of zero attracting
        // fixed points for this exact map/parameter. Plus infinity,
        // superattracting (multiplier 0, matches critical_points()'s own
        // infinity rule for this map).
        RationalMap m = RationalMap::mandelbrot();
        const Cplx a{-1.0, 0.0};
        const auto fps = m.fixed_points(a);
        check(fps.size() == 3, "basilica: 3 fixed points (golden ratio pair + infinity)");

        const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
        int n_golden_pair = 0, n_inf = 0, n_finite_attracting = 0;
        for (const auto& fp : fps) {
            if (std::isinf(fp.point.real())) {
                ++n_inf;
                check(close(fp.multiplier, Cplx(0, 0), 1e-9),
                      "basilica: infinity is superattracting (multiplier 0)");
            } else {
                if (close(fp.point, Cplx(golden, 0), 1e-6) ||
                    close(fp.point, Cplx(1.0 - golden, 0), 1e-6))
                    ++n_golden_pair;
                if (std::abs(fp.multiplier) < 1.0) ++n_finite_attracting;
            }
        }
        check(n_golden_pair == 2, "basilica: the two finite fixed points are the golden ratio pair");
        check(n_inf == 1, "basilica: infinity is among the fixed points");
        // Infinity itself IS attracting here (superattracting, even) -- the
        // "zero attracting fixed points" fact from find_attractors/CLAUDE.md
        // is specifically about the FINITE plane (the classical,
        // pre-sphere-first framing this project explicitly rejects
        // everywhere else); on the sphere, infinity being an attracting
        // fixed point is unremarkable and expected.
        check(n_finite_attracting == 0,
              "basilica: ZERO finite attracting fixed points (matches find_attractors exactly)");
    }
    {
        // Newton's method: fixed points of N are exactly the roots of
        // z^3-1, each superattracting (multiplier 0, the whole point of
        // Newton's method) -- plus infinity, this time an ORDINARY
        // (non-critical) fixed point with multiplier 3/2 (repelling),
        // matching the p-q==1 formula and the earlier hand-derivation that
        // motivated find_attractors' pole-seed perturbation fix.
        RationalMap m = RationalMap::newton_cubic();
        const Cplx a{0.0, 0.0};
        const auto fps = m.fixed_points(a);
        check(fps.size() == 4, "newton_cubic: 4 fixed points (3 roots + infinity)");

        const std::vector<Cplx> cube_roots = {
            {1.0, 0.0}, {-0.5, 0.8660254037844386}, {-0.5, -0.8660254037844386}};
        int n_roots_superattracting = 0, n_inf_repelling = 0;
        for (const auto& fp : fps) {
            if (std::isinf(fp.point.real())) {
                if (close(fp.multiplier, Cplx(1.5, 0.0), 1e-6)) ++n_inf_repelling;
            } else {
                for (Cplx r : cube_roots)
                    if (close(fp.point, r, 1e-6) && close(fp.multiplier, Cplx(0, 0), 1e-9))
                        ++n_roots_superattracting;
            }
        }
        check(n_roots_superattracting == 3,
              "newton_cubic: all 3 roots are superattracting fixed points (multiplier 0)");
        check(n_inf_repelling == 1,
              "newton_cubic: infinity is an ordinary fixed point with multiplier 3/2 (repelling)");
    }
    {
        RationalMap m("mobius");
        push_pole_direct(m, {0, 0}, {1, 0}, 1);
        push_pole_direct(m, {0, 0}, {1, 0}, 1);
        const auto fps = m.fixed_points({0, 0});
        // R(z)=2/z: R(z)=z => z^2=2 => z=+-sqrt(2), each with multiplier -1
        // (R is an involution, R(R(z))=z, so this is the standard neutral
        // multiplier of an involution's fixed points). p_deg=1 < q_deg=2,
        // so infinity is NOT a fixed point here -- R(z)->0 as z->infinity,
        // not infinity; 0 and infinity actually form their own 2-cycle
        // (R(0)=infinity, R(infinity)=0), which is exactly why they are
        // NOT in this fixed-points list.
        check(fps.size() == 2, "R(z)=2/z: exactly 2 fixed points (+-sqrt(2)), not infinity");
        bool at_pole = false, any_inf = false;
        for (const auto& fp : fps) {
            if (std::isinf(fp.point.real())) any_inf = true;
            else if (std::abs(fp.point) < 1e-6) at_pole = true;
            else check(close(fp.multiplier, Cplx(-1, 0), 1e-6),
                       "R(z)=2/z: each fixed point has multiplier -1 (involution)");
        }
        check(!at_pole, "R(z)=2/z: the pole at the origin is correctly excluded, not a fixed point");
        check(!any_inf, "R(z)=2/z: infinity is correctly excluded (R sends infinity to 0, not itself)");
    }

    // ---- enabled/disabled terms ----------------------------------------------
    std::printf("\nenabled/disabled terms:\n");
    {
        RationalMap m = RationalMap::mandelbrot();
        m.poly_terms()[1].enabled = false;   // mute the 'a' term
        check(close(m.eval({2.0, 0.0}, {5.0, 0.0}), Cplx(4.0, 0.0)),
              "disabling the parameter term drops it from eval");
    }

    // ---- to_formula: exact for integer-coefficient presets --------------------
    std::printf("\nto_formula:\n");
    check(RationalMap::mandelbrot().to_formula() == "z^2 + a", "mandelbrot formula");
    check(RationalMap::multibrot(3).to_formula() == "z^3 + a", "multibrot(3) formula");
    check(RationalMap::mcmullen(2).to_formula() == "z^2 + a/z^2", "mcmullen(2) formula");
    {
        // The pole at the origin is now an explicit PoleTerm (add_poly
        // rejects negative exponents), so it prints in PoleTerm's own
        // "1/z^n" notation (see to_formula's pole-term branch, same shape
        // mcmullen(2)'s "a/z^2" above already exercises) rather than a
        // literal "z^-2" substring.
        const std::string f = RationalMap::newton_cubic().to_formula();
        check(f.find("z") != std::string::npos && f.find("/z^2") != std::string::npos,
              "newton_cubic formula mentions z and the pole 1/z^2");
    }

    // ---- serialize/deserialize round-trip --------------------------------------
    std::printf("\nserialize/deserialize round-trip:\n");
    {
        RationalMap presets[] = {
            RationalMap::mandelbrot(), RationalMap::multibrot(5),
            RationalMap::mcmullen(3), RationalMap::newton_cubic(),
        };
        bool all_ok = true;
        for (auto& m : presets) {
            const std::string text = m.serialize();
            RationalMap round;
            std::string err;
            if (!RationalMap::deserialize(text, round, err)) { all_ok = false; continue; }
            if (round.name() != m.name()) all_ok = false;
            for (Cplx z : zs) for (Cplx a : as)
                if (!close(round.eval(z, a), m.eval(z, a))) all_ok = false;
        }
        check(all_ok, "round-tripped presets evaluate identically to the originals");
    }
    {
        std::string err;
        RationalMap out;
        check(!RationalMap::deserialize("poly 1 0 2 0 1\nend\n", out, err),
              "deserialize without a 'map' header fails");
        check(!err.empty(), "failure sets a non-empty error message");
    }
    {
        std::string err;
        RationalMap out;
        check(!RationalMap::deserialize("map x\npoly 1 0 2\nend\n", out, err),
              "malformed poly line (missing fields) fails");
    }

    // ---- FamilyLibrary -----------------------------------------------------------
    std::printf("\nFamilyLibrary:\n");
    {
        FamilyLibrary lib = FamilyLibrary::with_defaults();
        const std::string text = lib.serialize();
        FamilyLibrary round;
        std::string err;
        const bool ok = FamilyLibrary::deserialize(text, round, err);
        check(ok, "library round-trips through serialize/deserialize");
        check(round.size() == lib.size(), "round-tripped library has the same number of maps");

        bool names_ok = true;
        for (const auto& name : lib.names())
            if (round.find(name) == nullptr) names_ok = false;
        check(names_ok, "every original map name is present after round-trip");

        check(lib.find("mandelbrot") != nullptr, "with_defaults includes mandelbrot");
        check(lib.find("no-such-map") == nullptr, "find on a missing name returns null");
    }
    {
        FamilyLibrary lib;
        RationalMap m("dup");
        m.add_poly({1, 0}, 1);
        lib.add(m);
        m.add_poly({2, 0}, 2);
        lib.add(m);   // same name -> replaces, does not duplicate
        check(lib.size() == 1, "adding a map with an existing name replaces it, not duplicates");
        check(lib.find("dup")->poly_terms().size() == 2,
              "the replacement's terms are the ones kept");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
