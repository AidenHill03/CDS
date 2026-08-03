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

    // ---- pole_locations: dedup and negative-exponent implication -----------
    std::printf("\npole_locations:\n");
    {
        RationalMap m("test");
        m.add_pole({1.0, 0.0}, {1.0, 0.0}, 1);
        m.add_pole({1.0, 0.0}, {2.0, 0.0}, 2);   // same location, different term
        m.add_pole({-1.0, 0.0}, {1.0, 0.0}, 1);
        auto locs = m.pole_locations({0, 0});
        check(locs.size() == 2, "duplicate pole locations are deduplicated");
    }
    {
        RationalMap m = RationalMap::newton_cubic();   // has z^-2, no explicit pole term
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
        m.add_pole({0, 0}, {1, 0}, 1);
        m.add_pole({0, 0}, {1, 0}, 1);
        auto cp = m.critical_points({0, 0});
        check(cp.empty(),
              "R(z)=2/z: correctly zero critical points despite the redundant construction");
    }
    {
        RationalMap m = RationalMap::newton_cubic();
        const Cplx a{0, 0};
        auto cp = m.critical_points(a);
        // Ordinary: z^3=1, the three roots Newton's method converges to
        // (superattracting fixed points). Pole at 0: true order 2 (from the
        // z^-2 term), multiplicity 1 -- this is the dynamically informative
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
        const std::string f = RationalMap::newton_cubic().to_formula();
        check(f.find("z") != std::string::npos && f.find("z^-2") != std::string::npos,
              "newton_cubic formula mentions z and z^-2");
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
