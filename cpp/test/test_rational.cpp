// =============================================================================
// test_rational.cpp -- property-based checks for the term-based RationalMap.
//
// Mirrors test_renderer.cpp's style: presets are checked against their
// documented closed forms, and serialization is checked as a round-trip
// property, rather than comparing against golden output.
// =============================================================================
#include "cdx/rational.hpp"

#include <cmath>
#include <cstdio>
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
