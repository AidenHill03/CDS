// =============================================================================
// test_analysis.cpp -- property-based checks for the analysis layer.
//
// Mirrors the rest of this suite's style: known dynamical facts and
// invariants, not golden output. The two headline cases are chosen
// specifically because they are where a naive implementation goes wrong:
//   * Newton z^3-1 has three attracting FIXED points -- the easy case.
//   * z^2-1 (the basilica) has an attracting 2-CYCLE and ZERO attracting
//     fixed points -- the case a fixed-point-only search silently misses,
//     which is exactly why find_attractors iterates critical orbits
//     instead of solving for fixed points.
// =============================================================================
#include "cdx/analysis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static bool close(Cplx a, Cplx b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

static bool has_point_near(const std::vector<Cplx>& pts, Cplx target, double tol = 1e-6) {
    for (Cplx p : pts) if (close(p, target, tol)) return true;
    return false;
}

// =============================================================================
// Regression oracle for the "exit cycle detection on convergence instead of
// fixed settling budgets" speed optimization (cdx/src/analysis.cpp): a
// faithful copy of the PRE-OPTIMIZATION find_attractors_from_seeds, kept here
// ONLY so the optimized version can be checked against it forever, not
// against a one-time snapshot of expected values that would rot as the
// algorithm evolves. This is deliberately the OLD (slow: unconditional
// burn_in, then a bounded closure search against a single fixed anchor)
// two-phase structure, unchanged since before that optimization -- see the
// "identical math" test below for what it's compared against and why.
// =============================================================================
namespace {

bool old_is_finite(Cplx z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

double old_chordal(Cplx z, Cplx w) {
    return chordal_distance(z.real(), z.imag(), w.real(), w.imag());
}

bool old_counts_as_infinite(Cplx z, double inf_cutoff) {
    return !old_is_finite(z) || std::abs(z) > inf_cutoff;
}

void old_add_cycle(std::vector<Cycle>& cycles, std::vector<Cplx> new_cycle, double tol) {
    for (const auto& existing : cycles) {
        if (existing.points.size() != new_cycle.size()) continue;
        const std::size_t n = existing.points.size();
        for (std::size_t shift = 0; shift < n; ++shift) {
            bool all_close = true;
            for (std::size_t i = 0; i < n; ++i) {
                if (old_chordal(new_cycle[(i + shift) % n], existing.points[i]) >= tol) {
                    all_close = false;
                    break;
                }
            }
            if (all_close) return;
        }
    }
    Cycle c;
    c.points = std::move(new_cycle);
    c.id = static_cast<int>(cycles.size()) + 1;
    cycles.push_back(std::move(c));
}

std::vector<Cplx> old_confirm_weakly_attracting(std::vector<Cplx>& orbit, const RationalMap& map,
                                                Cplx a, const FindAttractorsOptions& opts) {
    if (!opts.verify_multiplier || !opts.confirm_weakly_attracting) return {};

    Cplx z = orbit.back();
    for (int n = 0; n < opts.extended_max_period; ++n) {
        z = map.eval(z, a);
        if (old_counts_as_infinite(z, opts.inf_cutoff)) return {};
        orbit.push_back(z);
    }

    std::vector<Cplx> loose_orbit;
    loose_orbit.push_back(z);
    int loose_found = 0;
    for (int k = 0; k < opts.max_period; ++k) {
        Cplx zn = map.eval(loose_orbit.back(), a);
        if (old_counts_as_infinite(zn, opts.inf_cutoff)) return {};
        loose_orbit.push_back(zn);
        if (old_chordal(zn, loose_orbit.front()) < opts.loose_tol) { loose_found = k + 1; break; }
    }
    if (loose_found <= 0) return {};

    Cplx z0 = loose_orbit.front();
    for (int iter = 0; iter < opts.newton_iterations; ++iter) {
        Cplx zk = z0;
        Cplx deriv_prod(1.0, 0.0);
        bool ok = true;
        for (int i = 0; i < loose_found; ++i) {
            if (!old_is_finite(zk)) { ok = false; break; }
            deriv_prod *= map.deriv(zk, a);
            zk = map.eval(zk, a);
        }
        if (!ok || !old_is_finite(zk)) return {};
        const Cplx gprime = deriv_prod - Cplx(1.0, 0.0);
        if (std::abs(gprime) < 1e-9) return {};
        z0 -= (zk - z0) / gprime;
        if (!old_is_finite(z0)) return {};
    }

    std::vector<Cplx> refined;
    Cplx zk = z0;
    Cplx multiplier(1.0, 0.0);
    for (int i = 0; i < loose_found; ++i) {
        if (!old_is_finite(zk)) return {};
        refined.push_back(zk);
        multiplier *= map.deriv(zk, a);
        zk = map.eval(zk, a);
    }
    if (old_chordal(zk, z0) >= opts.tol * 1e2) return {};
    if (!(std::abs(multiplier) < 1.0 - opts.attracting_margin)) return {};
    return refined;
}

std::vector<Cycle> old_find_attractors_from_seeds(const std::vector<Cplx>& seeds,
                                                  const RationalMap& map, Cplx a,
                                                  const FindAttractorsOptions& opts,
                                                  int* unresolved_count) {
    const double kInfLocal = std::numeric_limits<double>::infinity();
    std::vector<Cycle> cycles;
    if (unresolved_count) *unresolved_count = 0;

    for (Cplx seed : seeds) {
        Cplx z = old_is_finite(seed) ? seed : Cplx(opts.inf_cutoff, 0.0);
        if (old_is_finite(z)) {
            for (Cplx p : map.pole_locations(a)) {
                if (std::abs(z - p) < 1e-9) { z += Cplx(1e-4, 3.7e-5); break; }
            }
        }

        bool at_inf = false;
        for (int n = 0; n < opts.burn_in; ++n) {
            z = map.eval(z, a);
            if (old_counts_as_infinite(z, opts.inf_cutoff)) { at_inf = true; break; }
        }

        if (at_inf) {
            if (opts.verify_multiplier) {
                bool infinity_attracting = false;
                for (const FixedPoint& fp : map.fixed_points(a)) {
                    if (!old_is_finite(fp.point) && std::abs(fp.multiplier) < 1.0) {
                        infinity_attracting = true;
                        break;
                    }
                }
                if (!infinity_attracting) {
                    if (unresolved_count) ++*unresolved_count;
                    continue;
                }
            }
            old_add_cycle(cycles, {Cplx(kInfLocal, 0.0)}, opts.tol);
            continue;
        }

        std::vector<Cplx> orbit;
        orbit.push_back(z);
        int found = 0;
        bool hit_inf_mid_orbit = false;
        for (int k = 0; k < opts.max_period; ++k) {
            Cplx zn = map.eval(orbit.back(), a);
            const bool zn_inf = old_counts_as_infinite(zn, opts.inf_cutoff);
            if (zn_inf) zn = Cplx(kInfLocal, 0.0);
            orbit.push_back(zn);
            if (old_chordal(zn, orbit.front()) < opts.tol) { found = k + 1; break; }
            if (zn_inf) { hit_inf_mid_orbit = true; break; }
        }

        if (found <= 0 || hit_inf_mid_orbit) {
            std::vector<Cplx> refined = !hit_inf_mid_orbit
                ? old_confirm_weakly_attracting(orbit, map, a, opts)
                : std::vector<Cplx>{};
            if (!refined.empty()) {
                old_add_cycle(cycles, std::move(refined), opts.tol * 1e3);
                continue;
            }
            if (unresolved_count) ++*unresolved_count;
            continue;
        }

        std::vector<Cplx> cyc(orbit.begin(), orbit.begin() + found);
        if (opts.verify_multiplier) {
            const bool has_inf = std::any_of(cyc.begin(), cyc.end(),
                                             [](Cplx zz) { return !old_is_finite(zz); });
            if (!has_inf) {
                Cplx multiplier(1.0, 0.0);
                for (Cplx zc : cyc) multiplier *= map.deriv(zc, a);
                if (!(std::abs(multiplier) < 1.0)) {
                    if (unresolved_count) ++*unresolved_count;
                    continue;
                }
            }
        }
        old_add_cycle(cycles, std::move(cyc), opts.tol * 1e3);
    }
    return cycles;
}

// (period, |multiplier|) for a cycle -- the Inf-cycle case reuses
// fixed_points()' own algebraic multiplier there, same as dynamical_facts().
std::pair<int, double> old_cycle_signature(const Cycle& cyc, const RationalMap& map, Cplx a) {
    if (cyc.points.size() == 1 && !old_is_finite(cyc.points[0])) {
        for (const auto& fp : map.fixed_points(a)) {
            if (!old_is_finite(fp.point)) return {1, std::abs(fp.multiplier)};
        }
    }
    Cplx multiplier(1.0, 0.0);
    for (Cplx z : cyc.points) multiplier *= map.deriv(z, a);
    return {static_cast<int>(cyc.points.size()), std::abs(multiplier)};
}

}  // namespace

int main() {
    std::printf("=== cdx analysis tests ===\n");

    // ---- find_attractors: Newton z^3-1 -----------------------------------------
    std::printf("\nfind_attractors: Newton z^3-1:\n");
    {
        RationalMap m = RationalMap::newton_cubic();
        const Cplx a{0.0, 0.0};   // param is unused by this family
        const auto cycles = find_attractors(m, a);

        check(cycles.size() == 3, "exactly 3 attracting cycles");
        bool all_fixed = true;
        for (const auto& c : cycles) if (c.points.size() != 1) all_fixed = false;
        check(all_fixed, "all three are fixed points (period 1)");

        const std::vector<Cplx> roots = {
            {1.0, 0.0}, {-0.5, 0.8660254037844386}, {-0.5, -0.8660254037844386}};
        bool all_roots_found = true;
        for (Cplx r : roots) {
            bool found = false;
            for (const auto& c : cycles) if (close(c.points[0], r, 1e-4)) found = true;
            if (!found) all_roots_found = false;
        }
        check(all_roots_found, "the three fixed points are the cube roots of unity");

        // Render the discovered cycles' basins and check the same properties
        // test_renderer.cpp's hand-coded Newton3 test checks -- but here the
        // cycles came from find_attractors, not from hardcoding {1,w,w^2}.
        Renderer r(Map::custom(m, a), Viewport{{0.0, 0.0}, 2.0, 251},
                   RenderSettings{200, 2.0, 1e-6, 0});
        Image basin = r.render_basin(cycles);
        long unresolved = 0, counts[4] = {0, 0, 0, 0};
        for (double v : basin.data) {
            const int k = static_cast<int>(v);
            if (k == 0) ++unresolved;
            if (k >= 1 && k <= 3) ++counts[k];
        }
        const double frac_un = static_cast<double>(unresolved) / basin.data.size();
        check(frac_un < 0.02, "nearly every pixel resolves (unresolved fraction ~0)");
        const double b1 = counts[1], b2 = counts[2], b3 = counts[3];
        const double mx = std::max({b1, b2, b3}), mn = std::min({b1, b2, b3});
        check(mn > 0 && (mx - mn) / mx < 0.10, "three near-equal basins");
    }

    // ---- find_attractors: z^2-1 (the basilica) ---------------------------------
    std::printf("\nfind_attractors: z^2-1 (basilica):\n");
    {
        RationalMap m = RationalMap::mandelbrot();
        const Cplx a{-1.0, 0.0};
        const auto cycles = find_attractors(m, a);

        int n_fixed = 0, n_2cycle = 0, n_inf = 0;
        for (const auto& c : cycles) {
            if (c.points.size() == 1 && std::isinf(c.points[0].real())) ++n_inf;
            else if (c.points.size() == 1) ++n_fixed;
            else if (c.points.size() == 2) ++n_2cycle;
        }
        check(cycles.size() == 2, "exactly 2 attracting cycles (the 2-cycle and infinity)");
        check(n_fixed == 0, "ZERO attracting fixed points (the fixed-point-only blind spot)");
        check(n_2cycle == 1, "one attracting 2-cycle");
        check(n_inf == 1, "one cycle at infinity");

        // The basilica's 2-cycle is the critical orbit of 0 itself: {0,-1}.
        for (const auto& c : cycles) {
            if (c.points.size() == 2) {
                check(has_point_near(c.points, Cplx(0.0, 0.0)) &&
                      has_point_near(c.points, Cplx(-1.0, 0.0)),
                      "the 2-cycle is exactly {0, -1}");
            }
        }
    }

    // ---- find_attractors: multiplier verification rejects non-attracting -----
    std::printf("\nfind_attractors: multiplier verification:\n");
    {
        // z (the identity map, as a poly term) has multiplier 1 everywhere --
        // every point is a (non-attracting, neutral) fixed point. With
        // verify_multiplier on, none should be reported as attracting.
        RationalMap m("identity");
        m.add_poly({1, 0}, 1);
        FindAttractorsOptions opts;
        opts.burn_in = 5;   // tiny burn-in is enough; nothing moves anyway
        const auto cycles = find_attractors(m, {0, 0}, opts);
        check(cycles.empty(), "the identity map has no attracting cycles (multiplier is always 1)");
    }

    // ---- wada_diagnostic --------------------------------------------------------
    std::printf("\nwada_diagnostic:\n");
    {
        // Newton's basins share a genuine Wada boundary: every boundary
        // pixel should tend toward seeing all 3 basins as resolution grows.
        RationalMap m = RationalMap::newton_cubic();
        const auto cycles = find_attractors(m, {0, 0});
        Renderer r(Map::custom(m, {0, 0}), Viewport{{0.0, 0.0}, 2.0, 121},
                   RenderSettings{200, 2.0, 1e-6, 0});
        Image basin = r.render_basin(cycles);

        const WadaStats stats = wada_diagnostic(basin);
        check(stats.n_basins == 3, "3 basins detected");
        check(stats.unresolved_fraction < 0.02, "low unresolved fraction");
        check(stats.boundary_fraction > 0.0, "some boundary pixels exist");
        check(stats.wada_fraction > 0.0 && !std::isnan(stats.wada_fraction),
              "a positive fraction of boundary pixels see all 3 basins");
        check(stats.radius_px >= 1, "radius scales to at least 1 pixel");

        // resolution-scaling: a coarser image should still report a
        // reasonable (nonzero, finite) radius, not a fixed constant.
        Renderer r2(Map::custom(m, {0, 0}), Viewport{{0.0, 0.0}, 2.0, 41},
                    RenderSettings{200, 2.0, 1e-6, 0});
        const WadaStats stats2 = wada_diagnostic(r2.render_basin(cycles));
        check(stats2.radius_px >= 1 && stats2.radius_px <= stats.radius_px,
              "radius_px scales down for a lower-resolution image");
    }
    {
        // Single-basin image: no boundary, wada_fraction undefined (NaN).
        Image single(20, 20);
        for (double& v : single.data) v = 1.0;
        const WadaStats stats = wada_diagnostic(single);
        check(stats.n_basins == 1, "single basin detected");
        check(std::isnan(stats.wada_fraction), "wada_fraction is NaN with fewer than 2 basins");
    }

    // ---- hausdorff_distance ------------------------------------------------------
    std::printf("\nhausdorff_distance:\n");
    {
        const std::vector<Cplx> pts = {{0, 0}, {1, 0}, {0, 1}, {-1, -1}, {2, -1}};
        const auto self = hausdorff_distance(pts, pts);
        check(self.chordal == 0.0, "chordal Hausdorff of a set against itself is 0");
        check(self.euclidean == 0.0, "euclidean Hausdorff of a set against itself is 0");
    }
    {
        // Well-separated points (spacing >> delta), each translated by the
        // same small delta: the nearest-neighbour match for each translated
        // point is unambiguously its own pre-image, so both directed
        // distances (each way) are exactly the per-point translate
        // distance, computed independently via the SAME primitives
        // hausdorff_distance itself is built on (abs() and chordal_distance)
        // -- not an approximation.
        const std::vector<Cplx> pts = {{0, 0}, {10, 0}, {0, 10}, {-10, -10}, {20, -5}};
        const Cplx delta{0.01, -0.02};
        std::vector<Cplx> translated;
        for (Cplx p : pts) translated.push_back(p + delta);

        double expect_euclidean = 0.0, expect_chordal = 0.0;
        for (Cplx p : pts) {
            expect_euclidean = std::max(expect_euclidean, std::abs(delta));
            expect_chordal = std::max(expect_chordal,
                chordal_distance(p.real(), p.imag(), (p + delta).real(), (p + delta).imag()));
        }

        const auto d = hausdorff_distance(pts, translated);
        check(close(Cplx(d.euclidean, 0), Cplx(expect_euclidean, 0), 1e-9),
              "euclidean Hausdorff of a translate is exactly the translation distance");
        check(close(Cplx(d.chordal, 0), Cplx(expect_chordal, 0), 1e-9),
              "chordal Hausdorff of a translate matches the chordal translation distance");
    }
    {
        // Directed asymmetry: target has an extra far-away point with
        // nothing near it in julia_points -- target_to_julia should be
        // large while julia_to_target (every julia point has a close
        // target neighbour) stays small.
        const std::vector<Cplx> julia = {{0, 0}, {1, 0}, {0, 1}};
        const std::vector<Cplx> target = {{0, 0}, {1, 0}, {0, 1}, {100, 100}};
        const auto d = hausdorff_distance(julia, target);
        check(d.euclidean_julia_to_target < 1e-9, "julia_to_target is tiny (every julia point matched)");
        check(d.euclidean_target_to_julia > 100.0, "target_to_julia is large (the far point is missed)");
        check(close(Cplx(d.euclidean, 0), Cplx(d.euclidean_target_to_julia, 0)),
              "symmetric distance is the max of the two directed distances");
    }
    {
        const auto d = hausdorff_distance({}, {{0, 0}});
        check(std::isinf(d.chordal) && std::isinf(d.euclidean),
              "an empty side reports infinity, not a false zero match");
    }

    // ---- extract_boundary_points -------------------------------------------------
    std::printf("\nextract_boundary_points:\n");
    {
        // Two basins split down the middle column: boundary pixels are
        // exactly those adjacent to the split.
        Image labels(4, 4);
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                labels.at(col, row) = col < 2 ? 1.0 : 2.0;
        Viewport v{{0.0, 0.0}, 2.0, 4};
        const auto boundary = extract_boundary_points(labels, v);
        check(!boundary.empty(), "boundary points were extracted");
        check(boundary.size() == 8, "8 boundary pixels (2 columns x 4 rows either side of the split)");
    }

    // ---- dynamical_facts ---------------------------------------------------------
    std::printf("\ndynamical_facts:\n");
    {
        RationalMap m = RationalMap::newton_cubic();
        const Cplx a{0.0, 0.0};
        const auto facts = dynamical_facts(m, a);

        check(facts.degree == 3, "newton_cubic: degree 3");
        check(facts.critical_points.size() == 4,
              "newton_cubic: 4 critical points (3 roots + pole), matches critical_points()");
        check(facts.pole_locations.size() == 1 && facts.pole_orders.size() == 1 &&
              facts.pole_orders[0] == 2,
              "newton_cubic: one pole at the origin, order 2");
        check(facts.fixed_points.size() == 4,
              "newton_cubic: 4 fixed points (3 roots + infinity)");

        check(facts.attracting_cycles.size() == 3,
              "newton_cubic: 3 attracting cycles, matches find_attractors() directly");
        bool all_period1_superattracting = true;
        for (const auto& ac : facts.attracting_cycles) {
            if (ac.period != 1) all_period1_superattracting = false;
            if (std::abs(ac.multiplier) > 1e-9) all_period1_superattracting = false;
        }
        check(all_period1_superattracting,
              "newton_cubic: all 3 attracting cycles are superattracting fixed points");
    }
    {
        RationalMap m = RationalMap::mandelbrot();
        const Cplx a{-1.0, 0.0};   // the basilica
        const auto facts = dynamical_facts(m, a);

        check(facts.degree == 2, "basilica: degree 2");
        check(facts.pole_locations.empty() && facts.pole_orders.empty(),
              "basilica: no poles");
        check(facts.fixed_points.size() == 3,
              "basilica: 3 fixed points, matches RationalMap::fixed_points() directly");

        check(facts.attracting_cycles.size() == 2,
              "basilica: 2 attracting cycles (the 2-cycle and infinity)");
        bool found_2cycle = false, found_inf = false;
        for (const auto& ac : facts.attracting_cycles) {
            if (ac.period == 2) {
                found_2cycle = true;
                // {0,-1} contains the critical point 0 itself -> superattracting.
                check(std::abs(ac.multiplier) < 1e-9,
                      "basilica: the 2-cycle is superattracting (it contains the critical point)");
            } else if (ac.period == 1 && std::isinf(ac.points[0].real())) {
                found_inf = true;
                check(std::abs(ac.multiplier) < 1e-9,
                      "basilica: infinity is superattracting, matches fixed_points()");
            }
        }
        check(found_2cycle && found_inf, "basilica: both expected cycle shapes are present");
    }
    {
        // Custom discovery options thread through to find_attractors.
        RationalMap m = RationalMap::mandelbrot();
        const Cplx a{-1.0, 0.0};
        FindAttractorsOptions opts;
        opts.burn_in = 5;   // absurdly small -- likely too little to converge
        const auto facts = dynamical_facts(m, a, opts);
        check(facts.degree == 2, "custom opts: facts unrelated to discovery are unaffected");
        // Not asserting a specific cycle count here: the point is only that
        // opts is honored (this is exercised properly by
        // find_attractors' own tests); a degenerate burn_in is just a cheap
        // way to prove the parameter is actually threaded through.
    }

    // ---- polynomial_escape_certified (Stage 1) ---------------------------------
    std::printf("\npolynomial_escape_certified:\n");
    {
        check(polynomial_escape_certified(RationalMap::mandelbrot()) == true,
              "z^2+a (mandelbrot): certified -- no poles, degree 2");
        check(polynomial_escape_certified(RationalMap::multibrot(5)) == true,
              "z^5+a (multibrot): certified -- no poles, degree 5");
        check(polynomial_escape_certified(RationalMap::newton_cubic()) == false,
              "Newton z^3-1: NOT certified -- has a pole at the origin");
        check(polynomial_escape_certified(RationalMap::mcmullen(2)) == false,
              "mcmullen z^2+a/z^2: NOT certified -- has a pole, even though infinity "
              "happens to be attracting there too (see the next section)");

        // Nova-style hand-built rational map (a pole term present, structurally,
        // regardless of its strength at any one parameter) -- degree >= 2 alone
        // is not enough; a pole anywhere disqualifies the fast path.
        RationalMap nova("nova");
        nova.add_poly({0.75, 0.0}, 1, 0, "linear");
        nova.add_pole({0.0, 0.0}, {0.25, 0.0}, 3, 0, "pole");
        check(polynomial_escape_certified(nova) == false,
              "a hand-built Nova-style map (linear term + a pole): NOT certified");

        RationalMap linear("linear");
        linear.add_poly({2.0, 0.0}, 1, 0, "2z");
        check(polynomial_escape_certified(linear) == false,
              "a degree-1 polynomial (no poles, but degree < 2): NOT certified -- "
              "infinity's multiplier there is nonzero (an ORDINARY, not "
              "superattracting, fixed point), so the escape trap doesn't hold");

        // A legacy negative-exponent PolyTerm (add_poly itself now rejects one --
        // see test_rational.cpp's own "legacy" test -- but an old deserialized
        // map can still carry one, pushed directly onto poly_terms() the same
        // way that test does) also implies a pole at the origin, exactly as
        // RationalMap::degree's own den_deg logic already treats it.
        RationalMap negexp("negexp");
        PolyTerm legacy_term;
        legacy_term.coeff = {1.0, 0.0};
        legacy_term.exponent = -3;
        negexp.poly_terms().push_back(legacy_term);
        check(polynomial_escape_certified(negexp) == false,
              "a map whose only term is a legacy negative-exponent PolyTerm: NOT "
              "certified -- it implies a pole at the origin too");
    }

    // ---- find_attractors: infinity verified via fixed_points(), not assumed (Stage 1) --
    std::printf("\nfind_attractors: infinity attractor is VERIFIED, not assumed from a "
               "large excursion:\n");
    {
        // mcmullen(2) at a=1: a RATIONAL map (has a pole at the origin) whose
        // infinity is nonetheless genuinely superattracting (numerator degree
        // exceeds denominator degree by 2 after clearing denominators) -- the
        // positive case: a map WITH poles where find_attractors must still
        // correctly report infinity as attracting, not just for a certified
        // polynomial.
        RationalMap m = RationalMap::mcmullen(2);
        const Cplx a{1.0, 0.0};
        const auto cycles = find_attractors(m, a);
        bool found_inf = false;
        for (const auto& c : cycles) {
            if (c.points.size() == 1 && std::isinf(c.points[0].real())) found_inf = true;
        }
        check(found_inf,
              "mcmullen(2) at a=1 (a RATIONAL map, has a pole): infinity is still "
              "correctly found as attracting -- the fix doesn't just favour "
              "certified polynomials");

        const auto fp = m.fixed_points(a);
        bool inf_multiplier_near_zero = false;
        for (const auto& p : fp) {
            if (!std::isfinite(p.point.real())) inf_multiplier_near_zero = std::abs(p.multiplier) < 1e-9;
        }
        check(inf_multiplier_near_zero,
              "...and fixed_points() independently confirms its multiplier is ~0 "
              "(superattracting), consistent with find_attractors' own inclusion");
    }
    {
        // Newton z^4-1 = (3/4)z + (1/4)z^-3: infinity is an ORDINARY (repelling)
        // fixed point there (multiplier degree/(degree-1) = 4/3, matching the
        // 3/2 pattern newton_cubic's own z^3-1 case already has) -- the
        // regression case find_attractors must NOT report as attracting, even
        // though a naive "orbit got large during burn-in" heuristic could.
        RationalMap m("newton4");
        m.add_poly({0.75, 0.0}, 1, 0, "(3/4)z");
        m.add_pole({0.0, 0.0}, {0.25, 0.0}, 3, 0, "1/(4z^3)");
        const Cplx a{0.0, 0.0};

        const auto fp = m.fixed_points(a);
        bool inf_is_repelling = false;
        for (const auto& p : fp) {
            if (!std::isfinite(p.point.real())) inf_is_repelling = std::abs(p.multiplier) > 1.0;
        }
        check(inf_is_repelling,
              "sanity: fixed_points() independently confirms infinity is REPELLING "
              "here (multiplier 4/3), the case that must be rejected");

        const auto cycles = find_attractors(m, a);
        check(cycles.size() == 4, "Newton z^4-1: exactly 4 attracting cycles");
        bool any_inf = false;
        for (const auto& c : cycles) {
            if (c.points.size() == 1 && !std::isfinite(c.points[0].real())) any_inf = true;
        }
        check(!any_inf,
              "...and NONE of them is infinity -- find_attractors correctly rejects "
              "it despite infinity's own repelling fixed point sitting arbitrarily "
              "close to where a large critical-orbit excursion could otherwise be "
              "mistaken for convergence");

        const std::vector<Cplx> fourth_roots = {
            {1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}};
        bool all_found = true;
        for (Cplx r : fourth_roots) {
            bool found = false;
            for (const auto& c : cycles) {
                if (c.points.size() == 1 && close(c.points[0], r, 1e-4)) found = true;
            }
            if (!found) all_found = false;
        }
        check(all_found, "...and the four cycles found ARE the four fourth-roots of unity "
              "(+-1, +-i), each its own attracting fixed point");
    }

    // ---- MILESTONE ACCEPTANCE TEST: rational Julia is escape_radius-INVARIANT --------
    // (Stage 2) -- the Julia set is an invariant of the MAP, so escape_radius
    // must no longer touch rational classification at all; a polynomial's
    // fast path is unaffected either way. Baked as a hard test per the
    // milestone's own instruction -- must pass now and stay passing.
    std::printf("\nMILESTONE ACCEPTANCE TEST: rational Julia is escape_radius-invariant, "
               "polynomial Julia unchanged:\n");
    {
        RationalMap newton4("newton4");   // (3/4)z + (1/4)z^-3 -- Newton's method for z^4-1
        newton4.add_poly({0.75, 0.0}, 1, 0, "(3/4)z");
        newton4.add_pole({0.0, 0.0}, {0.25, 0.0}, 3, 0, "1/(4z^3)");
        const Cplx a{0.0, 0.0};
        check(polynomial_escape_certified(newton4) == false,
              "sanity: Newton z^4-1 genuinely takes the rational path");
        const auto cycles = find_attractors(newton4, a);
        check(cycles.size() == 4, "sanity: 4 attracting cycles (the 4 fourth-roots of unity)");

        const Viewport v{{0.0, 0.0}, 2.0, 121};
        Renderer r2(Map::custom(newton4, a), v, RenderSettings{100, 2.0, 1e-6, 1});
        Renderer r10(Map::custom(newton4, a), v, RenderSettings{100, 10.0, 1e-6, 1});
        Image labels2, labels10;
        const Image vals2 = r2.render_julia(nullptr, cycles, &labels2);
        const Image vals10 = r10.render_julia(nullptr, cycles, &labels10);

        bool values_identical = true, labels_identical = true;
        for (std::size_t i = 0; i < vals2.data.size(); ++i) {
            if (vals2.data[i] != vals10.data[i]) values_identical = false;
            if (labels2.data[i] != labels10.data[i]) labels_identical = false;
        }
        check(values_identical,
              "ACCEPTANCE: rational Julia (Newton z^4-1) at escape_radius=2 and "
              "escape_radius=10 produces BYTE-IDENTICAL smooth values");
        check(labels_identical,
              "ACCEPTANCE: ...and byte-identical basin labels too, not just the "
              "smooth channel");

        // Newton z^4-1 classifies the 4 root basins, with the Julia set (the
        // common boundary) as unresolved pixels -- not one dominant basin,
        // not everything unresolved.
        int counts[5] = {0, 0, 0, 0, 0};   // index 0 = unresolved, 1..4 = the 4 roots
        for (double v : labels2.data) {
            const int k = static_cast<int>(v);
            if (k >= 0 && k <= 4) ++counts[k];
        }
        const int total = static_cast<int>(labels2.data.size());
        check(counts[1] > 0 && counts[2] > 0 && counts[3] > 0 && counts[4] > 0,
              "Newton z^4-1 Julia: all 4 root basins are actually present in the image, "
              "not just discovered by find_attractors");
        check(counts[0] > 0 && counts[0] < total,
              "...and SOME pixels are unresolved (the Julia set boundary itself), but not "
              "the whole image -- a real classification, not everything-or-nothing");

        // Polynomial Julia (mandelbrot) is UNCHANGED: same escape-time values
        // regardless of what cycles/labels are passed (both ignored on the
        // certified path), and produces the SAME output the pre-Stage-2 code
        // did (compared directly against a call with the OLD, cycles-free
        // signature shape -- both defaulted here).
        RationalMap mandel = RationalMap::mandelbrot();
        const Cplx pa{-0.7269, 0.1889};
        check(polynomial_escape_certified(mandel) == true,
              "sanity: mandelbrot is certified -- takes the unchanged fast path");
        Renderer rp(Map::custom(mandel, pa), Viewport{{0.0, 0.0}, 1.5, 121},
                   RenderSettings{200, 2.0, 1e-6, 1});
        Image poly_labels;
        const Image poly_vals_with_cycles =
            rp.render_julia(nullptr, find_attractors(mandel, pa), &poly_labels);
        const Image poly_vals_default = rp.render_julia();   // no cycles/labels at all
        bool poly_identical = true;
        for (std::size_t i = 0; i < poly_vals_with_cycles.data.size(); ++i) {
            if (poly_vals_with_cycles.data[i] != poly_vals_default.data[i]) poly_identical = false;
        }
        check(poly_identical,
              "ACCEPTANCE: a certified polynomial's Julia render is IDENTICAL whether or "
              "not cycles/labels are passed -- the fast path genuinely ignores them, not "
              "just happens to agree this run");
        bool poly_labels_all_zero = true;
        for (double v : poly_labels.data) if (v != 0.0) poly_labels_all_zero = false;
        check(poly_labels_all_zero,
              "...and `labels` stays all-zero for the certified path -- no basin concept "
              "there, matching escape-time's own single-channel output");
    }

    // ---- infinity-attracting rational Julia routes infinity as a basin -------------
    std::printf("\ninfinity-attracting rational Julia: infinity is a genuine basin, not a "
               "false |z|>R ring:\n");
    {
        // z^3 + a/z^2 -- a custom rational map with no built-in equivalent,
        // whose only attracting cycle (for this a) is infinity itself (see
        // test_custom.cpp's own identical construction).
        RationalMap m("mixed");
        m.add_poly({1.0, 0.0}, 3, 0, "z^3");
        m.add_pole({0.0, 0.0}, {1.0, 0.0}, 2, 1, "a/z^2");
        const Cplx a{0.4, 0.3};
        const auto cycles = find_attractors(m, a);
        check(cycles.size() == 1 && !std::isfinite(cycles[0].points[0].real()),
              "sanity: infinity is this map's only attracting cycle at this parameter");

        const Viewport v{{0.0, 0.0}, 2.0, 101};
        Renderer r2(Map::custom(m, a), v, RenderSettings{100, 2.0, 1e-6, 1});
        Renderer r10(Map::custom(m, a), v, RenderSettings{100, 10.0, 1e-6, 1});
        Image labels2, labels10;
        const Image vals2 = r2.render_julia(nullptr, cycles, &labels2);
        const Image vals10 = r10.render_julia(nullptr, cycles, &labels10);

        bool any_resolved = false;
        for (double v2 : labels2.data) if (v2 > 0.0) any_resolved = true;
        check(any_resolved,
              "infinity is captured as a genuine, labelled basin -- not left unresolved "
              "the way an unmodelled escape would be");

        bool identical = true;
        for (std::size_t i = 0; i < labels2.data.size(); ++i) {
            if (labels2.data[i] != labels10.data[i]) identical = false;
            if (vals2.data[i] != vals10.data[i]) identical = false;
        }
        check(identical,
              "...and identically so regardless of escape_radius (2 vs 10) -- infinity's "
              "own basin membership is decided by the chordal metric, never a |z|>R test");
    }

    // ---- MILESTONE ACCEPTANCE TEST (Stage 3): rational Green's, both potentials ------
    // escape_radius-INVARIANT for a rational map; a certified polynomial's fast
    // path is unaffected by `potential`/`cycles`/`exact` entirely, mirroring
    // Stage 2's own render_julia acceptance test exactly.
    std::printf("\nMILESTONE ACCEPTANCE TEST (Stage 3): rational Green's is "
               "escape_radius-invariant, polynomial Green's unaffected by `potential`:\n");
    {
        RationalMap mandel = RationalMap::mandelbrot();
        const Cplx pa{-0.7269, 0.1889};
        check(polynomial_escape_certified(mandel) == true,
              "sanity: mandelbrot is certified -- takes the unchanged fast path");
        Renderer rp(Map::custom(mandel, pa), Viewport{{0.0, 0.0}, 1.5, 101},
                   RenderSettings{200, 2.0, 1e-6, 1});
        const Image g_default = rp.render_greens();
        Image exact_pragmatic, exact_conformal;
        const Image g_pragmatic = rp.render_greens(nullptr, find_attractors(mandel, pa),
                                                    GreensPotential::Pragmatic, &exact_pragmatic);
        const Image g_conformal = rp.render_greens(nullptr, find_attractors(mandel, pa),
                                                    GreensPotential::Conformal, &exact_conformal);
        bool poly_identical = true;
        for (std::size_t i = 0; i < g_default.data.size(); ++i) {
            if (g_default.data[i] != g_pragmatic.data[i]) poly_identical = false;
            if (g_default.data[i] != g_conformal.data[i]) poly_identical = false;
        }
        check(poly_identical,
              "ACCEPTANCE: a certified polynomial's Green's render is IDENTICAL regardless "
              "of `potential` or whether cycles are passed -- PRAGMATIC and "
              "CONFORMAL-Boettcher already coincide at a certified polynomial's own "
              "infinity, and the fast path ignores the selector entirely");

        RationalMap newton4("newton4");   // (3/4)z + (1/4)z^-3 -- Newton's method for z^4-1
        newton4.add_poly({0.75, 0.0}, 1, 0, "(3/4)z");
        newton4.add_pole({0.0, 0.0}, {0.25, 0.0}, 3, 0, "1/(4z^3)");
        const Cplx a{0.0, 0.0};
        check(polynomial_escape_certified(newton4) == false,
              "sanity: Newton z^4-1 genuinely takes the rational path");
        const auto cycles = find_attractors(newton4, a);
        check(cycles.size() == 4, "sanity: 4 attracting cycles (the 4 fourth-roots of unity)");

        const Viewport v{{0.0, 0.0}, 2.0, 121};
        for (GreensPotential pot : {GreensPotential::Pragmatic, GreensPotential::Conformal}) {
            Renderer r2(Map::custom(newton4, a), v, RenderSettings{100, 2.0, 1e-6, 1});
            Renderer r10(Map::custom(newton4, a), v, RenderSettings{100, 10.0, 1e-6, 1});
            Image exact2, exact10;
            const Image vals2 = r2.render_greens(nullptr, cycles, pot, &exact2);
            const Image vals10 = r10.render_greens(nullptr, cycles, pot, &exact10);

            bool values_identical = true, exact_identical = true;
            for (std::size_t i = 0; i < vals2.data.size(); ++i) {
                if (vals2.data[i] != vals10.data[i]) values_identical = false;
                if (exact2.data[i] != exact10.data[i]) exact_identical = false;
            }
            const char* label = pot == GreensPotential::Pragmatic ? "PRAGMATIC" : "CONFORMAL";
            std::printf("  (%s)\n", label);
            check(values_identical,
                  "ACCEPTANCE: rational Green's (Newton z^4-1) at escape_radius=2 and "
                  "escape_radius=10 produces BYTE-IDENTICAL values");
            check(exact_identical,
                  "...and byte-identical `exact` flags too -- escape_radius plays no role "
                  "anywhere in this path");

            if (pot == GreensPotential::Conformal) {
                bool some_exact = false;
                for (double e : exact2.data) if (e != 0.0) some_exact = true;
                check(some_exact,
                      "CONFORMAL genuinely computes the Boettcher potential for at least "
                      "some pixels here (each root is superattracting, local degree 2) -- "
                      "not silently falling back to Pragmatic for everything");
            }
        }
    }

    // ---- Conformal (Koenigs) potential on a geometrically attracting fixed point -----
    // R(z) = lambda*z + (1-lambda), an affine (degree-1, hence NOT
    // escape-radius-certified -- see polynomial_escape_certified's own "degree
    // < 2" test case) map with EXACT global dynamics: R(z)-1 = lambda*(z-1)
    // for every z, so the fixed point at z=1 has multiplier EXACTLY lambda,
    // and every finite point converges to it geometrically at that exact
    // rate -- ideal ground truth for testing the numerically-ESTIMATED
    // Koenigs potential against.
    std::printf("\nConformal (Koenigs) potential on a geometrically attracting fixed point:\n");
    {
        const double lambda = 0.3;
        RationalMap affine("affine");
        affine.add_poly({lambda, 0.0}, 1, 0, "lambda*z");
        affine.add_poly({1.0 - lambda, 0.0}, 0, 0, "1-lambda");
        const Cplx a{0.0, 0.0};
        check(polynomial_escape_certified(affine) == false,
              "sanity: a degree-1 map is not escape_radius-certified -- takes the rational "
              "path despite having no poles");

        // NOT via find_attractors: that seeds from distinct_critical_points,
        // and a degree-1 map has NONE (Riemann-Hurwitz: total multiplicity
        // == 2d-2 == 0) -- a genuine, expected blind spot of critical-orbit
        // discovery for a map with no critical points at all, not something
        // to route around. fixed_points() finds z=1 directly instead, the
        // same way this map's ground-truth multiplier is verified below.
        const auto fps = affine.fixed_points(a);
        bool found_p1 = false;
        for (const auto& fp : fps) {
            if (close(fp.point, Cplx{1.0, 0.0}, 1e-6)) {
                found_p1 = true;
                check(close(fp.multiplier, Cplx{lambda, 0.0}, 1e-6),
                      "fixed_points independently confirms the multiplier at z=1 is exactly "
                      "lambda -- the ground truth this test's Koenigs estimate is checked "
                      "against");
            }
        }
        check(found_p1, "sanity: z=1 is a fixed point of R(z)=lambda*z+(1-lambda)");

        const std::vector<Cycle> cycles = {Cycle{{Cplx{1.0, 0.0}}, 1}};

        const Viewport v{{1.0, 0.0}, 6.0, 121};   // centred on the attractor
        Renderer r(Map::custom(affine, a), v, RenderSettings{200, 2.0, 1e-9, 1});
        Image exact;
        const Image g = r.render_greens(nullptr, cycles, GreensPotential::Conformal, &exact);

        bool most_exact = false;
        {
            int n_exact = 0, n_resolved = 0;
            for (std::size_t i = 0; i < g.data.size(); ++i) {
                if (g.data[i] != 0.0 || exact.data[i] != 0.0) ++n_resolved;
                if (exact.data[i] != 0.0) ++n_exact;
            }
            most_exact = n_resolved > 0 && n_exact > n_resolved / 2;
        }
        check(most_exact,
              "the geometric (Koenigs) branch is genuinely selected for most resolved "
              "pixels here -- not falling back to Pragmatic");

        // Sample real-axis points at increasing distance from the attractor
        // (this map's dynamics are real-preserving) and confirm G grows
        // monotonically farther from z=1 -- G_p is supposed to vanish toward
        // the basin boundary and grow toward the attractor's own approach,
        // exactly as the certified-polynomial Boettcher formula already
        // does; a geometric basin should show the SAME qualitative shape
        // even though its magnitude isn't sign-pinned the way Boettcher's
        // is (see conformal_potential's own doc comment).
        auto conformal_at = [&](double re) {
            Renderer r1(Map::custom(affine, a), Viewport{{re, 0.0}, 0.01, 3},
                       RenderSettings{200, 2.0, 1e-9, 1});
            Image ex1;
            const Image gv = r1.render_greens(nullptr, cycles, GreensPotential::Conformal, &ex1);
            return std::make_pair(gv.at(1, 1), ex1.at(1, 1));
        };
        const auto [g_near, ex_near] = conformal_at(1.5);     // |z-1| = 0.5
        const auto [g_far,  ex_far ] = conformal_at(4.0);     // |z-1| = 3.0
        check(ex_near != 0.0 && ex_far != 0.0,
              "both sample points resolve via the genuine Koenigs estimate (exact), not a "
              "Pragmatic fallback");
        check(g_far > g_near,
              "CONSISTENT WITH THE MULTIPLIER: a point 6x farther from the attractor (and "
              "so needing more lambda-scaled steps to converge) gets a strictly LARGER "
              "Conformal value -- the potential grows away from the attractor, not a flat "
              "or inverted field");
    }

    // ---- Conformal potential flags the near-parabolic case as approximate -----------
    // Same affine construction, but with lambda pushed right up against the
    // unit circle (0.99): still technically geometric, but exactly the
    // regime this milestone's own spec calls out as too close to parabolic
    // to trust ("approximate with the pragmatic rate and CLEARLY label it as
    // not the exact conformal potential") -- conformal_potential's own
    // lambda_est < 0.98 acceptance threshold is deliberately conservative
    // about this boundary, not just the literal |lambda|==1 point.
    std::printf("\nConformal potential falls back to Pragmatic, flagged inexact, near the "
               "parabolic boundary:\n");
    {
        const double lambda = 0.99;
        RationalMap affine("affine_slow");
        affine.add_poly({lambda, 0.0}, 1, 0, "lambda*z");
        affine.add_poly({1.0 - lambda, 0.0}, 0, 0, "1-lambda");
        const Cplx a{0.0, 0.0};
        // Same reasoning as the lambda=0.3 case above: built directly from
        // the map's own algebra, not find_attractors (no critical points to
        // seed from at degree 1).
        const std::vector<Cycle> cycles = {Cycle{{Cplx{1.0, 0.0}}, 1}};

        const Viewport v{{1.0, 0.0}, 6.0, 41};
        Renderer r(Map::custom(affine, a), v, RenderSettings{4000, 2.0, 1e-9, 1});
        Image exact_conf, exact_prag;
        const Image g_conf = r.render_greens(nullptr, cycles, GreensPotential::Conformal, &exact_conf);
        const Image g_prag = r.render_greens(nullptr, cycles, GreensPotential::Pragmatic, &exact_prag);

        int n_resolved = 0, n_fallback = 0, n_matches_pragmatic = 0;
        for (std::size_t i = 0; i < g_conf.data.size(); ++i) {
            if (g_prag.data[i] == 0.0) continue;   // unresolved pixel, not relevant here
            ++n_resolved;
            if (exact_conf.data[i] == 0.0) {
                ++n_fallback;
                if (g_conf.data[i] == g_prag.data[i]) ++n_matches_pragmatic;
            }
        }
        check(n_resolved > 0, "sanity: at least some pixels resolved within max_iter");
        check(n_fallback == n_resolved,
              "FLAGGED APPROXIMATE: every resolved pixel falls back (exact == 0) at "
              "lambda=0.99 -- the near-parabolic regime is honestly reported as "
              "inexact, not silently given a fabricated Boettcher/Koenigs value");
        check(n_matches_pragmatic == n_fallback,
              "...and the fallback VALUE is genuinely the Pragmatic value, not left at a "
              "stale 0 -- 'approximate with the pragmatic rate', per the spec, not just "
              "flagged and abandoned");
    }

    // ---- Parameter (escape-time) restored: quadratic still reproduces Mandelbrot ---
    // The Stage 4 multi-critical/escape-radius-free detour into render_
    // parameter was retired -- render_parameter is back to its pre-Stage-4
    // escape-time form (see its own header doc comment). This is the
    // property test standing in for that revert: Parameter is again an
    // escape_radius-governed VISUALIZATION, not an escape_radius-invariant
    // set-membership computation (that's Julia/Green's own job).
    std::printf("\nParameter (escape-time, restored): quadratic reproduces the Mandelbrot "
               "set, escape_radius genuinely changes the render:\n");
    {
        RationalMap mandel = RationalMap::mandelbrot();
        const Viewport v{{-0.5, 0.0}, 1.5, 81};
        Renderer r_r2(Map::custom(mandel), v, RenderSettings{80, 2.0, 1e-6, 1});
        Renderer r_r10(Map::custom(mandel), v, RenderSettings{80, 10.0, 1e-6, 1});
        const Image param_r2 = r_r2.render_parameter();
        const Image param_r10 = r_r10.render_parameter();

        // c=0 (deep in the set) has escape-time exactly 0 regardless of
        // escape_radius.
        const Cplx c_in{0.0, 0.0};
        Renderer r_probe(Map::custom(mandel), Viewport{c_in, 0.01, 1}, RenderSettings{80, 2.0, 1e-6, 1});
        check(r_probe.render_parameter().data[0] == 0.0,
              "c=0 (deep in the Mandelbrot set) has escape-time exactly 0");

        bool r2_has_nonzero = false, r10_has_nonzero = false;
        for (double v2 : param_r2.data) if (v2 > 0.0) r2_has_nonzero = true;
        for (double v10 : param_r10.data) if (v10 > 0.0) r10_has_nonzero = true;
        check(r2_has_nonzero && r10_has_nonzero,
              "sanity: this window has genuinely escaping (outside-the-set) pixels at "
              "both escape_radius settings");

        bool differs = false;
        for (std::size_t i = 0; i < param_r2.data.size(); ++i) {
            if (param_r2.data[i] != param_r10.data[i]) differs = true;
        }
        check(differs,
              "RESTORED BEHAVIOR: escape_radius=2 vs escape_radius=10 give DIFFERENT "
              "Parameter renders -- unlike Julia/Green's, this mode is a VISUALIZATION "
              "where escape_radius is a real tuning knob, not an invariant of the map");
    }

    // ---- Parameter (Stage 1): a rational family is sphere-aware, multi-critical, escape_radius-free --
    std::printf("\nParameter (Stage 1): a rational family shows real detail via the complete "
               "attractor backbone, with NO escape_radius dependence:\n");
    {
        // McMullen2 (z^2 + a/z^2): a built-in RATIONAL family (has a pole,
        // not certified) -- exactly the case whose Parameter render went
        // uniform under the EARLIER, reverted escape-to-infinity-only
        // attempt (every pixel ran to max_iter, since McMullen2's own
        // critical orbits don't need to reach INFINITY specifically to be
        // "interesting" -- they can settle into finite attracting cycles
        // that path had no way to report as anything but "unresolved").
        // Stage 1 fixes exactly that gap: classification is against the
        // FULL complete-attractor set (fixed point, cycle, OR infinity),
        // not just infinity alone.
        Renderer mc(Map(Family::McMullen2, Cplx(1.0, 0.0)), Viewport{{0.0, 0.0}, 1.0, 61},
                   RenderSettings{200, 2.0, 1e-6, 1});
        check(mc.map().escape_certified() == false,
              "sanity: McMullen2 is a built-in RATIONAL (non-certified) family");

        const Image mc_param = mc.render_parameter();
        bool mc_some_nonzero = false, mc_some_zero = false;
        for (double v : mc_param.data) { if (v > 0.0) mc_some_nonzero = true; else mc_some_zero = true; }
        check(mc_some_nonzero && mc_some_zero,
              "McMullen2's Parameter render has BOTH resolved (nonzero smooth rate) and "
              "unresolved (0) pixels -- genuine sphere-aware classification, not a uniform "
              "plane the way the earlier infinity-only attempt produced");

        bool mc_has_variety = false;
        double first_nonzero = 0.0;
        for (double v : mc_param.data) {
            if (v <= 0.0) continue;
            if (first_nonzero == 0.0) { first_nonzero = v; continue; }
            if (v != first_nonzero) { mc_has_variety = true; break; }
        }
        check(mc_has_variety,
              "...and the resolved pixels take on MORE THAN ONE smooth-rate value -- real "
              "gradient detail, not a flat resolved/unresolved binary");

        // escape_radius plays NO role in the rational path anymore -- see
        // render_parameter's own header doc comment. Unlike the certified-
        // polynomial case above (a legitimate visualization knob there),
        // this is now a genuine invariance property.
        Renderer mc_r10(Map(Family::McMullen2, Cplx(1.0, 0.0)), Viewport{{0.0, 0.0}, 1.0, 61},
                        RenderSettings{200, 10.0, 1e-6, 1});
        const Image mc_param_r10 = mc_r10.render_parameter();
        bool mc_differs = false;
        for (std::size_t i = 0; i < mc_param.data.size(); ++i) {
            if (mc_param.data[i] != mc_param_r10.data[i]) mc_differs = true;
        }
        check(!mc_differs,
              "RETIRES escape_radius for rational Parameter: escape_radius=2 vs "
              "escape_radius=10 give the IDENTICAL McMullen2 Parameter render");
    }

    // ---- Parameter (Stage 1): Nova (a genuinely multi-critical Custom map) --------
    std::printf("\nParameter (Stage 1): Nova (Custom, multi-critical) shows a smooth "
               "gradient, not the earlier attempt's flat/categorical plane:\n");
    {
        RationalMap nova("nova");
        nova.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
        nova.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
        nova.add_poly({1.0, 0.0}, 0, 1, "a");

        Renderer r(Map::custom(nova), Viewport{{0.0, 0.0}, 2.0, 61}, RenderSettings{150, 2.0, 1e-6, 1});
        check(r.map().escape_certified() == false, "sanity: Nova is a rational (non-certified) map");

        const Image nova_param = r.render_parameter();
        bool some_nonzero = false, some_zero = false;
        for (double v : nova_param.data) { if (v > 0.0) some_nonzero = true; else some_zero = true; }
        check(some_nonzero && some_zero,
              "Nova's Parameter render has both resolved and unresolved pixels");

        std::vector<double> distinct_vals;
        for (double v : nova_param.data) {
            if (v <= 0.0) continue;
            bool seen = false;
            for (double d : distinct_vals) if (d == v) { seen = true; break; }
            if (!seen) distinct_vals.push_back(v);
            if (distinct_vals.size() > 5) break;
        }
        check(distinct_vals.size() > 5,
              "Nova's resolved pixels take on a genuine SPREAD of smooth-rate values "
              "(not a small handful of categorical levels) -- the multi-critical "
              "Mandelbrot-glow gradient the earlier infinity-only attempt could not produce");
    }

    // ---- Parameter_basin: find_attractors_from_seeds' unresolved_count -------------
    // Pure-function test of the counting/clustering logic itself, no
    // rendering involved: z^2+c at c=0.2499 (just inside the main
    // cardioid's own cusp, c=0.25) has a genuinely, if weakly, attracting
    // finite fixed point (multiplier 0.98) -- MULTIPLIER-CONFIRMED
    // detection (this batch's own fix) resolves it correctly, where the
    // OLD closure-tolerance-only detection rejected it as unresolved
    // purely because it converged too slowly to close within the strict
    // budget (see find_attractors_from_seeds' own doc comment). c=0.25
    // EXACTLY, the cusp itself, is the genuinely parabolic case (multiplier
    // exactly 1) this confirmation path must still correctly reject, not
    // a false positive from Newton's own floating-point residual there
    // (see FindAttractorsOptions::attracting_margin's own doc comment).
    std::printf("\nParameter_basin: find_attractors_from_seeds' unresolved_count "
               "(pure-function test):\n");
    {
        RationalMap quad("quad");
        quad.add_poly({1.0, 0.0}, 2, 0, "z^2");
        quad.add_poly({1.0, 0.0}, 0, 1, "a");

        const Cplx c_inside{0.0, 0.0};
        int unresolved_inside = -1;
        const auto cyc_inside = find_attractors_from_seeds(quad.distinct_critical_points(c_inside),
                                                           quad, c_inside, {}, &unresolved_inside);
        check(cyc_inside.size() == 2 && unresolved_inside == 0,
              "c=0 (deep inside the cardioid): 2 distinct attracting cycles (the "
              "superattracting fixed point at 0, plus infinity), nothing unresolved");

        const Cplx c_weak{0.2499, 0.0};
        int unresolved_weak = -1;
        const auto cyc_weak = find_attractors_from_seeds(quad.distinct_critical_points(c_weak),
                                                         quad, c_weak, {}, &unresolved_weak);
        check(cyc_weak.size() == 2 && unresolved_weak == 0,
              "FIX: c=0.2499 (genuinely, weakly attracting -- multiplier ~0.98) now resolves "
              "to 2 distinct attracting cycles (the finite fixed point AND infinity), "
              "nothing unresolved -- multiplier confirmation caught what strict closure "
              "tolerance alone missed");
        check(cyc_weak[0].points.size() == 1,
              "...and the resolved cycle is genuinely period-1 (a single point), not a "
              "spurious multi-point 'cycle' from mistaking convergence time for period "
              "(the exact bug this fix's own Newton-polish re-anchoring avoids)");
        check(close(cyc_weak[0].points[0], Cplx{0.49, 0.0}, 1e-6),
              "...at the analytically correct fixed point z=0.49 (solving z^2-z+0.2499=0)");

        const Cplx c_true_cusp{0.25, 0.0};
        int unresolved_cusp = -1;
        const auto cyc_cusp = find_attractors_from_seeds(quad.distinct_critical_points(c_true_cusp),
                                                         quad, c_true_cusp, {}, &unresolved_cusp);
        check(cyc_cusp.size() == 1,
              "SAFETY: c=0.25 EXACTLY (the cusp itself, multiplier exactly 1 -- genuinely "
              "parabolic) still correctly finds only infinity, NOT a false-positive "
              "'attracting' cycle from Newton's own floating-point residual there");
        check(unresolved_cusp >= 1,
              "...and the parabolic critical orbit is honestly reported as unresolved, "
              "not silently dropped");

        const Cplx c_far{5.0, 5.0};
        int unresolved_far = -1;
        const auto cyc_far = find_attractors_from_seeds(quad.distinct_critical_points(c_far),
                                                        quad, c_far, {}, &unresolved_far);
        check(cyc_far.size() == 1 && unresolved_far == 0,
              "c=5+5i (far outside): both critical-point seeds (0 and infinity) converge to "
              "the SAME attractor (infinity) -- deduped to exactly 1 cycle, nothing "
              "unresolved (infinity genuinely IS confirmed attracting there)");
    }

    // ---- Parameter_basin: render, count == number of distinct attracting cycles ----
    std::printf("\nParameter_basin: render_parameter_basin counts distinct attracting "
               "cycles per pixel:\n");
    {
        RationalMap quad("quad");
        quad.add_poly({1.0, 0.0}, 2, 0, "z^2");
        quad.add_poly({1.0, 0.0}, 0, 1, "a");

        auto count_at = [&](Cplx c) {
            Renderer r(Map::custom(quad, c), Viewport{c, 0.001, 3}, RenderSettings{100, 2.0, 1e-6, 1});
            Image unresolved;
            const Image counts = r.render_parameter_basin(nullptr, &unresolved);
            return std::make_pair(counts.at(1, 1), unresolved.at(1, 1));
        };

        auto [count_inside, unresolved_inside2] = count_at(Cplx{0.0, 0.0});
        check(count_inside == 2.0 && unresolved_inside2 == 0.0,
              "RENDER matches the pure-function result at c=0: count=2, unresolved=0");

        auto [count_weak, unresolved_weak2] = count_at(Cplx{0.2499, 0.0});
        check(count_weak == 2.0 && unresolved_weak2 == 0.0,
              "RENDER matches the pure-function result at c=0.2499 (post-fix): count=2, "
              "unresolved=0");

        auto [count_cusp, unresolved_cusp2] = count_at(Cplx{0.25, 0.0});
        check(count_cusp == 1.0 && unresolved_cusp2 >= 1.0,
              "RENDER matches the pure-function result at the TRUE cusp c=0.25: count=1, "
              "unresolved>=1 -- count and unresolved are tracked as SEPARATE channels, "
              "never conflated, and the genuinely parabolic case still isn't a false positive");

        auto [count_far, unresolved_far2] = count_at(Cplx{5.0, 5.0});
        check(count_far == 1.0 && unresolved_far2 == 0.0,
              "RENDER matches the pure-function result far outside: count=1 (infinity "
              "only), unresolved=0");

        // A REAL render (not just three probed pixels) actually shows the
        // count changing across the image -- this is what makes "sharp
        // color edges at bifurcation boundaries" (the app's own coloring
        // requirement) possible at all.
        Renderer r_full(Map::custom(quad, Cplx{0.0, 0.0}), Viewport{{-0.5, 0.0}, 1.5, 81},
                        RenderSettings{100, 2.0, 1e-6, 1});
        const Image full = r_full.render_parameter_basin();
        bool has_count_1 = false, has_count_2 = false;
        for (double v : full.data) {
            if (v == 1.0) has_count_1 = true;
            if (v == 2.0) has_count_2 = true;
        }
        check(has_count_1 && has_count_2,
              "COUNT CHANGES ACROSS A KNOWN BIFURCATION: a real render spanning the "
              "Mandelbrot set's own cusp shows BOTH count=1 (exterior) and count=2 "
              "(interior) pixels, not a uniform plane");
    }

    // ---- Parameter_basin: 2+ coexisting FINITE attracting cycles -------------------
    std::printf("\nParameter_basin: a genuinely multi-critical family shows 2+ coexisting "
               "FINITE attracting cycles:\n");
    {
        // z^3 - 0.9z + a -- two independent finite critical points (+-
        // sqrt(0.3)), so (unlike z^2+a) its critical orbits can land on
        // TWO DIFFERENT finite attracting cycles simultaneously. `a` below
        // was found by an empirical scan (not hand-derived) and verified
        // directly against find_attractors before being hardcoded here;
        // small perturbations of it give the same count, so this is not a
        // knife-edge value.
        RationalMap cubic("cubic2");
        cubic.add_poly({1.0, 0.0}, 3, 0, "z^3");
        cubic.add_poly({-0.9, 0.0}, 1, 0, "-0.9z");
        cubic.add_poly({1.0, 0.0}, 0, 1, "a");

        const Cplx a_two_finite{-0.02986271158692677, -0.6105286348382948};
        const auto cycles = find_attractors(cubic, a_two_finite);
        int n_finite_cycles = 0;
        for (const auto& cyc : cycles) {
            bool all_finite = true;
            for (Cplx pt : cyc.points) if (!std::isfinite(pt.real())) all_finite = false;
            if (all_finite) ++n_finite_cycles;
        }
        check(n_finite_cycles >= 2,
              "sanity: find_attractors independently confirms >= 2 distinct FINITE "
              "attracting cycles at this parameter (two coexisting period-2 cycles)");

        Renderer r(Map::custom(cubic, a_two_finite), Viewport{a_two_finite, 0.001, 3},
                  RenderSettings{100, 2.0, 1e-6, 1});
        const Image counts = r.render_parameter_basin();
        check(counts.at(1, 1) == static_cast<double>(cycles.size()),
              "render_parameter_basin's own count matches find_attractors' cycle count "
              "exactly at this parameter (2 finite cycles + infinity = 3)");

        // A parameter far away collapses back to just infinity -- the
        // count genuinely varies with `a`, not a constant baked into the
        // family's own structure.
        Renderer r_far(Map::custom(cubic, Cplx{5.0, 5.0}), Viewport{{5.0, 5.0}, 0.001, 3},
                       RenderSettings{100, 2.0, 1e-6, 1});
        check(r_far.render_parameter_basin().at(1, 1) == 1.0,
              "...and a distant parameter for the SAME family collapses to count=1 "
              "(everything converges to infinity) -- a genuine bifurcation, not a "
              "fixed per-family constant");
    }

    // ---- Parameter_basin: a parameter-independent family gives a constant count ----
    std::printf("\nParameter_basin: Newton z^3-1 (no free parameter) gives a CONSTANT "
               "count everywhere:\n");
    {
        RationalMap newton3("newton3");
        newton3.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
        newton3.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
        check(polynomial_escape_certified(newton3) == false, "sanity: has a pole, not certified");

        const auto cycles = find_attractors(newton3, Cplx{0.0, 0.0});
        check(cycles.size() == 3, "sanity: find_attractors independently confirms 3 "
              "attracting cycles (the 3 cube roots of unity, each superattracting)");

        Renderer r(Map::custom(newton3), Viewport{{0.0, 0.0}, 2.0, 31},
                  RenderSettings{100, 2.0, 1e-6, 1});
        const Image counts = r.render_parameter_basin();
        bool all_three = true;
        for (double v : counts.data) if (v != 3.0) all_three = false;
        check(all_three,
              "every pixel reports count=3, regardless of the (ignored) parameter value -- "
              "newton_cubic has no term depending on `a` at all, so its dynamics -- and "
              "this count -- cannot vary across the plane");
    }

    // ---- Parameter_basin: requires a Custom-wrapped map -----------------------------
    std::printf("\nParameter_basin: a genuine built-in Family with no RationalMap behind "
               "it degrades honestly:\n");
    {
        Renderer r(Map(Family::Quadratic, Cplx{0.0, 0.0}), Viewport{{-0.5, 0.0}, 1.5, 21},
                  RenderSettings{80, 2.0, 1e-6, 1});
        check(r.map().custom_map() == nullptr,
              "sanity: a genuine Family::Quadratic Map has no RationalMap behind it");
        Image unresolved;
        const Image counts = r.render_parameter_basin(nullptr, &unresolved);
        bool all_zero = true, unresolved_all_zero = true;
        for (double v : counts.data) if (v != 0.0) all_zero = false;
        for (double v : unresolved.data) if (v != 0.0) unresolved_all_zero = false;
        check(all_zero && unresolved_all_zero,
              "degrades to an honest all-zero image (both channels) rather than crashing "
              "or guessing -- this path is never actually reached by the app itself, which "
              "always renders through Map::custom (see app/session.py's render_map)");
    }

    // ---- fact sheet self-consistency: fixed-points table vs attracting-cycles table ---
    // The SECOND symptom this fix addresses, distinct from Parameter_basin's
    // own artifacts: dynamical_facts() bundles RationalMap::fixed_points()
    // (an ALGEBRAIC root-find + exact deriv() evaluation, entirely
    // independent of any iterative discovery) alongside find_attractors()'
    // own attracting_cycles (previously closure-tolerance-only). Before
    // this fix, the SAME map at the SAME parameter could have its fixed-
    // points table correctly mark a point attracting (multiplier < 1)
    // while its attracting-cycles table omitted it entirely -- both
    // computed from ONE dynamical_facts() call, both about the exact same
    // point, disagreeing with each other.
    std::printf("\nfact sheet self-consistency: fixed-points table agrees with "
               "attracting-cycles table (the SAME map, SAME parameter):\n");
    {
        RationalMap quad("quad");
        quad.add_poly({1.0, 0.0}, 2, 0, "z^2");
        quad.add_poly({1.0, 0.0}, 0, 1, "a");
        const Cplx c{0.2499, 0.0};   // multiplier ~0.98 -- genuinely, weakly attracting
        const auto facts = dynamical_facts(quad, c);

        const FixedPoint* weak_fp = nullptr;
        for (const auto& fp : facts.fixed_points) {
            if (close(fp.point, Cplx{0.49, 0.0}, 1e-6)) { weak_fp = &fp; break; }
        }
        check(weak_fp != nullptr, "sanity: the algebraic fixed-points table finds z=0.49 "
              "at all (RationalMap::fixed_points, independent of any iteration)");
        check(weak_fp != nullptr && std::abs(weak_fp->multiplier) < 1.0,
              "ORACLE: the fixed-points table's own ALGEBRAIC multiplier correctly marks "
              "z=0.49 as attracting (|multiplier| < 1) -- this is the ground truth the "
              "attracting-cycles table must agree with");

        bool cycles_table_agrees = false;
        for (const auto& ac : facts.attracting_cycles) {
            for (Cplx pt : ac.points) {
                if (close(pt, Cplx{0.49, 0.0}, 1e-6)) cycles_table_agrees = true;
            }
        }
        check(cycles_table_agrees,
              "FIX: the SAME dynamical_facts() call's attracting-cycles table (fed by "
              "find_attractors, this batch's own fix) now agrees -- it includes z=0.49 "
              "too, where before this fix it silently omitted it despite the fixed-points "
              "table, computed in the SAME call, already knowing it was attracting");
    }

    // ---- IDENTICAL MATH (hard gate): interleaved settle-and-detect vs the OLD ------
    // fixed-budget two-phase algorithm -- a pure SPEED optimization
    // (cdx/src/analysis.cpp: exit cycle detection on convergence instead of
    // burning a fixed budget every seed) must never change WHAT is found,
    // only how fast. old_find_attractors_from_seeds above is a faithful,
    // frozen copy of the pre-optimization algorithm, kept specifically as
    // this regression's own oracle. Compared across a grid of parameters
    // spanning interior/exterior/boundary on Mandelbrot (z^2+a) and across
    // Nova (Newton's method for z^3-1, plus an additive parameter -- the
    // family the optimization was actually diagnosed and profiled against,
    // see cdx/test/diagnose_parameter_basin.cpp), using the SAME seeds and
    // options for both: same cycle count, same unresolved count, and for
    // every cycle a (period, |multiplier|) match on the other side -- exact
    // on period (an integer, and the SAME minimal period the SAME tolerance
    // must produce either way), close on |multiplier| (both algorithms
    // settle to within `tol`/`loose_tol` of the true cycle, just via
    // different step counts, so points differ only at the numerical-noise
    // level, not the multiplier itself).
    std::printf("\nIDENTICAL MATH (hard gate): interleaved settle-and-detect matches the OLD "
               "fixed-budget algorithm exactly, across a grid on Mandelbrot and Nova:\n");
    {
        RationalMap nova("nova");
        nova.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
        nova.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
        nova.add_poly({1.0, 0.0}, 0, 1, "a");

        RationalMap mandel = RationalMap::mandelbrot();

        auto compare_at = [&](const RationalMap& map, Cplx a, int& n_mismatches,
                              int& n_params_checked) {
            const auto seeds = map.distinct_critical_points(a);
            int unresolved_new = -1, unresolved_old = -1;
            const auto cycles_new = find_attractors_from_seeds(seeds, map, a, {}, &unresolved_new);
            const auto cycles_old =
                old_find_attractors_from_seeds(seeds, map, a, {}, &unresolved_old);
            ++n_params_checked;

            if (cycles_new.size() != cycles_old.size() || unresolved_new != unresolved_old) {
                ++n_mismatches;
                std::printf("    MISMATCH a=(%.6f%+.6fi): new count=%zu unresolved=%d | old count=%zu unresolved=%d\n",
                           a.real(), a.imag(), cycles_new.size(), unresolved_new, cycles_old.size(), unresolved_old);
                return;
            }
            std::vector<std::pair<int, double>> sig_new, sig_old;
            for (const auto& c : cycles_new) sig_new.push_back(old_cycle_signature(c, map, a));
            for (const auto& c : cycles_old) sig_old.push_back(old_cycle_signature(c, map, a));
            std::vector<bool> matched(sig_old.size(), false);
            for (const auto& [period, mult] : sig_new) {
                bool found_match = false;
                for (std::size_t j = 0; j < sig_old.size(); ++j) {
                    if (matched[j]) continue;
                    if (sig_old[j].first == period && std::abs(sig_old[j].second - mult) < 1e-6) {
                        matched[j] = true;
                        found_match = true;
                        break;
                    }
                }
                if (!found_match) {
                    ++n_mismatches;
                    std::printf("    MISMATCH a=(%.6f%+.6fi): no old match for new (period=%d, |mult|=%.9f)\n",
                               a.real(), a.imag(), period, mult);
                    return;
                }
            }
        };

        int n_mismatches = 0, n_checked = 0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                const double re = -2.0 + 2.5 * i / 8.0;    // -2.0 .. 0.5
                const double im = -1.2 + 2.4 * j / 8.0;    // -1.2 .. 1.2
                compare_at(mandel, Cplx{re, im}, n_mismatches, n_checked);
            }
        }
        const int mandel_checked = n_checked;
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                const double re = -0.6 + 0.4 * i / 6.0;    // -0.6 .. -0.2
                const double im = -0.15 + 0.3 * j / 6.0;   // -0.15 .. 0.15
                compare_at(nova, Cplx{re, im}, n_mismatches, n_checked);
            }
        }
        std::printf("  checked %d parameters (%d Mandelbrot interior/exterior/boundary, %d "
                   "Nova), %d mismatch%s\n", n_checked, mandel_checked, n_checked - mandel_checked,
                   n_mismatches, n_mismatches == 1 ? "" : "es");
        check(n_checked > 0, "sanity: the grid actually ran");
        check(n_mismatches == 0,
              "every parameter's cycle count, unresolved count, and per-cycle (period, "
              "|multiplier|) signature is IDENTICAL between the interleaved early-exit "
              "algorithm and the frozen pre-optimization reference -- the optimization "
              "changed WHEN closure is detected, never WHAT is found");
    }

    // =============================================================================
    // complete_attractors: find_attractors UNION algebraically-attracting
    // fixed points (cosmetic-batch follow-on: "every attracting fixed point
    // must appear as an attracting cycle, everywhere it's used").
    //
    // AUDIT NOTE on reproduction: an initial broad search (random
    // polynomials and multi-pole rational maps, degree 3-7; Newton/
    // McMullen/relaxed-Newton families; roughly 200,000 total (map,
    // parameter) evaluations, filtered to |multiplier| < 0.9) found no
    // naturally-occurring miss -- but that filter was itself too
    // conservative. cdx_diagnose_parameter_basin.cpp's OWN "before vs
    // after" sanity check (comparing render_parameter_basin's real output
    // against a direct find_attractors_from_seeds call) started failing
    // once render_parameter_basin was wired to complete_attractors_from_
    // seeds, and pinning down exactly why surfaced a REAL, reproducible
    // case on the exact Nova family/viewport that file already uses: at
    // a=(-0.345714,-0.100000), fixed_points() reports (0.74392,-0.0516)
    // as attracting (|multiplier|=0.9652 -- not superattracting, not so
    // close to 1 that it needs anything special), but find_attractors
    // (default options, confirm_weakly_attracting ON) mislabels it as a
    // SPURIOUS PERIOD-9 cycle of the identical point repeated nine times,
    // rather than period-1 -- hypothesis (c) from the audit (period-1
    // detection), root-caused to confirm_weakly_attracting's own final-
    // trial acceptance (analysis.cpp, the "bool accept = true" default
    // before the current_anchor_step < opts.extended_max_period gate):
    // at the LAST anchor trial, the k==1 candidate's own extra
    // confirmation step (chordal(f(zn),zn) < loose_tol) is bypassed
    // entirely, so if the orbit still hasn't fully settled by then (slow,
    // and evidently non-real/rotating multiplier), whichever k first
    // satisfies the plain chordal(zn,anchor) test gets accepted --
    // observed to be k=9 here, not k=1. This is a genuine, PRE-EXISTING
    // gap in find_attractors' own closure detection, out of scope to fix
    // in THIS batch (which reconciles against the algebraic ground truth
    // instead of patching the numerical search further), but it is
    // exactly the kind of case complete_attractors exists to catch
    // regardless of its numerical root cause. See test C below for the
    // pinned-down case itself, and cdx_diagnose_parameter_basin.cpp's own
    // updated "STAGE 2 VERIFICATION" section for where this was found.
    // Test E (infinity) still uses a deliberately-restricted seed list --
    // this real case doesn't happen to cover infinity, so that part is
    // demonstrated directly instead.
    // =============================================================================
    std::printf("\n\n=== complete_attractors ===\n");

    // ---- A: ground-truth invariant -- every algebraically-attracting fixed --------
    // point appears in complete_attractors' output, across a battery.
    std::printf("\nA. ground-truth invariant (fixed-points table vs complete_attractors, "
               "across a battery):\n");
    {
        auto check_battery = [](const char* label, RationalMap& map, Cplx a) {
            const auto fps = map.fixed_points(a);
            const auto cycles = complete_attractors(map, a);
            for (const auto& fp : fps) {
                if (!(std::abs(fp.multiplier) < 1.0)) continue;   // not attracting
                bool found = false;
                for (const auto& c : cycles) {
                    if (c.points.size() != 1) continue;
                    const bool fp_inf = !std::isfinite(fp.point.real());
                    const bool c_inf = !std::isfinite(c.points[0].real());
                    if (fp_inf != c_inf) continue;
                    if (fp_inf || close(fp.point, c.points[0], 1e-4)) { found = true; break; }
                }
                char msg[256];
                std::snprintf(msg, sizeof msg,
                             "%s: fixed point (%.4g%+.4gi) |mult|=%.4g is represented in "
                             "complete_attractors", label, fp.point.real(), fp.point.imag(),
                             std::abs(fp.multiplier));
                check(found, msg);
            }
        };

        RationalMap newton3 = RationalMap::newton_cubic();
        check_battery("newton_cubic", newton3, Cplx{0.0, 0.0});

        RationalMap mcm = RationalMap::mcmullen(3);
        for (double re = -1.0; re <= 1.0; re += 0.25)
            for (double im = -1.0; im <= 1.0; im += 0.25)
                check_battery("mcmullen(3)", mcm, Cplx{re, im});

        RationalMap nova("nova");
        nova.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
        nova.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
        nova.add_poly({1.0, 0.0}, 0, 1, "a");
        for (double re = -0.6; re <= -0.28; re += 0.04)
            for (double im = -0.15; im <= 0.15; im += 0.04)
                check_battery("nova", nova, Cplx{re, im});

        std::string err;
        RationalMap custom_map = RationalMap::from_expression(
            "z^3 + a/(z-1) + a/(z+1)", "a", {}, "custom_multi_pole");
        for (double re = -1.5; re <= 1.5; re += 0.3)
            for (double im = -1.5; im <= 1.5; im += 0.3)
                check_battery("custom(two poles)", custom_map, Cplx{re, im});
    }

    // ---- B: no double-counting -------------------------------------------------------
    std::printf("\nB. no double-counting:\n");
    {
        // newton_cubic: all 3 roots are ALREADY superattracting fixed points
        // find_attractors' own critical seeding finds directly (each root IS
        // one of its own critical points) -- the union must not add a SECOND
        // copy of any of them.
        RationalMap newton3 = RationalMap::newton_cubic();
        const auto before = find_attractors(newton3, Cplx{0.0, 0.0});
        const auto after = complete_attractors(newton3, Cplx{0.0, 0.0});
        check(before.size() == 3, "sanity: find_attractors alone already finds all 3 roots");
        check(after.size() == before.size(),
              "complete_attractors adds NOTHING on a map find_attractors already covers "
              "completely -- no inflation on an already-correct map");

        // Explicit duplicate-representation check: force the union to
        // consider a fixed point find_attractors ALREADY found (via a
        // full, unrestricted seed list) -- it must still end up
        // represented exactly once.
        for (const FixedPoint& fp : newton3.fixed_points(Cplx{0.0, 0.0})) {
            if (!(std::abs(fp.multiplier) < 1.0)) continue;
            int n_matches = 0;
            for (const auto& c : after) {
                if (c.points.size() != 1) continue;
                const bool fp_inf = !std::isfinite(fp.point.real());
                const bool c_inf = !std::isfinite(c.points[0].real());
                if (fp_inf != c_inf) continue;
                if (fp_inf || close(fp.point, c.points[0], 1e-4)) ++n_matches;
            }
            char msg[256];
            std::snprintf(msg, sizeof msg,
                         "fixed point (%.4g%+.4gi) is represented EXACTLY once, not "
                         "duplicated", fp.point.real(), fp.point.imag());
            check(n_matches == 1, msg);
        }
    }

    // ---- C: the REAL reproduced case -- Nova, a=(-0.345714,-0.100000) --------------
    // find_attractors (default options) mislabels the algebraically-
    // attracting fixed point (0.74392,-0.0516) as a spurious period-9
    // cycle -- see the AUDIT NOTE above for how this was found and root-
    // caused. complete_attractors recovers it correctly.
    std::printf("\nC. the REAL reproduced case (Nova, a=(-0.345714,-0.100000)) -- "
               "find_attractors mislabels a genuine period-1 fixed point as a spurious "
               "period-9 cycle:\n");
    {
        RationalMap nova("nova_repro");
        nova.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
        nova.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
        nova.add_poly({1.0, 0.0}, 0, 1, "a");
        const Cplx a{-0.345714, -0.100000};
        const Cplx target{0.74392, -0.0516219};

        const auto fps = nova.fixed_points(a);
        const FixedPoint* target_fp = nullptr;
        for (const auto& fp : fps) {
            if (close(fp.point, target, 1e-3)) { target_fp = &fp; break; }
        }
        check(target_fp != nullptr && std::abs(target_fp->multiplier) < 0.9701 &&
              std::abs(target_fp->multiplier) > 0.9,
              "sanity: fixed_points() reports this point attracting with |mult|~0.965 -- "
              "genuinely attracting, not superattracting, not pathologically close to 1");

        const auto broken_cycles = find_attractors(nova, a);   // default opts -- production
        bool mislabeled_as_period9 = false;
        for (const auto& c : broken_cycles) {
            if (c.points.size() == 9 && close(c.points[0], target, 1e-3)) mislabeled_as_period9 = true;
        }
        check(mislabeled_as_period9,
              "REPRODUCED: find_attractors' own default-options output labels this point's "
              "orbit as a period-9 cycle of the identical point repeated 9 times, not the "
              "period-1 fixed point it actually is");
        bool correctly_period1_in_broken = false;
        for (const auto& c : broken_cycles) {
            if (c.points.size() == 1 && close(c.points[0], target, 1e-3)) correctly_period1_in_broken = true;
        }
        check(!correctly_period1_in_broken,
              "...and find_attractors' own output has NO separate, correctly-labeled "
              "period-1 entry for it either -- this fixed point is genuinely unrepresented "
              "as what it actually is, not just duplicated");

        const auto fixed_cycles = complete_attractors(nova, a);
        bool correctly_period1_in_fixed = false;
        for (const auto& c : fixed_cycles) {
            if (c.points.size() == 1 && close(c.points[0], target, 1e-3)) correctly_period1_in_fixed = true;
        }
        check(correctly_period1_in_fixed,
              "FIX: complete_attractors' output includes this point as its own correct "
              "period-1 cycle -- recovered from fixed_points() regardless of find_"
              "attractors' own period-9 misdetection");

        // Impact: a pixel AT this point resolves to a real basin with the
        // complete set; with the broken set it either stays unresolved or
        // (since the spurious period-9 "cycle" IS technically present,
        // just mislabeled) can only ever match via that wrong period,
        // never as the clean, correct basin the point actually belongs to.
        RenderSettings rset{50, 2.0, 1e-6, 1};
        Viewport v{target, 0.01, 3};
        Renderer r_broken(Map::custom(nova, a), v, rset);
        Renderer r_fixed(Map::custom(nova, a), v, rset);
        const Image labels_broken = r_broken.render_basin(broken_cycles);
        const Image labels_fixed = r_fixed.render_basin(fixed_cycles);
        const int center = v.resolution / 2;
        std::printf("  BEFORE (find_attractors): %zu cycles total, pixel-at-point label=%.0f\n",
                   broken_cycles.size(), labels_broken.at(center, center));
        std::printf("  AFTER  (complete_attractors): %zu cycles total, pixel-at-point label=%.0f\n",
                   fixed_cycles.size(), labels_fixed.at(center, center));
        check(labels_fixed.at(center, center) != 0.0,
              "basin-mode region impact: the pixel at this fixed point resolves to a real "
              "attractor id with the complete set");
    }

    // ---- E: infinity, via a deliberately-restricted seed list -----------------------
    // (the real Nova case above doesn't happen to cover infinity, so this
    // demonstrates the union mechanism directly, the same way as before).
    std::printf("\nE. infinity as the attracting fixed point (simulated via an "
               "EMPTY seed list) -- impact on basin classification and Parameter_basin's "
               "count, plus an infinity case:\n");
    {
        // z^2 + a at a=-0.5: two finite fixed points, z=-0.366... genuinely
        // (not superattracting, not near-parabolic) attracting (|mult|~0.73),
        // z=1.366... repelling. An EMPTY seed list stands in for "critical
        // seeding found/used nothing" -- the most extreme, unambiguous case
        // of hypothesis (a) (incomplete critical-point enumeration) from the
        // audit, and it exercises the EXACT SAME complete_attractors_from_
        // seeds entry point render_parameter_basin calls in production.
        RationalMap quad("quad_demo");
        quad.add_poly({1.0, 0.0}, 2, 0, "z^2");
        quad.add_poly({1.0, 0.0}, 0, 1, "a");
        const Cplx a{-0.5, 0.0};

        const auto fps = quad.fixed_points(a);
        const FixedPoint* attracting_fp = nullptr;
        for (const auto& fp : fps) {
            if (std::abs(fp.multiplier) < 0.9) { attracting_fp = &fp; break; }
        }
        check(attracting_fp != nullptr,
              "sanity: z^2-0.5 has a genuinely (not weakly) attracting finite fixed point");

        int unresolved_before = -1, unresolved_after = -1;
        const auto cycles_before =
            find_attractors_from_seeds({}, quad, a, {}, &unresolved_before);
        const auto cycles_after =
            complete_attractors_from_seeds({}, quad, a, {}, &unresolved_after);
        check(cycles_before.empty(),
              "BEFORE: an empty seed list means find_attractors_from_seeds finds nothing "
              "at all -- the simulated miss");
        // z^2+a is a bare polynomial (no poles), so infinity is trivially
        // superattracting there TOO (diff=2, multiplier=0) -- the union
        // correctly recovers BOTH it and the finite attracting fixed point,
        // not just one, since both satisfy the same |multiplier|<1 test.
        bool has_finite = false, has_infinity = false;
        for (const auto& c : cycles_after) {
            if (c.points.size() != 1) continue;
            if (!std::isfinite(c.points[0].real())) has_infinity = true;
            else if (close(c.points[0], attracting_fp->point)) has_finite = true;
        }
        check(cycles_after.size() == 2 && has_finite && has_infinity,
              "AFTER: complete_attractors_from_seeds recovers BOTH the finite attracting "
              "fixed point AND infinity (also trivially superattracting for this bare "
              "polynomial) from fixed_points() alone, even with zero critical seeds");
        check(unresolved_before == 0 && unresolved_after == 0,
              "unresolved_count is unaffected by the union either way (0 seeds attempted, "
              "0 seeds unresolved -- the union recovers MISSING attractors, it does not "
              "reinterpret a genuinely failed seed as resolved)");

        // Impact on basin classification: a pixel AT the attracting fixed
        // point itself must resolve immediately (n=0) to a real cycle id
        // with the complete set, vs staying unresolved (label 0) forever
        // with the incomplete one.
        RenderSettings rset{50, 2.0, 1e-6, 1};
        Viewport v{attracting_fp->point, 0.01, 3};   // tiny viewport centred ON the fixed point
        Renderer r_before(Map::custom(quad, a), v, rset);
        Renderer r_after(Map::custom(quad, a), v, rset);
        const Image labels_before = r_before.render_basin(cycles_before);
        const Image labels_after = r_after.render_basin(cycles_after);
        const int center = v.resolution / 2;
        check(labels_before.at(center, center) == 0.0,
              "BEFORE (incomplete cycles): the pixel AT the attracting fixed point is "
              "unresolved -- render_basin has no attractor to classify it against");
        check(labels_after.at(center, center) != 0.0,
              "AFTER (complete cycles): the SAME pixel now correctly resolves to the "
              "recovered attractor's basin -- this is the basin-mode region the fix claims");

        // Impact on Parameter_basin's count, at this exact parameter pixel:
        // before=0 distinct attracting cycles (nothing seeded), after=2
        // (the finite fixed point AND infinity, both recovered) -- the
        // count-region impact this synthetic construction demonstrates.
        // (A REAL render_parameter_basin pixel for z^2+a has only ONE
        // finite critical orbit to seed from, so it would show count=1
        // either way -- escaping to infinity XOR converging to the finite
        // point, never counting both from a single orbit; this empty-seed
        // simulation exposes BOTH simultaneously because fixed_points()
        // reports every algebraically attracting point regardless of any
        // orbit's actual fate, which is exactly the point of a pure
        // completeness union.)
        check(static_cast<int>(cycles_before.size()) == 0 &&
              static_cast<int>(cycles_after.size()) == 2,
              "Parameter_basin count-region impact at this parameter: 0 -> 2 distinct "
              "attracting cycles (BEFORE undercounts by exactly the recovered fixed point)");

        // ---- E: infinity as the attracting fixed point -----------------------------
        // P(z)=2z^2-2z+a, Q(z)=z-1 -- degree(P)-degree(Q)=1, so infinity is
        // FIXED (not itself critical, since diff<2) with multiplier
        // Q_lead/P_lead = 1/2, clearly attracting and not near 1.
        std::string err;
        RationalMap inf_map = RationalMap::from_expression(
            "(2*z^2 - 2*z + a) / (z - 1)", "a", {}, "inf_attracting");
        const Cplx a_inf{0.3, 0.0};
        const auto inf_fps = inf_map.fixed_points(a_inf);
        const FixedPoint* inf_fp = nullptr;
        for (const auto& fp : inf_fps) {
            if (!std::isfinite(fp.point.real()) && std::abs(fp.multiplier) < 0.9) {
                inf_fp = &fp;
                break;
            }
        }
        check(inf_fp != nullptr,
              "sanity: this map's fixed_points() reports infinity as clearly attracting "
              "(|mult| ~ 0.5), not near 1");

        const auto inf_cycles_before = find_attractors_from_seeds({}, inf_map, a_inf);
        const auto inf_cycles_after = complete_attractors_from_seeds({}, inf_map, a_inf);
        check(inf_cycles_before.empty(),
              "BEFORE: with nothing seeded, infinity's own attraction is never checked at "
              "all -- the simulated miss applies to infinity exactly as to any finite point");
        check(inf_cycles_after.size() == 1 && !std::isfinite(inf_cycles_after[0].points[0].real()),
              "AFTER: complete_attractors_from_seeds counts infinity as its own period-1 "
              "{Inf} cycle, recovered from fixed_points() alone");
    }

    // ---- D: no regression -- an already-complete map renders identically ------------
    std::printf("\nD. no regression: an already-complete map's basin render is unchanged:\n");
    {
        RationalMap newton3 = RationalMap::newton_cubic();
        const Cplx a{0.0, 0.0};
        const auto old_cycles = find_attractors(newton3, a);
        const auto new_cycles = complete_attractors(newton3, a);
        check(old_cycles.size() == new_cycles.size(),
              "newton_cubic (no missed attractors) has the SAME cycle count before/after");

        RenderSettings rset{50, 2.0, 1e-6, 1};
        Viewport v{{0.0, 0.0}, 2.0, 41};
        Renderer r_old(Map::custom(newton3, a), v, rset);
        Renderer r_new(Map::custom(newton3, a), v, rset);
        const Image labels_old = r_old.render_basin(old_cycles);
        const Image labels_new = r_new.render_basin(new_cycles);
        bool identical = labels_old.data.size() == labels_new.data.size();
        for (std::size_t i = 0; identical && i < labels_old.data.size(); ++i) {
            if (labels_old.data[i] != labels_new.data[i]) identical = false;
        }
        check(identical,
              "render_basin's output is BYTE-IDENTICAL between find_attractors' own cycles "
              "and complete_attractors' cycles, for a map with no missed attractors -- the "
              "union changes nothing when there is nothing to add");
    }

    // =============================================================================
    // Parameter_basin unresolved-count reconciliation (cosmetic-batch follow-on:
    // "reconcile the Parameter_basin unresolved count with the injected
    // attractors, so color_parameter_basin stops painting confirmed-attractor
    // regions as unresolved"). unresolved_count must mean "critical orbits
    // explained by NO attractor in the COMPLETE set", not just the critical-
    // seeded pass's own raw tally -- see complete_attractors_from_seeds' own
    // doc comment.
    // =============================================================================
    std::printf("\n\n=== Parameter_basin unresolved-count reconciliation ===\n");

    // ---- reconciliation: an unresolved orbit heading toward an injected fixed point ---
    std::printf("\nreconciliation: an unresolved critical orbit heading toward an injected "
               "fixed point is re-attributed as resolved:\n");
    {
        RationalMap quad("quad_recon");
        quad.add_poly({1.0, 0.0}, 2, 0, "z^2");
        quad.add_poly({1.0, 0.0}, 0, 1, "a");
        const Cplx a{-0.5, 0.0};

        // Artificially tiny closure budget (max_period=0, weak-confirm off) --
        // NOT find_attractors' own default settings -- forces the strict pass
        // to genuinely fail to detect closure even though burn_in=45 is more
        // than enough for this orbit to have numerically converged extremely
        // close to the true fixed point (|mult|=0.732, so ~0.732^45 residual)
        // by the time it gives up. This isolates the RECONCILIATION mechanism
        // itself, independent of confirm_weakly_attracting's own (separate,
        // already-tested) extended search.
        FindAttractorsOptions opts;
        opts.burn_in = 45;
        opts.max_period = 0;
        opts.confirm_weakly_attracting = false;

        int unresolved_before = -1;
        const auto before = find_attractors_from_seeds({Cplx(0.0, 0.0)}, quad, a, opts,
                                                        &unresolved_before);
        check(before.empty() && unresolved_before == 1,
              "sanity: with this artificially tiny budget, find_attractors_from_seeds alone "
              "genuinely fails to close -- 0 cycles, 1 unresolved seed");

        int unresolved_after = -1, certain_after = -1;
        const auto after = complete_attractors_from_seeds({Cplx(0.0, 0.0)}, quad, a, opts,
                                                           &unresolved_after, &certain_after);
        check(after.size() == 2, "AFTER: the union still injects both algebraically-"
              "attracting fixed points (z=-0.366... and infinity) regardless of the tiny "
              "budget");
        check(certain_after == 2, "...both are CERTAIN (injected, not numerically discovered)");
        check(unresolved_after == 0,
              "FIX: unresolved_count drops from 1 to 0 -- the seed's own orbit was actually "
              "heading toward the injected fixed point (its final live point sits within "
              "the reconciliation tolerance of it), so it is re-attributed as resolved, not "
              "left as a phantom residual");
        check(unresolved_before - unresolved_after == 1,
              "the drop is exactly the one re-attributed orbit, not more (nothing else to "
              "reconcile here)");

        // NO REGRESSION: reconciliation touches unresolved_count ONLY -- the
        // returned cycle set itself is byte-for-byte the same whether or not
        // a caller even asks for unresolved_count at all.
        const auto no_unresolved_tracking =
            complete_attractors_from_seeds({Cplx(0.0, 0.0)}, quad, a, opts);
        check(no_unresolved_tracking.size() == after.size(),
              "NO REGRESSION: the cycle count is identical whether or not unresolved_count "
              "is tracked -- reconciliation only moves the resolved/unresolved boundary, it "
              "never adds or removes an attractor");
    }

    // ---- genuine unresolved: a parabolic fixed point is NEVER reconciled away ---------
    std::printf("\ngenuine unresolved: a parabolic (non-attracting) fixed point stays "
               "unresolved, not swept away by the reconciliation:\n");
    {
        RationalMap quad("quad_parabolic");
        quad.add_poly({1.0, 0.0}, 2, 0, "z^2");
        quad.add_poly({1.0, 0.0}, 0, 1, "a");
        const Cplx a{0.25, 0.0};   // cusp of the main cardioid: z=0.5, multiplier EXACTLY 1

        const auto fps = quad.fixed_points(a);
        const FixedPoint* parabolic_fp = nullptr;
        for (const auto& fp : fps) {
            if (close(fp.point, Cplx{0.5, 0.0}, 1e-6)) { parabolic_fp = &fp; break; }
        }
        check(parabolic_fp != nullptr && std::abs(parabolic_fp->multiplier - 1.0) < 1e-6,
              "sanity: z=0.5 is fixed with multiplier 1 (parabolic, to numerical-root-find "
              "precision) -- genuinely NOT "
              "attracting (|mult| < 1 fails), so the union must never inject it");

        // find_attractors' own DEFAULT budget -- no artificial restriction
        // needed here; a parabolic orbit's algebraic (not geometric)
        // convergence rate genuinely never closes within it.
        const FindAttractorsOptions opts;
        int unresolved_before = -1;
        const auto before = find_attractors_from_seeds({Cplx(0.0, 0.0)}, quad, a, opts,
                                                        &unresolved_before);
        check(unresolved_before == 1,
              "sanity: the critical orbit toward the parabolic point genuinely does not "
              "close within find_attractors' own default budget");

        int unresolved_after = -1, certain_after = -1;
        const auto after = complete_attractors_from_seeds({Cplx(0.0, 0.0)}, quad, a, opts,
                                                           &unresolved_after, &certain_after);
        check(after.size() == 1 && certain_after == 1,
              "infinity IS injected here too (trivially superattracting for this bare "
              "polynomial) -- but that attractor is unrelated to the still-unresolved "
              "parabolic orbit");
        bool parabolic_injected = false;
        for (const auto& c : after) {
            if (c.points.size() == 1 && close(c.points[0], Cplx{0.5, 0.0}, 1e-4)) {
                parabolic_injected = true;
            }
        }
        check(!parabolic_injected,
              "the parabolic fixed point is NEVER injected -- union_in_attracting_fixed_"
              "points' own |multiplier| < 1.0 gate correctly excludes it");
        check(unresolved_after == unresolved_before,
              "GENUINE UNRESOLVED, unaffected: unresolved_count is UNCHANGED by the "
              "reconciliation -- this orbit's endpoint matches no complete attractor (the "
              "parabolic point was never injected, and the orbit wasn't heading toward "
              "infinity either), so it correctly remains a residual, not silently swept "
              "away");
    }

    // =============================================================================
    // Period coloring: per_seed_outcomes and Renderer::render_parameter_period
    // (side-by-side comparison with Parameter_basin's count coloring, matching
    // the Lindsey-Koch-Sharland bicritical-maps literature's own convention).
    // Family under test: relaxed Newton of p(z) = z^n - 1, N_a(z) = z -
    // a(z^n-1)/(n z^{n-1}) = ((n-a)z^n + a) / (n z^{n-1}), n = 6.
    // =============================================================================
    std::printf("\n\n=== period coloring (per_seed_outcomes, render_parameter_period) ===\n");

    auto relaxed_newton_power = [](int n) {
        std::string src = "((" + std::to_string(n) + "-a)*z^" + std::to_string(n) + " + a) / (" +
                          std::to_string(n) + "*z^" + std::to_string(n - 1) + ")";
        return RationalMap::from_expression(src, "a", {}, "relaxed_newton");
    };
    auto free_critical_points = [](int n, Cplx a) {
        // z_c(a) = (a(n-1)/(n-a))^(1/n) * omega^j, j=0..n-1 -- see
        // CriticalPointFamily::RelaxedNewtonPower's own C++ doc comment.
        const Cplx omega(std::cos(2 * M_PI / n), std::sin(2 * M_PI / n));
        const Cplx ratio = a * static_cast<double>(n - 1) / (static_cast<double>(n) - a);
        const Cplx base = std::pow(ratio, 1.0 / static_cast<double>(n));
        std::vector<Cplx> seeds;
        seeds.reserve(static_cast<std::size_t>(n));
        Cplx w(1.0, 0.0);
        for (int j = 0; j < n; ++j) { seeds.push_back(base * w); w *= omega; }
        return seeds;
    };
    auto gcd_fn = [](long x, long y) { while (y) { long t = y; y = x % y; x = t; } return x; };

    const int n = 6;
    RationalMap nova_n = relaxed_newton_power(n);

    // ---- per_seed_outcomes: a seed's own outcome, not the union's --------------------
    std::printf("\nper_seed_outcomes: reports what THIS seed converged to, not the "
               "deduplicated attractor set:\n");
    {
        const Cplx a{1.3, 0.2};   // inside |a-1|<1
        const auto z0_outcome = per_seed_outcomes({Cplx(0.0, 0.0)}, nova_n, a, {});
        check(z0_outcome.size() == 1 && !z0_outcome[0].resolved,
              "z=0 seeded ALONE at a=1.3+0.2i is genuinely undetermined -- it does NOT "
              "inherit the 6 roots-of-unity fixed points the way complete_attractors_"
              "from_seeds' own algebraic union would (measured directly: seeding that "
              "function with z=0 alone still returns all 6, via the union, none of them "
              "what z=0's own orbit actually did)");

        const auto free_outcomes = per_seed_outcomes(free_critical_points(n, a), nova_n, a, {});
        check(free_outcomes.size() == 6 &&
              std::all_of(free_outcomes.begin(), free_outcomes.end(),
                         [](const SeedOutcome& o) { return o.resolved && o.period == 1 &&
                                                            !o.is_infinity; }),
              "all 6 free critical points independently resolve to period 1 inside "
              "|a-1|<1 -- each seed's OWN outcome, not a shared/injected one");
    }

    // ---- VALIDATION A: disc |a-1| < 1 is uniformly period 1 --------------------------
    std::printf("\nVALIDATION A: |a-1| < 1 is uniformly period 1:\n");
    {
        Viewport v{{1.0, 0.0}, 0.6, 15};
        Renderer r(Map::custom(nova_n, Cplx(0, 0)), v, RenderSettings{200, 2.0, 1e-6, 1});
        Image undetermined, is_inf;
        Image periods = r.render_parameter_period(CriticalPointFamily::RelaxedNewtonPower, n,
                                                   nullptr, &undetermined, &is_inf);
        int n_checked = 0, n_period1 = 0;
        for (int row = 0; row < v.resolution; ++row) {
            for (int col = 0; col < v.resolution; ++col) {
                const Cplx a = v.coord(col, row);
                if (std::abs(a - Cplx(1.0, 0.0)) >= 0.98) continue;   // strictly inside, with margin
                ++n_checked;
                if (periods.at(col, row) == 1.0 && undetermined.at(col, row) == 0.0 &&
                    is_inf.at(col, row) == 0.0) {
                    ++n_period1;
                }
            }
        }
        check(n_checked > 0, "sanity: the grid actually samples points strictly inside |a-1|<1");
        check(n_period1 == n_checked,
              "every sampled point strictly inside |a-1| < 1 is period 1, none "
              "undetermined or infinity");
    }

    // ---- VALIDATION B: superattracting centers a_k = n - (n-1)*omega_k --------------
    std::printf("\nVALIDATION B: superattracting centers a_k = n - (n-1)*omega_k, "
               "period = n/gcd(k,n):\n");
    for (int k = 0; k < n; ++k) {
        const Cplx omega_k(std::cos(2 * M_PI * k / n), std::sin(2 * M_PI * k / n));
        const Cplx a_k = static_cast<double>(n) - static_cast<double>(n - 1) * omega_k;
        const long expected_period = n / gcd_fn(k == 0 ? n : k, n);   // k=0 -> period 1

        Viewport v{a_k, 0.001, 3};
        Renderer r(Map::custom(nova_n, Cplx(0, 0)), v, RenderSettings{200, 2.0, 1e-6, 1});
        Image undetermined, is_inf;
        Image periods = r.render_parameter_period(CriticalPointFamily::RelaxedNewtonPower, n,
                                                   nullptr, &undetermined, &is_inf);
        const int center = v.resolution / 2;
        char msg[192];
        std::snprintf(msg, sizeof msg,
                     "k=%d: a_k=(%.6f,%.6f) reports period %.0f (expected %ld), "
                     "undetermined=%.0f, is_inf=%.0f", k, a_k.real(), a_k.imag(),
                     periods.at(center, center), expected_period, undetermined.at(center, center),
                     is_inf.at(center, center));
        check(periods.at(center, center) == static_cast<double>(expected_period) &&
              undetermined.at(center, center) == 0.0 && is_inf.at(center, center) == 0.0, msg);
    }

    // ---- VALIDATION C: exterior |a-n| > n is the infinity class ----------------------
    std::printf("\nVALIDATION C: |a-n| > n is the infinity class:\n");
    {
        Viewport v{{3.0 * n, 0.0}, 0.001, 3};
        Renderer r(Map::custom(nova_n, Cplx(0, 0)), v, RenderSettings{200, 2.0, 1e-6, 1});
        Image undetermined, is_inf;
        Image periods = r.render_parameter_period(CriticalPointFamily::RelaxedNewtonPower, n,
                                                   nullptr, &undetermined, &is_inf);
        const int center = v.resolution / 2;
        check(is_inf.at(center, center) == 1.0 && undetermined.at(center, center) == 0.0,
              "a well outside |a-n|=n reports the infinity class, not undetermined");
    }

    // ---- NO REGRESSION: a map with no RationalMap behind it degrades honestly -------
    std::printf("\nno RationalMap behind the map: degrades to all-undetermined, never a "
               "fabricated period:\n");
    {
        Renderer r(Map(Family::Quadratic, Cplx(0, 0)), Viewport{{0, 0}, 1.0, 5},
                  RenderSettings{100, 2.0, 1e-6, 1});
        Image undetermined;
        Image periods = r.render_parameter_period(CriticalPointFamily::RelaxedNewtonPower, n,
                                                   nullptr, &undetermined);
        bool all_undetermined = true, all_zero_period = true;
        for (double v : undetermined.data) if (v != 1.0) all_undetermined = false;
        for (double v : periods.data) if (v != 0.0) all_zero_period = false;
        check(all_undetermined && all_zero_period,
              "a built-in Family with no RationalMap behind it (never actually reached by "
              "the app, which always renders through Map::custom) is honestly all-"
              "undetermined, not a guess");
    }

    // ---- Stage 2 (spatial continuation): pixel-identical vs cold-only, plus timing ----
    std::printf("\nStage 2 (spatial continuation): Parameter_basin's continued counts are "
               "PIXEL-IDENTICAL to a cold-only reference, on Nova:\n");
    {
        RationalMap nova("nova");
        nova.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
        nova.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
        nova.add_poly({1.0, 0.0}, 0, 1, "a");

        const Viewport v{{0.0, 0.0}, 3.0, 121};
        const RenderSettings settings{150, 2.0, 1e-6, 1};
        Renderer r(Map::custom(nova), v, settings);

        Image unresolved, certain;
        const Image continued = r.render_parameter_basin(nullptr, &unresolved, &certain);

        // Cold-only reference: the EXACT pre-Stage-2 per-pixel algorithm --
        // distinct_critical_points(p) and complete_attractors_from_seeds
        // with no fp_predictor -- reproduced independently here rather than
        // by disabling continuation inside the renderer, so this is a
        // genuine second implementation, not the same code timed twice.
        const FindAttractorsOptions opts;
        Image cold_counts(v.resolution, v.resolution);
        Image cold_unresolved(v.resolution, v.resolution);
        Image cold_certain(v.resolution, v.resolution);
        const auto t_cold_start = std::chrono::steady_clock::now();
        for (int col = 0; col < v.resolution; ++col) {
            for (int row = 0; row < v.resolution; ++row) {
                const Cplx p = v.coord(col, row);
                const std::vector<Cplx> crit_pts = nova.distinct_critical_points(p);
                int unresolved_n = 0, certain_n = 0;
                const std::vector<Cycle> cycles = complete_attractors_from_seeds(
                    crit_pts, nova, p, opts, &unresolved_n, &certain_n);
                cold_counts.at(col, row) = static_cast<double>(cycles.size());
                cold_unresolved.at(col, row) = static_cast<double>(unresolved_n);
                cold_certain.at(col, row) = static_cast<double>(certain_n);
            }
        }
        const auto t_cold_end = std::chrono::steady_clock::now();

        bool identical = true;
        for (std::size_t i = 0; i < continued.data.size(); ++i) {
            if (continued.data[i] != cold_counts.data[i] ||
                unresolved.data[i] != cold_unresolved.data[i] ||
                certain.data[i] != cold_certain.data[i]) {
                identical = false;
                break;
            }
        }
        check(identical,
              "render_parameter_basin's count/unresolved/certain images are BYTE-IDENTICAL "
              "to the independently-computed cold-only reference -- continuation changed "
              "nothing about the answer, only how it was obtained");

        // Timing: render_parameter_basin's own (continuation-enabled) time vs the
        // cold-only reference loop above, same map/viewport/settings/work.
        const auto t_warm_start = std::chrono::steady_clock::now();
        r.render_parameter_basin(nullptr, nullptr, nullptr);
        const auto t_warm_end = std::chrono::steady_clock::now();

        const double cold_ms =
            std::chrono::duration<double, std::milli>(t_cold_end - t_cold_start).count();
        const double warm_ms =
            std::chrono::duration<double, std::milli>(t_warm_end - t_warm_start).count();
        std::printf("  TIMING (Nova, %dx%d Parameter_basin): cold-only=%.1fms  "
                   "continuation-enabled=%.1fms  (%.2fx)\n",
                   v.resolution, v.resolution, cold_ms, warm_ms,
                   warm_ms > 0.0 ? cold_ms / warm_ms : 0.0);
    }

    std::printf("\nStage 2: render_parameter's rational path also renders successfully with "
               "continuation engaged (Nova), still shows real detail:\n");
    {
        RationalMap nova("nova");
        nova.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
        nova.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
        nova.add_poly({1.0, 0.0}, 0, 1, "a");
        Renderer r(Map::custom(nova), Viewport{{0.0, 0.0}, 2.0, 121}, RenderSettings{150, 2.0, 1e-6, 1});
        const Image img = r.render_parameter();
        bool some_nonzero = false, some_zero = false;
        for (double v : img.data) { if (v > 0.0) some_nonzero = true; else some_zero = true; }
        check(some_nonzero && some_zero,
              "Nova's Parameter render (continuation-enabled) still has both resolved and "
              "unresolved pixels -- Stage 1's own sphere-aware detail is preserved");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
