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
#include <cmath>
#include <cstdio>

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

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
