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

    // ---- Parameter (escape-time, restored): a rational family renders quickly again --
    std::printf("\nParameter (escape-time, restored): a rational family is fast again "
               "(early exit on escape), not uniform:\n");
    {
        // McMullen2 (z^2 + a/z^2): a built-in RATIONAL family (has a pole,
        // not certified) -- exactly the case whose Parameter render went
        // uniform under the retired Stage 4 escape-to-infinity path (every
        // pixel ran to max_iter, since McMullen2's own critical orbits
        // don't need to reach INFINITY specifically to be "interesting" --
        // they can settle into finite attracting cycles that the retired
        // path had no way to report as anything but "unresolved").
        Renderer mc(Map(Family::McMullen2, Cplx(1.0, 0.0)), Viewport{{0.0, 0.0}, 2.0, 61},
                   RenderSettings{200, 2.0, 1e-6, 1});
        check(mc.map().escape_certified() == false,
              "sanity: McMullen2 is a built-in RATIONAL (non-certified) family");

        const Image mc_param = mc.render_parameter();
        bool mc_some_nonzero = false, mc_some_zero = false;
        for (double v : mc_param.data) { if (v > 0.0) mc_some_nonzero = true; else mc_some_zero = true; }
        check(mc_some_nonzero && mc_some_zero,
              "RESTORED BEHAVIOR: McMullen2's Parameter render has BOTH escaped (nonzero) "
              "and non-escaped (0) pixels -- a real escape-time classification, not the "
              "retired path's uniform max_iter plane");

        // escape_radius genuinely matters again for this rational family
        // too -- a different, real acceptance target from the retired
        // path's escape-radius INVARIANCE (which correctly no longer
        // applies here at all, per render_parameter's own doc comment).
        Renderer mc_r10(Map(Family::McMullen2, Cplx(1.0, 0.0)), Viewport{{0.0, 0.0}, 2.0, 61},
                        RenderSettings{200, 10.0, 1e-6, 1});
        bool mc_differs = false;
        const Image mc_param_r10 = mc_r10.render_parameter();
        for (std::size_t i = 0; i < mc_param.data.size(); ++i) {
            if (mc_param.data[i] != mc_param_r10.data[i]) mc_differs = true;
        }
        check(mc_differs,
              "...and escape_radius genuinely changes McMullen2's Parameter render too, "
              "the same VISUALIZATION-knob behavior as the certified-polynomial case above");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
