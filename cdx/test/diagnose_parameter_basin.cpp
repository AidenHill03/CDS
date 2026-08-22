// =============================================================================
// diagnose_parameter_basin.cpp -- MEASUREMENT ONLY, no production behavior
// changed by this file. Diagnoses the Parameter_basin "thick unresolved
// boundary band + scattered bulk specks + slowness" problem on a Nova-style
// rational family, per the batch's own Stage 1 spec.
//
// This is a REPORT tool, not a PASS/FAIL test suite (though it includes a
// couple of basic sanity checks using the same [PASS]/[FAIL] convention as
// the rest of this project's tests, to confirm the tool itself is measuring
// what it claims to). It does not modify cdx/src/renderer.cpp or
// cdx/src/analysis.cpp at all -- every measurement here is taken by calling
// the EXISTING, unmodified public API (RationalMap::eval/deriv/
// distinct_critical_points, find_attractors_from_seeds, Renderer::
// render_parameter_basin) either directly or with a hand-rolled instrumented
// REIMPLEMENTATION of find_attractors_from_seeds' own per-seed loop (kept
// deliberately identical in its default-settings behavior -- see
// instrumented_classify's own comment -- so what it measures is genuinely
// what the production loop does, not a different algorithm).
//
// "Nova": Newton's method for z^3-1 -- (2/3)z + (1/3)z^-2, Newton3's own
// exact shape -- PLUS an ADDITIVE parameter a (a constant term, param_power
// 1, exponent 0). The +a term does not change where the derivative
// vanishes (a constant's derivative is 0), so this map's critical points
// are the SAME 3 cube-roots-of-unity + the order-2 pole at the origin
// Newton3 always has; what changes is where those orbits actually LAND,
// since Newton's own root-finding guarantee no longer applies once the
// iteration is perturbed by +a. This is the classical "Nova fractal"
// construction, well known to produce rich boundary structure between the
// three root basins in the parameter plane -- exactly the kind of
// structure this diagnostic needs to exercise.
// =============================================================================
#include "cdx/analysis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cdx;
using Clock = std::chrono::steady_clock;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static bool is_finite_cplx(Cplx z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

static double chordal(Cplx z, Cplx w) {
    return chordal_distance(z.real(), z.imag(), w.real(), w.imag());
}

static RationalMap build_nova() {
    RationalMap nova("nova");
    nova.add_poly({2.0 / 3.0, 0.0}, 1, 0, "(2/3)z");
    nova.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "(1/3)z^-2");
    nova.add_poly({1.0, 0.0}, 0, 1, "a");   // additive parameter -- Newton3 + a
    return nova;
}

// -----------------------------------------------------------------------------
// A hand-instrumented REIMPLEMENTATION of find_attractors_from_seeds' own
// per-seed body (cdx/src/analysis.cpp), extended to ALSO try a much looser
// closure tolerance over a much larger period budget when the strict
// (default) settings fail to resolve -- purely for measurement, never
// called from renderer.cpp. Mirrors the production loop's burn-in, pole
// nudge, and infinity handling exactly so the "did it resolve at strict
// settings" half of this function's own result is a faithful reproduction
// of what find_attractors_from_seeds itself would report, not an
// approximation of it.
// -----------------------------------------------------------------------------
struct SeedDiagnosis {
    bool   resolved_strict   = false;   // matches find_attractors_from_seeds' own verdict
    bool   found_loose_cycle = false;   // a candidate period closed at the LOOSE tolerance
    int    loose_period      = 0;
    Cplx   loose_multiplier  {0.0, 0.0};
    bool   loose_attracting  = false;   // |loose_multiplier| < 1
    long long iterations_run = 0;       // burn-in + period-search steps actually executed
};

static SeedDiagnosis diagnose_seed(const RationalMap& map, Cplx a, Cplx seed,
                                   const FindAttractorsOptions& strict_opts,
                                   double loose_tol, int loose_max_period) {
    SeedDiagnosis diag;

    Cplx z = is_finite_cplx(seed) ? seed : Cplx(strict_opts.inf_cutoff, 0.0);
    if (is_finite_cplx(z)) {
        for (Cplx p : map.pole_locations(a)) {
            if (std::abs(z - p) < 1e-9) { z += Cplx(1e-4, 3.7e-5); break; }
        }
    }

    bool at_inf = false;
    for (int n = 0; n < strict_opts.burn_in; ++n) {
        z = map.eval(z, a);
        ++diag.iterations_run;
        if (!is_finite_cplx(z) || std::abs(z) > strict_opts.inf_cutoff) { at_inf = true; break; }
    }

    if (at_inf) {
        bool infinity_attracting = false;
        for (const FixedPoint& fp : map.fixed_points(a)) {
            if (!is_finite_cplx(fp.point) && std::abs(fp.multiplier) < 1.0) {
                infinity_attracting = true;
                break;
            }
        }
        diag.resolved_strict = infinity_attracting;
        // Infinity excursions aren't the "slow-but-real vs genuinely-non-
        // converging" question this diagnostic is measuring (that question
        // is specifically about the closure-tolerance/period-budget
        // trade-off for a FINITE candidate cycle) -- report as resolved or
        // not and stop there.
        return diag;
    }

    // ---- strict pass: EXACTLY find_attractors_from_seeds' own loop -----------
    std::vector<Cplx> orbit;
    orbit.reserve(static_cast<std::size_t>(strict_opts.max_period) + 1);
    orbit.push_back(z);
    int found = 0;
    bool hit_inf_mid = false;
    for (int k = 0; k < strict_opts.max_period; ++k) {
        Cplx zn = map.eval(orbit.back(), a);
        ++diag.iterations_run;
        const bool zn_inf = !is_finite_cplx(zn) || std::abs(zn) > strict_opts.inf_cutoff;
        if (zn_inf) zn = Cplx(std::numeric_limits<double>::infinity(), 0.0);
        orbit.push_back(zn);
        if (chordal(zn, orbit.front()) < strict_opts.tol) { found = k + 1; break; }
        if (zn_inf) { hit_inf_mid = true; break; }
    }
    if (found > 0 && !hit_inf_mid) {
        std::vector<Cplx> cyc(orbit.begin(), orbit.begin() + found);
        Cplx multiplier(1.0, 0.0);
        for (Cplx zc : cyc) multiplier *= map.deriv(zc, a);
        diag.resolved_strict = std::abs(multiplier) < 1.0;
    }
    if (diag.resolved_strict) return diag;

    // ---- loose pass: same orbit, continued past max_period, looser tol -------
    // Continues iterating the SAME orbit (not restarted) -- burn-in already
    // happened once, and a genuinely slow-but-real cycle should still be
    // converging monotonically toward it.
    for (int k = strict_opts.max_period; k < loose_max_period; ++k) {
        Cplx zn = map.eval(orbit.back(), a);
        ++diag.iterations_run;
        const bool zn_inf = !is_finite_cplx(zn) || std::abs(zn) > strict_opts.inf_cutoff;
        if (zn_inf) zn = Cplx(std::numeric_limits<double>::infinity(), 0.0);
        orbit.push_back(zn);
        if (chordal(zn, orbit.front()) < loose_tol) {
            diag.found_loose_cycle = true;
            diag.loose_period = k + 1;
            break;
        }
        if (zn_inf) break;
    }
    if (diag.found_loose_cycle) {
        std::vector<Cplx> cyc(orbit.begin(), orbit.begin() + diag.loose_period);
        Cplx multiplier(1.0, 0.0);
        for (Cplx zc : cyc) multiplier *= map.deriv(zc, a);
        diag.loose_multiplier = multiplier;
        diag.loose_attracting = std::abs(multiplier) < 1.0;
    }
    return diag;
}

int main() {
    std::printf("=== Parameter_basin diagnostic: Nova family (MEASUREMENT ONLY) ===\n\n");

    const RationalMap nova = build_nova();
    check(polynomial_escape_certified(nova) == false, "sanity: Nova has a pole, not certified");
    check(nova.critical_points_constant(),
          "sanity: Nova's critical points don't depend on `a` (the +a term is a constant, "
          "derivative 0) -- confirmed, not assumed, since Stage 1's own profiling below "
          "depends on knowing whether distinct_critical_points(p) is paid once or per pixel");
    const auto crit_pts = nova.distinct_critical_points(Cplx(0.0, 0.0));
    std::printf("  Nova's own critical points (%zu): ", crit_pts.size());
    for (Cplx c : crit_pts) {
        if (is_finite_cplx(c)) std::printf("(%.4f%+.4fi) ", c.real(), c.imag());
        else std::printf("(inf) ");
    }
    std::printf("\n\n");

    // ---- AUDIT: FindAttractorsOptions' fields and defaults --------------------
    std::printf("---- AUDIT: FindAttractorsOptions fields and defaults ----\n");
    FindAttractorsOptions defaults;
    std::printf("  burn_in          = %d\n", defaults.burn_in);
    std::printf("  max_period       = %d\n", defaults.max_period);
    std::printf("  tol              = %.3e\n", defaults.tol);
    std::printf("  inf_cutoff       = %.3e\n", defaults.inf_cutoff);
    std::printf("  verify_multiplier= %s\n", defaults.verify_multiplier ? "true" : "false");
    std::printf("  (NOTE: there is no 'max_iter' field on this struct -- the effective\n"
               "   per-seed iteration budget is burn_in + max_period = %d + %d = %d total\n"
               "   steps before a seed is given up on as unresolved.)\n\n",
               defaults.burn_in, defaults.max_period, defaults.burn_in + defaults.max_period);

    std::printf("---- AUDIT: acceptance test inside find_attractors_from_seeds ----\n"
               "  Closure-first, THEN multiplier: a seed is accepted ONLY if (a) its orbit\n"
               "  returns within `tol` (chordal) of its OWN post-burn-in starting point\n"
               "  within `max_period` additional steps (found > 0), AND (b) the product of\n"
               "  deriv() around the resulting cycle has |multiplier| < 1. If (a) fails --\n"
               "  no closure within the strict tol/max_period budget -- the seed is rejected\n"
               "  as unresolved and (b) is NEVER REACHED, regardless of whether a genuine\n"
               "  attracting cycle exists nearby that would have closed given a looser\n"
               "  tolerance or a few more iterations. This is the exact mechanism this\n"
               "  diagnostic measures below.\n\n");

    std::printf("---- AUDIT: does Cycle already carry a multiplier? ----\n"
               "  NO -- cdx::Cycle (cdx/include/cdx/renderer.hpp) has only {points, id}.\n"
               "  The per-cycle multiplier IS already computed as a LOCAL variable inside\n"
               "  find_attractors_from_seeds' own accept/reject check (cdx/src/analysis.cpp,\n"
               "  ~line 175), but it is discarded after the check, not stored anywhere. A\n"
               "  SEPARATE, richer struct -- DynamicalFacts::AttractingCycle -- DOES carry a\n"
               "  `multiplier` field, but that is a different type built by a different\n"
               "  function (dynamical_facts()), not what render_parameter_basin consumes.\n"
               "  CORRECTION to this batch's own premise: propagating the multiplier is\n"
               "  still cheap (the computation already happens inline; it's a matter of\n"
               "  keeping the value rather than recomputing it), but it is not \"already\n"
               "  there for free\" on Cycle itself -- Stage 2 will need to add it.\n\n");

    // ---- MEASURE: render Parameter_basin on Nova, find unresolved pixels ------
    // Located by an exploratory scan (not guessed): a coarse full-plane
    // render's own unresolved-pixel density was swept with a sliding
    // window to find the densest region, then progressively zoomed until
    // the pattern was unambiguously a connected, CURVED boundary band
    // (not scattered specks) -- an ASCII render of THIS exact viewport
    // confirmed three clear connected bands plus scattered bulk specks
    // before this value was hardcoded here.
    const Viewport v{{-0.44, 0.0}, 0.1, 141};
    const RenderSettings settings{100, 2.0, 1e-6, 1};
    Renderer r(Map::custom(nova), v, settings);

    const auto t_render_start = Clock::now();
    Image unresolved_img;
    const Image counts = r.render_parameter_basin(nullptr, &unresolved_img);
    const auto t_render_end = Clock::now();
    const double render_ms =
        std::chrono::duration<double, std::milli>(t_render_end - t_render_start).count();

    int n_unresolved_pixels = 0, n_zero_count = 0;
    for (std::size_t i = 0; i < counts.data.size(); ++i) {
        if (unresolved_img.data[i] > 0.0) ++n_unresolved_pixels;
        if (counts.data[i] == 0.0) ++n_zero_count;
    }
    std::printf("---- MEASURE: render_parameter_basin on Nova, %dx%d ----\n", v.resolution,
               v.resolution);
    std::printf("  render time: %.1f ms\n", render_ms);
    std::printf("  pixels with unresolved > 0: %d / %d (%.1f%%)\n", n_unresolved_pixels,
               static_cast<int>(counts.data.size()),
               100.0 * n_unresolved_pixels / static_cast<double>(counts.data.size()));
    std::printf("  pixels with count == 0 (nothing confirmed at all): %d (%.1f%%)\n\n",
               n_zero_count, 100.0 * n_zero_count / static_cast<double>(counts.data.size()));

    // ---- classify sampled unresolved pixels: bulk speck vs boundary band ------
    // Bulk speck: this pixel's own count differs from a genuine confirmed
    // count that a clear MAJORITY of its 8-neighbourhood shares (i.e. it
    // sits inside an otherwise-solid, resolved region). Boundary band: a
    // majority of its 8-neighbourhood is ALSO unresolved (part of a
    // connected unresolved region, not an isolated pixel).
    auto at = [&](int col, int row) { return row * v.resolution + col; };
    std::vector<std::pair<int, int>> speck_pixels, band_pixels;
    for (int row = 1; row < v.resolution - 1; ++row) {
        for (int col = 1; col < v.resolution - 1; ++col) {
            if (unresolved_img.data[at(col, row)] <= 0.0) continue;
            int neighbours_unresolved = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    if (unresolved_img.data[at(col + dc, row + dr)] > 0.0) ++neighbours_unresolved;
                }
            }
            if (neighbours_unresolved >= 5) band_pixels.push_back({col, row});
            else if (neighbours_unresolved <= 2) speck_pixels.push_back({col, row});
        }
    }
    // Interior pixels only (the classification loop above excludes the
    // outermost ring, which has no full 8-neighbourhood) -- computed
    // directly rather than assumed, so "ambiguous" below is exact.
    int n_unresolved_interior = 0;
    for (int row = 1; row < v.resolution - 1; ++row)
        for (int col = 1; col < v.resolution - 1; ++col)
            if (unresolved_img.data[at(col, row)] > 0.0) ++n_unresolved_interior;

    std::printf("---- CLASSIFY: unresolved-pixel spatial pattern (interior pixels only) ----\n");
    std::printf("  boundary-band-like (>=5/8 neighbours also unresolved): %zu\n",
               band_pixels.size());
    std::printf("  bulk-speck-like (<=2/8 neighbours unresolved, isolated): %zu\n",
               speck_pixels.size());
    std::printf("  (ambiguous, neither clearly banded nor isolated, 3-4/8): %d\n\n",
               n_unresolved_interior -
               static_cast<int>(band_pixels.size()) - static_cast<int>(speck_pixels.size()));

    // ---- WHY: for a sample of each, re-run every critical orbit with instrumentation --
    auto sample_and_classify = [&](const std::vector<std::pair<int, int>>& pixels,
                                   const char* label, int max_samples) {
        int n_slow_but_real = 0, n_genuinely_stuck = 0, n_orbits_checked = 0;
        const int step = std::max(1, static_cast<int>(pixels.size()) / max_samples);
        for (std::size_t i = 0; i < pixels.size(); i += static_cast<std::size_t>(step)) {
            const Cplx p = v.coord(pixels[i].first, pixels[i].second);
            for (Cplx seed : crit_pts) {
                const SeedDiagnosis diag = diagnose_seed(nova, p, seed, FindAttractorsOptions{},
                                                         /*loose_tol=*/1e-4,
                                                         /*loose_max_period=*/4000);
                if (diag.resolved_strict) continue;   // this orbit wasn't the unresolved one
                ++n_orbits_checked;
                if (diag.found_loose_cycle && diag.loose_attracting) ++n_slow_but_real;
                else ++n_genuinely_stuck;
            }
        }
        std::printf("  [%s] sampled %d unresolved critical orbits (from ~%d pixels):\n", label,
                   n_orbits_checked, std::min(max_samples, static_cast<int>(pixels.size())));
        std::printf("      slow-but-real (loose closure found, |multiplier| < 1): %d (%.0f%%)\n",
                   n_slow_but_real,
                   n_orbits_checked ? 100.0 * n_slow_but_real / n_orbits_checked : 0.0);
        std::printf("      genuinely-non-converging (no attracting loose closure): %d (%.0f%%)\n",
                   n_genuinely_stuck,
                   n_orbits_checked ? 100.0 * n_genuinely_stuck / n_orbits_checked : 0.0);
    };

    std::printf("---- WHY: slow-but-real vs genuinely-non-converging ----\n");
    sample_and_classify(band_pixels, "boundary band", 40);
    sample_and_classify(speck_pixels, "bulk specks", 40);
    std::printf("\n");

    // ---- PROFILE: distinct_critical_points(p) vs orbit iteration -------------
    std::printf("---- PROFILE: per-pixel cost breakdown ----\n");
    const bool cp_fixed = nova.critical_points_constant();
    long long total_discovery_iters = 0;
    long long unresolved_discovery_iters = 0;
    int n_pixels_with_any_unresolved = 0;

    const auto t_cp_start = Clock::now();
    if (!cp_fixed) {
        for (int row = 0; row < v.resolution; ++row)
            for (int col = 0; col < v.resolution; ++col)
                (void)nova.distinct_critical_points(v.coord(col, row));
    } else {
        (void)nova.distinct_critical_points(Cplx(1.0, 0.0));
    }
    const auto t_cp_end = Clock::now();
    const double cp_ms = std::chrono::duration<double, std::milli>(t_cp_end - t_cp_start).count();

    // loose_max_period == strict max_period below (both from a default-
    // constructed FindAttractorsOptions) so diagnose_seed's own "loose
    // pass" loop never executes (its guard is `k < loose_max_period`
    // starting at `k = strict max_period`) -- this measures EXACTLY
    // find_attractors_from_seeds' own strict-pass cost, once per seed per
    // pixel, not the (much more expensive) extended loose search used
    // above for classification.
    const auto t_disc_start = Clock::now();
    for (int row = 0; row < v.resolution; ++row) {
        for (int col = 0; col < v.resolution; ++col) {
            const Cplx p = v.coord(col, row);
            bool any_unresolved_here = false;
            for (Cplx seed : crit_pts) {
                const SeedDiagnosis diag = diagnose_seed(nova, p, seed, FindAttractorsOptions{},
                                                         1e-4, FindAttractorsOptions{}.max_period);
                total_discovery_iters += diag.iterations_run;
                if (!diag.resolved_strict) {
                    unresolved_discovery_iters += diag.iterations_run;
                    any_unresolved_here = true;
                }
            }
            if (any_unresolved_here) ++n_pixels_with_any_unresolved;
        }
    }
    const auto t_disc_end = Clock::now();
    const double disc_ms =
        std::chrono::duration<double, std::milli>(t_disc_end - t_disc_start).count();

    std::printf("  distinct_critical_points(p) total: %.1f ms (%s -- Nova's own critical "
               "points don't depend on `a`)\n", cp_ms,
               cp_fixed ? "paid ONCE for the whole render" : "paid PER PIXEL");
    std::printf("  discovery (burn-in + closure search) total: %.1f ms\n", disc_ms);
    std::printf("  discovery iterations (all seeds, all pixels): %lld\n", total_discovery_iters);
    std::printf("  discovery iterations spent on seeds that end up UNRESOLVED: %lld (%.1f%% of "
               "all discovery iterations)\n", unresolved_discovery_iters,
               total_discovery_iters
                   ? 100.0 * static_cast<double>(unresolved_discovery_iters) /
                         static_cast<double>(total_discovery_iters)
                   : 0.0);
    std::printf("  pixels with >=1 unresolved seed: %d / %d\n\n", n_pixels_with_any_unresolved,
               v.resolution * v.resolution);

    // Cross-check: this file's own diagnose_seed reimplementation should
    // agree with the REAL, unmodified render_parameter_basin (via
    // find_attractors_from_seeds directly) on which pixels have any
    // unresolved seed -- confirms every measurement above reflects
    // production behavior, not a diverged reimplementation.
    check(n_pixels_with_any_unresolved == n_unresolved_pixels,
          "the instrumented reimplementation's own unresolved-pixel count matches "
          "render_parameter_basin's real, unmodified output exactly");

    // ---- SUPPLEMENTARY: is find_attractors_from_seeds' eval() cost avoidable? -----
    // RationalMap::eval(z, a) redoes every term's effective_coeff/
    // effective_location/effective_strength on EVERY call (documented in
    // rational.hpp) -- exactly what RationalMap::compile(a) -> CompiledMap
    // exists to amortize for a FIXED `a` used across many iterations (the
    // SAME pattern every other render_* method's own hot loop already
    // uses). find_attractors_from_seeds does NOT use it -- calls raw
    // map.eval(z, a) every iteration, even though `a` never changes across
    // one seed's own burn-in+closure loop. Quantify the gap directly: the
    // SAME number of steps, on the SAME orbit, through both paths.
    std::printf("\n---- SUPPLEMENTARY: eval(z,a) vs compile(a)+step(z) for the SAME orbit ----\n");
    {
        const Cplx probe_a{0.3, -0.2};
        const CompiledMap compiled = nova.compile(probe_a);
        constexpr int kProbeIters = 2'000'000;

        double zr = 0.2, zi = 0.1;
        Cplx z{0.2, 0.1};
        const auto t_eval_start = Clock::now();
        for (int i = 0; i < kProbeIters; ++i) z = nova.eval(z, probe_a);
        const auto t_eval_end = Clock::now();
        const double eval_ms =
            std::chrono::duration<double, std::milli>(t_eval_end - t_eval_start).count();

        const auto t_step_start = Clock::now();
        for (int i = 0; i < kProbeIters; ++i) compiled.step(zr, zi);
        const auto t_step_end = Clock::now();
        const double step_ms =
            std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();

        std::printf("  %d iterations: eval(z,a) %.1f ms | compile(a)+step(z) %.1f ms (%.2fx)\n",
                   kProbeIters, eval_ms, step_ms, step_ms > 0 ? eval_ms / step_ms : 0.0);
        std::printf("  (find_attractors_from_seeds' own inner loop uses eval(z,a) exclusively "
                   "-- see cdx/src/analysis.cpp -- despite `a` being fixed for the full "
                   "burn-in+closure search of one seed, exactly the case compile()+step() "
                   "exists for.)\n");
        check(std::abs(z.real() - zr) < 1e-6 && std::abs(z.imag() - zi) < 1e-6,
              "sanity: eval(z,a) and compile(a)+step(z) compute the SAME orbit -- the "
              "speedup above would be a free, behavior-preserving win if wired in");
    }

    std::printf("%s (%d failure%s)\n",
                failures == 0 ? "\nALL SANITY CHECKS PASSED" : "\nSOME SANITY CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
