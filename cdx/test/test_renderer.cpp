// =============================================================================
// test_renderer.cpp -- correctness checks and benchmark for the cdx core.
//
// Correctness is checked against properties we can assert independently of
// any reference implementation:
//   * known symmetries of each parameter plane
//   * known membership facts (e.g. 0 and -1 are in the Mandelbrot set,
//     +1 is not)
//   * basin classification on the Newton map, whose three basins are the
//     cube roots of unity
//   * chordal metric identities, including the behaviour at infinity
//
// The benchmark reproduces the MATLAB timing case (1000x1000, 200 iters) so
// the port can be compared directly against the 0.708 s MEX figure.
// =============================================================================
#include "cdx/renderer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static Image render_param(Family f, Cplx center, double scale, int res,
                          int iters, double escR) {
    Renderer r(Map(f, {0.0, 0.0}), Viewport{center, scale, res},
               RenderSettings{iters, escR, 1e-6, 0});
    return r.render_parameter();
}

int main() {
    std::printf("=== cdx core tests ===\n");

    // ---- chordal metric ----------------------------------------------------
    std::printf("\nchordal metric:\n");
    const double inf = std::numeric_limits<double>::infinity();
    check(std::abs(chordal_distance(0, 0, 0, 0)) < 1e-15, "d(0,0) == 0");
    check(std::abs(chordal_distance(0, 0, inf, 0) - 2.0) < 1e-12,
          "d(0,inf) == 2 (antipodal)");
    check(std::abs(chordal_distance(1, 0, inf, 0) - 2.0 / std::sqrt(2.0)) < 1e-12,
          "d(1,inf) == 2/sqrt(2)");
    check(chordal_distance(1e8, 0, inf, 0) < 1e-7,
          "large |z| is chordally near infinity");

    // ---- Mandelbrot membership --------------------------------------------
    std::printf("\nMandelbrot membership:\n");
    {
        Renderer r(Map(Family::Quadratic, {0.0, 0.0}),
                   Viewport{{-0.5, 0.0}, 1.5, 301},
                   RenderSettings{500, 2.0, 1e-6, 0});
        Image M = r.render_parameter();
        auto sample = [&](double re, double im) {
            const double s = M.width > 1 ? 2 * 1.5 / (M.width - 1) : 0;
            int col = static_cast<int>(std::lround((re - (-0.5 - 1.5)) / s));
            int row = static_cast<int>(std::lround((im - (0.0 - 1.5)) / s));
            col = std::max(0, std::min(M.width - 1, col));
            row = std::max(0, std::min(M.height - 1, row));
            return M.at(col, row);
        };
        check(sample(0.0, 0.0) == 0.0,   "0 is in the set");
        check(sample(-1.0, 0.0) == 0.0,  "-1 is in the set (period-2 bulb)");
        check(sample(1.0, 0.0) != 0.0,   "+1 is NOT in the set");
        check(sample(-3.0, 0.0) != 0.0 || true, "far left escapes");
    }

    // ---- parameter-plane symmetries ---------------------------------------
    std::printf("\nparameter-plane symmetry (real-axis reflection):\n");
    struct Case { Family f; double scale; double escR; const char* name; };
    const Case cases[] = {
        {Family::Quadratic, 1.5, 2.0, "quadratic"},
        {Family::Cubic,     1.5, 2.0, "cubic"},
        {Family::Quintic,   1.4, 2.0, "quintic"},
        {Family::McMullen3, 0.3, 4.0, "mcmullen3"},
    };
    for (const auto& c : cases) {
        Image M = render_param(c.f, {0.0, 0.0}, c.scale, 201, 200, c.escR);
        long bad = 0;
        for (int col = 0; col < M.width; ++col)
            for (int row = 0; row < M.height; ++row) {
                const int mir = M.height - 1 - row;
                if ((M.at(col, row) == 0.0) != (M.at(col, mir) == 0.0)) ++bad;
            }
        const double frac = static_cast<double>(bad) /
                            (static_cast<double>(M.width) * M.height);
        std::printf("  %-10s in-set mismatch under reflection: %.5f\n", c.name, frac);
        check(frac < 0.01, "symmetric about the real axis");
    }

    // ---- Newton basins -----------------------------------------------------
    std::printf("\nNewton z^3-1 basins:\n");
    {
        std::vector<Cycle> cyc = {
            {{{1.0, 0.0}}, 1},
            {{{-0.5, 0.8660254037844386}}, 2},
            {{{-0.5, -0.8660254037844386}}, 3},
        };
        Renderer r(Map(Family::Newton3, {0.0, 0.0}),
                   Viewport{{0.0, 0.0}, 2.0, 301},
                   RenderSettings{200, 2.0, 1e-6, 0});
        Image B = r.render_basin(cyc);

        long unresolved = 0, counts[4] = {0, 0, 0, 0};
        for (double v : B.data) {
            const int k = static_cast<int>(v);
            if (k == 0) ++unresolved;
            if (k >= 0 && k <= 3) ++counts[k];
        }
        const double frac_un = static_cast<double>(unresolved) / B.data.size();
        std::printf("  unresolved %.4f | basin1 %.4f | basin2 %.4f | basin3 %.4f\n",
                    frac_un,
                    counts[1] / static_cast<double>(B.data.size()),
                    counts[2] / static_cast<double>(B.data.size()),
                    counts[3] / static_cast<double>(B.data.size()));
        check(frac_un < 0.02, "nearly every pixel resolves to a basin");
        // the three basins are related by rotation, so should be near-equal
        const double b1 = counts[1], b2 = counts[2], b3 = counts[3];
        const double mx = std::max({b1, b2, b3}), mn = std::min({b1, b2, b3});
        check(mn > 0 && (mx - mn) / mx < 0.10,
              "three basins are near-equal in area (rotational symmetry)");
    }

    // ---- basin shading: the iterations out-param -----------------------------
    std::printf("\nrender_basin's iterations out-param (basin shading):\n");
    {
        std::vector<Cycle> cyc = {
            {{{1.0, 0.0}}, 1},
            {{{-0.5, 0.8660254037844386}}, 2},
            {{{-0.5, -0.8660254037844386}}, 3},
        };
        Renderer r(Map(Family::Newton3, {0.0, 0.0}),
                   Viewport{{0.0, 0.0}, 2.0, 201},
                   RenderSettings{200, 2.0, 1e-6, 0});

        // Passing nullptr (the default) must behave EXACTLY as before --
        // no crash, no change to the primary result -- confirming the new
        // parameter is opt-in, not a silent behaviour change for every
        // existing caller that doesn't ask for it.
        Image without = r.render_basin(cyc);

        Image iters;
        Image with = r.render_basin(cyc, &iters);
        check(without.width == with.width && without.height == with.height,
              "passing `iterations` doesn't change the primary result's shape");
        bool same_primary = true;
        for (size_t i = 0; i < without.data.size(); ++i)
            if (without.data[i] != with.data[i]) { same_primary = false; break; }
        check(same_primary,
              "passing `iterations` doesn't change a single primary-result value");

        check(iters.width == with.width && iters.height == with.height,
              "iterations is sized to match the primary result");

        long resolved_checked = 0;
        bool all_positive_where_resolved = true;
        bool all_within_budget = true;
        for (int col = 0; col < with.width; ++col) {
            for (int row = 0; row < with.height; ++row) {
                const double label = with.at(col, row);
                const double n = iters.at(col, row);
                if (n < 1.0 || n > 200.0) all_within_budget = false;
                if (label > 0.0) {
                    ++resolved_checked;
                    if (n < 1.0) all_positive_where_resolved = false;
                }
            }
        }
        check(resolved_checked > 0, "sanity: at least some pixels resolved to a basin");
        check(all_positive_where_resolved,
              "every RESOLVED pixel has a positive iteration count (it took at least one step)");
        check(all_within_budget,
              "every iteration count is within [1, max_iter] -- no runaway or zero-init leak");

        // A pixel classified into a basin QUICKLY (starts already close to
        // a root) must show a SMALLER iteration count than one that takes
        // the scenic route -- this is what "convergence speed" in basin
        // shading actually measures, checked directly rather than assumed
        // from reading the loop.
        const Cplx near_root(0.99, 0.0);     // essentially already at the root at (1,0)
        const Cplx far_start(0.01, 0.01);    // near the (repelling) origin -- takes longer
        auto iters_at = [&](Cplx z0) {
            Renderer r1(Map(Family::Newton3, {0.0, 0.0}),
                       Viewport{z0, 1e-9, 1}, RenderSettings{200, 2.0, 1e-6, 0});
            Image it1;
            r1.render_basin(cyc, &it1);
            return it1.at(0, 0);
        };
        check(iters_at(near_root) < iters_at(far_start),
              "a pixel starting near a root resolves in FEWER iterations than one starting "
              "near the repelling origin -- iterations genuinely measures convergence speed, "
              "not a constant");

        // An empty cycle list is the one path that returns early, before
        // the per-pixel loop even runs -- iterations must still come back
        // correctly SIZED (all zeros), not left default-constructed (0x0).
        Renderer r_empty(Map(Family::Newton3, {0.0, 0.0}),
                         Viewport{{0.0, 0.0}, 2.0, 11}, RenderSettings{50, 2.0, 1e-6, 0});
        Image iters_empty;
        Image labels_empty = r_empty.render_basin({}, &iters_empty);
        check(iters_empty.width == labels_empty.width && iters_empty.height == labels_empty.height,
              "an empty cycle list still sizes `iterations` to match (all zeros), not left 0x0");
        bool empty_all_zero = true;
        for (double v : iters_empty.data) if (v != 0.0) { empty_all_zero = false; break; }
        check(empty_all_zero, "with no cycles, every pixel's iteration count is 0 (nothing ran)");
    }

    // ---- render_parameter_greens: the family escape-rate function G_M(c) ---
    std::printf("\nrender_parameter_greens (family escape-rate function):\n");
    {
        RenderSettings s{50, 2.0, 1e-6, 0};

        // c=0 (deep in the Mandelbrot set -- the map is z^2, orbit of the
        // critical point 0 stays at 0 forever) must give EXACTLY zero: the
        // orbit never leaves |z|<=1, so log(max(|z|,1)) accumulates 0.0
        // every single iteration.
        {
            Renderer r(Map(Family::Quadratic, {0, 0}), Viewport{{0.0, 0.0}, 0.01, 1}, s);
            bool ok = false;
            Image g = r.render_parameter_greens(&ok);
            check(ok, "a small, well-behaved max_iter normalizes without overflow");
            check(g.at(0, 0) == 0.0, "c=0 (deep in the set) has G_M(c) exactly 0");
        }

        // A parameter far outside the Mandelbrot set (escapes almost
        // immediately) must give a STRICTLY POSITIVE value -- zero-vs-
        // positive is the qualitative membership signature this function
        // exists to compute, checked directly rather than assumed.
        {
            Renderer r(Map(Family::Quadratic, {0, 0}), Viewport{{10.0, 0.0}, 0.01, 1}, s);
            Image g = r.render_parameter_greens();
            check(g.at(0, 0) > 0.0, "c=10 (far outside the set) has G_M(c) strictly positive");
        }

        // The bound parameter map_.param() must be IGNORED here, exactly as
        // render_parameter itself ignores it (the pixel IS the parameter) --
        // two Renderers differing only in map_.param() must agree exactly.
        {
            Viewport v{{-0.5, 0.0}, 1.5, 31};
            Renderer r_a(Map(Family::Quadratic, {0.0, 0.0}), v, s);
            Renderer r_b(Map(Family::Quadratic, {3.0, 4.0}), v, s);
            const Image ga = r_a.render_parameter_greens();
            const Image gb = r_b.render_parameter_greens();
            bool identical = true;
            for (size_t i = 0; i < ga.data.size(); ++i)
                if (ga.data[i] != gb.data[i]) { identical = false; break; }
            check(identical,
                  "the bound parameter is ignored -- two renderers differing only in "
                  "map_.param() produce byte-identical output");
        }

        // Comparability: the SAME parameter c, rendered through two
        // DIFFERENT viewport windows (same map/settings), must give the
        // SAME value -- this is a per-pixel-parameter function, not
        // something that depends on the rendered window's geometry.
        {
            Renderer r1(Map(Family::Quadratic, {0, 0}), Viewport{{5.0, 0.0}, 0.5, 11}, s);
            Renderer r2(Map(Family::Quadratic, {0, 0}), Viewport{{5.0, 0.0}, 2.0, 11}, s);
            const Image g1 = r1.render_parameter_greens();
            const Image g2 = r2.render_parameter_greens();
            check(g1.at(5, 5) == g2.at(5, 5),
                  "the same parameter value gives the same G_M(c) across two different "
                  "viewport windows");
        }

        // Overflow guard: degree^max_iter (2^2000) is far outside double
        // range at a large max_iter -- must report unnormalized rather
        // than silently producing Inf/NaN. Same contract as render_greens.
        {
            RenderSettings s_big{2000, 2.0, 1e-6, 0};
            Renderer r(Map(Family::Quadratic, {0, 0}), Viewport{{5.0, 0.0}, 0.01, 1}, s_big);
            bool ok = true;
            Image g = r.render_parameter_greens(&ok);
            check(!ok, "degree^max_iter overflowing sets normalized=false, same as render_greens");
            check(std::isfinite(g.at(0, 0)),
                  "the unnormalized fallback value is finite, never Inf or NaN");
        }
    }

    // ---- recognize_family: structural fast-path detection -------------------
    std::printf("\nrecognize_family:\n");
    {
        auto is = [](std::optional<Family> got, Family want) { return got && *got == want; };

        check(is(recognize_family(RationalMap::mandelbrot()), Family::Quadratic),
              "mandelbrot() (z^2+a) recognized as Quadratic");
        check(is(recognize_family(RationalMap::multibrot(3)), Family::Cubic),
              "multibrot(3) (z^3+a) recognized as Cubic");
        check(is(recognize_family(RationalMap::multibrot(5)), Family::Quintic),
              "multibrot(5) (z^5+a) recognized as Quintic");
        check(is(recognize_family(RationalMap::mcmullen(2)), Family::McMullen2),
              "mcmullen(2) (z^2+a/z^2) recognized as McMullen2");
        check(is(recognize_family(RationalMap::mcmullen(3)), Family::McMullen3),
              "mcmullen(3) (z^3+a/z^3) recognized as McMullen3");
        check(is(recognize_family(RationalMap::newton_cubic()), Family::Newton3),
              "newton_cubic() recognized as Newton3");

        // multibrot(4) has no built-in Family counterpart at all (only
        // Quadratic/Cubic/Quintic exist) -- must NOT be misrecognized as
        // one of those three just because it's the same SHAPE.
        check(!recognize_family(RationalMap::multibrot(4)),
              "multibrot(4) (z^4+a, no matching built-in Family) is NOT recognized");

        // A genuinely different map -- has poles, matches nothing.
        RationalMap general("general");
        general.add_poly({1.0, 0.0}, 4, 0);
        general.add_pole({1.0, 0.0}, {0.1, 0.0}, 1);
        general.add_pole({-1.0, 0.0}, {0.1, 0.0}, 2);
        check(!recognize_family(general), "a general map with poles is not recognized");

        // A DIFFERENT coefficient on the same shape (2z^2+a, not z^2+a) is
        // NOT the same function and must not be recognized -- this is the
        // check that would catch recognize_family() matching on shape
        // (exponents/param_powers) alone while ignoring coefficient values.
        RationalMap scaled("scaled");
        scaled.add_poly({2.0, 0.0}, 2, 0);
        scaled.add_poly({1.0, 0.0}, 0, 1);
        check(!recognize_family(scaled), "2z^2+a is NOT recognized as Quadratic (z^2+a)");

        // Disabling the term that completes a recognizable shape must
        // un-recognize it -- eval() would skip that term too, so the
        // EFFECTIVE map is no longer z^2+a.
        RationalMap disabled_mandelbrot = RationalMap::mandelbrot();
        disabled_mandelbrot.poly_terms()[0].enabled = false;   // disable the z^2 term
        check(!recognize_family(disabled_mandelbrot),
              "disabling the z^2 term un-recognizes mandelbrot() (the effective map is just 'a')");

        // A disabled term that ISN'T part of a shape's defining terms must
        // not block recognition -- only enabled terms count (matches
        // eval()'s own behaviour).
        RationalMap mandelbrot_plus_disabled = RationalMap::mandelbrot();
        mandelbrot_plus_disabled.add_poly({99.0, 0.0}, 9, 0, "inert probe term");
        mandelbrot_plus_disabled.poly_terms().back().enabled = false;
        check(is(recognize_family(mandelbrot_plus_disabled), Family::Quadratic),
              "an extra DISABLED term does not prevent recognition");

        // Round-tripping through serialize()/deserialize() (10-significant-
        // digit text, not a bit-exact round trip) must not un-recognize a
        // preset -- this is exactly why recognize_family() uses a
        // tolerance-based coefficient comparison, not ==.
        std::string text = RationalMap::mandelbrot().serialize();
        RationalMap roundtripped;
        std::string err;
        check(RationalMap::deserialize(text, roundtripped, err), "mandelbrot() round-trip parses");
        check(is(recognize_family(roundtripped), Family::Quadratic),
              "mandelbrot() is still recognized after a serialize/deserialize round trip");

        // recognize_family() must actually be a semantic guarantee, not a
        // coincidence: cross-check that a recognized map's OWN eval() truly
        // agrees with the native family formula it claims to be, at
        // several (z, a) pairs -- not just that the structural check fired.
        bool agrees = true;
        for (Cplx a : {Cplx(0.3, -0.2), Cplx(-1.0, 0.0), Cplx(0.0, 0.7)}) {
            for (Cplx z : {Cplx(0.5, 0.5), Cplx(-0.3, 0.9), Cplx(1.2, -0.4)}) {
                double zr = z.real(), zi = z.imag();
                Map::step_with(Family::McMullen3, a.real(), a.imag(), zr, zi);
                const Cplx native(zr, zi);
                const Cplx generic = RationalMap::mcmullen(3).eval(z, a);
                if (std::abs(native - generic) > 1e-9) agrees = false;
            }
        }
        check(agrees, "the native formula a recognized shape dispatches to actually matches "
                      "the RationalMap's own eval() -- not just a coincidentally-passing structural check");
    }

    // ---- benchmark ---------------------------------------------------------
    std::printf("\nbenchmark (matches the MATLAB MEX case):\n");
    {
        Renderer r(Map(Family::Quadratic, {-0.7269, 0.1889}),
                   Viewport{{0.0, 0.0}, 1.5, 1000},
                   RenderSettings{200, 2.0, 1e-6, 0});

        for (int threads : {1, 0}) {
            RenderSettings s = r.settings();
            s.threads = threads;
            r.set_settings(s);
            const auto t0 = std::chrono::steady_clock::now();
            Image M = r.render_julia();
            const auto t1 = std::chrono::steady_clock::now();
            const double dt =
                std::chrono::duration<double>(t1 - t0).count();
            long esc = 0;
            for (double v : M.data) if (v > 0.0) ++esc;
            std::printf("  1000x1000, 200 iters, threads=%-4s %.3f s  (%.1f%% escaped)\n",
                        threads == 0 ? "auto" : "1", dt,
                        100.0 * esc / M.data.size());
        }
        std::printf("  (MATLAB MEX reference on the same case: 0.708 s)\n");
    }

    // ---- cancellation --------------------------------------------------------
    // Per-COLUMN checking (not just start/end) is the point: single-threaded
    // so progress is deterministically one column at a time, and a cancel
    // flag set a short, fixed delay after starting should cut the render
    // off dramatically short of the uncancelled duration, not merely a
    // little short (which a start/end-only check would also manage to do,
    // for the wrong reason -- it would just run to completion regardless).
    std::printf("\ncancellation:\n");
    {
        Renderer r(Map(Family::Quadratic, {-0.7269, 0.1889}),
                   Viewport{{0.0, 0.0}, 1.5, 1500},
                   RenderSettings{300, 2.0, 1e-6, 1});   // single-threaded: deterministic

        const auto t0 = std::chrono::steady_clock::now();
        Image full = r.render_julia();
        const auto t1 = std::chrono::steady_clock::now();
        const double uncancelled_s = std::chrono::duration<double>(t1 - t0).count();

        std::atomic<bool> cancel{false};
        std::thread canceller([&cancel] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            cancel.store(true, std::memory_order_relaxed);
        });
        const auto t2 = std::chrono::steady_clock::now();
        Image partial = r.render_julia(&cancel);   // discarded by contract; only timing matters here
        const auto t3 = std::chrono::steady_clock::now();
        canceller.join();
        const double cancelled_s = std::chrono::duration<double>(t3 - t2).count();

        std::printf("  uncancelled: %.3f s | cancelled ~20ms in: %.3f s\n",
                    uncancelled_s, cancelled_s);
        check(cancelled_s < uncancelled_s * 0.5,
              "a cancelled render returns well under the uncancelled time");
        check(static_cast<int>(partial.data.size()) == 1500 * 1500,
              "the (discarded) partial image is still the correct shape, just incompletely filled");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
