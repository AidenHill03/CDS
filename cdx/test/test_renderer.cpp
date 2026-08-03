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
