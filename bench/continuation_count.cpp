// bench/continuation_count.cpp -- how often does the neighbour continuation of
// critical points and fixed points (commit 23a4e42) actually engage for the
// relaxed Newton family, and what does a cold solve cost?
//
// Walks parameter-plane columns exactly as Renderer::parallel_columns'
// workers do (previous row's roots as the predictor, reset per column) and
// counts ContinuedCriticalPoints::continued / ContinuedFixedPoints::continued.
// Also times the calls. A measurement tool, not a test.
//
//   c++ -std=c++17 -O3 -Icdx/include bench/continuation_count.cpp cdx/build/libcdx_core.a \
//       -o bench/out/continuation_count && bench/out/continuation_count
#include "cdx/rational.hpp"
#include "cdx/renderer.hpp"

#include <chrono>
#include <cstdio>
#include <string>

using namespace cdx;

int main() {
    for (int n : {5, 8}) {
        const std::string src = "z - a*(z^" + std::to_string(n) + "-1)/(" + std::to_string(n) + "*z^" +
                                std::to_string(n - 1) + ")";
        const RationalMap m = RationalMap::from_expression(src, "a", {}, "relaxed_newton");
        const Viewport v{{static_cast<double>(n), 0.0}, static_cast<double>(n), 130};
        long pixels = 0, crit_cont = 0, fp_cont = 0;
        double t_crit = 0.0, t_fp = 0.0, t_poles = 0.0;
        std::size_t n_ordinary = 0, n_expected = 0;
        for (int col = 0; col < v.resolution; col += 13) {
            std::vector<Cplx> prev_ordinary, prev_fp;
            for (int row = 0; row < v.resolution; ++row) {
                const Cplx a = v.coord(col, row);
                auto t0 = std::chrono::steady_clock::now();
                const ContinuedCriticalPoints cc = m.distinct_critical_points_continued(a, prev_ordinary);
                auto t1 = std::chrono::steady_clock::now();
                const ContinuedFixedPoints cf = m.fixed_points_continued(a, prev_fp);
                auto t2 = std::chrono::steady_clock::now();
                (void)m.pole_locations(a);
                auto t3 = std::chrono::steady_clock::now();
                t_crit += std::chrono::duration<double>(t1 - t0).count();
                t_fp += std::chrono::duration<double>(t2 - t1).count();
                t_poles += std::chrono::duration<double>(t3 - t2).count();
                if (row > 0) {   // row 0 of a column is cold by design
                    ++pixels;
                    crit_cont += cc.continued ? 1 : 0;
                    fp_cont += cf.continued ? 1 : 0;
                }
                n_ordinary = cc.ordinary_raw.size();
                n_expected = static_cast<std::size_t>(2 * n - 2);
                prev_ordinary = cc.ordinary_raw;
                prev_fp = cf.raw;
            }
        }
        const double calls = static_cast<double>(pixels + 10);
        std::printf("n=%d: critical points continued at %ld of %ld pixels (%.1f%%); fixed points at %ld (%.1f%%)\n",
                    n, crit_cont, pixels, 100.0 * crit_cont / pixels, fp_cont, 100.0 * fp_cont / pixels);
        std::printf("      predictor carries %zu ordinary critical points; the derivative numerator has degree %zu\n",
                    n_ordinary, n_expected);
        std::printf("      per pixel: critical points %.1f us, fixed points %.1f us, one pole_locations call %.2f us\n",
                    1e6 * t_crit / calls, 1e6 * t_fp / calls, 1e6 * t_poles / calls);
    }
    return 0;
}
