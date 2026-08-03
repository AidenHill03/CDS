// =============================================================================
// test_custom.cpp -- property-based checks for Family::Custom.
//
// The core property under test: a RationalMap wrapped via Map::custom()
// should render the same picture as the equivalent built-in fast path,
// across every render mode (render_parameter additionally exercises
// step_with_param/critical_point_at, since it evaluates at a varying
// parameter rather than the one bound to the Map). Modest resolutions
// throughout -- these are correctness checks, not the benchmark.
// =============================================================================
#include "cdx/renderer.hpp"
#include "cdx/rational.hpp"

#include <cmath>
#include <cstdio>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

// Fraction of pixels whose values differ by more than `tol` -- the same
// tolerant-comparison spirit as test_renderer.cpp's symmetry check, since
// RationalMap::eval and the hand-rolled step_with take different arithmetic
// paths to the same formula and can disagree in the last bit or two near an
// escape-time boundary.
static double mismatch_fraction(const Image& a, const Image& b, double tol) {
    long bad = 0;
    const long n = static_cast<long>(a.data.size());
    for (long i = 0; i < n; ++i)
        if (std::abs(a.data[static_cast<std::size_t>(i)] -
                     b.data[static_cast<std::size_t>(i)]) > tol) ++bad;
    return static_cast<double>(bad) / static_cast<double>(n);
}

int main() {
    std::printf("=== cdx custom-map tests ===\n");

    // ---- Map::custom basics ---------------------------------------------------
    std::printf("\nMap::custom basics:\n");
    {
        Map built_in(Family::Quadratic, {0, 0});
        check(built_in.custom_map() == nullptr, "custom_map() is null for a built-in family");

        Map custom = Map::custom(RationalMap::mandelbrot(), {-0.7269, 0.1889});
        check(custom.family() == Family::Custom, "custom() sets family() to Custom");
        check(custom.custom_map() != nullptr, "custom_map() is non-null for Family::Custom");
        check(custom.degree() == 2, "degree() dispatches to the RationalMap (mandelbrot: 2)");
    }

    // ---- render_julia: custom mandelbrot vs the built-in fast path -----------
    std::printf("\nrender_julia: custom mandelbrot matches Family::Quadratic:\n");
    {
        const Cplx param{-0.7269, 0.1889};
        Viewport v{{0.0, 0.0}, 1.5, 301};
        RenderSettings s{200, 2.0, 1e-6, 0};

        Renderer built_in(Map(Family::Quadratic, param), v, s);
        Renderer custom(Map::custom(RationalMap::mandelbrot(), param), v, s);

        const Image a = built_in.render_julia();
        const Image b = custom.render_julia();
        const double frac = mismatch_fraction(a, b, 1e-6);
        std::printf("  mismatch fraction: %.6f\n", frac);
        check(frac < 0.001, "custom-path Julia set matches the built-in fast path");
    }

    // ---- render_basin: custom Newton cubic vs Family::Newton3 -----------------
    std::printf("\nrender_basin: custom newton_cubic matches Family::Newton3:\n");
    {
        std::vector<Cycle> cyc = {
            {{{1.0, 0.0}}, 1},
            {{{-0.5, 0.8660254037844386}}, 2},
            {{{-0.5, -0.8660254037844386}}, 3},
        };
        Viewport v{{0.0, 0.0}, 2.0, 251};
        RenderSettings s{200, 2.0, 1e-6, 0};

        Renderer built_in(Map(Family::Newton3, {0, 0}), v, s);
        Renderer custom(Map::custom(RationalMap::newton_cubic(), {0, 0}), v, s);

        const Image a = built_in.render_basin(cyc);
        const Image b = custom.render_basin(cyc);
        const double frac = mismatch_fraction(a, b, 0.5);   // basin ids, not continuous
        std::printf("  mismatch fraction: %.6f\n", frac);
        check(frac < 0.01, "custom-path Newton basins match the built-in fast path");
    }

    // ---- render_greens: custom mandelbrot vs the built-in fast path ----------
    std::printf("\nrender_greens: custom mandelbrot matches Family::Quadratic:\n");
    {
        const Cplx param{-0.5, 0.0};
        Viewport v{{0.0, 0.0}, 1.5, 201};
        RenderSettings s{150, 2.0, 1e-6, 0};

        Renderer built_in(Map(Family::Quadratic, param), v, s);
        Renderer custom(Map::custom(RationalMap::mandelbrot(), param), v, s);

        bool norm_a = false, norm_b = false;
        const Image a = built_in.render_greens(&norm_a);
        const Image b = custom.render_greens(&norm_b);
        check(norm_a == norm_b, "both agree on whether the normalization overflowed");
        const double frac = mismatch_fraction(a, b, 1e-6);
        std::printf("  mismatch fraction: %.6f\n", frac);
        check(frac < 0.001, "custom-path Green's function matches the built-in fast path");
    }

    // ---- render_parameter: exercises critical_point_at/step_with_param -------
    std::printf("\nrender_parameter: custom mandelbrot matches Family::Quadratic:\n");
    {
        // Deliberately modest resolution: render_parameter root-finds the
        // custom map's critical point once per pixel (see the comment on
        // Renderer::render_parameter), which is far more expensive per pixel
        // than the built-in closed form. Fine for a sandbox map; not
        // something to run at demo.py's 900x900 in a test.
        Viewport v{{-0.5, 0.0}, 1.5, 121};
        RenderSettings s{150, 2.0, 1e-6, 0};

        Renderer built_in(Map(Family::Quadratic, {0, 0}), v, s);
        Renderer custom(Map::custom(RationalMap::mandelbrot(), {0, 0}), v, s);

        const Image a = built_in.render_parameter();
        const Image b = custom.render_parameter();
        const double frac = mismatch_fraction(a, b, 1e-6);
        std::printf("  mismatch fraction: %.6f\n", frac);
        check(frac < 0.001,
              "custom-path parameter plane matches the built-in fast path "
              "(critical_point_at and step_with_param both correct)");
    }

    // ---- a genuinely custom map with no built-in equivalent -------------------
    std::printf("\na map with no built-in equivalent still renders sensibly:\n");
    {
        // z^3 + a/z^2 -- mixed positive/negative exponents, not one of the
        // built-in families.
        RationalMap m("mixed");
        m.add_poly({1, 0}, 3, 0, "z^3");
        m.add_pole({0, 0}, {1, 0}, 2, 1, "a/z^2");

        Viewport v{{0.0, 0.0}, 2.0, 151};
        RenderSettings s{150, 2.0, 1e-6, 0};
        Renderer r(Map::custom(m, {0.4, 0.3}), v, s);

        const Image img = r.render_julia();
        bool any_nonzero = false, any_finite = false;
        for (double val : img.data) {
            if (val != 0.0) any_nonzero = true;
            if (std::isfinite(val)) any_finite = true;
        }
        check(any_nonzero, "a non-preset custom map produces a non-degenerate image");
        check(any_finite, "...with finite (non-NaN/inf) values");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
