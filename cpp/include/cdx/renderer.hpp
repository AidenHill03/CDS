// =============================================================================
// cdx/renderer.hpp -- Complex Dynamics core rendering library.
//
// Standalone C++ port of the MATLAB/MEX kernels. No MATLAB dependency; this
// is the numerics core that the Python bindings and eventual Qt UI sit on.
//
// Design notes
//   * Map is a small value type: family + parameter. Copyable, cheap.
//   * Renderer holds map + viewport + settings and produces Images. Keeping
//     it an object (rather than free functions with long argument lists)
//     gives a natural home for state that later features need to retain --
//     notably a high-precision reference orbit for perturbation-based deep
//     zoom, which is the flagship capability this port is meant to unlock.
//   * The inner loops hand-roll real/imaginary arithmetic instead of using
//     std::complex. libstdc++/libc++ operator* carries inf/nan branch
//     handling that is measurably slower, and profiling on the MATLAB side
//     established that this arithmetic IS the bottleneck.
//   * All classification is sphere-aware: infinity is an ordinary point,
//     and basin membership uses the chordal metric.
// =============================================================================
#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace cdx {

using Cplx = std::complex<double>;

// -----------------------------------------------------------------------------
// Map families the core implements. Adding a family means adding an enumerator,
// a case in Map::step, its degree, and its critical point.
// -----------------------------------------------------------------------------
enum class Family {
    Quadratic,   // z^2 + a          (Mandelbrot / quadratic Julia)
    Cubic,       // z^3 + a
    Quintic,     // z^5 + a
    McMullen2,   // z^2 + a/z^2
    McMullen3,   // z^3 + a/z^3
    Newton3      // Newton map of z^3 - 1  (parameter unused)
};

std::string to_string(Family f);
bool        family_from_string(const std::string& s, Family& out);

// -----------------------------------------------------------------------------
// A concrete map: a family with its parameter bound.
// -----------------------------------------------------------------------------
class Map {
public:
    Map() = default;
    Map(Family f, Cplx param) : family_(f), param_(param) {}

    Family family() const { return family_; }
    Cplx   param()  const { return param_; }
    void   set_param(Cplx p) { param_ = p; }

    // Degree of the map as a rational map of the sphere.
    int degree() const;

    // One iteration. Hot path: takes and returns components by reference to
    // avoid constructing complex temporaries in the inner loop.
    void step(double& zr, double& zi) const;

    // Same, but with an explicit parameter -- used by parameter-plane
    // rendering, where the parameter varies per pixel.
    static void step_with(Family f, double pr, double pi, double& zr, double& zi);

    // The critical point whose orbit determines parameter-plane membership,
    // for the given parameter value. For z^n + a this is always 0; for the
    // McMullen families the critical points satisfy z^(2n) = a and move with
    // the parameter (they are rotations of one another and share escape
    // behaviour, so one representative suffices).
    static Cplx critical_point(Family f, Cplx param);

private:
    Family family_ = Family::Quadratic;
    Cplx   param_  = {0.0, 0.0};
};

// -----------------------------------------------------------------------------
// The square view window, in the plane the map acts on.
// -----------------------------------------------------------------------------
struct Viewport {
    Cplx   center     {0.0, 0.0};
    double scale      = 1.5;   // half-width
    int    resolution = 800;   // pixels per side

    double pixel_size() const {
        return resolution > 1 ? 2.0 * scale / (resolution - 1) : 0.0;
    }
    // Data coordinate of a pixel (col, row), row 0 at the bottom.
    Cplx coord(int col, int row) const {
        const double s = pixel_size();
        return { center.real() - scale + s * col,
                 center.imag() - scale + s * row };
    }
};

// -----------------------------------------------------------------------------
// Iteration/tolerance settings shared by the render modes.
// -----------------------------------------------------------------------------
struct RenderSettings {
    int    max_iter      = 200;
    double escape_radius = 2.0;
    double tol           = 1e-6;   // chordal tolerance for basin membership
    int    threads       = 0;      // 0 = hardware concurrency
};

// -----------------------------------------------------------------------------
// An attracting cycle: the points of the cycle plus the basin id to label
// pixels with. A fixed point is simply a cycle of length one. Points may be
// infinite (use std::numeric_limits<double>::infinity() in the real part).
// -----------------------------------------------------------------------------
struct Cycle {
    std::vector<Cplx> points;
    int               id = 1;
};

// -----------------------------------------------------------------------------
// Render result. Row-major, row 0 at the bottom (matching Viewport::coord), so
// callers that want image-style top-down order should flip.
// -----------------------------------------------------------------------------
struct Image {
    int                 width  = 0;
    int                 height = 0;
    std::vector<double> data;

    Image() = default;
    Image(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h, 0.0) {}

    double&       at(int col, int row)       { return data[static_cast<size_t>(row) * width + col]; }
    const double& at(int col, int row) const { return data[static_cast<size_t>(row) * width + col]; }
};

// -----------------------------------------------------------------------------
// Chordal (spherical) distance. Handles infinite arguments, so infinity is an
// ordinary point rather than an escape condition:
//     d(z,w)   = 2|z-w| / sqrt((1+|z|^2)(1+|w|^2))
//     d(z,inf) = 2 / sqrt(1+|z|^2)
// -----------------------------------------------------------------------------
double chordal_distance(double zr, double zi, double wr, double wi);

// -----------------------------------------------------------------------------
// The renderer.
// -----------------------------------------------------------------------------
class Renderer {
public:
    Renderer() = default;
    Renderer(Map m, Viewport v, RenderSettings s)
        : map_(m), view_(v), settings_(s) {}

    // --- configuration -------------------------------------------------------
    const Map&            map()      const { return map_; }
    const Viewport&       viewport() const { return view_; }
    const RenderSettings& settings() const { return settings_; }

    void set_map(const Map& m)                 { map_ = m; }
    void set_viewport(const Viewport& v)       { view_ = v; }
    void set_settings(const RenderSettings& s) { settings_ = s; }

    // Convenience: zoom the viewport toward a point by a factor (>1 zooms in).
    void zoom(Cplx target, double factor);

    // --- render modes --------------------------------------------------------

    // Escape-time Julia set of the bound map. Value is the smooth escape
    // count n + 1 - log(log|z|)/log 2, or 0 for orbits that never escaped.
    Image render_julia() const;

    // Parameter plane: each pixel is a parameter value, the orbit starts at
    // that map's critical point. Reproduces the Mandelbrot/multibrot sets and
    // the McMullenbrot. The bound parameter is ignored.
    Image render_parameter() const;

    // Basin classification against a set of attracting cycles, in the chordal
    // metric. Value is the cycle id, or 0 for unresolved pixels.
    Image render_basin(const std::vector<Cycle>& cycles) const;

    // Green's function (dynamical potential). Accumulates log(max(|z|,1))
    // over the orbit and divides by degree^max_iter.
    //
    // NOTE: degree^max_iter overflows double for even moderate max_iter
    // (2^200 is far outside range). When it would overflow, the result is
    // returned UNNORMALIZED and `normalized` is set false -- values remain
    // comparable within one image but not across different max_iter.
    Image render_greens(bool* normalized = nullptr) const;

    // Deepest zoom the double-precision grid can still resolve: below this
    // half-width, neighbouring pixels round to the same double and the image
    // degenerates. Beating it requires extended precision or perturbation
    // theory (see the development plan).
    double precision_floor() const;

private:
    // Runs `body(col)` for every column, across the configured thread count.
    template <typename F>
    void parallel_columns(F body) const;

    Map            map_;
    Viewport       view_;
    RenderSettings settings_;
};

}  // namespace cdx
