// =============================================================================
// cdx/renderer.hpp -- Complex Dynamics core rendering library.
//
// Standalone C++ port of the MATLAB/MEX kernels. No MATLAB dependency; this
// is the numerics core that the Python bindings and eventual Qt UI sit on.
//
// Design notes
//   * Map is a small value type: family + parameter. Copyable, cheap -- even
//     for Family::Custom, which carries a RationalMap via shared_ptr rather
//     than by value, so copying a Map is always just a refcount bump.
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

#include "cdx/rational.hpp"

#include <atomic>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cdx {

using Cplx = std::complex<double>;

// -----------------------------------------------------------------------------
// Map families the core implements. Adding a built-in family means adding an
// enumerator, a case in Map::step_with, its degree, and its critical point.
//
// Custom is different: it carries no fixed formula of its own. A Map with
// family() == Custom holds a RationalMap (see Map::custom()) and every
// family-keyed operation defers to that object's eval/deriv/critical_points
// instead. It exists so a user-edited RationalMap can be rendered through
// exactly the same Renderer as the built-ins, without the built-ins paying
// any indirection cost -- see the note on step_with_param below.
// -----------------------------------------------------------------------------
enum class Family {
    Quadratic,   // z^2 + a          (Mandelbrot / quadratic Julia)
    Cubic,       // z^3 + a
    Quintic,     // z^5 + a
    McMullen2,   // z^2 + a/z^2
    McMullen3,   // z^3 + a/z^3
    Newton3,     // Newton map of z^3 - 1  (parameter unused)
    Custom       // a general RationalMap; see Map::custom()
};

std::string to_string(Family f);
bool        family_from_string(const std::string& s, Family& out);

// Detects whether `m` is STRUCTURALLY one of the built-in Family shapes
// above (z^n + a for n in {2,3,5}, z^n + a/z^n for n in {2,3}, or the fixed
// Newton3 formula) -- see renderer.cpp for the exact per-shape checks. A
// hit means a Custom map wrapping `m` can render through step_with's
// native formula directly, at zero extra per-iteration cost over an
// ordinary built-in Family render, instead of the generic (if now
// hand-rolled and reasonably fast -- see CompiledMap) RationalMap::compile
// path. A miss (nullopt) is always safe, just slower: this is a pure
// performance dispatch, never a correctness one. Only ENABLED terms are
// considered, matching what eval() actually computes -- a disabled term
// that happens to complete a recognizable shape does not count, since it
// contributes nothing to the map's actual behaviour.
std::optional<Family> recognize_family(const RationalMap& m);

// -----------------------------------------------------------------------------
// A concrete map: a family with its parameter bound, or (for Custom) a
// RationalMap with its parameter bound.
// -----------------------------------------------------------------------------
class Map {
public:
    Map() = default;
    Map(Family f, Cplx param) : family_(f), param_(param) {}

    // Wraps a RationalMap as a Family::Custom map. `param` is the bound
    // value of the map's own free parameter `a`, used by step()/degree()
    // exactly as param_ is for a built-in family; render_parameter varies
    // it per pixel regardless of what is bound here (see step_with_param).
    static Map custom(RationalMap m, Cplx param = {0.0, 0.0});

    Family family() const { return family_; }
    Cplx   param()  const { return param_; }
    void   set_param(Cplx p) { param_ = p; }

    // The wrapped RationalMap, or nullptr unless family() == Custom.
    const RationalMap* custom_map() const { return custom_.get(); }

    // Degree of the map as a rational map of the sphere.
    int degree() const;

    // One iteration. Hot path: takes and returns components by reference to
    // avoid constructing complex temporaries in the inner loop.
    void step(double& zr, double& zi) const;

    // Same, but with an explicit parameter -- used by parameter-plane
    // rendering, where the parameter varies per pixel. Static and keyed
    // purely on Family because every built-in family's formula is fully
    // determined by the enum value; it does NOT handle Custom, which has no
    // formula outside its bound RationalMap instance. render_parameter uses
    // the instance method step_with_param below instead, which does.
    static void step_with(Family f, double pr, double pi, double& zr, double& zi);

    // The critical point whose orbit determines parameter-plane membership,
    // for the given parameter value. For z^n + a this is always 0; for the
    // McMullen families the critical points satisfy z^(2n) = a and move with
    // the parameter (they are rotations of one another and share escape
    // behaviour, so one representative suffices). Same caveat as step_with:
    // static, built-in-only. Use critical_point_at for Custom.
    static Cplx critical_point(Family f, Cplx param);

    // Instance counterparts of step_with/critical_point that also handle
    // Custom, by dispatching to the bound RationalMap's eval/critical_points
    // instead of the static per-family formula. These are what
    // Renderer::render_parameter calls; for a built-in family they just
    // forward to the static functions above; for a Custom map they are the
    // ONLY way to evaluate it at a parameter other than the one bound to
    // this instance, since a Custom map's shape lives on the object, not in
    // the Family enum.
    //
    // For Custom, critical_point_at picks the first critical point
    // RationalMap::critical_points returns (see that method's documented
    // scope and known limitation), or {0,0} if the map has none.
    Cplx critical_point_at(Cplx p) const;
    void step_with_param(Cplx p, double& zr, double& zi) const;

private:
    Family family_ = Family::Quadratic;
    Cplx   param_  = {0.0, 0.0};
    std::shared_ptr<const RationalMap> custom_;   // non-null iff family_ == Custom
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
    //
    // Every mode below takes an optional `cancel`: a caller-owned flag,
    // checked ONCE PER COLUMN (not just at the start and end -- a render
    // that only checked at the boundaries would still block a caller for
    // the full render duration, which defeats the purpose). ~1000 checks
    // per render is free relative to the per-pixel work; see
    // parallel_columns. When cancel is set mid-render, the render loop
    // returns early with a PARTIAL image; the caller is expected to discard
    // it, not display it. cancel may be nullptr (the default), meaning
    // "never cancel," identical to the pre-cancellation behaviour.

    // Escape-time Julia set of the bound map. Value is the smooth escape
    // count n + 1 - log(log|z|)/log 2, or 0 for orbits that never escaped.
    Image render_julia(const std::atomic<bool>* cancel = nullptr) const;

    // Parameter plane: each pixel is a parameter value, the orbit starts at
    // that map's critical point. Reproduces the Mandelbrot/multibrot sets and
    // the McMullenbrot. The bound parameter is ignored.
    Image render_parameter(const std::atomic<bool>* cancel = nullptr) const;

    // Basin classification against a set of attracting cycles, in the chordal
    // metric. Value is the cycle id, or 0 for unresolved pixels.
    //
    // If `iterations` is given, it is replaced with an Image the same size
    // as the result, holding the iteration count (n+1, matching
    // render_julia's own n+1 convention) each pixel took to resolve -- for
    // BASIN SHADING (hue = basin id from the primary result, brightness =
    // this convergence speed; see app/colour.py's colour_basin). No extra
    // computation: the per-pixel loop already counts iterations to decide
    // when to stop, this just retains that count instead of discarding it.
    // For an UNRESOLVED pixel (primary result 0), the count is however many
    // iterations were actually run (max_iter if the orbit neither resolved
    // nor blew up, fewer if it hit is_bad first) -- not a meaningful
    // "convergence speed" since nothing converged, callers should treat an
    // unresolved pixel's shading as undefined and colour it by the primary
    // result being 0 instead (exactly what colour_basin already does).
    Image render_basin(const std::vector<Cycle>& cycles, Image* iterations = nullptr,
                       const std::atomic<bool>* cancel = nullptr) const;

    // Green's function (dynamical potential), G_f(z) = lim_{n->inf}
    // d^-n log+|f^n(z)|. The pixel is an INITIAL CONDITION on the
    // DYNAMICAL plane -- the orbit starts AT the pixel, under the map's
    // own BOUND parameter map_.param() (fixed for the whole render,
    // exactly like render_julia). When the orbit escapes at iteration n,
    // the result is log|z_n| / degree^n; non-escaping pixels are exactly
    // 0. Normalizing at each pixel's OWN escape iteration (not against a
    // single degree^max_iter for the whole image) is what makes this the
    // actual escape-rate potential rather than a near-constant field --
    // see the .cpp for the measured difference.
    Image render_greens(const std::atomic<bool>* cancel = nullptr) const;

    // The FAMILY ESCAPE-RATE FUNCTION on the PARAMETER plane -- for the
    // quadratic family this is G_M(c), the Mandelbrot set's own Green's
    // function. This is a DIFFERENT function on a DIFFERENT space from
    // render_greens above, and the two must never be confused or treated
    // as reparameterizations of each other:
    //   - the pixel is a PARAMETER (like render_parameter), not an
    //     initial condition -- map_.param() is IGNORED, exactly as
    //     render_parameter ignores it;
    //   - the orbit starts at THAT parameter's CRITICAL POINT (again like
    //     render_parameter -- Fatou: the critical orbit is what actually
    //     determines escape/connectedness), not at the pixel itself;
    //   - the accumulated quantity is the escape rate of THAT critical
    //     orbit (render_greens' own log(max(|z|,1)) accumulation,
    //     substituted for render_parameter's escape-time test -- this is
    //     the one piece genuinely shared between the two parents).
    //
    // Structurally: render_parameter's per-pixel critical-point/step
    // dispatch (three paths -- recognized/critical-points-fixed/general,
    // see render_parameter's own comment) with render_greens' escape-rate-
    // potential body (per-pixel normalization at its own escape iteration,
    // see render_greens' own comment) substituted for render_parameter's
    // escape test. NOT a call to either existing method with different
    // arguments.
    //
    // degree, unlike a per-pixel critical point, is trustworthy to
    // compute ONCE for the whole render even here: RationalMap::degree(
    // Cplx a) is purely structural and never actually reads `a` (see its
    // own implementation), so it cannot differ from one parameter-plane
    // pixel to the next.
    Image render_parameter_greens(const std::atomic<bool>* cancel = nullptr) const;

    // Deepest zoom the double-precision grid can still resolve: below this
    // half-width, neighbouring pixels round to the same double and the image
    // degenerates. Beating it requires extended precision or perturbation
    // theory (see the development plan).
    double precision_floor() const;

private:
    // Runs `body(col)` for every column, across the configured thread count.
    // Checks `cancel` (if non-null) once per column, on every worker thread;
    // a set flag stops that thread taking any further columns. Does not
    // itself clear or own `cancel` -- purely a check.
    template <typename F>
    void parallel_columns(F body, const std::atomic<bool>* cancel = nullptr) const;

    Map            map_;
    Viewport       view_;
    RenderSettings settings_;
};

}  // namespace cdx
