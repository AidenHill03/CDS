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

    // Stage 1's polynomial_escape_certified, family-aware -- true for the
    // three polynomial built-ins (Quadratic/Cubic/Quintic: z^n+a, no
    // poles), false for the three rational built-ins (McMullen2/McMullen3:
    // poles at 0 and infinity's own chart; Newton3: a pole at the origin),
    // and for Custom delegates to polynomial_escape_certified(*custom_map())
    // (see cdx/analysis.hpp -- implemented in renderer.cpp, which includes
    // analysis.hpp itself, rather than pulled in here, to avoid a header
    // cycle: analysis.hpp already includes renderer.hpp for Cycle/Image).
    bool escape_certified() const;

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
// Stage 3's two selectable Green's-function potentials for a RATIONAL map's
// dynamical-plane render_greens (and, per-critical-orbit, render_parameter_
// greens) -- see render_greens' own doc comment for what each means
// mathematically. Meaningless for a CERTIFIED polynomial, which ignores this
// entirely and keeps its own existing fast path (see Map::escape_certified) --
// where PRAGMATIC and CONFORMAL-Boettcher already coincide exactly.
// -----------------------------------------------------------------------------
enum class GreensPotential {
    Pragmatic,   // smooth chordal convergence-rate toward the reached attractor
    Conformal,   // log|phi(z)| -- Boettcher (superattracting) or Koenigs
                // (geometrically attracting) coordinate, estimated numerically;
                // falls back to Pragmatic (flagged, see render_greens) when the
                // orbit's own local behaviour is too close to parabolic to tell
};

// -----------------------------------------------------------------------------
// Which closed-form recipe render_parameter_period (below) uses to compute
// the PER-PIXEL "tracked" critical points to seed period detection from --
// see that method's own doc comment for why this can't just be RationalMap::
// distinct_critical_points(a) (that returns every critical point the map
// has, including ones structurally unrelated to the specific orbit a
// literature figure tracks, e.g. z=0 for the relaxed-Newton family below --
// see per_seed_outcomes' own doc comment for a measured example of that
// going wrong). One enumerator per family with a known closed form,
// mirroring how Family itself dispatches built-in shapes -- NOT a plugin
// registry (see modules/README.md's own rule against one until at least
// three concrete needs exist); add a case here, and its own branch in
// render_parameter_period, when a second family's own closed form is
// needed (e.g. a KLS bicritical map's two critical points).
// -----------------------------------------------------------------------------
enum class CriticalPointFamily {
    // Relaxed Newton of p(z) = z^n - 1: N_a(z) = z - a(z^n-1)/(n z^{n-1}) =
    // ((n-a)z^n + a) / (n z^{n-1}). The n FREE critical points (excluding
    // the structural one at z=0, which always maps directly to infinity
    // and is tracked separately, if at all -- see render_parameter_period)
    // are z_c(a) = (a(n-1)/(n-a))^(1/n) * omega^j, j=0..n-1, omega =
    // e^(2*pi*i/n). `n` is passed to render_parameter_period explicitly
    // (it is a FAMILY-structural choice, not the map's own single active
    // dynamical parameter `a`, and the map's own degree(a) equals n for
    // this family -- see render_parameter_period's own doc comment for
    // how a caller that only has a RationalMap, not the n it was built
    // with, can still recover it).
    RelaxedNewtonPower,
};

// -----------------------------------------------------------------------------
// How render_parameter's own rational path (below) combines MULTIPLE free
// critical orbits' smooth chordal rates into the one value a pixel gets --
// see render_parameter's own doc comment for the full derivation. This
// enum's own name and case shapes deliberately echo an EARLIER, reverted
// attempt at exactly this idea (git history: "AllCaptured/FastestCapture/
// PerCritical"), which was retired because it only ever asked "does the
// critical orbit escape to infinity" -- see render_parameter's own doc
// comment for why THIS version (sphere-aware, any attractor, not just
// infinity) fixes the gap that made the earlier one useless on a Nova-like
// family. Meaningless for a CERTIFIED polynomial (which has, at most, ONE
// finite critical point that ever distinguishes parameter pixels -- see
// render_parameter's own comment for why -- so there is nothing for a
// strategy to combine) and for a non-Custom built-in RATIONAL family
// (McMullen2/McMullen3/Newton3), which keeps using its existing single-
// representative critical point (see Map::critical_point's own doc
// comment) for the same reason Map::critical_point already gave: every
// finite critical point of that SPECIFIC family shares escape/capture
// behaviour by symmetry, so all three strategies degenerate to the
// identical single-orbit result there. Only a genuinely multi-critical
// Custom rational map (RationalMap::distinct_critical_points returning
// more than one point) is where these three actually differ.
//
// A DELIBERATE TUNING KNOB, not a settled design choice -- exposed as an
// app.settings.Settings field (parameter_strategy) specifically so it can
// be compared in-app; Slowest is the current default, not a claim that it
// is the "right" one.
// -----------------------------------------------------------------------------
enum class ParameterStrategy {
    // 0 iff AT LEAST ONE free critical orbit never settles onto any
    // complete attractor within max_iter (a genuine "still open/
    // undetermined here" signal, the sphere-aware analog of "in the
    // filled/connectedness set" -- matching every other mode's own "0 =
    // filled/unresolved" convention); otherwise the SLOWEST-to-settle
    // orbit's own smooth value -- i.e. whichever one took the longest to
    // resolve. DEFAULT: large near the connectedness-locus boundary
    // (where at least one critical orbit is still slowly deciding which
    // attractor it belongs to), small deep in a region every critical
    // orbit resolves into quickly -- the multi-critical "Mandelbrot glow"
    // analog.
    Slowest,
    // The pixel is driven by whichever free critical orbit resolves
    // FASTEST (0 only if NONE of them resolve at all) -- "how quickly
    // does structure emerge here," rather than "is everything nice here."
    Fastest,
    // Track exactly ONE distinct critical point's own orbit (selected by
    // render_parameter's own `critical_index`, into the SAME distinct_
    // critical_points(a) ordering for every pixel -- clamped into range
    // per pixel, since how many free critical points there are can itself
    // vary with `a`), ignoring the rest entirely -- gives one
    // independent, classical-Mandelbrot-style render per critical point,
    // for a caller that wants to let the user flip between them rather
    // than see them pre-combined.
    PerCritical,
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

    // Julia set of the bound map -- TWO PATHS, chosen internally via
    // Map::escape_certified() (Stage 1's polynomial_escape_certified,
    // family-aware):
    //
    //   CERTIFIED POLYNOMIAL (no poles, degree >= 2 -- infinity is ALWAYS
    //   superattracting there, so a fixed |z| > escape_radius test is a
    //   provably forward-invariant trap): today's escape-time fast path,
    //   UNCHANGED -- `cycles`/`labels` are ignored entirely. Value is the
    //   smooth escape count n + 1 - log(log|z|)/log 2, or 0 for orbits
    //   that never escaped.
    //
    //   RATIONAL (has poles -- infinity may be repelling, attracting, or
    //   not even fixed; a fixed escape radius is not a validated trap in
    //   general): sphere-aware classification against `cycles` (see
    //   render_basin below -- SAME chordal-metric-against-found-attractors
    //   idea, including infinity as an ordinary point when it's one of the
    //   entries), structured the same way. Value is a SMOOTH chordal
    //   analog of escape-time -- a continuous "approach rate" toward
    //   whichever attractor the orbit reached, log-log-interpolated
    //   between the iteration where the chordal distance was still above
    //   `settings().tol` and the one where it first dropped below --
    //   or 0 for pixels that never resolved (matching the polynomial
    //   path's own "0 = never escaped/resolved" convention). If `labels`
    //   is given, it is replaced with an Image the same size as the
    //   result, holding which attractor (Cycle::id) each pixel reached (0
    //   = unresolved) -- for PER-BASIN-HUE coloring, the same (label,
    //   rate) split render_basin already returns as (primary, iterations).
    //   escape_radius plays NO role anywhere in this path.
    //
    // NO escape_radius anywhere in the rational path -- unlike the
    // certified-polynomial path, whose SET is R-invariant already (any
    // large-enough R gives the identical escaping set, hence identical
    // classification), a rational map has no such blanket guarantee, so
    // this path never reads settings().escape_radius at all: the Julia
    // set is an invariant of the MAP, not of an arbitrary numeric cutoff.
    Image render_julia(const std::atomic<bool>* cancel = nullptr,
                       const std::vector<Cycle>& cycles = {}, Image* labels = nullptr) const;

    // Parameter plane: each pixel is a parameter value. The bound parameter
    // is ignored. Reproduces the Mandelbrot/multibrot sets and the
    // McMullenbrot. TWO PATHS, same certification dispatch as render_julia/
    // render_greens (Map::escape_certified):
    //
    //   CERTIFIED POLYNOMIAL: UNCHANGED from before this milestone --
    //   escape_radius-based escape-TIME, orbit starts at the map's own
    //   SINGLE finite critical point (every finite critical point of a
    //   z^d+a-shaped family coincides at 0; infinity is trivially,
    //   uninformatively superattracting there, deliberately excluded --
    //   see Map::critical_point's own doc comment). `strategy`/
    //   `critical_index` are ignored entirely -- there is nothing for a
    //   strategy to combine with only one orbit. NOTE: this mode is a
    //   VISUALIZATION, not a set-membership computation -- unlike the
    //   Julia SET or Green's function, which are invariants of the map
    //   and so must never depend on escape_radius (see their own doc
    //   comments and CLAUDE.md), the escape-TIME shading here is
    //   explicitly an artifact of where the cutoff is drawn, the same way
    //   classical Mandelbrot-set escape-time renders always have been.
    //   escape_radius is a legitimate tuning knob for the POLYNOMIAL path
    //   specifically; the milestone's own escape-radius-invariance
    //   acceptance test applies to Julia/Green's, deliberately not here.
    //
    //   RATIONAL: escape-radius-free and MULTI-critical -- retires the
    //   PRIOR monocritical |z|>escape_radius test this path used to share
    //   with the polynomial one. HISTORY, not assumed but confirmed by
    //   reading it: an EARLIER multi-critical rational attempt here (the
    //   "AllCaptured/FastestCapture/PerCritical" ParameterStrategy this
    //   enum's own name and cases deliberately echo) was tried and
    //   reverted, because it only ever asked "does the critical orbit
    //   escape to infinity" -- useless for a family (Nova-like ones)
    //   whose interesting parameter structure is FINITE attracting
    //   cycles, which just ran every pixel to max_iter and returned a
    //   uniform plane. THIS path fixes exactly that gap: every free
    //   critical point at that pixel's parameter (RationalMap::
    //   distinct_critical_points, UNFILTERED -- deliberately including
    //   any pole-related one, unlike render_parameter_period's own
    //   family-specific "free" subset, since a pole-related critical
    //   orbit failing to settle is exactly as informative here as any
    //   other) has its own orbit classified against the FULL sphere-
    //   aware attractor set at that parameter (complete_attractors_
    //   from_seeds -- fixed point, cycle, OR infinity, not just infinity
    //   alone -- see that function's own doc comment), via the SAME
    //   classify_rational_orbit smooth-chordal-rate machinery render_
    //   julia_rational/render_greens_rational/render_parameter_greens_
    //   rational already use (nothing bespoke here either). `strategy`
    //   (ParameterStrategy, see its own doc comment) combines the
    //   resulting per-critical-point smooth values into the one value
    //   this pixel gets; `critical_index` only matters for
    //   ParameterStrategy::PerCritical. Every strategy's own "0" case is
    //   exactly "nothing usable to report" (no tracked critical point at
    //   all, or -- for Slowest -- at least one that never settles at
    //   all), matching every other mode's own "0 = filled/unresolved"
    //   convention -- see ParameterStrategy's own doc comment for exactly
    //   what counts as 0 under each strategy. COLORS through the SAME
    //   color_escape_time palette/scaling pipeline the polynomial path's
    //   own output already goes through -- no bespoke scheme, no shape
    //   change to what this method returns (a plain Image, same as
    //   before) -- app/session.py's own dispatch needs no changes either.
    Image render_parameter(const std::atomic<bool>* cancel = nullptr,
                           ParameterStrategy strategy = ParameterStrategy::Slowest,
                           int critical_index = 0) const;

    // Parameter_basin: each pixel is a parameter value a; the pixel value
    // is the NUMBER OF DISTINCT ATTRACTING CYCLES the map has at that a
    // (infinity counts as one when it's the limit of a critical orbit).
    // Escape-radius-free -- this is a genuine multi-attractor question,
    // the one render_parameter's own retired Stage 4 detour was trying to
    // answer before being pulled back to plain escape-time (see its own
    // doc comment).
    //
    // METHOD, per pixel: get EVERY distinct critical point at that
    // parameter (RationalMap::distinct_critical_points -- by Fatou, every
    // attracting cycle attracts at least one critical point, so this is
    // enough to find them all; no per-pixel find_attractors call that
    // would redundantly re-root-find the SAME critical points a second
    // time -- see find_attractors_from_seeds' own doc comment), then run
    // complete_attractors_from_seeds against THAT seed list (the SAME
    // burn-in + chordal-closure-detection + attracting-multiplier-
    // verification + chordal dedup machinery find_attractors already uses
    // everywhere else in this codebase, UNIONED with every algebraically-
    // attracting fixed point RationalMap::fixed_points already knows
    // exactly -- see complete_attractors' own doc comment for why plain
    // find_attractors alone is not a completeness guarantee). The
    // result's own .size() is the distinct-attracting-cycle count;
    // `unresolved`, if given, is replaced with an Image the same size as
    // the result holding how many of that pixel's critical orbits are
    // explained by NO attractor in the complete set (complete_attractors_
    // from_seeds' own `unresolved_count` out-param, ALREADY reconciled
    // against the union -- an orbit that failed to close numerically but
    // actually landed on an injected fixed point is not counted here) --
    // tracked SEPARATELY from the count, so a caller can tell "0
    // attracting cycles, nothing else interesting either" apart from "0
    // attracting cycles because everything here is still unresolved"
    // rather than the two being silently conflated into the same 0.
    //
    // `certain`, if given, is replaced with an Image the same size as the
    // result holding how many of that pixel's cycles are CERTAIN --
    // algebraically injected fixed points (complete_attractors_from_seeds'
    // own `certain_count` out-param), not numerically-discovered closures
    // still subject to find_attractors' own finite budget/tolerance. Lets
    // a caller (the Python coloring layer -- see app/color.py's
    // color_parameter_basin) weigh a certain attractor differently from
    // ordinary unresolved noise, e.g. still showing a pixel's count color
    // even when unresolved > count, PROVIDED at least one of that count is
    // certain (a genuinely unrelated Siegel/Herman/parabolic residual
    // elsewhere in the SAME pixel shouldn't retroactively make an exact,
    // algebraically-known attractor look unconfirmed).
    //
    // KNOWN RESIDUAL, honestly: this counts ATTRACTING cycles specifically
    // (multiplier strictly < 1, find_attractors' own existing criterion).
    // A Siegel disc / Herman ring (irrationally neutral, multiplier on the
    // unit circle) or a parabolic cycle (multiplier a root of unity) is
    // NEITHER attracting NOR does a critical orbit inside one ever close
    // chordally onto anything within max_period -- it shows up as
    // unresolved, not as a phantom attracting cycle and not silently
    // dropped either. Correct for HYPERBOLIC maps (attracting cycles are
    // the whole story); a genuine limitation for non-hyperbolic ones,
    // which this reports honestly via `unresolved` rather than hiding.
    //
    // Requires a Custom-wrapped map (map_.custom_map() non-null) -- a
    // genuine built-in Family with no RationalMap behind it (never
    // actually reached by the app itself, which always renders through
    // Map::custom -- see app/session.py's render_map) has no RationalMap
    // for find_attractors_from_seeds to call eval()/deriv() on; returns
    // an all-zero degenerate image with `unresolved`/`certain` all-zero
    // too in that case, rather than guessing.
    Image render_parameter_basin(const std::atomic<bool>* cancel = nullptr,
                                 Image* unresolved = nullptr, Image* certain = nullptr) const;

    // Parameter_period: each pixel is a parameter value a; the pixel value
    // is the PERIOD of the attracting cycle a TRACKED critical orbit
    // converges to -- reproducing the period-colored parameter-plane
    // figures in the Lindsey-Koch-Sharland bicritical-maps literature,
    // for side-by-side comparison with Parameter_basin's own count
    // coloring. Escape-radius-free, same as Parameter_basin.
    //
    // WHY THIS IS A SEPARATE METHOD FROM render_parameter_basin, not a
    // second output of it: the two ask genuinely different questions.
    // Parameter_basin seeds from EVERY critical point the map has
    // (RationalMap::distinct_critical_points(a)) and reports how many
    // DISTINCT attractors that whole set reaches -- correct for counting,
    // wrong for period, because (per_seed_outcomes' own doc comment,
    // confirmed by direct measurement) the complete/deduplicated
    // attractor set can contain algebraically-injected fixed points no
    // TRACKED seed's own orbit ever reaches, and can (for a bicritical
    // family) contain several SIMULTANEOUSLY attracting cycles from
    // critical points that are not the one a literature figure means by
    // "the free critical orbit". This method therefore takes an EXPLICIT
    // `seed_family` recipe (see CriticalPointFamily's own doc comment)
    // computing exactly the critical points a period-coloring consumer
    // means to track, per pixel (they are generally PARAMETER-DEPENDENT,
    // e.g. this family's own z_c(a) formula -- there is no `cp_fixed`
    // once-per-render optimization here the way Parameter_basin has for a
    // parameter-INDEPENDENT critical point set).
    //
    // METHOD, per pixel: compute the tracked seeds via `seed_family`'s own
    // closed form at that pixel's `a`, then per_seed_outcomes against
    // them (the SAME burn-in + closure-detection + attracting-multiplier-
    // verification + algebraic reconciliation machinery every other
    // discovery path in this codebase uses -- nothing bespoke here
    // either). PERIOD MEASURED IN THE RAW Z-PLANE: per_seed_outcomes'
    // own period is literally a discovered Cycle's points.size(), the
    // SAME quantity find_attractors/complete_attractors already report --
    // never computed in a symmetry quotient (e.g. w = z^n for this
    // family), which would collapse a genuine z-plane period-P star cycle
    // into a fixed point and silently mislabel every color.
    //
    // POLICY when the tracked seeds disagree (checked in the order
    // `seed_family` generates them): the FIRST one that resolves wins --
    // by the family's own equivariance (z -> omega*z symmetry) every one
    // of RelaxedNewtonPower's own n seeds is guaranteed to agree in
    // exact arithmetic, so this is a tie-break for numerical stragglers,
    // not a real policy choice for THAT family; a future, genuinely
    // asymmetric family (a bicritical one, whose two critical points CAN
    // converge to two actually-different cycles) would need this
    // documented explicitly wherever it is used -- this method does not
    // attempt to detect or flag such a disagreement itself (no "mixed"
    // class yet; not needed by any family this method currently supports).
    //
    // The pixel value is the period (>= 1) when a tracked seed resolves to
    // a FINITE cycle; 0 (paired with `undetermined` set) when NONE of the
    // tracked seeds resolve within budget -- a genuine residual (Siegel/
    // Herman/parabolic), never a fabricated period; also 0 (paired with
    // `is_infinity` set instead) when the resolved cycle is the point at
    // infinity -- kept OUT of the ordinary period value and its own golden-
    // hue color family entirely, since "converges to infinity" is a
    // dynamically different kind of behaviour from "converges to an
    // ordinary period-1 cycle in the plane", not merely a variant of it.
    //
    // `undetermined`, if given, is replaced with an Image the same size as
    // the result: 1.0 where no tracked seed resolved (see above), 0.0
    // elsewhere. `is_infinity`, if given, is replaced with an Image the
    // same size: 1.0 where the resolved cycle is the point at infinity,
    // 0.0 elsewhere. At most one of the two is ever 1.0 for the same pixel.
    //
    // Requires a Custom-wrapped map (map_.custom_map() non-null), same as
    // Parameter_basin -- degrades to an honest all-undetermined image
    // otherwise, rather than guessing. CURRENTLY SUPPORTS EXACTLY ONE
    // seed_family (RelaxedNewtonPower) -- calling this on a map that is
    // not structurally that family produces seeds that do not correspond
    // to its actual critical points, and the result is meaningless (not a
    // crash: those seeds' own orbits are followed under the map's real
    // dynamics regardless, they simply are not the family's own critical
    // points) -- this is a scope limitation to document at every call
    // site, not a case this method itself detects or guards against.
    Image render_parameter_period(CriticalPointFamily seed_family, int n,
                                  const std::atomic<bool>* cancel = nullptr,
                                  Image* undetermined = nullptr,
                                  Image* is_infinity = nullptr) const;

    // Basin classification against a set of attracting cycles, in the chordal
    // metric. Value is the cycle id, or 0 for unresolved pixels.
    //
    // If `iterations` is given, it is replaced with an Image the same size
    // as the result, holding the iteration count (n+1, matching
    // render_julia's own n+1 convention) each pixel took to resolve -- for
    // BASIN SHADING (hue = basin id from the primary result, brightness =
    // this convergence speed; see app/color.py's color_basin). No extra
    // computation: the per-pixel loop already counts iterations to decide
    // when to stop, this just retains that count instead of discarding it.
    // For an UNRESOLVED pixel (primary result 0), the count is however many
    // iterations were actually run (max_iter if the orbit neither resolved
    // nor blew up, fewer if it hit is_bad first) -- not a meaningful
    // "convergence speed" since nothing converged, callers should treat an
    // unresolved pixel's shading as undefined and color it by the primary
    // result being 0 instead (exactly what color_basin already does).
    Image render_basin(const std::vector<Cycle>& cycles, Image* iterations = nullptr,
                       const std::atomic<bool>* cancel = nullptr) const;

    // Green's function (dynamical potential). TWO PATHS, same certification
    // dispatch as render_julia (Map::escape_certified):
    //
    //   CERTIFIED POLYNOMIAL: UNCHANGED -- G_f(z) = lim_{n->inf} d^-n
    //   log+|f^n(z)|. When the orbit escapes at iteration n, the result is
    //   log|z_n| / degree^n; non-escaping pixels are exactly 0. `cycles`/
    //   `potential`/`exact` are ignored entirely (PRAGMATIC and CONFORMAL-
    //   Boettcher already coincide exactly at a certified polynomial's own
    //   infinity -- see GreensPotential's own doc comment).
    //
    //   RATIONAL: sphere-aware, escape_radius-free, against `cycles` (the
    //   SAME found-attractor set render_julia/render_basin take) -- see
    //   the .cpp for the actual potential derivations. `potential` selects
    //   PRAGMATIC (the SAME smooth chordal approach-rate render_julia's own
    //   rational path computes -- reused, not re-derived) or CONFORMAL
    //   (log|phi(z)| for the Boettcher/Koenigs coordinate at the reached
    //   attractor, estimated numerically from the orbit's own last two
    //   chordal distances before crossing tol -- see the .cpp for the
    //   superattracting-vs-geometric-vs-parabolic classification this
    //   estimate is built on). If `exact` is given, it is replaced with an
    //   Image the same size as the result, 1.0 where CONFORMAL was
    //   genuinely computed and 0.0 where it fell back to PRAGMATIC (the
    //   orbit's own local behaviour was too close to parabolic to
    //   distinguish, or the pixel never resolved at all) -- PRAGMATIC
    //   itself has no such fallback, `exact` is meaningless for it.
    //
    // Normalizing at each pixel's OWN (escape or attractor-crossing)
    // iteration -- not against a single degree^max_iter for the whole
    // image -- is what makes this the actual escape-rate potential rather
    // than a near-constant field, in EITHER path; see the .cpp for the
    // measured difference on the certified-polynomial path specifically.
    Image render_greens(const std::atomic<bool>* cancel = nullptr,
                        const std::vector<Cycle>& cycles = {},
                        GreensPotential potential = GreensPotential::Pragmatic,
                        Image* exact = nullptr) const;

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
    //
    // CERTIFIED POLYNOMIAL: UNCHANGED, exactly as above. `potential`/`exact`
    // ignored.
    //
    // RATIONAL: escape-radius-free, but deliberately NOT a per-pixel
    // find_attractors -- that would mean a full root-find at EVERY
    // parameter pixel (attractors genuinely depend on the parameter here,
    // unlike render_greens' single fixed map), which is exactly the
    // per-pixel-root-find cost this codebase's own performance notes warn
    // against. This function only ever asked "does the critical orbit
    // escape to infinity" (the classical Mandelbrot-style connectedness
    // question) -- not "which of several attractors captures it" -- so the
    // rational generalization keeps that SAME single question, just asked
    // chordally: track the critical orbit's chordal distance to infinity
    // specifically (a fixed, trivial one-point attractor, no root-find at
    // all) and apply the SAME Pragmatic/Conformal-Boettcher potential
    // render_greens uses for an infinity attractor. If this particular
    // parameter's infinity isn't actually attracting, the orbit simply
    // never gets chordally close to it, stays unresolved, and the pixel is
    // 0 -- "in the connectedness locus" -- exactly the existing convention.
    // Multi-critical-point classification is Parameter_basin's own job
    // (render_parameter_basin -- counts attracting cycles per parameter,
    // not an escape potential) -- this function stays single-critical/
    // single-attractor by design.
    Image render_parameter_greens(const std::atomic<bool>* cancel = nullptr,
                                  GreensPotential potential = GreensPotential::Pragmatic,
                                  Image* exact = nullptr) const;

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

    // render_julia's own two paths (see its own doc comment for what each
    // is) -- split out so the dispatcher itself (render_julia) just picks
    // one, rather than one function trying to be both. Both are private:
    // parallel_columns is, and both need it, the same as every other
    // render_* method already gets it via being a Renderer method itself.
    Image render_julia_polynomial(const std::atomic<bool>* cancel) const;
    Image render_julia_rational(const std::vector<Cycle>& cycles, Image* labels,
                                const std::atomic<bool>* cancel) const;

    // render_parameter's own two paths, same split rationale as render_
    // julia's -- see render_parameter's own doc comment for what each is.
    Image render_parameter_polynomial(const std::atomic<bool>* cancel) const;
    Image render_parameter_rational(ParameterStrategy strategy, int critical_index,
                                    const std::atomic<bool>* cancel) const;

    // render_greens' own two paths, same split rationale as render_julia's.
    Image render_greens_polynomial(const std::atomic<bool>* cancel) const;
    Image render_greens_rational(const std::vector<Cycle>& cycles, GreensPotential potential,
                                 Image* exact, const std::atomic<bool>* cancel) const;

    // render_parameter_greens' own two paths. The rational path's "cycles"
    // is always the fixed, single-point infinity attractor -- see
    // render_parameter_greens' own doc comment for why.
    Image render_parameter_greens_polynomial(const std::atomic<bool>* cancel) const;
    Image render_parameter_greens_rational(GreensPotential potential, Image* exact,
                                           const std::atomic<bool>* cancel) const;

    Map            map_;
    Viewport       view_;
    RenderSettings settings_;
};

}  // namespace cdx
