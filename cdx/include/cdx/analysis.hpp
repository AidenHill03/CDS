// =============================================================================
// cdx/analysis.hpp -- numerical analysis layer: attractor discovery and
// diagnostics, ported from the MATLAB prototype (matlab-reference/).
//
// Components, each building on the previous:
//   1. find_attractors     -- discovers attracting cycles via critical orbits
//   2. wada_diagnostic      -- Wada-boundary signatures on a basin image
//   3. hausdorff_distance   -- Julia-set-vs-target distance, both metrics
//   4. dynamical_facts      -- bundles the above with RationalMap's own
//                              algebraic facts into one data-extraction call
//
// verify_conditions -- per-region numerical checks of the Fisher-Hill-
// Lazebnik-Thompson paper's conditions (2.4)/(2.5) -- was attempted here and
// removed. It needs a target region, a Runge preprocessing map, and the
// model map, none of which the engine has any notion of; it belongs in
// modules/approximation/ instead. See ARCHITECTURE.md's "Deferred: the
// approximation module" section for the construction and the two specific
// mistakes worth not repeating.
//
// All of this operates through the SAME sphere-first primitives as the rest
// of this codebase (chordal_distance, Cycle, Image) -- infinity is an
// ordinary point here too, not a special case to work around.
// =============================================================================
#pragma once

#include "cdx/rational.hpp"
#include "cdx/renderer.hpp"

#include <cstddef>
#include <vector>

namespace cdx {

// -----------------------------------------------------------------------------
// 1. Attracting-cycle discovery.
//
// By Fatou's theorem, every attracting cycle of a rational map attracts at
// least one critical point, so iterating every critical orbit past its
// transient and detecting the cycle it lands on finds EVERY attracting
// cycle -- no symbolic solving, no assuming the period in advance, and no
// "attracting fixed points only" blind spot (a fixed point is just the
// period-1 case; z^2-1's basilica has a period-2 attracting cycle and ZERO
// attracting fixed points, which a fixed-point-only search would miss
// entirely). Ports matlab-reference/FindAttractors.m.
//
// DEPENDS ON RationalMap::distinct_critical_points BEING COMPLETE --
// including poles and infinity, not just ordinary derivative zeros -- or
// this silently misses cycles whose only attracting critical point is a
// pole (e.g. a Newton-map-style RationalMap, where the informative critical
// point sits at a pole and the "obvious" ones are trivial superattracting
// fixed points) or infinity itself.
// -----------------------------------------------------------------------------
struct FindAttractorsOptions {
    int    burn_in          = 500;   // iterations discarded as transient
    int    max_period       = 64;    // largest cycle period detected
    double tol               = 1e-9; // chordal tolerance for cycle closure and dedup
    double inf_cutoff        = 1e12; // |z| beyond which a value counts as infinity
    bool   verify_multiplier = true; // reject candidates with |multiplier| >= 1
};

std::vector<Cycle> find_attractors(const RationalMap& map, Cplx a,
                                   const FindAttractorsOptions& opts = {});

// Same discovery, but seeded from a caller-supplied critical-point list
// instead of calling map.distinct_critical_points(a) itself. find_attractors
// above is now a thin wrapper over this. Exists for a caller that ALSO
// needs the critical points themselves for its own per-orbit use --
// render_parameter_basin (renderer.cpp), which counts DISTINCT attracting
// cycles per PARAMETER PIXEL, is the current one: distinct_critical_points
// is a real, sometimes root-finding-heavy cost (see cdx_test_custom's own
// benchmark comment), so a per-pixel caller computes it once and passes it
// here rather than paying for it twice.
//
// `unresolved_count`, if given, is set to how many of `seeds` did NOT end
// up contributing a cycle to the result -- a critical orbit that never
// closed onto anything within `opts.max_period` (parabolic/rotation-domain/
// slow), that closed onto a cycle but failed the attracting-multiplier
// check, or that excursed past inf_cutoff during burn-in without infinity
// actually being attracting there. render_parameter_basin needs this to
// report "unresolved" honestly rather than silently folding it into the
// attracting-cycle count (see its own doc comment) -- every other existing
// caller ignores it (nullptr, the default), unchanged.
std::vector<Cycle> find_attractors_from_seeds(const std::vector<Cplx>& seeds,
                                              const RationalMap& map, Cplx a,
                                              const FindAttractorsOptions& opts = {},
                                              int* unresolved_count = nullptr);

// -----------------------------------------------------------------------------
// Polynomial escape-radius certification.
//
// TRUE iff `map` has NO poles anywhere in its STRUCTURE (no enabled
// PoleTerm, and no enabled PolyTerm with a negative exponent -- a negative
// exponent implies a pole at the origin too, see RationalMap::degree's own
// den_deg logic, which this mirrors) and its degree is >= 2. Parameter-
// independent: which TERMS exist doesn't depend on `a`'s value, only their
// coefficients do (and RationalMap::degree itself never reads `a` either).
//
// With no poles anywhere, infinity is the map's only point sent to itself
// as |z| grows (R(z) -> infinity as |z| -> infinity, with no OTHER pole to
// compete for where a large z lands), and for degree >= 2 it is ALWAYS
// superattracting there (multiplier 0 -- see RationalMap::fixed_points'
// own w=1/z-chart computation: diff = degree >= 2 forces multiplier
// exactly 0). The classical |z| > R escape test is therefore a PROVABLY
// forward-invariant trap for any large enough R: once |z| exceeds it,
// every later iterate is even larger, so the escaping set -- and the
// Julia set as its boundary -- is independent of exactly which R was
// used. This is what makes today's polynomial escape-time fast path
// correct, not just fast.
//
// A rational map WITH poles has no such blanket guarantee: infinity may be
// repelling (Newton's method: multiplier 3/2), or attracting despite the
// poles (e.g. RationalMap::mcmullen(n>=2): numerator degree exceeds
// denominator degree by n after clearing denominators, so infinity is
// STILL superattracting there even though the map has a pole at the
// origin) -- either way, a fixed escape radius is not a validated trap in
// general, so classification must go through the sphere-aware (chordal)
// path instead of this fast path. This predicate is therefore
// CONSERVATIVE, not exhaustive: it certifies the case that is always
// safe (no poles at all), not every case where infinity happens to be
// attracting. Extending the fast path to cover a rational map with poles
// whose infinity is PROVABLY attracting (mcmullen-style) is future work,
// not required here -- the sphere-aware path already handles that case
// correctly, just not via the fast path's own machinery.
// -----------------------------------------------------------------------------
bool polynomial_escape_certified(const RationalMap& map);

// -----------------------------------------------------------------------------
// 2. Wada-boundary diagnostic on a basin label image (Renderer::render_basin's
// output: 0 = unresolved, k = basin k). Ports matlab-reference/WadaDiagnostic.m.
//
// wada_fraction is the fraction of boundary pixels (>= 2 distinct basins
// within the neighbourhood radius) whose neighbourhood contains ALL
// n_basins labels -- the numerical signature of a genuine Wada boundary,
// where every boundary point borders every basin.
//
// RESOLUTION DEPENDENCE. The true Wada property is an infinite-resolution
// statement (every boundary point has all d basins in EVERY neighbourhood);
// at any finite pixel resolution a small neighbourhood typically shows only
// the locally-dominant basins even for a genuinely Wada map, because a
// third basin's presence can be at a finer scale than the neighbourhood.
// radius_fraction is deliberately a FRACTION of the image's resolution, not
// a fixed pixel count: a fixed pixel radius covers a physically SMALLER
// region as resolution rises, which makes wada_fraction artificially FALL
// with resolution even when the map itself is unchanged. Read wada_fraction
// as a TREND across resolutions, not an absolute value at one setting -- it
// should rise toward 1 as resolution increases for a genuine Wada
// configuration, and plateau below 1 for a non-Wada one (e.g. two basins
// sharing a plain arc away from the others).
// -----------------------------------------------------------------------------
struct WadaStats {
    int    n_basins            = 0;
    double unresolved_fraction = 0.0;
    double boundary_fraction   = 0.0;
    double wada_fraction       = 0.0;   // NaN if fewer than 2 basins, or no boundary pixels
    int    radius_px           = 0;     // the actual pixel radius used, after resolution-scaling
};

WadaStats wada_diagnostic(const Image& labels, double radius_fraction = 0.004);

// -----------------------------------------------------------------------------
// 3. Hausdorff distance between two point sets, in both the chordal
// (sphere-aware) and Euclidean metrics, with the two DIRECTED distances
// exposed separately. Ports matlab-reference/HausdorffBasin.m.
//
// Symmetric Hausdorff = max(sup_{a in A} inf_{b in B} d(a,b),
//                            sup_{b in B} inf_{a in A} d(a,b)).
// The two directed pieces diagnose different failures: large
// julia_to_target means the computed set has spurious structure far from
// the target; large target_to_julia means part of the target has no nearby
// match in the computed set (a missed piece of boundary).
//
// Inputs larger than max_points are subsampled (evenly, not randomly) to
// keep the O(n*m) distance computation tractable at high resolution.
// -----------------------------------------------------------------------------
struct HausdorffResult {
    double chordal                   = 0.0;
    double euclidean                 = 0.0;
    double chordal_julia_to_target   = 0.0;
    double chordal_target_to_julia   = 0.0;
    double euclidean_julia_to_target = 0.0;
    double euclidean_target_to_julia = 0.0;
};

HausdorffResult hausdorff_distance(const std::vector<Cplx>& julia_points,
                                   const std::vector<Cplx>& target_points,
                                   std::size_t max_points = 4000);

// Extracts a basin-label image's boundary pixels (adjacent to a different
// nonzero label) as complex points in the view's coordinate plane -- the
// discretized Julia set a Renderer::render_basin() image implies. This is
// how julia_points is obtained in practice for hausdorff_distance above.
std::vector<Cplx> extract_boundary_points(const Image& labels, const Viewport& view);

// -----------------------------------------------------------------------------
// 4. dynamical_facts -- everything app/session's data-extraction call needs,
// in one report: RationalMap's own algebraic facts (degree, critical points,
// pole locations and orders, fixed points) plus find_attractors' dynamical
// ones (the attracting cycles, each with its period and multiplier).
//
// This is a pure function of (map, parameter, discovery options) -> data --
// the same seam every analysis in this file has (see ARCHITECTURE.md's "The
// seam"), just bundling several existing ones instead of computing something
// new. It exists because a sandbox session wants ALL of this together for
// one map, not because any single piece of it is a new capability.
// -----------------------------------------------------------------------------
struct DynamicalFacts {
    struct AttractingCycle {
        std::vector<Cplx> points;
        int  period = 0;
        Cplx multiplier;   // product of deriv() around the cycle; see .cpp for the Inf case
    };

    int                        degree = 0;
    std::vector<Cplx>          critical_points;    // with multiplicity, see RationalMap::critical_points
    std::vector<AttractingCycle> attracting_cycles;
    std::vector<Cplx>          pole_locations;
    std::vector<int>           pole_orders;        // parallel to pole_locations
    std::vector<FixedPoint>    fixed_points;        // ALL fixed points, not just attracting ones
};

DynamicalFacts dynamical_facts(const RationalMap& map, Cplx a,
                               const FindAttractorsOptions& opts = {});

}  // namespace cdx
