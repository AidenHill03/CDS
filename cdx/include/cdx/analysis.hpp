// =============================================================================
// cdx/analysis.hpp -- numerical analysis layer: attractor discovery and
// diagnostics, ported from the MATLAB prototype (matlab-reference/).
//
// Components, each building on the previous:
//   1. find_attractors     -- discovers attracting cycles via critical orbits
//   1b. complete_attractors -- find_attractors UNION algebraically-attracting
//                              fixed points (RationalMap::fixed_points), the
//                              set every real consumer (fact sheet, basin,
//                              Parameter_basin) should use -- see its own
//                              doc comment for why find_attractors alone is
//                              not a complete-by-construction guarantee
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
    int    burn_in          = 500;   // SAFETY CAP on transient-elimination steps, not a
                                      // mandatory count -- see find_attractors_from_seeds'
                                      // own doc comment for how closure is detected earlier
    int    max_period       = 64;    // largest cycle period detected
    double tol               = 1e-9; // chordal tolerance for cycle closure and dedup
    double inf_cutoff        = 1e12; // |z| beyond which a value counts as infinity
    bool   verify_multiplier = true; // reject candidates with |multiplier| >= 1

    // WEAKLY ATTRACTING cycle confirmation (multiplier close to, but under,
    // 1): a cycle that is genuinely attracting but converges slowly can
    // fail to close within `tol`/`max_period` above -- the strict pass
    // alone would then reject it as unresolved even though it is real.
    // When `confirm_weakly_attracting` (on by default) and a seed's
    // strict pass fails, the SAME orbit continues for up to
    // `extended_max_period` MORE steps looking for a closure at the much
    // looser `loose_tol`; if found, the candidate periodic point is
    // Newton-polished (`newton_iterations` steps) against f^p(z)-z=0 and
    // the EXACT multiplier is recomputed at the refined point before
    // accepting -- see find_attractors_from_seeds' own doc comment for
    // why this is strictly more correct, never less, than the strict
    // pass alone. Only ever consulted when opts.verify_multiplier is also
    // true (there is nothing to "confirm" if multiplier verification
    // itself is disabled).
    bool   confirm_weakly_attracting = true;
    double loose_tol             = 1e-4;  // closure tolerance for the extended search
    int    extended_max_period   = 500;   // additional steps tried beyond max_period
    // Generous relative to the "few Newton iterations" a SIMPLE root would
    // need: Newton's method only converges LINEARLY (not quadratically)
    // at a DOUBLE root, i.e. exactly a parabolic candidate -- see
    // attracting_margin's own comment for why that specific case matters
    // here. Measured: ~20 iterations to reach the double-precision noise
    // floor starting from a loose_tol=1e-4 candidate; 25 leaves headroom.
    int    newton_iterations     = 25;    // Newton-polish steps on the candidate point

    // Safety margin below 1.0 the CONFIRMATION path's own refined
    // multiplier must clear: |multiplier| < 1.0 - attracting_margin. NOT
    // applied to the strict pass's own multiplier check (a cycle that
    // already closed within `tol` chordally in only `max_period` steps
    // couldn't have done so if it were genuinely parabolic, so that
    // check's plain < 1.0 is already reliable). It IS needed here because
    // Newton's method loses its normal quadratic convergence at a
    // genuinely parabolic point (multiplier exactly 1 means f^p(z)-z has
    // a DOUBLE root there, and Newton's method only converges LINEARLY at
    // a double root) -- polishing a truly parabolic candidate still lands
    // extremely close to the exact point, but the multiplier computed
    // there comes out as 1.0 minus a small floating-point residual
    // (observed ~1e-9, the double-precision noise floor for this
    // computation) rather than being reliably >= 1.0. Without this
    // margin, that residual alone would falsely confirm a genuinely
    // parabolic cycle as attracting -- exactly the false-positive this
    // whole confirmation path exists to avoid, not introduce. 1e-6 is
    // comfortably above the observed noise floor while still well below
    // any multiplier a real, if very weakly, attracting cycle would have.
    double attracting_margin     = 1e-6;
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
//
// WEAKLY ATTRACTING CYCLES (multiplier close to, but under, 1 in
// magnitude): the strict pass alone accepts a cycle by CLOSURE first
// (chordal(z_k, z_0) < opts.tol within opts.max_period steps) and only
// THEN checks the multiplier -- so a genuinely attracting cycle that
// simply converges too slowly to close within that strict budget is
// rejected as unresolved before the multiplier is ever examined, even
// though the multiplier alone (the actual mathematical definition of
// "attracting") would confirm it. This showed up as two SEPARATE bugs
// sharing one root cause: Parameter_basin's own artifact count-regions
// and unresolved band/specks on weakly-hyperbolic families, AND a fact
// sheet whose fixed-points table (RationalMap::fixed_points, an
// independent ALGEBRAIC root-find + exact deriv() evaluation, no
// iteration involved) correctly marks such a point attracting while its
// attracting-cycles table (fed by this very function) omits it -- same
// map, same parameter, two different verdicts.
//
// When `opts.confirm_weakly_attracting` (default on) and the strict pass
// fails for a seed, the SAME orbit is continued for up to
// `opts.extended_max_period` further steps, checking closure against the
// much looser `opts.loose_tol`. A loose closure only identifies an
// APPROXIMATE periodic point -- if the orbit is still converging slowly,
// it isn't yet ON the cycle, so its raw multiplier there would be
// unreliable. The candidate is therefore Newton-polished
// (`opts.newton_iterations` steps, solving f^p(z)-z=0 via g'(z) =
// (f^p)'(z)-1, itself just the SAME product-of-deriv() the strict path's
// own multiplier check already computes) to the TRUE periodic point,
// where the EXACT multiplier is recomputed and checked. This is strictly
// MORE correct than the strict pass alone: it can only ever ADD a
// genuinely attracting cycle the strict pass missed on convergence speed
// alone, never accept anything the multiplier itself would reject (a
// repelling or parabolic candidate fails the SAME |multiplier| < 1 test,
// now evaluated more accurately, if anything making a false accept LESS
// likely), and a cycle the strict pass already found closes and returns
// exactly as before (this extended search never runs for it at all).
//
// SPEED: both passes above exit as soon as closure is confirmed rather than
// always burning their full settling budget (`burn_in`/`extended_max_period`)
// first -- a deep-interior, quickly-converging orbit typically settles in
// tens of steps, not hundreds, and previously paid the FULL budget
// regardless. This changes ONLY when closure is detected, never what is
// found: every accepted cycle's period and multiplier are identical to what
// the fixed-budget search would have found, just often discovered sooner
// (see analysis.cpp's own, fuller comment on the mechanism and the two
// numerical pitfalls -- chordal saturation near infinity, and a
// slowly-converging multi-point cycle's non-monotonic residual -- that a
// naive early-exit falls into).
// `unresolved_endpoints`, if given, gets one Cplx appended per seed that
// contributes to `unresolved_count` -- the LAST live orbit point that seed
// reached before being counted unresolved (the finite point right before
// an infinity excursion, or the final point of however far the settle/
// closure/weak-confirm search got before giving up, or the closed-but-not-
// attracting candidate's own point when multiplier verification is what
// rejected it). Exists so complete_attractors_from_seeds (below) can check
// whether an "unresolved" orbit was actually heading toward an attractor
// the algebraic union recovers -- see its own doc comment. Ignored (never
// even collected) when nullptr, same as `unresolved_count` itself; the two
// are always the same length when both are given, one entry per
// increment, in the same order.
std::vector<Cycle> find_attractors_from_seeds(const std::vector<Cplx>& seeds,
                                              const RationalMap& map, Cplx a,
                                              const FindAttractorsOptions& opts = {},
                                              int* unresolved_count = nullptr,
                                              std::vector<Cplx>* unresolved_endpoints = nullptr);

// -----------------------------------------------------------------------------
// 1b. complete_attractors -- find_attractors' own critical-seeded cycles,
// UNIONED with every algebraically-attracting fixed point RationalMap::
// fixed_points(a) reports (|multiplier| < 1, including infinity), each
// added as its own period-1 Cycle when no critical-seeded cycle already
// represents it (a period-1 cycle at the same point, chordally, within
// opts.tol * 1e3 -- the same "looser dedupe" tolerance find_attractors
// itself already uses between cycles it discovers).
//
// WHY THIS EXISTS, on top of find_attractors already existing: by Fatou's
// theorem every attracting cycle attracts at least one critical point, so
// find_attractors' critical-seeded search is complete IN THE LIMIT of
// exact arithmetic and unbounded iteration -- but it is still a finite-
// budget NUMERICAL SEARCH (opts.burn_in/max_period/tol, and see
// find_attractors_from_seeds' own doc comment for the weakly-attracting
// case this already had to special-case once), not a proof. A period-1
// fixed point, by contrast, is available from RationalMap::fixed_points
// EXACTLY and CHEAPLY -- a single polynomial root-find, no orbit iteration
// at all -- with its multiplier computed analytically via deriv(), not
// estimated from a numerically-detected closure. There is no reason a
// consumer that wants "every attracting cycle" should ever fail to
// include a period-1 one this cheaper, exact source already knows about,
// regardless of whether the numerical search alone would have found it.
//
// This is a PURE ADDITION over find_attractors' own output: every cycle
// find_attractors returns is kept unchanged (including any period-1 one
// it already found on its own, whose point and derived multiplier are
// left exactly as discovered, not replaced), and the only thing ever
// APPENDED is a fixed point find_attractors did not already represent.
// Cannot double-count (the dedup check above) and cannot regress a map
// that was already complete (nothing new passes the "already represented"
// check, so cycles come back byte-for-byte identical).
//
// Every REAL consumer of "the attractor set" (dynamical_facts' own
// attracting_cycles, session.py's basin-mode cycle source, Renderer::
// render_parameter_basin's per-pixel count) should call this instead of
// find_attractors directly. find_attractors itself is UNCHANGED and still
// exists in its own right: several existing tests (test_analysis.cpp) and
// cdx_diagnose_parameter_basin specifically exercise the PURE
// critical-seeded algorithm's own properties (unresolved-count behaviour,
// weak-attraction confirmation, before/after performance) and must keep
// doing so unperturbed by this reconciliation layer.
std::vector<Cycle> complete_attractors(const RationalMap& map, Cplx a,
                                       const FindAttractorsOptions& opts = {});

// Same reconciliation, seeded from a caller-supplied critical-point list --
// the complete_attractors analogue of find_attractors_from_seeds, for a
// caller (render_parameter_basin) that already has its own seed list in
// hand and would otherwise pay for distinct_critical_points(a) twice.
//
// `unresolved_count` MEANS "critical orbits explained by NO attractor in
// the COMPLETE set" -- not just the critical-seeded pass's own count.
// find_attractors_from_seeds' raw tally (an orbit the strict/weak-confirm
// search itself failed to close, or closed onto something not attracting)
// can OVER-report once the fixed-point union adds a cycle that same orbit
// was actually heading toward all along -- e.g. an orbit whose closure
// detection genuinely failed (see find_attractors_from_seeds' own doc
// comment on its finite budget/tolerance), but whose LAST live point
// (find_attractors_from_seeds' own `unresolved_endpoints`) sits right on
// top of a fixed point the union recovers. Such an orbit is not a residual
// miss -- it found exactly where it was going, just not via a numerically
// clean closure -- so after the union runs, every unresolved endpoint is
// swept against the COMPLETE cycle set (critical-seeded and injected
// alike) at the same opts.tol * 1e3 chordal tolerance the injection dedup
// itself uses; a match decrements unresolved_count (the attractor is
// already present in the returned cycles -- nothing is added a second
// time). What remains after this sweep is a genuine residual: a critical
// orbit no complete attractor explains at all (Siegel/Herman/parabolic, or
// a candidate that closed but was not actually attracting).
//
// `certain_count`, if given, is set to how many of the returned cycles
// were ADDED by the algebraic union (i.e. are exact fixed points RationalMap
// ::fixed_points reports, not numerically-discovered closures) -- distinct
// from the find_attractors_from_seeds-discovered ones, which remain subject
// to that search's own finite budget/tolerance. A caller that wants to
// weigh a CERTAIN attractor differently from unresolved noise (e.g.
// Renderer::render_parameter_basin's own coloring policy -- see its doc
// comment) reads this rather than re-deriving which cycles are which.
// `fp_predictor`/`fp_raw_out` (Stage 2, spatial continuation): the internal
// algebraic union's own cold RationalMap::fixed_points(a) call is exactly
// the kind of per-pixel root-find continuation exists to speed up -- a
// caller sweeping a grid of nearby `a` (Renderer::render_parameter_basin/
// render_parameter_rational) can pass the PRECEDING pixel's own `fp_raw_out`
// as `fp_predictor` here, and get RationalMap::fixed_points_continued's own
// warm-started result instead of a cold solve; `fp_raw_out`, if given, is
// set to the raw predictor to feed into the NEXT pixel's own call. Both
// default to nullptr, meaning "cold fixed_points(a), exactly as before this
// parameter existed" -- every other caller of this function is unaffected.
// The returned cycles are IDENTICAL either way, same guarantee as
// RationalMap::fixed_points_continued's own contract.
std::vector<Cycle> complete_attractors_from_seeds(const std::vector<Cplx>& seeds,
                                                  const RationalMap& map, Cplx a,
                                                  const FindAttractorsOptions& opts = {},
                                                  int* unresolved_count = nullptr,
                                                  int* certain_count = nullptr,
                                                  const std::vector<Cplx>* fp_predictor = nullptr,
                                                  std::vector<Cplx>* fp_raw_out = nullptr);

// -----------------------------------------------------------------------------
// 1c. per_seed_outcomes -- WHAT DID *THIS SPECIFIC* SEED CONVERGE TO, not the
// deduplicated attractor SET find_attractors/complete_attractors return.
//
// WHY THIS EXISTS, distinct from complete_attractors_from_seeds: that
// function answers "what is the complete set of attractors this parameter
// has" -- correct for basin-count coloring, where every attractor counts
// once no matter which (or how many) seeds found it. A PERIOD-coloring
// consumer asks a different question: "what does THIS PARTICULAR critical
// orbit (or family of symmetric-equivalent ones) converge to" -- and the
// complete, deduplicated cycle SET is not enough to answer it. Two
// concrete failure modes if a caller tried to read complete_attractors_
// from_seeds' own output for this instead:
//   (a) the algebraic union adds every algebraically-attracting fixed
//       point regardless of which seed (if any) actually reaches it --
//       verified directly (relaxed-Newton-of-z^n-1, one seed at the
//       origin only): seeding with just one UNRELATED critical point
//       still returns every one of the n roots-of-unity fixed points,
//       all via the union, none of them what that seed's own orbit did.
//   (b) a map can have several SIMULTANEOUSLY attracting cycles from
//       DIFFERENT critical points (the whole point of studying bicritical
//       families) -- "the complete set" conflates them; a period-coloring
//       consumer needs to know which cycle EACH tracked critical point's
//       own orbit reached, not the union of all of them.
//
// Returns one SeedOutcome per seed, in the SAME order as `seeds`, each
// computed as if that seed were the only one given to find_attractors_
// from_seeds (seeds are independent in that function -- see its own
// per-seed loop -- so this is not an approximation, just a different way
// of reading the identical computation) PLUS the same algebraic
// reconciliation complete_attractors_from_seeds applies (an unresolved
// seed whose own endpoint lands on an algebraically-attracting fixed
// point is reported resolved, period 1, at the same opts.tol * 1e3
// chordal tolerance) -- so a seed that is genuinely heading to an exact
// fixed point is not falsely reported "undetermined" just because its own
// numerical closure detection missed it.
//
// `period` is measured in the RAW z-plane this seed's orbit actually
// lives in -- literally cyc.points.size() for whichever cycle was found,
// nothing quotiented out. `is_infinity` is set when that cycle is the
// point at infinity (still period 1, but reported separately -- see
// Renderer::render_parameter_period's own doc comment for why a caller
// usually wants to color that differently from an ordinary finite period-
// 1 cycle). `resolved` is false ("undetermined") exactly when this seed's
// orbit is a genuine residual no complete attractor explains -- never a
// fabricated period; `period`/`is_infinity` are both left at their
// zero-value defaults in that case.
struct SeedOutcome {
    int  period      = 0;       // 0 iff !resolved -- never a real period otherwise
    bool is_infinity = false;
    bool resolved    = false;
};

std::vector<SeedOutcome> per_seed_outcomes(const std::vector<Cplx>& seeds, const RationalMap& map,
                                           Cplx a, const FindAttractorsOptions& opts = {});

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
