// =============================================================================
// cdx/roots.hpp -- complex polynomial root-finder (Aberth-Ehrlich).
//
// WHY THIS EXISTS. A RationalMap's critical points are wherever its analytic
// derivative vanishes. Built-in families get that in closed form; a
// user-edited RationalMap only has the derivative as another rational
// function, so its critical points have to be found numerically: clear
// denominators, and root the resulting polynomial.
//
// ALGORITHM. Aberth-Ehrlich: simultaneous iteration over all n roots at once,
// each correction weighted by a Newton term and a repulsion term from the
// other current estimates. Cubic convergence once estimates are close, and
// noticeably more robust than Durand-Kerner on polynomials with
// closely-spaced roots -- the common case here, since a near-degenerate
// parameter can push two critical points close together.
//
// COEFFICIENT ORDER. Ascending: coeffs[k] is the coefficient of z^k, so
// coeffs.back() is nominally the leading (highest-degree) term. This matches
// how RationalMap builds these polynomials: a term's `exponent` maps
// directly to an array index, with no reversal step.
// =============================================================================
#pragma once

#include <complex>
#include <vector>

namespace cdx {

using Cplx = std::complex<double>;

// coeffs[k] * z^k, for k = 0 .. coeffs.size()-1. Trailing (high-index)
// entries that are negligible relative to the polynomial's own scale are
// trimmed internally before root-finding; see roots()'s documentation.
struct Polynomial {
    std::vector<Cplx> coeffs;
};

// Highest index k with |coeffs[k]| non-negligible relative to the
// polynomial's own scale, or -1 if every coefficient is (identically or
// relatively) zero. Exposed -- not just an internal step of roots() --
// because callers building their own polynomials (e.g. RationalMap,
// determining the true degree of a numerator or denominator it just
// constructed) need the exact same "is this coefficient real or
// cancellation noise" judgment call that roots() makes internally.
int effective_degree(const Polynomial& p);

// Finds all roots of `p` via the Aberth-Ehrlich method.
//
// Returns exactly effective_degree(p) roots, WITH MULTIPLICITY: a genuine
// double root comes back as two numerically-close (not necessarily
// bit-identical) estimates rather than being collapsed to one. Deciding what
// counts as "close enough to be the same root" is left to the caller, since
// that depends on what the roots are being used for.
//
// Polynomials of effective degree <= 0 -- identically zero, or a nonzero
// constant -- have no roots; an empty vector is returned for both.
//
// `converged` (optional, same shape as Renderer::render_greens's
// `normalized`): set to false if the iteration hit its cap before every
// root's correction fell below the convergence tolerance. The returned
// values are still the best estimates found, just not guaranteed to be as
// precise as a converged run -- this happens mainly for polynomials with
// genuine multiple roots, where convergence degrades from cubic to linear.
std::vector<Cplx> roots(const Polynomial& p, bool* converged = nullptr);

// -----------------------------------------------------------------------------
// Spatial continuation's own numerical primitive -- a genuinely SMALLER,
// DIFFERENT thing than roots() above: plain (non-simultaneous) Newton
// iteration from a single already-good starting point, no Aberth repulsion
// term and no Cauchy-bound initial circle. WHY THIS EXISTS: a caller
// rendering a grid of nearby parameter values (e.g. Renderer::render_
// parameter_rational/render_parameter_basin) already has, at each pixel, an
// excellent starting guess for each root -- the corresponding root the
// IMMEDIATELY PRECEDING pixel already found -- and refining from there with
// a couple of Newton steps is far cheaper than roots()'s full cold
// simultaneous solve. Deciding what counts as a SOUND continuation (root
// count unchanged, no collision with a pole, no two roots collapsing
// together) is caller context roots.hpp does not have (it knows nothing
// about poles); see RationalMap's own continuation entry points
// (distinct_critical_points_continued, fixed_points_continued) for that
// policy -- this function only ever answers "did THIS ONE guess converge."
//
// DELIBERATELY LOOSER convergence bar than roots()'s own kConvRelTol
// (1e-12): that value is calibrated for a cubically-convergent (once
// close) SIMULTANEOUS iteration run up to 100 times, and holding plain,
// single-root Newton -- only quadratically convergent, and capped at a
// handful of steps -- to the same bar would make it fail to converge from
// realistic predictors almost every time, defeating the entire point of a
// cheap warm start. Measured directly: a few Newton steps from an
// already-close predictor reliably lands around 1e-9 to 1e-7 relative
// correction, essentially never 1e-12 -- and nothing downstream needs
// that: this project's own working tolerances (RenderSettings::tol's
// chordal membership, distinct_critical_points' own dedup rel_tol) sit at
// 1e-4 to 1e-6, so a root known to 1e-9 has ample margin below anything a
// continued root actually has to satisfy. Same kTinyDen guard on a
// vanishing derivative as roots() uses. Operates directly on `p`'s own
// (trimmed) coefficients; unlike roots(), no monic normalization is
// needed or performed -- Newton's own correction f(z)/f'(z) is invariant
// under scaling f by a nonzero constant, so it would be pure wasted work
// here.
struct NewtonRefineResult {
    Cplx z;
    bool converged = false;
};
NewtonRefineResult newton_refine(const Polynomial& p, Cplx guess, int max_iters = 4);

}  // namespace cdx
