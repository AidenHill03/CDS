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

}  // namespace cdx
