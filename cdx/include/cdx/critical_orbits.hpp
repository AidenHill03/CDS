// cdx/critical_orbits.hpp -- parameter planes from the free critical orbits
// alone.
//
// WHY. Every attracting cycle of a rational map attracts at least one
// critical point (Fatou). A structurally pre-fixed critical point (see
// FreeCriticalStructure) does so for every parameter at once, so its
// attractor is known in closed form; every OTHER attracting cycle must
// attract a FREE critical point. Following the free critical orbits with
// inline period detection therefore finds every attracting cycle that
// matters at a parameter pixel, and tells which one each critical point
// lands on -- with no fixed-point solve, no attractor discovery, no
// algebraic union and no reconciliation sweep. Those (complete_attractors
// and friends, analysis.hpp) stay for SINGLE-parameter uses: the Facts
// panel, dynamical-plane basin, overlays.
//
// THE PER-ORBIT METHOD (track_critical_orbit), one iteration loop:
//   * Escape to infinity, when infinity attracts at this parameter:
//       - a certified polynomial: |z| > R, R the certified radius (the
//         escape-time test, unchanged);
//       - a rational map: infinity's status is read off the compiled
//         coefficients (InfinityStatus): superattracting when deg P - deg Q
//         >= 2, multiplier lead(Q)/lead(P) when the difference is 1, not
//         fixed otherwise. The orbit has arrived once its chordal distance
//         to infinity is below `tol`.
//   * Otherwise, periodicity checking (Brent): a reference point z_ref is
//     reset at iterations 1, 2, 4, ..., 128, then every 128 (see kMaxRefAge
//     in critical_orbits.cpp for why the age is capped); when the chordal distance from
//     z_n to z_ref drops below `closure_eps`, p = n - n_ref is a period
//     candidate. p is reduced to its minimal divisor q with f^q(z_ref)
//     still within closure_eps, a few Newton steps on f^q(z) - z refine the
//     cycle point, the refined cycle is reduced AGAIN to its minimal period
//     (an orbit spiralling into a fixed point with multiplier ~e^(2 pi i/3)
//     closes at p = 3 long before it closes at p = 1; Newton lands on the
//     fixed point, and only the refined points show q = 1), and the
//     multiplier lambda = prod f'(z_i) decides: attracting iff
//     |lambda| < 1 - attracting_margin. A candidate that fails is
//     forgotten and the orbit continues.
//   * Neutral, parabolic, Siegel, a pole hit, or out of budget:
//     Undetermined. Never a fabricated cycle.
//
// SMOOTH VALUE (CriticalOrbitOutcome::nu) -- the attracting analogue of the
// smooth escape count, so `parameter`'s colouring keeps its meaning (the
// fractional iteration at which the orbit comes within `tol` of where it
// is going):
//   (Origin: an orbit exactly at distance tol at step n gets n - 1, the
//   reconciled path's interpolation convention, so colours keep their meaning.)
//   * geometric cycle (|lambda| > 0): nu = n - 1 + q log(tol / d_n) / log|lambda|,
//     d_n the chordal distance from z_n to the nearest refined cycle point.
//     It is the same at every n in the linear regime (d_{n+q} = |lambda| d_n),
//     so it is evaluated at the detection step.
//   * superattracting (|lambda| below kSuperattracting): the log-log form
//     nu = n - 1 + q log_k(log tol / log d_n), k = 2 (a simple critical point on
//     the cycle).
//   * infinity: the escape analogue nu = n - 1 - log_k(log|z_n| / log R_inf),
//     k = deg P - deg Q, R_inf = 2 / tol (chordal distance tol from
//     infinity); geometric infinity uses the cycle formula with d the
//     chordal distance to infinity.
// -----------------------------------------------------------------------------
#pragma once

#include <vector>

#include "cdx/rational.hpp"

namespace cdx {

// Infinity's dynamical status for ONE compiled map (one parameter value).
struct InfinityStatus {
    bool fixed = false;       // f(inf) == inf
    int  local_degree = 1;    // >= 2: superattracting with this local degree
    Cplx multiplier{1.0, 0.0};// in the w = 1/z chart; 0 when superattracting
    Cplx image{0.0, 0.0};     // f(inf) when !fixed
    bool attracting() const { return fixed && (local_degree >= 2 || std::abs(multiplier) < 1.0); }
};

InfinityStatus infinity_status(const CompiledMap& f);

// f(z) and f'(z) for a compiled map, in std::complex (cold path: Newton
// refinement and multipliers, a few dozen calls per orbit, not per step).
// A pole maps to (1e300, 0), the same sentinel CompiledMap::step uses.
Cplx compiled_eval(const CompiledMap& f, Cplx z);
Cplx compiled_deriv(const CompiledMap& f, Cplx z);

struct CriticalOrbitOptions {
    int    budget = 200;               // iterations before "out of budget"
    double tol = 1e-6;                 // chordal "arrived" distance (the smooth value's threshold)
    // Chordal Brent closure distance. Only a CANDIDATE: Newton refinement,
    // the refined closure check and the multiplier decide. Loose on purpose
    // (the reconciled path's weakly-attracting confirmation used the same
    // 1e-4): a tight one cannot close a slowly attracting orbit in budget.
    double closure_eps = 1e-4;
    double attracting_margin = 1e-6;   // |lambda| < 1 - margin to count as attracting
    int    newton_iterations = 30;
    // Escape to infinity for a certified polynomial: |z|^2 > R2 (R2 <= 0:
    // not a certified polynomial; infinity is handled from InfinityStatus).
    double poly_R2 = 0.0;
    int    poly_degree = 2;
    double poly_offset = 0.0;          // log|a_d| / (d - 1), see renderer.cpp's PolyEscape
};

struct CriticalOrbitOutcome {
    enum class Kind { Undetermined, Infinity, Cycle };
    Kind kind = Kind::Undetermined;
    int  period = 0;                   // 1 for infinity
    std::vector<Cplx> cycle;           // refined cycle points (period entries); {inf} for infinity
    Cplx multiplier{1.0, 0.0};
    double nu = 0.0;                   // smooth value (see header); 0 when undetermined
    int  steps = 0;                    // iterations used
    bool resolved() const { return kind != Kind::Undetermined; }
};

// Follows ONE critical orbit from c0 under `f` (see the header comment).
// `inf` is infinity's status for the same map (ignored when opts.poly_R2 > 0).
CriticalOrbitOutcome track_critical_orbit(const CompiledMap& f, const InfinityStatus& inf, Cplx c0,
                                          const CriticalOrbitOptions& opts);

// TRUE if a and b reached the same attractor: both infinity, or the same
// period with a point of one chordally within `tol` of the first point of
// the other.
bool same_attractor(const CriticalOrbitOutcome& a, const CriticalOrbitOutcome& b, double tol);

}  // namespace cdx
