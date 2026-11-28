// =============================================================================
// cdx/special_polys.hpp -- coefficient tables of the classical polynomials
// the rational-expression grammar exposes as built-in functions
// (hermite(n, x), legendre(n, x), laguerre(n, alpha, x), ... -- see
// rational_parser.hpp's grammar comment for the full list and syntax).
//
// WHAT THIS RETURNS. A polynomial in a formal variable t (ascending powers),
// whose coefficients are themselves polynomials in up to two SYMBOLS -- the
// family's own parameters: alpha (generalized Laguerre, Jacobi), lambda
// (Gegenbauer, stored in alpha's slot) and beta (Jacobi). Keeping them as
// exact-shape polynomials in the symbols, rather than evaluating them, is
// what lets `laguerre(4, a, z)` produce coefficients that are compact
// Horner forms in `a` (O(n) ParamExpr nodes each) instead of the
// exponentially-shared trees a recurrence run directly on ParamExpr would
// build. The parser (rational_parser.cpp) then substitutes its own argument
// expression for t, in the Fraction<PolyZ> algebra.
//
// HOW EACH FAMILY IS COMPUTED.
//   Recurrences (as the request table specified): Hermite H and He,
//   Legendre, Laguerre / generalized Laguerre, Chebyshev T and U,
//   Gegenbauer, Bessel. All divisions in them are by integers (k+1), never
//   by a symbol, so they stay inside the polynomial-in-symbols ring.
//   Jacobi: the explicit binomial sum
//       P_n^(a,b)(t) = sum_s C(n+a, n-s) C(n+b, s) ((t-1)/2)^s ((t+1)/2)^(n-s)
//   instead of the three-term recurrence, because the recurrence divides by
//   2(k+1)(k+a+b+1)(2k+a+b) -- a polynomial in the symbols, which this ring
//   (and ParamExpr, which has no Div node) cannot represent.
//   Bernoulli / Euler: from the Bernoulli numbers B_0..B_(n+1), computed
//   EXACTLY with a small big-integer type (B_k = sum_j (-1)^j j! S(k,j)/(j+1)
//   over a common denominator lcm(1..k+1)), then
//       B_n(t) = sum_j C(n,j) B_(n-j) t^j
//       E_n(t) = sum_j 2/(n+1) C(n+1,j) B_(n+1-j) (1 - 2^(n+1-j)) t^j
//   -- each coefficient a single product, so no cancellation after the one
//   rounding of each exact B_k to floating point.
//   Cyclotomic: x^n - 1 divided exactly (integer synthetic division) by
//   Phi_d for every proper divisor d of n.
//
// Arithmetic is long double (80-bit on GCC/Clang x86, plain double on MSVC
// -- both well within what n <= 40 needs: the largest coefficient magnitude
// is Bessel's ~1e57, the doubles' own range is 1e308).
// =============================================================================
#pragma once

#include <string>
#include <vector>

namespace cdx {
namespace special {

// A polynomial in two symbols s0 (alpha / lambda) and s1 (beta):
// c[i][j] is the coefficient of s0^i s1^j. Never empty (at least {{0}}).
struct SymPoly {
    std::vector<std::vector<long double>> c{{0.0L}};
};

// A polynomial in t with SymPoly coefficients, ascending powers. May have
// trailing all-zero coefficients only for the identically-zero polynomial.
using TPoly = std::vector<SymPoly>;

enum class Family {
    Hermite, HermiteE, Legendre, Laguerre, Gegenbauer, Jacobi,
    ChebyshevT, ChebyshevU, Bessel, Bernoulli, Euler, Cyclotomic,
};

constexpr int kMaxDegree = 40;   // hard limit (parse error above)
constexpr int kWarnDegree = 25;  // parse WARNING above (coefficient growth)

// The degree-n member of `family` (n already validated: 0 <= n <= kMaxDegree,
// and n >= 1 for Cyclotomic). For Laguerre, s0 is alpha (plain Laguerre is
// the caller substituting alpha = 0); for Gegenbauer s0 is lambda; for
// Jacobi s0 is alpha and s1 is beta. Families without parameters return
// constant SymPolys.
TPoly special_poly(Family family, int n);

// Exact Bernoulli number B_k (B_1 = -1/2 convention), rounded once to long
// double. Exposed for tests.
long double bernoulli_number(int k);

// TRUE iff the SymPoly is the constant polynomial (no s0/s1 dependence).
bool is_constant(const SymPoly& p);

}  // namespace special
}  // namespace cdx
