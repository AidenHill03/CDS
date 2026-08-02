// =============================================================================
// cdx/roots.cpp -- Aberth-Ehrlich complex polynomial root-finder.
// =============================================================================
#include "cdx/roots.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace cdx {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Coefficients below this fraction of the polynomial's own max coefficient
// magnitude are treated as zero when determining the effective degree.
// A couple of orders above machine epsilon, enough to absorb the
// cancellation noise of RationalMap's term arithmetic without discarding a
// genuinely small (but real) leading coefficient from a user-typed formula.
constexpr double kTrimRelTol = 1e-12;

// Per-root correction, relative to the root's own magnitude, below which
// that root is considered converged.
constexpr double kConvRelTol = 1e-12;

constexpr int kMaxIterations = 100;

// Floor used to keep the Newton term and the Aberth repulsion sum finite
// when two estimates coincide or a derivative vanishes, in the same spirit
// as renderer.cpp's kTinyDen guards on near-zero denominators.
constexpr double kTinyDen = 1e-300;

// Evaluates p(z) and p'(z) together via Horner's method / synthetic
// division, one pass over ascending-order coefficients c[0..n].
std::pair<Cplx, Cplx> eval_with_deriv(const std::vector<Cplx>& c, Cplx z) {
    const int n = static_cast<int>(c.size()) - 1;
    Cplx p = c[n];
    Cplx dp(0.0, 0.0);
    for (int k = n - 1; k >= 0; --k) {
        dp = dp * z + p;
        p  = p  * z + c[k];
    }
    return {p, dp};
}

}  // namespace

// -----------------------------------------------------------------------------
int effective_degree(const Polynomial& p) {
    const std::vector<Cplx>& c = p.coeffs;
    double maxabs = 0.0;
    for (const Cplx& v : c) maxabs = std::max(maxabs, std::abs(v));
    if (maxabs == 0.0) return -1;

    const double tol = kTrimRelTol * maxabs;
    for (int k = static_cast<int>(c.size()) - 1; k >= 0; --k) {
        if (std::abs(c[k]) >= tol) return k;
    }
    return -1;
}

// -----------------------------------------------------------------------------
std::vector<Cplx> roots(const Polynomial& poly, bool* converged) {
    if (converged) *converged = true;

    const int deg = effective_degree(poly);
    if (deg <= 0) return {};   // identically zero, or a nonzero constant

    // Trim to the effective degree and normalize to monic form. This does
    // not change the roots (dividing by a nonzero constant is a no-op on
    // the zero set) but simplifies the iteration and the initial-guess
    // radius, both usually stated for monic polynomials.
    std::vector<Cplx> c(poly.coeffs.begin(), poly.coeffs.begin() + deg + 1);
    const Cplx lead = c[deg];
    for (Cplx& v : c) v /= lead;

    const int n = deg;

    // Cauchy's bound for a monic polynomial: every root satisfies
    // |z| <= 1 + max_k |c_k| (k < n). Initial guesses are spread evenly on
    // that circle, offset by half a step so real-coefficient polynomials
    // don't start with symmetric conjugate pairs exactly opposite each
    // other (a known source of stagnation for simultaneous iteration).
    double radius = 1.0;
    for (int k = 0; k < n; ++k) radius = std::max(radius, 1.0 + std::abs(c[k]));

    std::vector<Cplx> z(n);
    for (int k = 0; k < n; ++k) {
        const double theta = 2.0 * kPi * (k + 0.5) / n;
        z[k] = Cplx(radius * std::cos(theta), radius * std::sin(theta));
    }

    bool ok = false;
    for (int iter = 0; iter < kMaxIterations && !ok; ++iter) {
        double max_rel_correction = 0.0;

        for (int i = 0; i < n; ++i) {
            auto [p, dp] = eval_with_deriv(c, z[i]);
            if (std::abs(dp) < kTinyDen) dp = Cplx(kTinyDen, 0.0);
            const Cplx newton = p / dp;

            Cplx repulsion(0.0, 0.0);
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                Cplx diff = z[i] - z[j];
                if (std::abs(diff) < kTinyDen) diff = Cplx(kTinyDen, 0.0);
                repulsion += Cplx(1.0, 0.0) / diff;
            }

            Cplx denom = Cplx(1.0, 0.0) - newton * repulsion;
            if (std::abs(denom) < kTinyDen) denom = Cplx(kTinyDen, 0.0);
            const Cplx correction = newton / denom;

            z[i] -= correction;
            const double rel = std::abs(correction) / std::max(1.0, std::abs(z[i]));
            max_rel_correction = std::max(max_rel_correction, rel);
        }

        if (max_rel_correction < kConvRelTol) ok = true;
    }

    if (converged) *converged = ok;
    return z;
}

}  // namespace cdx
