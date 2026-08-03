// =============================================================================
// test_roots.cpp -- property-based checks for the Aberth-Ehrlich root-finder.
//
// Mirrors test_renderer.cpp's style: checks against polynomials whose roots
// are known in closed form (via residual and set-matching), not golden
// output. Coefficients throughout are in cdx::Polynomial's ascending order:
// coeffs[k] is the coefficient of z^k.
// =============================================================================
#include "cdx/roots.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static Cplx eval_poly(const std::vector<Cplx>& c, Cplx z) {
    Cplx p(0.0, 0.0);
    for (int k = static_cast<int>(c.size()) - 1; k >= 0; --k) p = p * z + c[k];
    return p;
}

// Order-independent comparison: every expected root must have a distinct
// discovered root within tol, and the counts must match.
static bool sets_match(const std::vector<Cplx>& discovered,
                        const std::vector<Cplx>& expected, double tol) {
    if (discovered.size() != expected.size()) return false;
    std::vector<bool> used(discovered.size(), false);
    for (Cplx e : expected) {
        int best = -1;
        double best_d = 1e300;
        for (std::size_t i = 0; i < discovered.size(); ++i) {
            if (used[i]) continue;
            const double d = std::abs(discovered[i] - e);
            if (d < best_d) { best_d = d; best = static_cast<int>(i); }
        }
        if (best < 0 || best_d > tol) return false;
        used[static_cast<std::size_t>(best)] = true;
    }
    return true;
}

static double max_residual(const std::vector<Cplx>& c, const std::vector<Cplx>& r) {
    double m = 0.0;
    for (Cplx z : r) m = std::max(m, std::abs(eval_poly(c, z)));
    return m;
}

int main() {
    std::printf("=== cdx roots tests ===\n");

    // ---- z^3 - 1: the cube roots of unity ------------------------------------
    std::printf("\nz^3 - 1:\n");
    {
        Polynomial p{{{-1, 0}, {0, 0}, {0, 0}, {1, 0}}};
        bool converged = false;
        auto r = roots(p, &converged);
        check(converged, "converges");
        check(r.size() == 3, "3 roots for a cubic");
        const std::vector<Cplx> expected = {
            {1.0, 0.0}, {-0.5, 0.8660254037844386}, {-0.5, -0.8660254037844386}};
        check(sets_match(r, expected, 1e-9), "roots are the cube roots of unity");
        check(max_residual(p.coeffs, r) < 1e-9, "residual |p(root)| is tiny");
    }

    // ---- z^4 - a for a couple of parameter values ----------------------------
    std::printf("\nz^4 - a:\n");
    for (Cplx a : {Cplx(16.0, 0.0), Cplx(1.0, 2.0)}) {
        Polynomial p{{-a, {0, 0}, {0, 0}, {0, 0}, {1, 0}}};
        bool converged = false;
        auto r = roots(p, &converged);
        check(converged, "converges");
        check(r.size() == 4, "4 roots for a quartic");
        bool fourth_powers_match = true;
        for (Cplx z : r) {
            const Cplx z4 = z * z * z * z;
            if (std::abs(z4 - a) > 1e-8) fourth_powers_match = false;
        }
        check(fourth_powers_match, "every root's 4th power recovers a");
        check(max_residual(p.coeffs, r) < 1e-8, "residual |p(root)| is tiny");
    }

    // ---- (z-1)(z-2)(z+3) = z^3 + 0z^2 - 7z + 6 --------------------------------
    std::printf("\n(z-1)(z-2)(z+3):\n");
    {
        Polynomial p{{{6, 0}, {-7, 0}, {0, 0}, {1, 0}}};
        auto r = roots(p);
        const std::vector<Cplx> expected = {{1, 0}, {2, 0}, {-3, 0}};
        check(sets_match(r, expected, 1e-8), "roots are 1, 2, -3");
    }

    // ---- (z-2)^3: a genuine triple root ---------------------------------------
    // Convergence degrades from cubic to linear on a true multiple root, so
    // this only checks that the iteration lands in the neighbourhood, not
    // that it reaches the same precision as the well-separated cases above.
    std::printf("\n(z-2)^3 (triple root):\n");
    {
        Polynomial p{{{-8, 0}, {12, 0}, {-6, 0}, {1, 0}}};
        auto r = roots(p);
        check(r.size() == 3, "3 roots (with multiplicity) for a cubic");
        bool all_near_two = true;
        for (Cplx z : r)
            if (std::abs(z - Cplx(2.0, 0.0)) > 1e-3) all_near_two = false;
        check(all_near_two, "all three estimates cluster near the true root");
    }

    // ---- degree <= 0: no roots --------------------------------------------------
    std::printf("\ndegenerate inputs have no roots:\n");
    check(roots(Polynomial{{}}).empty(), "empty coefficient list");
    check(roots(Polynomial{{{5, 0}}}).empty(), "nonzero constant polynomial");
    check(roots(Polynomial{{{0, 0}}}).empty(), "identically zero polynomial");

    // ---- leading (high-index) near-zero coefficients are trimmed --------------
    std::printf("\nleading near-zero coefficients are trimmed:\n");
    {
        // Nominally cubic, but the z^3 coefficient is negligible: behaves
        // like z^2 - 1, not like a cubic with a runaway third root.
        Polynomial p{{{-1, 0}, {0, 0}, {1, 0}, {1e-15, 0}}};
        auto r = roots(p);
        const std::vector<Cplx> expected = {{1, 0}, {-1, 0}};
        check(r.size() == 2, "trimmed to degree 2");
        check(sets_match(r, expected, 1e-6), "roots match z^2-1, not a spurious cubic");
    }

    // ---- linear polynomial: n=1 has no repulsion term, still works -----------
    std::printf("\nlinear polynomial:\n");
    {
        Polynomial p{{{3, 0}, {2, 0}}};   // 2z + 3 = 0  =>  z = -1.5
        bool converged = false;
        auto r = roots(p, &converged);
        check(converged, "converges");
        check(r.size() == 1 && std::abs(r[0] - Cplx(-1.5, 0.0)) < 1e-12,
              "2z+3 has the single root -1.5");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
