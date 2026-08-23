// =============================================================================
// test_rational_parser.cpp -- oracle test for cdx::parse_rational.
//
// THE ORACLE. A completely SEPARATE, direct recursive-descent evaluator
// (DirectEval below) implements the same grammar but interprets it straight
// to a Cplx value for one (z, parameter-binding) pair, with no polynomial/
// fraction structure at all. It shares no code with rational_parser.cpp's
// own Fraction<PolyZ> algebra, so a bug in THAT algebra (a wrong sign in
// frac_sub, a degree-bookkeeping slip in poly_mul, ...) has no reason to be
// replicated here -- agreement between the two, at many random points, is
// real evidence the canonical P/Q form means what the source expression
// means, not just that the same code was asked twice.
// =============================================================================
#include "cdx/rational_parser.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

// -----------------------------------------------------------------------------
// Independent direct-evaluation oracle -- see file header.
// -----------------------------------------------------------------------------
namespace {

struct DirectEval {
    const std::string& src;
    std::size_t pos = 0;
    Cplx z;
    const std::map<std::string, Cplx>& params;

    void skip_ws() { while (pos < src.size() && std::isspace((unsigned char)src[pos])) ++pos; }
    bool accept(char c) {
        skip_ws();
        if (pos < src.size() && src[pos] == c) { ++pos; return true; }
        return false;
    }

    Cplx expr() {
        Cplx v = term();
        for (;;) {
            skip_ws();
            if (accept('+')) v += term();
            else if (accept('-')) v -= term();
            else return v;
        }
    }
    Cplx term() {
        Cplx v = factor();
        for (;;) {
            skip_ws();
            if (accept('*')) v *= factor();
            else if (accept('/')) v /= factor();
            else return v;
        }
    }
    Cplx factor() {
        Cplx b = unary();
        skip_ws();
        if (accept('^')) {
            int n = int_exp();
            const bool inv = n < 0;
            int an = inv ? -n : n;
            Cplx acc(1.0, 0.0);
            for (int i = 0; i < an; ++i) acc *= b;
            return inv ? Cplx(1.0, 0.0) / acc : acc;
        }
        return b;
    }
    int int_exp() {
        skip_ws();
        const bool paren = accept('(');
        skip_ws();
        bool neg = false;
        if (accept('-')) neg = true; else accept('+');
        skip_ws();
        const std::size_t start = pos;
        while (pos < src.size() && std::isdigit((unsigned char)src[pos])) ++pos;
        long v = std::strtol(src.substr(start, pos - start).c_str(), nullptr, 10);
        if (paren) accept(')');
        return static_cast<int>(neg ? -v : v);
    }
    Cplx unary() {
        skip_ws();
        if (accept('-')) return -unary();
        if (accept('+')) return unary();
        return atom();
    }
    Cplx atom() {
        skip_ws();
        const char c = src[pos];
        if (c == '(') { ++pos; Cplx v = expr(); accept(')'); return v; }
        if (std::isdigit((unsigned char)c) || c == '.') {
            const char* start = src.c_str() + pos;
            char* end = nullptr;
            double v = std::strtod(start, &end);
            pos += static_cast<std::size_t>(end - start);
            if (pos < src.size() && (src[pos] == 'i' || src[pos] == 'j')) { ++pos; return Cplx(0.0, v); }
            return Cplx(v, 0.0);
        }
        const std::size_t start = pos;
        while (pos < src.size() && std::isalnum((unsigned char)src[pos])) ++pos;
        const std::string id = src.substr(start, pos - start);
        if (id == "z") return z;
        if (id == "i" || id == "j") return Cplx(0.0, 1.0);
        auto it = params.find(id);
        return it == params.end() ? Cplx(0.0, 0.0) : it->second;
    }
};

Cplx direct_eval(const std::string& src, Cplx z, const std::map<std::string, Cplx>& params) {
    DirectEval ev{src, 0, z, params};
    return ev.expr();
}

bool close(Cplx a, Cplx b, double tol) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) < tol * scale;
}

// Parses `source`, then compares CanonicalRational::eval against direct_eval
// at `n_samples` random (z, params) points -- skipping points where either
// side blows up (near a pole/singularity, where floating-point noise alone
// can dominate) rather than papering over a real mismatch elsewhere.
void oracle_check(const std::string& source, const std::vector<std::string>& expected_params,
                  int n_samples = 200) {
    std::printf("\n%s\n", source.c_str());
    CanonicalRational cr;
    std::string error;
    const bool ok = parse_rational(source, cr, error);
    check(ok, "parses successfully");
    if (!ok) { std::printf("    error: %s\n", error.c_str()); return; }

    check(cr.parameters == expected_params, "parameter list matches expected");
    if (cr.parameters != expected_params) {
        std::printf("    got:");
        for (const auto& p : cr.parameters) std::printf(" %s", p.c_str());
        std::printf("\n    want:");
        for (const auto& p : expected_params) std::printf(" %s", p.c_str());
        std::printf("\n");
    }

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> unif(-2.5, 2.5);

    int n_checked = 0, n_mismatch = 0;
    for (int i = 0; i < n_samples; ++i) {
        const Cplx z{unif(rng), unif(rng)};
        std::map<std::string, Cplx> params;
        for (const auto& name : cr.parameters) params[name] = Cplx(unif(rng), unif(rng));

        const Cplx want = direct_eval(source, z, params);
        if (!std::isfinite(want.real()) || !std::isfinite(want.imag()) || std::abs(want) > 1e8) continue;
        const Cplx got = cr.eval(z, params);
        if (!std::isfinite(got.real()) || !std::isfinite(got.imag()) || std::abs(got) > 1e8) continue;

        ++n_checked;
        if (!close(got, want, 1e-6)) {
            ++n_mismatch;
            if (n_mismatch <= 3) {
                std::printf("    MISMATCH z=(%.4f%+.4fi) got=(%.9f%+.9fi) want=(%.9f%+.9fi)\n",
                           z.real(), z.imag(), got.real(), got.imag(), want.real(), want.imag());
            }
        }
    }
    std::printf("    checked %d/%d finite sample points, %d mismatches\n", n_checked, n_samples,
               n_mismatch);
    check(n_checked > n_samples / 4, "sanity: most sample points were usable (not all near a pole)");
    check(n_mismatch == 0, "canonical P/Q evaluates identically to the source expression, "
          "at every usable sample point");
}

}  // namespace

int main() {
    std::printf("=== cdx rational-parser tests ===\n");

    std::printf("\nORACLE: canonical P/Q matches direct evaluation of the source:\n");
    oracle_check("z^2 + a", {"a"});                                  // mandelbrot
    oracle_check("z^3 + a/z^3", {"a"});                              // mcmullen(3)
    oracle_check("z^5 + a/z^5", {"a"});                              // mcmullen(5)
    oracle_check("z - (z^3-1)/(3*z^2)", {});                         // newton cubic (no free param)
    oracle_check("(2/3)*z + (1/3)/z^2 + a", {"a"});                  // nova
    oracle_check("a*z^3 - b*z + c", {"a", "b", "c"});                // multi-parameter
    oracle_check("1/(1/(z+1) + a)", {"a"});                          // nested rational
    oracle_check("(z + 1/(z-a)) / (z - 1/(z+b))", {"a", "b"});       // deeper nesting, 2 params
    oracle_check("z^-2 + a*z^-1", {"a"});                            // negative exponents
    oracle_check("(a+b)*z^2 + (a-b)", {"a", "b"});                   // parameter-only coefficients
    oracle_check("2i*z + 3.5e-1", {});                               // scientific notation, 2i shorthand

    // ---- collapse to a single fraction, exactly (the milestone's own example) ----
    std::printf("\nCOLLAPSE: z^n + a/z^n normalizes to (z^2n+a)/z^n exactly:\n");
    {
        CanonicalRational cr;
        std::string error;
        check(parse_rational("z^3 + a/z^3", cr, error), "parses");
        check(cr.P.coeffs.size() == 7, "P has degree 6 (z^6+a): 7 coefficients");
        check(cr.Q.coeffs.size() == 4, "Q has degree 3 (z^3): 4 coefficients");
        std::map<std::string, Cplx> params{{"a", Cplx(1.0, 0.0)}};
        // P should be [a, 0, 0, 0, 0, 0, 1] (a + z^6); Q should be [0,0,0,1] (z^3).
        check(cr.P.coeffs[0]->eval(params) == Cplx(1.0, 0.0), "P's constant term is exactly a (=1 here)");
        check(cr.P.coeffs[6]->eval({}) == Cplx(1.0, 0.0), "P's z^6 coefficient is exactly 1");
        for (int k = 1; k < 6; ++k) {
            check(cr.P.coeffs[static_cast<std::size_t>(k)]->is_zero_literal(),
                  "P's intermediate coefficients are structurally zero");
        }
        check(cr.Q.coeffs[3]->eval({}) == Cplx(1.0, 0.0), "Q's z^3 coefficient is exactly 1");
        for (int k = 0; k < 3; ++k) {
            check(cr.Q.coeffs[static_cast<std::size_t>(k)]->is_zero_literal(),
                  "Q's lower coefficients are structurally zero");
        }
    }

    // ---- parameter list: correct, sorted, deduplicated ---------------------
    std::printf("\nparameter-list extraction:\n");
    {
        CanonicalRational cr;
        std::string error;
        check(parse_rational("z^2 + a", cr, error) && cr.parameters == std::vector<std::string>{"a"},
              "single parameter");
        check(parse_rational("z - (z^3-1)/(3*z^2)", cr, error) && cr.parameters.empty(),
              "no free parameters at all");
        check(parse_rational("c*z^2 + b*z + a", cr, error) &&
              cr.parameters == std::vector<std::string>{"a", "b", "c"},
              "multiple parameters come back SORTED, not in appearance order");
        check(parse_rational("a*z + a*a", cr, error) && cr.parameters == std::vector<std::string>{"a"},
              "a repeated parameter is deduplicated to one entry");
        check(parse_rational("z + pi + e", cr, error) &&
              cr.parameters == std::vector<std::string>{"e", "pi"},
              "'pi'/'e' are ordinary parameters here, not built-in constants (see header's own "
              "divergence-from-Expr note)");
    }

    // ---- rejection: transcendental functions, unknown symbols, malformed input --
    std::printf("\nrejection: transcendental functions, unknown symbols, malformed input:\n");
    {
        CanonicalRational cr;
        std::string error;
        check(!parse_rational("sin(z) + a", cr, error), "sin(z) is rejected");
        check(error.find("transcendental") != std::string::npos,
              "...with a message specifically calling out 'transcendental', not a generic parse error");
        check(!parse_rational("exp(a*z)", cr, error), "exp(...) is rejected");
        check(!parse_rational("sqrt(z)", cr, error), "sqrt(...) is rejected (also transcendental here)");

        check(!parse_rational("foo(z)", cr, error), "an unrecognized function call is rejected");
        check(error.find("transcendental") == std::string::npos,
              "...with a DIFFERENT message than the transcendental case (not a rational map "
              "vs not a supported construct at all)");

        check(!parse_rational("z^a", cr, error), "a PARAMETER exponent is rejected (must be a "
              "literal integer)");
        check(!parse_rational("z^1.5", cr, error), "a non-integer exponent is rejected");
        check(!parse_rational("z^^2", cr, error), "malformed exponent syntax is rejected");

        check(!parse_rational("z +", cr, error), "trailing operator is rejected");
        check(!parse_rational("(z + 1", cr, error), "unmatched '(' is rejected");
        check(!parse_rational("z + 1)", cr, error), "unmatched ')' is rejected");
        check(!parse_rational("", cr, error), "empty input is rejected");
        check(!parse_rational("z # a", cr, error), "an unexpected character is rejected");

        check(!parse_rational("5/(z-z)", cr, error), "a literal zero denominator is rejected");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
