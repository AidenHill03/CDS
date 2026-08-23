// =============================================================================
// test_rational_params.cpp -- Stage 3 of the P/Q milestone: single active-
// parameter selection with constant substitution, and authored-source
// preservation through serialize()/deserialize().
// =============================================================================
#include "cdx/rational.hpp"
#include "cdx/rational_parser.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static bool close(Cplx a, Cplx b, double tol = 1e-6) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) < tol * scale;
}

int main() {
    std::printf("=== cdx RationalMap::from_expression (Stage 3) tests ===\n");

    // ---- exactly one parameter -> auto-active ------------------------------------
    std::printf("\none-parameter input auto-selects:\n");
    {
        RationalMap m = RationalMap::from_expression("z^2 + a", "", {}, "auto1");
        check(m.is_pq_backed(), "sanity: built P/Q-backed");
        check(m.pq_active_param() == "a", "the single parsed parameter ('a') is auto-active "
              "with active_param left empty");
        check(m.pq_fixed_params().empty(), "nothing was substituted (there was only one "
              "parameter, and it's the active one)");
        check(close(m.eval(Cplx{0.5, -0.3}, Cplx{0.1, 0.2}),
                    Cplx{0.5, -0.3} * Cplx{0.5, -0.3} + Cplx{0.1, 0.2}),
              "computes z^2+a correctly with the auto-selected active parameter");
    }
    {
        // Explicitly naming the one-and-only parameter is also accepted
        // (not just leaving active_param empty).
        RationalMap m = RationalMap::from_expression("z^2 + a", "a", {}, "auto2");
        check(m.pq_active_param() == "a", "explicitly naming the single parameter also works");
    }
    {
        // Zero parameters -- Newton's method typed directly, no `a` at all.
        RationalMap m = RationalMap::from_expression("z - (z^3-1)/(3*z^2)", "", {}, "newton");
        check(m.pq_active_param().empty(), "a genuinely zero-parameter source has NO active "
              "parameter (not an error -- see from_expression's own doc comment)");
        check(m.pq_fixed_params().empty(), "...and nothing was substituted");
    }

    // ---- multi-parameter substitution matches the fully-substituted expression ----
    std::printf("\nmulti-parameter substitution: dynamics match the fully-substituted "
               "expression:\n");
    {
        const std::string src = "a*z^3 - b*z + c";
        const Cplx b_val{0.9, 0.0}, c_val{0.1, -0.2};
        RationalMap m = RationalMap::from_expression(src, "a", {{"b", b_val}, {"c", c_val}},
                                                      "cubic_sub");
        check(m.pq_active_param() == "a", "'a' stays the active (live) parameter");
        check(m.pq_fixed_params().size() == 2 &&
              close(m.pq_fixed_params().at("b"), b_val) &&
              close(m.pq_fixed_params().at("c"), c_val),
              "pq_fixed_params() reports the exact substituted values for b and c");

        // Independent oracle: parse the SAME source once via Stage 1 directly
        // (not through from_expression at all) and evaluate it with ALL
        // THREE parameters bound explicitly -- this exercises a completely
        // different code path (CanonicalRational::eval with a 3-entry
        // binding) than from_expression's own substitute-then-reduce one.
        CanonicalRational cr;
        std::string error;
        check(parse_rational(src, cr, error) && cr.parameters.size() == 3,
              "sanity: the oracle's own parse sees all three parameters");

        std::mt19937 rng(4242);
        std::uniform_real_distribution<double> unif(-2.0, 2.0);
        int n_mismatch = 0;
        for (int i = 0; i < 100; ++i) {
            const Cplx z{unif(rng), unif(rng)};
            const Cplx a_val{unif(rng), unif(rng)};
            const Cplx want = cr.eval(z, {{"a", a_val}, {"b", b_val}, {"c", c_val}});
            const Cplx got = m.eval(z, a_val);
            if (!close(got, want, 1e-6)) ++n_mismatch;
        }
        check(n_mismatch == 0, "from_expression's substituted-and-reduced map matches the "
              "independently-evaluated fully-bound original at 100 random (z,a) points");

        // deriv() too, not just eval() -- substitution must not silently
        // break the closed-form derivative.
        int n_deriv_mismatch = 0;
        for (int i = 0; i < 50; ++i) {
            const Cplx z{unif(rng), unif(rng)};
            const Cplx a_val{unif(rng), unif(rng)};
            // Finite-difference oracle for the derivative of the FULLY BOUND
            // original expression (independent of both from_expression AND
            // RationalMap::deriv's own closed-form implementation).
            const double h = 1e-6;
            const Cplx f0 = cr.eval(z, {{"a", a_val}, {"b", b_val}, {"c", c_val}});
            const Cplx f1 = cr.eval(z + Cplx(h, 0), {{"a", a_val}, {"b", b_val}, {"c", c_val}});
            const Cplx want = (f1 - f0) / h;
            const Cplx got = m.deriv(z, a_val);
            if (!close(got, want, 1e-3)) ++n_deriv_mismatch;
        }
        check(n_deriv_mismatch == 0, "deriv() matches a finite-difference oracle on the "
              "fully-bound original, at 50 random (z,a) points");
    }

    // ---- serialize()/deserialize(): authored string AND dynamics survive ----------
    std::printf("\nsaved+reloaded family: recovers its authored string AND computes "
               "identically:\n");
    {
        const std::string src = "a*z^3 - b*z + c";
        const Cplx b_val{0.9, 0.0}, c_val{0.1, -0.2};
        RationalMap m = RationalMap::from_expression(src, "a", {{"b", b_val}, {"c", c_val}},
                                                      "roundtrip");
        const std::string blob = m.serialize();
        RationalMap loaded;
        std::string error;
        check(RationalMap::deserialize(blob, loaded, error), "deserializes without error");
        check(loaded.name() == "roundtrip", "...name preserved");
        check(loaded.is_pq_backed(), "...comes back P/Q-backed");
        check(loaded.pq_source() == src, "...recovers the ORIGINAL multi-parameter authored "
              "string EXACTLY ('a*z^3 - b*z + c'), not the reduced single-parameter form");
        check(loaded.to_formula() == src, "...and to_formula() agrees");
        check(loaded.pq_active_param() == "a", "...recovers which parameter was active");
        check(loaded.pq_fixed_params().size() == 2 &&
              close(loaded.pq_fixed_params().at("b"), b_val) &&
              close(loaded.pq_fixed_params().at("c"), c_val),
              "...recovers the exact substituted values for b and c");

        std::mt19937 rng(99);
        std::uniform_real_distribution<double> unif(-2.0, 2.0);
        int n_mismatch = 0;
        for (int i = 0; i < 50; ++i) {
            const Cplx z{unif(rng), unif(rng)};
            const Cplx a_val{unif(rng), unif(rng)};
            if (!close(loaded.eval(z, a_val), m.eval(z, a_val))) ++n_mismatch;
        }
        check(n_mismatch == 0, "...and computes IDENTICALLY to the original at 50 random "
              "(z,a) points");
    }
    {
        // A round trip through FamilyLibrary too (the sandbox's own save
        // file format), not just a single RationalMap::serialize() call.
        RationalMap m = RationalMap::from_expression("a*z^2 + b", "a", {{"b", Cplx{0.25, 0.0}}},
                                                      "lib_member");
        FamilyLibrary lib;
        lib.add(m);
        const std::string blob = lib.serialize();
        FamilyLibrary loaded;
        std::string error;
        check(FamilyLibrary::deserialize(blob, loaded, error), "a library containing a P/Q "
              "multi-parameter map round-trips through FamilyLibrary::serialize/deserialize");
        const RationalMap* found = loaded.find("lib_member");
        check(found != nullptr, "...and the member is findable by name afterward");
        check(found != nullptr && found->pq_source() == "a*z^2 + b",
              "...with its authored source intact");
    }

    // ---- error handling: ambiguous/invalid parameter selection --------------------
    std::printf("\nerror handling: ambiguous or invalid parameter selection:\n");
    {
        bool threw = false;
        try {
            RationalMap::from_expression("a*z^2 + b", "", {{"b", Cplx{0, 0}}}, "x");
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "two parameters with active_param left EMPTY throws -- no default when "
              "the choice is genuinely ambiguous");
    }
    {
        bool threw = false;
        try {
            RationalMap::from_expression("a*z^2 + b", "c", {{"b", Cplx{0, 0}}}, "x");
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "active_param naming something OTHER than one of the source's own "
              "parameters throws");
    }
    {
        bool threw = false;
        try {
            RationalMap::from_expression("a*z^2 + b", "a", {}, "x");   // missing b's value
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a missing fixed value for a non-active parameter throws");
    }
    {
        bool threw = false;
        try {
            RationalMap::from_expression("z^2 + a", "b", {}, "x");   // 'b' isn't even parsed
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "active_param naming a parameter the source doesn't even reference "
              "throws");
    }
    {
        bool threw = false;
        try {
            RationalMap::from_expression("z^2 + 1", "a", {}, "x");   // no parameters at all
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a non-empty active_param on a source with ZERO parameters throws");
    }
    {
        bool threw = false;
        try {
            RationalMap::from_expression("not valid ) syntax (", "", {}, "x");
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a source that fails to parse at all throws (surfacing the parser's own "
              "error, not silently producing a garbage map)");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
