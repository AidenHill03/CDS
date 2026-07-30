// =============================================================================
// test_expr.cpp -- property-based checks for the compact expression evaluator.
//
// Mirrors test_renderer.cpp's style: known closed-form values and rejection
// of malformed input, not golden output.
// =============================================================================
#include "cdx/expr.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static bool close(Cplx a, Cplx b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// Compiles `src`, asserting success, and returns the Expr (empty/invalid on
// failure, with `failures` already bumped and the parser error printed).
static Expr must_compile(const std::string& src) {
    Expr e;
    std::string err;
    const bool ok = e.compile(src, err);
    check(ok, ("compiles: " + src).c_str());
    if (!ok) std::printf("      error: %s\n", err.c_str());
    return e;
}

int main() {
    std::printf("=== cdx expr tests ===\n");

    // ---- arithmetic and precedence -------------------------------------------
    std::printf("\narithmetic and precedence:\n");
    {
        Expr e = must_compile("1+2*3");
        check(close(e(0, 0), Cplx(7, 0)), "1+2*3 == 7 (mul binds tighter than add)");
    }
    {
        Expr e = must_compile("(1+2)*3");
        check(close(e(0, 0), Cplx(9, 0)), "(1+2)*3 == 9 (parens override precedence)");
    }
    {
        Expr e = must_compile("2^3^2");
        check(close(e(0, 0), Cplx(512, 0)), "2^3^2 == 512 (right-associative ^)");
    }
    {
        Expr e = must_compile("-3+5");
        check(close(e(0, 0), Cplx(2, 0)), "-3+5 == 2 (unary minus)");
    }
    {
        Expr e = must_compile("--3");
        check(close(e(0, 0), Cplx(3, 0)), "--3 == 3 (nested unary)");
    }
    {
        Expr e = must_compile(" 1 +  2 ");
        check(close(e(0, 0), Cplx(3, 0)), "whitespace is ignored");
    }

    // ---- variables -------------------------------------------------------------
    std::printf("\nvariables z and a:\n");
    {
        Expr e = must_compile("z^2 + a");
        const Cplx zs[] = {{0, 0}, {1, 0}, {0, 1}, {2, -1}};
        const Cplx as[] = {{0, 0}, {-1, 0}, {0.3, 0.2}, {-2, 3}};
        bool all_ok = true;
        for (Cplx z : zs)
            for (Cplx a : as)
                if (!close(e(z, a), z * z + a)) all_ok = false;
        check(all_ok, "z^2+a matches z*z+a at sampled points");
    }
    {
        // A compiled Expr must be safe to evaluate repeatedly from independent
        // scratch buffers -- this is what the threaded renderer relies on
        // once Expr is wired in.
        Expr e = must_compile("z*a - 1");
        std::vector<Cplx> s1(e.stack_depth()), s2(e.stack_depth());
        const Cplx r1 = e.eval({2, 0}, {3, 0}, s1.data());
        const Cplx r2 = e.eval({0, 1}, {0, 1}, s2.data());
        check(close(r1, Cplx(5, 0)), "z*a-1 at (2,3) == 5");
        check(close(r2, Cplx(-2, 0)), "z*a-1 at (i,i) == i*i-1 == -2");
    }

    // ---- imaginary unit and numeric literals -----------------------------------
    std::printf("\nimaginary unit and numeric literals:\n");
    {
        Expr e = must_compile("2i");
        check(close(e(0, 0), Cplx(0, 2)), "2i == 0+2i (implicit-multiply shorthand)");
    }
    {
        Expr e = must_compile("3+4i");
        check(std::abs(std::abs(e(0, 0)) - 5.0) < 1e-9, "|3+4i| == 5");
    }
    {
        Expr e = must_compile("i^2");
        check(close(e(0, 0), Cplx(-1, 0)), "i^2 == -1");
    }
    {
        Expr e = must_compile("pi");
        check(std::abs(e(0, 0).real() - 3.14159265358979323846) < 1e-12, "pi constant");
    }

    // ---- built-in functions -----------------------------------------------------
    std::printf("\nbuilt-in functions:\n");
    check(close(must_compile("exp(0)")(0, 0), Cplx(1, 0)), "exp(0) == 1");
    check(close(must_compile("log(1)")(0, 0), Cplx(0, 0)), "log(1) == 0");
    check(close(must_compile("sin(0)")(0, 0), Cplx(0, 0)), "sin(0) == 0");
    check(close(must_compile("cos(0)")(0, 0), Cplx(1, 0)), "cos(0) == 1");
    check(close(must_compile("sqrt(4)")(0, 0), Cplx(2, 0)), "sqrt(4) == 2");
    check(close(must_compile("conj(3+4i)")(0, 0), Cplx(3, -4)), "conj(3+4i) == 3-4i");
    check(close(must_compile("abs(3+4i)")(0, 0), Cplx(5, 0)), "abs(3+4i) == 5");
    check(close(must_compile("re(3+4i)")(0, 0), Cplx(3, 0)), "re(3+4i) == 3");
    check(close(must_compile("im(3+4i)")(0, 0), Cplx(4, 0)), "im(3+4i) == 4");

    // ---- malformed input is rejected --------------------------------------------
    std::printf("\nmalformed input is rejected:\n");
    auto expect_fail = [](const std::string& src, const char* what) {
        Expr e;
        std::string err;
        const bool ok = e.compile(src, err);
        check(!ok && !err.empty(), what);
    };
    expect_fail("1+", "trailing operator fails");
    expect_fail("(1+2", "unclosed paren fails");
    expect_fail("1 2", "juxtaposed literals fail (no implicit multiplication)");
    expect_fail("bogus(1)", "unknown function name fails");
    expect_fail("", "empty expression fails");

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
