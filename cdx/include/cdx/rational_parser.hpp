// =============================================================================
// cdx/rational_parser.hpp -- general rational expression -> canonical P(z)/Q(z).
//
// WHY THIS EXISTS. cdx::Expr (expr.hpp) parses and EVALUATES an arbitrary
// formula in z and one hard-coded parameter `a` via a stack machine -- it has
// no notion of polynomial coefficient structure, works with exactly one
// parameter, and accepts transcendental functions (sin, exp, ...). None of
// that is a defect there (it exists to evaluate user formulas fast, not to
// analyze them), but it means Expr cannot answer "what are this map's poles"
// or "what is its degree" -- those need P(z) and Q(z) as actual polynomials,
// not an opaque bytecode program.
//
// This module parses the RATIONAL sublanguage only (+ - * / integer-^ parens
// unary-minus z i numbers and ARBITRARY named parameters -- no transcendental
// functions) directly into that structure: P and Q as vectors of
// per-degree-in-z coefficients, each coefficient itself an expression in the
// parameters (not a single number, since a coefficient like `a+b` or `2*a`
// is common and must stay symbolic until a caller supplies parameter
// values). Every subexpression is carried as a Fraction of two such
// polynomials-in-z as parsing proceeds -- + - * / and integer ^ are all
// FRACTION operations (see rational_parser.cpp), so the result is always a
// single P/Q, however deeply the input nests divisions.
//
// GRAMMAR (mirrors Expr's own precedence exactly, including its own
// unary-binds-tighter-than-power quirk -- see rational_parser.cpp for why
// that is deliberately preserved rather than "fixed"):
//     expr     := term (('+' | '-') term)*
//     term     := factor (('*' | '/') factor)*
//     factor   := unary ('^' int_exponent)?          -- ^ is NOT right-
//                                                        associative here
//                                                        (there is nothing
//                                                        to associate: the
//                                                        exponent is a bare
//                                                        integer literal,
//                                                        never a sub-
//                                                        expression -- see
//                                                        int_exponent below)
//     unary    := ('-' | '+')? atom
//     atom     := number | 'i' | 'j' | 'z' | identifier | '(' expr ')'
//     int_exponent := '('? ('-'|'+')? digit+ ')'?      -- a LITERAL signed
//                                                          integer, optionally
//                                                          parenthesized;
//                                                          never a parameter
//                                                          or general
//                                                          expression, since
//                                                          a polynomial's
//                                                          degree must be
//                                                          known at parse
//                                                          time
//
// Numbers accept the same decimal/scientific forms as strtod (matching
// Expr), and "<number>i"/"<number>j" is shorthand for <number>*i, the ONE
// exception Expr itself makes to "no implicit multiplication" -- matched
// here for consistency, and likewise the only exception.
//
// PARAMETERS. Any identifier that is not exactly "z", "i", or "j" is a
// parameter reference (this INCLUDES "a", "pi", "e" -- unlike Expr, which
// treats "a"/"c"/"lambda" as its one hard-coded parameter and "pi"/"e" as
// built-in constants; this parser has no built-in parameter or constant
// identifiers at all, since it supports arbitrarily many named parameters
// and there is no longer one privileged slot for Expr's "a" to occupy).
// parse_rational() collects every distinct one into CanonicalRational::
// parameters, sorted for a stable, deterministic order.
//
// FUNCTION CALLS ARE REJECTED, always, with a message that distinguishes a
// recognized transcendental name (out of scope: not a rational map) from an
// unrecognized one (not a supported construct at all) -- see
// rational_parser.cpp's KNOWN_TRANSCENDENTAL set.
//
// NOT DONE HERE, ON PURPOSE: symbolic GCD cancellation of common P/Q
// factors. A redundant common factor (e.g. from `(z-1)/(z-1) * f(z)`)
// cancels safely out of degree DIFFERENCES the same way
// RationalMap::own_fraction's own uncancelled common denominator already
// does (see rational.cpp's clear_denominators, whose "not necessarily
// reduced, still correct" reasoning applies unchanged here) -- reducing it
// properly needs polynomial GCD over a ring whose elements are themselves
// unevaluated parameter expressions, a substantially harder problem left
// for whoever actually needs it.
// =============================================================================
#pragma once

#include <complex>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cdx {

using Cplx = std::complex<double>;

// -----------------------------------------------------------------------------
// A small expression tree over NAMED PARAMETERS ONLY (no z). Built purely
// from Const/Param/Add/Sub/Mul/Neg -- no Div, no Pow node -- because every
// division and every power in the source grammar is resolved at the
// Fraction<PolyZ> level (see rational_parser.cpp): by the time a single
// coefficient is isolated (one entry of P or Q), it is always a sum of
// products of parameters and constants, never itself a fraction or a power
// of something other than a repeated product.
// -----------------------------------------------------------------------------
struct ParamExpr {
    enum class Kind { Const, Param, Add, Sub, Mul, Neg };

    Kind kind = Kind::Const;
    Cplx value{0.0, 0.0};                 // Kind::Const
    std::string name;                     // Kind::Param
    std::shared_ptr<ParamExpr> lhs, rhs;  // Add/Sub/Mul: both; Neg: lhs only

    Cplx eval(const std::map<std::string, Cplx>& params) const;

    // TRUE iff this node is structurally the literal constant 0 -- i.e. it
    // was built (directly, or by constant-folding two literal operands
    // during parsing) as Const(0), NOT merely a subexpression that happens
    // to evaluate to 0 for some particular parameter binding (e.g. "a - a"
    // is never folded this way; see the file header's own no-GCD-
    // cancellation note for why that distinction is deliberate).
    bool is_zero_literal() const { return kind == Kind::Const && value == Cplx(0.0, 0.0); }
};
using ParamExprPtr = std::shared_ptr<ParamExpr>;

ParamExprPtr param_const(Cplx v);
ParamExprPtr param_named(std::string name);

// Returns a NEW tree with every Param node named `name` replaced by
// Const(value) -- everything else rebuilt through the SAME constant-
// folding constructors parsing itself uses (param_add/sub/mul/neg), so a
// substitution that leaves no remaining reference to any OTHER parameter
// folds down exactly as far as parsing an already-substituted source
// would have. Used by a later stage (RationalMap::from_expression,
// rational.hpp) to fix every non-active parameter's value before reducing
// a multi-parameter CanonicalRational to the engine's own single-active-
// parameter P/Q form.
ParamExprPtr substitute_param(const ParamExprPtr& e, const std::string& name, Cplx value);

// -----------------------------------------------------------------------------
// A polynomial in z whose coefficients live in the ring of parameter
// expressions above. Ascending order, matching cdx::Polynomial's own
// convention (roots.hpp): coeffs[k] is the coefficient of z^k. Trailing
// entries that are STRUCTURALLY (see is_zero_literal above) the constant 0
// are trimmed as the polynomial is built; anything not resolvable to a
// literal at parse time is left exactly as authored -- evaluating it for a
// specific parameter binding, and trimming THAT numeric result, is a job
// for whoever consumes CanonicalRational (see RationalMap's own
// effective_degree-style handling for the analogous already-evaluated
// case).
// -----------------------------------------------------------------------------
struct PolyZ {
    std::vector<ParamExprPtr> coeffs;

    Cplx eval(Cplx z, const std::map<std::string, Cplx>& params) const;
};

// PolyZ arithmetic, exposed (not just used internally by the parser) so a
// consumer working directly with canonical P/Q -- e.g. the P/Q-backed
// RationalMap representation (rational.hpp) computing (P'Q - PQ')/Q^2 in
// closed form -- can build derived polynomials without duplicating this
// algebra. All four trim trailing structurally-zero coefficients (see
// ParamExpr::is_zero_literal); poly_mul of an empty PolyZ (the literal
// constant 0) with anything is the empty PolyZ.
PolyZ poly_add(const PolyZ& a, const PolyZ& b);
PolyZ poly_sub(const PolyZ& a, const PolyZ& b);
PolyZ poly_mul(const PolyZ& a, const PolyZ& b);
PolyZ poly_neg(const PolyZ& a);

// d/dz of a PolyZ: coefficient k becomes (k+1)*coeffs[k+1]. The empty PolyZ
// (degree -infinity, i.e. identically 0) for a degree-<=0 input.
PolyZ poly_deriv(const PolyZ& a);

// A degree-0 PolyZ holding exactly `c` (empty if c is the literal 0).
PolyZ poly_const(ParamExprPtr c);

// TRUE iff the parameter named `name` appears anywhere in this expression
// tree -- a purely structural (not numeric) check, used by callers that
// need to know whether a coefficient can vary with one specific parameter
// (e.g. RationalMap::critical_points_constant's P/Q-backed implementation).
bool references_param(const ParamExprPtr& e, const std::string& name);

// substitute_param, applied to every coefficient of a whole PolyZ, with
// trailing entries re-trimmed afterward -- substituting a parameter's
// value can turn a previously-nonzero leading coefficient into a literal
// zero (e.g. "b*z^2 + a" with b fixed to 0), which must lower the
// polynomial's own STRUCTURAL degree, not leave a phantom zero leading
// term.
PolyZ substitute_param(const PolyZ& p, const std::string& name, Cplx value);

// -----------------------------------------------------------------------------
// The canonical form every authored rational expression collapses to:
// R(z) = P(z) / Q(z), plus every distinct parameter name referenced
// anywhere in the source (sorted, deduplicated; never includes z/i/j).
// -----------------------------------------------------------------------------
struct CanonicalRational {
    PolyZ P, Q;
    std::vector<std::string> parameters;
    std::string source;   // the original authored text, preserved verbatim

    // R(z) = P(z)/Q(z) at the given z, with every name in `parameters`
    // bound via `params`. A name present in `parameters` but missing from
    // `params` evaluates as if bound to 0 (ParamExpr::eval's own
    // convention) -- callers that need to detect a genuinely missing
    // binding should check against `parameters` themselves before calling.
    Cplx eval(Cplx z, const std::map<std::string, Cplx>& params) const;
};

// Parses `source` -- a rational expression in z with arbitrary named
// parameters -- into canonical P(z)/Q(z) form. Returns false and fills
// `error` (with a 1-based character position where feasible, matching
// Expr's own convention) on any invalid input: an unknown symbol, a
// transcendental function call, malformed syntax, or a non-integer /
// non-literal exponent. `out` is left in an unspecified state on failure.
bool parse_rational(const std::string& source, CanonicalRational& out, std::string& error);

}  // namespace cdx
