// =============================================================================
// cdx/rational_parser.cpp -- recursive-descent parser building Fraction<PolyZ>
// values directly (no separate AST pass), mirroring cdx::Expr's own parser
// structure (expr.cpp) but producing canonical P(z)/Q(z) instead of bytecode.
//
// THE FRACTION ALGEBRA. Every subexpression the parser reduces is a Fraction
// -- a pair of PolyZ (numerator, denominator). Combining two subexpressions
// is ordinary fraction arithmetic:
//     x + y  =  (x.num*y.den + y.num*x.den) / (x.den*y.den)
//     x - y  =  (x.num*y.den - y.num*x.den) / (x.den*y.den)
//     x * y  =  (x.num*y.num) / (x.den*y.den)
//     x / y  =  (x.num*y.den) / (x.den*y.num)
//     -x     =  (-x.num) / x.den
//     x^n    =  (x.num^n) / (x.den^n)          n >= 0
//     x^n    =  (x.den^|n|) / (x.num^|n|)      n <  0
// A leaf (a number, i, z, or a parameter name) starts as num=that value,
// den=1. Because every operation stays within this closed algebra, the
// RESULT of parsing the whole expression is, unconditionally, a single
// Fraction -- there is no separate "collapse to one fraction" pass, the
// parser never produces anything else. Multiplying/dividing PolyZ never
// needs anything beyond ParamExpr::Add/Sub/Mul (see the header), since a
// polynomial's own coefficients only ever get summed and multiplied by this
// algebra, never divided directly -- any division always manufactures a NEW
// denominator PolyZ instead.
// =============================================================================
#include "cdx/rational_parser.hpp"
#include "cdx/special_polys.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>

namespace cdx {

// -----------------------------------------------------------------------------
// ParamExpr
// -----------------------------------------------------------------------------
ParamExprPtr param_const(Cplx v) {
    auto p = std::make_shared<ParamExpr>();
    p->kind = ParamExpr::Kind::Const;
    p->value = v;
    return p;
}

ParamExprPtr param_named(std::string name) {
    auto p = std::make_shared<ParamExpr>();
    p->kind = ParamExpr::Kind::Param;
    p->name = std::move(name);
    return p;
}

namespace {

ParamExprPtr param_binary(ParamExpr::Kind kind, ParamExprPtr lhs, ParamExprPtr rhs) {
    // Constant-fold immediately when both operands are already literals --
    // cheap, keeps the tree small, and is what lets PolyZ trimming (below)
    // recognize a genuinely-zero leading coefficient in the common case
    // (e.g. "z^3 + 0" or two literal coefficients that happen to cancel)
    // without attempting anything as general as symbolic simplification.
    if (lhs->kind == ParamExpr::Kind::Const && rhs->kind == ParamExpr::Kind::Const) {
        switch (kind) {
            case ParamExpr::Kind::Add: return param_const(lhs->value + rhs->value);
            case ParamExpr::Kind::Sub: return param_const(lhs->value - rhs->value);
            case ParamExpr::Kind::Mul: return param_const(lhs->value * rhs->value);
            default: break;
        }
    }
    auto p = std::make_shared<ParamExpr>();
    p->kind = kind;
    p->lhs = std::move(lhs);
    p->rhs = std::move(rhs);
    return p;
}

ParamExprPtr param_add(ParamExprPtr a, ParamExprPtr b) {
    if (a->is_zero_literal()) return b;
    if (b->is_zero_literal()) return a;
    return param_binary(ParamExpr::Kind::Add, std::move(a), std::move(b));
}
ParamExprPtr param_sub(ParamExprPtr a, ParamExprPtr b) {
    if (b->is_zero_literal()) return a;
    return param_binary(ParamExpr::Kind::Sub, std::move(a), std::move(b));
}
ParamExprPtr param_mul(ParamExprPtr a, ParamExprPtr b) {
    if (a->is_zero_literal() || b->is_zero_literal()) return param_const(Cplx(0.0, 0.0));
    if (a->kind == ParamExpr::Kind::Const && a->value == Cplx(1.0, 0.0)) return b;
    if (b->kind == ParamExpr::Kind::Const && b->value == Cplx(1.0, 0.0)) return a;
    return param_binary(ParamExpr::Kind::Mul, std::move(a), std::move(b));
}
ParamExprPtr param_neg(ParamExprPtr a) {
    if (a->kind == ParamExpr::Kind::Const) return param_const(-a->value);
    auto p = std::make_shared<ParamExpr>();
    p->kind = ParamExpr::Kind::Neg;
    p->lhs = std::move(a);
    return p;
}

}  // namespace

Cplx ParamExpr::eval(const std::map<std::string, Cplx>& params) const {
    switch (kind) {
        case Kind::Const: return value;
        case Kind::Param: {
            auto it = params.find(name);
            return it == params.end() ? Cplx(0.0, 0.0) : it->second;
        }
        case Kind::Add: return lhs->eval(params) + rhs->eval(params);
        case Kind::Sub: return lhs->eval(params) - rhs->eval(params);
        case Kind::Mul: return lhs->eval(params) * rhs->eval(params);
        case Kind::Neg: return -lhs->eval(params);
    }
    return Cplx(0.0, 0.0);
}

// -----------------------------------------------------------------------------
// PolyZ
// -----------------------------------------------------------------------------
Cplx PolyZ::eval(Cplx z, const std::map<std::string, Cplx>& params) const {
    // Horner, evaluating each coefficient against `params` as it's needed.
    Cplx acc(0.0, 0.0);
    for (std::size_t i = coeffs.size(); i-- > 0;) {
        acc = acc * z + coeffs[i]->eval(params);
    }
    return acc;
}

namespace {
// Drops trailing (highest-degree) entries that are structurally the
// literal 0 -- see ParamExpr::is_zero_literal's own comment on why this is
// a purely structural check, not a numeric one.
void trim(PolyZ& p) {
    while (!p.coeffs.empty() && p.coeffs.back()->is_zero_literal()) p.coeffs.pop_back();
}
PolyZ poly_zero() { return {}; }
}  // namespace

// PolyZ arithmetic -- exposed via rational_parser.hpp (see its own comment
// on why: a P/Q-backed RationalMap needs this same algebra, not just the
// parser).
PolyZ poly_const(ParamExprPtr c) {
    PolyZ p;
    if (!c->is_zero_literal()) p.coeffs.push_back(std::move(c));
    return p;
}

PolyZ poly_add(const PolyZ& a, const PolyZ& b) {
    PolyZ out;
    const std::size_t n = std::max(a.coeffs.size(), b.coeffs.size());
    out.coeffs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        ParamExprPtr av = i < a.coeffs.size() ? a.coeffs[i] : param_const(Cplx(0.0, 0.0));
        ParamExprPtr bv = i < b.coeffs.size() ? b.coeffs[i] : param_const(Cplx(0.0, 0.0));
        out.coeffs.push_back(param_add(av, bv));
    }
    trim(out);
    return out;
}

PolyZ poly_sub(const PolyZ& a, const PolyZ& b) {
    PolyZ out;
    const std::size_t n = std::max(a.coeffs.size(), b.coeffs.size());
    out.coeffs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        ParamExprPtr av = i < a.coeffs.size() ? a.coeffs[i] : param_const(Cplx(0.0, 0.0));
        ParamExprPtr bv = i < b.coeffs.size() ? b.coeffs[i] : param_const(Cplx(0.0, 0.0));
        out.coeffs.push_back(param_sub(av, bv));
    }
    trim(out);
    return out;
}

PolyZ poly_mul(const PolyZ& a, const PolyZ& b) {
    if (a.coeffs.empty() || b.coeffs.empty()) return poly_zero();
    PolyZ out;
    out.coeffs.assign(a.coeffs.size() + b.coeffs.size() - 1, nullptr);
    for (std::size_t i = 0; i < a.coeffs.size(); ++i) {
        for (std::size_t j = 0; j < b.coeffs.size(); ++j) {
            ParamExprPtr term = param_mul(a.coeffs[i], b.coeffs[j]);
            out.coeffs[i + j] = out.coeffs[i + j] ? param_add(out.coeffs[i + j], term) : term;
        }
    }
    for (auto& c : out.coeffs) if (!c) c = param_const(Cplx(0.0, 0.0));
    trim(out);
    return out;
}

PolyZ poly_neg(const PolyZ& a) {
    PolyZ out;
    out.coeffs.reserve(a.coeffs.size());
    for (const auto& c : a.coeffs) out.coeffs.push_back(param_neg(c));
    return out;
}

PolyZ poly_deriv(const PolyZ& a) {
    PolyZ out;
    if (a.coeffs.size() < 2) return out;   // degree <= 0: derivative is 0
    out.coeffs.reserve(a.coeffs.size() - 1);
    for (std::size_t k = 1; k < a.coeffs.size(); ++k) {
        // k * coeffs[k], via repeated param_add rather than a dedicated
        // "scale by integer" node -- ParamExpr has no Mul-by-plain-int
        // shortcut, and k is always small (a realistic map's degree), so
        // this is not worth a new node kind for.
        ParamExprPtr scaled = a.coeffs[k];
        for (std::size_t i = 1; i < k; ++i) scaled = param_add(scaled, a.coeffs[k]);
        out.coeffs.push_back(std::move(scaled));
    }
    trim(out);
    return out;
}

bool references_param(const ParamExprPtr& e, const std::string& name) {
    if (!e) return false;
    switch (e->kind) {
        case ParamExpr::Kind::Const: return false;
        case ParamExpr::Kind::Param: return e->name == name;
        case ParamExpr::Kind::Neg: return references_param(e->lhs, name);
        case ParamExpr::Kind::Add:
        case ParamExpr::Kind::Sub:
        case ParamExpr::Kind::Mul:
            return references_param(e->lhs, name) || references_param(e->rhs, name);
    }
    return false;
}

ParamExprPtr substitute_param(const ParamExprPtr& e, const std::string& name, Cplx value) {
    switch (e->kind) {
        case ParamExpr::Kind::Const:
            return e;
        case ParamExpr::Kind::Param:
            return e->name == name ? param_const(value) : e;
        case ParamExpr::Kind::Neg:
            return param_neg(substitute_param(e->lhs, name, value));
        case ParamExpr::Kind::Add:
            return param_add(substitute_param(e->lhs, name, value),
                             substitute_param(e->rhs, name, value));
        case ParamExpr::Kind::Sub:
            return param_sub(substitute_param(e->lhs, name, value),
                             substitute_param(e->rhs, name, value));
        case ParamExpr::Kind::Mul:
            return param_mul(substitute_param(e->lhs, name, value),
                             substitute_param(e->rhs, name, value));
    }
    return e;
}

PolyZ substitute_param(const PolyZ& p, const std::string& name, Cplx value) {
    PolyZ out;
    out.coeffs.reserve(p.coeffs.size());
    for (const auto& c : p.coeffs) out.coeffs.push_back(substitute_param(c, name, value));
    while (!out.coeffs.empty() && out.coeffs.back()->is_zero_literal()) out.coeffs.pop_back();
    return out;
}

namespace {

// -----------------------------------------------------------------------------
// Fraction<PolyZ> -- what every subexpression reduces to as parsing proceeds.
// -----------------------------------------------------------------------------
struct Fraction {
    PolyZ num, den;   // den is never structurally empty (see leaf constructors)
};

Fraction frac_const(Cplx v) { return {poly_const(param_const(v)), poly_const(param_const(Cplx(1.0, 0.0)))}; }
Fraction frac_z() {
    PolyZ p;
    p.coeffs.push_back(param_const(Cplx(0.0, 0.0)));
    p.coeffs.push_back(param_const(Cplx(1.0, 0.0)));
    return {p, poly_const(param_const(Cplx(1.0, 0.0)))};
}
Fraction frac_param(const std::string& name) {
    return {poly_const(param_named(name)), poly_const(param_const(Cplx(1.0, 0.0)))};
}

Fraction frac_add(const Fraction& x, const Fraction& y) {
    return {poly_add(poly_mul(x.num, y.den), poly_mul(y.num, x.den)), poly_mul(x.den, y.den)};
}
Fraction frac_sub(const Fraction& x, const Fraction& y) {
    return {poly_sub(poly_mul(x.num, y.den), poly_mul(y.num, x.den)), poly_mul(x.den, y.den)};
}
Fraction frac_mul(const Fraction& x, const Fraction& y) {
    return {poly_mul(x.num, y.num), poly_mul(x.den, y.den)};
}
Fraction frac_div(const Fraction& x, const Fraction& y) {
    return {poly_mul(x.num, y.den), poly_mul(x.den, y.num)};
}
Fraction frac_neg(const Fraction& x) { return {poly_neg(x.num), x.den}; }

Fraction frac_pow(const Fraction& x, int n) {
    const bool inv = n < 0;
    int an = inv ? -n : n;
    Fraction base = inv ? Fraction{x.den, x.num} : x;
    Fraction acc = frac_const(Cplx(1.0, 0.0));
    // Binary exponentiation -- an is small in every realistic authored map
    // (a rational map of degree in the hundreds would already be an
    // impractical render target), so this need not be more clever than
    // straightforward repeated squaring.
    Fraction b = base;
    while (an) {
        if (an & 1) acc = frac_mul(acc, b);
        b = frac_mul(b, b);
        an >>= 1;
    }
    return acc;
}

// d/dz of a PolyZ with the integer factor applied as ONE multiplication
// (param_const(k) * c) rather than poly_deriv's k-fold repeated addition --
// identical for the engine's own small-degree uses, but this one also
// serves Newton/Halley maps built on degree-40 special polynomials, where
// repeated addition would both grow symbolic coefficient trees linearly in
// k and accumulate k roundings for numeric ones.
PolyZ poly_deriv_scaled(const PolyZ& a) {
    PolyZ out;
    if (a.coeffs.size() < 2) return out;
    out.coeffs.reserve(a.coeffs.size() - 1);
    for (std::size_t k = 1; k < a.coeffs.size(); ++k)
        out.coeffs.push_back(param_mul(param_const(Cplx(static_cast<double>(k), 0.0)), a.coeffs[k]));
    trim(out);
    return out;
}

// d/dz of a whole Fraction -- the quotient rule, or just N'/D when the
// denominator is a constant (keeps d(p) of a polynomial a polynomial,
// instead of manufacturing a D^2 that every later operation would carry).
Fraction frac_deriv(const Fraction& f) {
    if (f.den.coeffs.size() <= 1) return {poly_deriv_scaled(f.num), f.den};
    return {poly_sub(poly_mul(poly_deriv_scaled(f.num), f.den),
                     poly_mul(f.num, poly_deriv_scaled(f.den))),
            poly_mul(f.den, f.den)};
}

// -----------------------------------------------------------------------------
// Special-polynomial built-ins (see rational_parser.hpp's grammar comment and
// special_polys.hpp for the coefficient computations).
// -----------------------------------------------------------------------------
struct SpecialFn {
    special::Family family;
    int symbols;              // how many of alpha/beta/lam the call carries
    const char* signature;    // for the arity error message
};

// Resolves a call name + argument count to a family. `known` is set iff the
// NAME is a special-polynomial function at all (whatever the count);
// returns false with `arity_msg` filled when the count is wrong for it.
bool resolve_special(const std::string& id, int nargs, bool& known, SpecialFn& fn,
                     std::string& arity_msg) {
    using F = special::Family;
    struct Entry { const char* name; F family; int symbols; const char* sig; };
    static const Entry kTable[] = {
        {"hermite", F::Hermite, 0, "2 arguments (n, x)"},
        {"H", F::Hermite, 0, "2 arguments (n, x)"},
        {"hermite_e", F::HermiteE, 0, "2 arguments (n, x)"},
        {"He", F::HermiteE, 0, "2 arguments (n, x)"},
        {"legendre", F::Legendre, 0, "2 arguments (n, x)"},
        {"chebyshev_t", F::ChebyshevT, 0, "2 arguments (n, x)"},
        {"T", F::ChebyshevT, 0, "2 arguments (n, x)"},
        {"chebyshev_u", F::ChebyshevU, 0, "2 arguments (n, x)"},
        {"U", F::ChebyshevU, 0, "2 arguments (n, x)"},
        {"gegenbauer", F::Gegenbauer, 1, "3 arguments (n, lam, x)"},
        {"jacobi", F::Jacobi, 2, "4 arguments (n, alpha, beta, x)"},
        {"bessel_poly", F::Bessel, 0, "2 arguments (n, x)"},
        {"bernoulli", F::Bernoulli, 0, "2 arguments (n, x)"},
        {"euler_poly", F::Euler, 0, "2 arguments (n, x)"},
        {"cyclotomic", F::Cyclotomic, 0, "2 arguments (n, x)"},
    };
    known = true;
    // The two overloaded names: argument count picks the family.
    if (id == "laguerre" || id == "L") {
        if (nargs == 2) { fn = {F::Laguerre, 0, ""}; return true; }
        if (nargs == 3) { fn = {F::Laguerre, 1, ""}; return true; }
        arity_msg = id + " expects 2 or 3 arguments (n, x) or (n, alpha, x)";
        return false;
    }
    if (id == "P") {
        if (nargs == 2) { fn = {F::Legendre, 0, ""}; return true; }
        if (nargs == 4) { fn = {F::Jacobi, 2, ""}; return true; }
        arity_msg = "P expects 2 arguments (n, x) or 4 (n, alpha, beta, x)";
        return false;
    }
    for (const auto& e : kTable) {
        if (id != e.name) continue;
        if (nargs != e.symbols + 2) {
            arity_msg = id + " expects " + e.sig;
            return false;
        }
        fn = {e.family, e.symbols, e.sig};
        return true;
    }
    known = false;
    return false;
}

bool is_special_name(const std::string& id) {
    bool known = false;
    SpecialFn fn;
    std::string msg;
    resolve_special(id, 2, known, fn, msg);
    return known;
}

// A SymPoly coefficient (polynomial in the family's symbols) as a ParamExpr,
// given each symbol's own ParamExpr (null when the family has none). Fully
// numeric symbols are evaluated directly in extended precision; otherwise a
// nested Horner form keeps the tree O(degree) per coefficient.
ParamExprPtr sym_to_param(const special::SymPoly& p, const ParamExprPtr& s0, const ParamExprPtr& s1) {
    using CL = std::complex<long double>;
    auto is_num = [](const ParamExprPtr& s) { return !s || s->kind == ParamExpr::Kind::Const; };
    if (is_num(s0) && is_num(s1)) {
        const CL a = s0 ? CL(s0->value.real(), s0->value.imag()) : CL(0.0L);
        const CL b = s1 ? CL(s1->value.real(), s1->value.imag()) : CL(0.0L);
        CL outer(0.0L);
        for (std::size_t i = p.c.size(); i-- > 0;) {
            CL inner(0.0L);
            for (std::size_t j = p.c[i].size(); j-- > 0;) inner = inner * b + p.c[i][j];
            outer = outer * a + inner;
        }
        return param_const(Cplx(static_cast<double>(outer.real()), static_cast<double>(outer.imag())));
    }
    auto horner = [](const std::vector<ParamExprPtr>& cs, const ParamExprPtr& s) {
        ParamExprPtr acc = param_const(Cplx(0.0, 0.0));
        for (std::size_t k = cs.size(); k-- > 0;) acc = param_add(s ? param_mul(acc, s) : acc, cs[k]);
        return acc;
    };
    std::vector<ParamExprPtr> inners;
    for (const auto& row : p.c) {
        std::vector<ParamExprPtr> cs;
        for (long double v : row) cs.push_back(param_const(Cplx(static_cast<double>(v), 0.0)));
        if (!s1) cs.resize(1);
        inners.push_back(horner(cs, s1));
    }
    if (!s0) inners.resize(1);
    return horner(inners, s0);
}

// -----------------------------------------------------------------------------
// Transcendental names rejected with a specific message -- the same set
// Expr itself recognizes as functions (expr.cpp), so "not supported here"
// and "recognized as a function name at all" stay in sync.
// -----------------------------------------------------------------------------
bool is_known_transcendental(const std::string& id) {
    static const std::set<std::string> kNames = {
        "exp", "log", "ln", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "sqrt", "conj", "abs", "re", "im", "arg",
    };
    return kNames.count(id) != 0;
}

// -----------------------------------------------------------------------------
// Parser state -- structurally mirrors expr.cpp's own Parser (same
// skip_ws/peek/accept helpers, same fail() convention), but each grammar
// rule returns a Fraction instead of emitting bytecode, and parameter names
// are collected into `params` as they're encountered.
// -----------------------------------------------------------------------------
struct Parser {
    const std::string& src;
    std::size_t pos = 0;
    std::string err;
    std::set<std::string> params;
    std::vector<std::string> warnings;

    explicit Parser(const std::string& s) : src(s) {}

    void skip_ws() {
        while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos]))) ++pos;
    }
    bool eof() { skip_ws(); return pos >= src.size(); }
    char peek() { skip_ws(); return pos < src.size() ? src[pos] : '\0'; }
    bool accept(char c) {
        skip_ws();
        if (pos < src.size() && src[pos] == c) { ++pos; return true; }
        return false;
    }

    bool fail(const std::string& msg) {
        if (err.empty()) err = msg + " (at character " + std::to_string(pos + 1) + ")";
        return false;
    }

    bool parse_expr(Fraction& out) {
        Fraction acc;
        if (!parse_term(acc)) return false;
        for (;;) {
            skip_ws();
            if (accept('+')) {
                Fraction rhs;
                if (!parse_term(rhs)) return false;
                acc = frac_add(acc, rhs);
            } else if (accept('-')) {
                Fraction rhs;
                if (!parse_term(rhs)) return false;
                acc = frac_sub(acc, rhs);
            } else {
                out = acc;
                return true;
            }
        }
    }

    bool parse_term(Fraction& out) {
        Fraction acc;
        if (!parse_factor(acc)) return false;
        for (;;) {
            skip_ws();
            if (accept('*')) {
                Fraction rhs;
                if (!parse_factor(rhs)) return false;
                acc = frac_mul(acc, rhs);
            } else if (accept('/')) {
                Fraction rhs;
                if (!parse_factor(rhs)) return false;
                acc = frac_div(acc, rhs);
            } else {
                out = acc;
                return true;
            }
        }
    }

    bool parse_factor(Fraction& out) {
        Fraction base;
        if (!parse_unary(base)) return false;
        skip_ws();
        if (accept('^')) {
            int n;
            if (!parse_int_exponent(n)) return false;
            out = frac_pow(base, n);
            return true;
        }
        out = base;
        return true;
    }

    // A literal signed integer, optionally parenthesized -- see the
    // header's own grammar comment for why this is deliberately NOT a
    // general sub-expression.
    bool parse_int_exponent(int& out) {
        skip_ws();
        const bool parenthesized = accept('(');
        skip_ws();
        bool neg = false;
        if (accept('-')) neg = true;
        else if (accept('+')) neg = false;
        skip_ws();
        if (pos >= src.size() || !std::isdigit(static_cast<unsigned char>(src[pos]))) {
            return fail("exponent must be a literal integer");
        }
        const std::size_t start = pos;
        while (pos < src.size() && std::isdigit(static_cast<unsigned char>(src[pos]))) ++pos;
        long v = std::strtol(src.substr(start, pos - start).c_str(), nullptr, 10);
        if (parenthesized) {
            skip_ws();
            if (!accept(')')) return fail("expected ')' after exponent");
        }
        out = static_cast<int>(neg ? -v : v);
        return true;
    }

    bool parse_unary(Fraction& out) {
        skip_ws();
        if (accept('-')) {
            Fraction v;
            if (!parse_unary(v)) return false;
            out = frac_neg(v);
            return true;
        }
        if (accept('+')) return parse_unary(out);
        return parse_atom(out);
    }

    bool parse_atom(Fraction& out) {
        skip_ws();
        if (pos >= src.size()) return fail("unexpected end of expression");
        const char c = src[pos];

        if (c == '(') {
            ++pos;
            if (!parse_expr(out)) return false;
            if (!accept(')')) return fail("expected ')'");
            return true;
        }

        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const char* start = src.c_str() + pos;
            char* end = nullptr;
            const double v = std::strtod(start, &end);
            if (end == start) return fail("malformed number");
            pos += static_cast<std::size_t>(end - start);
            // "<number>i"/"<number>j" shorthand for <number>*i -- the one
            // exception to "no implicit multiplication", matching Expr.
            if (pos < src.size() && (src[pos] == 'i' || src[pos] == 'j')) {
                ++pos;
                out = frac_const(Cplx(0.0, v));
            } else {
                out = frac_const(Cplx(v, 0.0));
            }
            return true;
        }

        if (std::isalpha(static_cast<unsigned char>(c))) {
            const std::size_t start = pos;
            // '_' is scanned as part of an identifier only so the special-
            // polynomial names (hermite_e, chebyshev_t, ...) can be
            // recognized; a parameter name containing one is still
            // rejected below, exactly as before.
            while (pos < src.size() &&
                   (std::isalnum(static_cast<unsigned char>(src[pos])) || src[pos] == '_')) ++pos;
            const std::string id = src.substr(start, pos - start);

            if (id == "z") { out = frac_z(); return true; }
            if (id == "i" || id == "j") { out = frac_const(Cplx(0.0, 1.0)); return true; }

            // Anything else immediately followed by '(' is a function call:
            // d(...) or a special polynomial; otherwise rejected, with a
            // message that distinguishes "not a rational map" from "not a
            // supported construct at all".
            skip_ws();
            if (pos < src.size() && src[pos] == '(') {
                if (id == "d") { ++pos; return parse_derivative(out); }
                if (is_special_name(id)) { ++pos; return parse_special(id, start, out); }
                if (is_known_transcendental(id)) {
                    return fail("transcendental function '" + id + "' is not supported -- "
                               "rational maps only (P(z)/Q(z), polynomial numerator and "
                               "denominator)");
                }
                return fail("unknown function '" + id + "' -- the only functions in "
                           "rational expressions are d(...) and the special polynomials "
                           "(hermite, legendre, laguerre, chebyshev_t, ...)");
            }

            if (id.find('_') != std::string::npos) {
                pos = start + id.find('_');
                return fail("unexpected character '_'");
            }

            // Any other identifier is a parameter reference -- including
            // names Expr treats specially ("a", "pi", "e"; see this file's
            // own header comment on why that divergence is deliberate).
            params.insert(id);
            out = frac_param(id);
            return true;
        }

        return fail(std::string("unexpected character '") + c + "'");
    }

    // d(expr): the '(' is already consumed.
    bool parse_derivative(Fraction& out) {
        Fraction inner;
        if (!parse_expr(inner)) return false;
        skip_ws();
        if (peek() == ',') return fail("d expects 1 argument (an expression in z)");
        if (!accept(')')) return fail("expected ')' after d(...)");
        out = frac_deriv(inner);
        return true;
    }

    // A family parameter (alpha, beta, lam) as a ParamExpr: it may involve
    // the map's parameters, but not z, and cannot divide by a parameter
    // (ParamExpr has no division).
    bool symbol_value(const Fraction& f, const std::string& fname, const char* what,
                      ParamExprPtr& out) {
        if (f.num.coeffs.size() > 1 || f.den.coeffs.size() > 1)
            return fail(fname + ": " + what + " must not depend on z");
        if (f.den.coeffs.empty()) return fail(fname + ": " + what + " has a zero denominator");
        const ParamExprPtr& d = f.den.coeffs[0];
        if (d->kind != ParamExpr::Kind::Const)
            return fail(fname + ": " + what + " must be a number or a polynomial in the "
                        "parameters (no division by a parameter)");
        const ParamExprPtr num = f.num.coeffs.empty() ? param_const(Cplx(0.0, 0.0)) : f.num.coeffs[0];
        out = param_mul(num, param_const(Cplx(1.0, 0.0) / d->value));
        return true;
    }

    // name(n, [alpha, [beta,]] x): the '(' is already consumed; `start` is
    // the name's own position (for the warning text only).
    bool parse_special(const std::string& id, std::size_t /*start*/, Fraction& out) {
        // Degree: a bare non-negative integer literal, then ','.
        skip_ws();
        const std::size_t dstart = pos;
        while (pos < src.size() && std::isdigit(static_cast<unsigned char>(src[pos]))) ++pos;
        if (pos == dstart) return fail(id + ": degree must be an integer literal");
        const std::string digits = src.substr(dstart, pos - dstart);
        skip_ws();
        if (peek() != ',' && peek() != ')') {
            pos = dstart;
            return fail(id + ": degree must be an integer literal");
        }
        const long n = digits.size() > 6 ? 1000000L : std::strtol(digits.c_str(), nullptr, 10);

        // The remaining arguments: ordinary expressions.
        std::vector<Fraction> args;
        while (accept(',')) {
            Fraction a;
            if (!parse_expr(a)) return false;
            args.push_back(std::move(a));
        }
        skip_ws();
        if (!accept(')')) return fail("expected ',' or ')' in " + id + "(...)");

        bool known = false;
        SpecialFn fn;
        std::string arity_msg;
        if (!resolve_special(id, static_cast<int>(args.size()) + 1, known, fn, arity_msg))
            return fail(arity_msg);

        if (n > special::kMaxDegree)
            return fail(id + ": degree must be at most " + std::to_string(special::kMaxDegree));
        if (fn.family == special::Family::Cyclotomic && n < 1)
            return fail(id + ": index must be at least 1");
        if (n > special::kWarnDegree && fn.family != special::Family::Cyclotomic) {
            warnings.push_back(id + "(" + std::to_string(n) + ", ...): degree above " +
                               std::to_string(special::kWarnDegree) +
                               " -- its monomial coefficients get large, and root-finding "
                               "(critical points, poles) may lose accuracy");
        }

        // Symbols: [alpha|lam] then [beta]. Plain Laguerre is alpha = 0.
        ParamExprPtr s0, s1;
        if (fn.family == special::Family::Laguerre && fn.symbols == 0) s0 = param_const(Cplx(0.0, 0.0));
        static const char* kNames[3][2] = {{"alpha", ""}, {"lam", ""}, {"alpha", "beta"}};
        const int row = fn.family == special::Family::Gegenbauer ? 1 :
                        fn.family == special::Family::Jacobi ? 2 : 0;
        if (fn.symbols >= 1 && !symbol_value(args[0], id, kNames[row][0], s0)) return false;
        if (fn.symbols >= 2 && !symbol_value(args[1], id, kNames[row][1], s1)) return false;
        const Fraction& x = args.back();

        const special::TPoly tp = special::special_poly(fn.family, static_cast<int>(n));
        std::vector<ParamExprPtr> coef;
        coef.reserve(tp.size());
        for (const auto& c : tp) coef.push_back(sym_to_param(c, s0, s1));
        while (!coef.empty() && coef.back()->is_zero_literal()) coef.pop_back();
        if (coef.empty()) { out = frac_const(Cplx(0.0, 0.0)); return true; }

        // p(x) by Horner, in the Fraction algebra: a rational x = N/D gives
        // the homogeneous sum c_k N^k D^(n-k) over D^n.
        const PolyZ one = poly_const(param_const(Cplx(1.0, 0.0)));
        Fraction acc{poly_const(coef.back()), one};
        for (std::size_t k = coef.size() - 1; k-- > 0;) {
            acc = frac_mul(acc, x);
            if (!coef[k]->is_zero_literal()) acc = frac_add(acc, Fraction{poly_const(coef[k]), one});
        }
        out = std::move(acc);
        return true;
    }
};

}  // namespace

Cplx CanonicalRational::eval(Cplx z, const std::map<std::string, Cplx>& params) const {
    return P.eval(z, params) / Q.eval(z, params);
}

bool parse_rational(const std::string& source, CanonicalRational& out, std::string& error) {
    Parser p(source);
    Fraction result;
    if (!p.parse_expr(result)) {
        error = p.err;
        return false;
    }
    if (!p.eof()) {
        error = "unexpected trailing input at character " + std::to_string(p.pos + 1);
        return false;
    }
    if (result.den.coeffs.empty()) {
        // Structurally impossible from this grammar (every leaf's own
        // denominator is the literal constant 1, and poly_mul of two
        // non-empty PolyZ is only empty if a factor was literally trimmed
        // to nothing -- i.e. a denominator that is EXACTLY the literal 0,
        // such as "z/0"), but report it explicitly rather than dividing by
        // an empty coefficient vector downstream.
        error = "expression has a literal zero denominator";
        return false;
    }

    out.P = std::move(result.num);
    out.Q = std::move(result.den);
    out.parameters.assign(p.params.begin(), p.params.end());
    out.warnings = std::move(p.warnings);
    std::sort(out.parameters.begin(), out.parameters.end());
    out.source = source;
    return true;
}

}  // namespace cdx
