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
            while (pos < src.size() && std::isalnum(static_cast<unsigned char>(src[pos]))) ++pos;
            const std::string id = src.substr(start, pos - start);

            if (id == "z") { out = frac_z(); return true; }
            if (id == "i" || id == "j") { out = frac_const(Cplx(0.0, 1.0)); return true; }

            // Anything else immediately followed by '(' is a function-call
            // attempt -- always rejected, with a message that distinguishes
            // "not a rational map" from "not a supported construct at all".
            skip_ws();
            if (pos < src.size() && src[pos] == '(') {
                if (is_known_transcendental(id)) {
                    return fail("transcendental function '" + id + "' is not supported -- "
                               "rational maps only (P(z)/Q(z), polynomial numerator and "
                               "denominator)");
                }
                return fail("unknown function '" + id + "' -- function calls are not "
                           "supported in rational expressions");
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
    std::sort(out.parameters.begin(), out.parameters.end());
    out.source = source;
    return true;
}

}  // namespace cdx
