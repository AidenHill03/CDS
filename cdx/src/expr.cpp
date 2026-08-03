// =============================================================================
// cdx/expr.cpp -- recursive-descent parser + stack-machine evaluator.
// =============================================================================
#include "cdx/expr.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace cdx {

namespace {

// -----------------------------------------------------------------------------
// Parser state. Produces code in a single pass, tracking the stack depth so
// callers can size scratch buffers exactly.
// -----------------------------------------------------------------------------
struct Parser {
    const std::string&  src;
    std::size_t         pos = 0;
    std::vector<Instr>& code;
    std::vector<Cplx>&  consts;
    std::string         err;
    std::size_t         depth = 0;
    std::size_t         max_depth = 0;

    Parser(const std::string& s, std::vector<Instr>& c, std::vector<Cplx>& k)
        : src(s), code(c), consts(k) {}

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

    void push(Op op, uint32_t arg = 0) {
        code.push_back({op, arg});
        switch (op) {
            case Op::PushConst: case Op::PushZ: case Op::PushA:
                ++depth; break;
            case Op::Add: case Op::Sub: case Op::Mul:
            case Op::Div: case Op::Pow:
                --depth; break;              // binary: 2 in, 1 out
            default: break;                  // unary: depth unchanged
        }
        if (depth > max_depth) max_depth = depth;
    }

    uint32_t add_const(Cplx v) {
        consts.push_back(v);
        return static_cast<uint32_t>(consts.size() - 1);
    }

    bool fail(const std::string& msg) {
        if (err.empty()) {
            err = msg + " (at character " + std::to_string(pos + 1) + ")";
        }
        return false;
    }

    // --- grammar ------------------------------------------------------------
    bool parse_expr() {
        if (!parse_term()) return false;
        for (;;) {
            skip_ws();
            if (accept('+'))      { if (!parse_term()) return false; push(Op::Add); }
            else if (accept('-')) { if (!parse_term()) return false; push(Op::Sub); }
            else return true;
        }
    }

    bool parse_term() {
        if (!parse_factor()) return false;
        for (;;) {
            skip_ws();
            if (accept('*'))      { if (!parse_factor()) return false; push(Op::Mul); }
            else if (accept('/')) { if (!parse_factor()) return false; push(Op::Div); }
            else return true;
        }
    }

    bool parse_factor() {
        if (!parse_unary()) return false;
        skip_ws();
        if (accept('^')) {
            if (!parse_factor()) return false;   // right-associative
            push(Op::Pow);
        }
        return true;
    }

    bool parse_unary() {
        skip_ws();
        if (accept('-')) {
            if (!parse_unary()) return false;
            push(Op::Neg);
            return true;
        }
        if (accept('+')) return parse_unary();
        return parse_atom();
    }

    bool parse_atom() {
        skip_ws();
        if (pos >= src.size()) return fail("unexpected end of expression");

        const char c = src[pos];

        // parenthesised sub-expression
        if (c == '(') {
            ++pos;
            if (!parse_expr()) return false;
            if (!accept(')')) return fail("expected ')'");
            return true;
        }

        // number literal (optionally followed by 'i')
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const char* start = src.c_str() + pos;
            char* end = nullptr;
            const double v = std::strtod(start, &end);
            if (end == start) return fail("malformed number");
            pos += static_cast<std::size_t>(end - start);
            // allow "2i" as shorthand for 2*i
            if (pos < src.size() && (src[pos] == 'i' || src[pos] == 'j')) {
                ++pos;
                push(Op::PushConst, add_const(Cplx(0.0, v)));
            } else {
                push(Op::PushConst, add_const(Cplx(v, 0.0)));
            }
            return true;
        }

        // identifier: variable, imaginary unit, constant, or function
        if (std::isalpha(static_cast<unsigned char>(c))) {
            const std::size_t start = pos;
            while (pos < src.size() &&
                   std::isalnum(static_cast<unsigned char>(src[pos]))) ++pos;
            const std::string id = src.substr(start, pos - start);

            if (id == "z")  { push(Op::PushZ); return true; }
            if (id == "a" || id == "c" || id == "lambda") { push(Op::PushA); return true; }
            if (id == "i" || id == "j") {
                push(Op::PushConst, add_const(Cplx(0.0, 1.0))); return true;
            }
            if (id == "pi") {
                push(Op::PushConst, add_const(Cplx(3.14159265358979323846, 0.0)));
                return true;
            }
            if (id == "e") {
                push(Op::PushConst, add_const(Cplx(2.71828182845904523536, 0.0)));
                return true;
            }

            // function call
            Op fop;
            if      (id == "exp")  fop = Op::Exp;
            else if (id == "log")  fop = Op::Log;
            else if (id == "sin")  fop = Op::Sin;
            else if (id == "cos")  fop = Op::Cos;
            else if (id == "sqrt") fop = Op::Sqrt;
            else if (id == "conj") fop = Op::Conj;
            else if (id == "abs")  fop = Op::Abs;
            else if (id == "re")   fop = Op::Re;
            else if (id == "im")   fop = Op::Im;
            else return fail("unknown identifier '" + id + "'");

            if (!accept('(')) return fail("expected '(' after '" + id + "'");
            if (!parse_expr()) return false;
            if (!accept(')')) return fail("expected ')'");
            push(fop);
            return true;
        }

        return fail(std::string("unexpected character '") + c + "'");
    }
};

}  // namespace

// -----------------------------------------------------------------------------
bool Expr::compile(const std::string& source, std::string& error) {
    code_.clear();
    constants_.clear();
    depth_ = 0;
    source_ = source;

    Parser p(source, code_, constants_);
    if (!p.parse_expr()) {
        error = p.err;
        code_.clear();
        return false;
    }
    if (!p.eof()) {
        error = "unexpected trailing input at character " +
                std::to_string(p.pos + 1);
        code_.clear();
        return false;
    }
    if (p.depth != 1) {
        error = "malformed expression (stack did not reduce to one value)";
        code_.clear();
        return false;
    }
    depth_ = p.max_depth;
    error.clear();
    return true;
}

// -----------------------------------------------------------------------------
Cplx Expr::eval(Cplx z, Cplx a, Cplx* st) const {
    std::size_t sp = 0;
    for (const Instr& in : code_) {
        switch (in.op) {
            case Op::PushConst: st[sp++] = constants_[in.arg]; break;
            case Op::PushZ:     st[sp++] = z; break;
            case Op::PushA:     st[sp++] = a; break;

            case Op::Add: { Cplx r = st[--sp]; st[sp-1] += r; break; }
            case Op::Sub: { Cplx r = st[--sp]; st[sp-1] -= r; break; }
            case Op::Mul: { Cplx r = st[--sp]; st[sp-1] *= r; break; }
            case Op::Div: { Cplx r = st[--sp]; st[sp-1] /= r; break; }
            case Op::Pow: {
                Cplx r = st[--sp];
                Cplx b = st[sp-1];
                // integer exponents are common (z^3) and much cheaper, and
                // avoid the branch cut std::pow introduces for negative bases
                if (r.imag() == 0.0 && r.real() == std::floor(r.real()) &&
                    std::abs(r.real()) <= 64.0) {
                    int n = static_cast<int>(r.real());
                    const bool inv = n < 0;
                    if (inv) n = -n;
                    Cplx acc(1.0, 0.0), base = b;
                    while (n) {
                        if (n & 1) acc *= base;
                        base *= base;
                        n >>= 1;
                    }
                    st[sp-1] = inv ? Cplx(1.0, 0.0) / acc : acc;
                } else {
                    st[sp-1] = std::pow(b, r);
                }
                break;
            }

            case Op::Neg:  st[sp-1] = -st[sp-1]; break;
            case Op::Exp:  st[sp-1] = std::exp(st[sp-1]); break;
            case Op::Log:  st[sp-1] = std::log(st[sp-1]); break;
            case Op::Sin:  st[sp-1] = std::sin(st[sp-1]); break;
            case Op::Cos:  st[sp-1] = std::cos(st[sp-1]); break;
            case Op::Sqrt: st[sp-1] = std::sqrt(st[sp-1]); break;
            case Op::Conj: st[sp-1] = std::conj(st[sp-1]); break;
            case Op::Abs:  st[sp-1] = Cplx(std::abs(st[sp-1]), 0.0); break;
            case Op::Re:   st[sp-1] = Cplx(st[sp-1].real(), 0.0); break;
            case Op::Im:   st[sp-1] = Cplx(st[sp-1].imag(), 0.0); break;
        }
    }
    return sp ? st[0] : Cplx(0.0, 0.0);
}

Cplx Expr::operator()(Cplx z, Cplx a) const {
    std::vector<Cplx> st(depth_ ? depth_ : 1);
    return eval(z, a, st.data());
}

}  // namespace cdx
