// =============================================================================
// cdx/expr.hpp -- compact complex-expression evaluator.
//
// WHY THIS EXISTS. The core hard-codes six families for speed, but the goal is
// a distributable application where a user can type an arbitrary map such as
//     z^5 + a/z^2 - 0.3
// and explore it. Two ways to support that:
//
//   (a) generate C++ from the expression and compile at run time -- native
//       speed, but requires a C++ toolchain on the END USER's machine, which
//       is incompatible with shipping a single self-contained executable;
//   (b) parse once into a compact stack machine and interpret it -- no
//       compiler needed, ships inside the binary, works anywhere.
//
// (b) is the only option compatible with distribution, so that is what this
// is. It costs perhaps 3-8x versus a hard-coded family, which is acceptable
// against a baseline that is already ~20x faster than the MATLAB original.
// Built-in families keep the fast path; this handles everything else.
//
// GRAMMAR (standard precedence, right-associative ^)
//     expr    := term (('+' | '-') term)*
//     term    := factor (('*' | '/') factor)*
//     factor  := unary ('^' factor)?          -- right associative
//     unary   := ('-' | '+')? atom
//     atom    := number | 'i' | 'z' | 'a' | func '(' expr ')' | '(' expr ')'
//     func    := exp | log | sin | cos | sqrt | conj | abs | re | im
//
// Numbers are real literals; the imaginary unit is `i` (so 2i is written
// 2*i or 2i -- both accepted). Variables are `z` (the point) and `a` (the
// parameter). Implicit multiplication is NOT supported except for the
// number-followed-by-i case, to keep parsing unambiguous.
// =============================================================================
#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace cdx {

using Cplx = std::complex<double>;

// -----------------------------------------------------------------------------
// Stack-machine opcodes.
// -----------------------------------------------------------------------------
enum class Op : std::uint8_t {
    PushConst,   // push constants_[arg]
    PushZ,       // push the current point z
    PushA,       // push the parameter a
    Add, Sub, Mul, Div, Pow,
    Neg,
    Exp, Log, Sin, Cos, Sqrt, Conj, Abs, Re, Im
};

struct Instr {
    Op       op;
    uint32_t arg = 0;   // index into the constant pool, for PushConst
};

// -----------------------------------------------------------------------------
// A parsed expression in z and a.
//
// Thread-safety: eval() writes only to a caller-supplied scratch stack, so a
// single Expr can be evaluated concurrently from many threads (which the
// renderer does).
// -----------------------------------------------------------------------------
class Expr {
public:
    Expr() = default;

    // Parse `source`. On failure returns false and fills `error`.
    bool compile(const std::string& source, std::string& error);

    bool valid() const { return !code_.empty(); }
    const std::string& source() const { return source_; }

    // Maximum stack depth the program needs; size scratch buffers with this.
    std::size_t stack_depth() const { return depth_; }

    // Evaluate at (z, a). `scratch` must have at least stack_depth() entries;
    // passing it in avoids allocating inside the iteration loop.
    Cplx eval(Cplx z, Cplx a, Cplx* scratch) const;

    // Convenience for non-hot paths: allocates its own scratch.
    Cplx operator()(Cplx z, Cplx a) const;

private:
    std::string        source_;
    std::vector<Instr> code_;
    std::vector<Cplx>  constants_;
    std::size_t        depth_ = 0;
};

}  // namespace cdx
