// =============================================================================
// cdx/composed.hpp -- rational maps kept in COMPOSED form.
//
//     f = g_k^(n_k) o ... o g_2^(n_2) o g_1^(n_1)
//
// where every stage g_i is an ordinary RationalMap with its parameter bound,
// applied n_i times in a row. The composition is a rational map of degree
// prod deg(g_i)^(n_i), but it is never expanded: a degree-10^7 map has no
// usable P/Q form (the coefficients are meaningless in double precision and
// the evaluation cost would be the degree), while the composed form costs
// sum n_i * (cost of g_i) per evaluation and is as accurate as its stages.
//
// This is a GENERAL capability, not tied to any research question: any map
// built as a composition (iterates of a map, a map conjugated by a Mobius
// transformation, a post-composition such as the Fisher-Hill-Lazebnik-
// Thompson construction M o P^n o R) can be rendered this way. Map::composed
// wraps one so that render_julia / render_basin / render_greens run on it
// unchanged -- they only ever call step().
//
// SPHERE-AWARE, stage by stage. Each stage is evaluated in both charts of the
// Riemann sphere, so infinity is an ordinary point all the way through the
// composition:
//   * a term-backed stage (polynomial terms + pole terms) knows its value at
//     infinity exactly from its terms (pole terms and negative powers vanish
//     there; any positive power sends infinity to infinity);
//   * a P/Q-backed stage is evaluated by Horner in z for |z| <= 1 and in the
//     reciprocal chart w = 1/z for |z| > 1, R(z) = z^(p-q) P~(w)/Q~(w) with
//     P~, Q~ the reversed polynomials -- no overflow for large |z|, and the
//     exact value at infinity (0, the ratio of leading coefficients, or inf).
// A point counts as infinite when it is non-finite or |z|^2 > 1e300 -- the
// same convention as chordal_distance -- and infinity is returned as
// (inf, 0), the representation Cycle documents. NaN in gives NaN out.
//
// WHAT IS NOT PROVIDED, honestly: critical points, fixed points and
// attractor discovery. Those are root-finding problems of the full degree,
// which is exactly what the composed form exists to avoid; a caller that
// knows its attractors (e.g. from a construction that certifies them) passes
// them to render_basin/render_julia/render_greens as Cycles. Parameter-plane
// modes do not apply either -- every stage's parameter is already bound --
// and return an all-zero image for a composed map.
// =============================================================================
#pragma once

#include "cdx/rational.hpp"

#include <complex>
#include <string>
#include <vector>

namespace cdx {

using Cplx = std::complex<double>;

// One stage of a composition: `map` at parameter `param`, applied `repeat`
// times (repeat >= 1).
struct ComposedStage {
    RationalMap map;
    Cplx        param{0.0, 0.0};
    int         repeat = 1;
};

class ComposedMap {
public:
    ComposedMap() = default;

    // Stages in APPLICATION order: stages[0] is applied first. Throws
    // std::invalid_argument for an empty list, a repeat < 1, a stage of
    // degree < 1, or a P/Q stage whose denominator is identically zero.
    explicit ComposedMap(std::vector<ComposedStage> stages);

    const std::vector<ComposedStage>& stages() const { return stages_; }

    // Total number of stage applications, sum of the repeats.
    int applications() const;

    // prod deg(g_i)^(n_i). A double: exact while below 2^53, and still the
    // right order of magnitude beyond that (log10_degree() is exact there).
    double degree() const { return degree_; }
    double log10_degree() const { return log10_degree_; }

    // f(z) on the sphere; infinity is Cplx(inf, 0) in and out.
    Cplx eval(Cplx z) const;

    // f'(z) by the chain rule, in the plane chart. NaN when z or any
    // intermediate value is infinite (the derivative there needs the other
    // chart, which a caller asking for a plane derivative does not want).
    Cplx deriv(Cplx z) const;

    // Hot-path form of eval, matching Map::step's calling convention.
    void step(double& zr, double& zi) const;

    // Short human-readable description, e.g. "3 stages, deg 2.0e+07".
    std::string describe() const;

    // Compiled form of one stage -- public only so the .cpp's helpers can see
    // it; not part of the API.
    struct Compiled {
        bool pq = false;
        int  repeat = 1;
        // term-backed: polynomial terms (nonzero coefficients only)
        std::vector<double> cr, ci;
        std::vector<int>    ce;
        // term-backed: simple poles, structure-of-arrays for the hot loop
        std::vector<double> lr, li, sr, si;
        // term-backed: poles of order >= 2
        std::vector<double> hlr, hli, hsr, hsi;
        std::vector<int>    hord;
        bool   inf_to_inf = false;     // value at infinity is infinity
        double inf_r = 0.0, inf_i = 0.0;
        // P/Q-backed: ascending coefficients, trailing exact zeros trimmed
        std::vector<double> pr, pi, qr, qi;
    };

private:
    std::vector<ComposedStage> stages_;
    std::vector<Compiled>      compiled_;
    double degree_ = 1.0;
    double log10_degree_ = 0.0;
};

}  // namespace cdx
