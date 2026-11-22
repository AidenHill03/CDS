// =============================================================================
// cdx/rational.hpp -- term-based rational map builder (the sandbox core).
//
// A RationalMap is held in PARTIAL-FRACTION form:
//
//     R(z) = SUM_k  c_k a^{q_k} z^{e_k}          (polynomial terms)
//          + SUM_j  s_j a^{r_j} / (z - p_j)^{m_j}   (pole terms)
//
// Why this form rather than a formula string:
//
//   * POLES ARE EXPLICIT OBJECTS. Each carries a location, an order and a
//     strength that can be inspected and edited individually. Since the
//     poles of r^n are r^{-n}(infinity) and their closure is the Julia set,
//     making poles first-class is what turns the program into an instrument
//     for controlling boundary geometry rather than just drawing it.
//   * TERMS ADD AND REMOVE CLEANLY. A UI can list terms and let the user
//     insert, delete and retune them, which a parsed string cannot support
//     without re-parsing and losing identity.
//   * IT SUBSUMES THE EXISTING FAMILIES. z^n + a/z^n is one polynomial term
//     plus one pole of order n at the origin with strength a; z^2 + a is one
//     polynomial term plus a constant term whose coefficient is the
//     parameter. Nothing is lost by moving to this representation.
//   * IT STAYS DIFFERENTIABLE IN CLOSED FORM. Each term's derivative is
//     another term of the same kind, so critical points come from a real
//     polynomial equation rather than numerical differentiation.
//
// PARAMETER BINDING. Every coefficient is c * a^q. q = 0 gives a fixed
// coefficient; q = 1 makes the term scale linearly with the parameter, which
// is how the classical families depend on it. Higher/negative q are allowed.
//
// A SECOND, P/Q-BACKED representation (Stage 2 of the P/Q milestone) lives
// alongside the term-based one above, in the SAME class: RationalMap::
// from_canonical builds a map directly from a cdx::CanonicalRational
// (rational_parser.hpp) instead of from terms, and every method below
// branches internally on which representation this particular instance
// holds (see is_pq_backed()) -- poly_/pole_ simply stay empty for a P/Q-
// backed map, and its own storage (pq_) stays empty for a term-based one.
// This keeps every existing consumer (Renderer, cdx::analysis, the Python
// bindings) working against the SAME `const RationalMap&`/`const
// RationalMap*` it already holds, with no signature changes anywhere --
// see ARCHITECTURE.md's own "insulated" framing for why that matters more
// than which representation happens to be faster to build a given map
// from. A P/Q-backed map is inherently SINGLE-parameter: whatever one
// parameter (if any) the source expression referenced is always bound to
// the `a` a caller passes to eval/deriv/etc, regardless of its literal
// name -- see from_canonical's own comment for why selecting among SEVERAL
// parsed parameters is deliberately not this class's job.
// =============================================================================
#pragma once

#include "cdx/rational_parser.hpp"

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace cdx {

using Cplx = std::complex<double>;

// -----------------------------------------------------------------------------
// c * a^param_power * z^exponent
// -----------------------------------------------------------------------------
struct PolyTerm {
    Cplx coeff       {1.0, 0.0};
    int  exponent    = 1;      // may be negative: that is a pole at the origin
    int  param_power = 0;      // multiply the coefficient by a^param_power
    bool enabled     = true;   // lets the UI mute a term without deleting it
    std::string label;         // optional user note

    Cplx effective_coeff(Cplx a) const;
};

// -----------------------------------------------------------------------------
// s * a^param_power / (z - location)^order
// -----------------------------------------------------------------------------
struct PoleTerm {
    Cplx location    {0.0, 0.0};
    Cplx strength    {1.0, 0.0};
    int  order       = 1;      // must be >= 1
    int  param_power = 0;
    bool enabled     = true;
    // If true, the pole LOCATION tracks the parameter (location = a). Useful
    // for families whose singularity moves through the plane as a varies.
    bool location_is_param = false;
    std::string label;

    Cplx effective_strength(Cplx a) const;
    Cplx effective_location(Cplx a) const;
};

// -----------------------------------------------------------------------------
// A fixed point R(point) == point, with its multiplier deriv(point, a) --
// |multiplier| < 1 attracting, > 1 repelling, == 1 neutral, == 0
// superattracting (always true when point is also a critical point).
// -----------------------------------------------------------------------------
struct FixedPoint {
    Cplx point;
    Cplx multiplier;
};

// -----------------------------------------------------------------------------
// Stage 2 (spatial continuation) results -- see RationalMap::distinct_
// critical_points_continued/fixed_points_continued's own doc comments for
// the full contract. `distinct`/`points` are byte-for-byte what the cold
// (non-continued) call of the same name would return -- continuation is
// purely a SPEED path, never a different answer. `ordinary_raw`/`raw` is
// what a caller sweeping a grid of nearby parameter values should feed back
// in as the NEXT spatially adjacent pixel's own predictor -- deliberately
// NOT `distinct`/`points` themselves, which (for critical points) mix in
// pole-copy/infinity entries that are cheap and always recomputed fresh,
// never continued, and (for fixed points) mix in the infinity entry the
// same way. `continued` is true iff continuation actually ran (false means
// a cold fallback fired instead) -- exposed for Stage 2's own speed/
// correctness verification and timing report, not needed by an ordinary
// caller.
struct ContinuedCriticalPoints {
    std::vector<Cplx> distinct;
    std::vector<Cplx> ordinary_raw;
    bool continued = false;
};
struct ContinuedFixedPoints {
    std::vector<FixedPoint> points;
    std::vector<Cplx> raw;
    bool continued = false;
};

// -----------------------------------------------------------------------------
// Which critical points are FREE -- see RationalMap::free_critical_structure.
// Computed once per render (it costs three cold critical/fixed-point solves),
// then applied per pixel by RationalMap::free_critical_points.
//
// THE RULE. A critical point c is STRUCTURALLY PRE-FIXED -- not free -- when,
// at each of three generic parameter values (kGenericParams in rational.cpp),
// c is a critical point at the same location and either c is a fixed point
// or f(c) is a fixed point, the same one at all three. Its orbit then says
// nothing about the parameter: it is the same at every `a`. Identification
// is by location, with two kinds of point identified by type instead,
// because their location is a closed-form consequence of the map's shape:
//   * infinity, when it is a critical point: pre-fixed iff infinity is a
//     fixed point at all three (every polynomial: f(inf) = inf);
//   * a pole of order >= 2 (critical in its own right): f(pole) = inf, so it
//     is pre-fixed iff infinity is a fixed point at all three -- even when
//     the pole MOVES with `a` (e.g. relaxed Newton of z^n - 1, whose z = 0 is
//     an order n-1 pole mapping to the fixed point at infinity).
// A finite, non-pole critical point that is pre-fixed but moves with `a`
// (e.g. c = a for (z - a)^2 + a) is NOT recognised and stays "free"; its
// orbit then reports a constant period 1, which is honest, just
// uninformative.
// -----------------------------------------------------------------------------
struct FreeCriticalStructure {
    std::vector<Cplx> excluded_finite;   // parameter-independent, pre-fixed, finite, non-pole
    bool exclude_infinity = false;       // infinity (when critical) is pre-fixed
    bool exclude_poles    = false;       // every pole-critical point is pre-fixed
    int  generic_free_count = 0;         // max #free distinct critical points at the generic params
};

// -----------------------------------------------------------------------------
// THE CRITICAL-POINT LABELLING CONVENTION -- what `critical_index` means in
// every mode (render_parameter's PerCritical, polynomial or rational, and
// render_parameter_period's FromMap), and what RationalMap::
// critical_point_labels returns for the Facts panel.
//
// Free critical points can collide (z^3 + a z + 1 at a = 0), and going once
// around a collision swaps them, so no global labelling is continuous
// everywhere: there must be a cut. The convention makes the cut
// deterministic, documented and independent of the viewport:
//   1. A fixed base parameter a* per map: 0.618 + 0.382i, nudged (by
//      +0.0137i steps) if any two free critical points there are within
//      1e-6 (relative) of each other or fewer than generic exist.
//   2. At a*, the free FINITE non-pole critical points (roots of the
//      derivative numerator) are labelled 0, 1, ... in order of (Re, Im),
//      each rounded to 1e-9.
//   3. For a pixel parameter a, those points are continued along the
//      straight segment a* -> a: adaptive steps, a secant predictor, a
//      Newton corrector on the derivative-numerator polynomial; a step is
//      halved when Newton fails, a point jumps more than a quarter of its
//      distance to its nearest neighbour, or two tracked points come within
//      the collapse tolerance (1e-9 relative).
//   4. Free critical points that are not tracked roots -- infinity and
//      pole copies, both closed-form -- follow, in (Re, Im) order.
// Labels index the FREE critical points only (FreeCriticalStructure): a
// structurally pre-fixed point -- infinity for a polynomial, or 0 and
// infinity for z^n + a/z^n -- never gets a label, so critical_index 0 can
// never silently mean "infinity", whose orbit is the same for every a.
// So the cuts are the rays from each collision point pointing directly
// away from a*, and they do not move when the user pans or zooms. If a
// continuation fails outright (the segment passes within ~1e-7 of a
// collision, i.e. the pixel lies ON a cut, or the root count changes), the
// pixel falls back to step 2's sort applied at a itself.
// -----------------------------------------------------------------------------
// The derivative numerator N_a(z) = sum_k c_k(a) z^k with every c_k(a)
// itself a polynomial in a, compiled once so the labelling continuation can
// evaluate N at a new parameter in a few dozen flops instead of rebuilding
// it through the P/Q expression evaluator (measured: that rebuild, paid on
// every continuation step including rejected ones, was nearly all of the
// labelling cost). Built by RationalMap::compile_deriv_numerator_in_a: fitted
// from 16 samples on |a| = 1 and accepted only if it reproduces
// deriv_numerator_poly at four other parameters (|a| up to ~4) to 1e-10
// relative; otherwise valid == false and callers use the direct path. (The
// parser only admits rational expressions and clears denominators in a too,
// so the coefficients are always polynomial in a; what fails is degree
// >= 16 in a, which 16 samples cannot capture.)
//
// The same fit serves any per-`a` coefficient vector that is polynomial in
// a: RationalMap::compile_polynomial_coefficients_in_a fits f's OWN
// coefficients (polynomial_coefficients) so the certified escape radius can
// be computed per parameter pixel without the expression evaluator.
struct PolyCoeffsInA {
    bool valid = false;
    int  z_terms = 0;                 // number of z-coefficients
    int  a_terms = 0;                 // number of a-coefficients per z-coefficient
    std::vector<Cplx> coef;           // coef[k * a_terms + j] multiplies a^j z^k
    // The coefficients (ascending in z) at `a` into `out`, resized to z_terms.
    void eval(Cplx a, std::vector<Cplx>& out) const;
};
using DerivNumeratorInA = PolyCoeffsInA;

struct CriticalLabelBase {
    bool  constant = false;           // critical points do not move: labels are just `points`
    Cplx  a_star{0.618, 0.382};
    std::vector<Cplx> points;         // free tracked critical points at a_star, label order
    int   expected_degree = 0;        // effective degree of the derivative numerator at a_star
    bool  closed_form_free = false;   // infinity / pole copies can be free (step 4 needed)
    FreeCriticalStructure free_s;
    DerivNumeratorInA fast;           // speed only: same polynomial, evaluated faster
};

// -----------------------------------------------------------------------------
// RationalMap::eval(z, a) redoes every term's effective_coeff/
// effective_location/effective_strength -- each an a^param_power call -- on
// EVERY invocation, even though every caller that matters (an escape-time
// loop) holds `a` fixed across many hundreds of calls in a row. It also
// walks generic std::complex-typed terms, whose operator* carries inf/nan
// branch handling and whose operator/ carries Smith's-algorithm branch
// handling -- both measurably expensive run resolution^2 * max_iter times
// (see CLAUDE.md's hand-rolled-arithmetic-in-hot-loops convention, already
// used throughout renderer.cpp's built-in Family formulas, just not
// previously extended to the generic RationalMap path).
//
// RationalMap::compile(a) does the parameter substitution once and flattens
// every term to raw double real/imag pairs, so CompiledMap::step() in the
// hot loop is a plain hand-rolled sum: no per-call exponentiation, no
// `enabled` branching, and no std::complex anywhere. See step()'s own
// comment for the exponent/division specifics.
// -----------------------------------------------------------------------------
class CompiledMap {
public:
    // c * z^exponent, coefficient already evaluated at the bound parameter.
    struct PolyOp { double cr, ci; int exponent; };
    // s * (z - location)^-order, likewise.
    struct PoleOp { double lr, li, sr, si; int order; };

    // Mutates (zr, zi) in place, matching Map::step's own calling
    // convention. Header-inline so a hot-loop call site can actually be
    // inlined, not just declare intent to be -- see cdx::detail::cipow in
    // this header for the arithmetic itself.
    //
    // BRANCHES ONCE per call on which representation this CompiledMap was
    // built from (is_pq_), not once per RENDER -- but that branch is
    // exactly as predictable as the existing `t.ci == 0.0` branch below
    // (same value on every one of the many thousands of calls in a single
    // render, so free after the first), the same reasoning this class's
    // own doc comment already applies to that branch.
    inline void step(double& zr, double& zi) const;

    // Cold-path companions (Newton refinement and multipliers in
    // critical_orbits.cpp -- a few dozen calls per orbit, not per step):
    // f(z) and f'(z) in std::complex, and infinity's status read off the
    // compiled coefficients (see InfinityStatus, critical_orbits.hpp).
    // eval_c maps a pole to (1e300, 0), the same sentinel step() uses.
    Cplx eval_c(Cplx z) const;
    Cplx deriv_c(Cplx z) const;
    // f(inf) == inf? If so, local degree there (>= 2: superattracting) and
    // the multiplier in the w = 1/z chart (0 when superattracting). If not,
    // `image` is f(inf) (the leading-coefficient ratio when deg P == deg Q,
    // else 0).
    void infinity_shape(bool& fixed, int& local_degree, Cplx& multiplier, Cplx& image) const;

private:
    friend class RationalMap;
    bool is_pq_ = false;
    std::vector<PolyOp> poly_;
    std::vector<PoleOp> pole_;
    // P/Q Horner form (used when is_pq_ is true; poly_/pole_ stay empty).
    // Ascending order, coefficients already evaluated at the bound
    // parameter -- exactly like poly_/pole_ above, just P(z)/Q(z) evaluated
    // directly instead of decomposed into partial-fraction terms (a P/Q
    // map has no term list to decompose in the first place).
    std::vector<double> pr_, pi_;   // P's coefficients
    std::vector<double> qr_, qi_;   // Q's coefficients
};

namespace detail {

// z^e (integer e, may be negative) via hand-rolled real/imag doubles --
// never std::complex. |e| <= 4 is unrolled as direct products: the range
// every built-in family and the overwhelming majority of sandbox terms
// actually use, and exactly where generic binary exponentiation wastes
// work -- z^2 costs one multiply this way, three via repeated-squaring
// (see ipow() in rational.cpp, which this mirrors for the general case).
// A negative e returns the reciprocal of z^|e| via a hand-rolled division
// -- 1/(pr+pi*i) = (pr-pi*i)/(pr^2+pi^2) -- with none of
// std::complex::operator/'s Smith's-algorithm branch handling.
inline void cipow(double zr, double zi, int e, double& outr, double& outi) {
    const int ae = e < 0 ? -e : e;
    double pr, pi;
    switch (ae) {
        case 0: pr = 1.0; pi = 0.0; break;
        case 1: pr = zr;  pi = zi;  break;
        case 2: pr = zr * zr - zi * zi; pi = 2.0 * zr * zi; break;
        case 3: {
            const double z2r = zr * zr - zi * zi, z2i = 2.0 * zr * zi;
            pr = z2r * zr - z2i * zi; pi = z2r * zi + z2i * zr;
            break;
        }
        case 4: {
            const double z2r = zr * zr - zi * zi, z2i = 2.0 * zr * zi;
            pr = z2r * z2r - z2i * z2i; pi = 2.0 * z2r * z2i;
            break;
        }
        default: {
            // Hand-rolled binary exponentiation for the rare larger
            // exponent -- same algorithm as rational.cpp's ipow(), real/
            // imag doubles throughout instead of std::complex.
            double br = zr, bi = zi;
            pr = 1.0; pi = 0.0;
            for (int n = ae; n; n >>= 1) {
                if (n & 1) {
                    const double nr = pr * br - pi * bi, ni = pr * bi + pi * br;
                    pr = nr; pi = ni;
                }
                const double nbr = br * br - bi * bi, nbi = 2.0 * br * bi;
                br = nbr; bi = nbi;
            }
            break;
        }
    }
    if (e >= 0) { outr = pr; outi = pi; return; }
    // One division (the reciprocal of den) shared by both components,
    // rather than two separate divisions -- real hardware division costs
    // several times a multiply, so trading a second divide for a multiply
    // is a real win, not just a style choice.
    const double den = pr * pr + pi * pi;
    const double inv_den = 1.0 / den;
    outr = pr * inv_den;
    outi = -pi * inv_den;
}

}  // namespace detail

inline void CompiledMap::step(double& zr, double& zi) const {
    if (is_pq_) {
        // Horner-evaluate P and Q at the current z, hand-rolled real/imag
        // throughout (no std::complex -- same convention as detail::cipow
        // above), then divide via the SAME one-division reciprocal trick.
        double pr = pr_.empty() ? 0.0 : pr_.back();
        double pi = pi_.empty() ? 0.0 : pi_.back();
        for (std::size_t k = pr_.size(); k-- > 1;) {
            const double nr = pr * zr - pi * zi + pr_[k - 1];
            const double ni = pr * zi + pi * zr + pi_[k - 1];
            pr = nr; pi = ni;
        }
        double qr = qr_.empty() ? 0.0 : qr_.back();
        double qi = qi_.empty() ? 0.0 : qi_.back();
        for (std::size_t k = qr_.size(); k-- > 1;) {
            const double nr = qr * zr - qi * zi + qr_[k - 1];
            const double ni = qr * zi + qi * zr + qi_[k - 1];
            qr = nr; qi = ni;
        }
        if (qr == 0.0 && qi == 0.0) { zr = 1e300; zi = 0.0; return; }
        const double inv_den = 1.0 / (qr * qr + qi * qi);
        const double numr = pr * qr + pi * qi;
        const double numi = pi * qr - pr * qi;
        zr = numr * inv_den;
        zi = numi * inv_den;
        return;
    }

    double sumr = 0.0, sumi = 0.0;
    for (const auto& t : poly_) {
        double pr, pi;
        detail::cipow(zr, zi, t.exponent, pr, pi);
        // A REAL coefficient (ci == 0) is the common case -- every built-in
        // preset's terms are all real (see mandelbrot/multibrot/mcmullen/
        // newton_cubic in rational.cpp), and so is a plain user-typed
        // number. Skip the full complex multiply's two wasted
        // multiply-by-zero-and-add operations for it; the branch itself is
        // free after the first call, since t.ci's value (and therefore
        // which side of it any one term takes) never changes across the
        // many thousands of step() calls in a single render.
        if (t.ci == 0.0) { sumr += t.cr * pr; sumi += t.cr * pi; }
        else { sumr += t.cr * pr - t.ci * pi; sumi += t.cr * pi + t.ci * pr; }
    }
    for (const auto& t : pole_) {
        const double dr = zr - t.lr, di = zi - t.li;
        // A point exactly at a pole maps to infinity; signal it with the
        // same huge sentinel RationalMap::eval() uses, so a caller's
        // escape test fires rather than seeing NaN.
        if (dr == 0.0 && di == 0.0) { zr = 1e300; zi = 0.0; return; }
        double invr, invi;
        detail::cipow(dr, di, -t.order, invr, invi);   // 1 / (z-location)^order
        if (t.si == 0.0) { sumr += t.sr * invr; sumi += t.sr * invi; }
        else { sumr += t.sr * invr - t.si * invi; sumi += t.sr * invi + t.si * invr; }
    }
    zr = sumr;
    zi = sumi;
}

// -----------------------------------------------------------------------------
// The map itself: a named collection of terms.
// -----------------------------------------------------------------------------
class RationalMap {
public:
    RationalMap() = default;
    explicit RationalMap(std::string name) : name_(std::move(name)) {}

    // --- P/Q-backed construction (Stage 2 of the P/Q milestone) --------------
    // Builds a P/Q-BACKED RationalMap directly from a Stage-1
    // CanonicalRational (rational_parser.hpp) instead of from terms --
    // poly_terms()/pole_terms() stay empty for a map built this way (see
    // is_pq_backed()). `cr` must have AT MOST ONE parameter: this engine
    // stays single-active-parameter, and selecting WHICH of several parsed
    // parameters plays that role -- substituting the rest as fixed
    // constants first -- is a LATER stage's job, layered on top of this
    // constructor via building a fresh single-parameter CanonicalRational
    // to pass in, not something this constructor does itself. Throws
    // std::invalid_argument if `cr` has more than one parameter.
    static RationalMap from_canonical(CanonicalRational cr, std::string name = "untitled");

    // --- P/Q-backed construction from a MULTI-parameter source (Stage 3) ----
    // Parses `source`, then reduces it to the engine's own single-active-
    // parameter P/Q form: `active_param` stays a free parameter (bound to
    // whatever `a` a caller passes to eval/deriv/etc, same as
    // from_canonical); every OTHER parameter the source references is
    // SUBSTITUTED with its value from `fixed_values`, turning that
    // coefficient position into a plain constant before construction (see
    // substitute_param, rational_parser.hpp) -- not carried forward as a
    // second live parameter, since this engine is single-active-parameter
    // by design (see the class comment).
    //
    // `active_param` may be EMPTY iff the source has zero or exactly one
    // parameter -- "exactly one parameter -> auto-active". For a source
    // with two or more parameters, `active_param` must name one of them
    // explicitly; there is no default to fall back on when the choice is
    // genuinely ambiguous.
    //
    // The ORIGINAL, unsubstituted source text, the chosen active
    // parameter, and every substituted value are all preserved (pq_source
    // () and to_formula() return the literal authored text, not a
    // reconstruction from the reduced P/Q; serialize()/deserialize()
    // round-trip all three -- see their own comments) even though every
    // OTHER method only ever touches the reduced, single-parameter P/Q.
    //
    // Throws std::invalid_argument if: `source` fails to parse;
    // `active_param` is non-empty but names something other than one of
    // `source`'s own parameters (or is given at all when there are zero);
    // `active_param` is empty when `source` has more than one parameter;
    // or `fixed_values` is missing an entry for some non-active parameter.
    static RationalMap from_expression(const std::string& source, const std::string& active_param,
                                       const std::map<std::string, Cplx>& fixed_values,
                                       std::string name = "untitled");

    // TRUE iff this instance is P/Q-backed (built via from_canonical or
    // from_expression, not add_poly/add_pole) -- every method below
    // branches on this internally, but a caller that only needs to know
    // WHICH representation a map is using (e.g. to skip a representation-
    // specific fast path -- see Renderer::recognize_family's own doc
    // comment) can ask directly rather than inferring it from poly_terms()
    // /pole_terms() both being empty (which is also true of a genuinely
    // empty term-based map).
    bool is_pq_backed() const { return pq_ != nullptr; }

    // P/Q-backed maps only: the numerator and denominator coefficients,
    // ASCENDING (index k multiplies z^k), evaluated at parameter `a`.
    // Returns false (and leaves P, Q untouched) for a term-backed map. Used
    // by ComposedMap (cdx/composed.hpp) to evaluate a P/Q stage in both
    // charts of the sphere, which needs the coefficients themselves rather
    // than CompiledMap's one-chart Horner form.
    bool pq_coefficients(Cplx a, std::vector<Cplx>& P, std::vector<Cplx>& Q) const;

    // The authored source expression this map was built from, if
    // is_pq_backed() -- empty string otherwise. For from_expression, this
    // is the ORIGINAL multi-parameter text (e.g. "b*z^2 + a"), NOT the
    // reduced single-parameter form computation actually uses -- prefer
    // this over to_formula() when it's non-empty: it's the user's own text
    // verbatim, not a reconstruction.
    const std::string& pq_source() const;

    // Which parsed parameter is active (bound to eval/deriv/etc's own
    // `a`), if is_pq_backed() -- empty string otherwise (including the
    // genuine zero-parameter case, e.g. Newton's method typed directly).
    const std::string& pq_active_param() const;

    // name -> value for every OTHER parameter from_expression substituted
    // as a constant -- empty if is_pq_backed() is false, or if the source
    // had at most one parameter (nothing to substitute).
    const std::map<std::string, Cplx>& pq_fixed_params() const;

    // --- P/Q-backed direct pole/zero addition (Stage 4) ----------------------
    // Forward root->factor on the canonical P/Q: adding a pole multiplies
    // Q by (z-location) (R_new(z) = R_old(z) / (z-location)); adding a
    // zero multiplies P by (z-location) (R_new(z) = R_old(z) *
    // (z-location)). The analogue, for a P/Q-backed map, of add_pole/
    // add_poly above (which only ever operate on a term-based one).
    // pq_source()/to_formula() update to match -- wrapping the existing
    // authored text in the new factor, which is exact (re-parsing the
    // result reproduces the identical P/Q, up to floating-point printing
    // precision), not an approximation. Calling this repeatedly at the
    // SAME location raises that pole's/zero's order by one each time --
    // not rejected as a collision (unlike add_pole's own term-based
    // check), since a P/Q map has no per-pole identity to collide with,
    // just a polynomial that can have as high a root multiplicity as the
    // caller asks for.
    //
    // PRESCRIBING a pole/zero so that R has some SPECIFIC desired
    // dynamical behavior (a chosen fixed point, a chosen critical orbit,
    // ...) is the INVERSE of this -- deliberately OUT OF SCOPE, left for
    // separate future work; this is only the forward direction (place a
    // root at a chosen LOCATION, see what the dynamics turn out to be).
    //
    // Throws std::logic_error if !is_pq_backed().
    void add_pole_at(Cplx location);
    void add_zero_at(Cplx location);

    // --- identity -----------------------------------------------------------
    const std::string& name() const { return name_; }
    void set_name(std::string n) { name_ = std::move(n); }
    const std::string& notes() const { return notes_; }
    void set_notes(std::string n) { notes_ = std::move(n); }

    // --- term management ----------------------------------------------------
    // Throws std::invalid_argument for exponent < 0: that represents a pole
    // (see the class comment's PARAMETER BINDING note and PolyTerm's own
    // comment), and every pole must go through add_pole below instead, so
    // there is exactly one representation for "a pole at a location" rather
    // than two a caller could accidentally create simultaneously (a
    // negative-exponent PolyTerm at the origin AND a PoleTerm there is the
    // SAME mathematical object told twice -- see add_pole's own comment for
    // why that matters). NOT retroactive: deserialize() and an already-
    // existing RationalMap built before this restriction (or hand-
    // constructed by pushing directly onto poly_terms(), which -- like
    // pole_terms() -- is still a live-mutable reference this function does
    // not gate) can still carry one; this only stops NEW ones being added
    // through this call.
    std::size_t add_poly(Cplx coeff, int exponent, int param_power = 0,
                         std::string label = {});
    // Throws std::invalid_argument if `location` coincides (within 1e-12,
    // matching pole_locations()'s own "same point" tolerance) with an
    // EXISTING enabled pole -- either another PoleTerm's location, or (for
    // backward compatibility with a map built before add_poly's own
    // restriction above, e.g. one loaded via deserialize) an existing
    // enabled PolyTerm with a negative exponent, which only ever means a
    // pole at the origin. Rejected rather than silently added: own_fraction
    // ()'s shared-denominator construction DOES combine two such
    // representations into the correct true order today (see
    // clear_denominators's own comment on why a redundant common factor
    // still cancels out of degree differences safely) -- this is not
    // papering over a live miscount -- but the result is a pole split
    // across two different term lists that the user can no longer edit as
    // the one thing it conceptually is, and a fragile invariant to keep
    // relying on every future consumer to preserve correctly by hand,
    // exactly the kind of shared-factor subtlety that cost real debugging
    // time getting critical_points() itself right (see CLAUDE.md). Closing
    // off the ambiguous state at its source is more robust than continuing
    // to lean on that cancellation elsewhere. Skipped (not checked at all,
    // so never rejected) when the colliding EXISTING pole has
    // location_is_param set: such a pole's location IS the parameter, so
    // whether it collides with a fixed location depends on which `a` is in
    // play at render time, not anything this call -- parameter-independent
    // by construction -- can know. (The pole being ADDED here can never
    // itself have location_is_param set: this signature has no such
    // parameter, so a fresh PoleTerm always starts with it false; setting
    // it is a later, direct edit via pole_terms()[i].)
    std::size_t add_pole(Cplx location, Cplx strength, int order = 1,
                         int param_power = 0, std::string label = {});

    void remove_poly(std::size_t i);
    void remove_pole(std::size_t i);
    void clear();

    std::vector<PolyTerm>&       poly_terms()       { return poly_; }
    const std::vector<PolyTerm>& poly_terms() const { return poly_; }
    std::vector<PoleTerm>&       pole_terms()       { return pole_; }
    const std::vector<PoleTerm>& pole_terms() const { return pole_; }

    // --- evaluation ---------------------------------------------------------
    Cplx eval(Cplx z, Cplx a) const;

    // Analytic derivative. Every term differentiates to a term of the same
    // kind, so this is exact rather than a finite difference.
    Cplx deriv(Cplx z, Cplx a) const;

    // Bind every term to a fixed parameter value and flatten to raw
    // real/imag doubles, once. Pays off whenever the same `a` will be
    // evaluated more than a handful of times -- which is always, for a
    // Renderer escape-time loop: `a` is fixed for an entire orbit
    // (render_julia/render_basin/render_greens: for the whole render;
    // render_parameter: for one pixel's worth of iterations). See
    // CompiledMap's own doc comment for what this buys over plain eval().
    CompiledMap compile(Cplx a) const;
    // compile(a) into `out`, reusing out's buffers (and a per-thread
    // parameter binding): no allocation once `out` has been filled once.
    // Bit-identical to compile(a). For per-PIXEL callers (the parameter
    // planes), where compile(a)'s allocations were most of the overhead.
    void compile_into(Cplx a, CompiledMap& out) const;

    // --- structure ----------------------------------------------------------
    // Degree as a rational map of the sphere: max(deg numerator, deg
    // denominator) after clearing denominators.
    int degree(Cplx a) const;

    // TRUE iff this map is STRUCTURALLY a polynomial of degree >= 2: the
    // denominator is a nonzero constant (no pole ANYWHERE, for ANY
    // parameter value) and the numerator's degree is at least 2 -- exactly
    // what cdx::analysis::polynomial_escape_certified needs to decide
    // whether the fixed-|z|>R escape-time fast path is a provably
    // forward-invariant trap (see that function's own doc comment for why
    // that specific condition is the right one). Structural, not numeric
    // at one parameter value, for the SAME reason critical_points_constant
    // () above is: a coefficient that happens to evaluate to 0 at one `a`
    // doesn't change what's STRUCTURALLY present for other values, and
    // this predicate has to hold for the whole family, not one point in
    // it. Representation-agnostic -- works identically whether this map is
    // term-based or P/Q-backed (is_pq_backed()) -- so callers outside this
    // class never need to know or care which.
    bool is_polynomial_structurally() const;

    // Locations of the finite poles (deduplicated), including any pole at the
    // origin implied by a negative polynomial exponent.
    std::vector<Cplx> pole_locations(Cplx a) const;

    // TRUE local order at each of pole_locations(a)'s entries, same index
    // for index (pole_orders(a)[k] is the order of pole_locations(a)[k]).
    // Not the nominal order of whichever term(s) happen to sit at that
    // location -- found from R's actual numerator/denominator, so it is
    // correct even when multiple terms share a location (orders do not
    // simply add) or a pole's strength happens to vanish at this particular
    // `a` (order 0 is possible: no real pole there for this parameter).
    std::vector<int> pole_orders(Cplx a) const;

    // Critical points on the RIEMANN SPHERE -- the complete set, from all
    // three sources a rational map can be critical at:
    //   * ordinary points: zeros of the derivative away from any pole,
    //     found by clearing denominators and rooting the resulting
    //     polynomial (see cdx/roots.hpp);
    //   * each pole of true local order m, which is itself critical (an
    //     m-to-1 map near it, same shape as an ordinary critical point of
    //     local degree m), contributing multiplicity m-1;
    //   * infinity, when the map is critical there, contributing
    //     multiplicity |p-q|-1, where p and q are the numerator and
    //     denominator degrees after clearing denominators, whenever
    //     |p-q| >= 2 (and only then -- |p-q| <= 1 means infinity maps
    //     through with local degree 1, not critical).
    // The point at infinity is represented as Cplx(inf, 0), matching
    // Cycle's convention (renderer.hpp) -- this project is sphere-first
    // throughout, and a map critical at infinity (which is most of the
    // built-in polynomial families: z^n + a is critical there for any
    // n >= 2) is not a special case to work around.
    //
    // For a degree-d map this is the complete set: the total count with
    // multiplicity is exactly 2d-2 (Riemann-Hurwitz). That invariant is
    // what the test suite checks, for every built-in family and for
    // randomly generated sandbox maps, precisely because it catches any
    // missing source for any map shape -- which is how the pole and
    // infinity sources above came to be added: an earlier version of this
    // function found only the ordinary points and silently undercounted.
    //
    // MULTIPLICITY. Returned with multiplicity, like cdx::roots(): a
    // multiplicity-k critical point appears k times (numerically close but
    // not necessarily identical estimates for the ordinary/derivative-root
    // case, since those come from Aberth-Ehrlich iteration; the exact same
    // value repeated for the pole/infinity cases, since those multiplicities
    // are known exactly rather than found numerically). Multiplicity is
    // real information for some purposes and noise for others -- e.g.
    // seeding one critical orbit per critical point for attractor
    // discovery, where iterating the same point twice just rediscovers the
    // same attractor a second time. Use distinct_critical_points() below
    // for that case; this function does not discard the information.
    //
    // No particular ordering guarantee.
    std::vector<Cplx> critical_points(Cplx a) const;

    // The same critical points, deduplicated: points within rel_tol of
    // each other (relative to their own magnitude; Cplx(inf,*) matches only
    // other infinite points, never a finite one) collapse to a single
    // representative. What a consumer seeding one orbit per critical point
    // wants -- see the multiplicity note on critical_points() above.
    //
    // rel_tol defaults looser than this codebase's usual "same point"
    // threshold (RenderSettings::tol's 1e-6) because Aberth-Ehrlich's
    // convergence on a genuine multiple root degrades from cubic to linear,
    // so two estimates of a legitimately repeated root can end up further
    // apart than that -- see cdx::roots()'s documentation of the same
    // effect.
    std::vector<Cplx> distinct_critical_points(Cplx a, double rel_tol = 1e-4) const;

    // Continuation-seeded variant of distinct_critical_points, for a caller
    // (Renderer::render_parameter_rational/render_parameter_basin) sweeping
    // a grid of nearby parameter values who already has, from the
    // IMMEDIATELY PRECEDING pixel's own call, an excellent per-root
    // starting guess: warm-starting the ordinary (non-pole, non-infinity)
    // critical points via a few Newton corrector steps (cdx::newton_refine)
    // on deriv_numerator_poly(a) is far cheaper than the full cold Aberth-
    // Ehrlich solve critical_points()/distinct_critical_points() always
    // perform. `predictor` should be empty for the first pixel of a sweep
    // -- an empty predictor can never match this call's own expected
    // ordinary-root count, so it naturally forces a cold solve with no
    // separate flag needed -- and otherwise exactly the PRECEDING call's
    // own `ordinary_raw` output.
    //
    // FALLS BACK to an ordinary cold solve -- IDENTICAL results, in every
    // way, to distinct_critical_points(a, rel_tol) -- whenever continuation
    // would be unsound: predictor's size doesn't match the ordinary
    // critical-point count at `a` (the parameter moved across a degree-
    // changing degeneracy), any single Newton correction fails to converge
    // within budget (the nearby root moved further than a few steps can
    // catch, or the local structure changed), a refined point ends up
    // coincident with a pole (a genuine nearby structural change, not
    // numerical noise -- see critical_points()'s own pole-exclusion
    // comment), or two refined points collapse onto each other (a lost
    // root). CORRECTNESS NEVER DEPENDS ON CONTINUATION SUCCEEDING: on ANY
    // of these triggers, `distinct` is EXACTLY what distinct_critical_
    // points(a, rel_tol) would have returned, at the (already-being-paid-
    // anyway, since continuation was attempted first) extra cost of the
    // failed attempt.
    //
    // Pole-copy and infinity contributions (critical_points()'s other two
    // sources, both closed-form/deterministic, not root-found at all) are
    // always recomputed fresh here, exactly as the cold path does -- there
    // is no root-finding cost there to save, so nothing about them is ever
    // continued.
    ContinuedCriticalPoints distinct_critical_points_continued(
        Cplx a, const std::vector<Cplx>& predictor, double rel_tol = 1e-4) const;

    // True iff critical_points(a) is provably the SAME SET for every `a` --
    // e.g. any z^n + a shape: the `+ a` term has exponent 0, so it never
    // enters the derivative and the only critical point is (and stays) 0.
    // Structural, not numeric: checks that no term feeding the derivative,
    // and no pole (a critical point in its own right, per the class comment
    // above), can vary with the parameter at all. This is what lets
    // Renderer::render_parameter call critical_points() ONCE per render for
    // a Custom map like this, instead of once per pixel -- normally the
    // dominant cost of rendering a Custom map's parameter plane, since each
    // call clears denominators and runs a full Aberth-Ehrlich root-find
    // (cdx::roots) from scratch.
    //
    // Conservative in one narrow, deliberate way: a poly term with
    // exponent == 0 is exempt even though ITS OWN value can depend on `a`
    // (that's exactly the "+a" case above) -- such a term never reaches the
    // derivative, but it DOES still feed into the numerator polynomial used
    // for pole-order and infinity-multiplicity bookkeeping elsewhere in
    // critical_points(). In the ordinary case that changes nothing (that
    // numerator's effective DEGREE is set by its highest-exponent term,
    // whose coefficient this function already requires to be
    // parameter-independent); it could only give a stale-but-wrong answer
    // if `a` grew so large that the exempted term's magnitude swamped that
    // leading term's fixed coefficient by more than roots.cpp's
    // kTrimRelTol (1e-12) -- i.e. |a| astronomically outside any parameter
    // range this project's viewports actually explore.
    bool critical_points_constant() const;

    // The free/pre-fixed split of this map's critical points -- see
    // FreeCriticalStructure's own comment for the rule. Parameter-
    // independent by construction; compute it once per render.
    FreeCriticalStructure free_critical_structure() const;

    // `distinct` (a distinct_critical_points(a)-style list AT `a`) with every
    // structurally pre-fixed entry removed, order otherwise preserved. A
    // finite entry within 1e-4 (relative, distinct_critical_points' own
    // dedup radius) of an excluded location counts as that location.
    std::vector<Cplx> free_critical_points(Cplx a, const std::vector<Cplx>& distinct,
                                           const FreeCriticalStructure& s) const;

    // The labelling convention above (see CriticalLabelBase): the base is
    // computed once per render, then critical_point_labels(a, base) gives the
    // free critical points at `a` IN LABEL ORDER. The one-argument form
    // builds the base itself (cold; for one-off callers such as the Facts
    // panel). continue_critical_labels is step 3 from an arbitrary already-
    // labelled start (from_a, from_pts) -- the warm-start primitive; it
    // returns false if the continuation failed.
    CriticalLabelBase critical_label_base() const;
    DerivNumeratorInA compile_deriv_numerator_in_a() const;
    // polynomial_coefficients(a) as polynomials in a (see PolyCoeffsInA);
    // valid == false when the map is not a polynomial or the fit fails.
    PolyCoeffsInA compile_polynomial_coefficients_in_a() const;
    std::vector<Cplx> critical_point_labels(Cplx a, const CriticalLabelBase& base) const;
    std::vector<Cplx> critical_point_labels(Cplx a) const;
    bool continue_critical_labels(const CriticalLabelBase& base, Cplx from_a,
                                  const std::vector<Cplx>& from_pts, Cplx to_a,
                                  std::vector<Cplx>& out) const;
    // Step 4 / the fallback: everything free at `a`, tracked points first
    // (given), then the closed-form ones, each group in (Re, Im) order when
    // `sort_tracked` is set.
    std::vector<Cplx> assemble_labels(Cplx a, const CriticalLabelBase& base,
                                      std::vector<Cplx> tracked, bool sort_tracked) const;

    // f's own coefficients at `a`, ASCENDING (index k multiplies z^k), when
    // is_polynomial_structurally(); returns false (and leaves `out`
    // untouched) otherwise. Trailing exact zeros are trimmed, so
    // out.back() is the leading coefficient a_d.
    bool polynomial_coefficients(Cplx a, std::vector<Cplx>& out) const;

    // All fixed points R(z) == z, found algebraically (root the polynomial
    // N(z) - z*D(z) after clearing denominators; NOT limited to attracting
    // ones -- for that, and for genuine higher-period cycles, see
    // cdx::find_attractors). Includes infinity, with its multiplier via the
    // w=1/z chart, whenever R(infinity) == infinity (i.e. the numerator
    // degree exceeds the denominator degree after clearing denominators):
    // multiplier 0 when the numerator/denominator degree gap is >= 2
    // (infinity is then also a critical point, so this matches
    // critical_points()'s infinity rule), or the reciprocal leading-
    // coefficient ratio when the gap is exactly 1 (an ordinary, non-critical
    // fixed point at infinity -- e.g. Newton's method's escape direction).
    //
    // No particular ordering guarantee.
    std::vector<FixedPoint> fixed_points(Cplx a) const;

    // Continuation-seeded variant of fixed_points, exactly mirroring
    // distinct_critical_points_continued's own contract above (predictor
    // = the preceding pixel's own `raw`; falls back to a cold fixed_
    // points(a) -- identical results -- on a count mismatch, a Newton
    // correction that fails to converge, a refined point coincident with
    // a pole, or two refined points collapsing together). The infinity
    // fixed point (closed-form/deterministic, never root-found) is always
    // recomputed fresh, same reasoning as the critical-point case.
    ContinuedFixedPoints fixed_points_continued(Cplx a, const std::vector<Cplx>& predictor) const;

    // Human-readable formula, e.g. "z^3 + a/z^3".
    std::string to_formula() const;

    // --- serialization ------------------------------------------------------
    // Simple line-oriented text, so saved families are diffable and
    // hand-editable rather than opaque.
    std::string serialize() const;
    static bool deserialize(const std::string& text, RationalMap& out,
                            std::string& error);

    // --- presets ------------------------------------------------------------
    static RationalMap mandelbrot();          // z^2 + a
    static RationalMap multibrot(int n);      // z^n + a
    static RationalMap mcmullen(int n);       // z^n + a/z^n
    static RationalMap newton_cubic();        // Newton map of z^3 - 1

private:
    // True iff an EXISTING enabled pole (a PoleTerm, or a negative-exponent
    // PolyTerm meaning the origin) sits at `location` -- see add_pole's own
    // comment. Skips any candidate with location_is_param set, for the
    // same reason add_pole itself does.
    bool pole_location_conflict(Cplx location) const;

    // Shared by from_canonical() and deserialize() (which builds `out`
    // incrementally as it reads lines, rather than through the public
    // factory) -- sets pq_/pq_param_/pq_dP_/pq_dQ_ from an already-
    // validated (<=1 parameter) CanonicalRational.
    void set_pq_backing(CanonicalRational cr);

    // The two polynomials critical_points()/fixed_points() (and Stage 2's
    // continuation counterparts) actually root -- see each one's own
    // comment in rational.cpp for the derivation. Factored out purely so
    // continuation can build the SAME polynomial a cold solve would,
    // without a second, divergence-prone copy of this construction.
    std::vector<Cplx> deriv_numerator_poly(Cplx a) const;   // P'Q-PQ' / cleared derivative numerator
    std::vector<Cplx> fixed_point_poly(Cplx a) const;       // P-zQ / N(z)-zD(z)

    // Appends the closed-form/deterministic sources (pole-copy multiplicity,
    // infinity multiplicity) to an already root-found (and, for critical
    // points, already pole-excluded) candidate list -- shared by each
    // method's cold path and Stage 2's continuation counterpart, so both
    // build the assembled result identically regardless of how the
    // root-found portion was obtained.
    std::vector<Cplx> assemble_critical_points(Cplx a, const std::vector<Cplx>& ordinary) const;
    std::vector<FixedPoint> assemble_fixed_points(Cplx a,
                                                  const std::vector<Cplx>& finite_candidates) const;

    std::string           name_ = "untitled";
    std::string           notes_;
    std::vector<PolyTerm> poly_;
    std::vector<PoleTerm> pole_;

    // P/Q-backed storage (Stage 2) -- null for a term-based map. Shared,
    // not owned uniquely: copying a RationalMap (e.g. Map::custom's own
    // std::make_shared<const RationalMap> wrap, or FamilyLibrary::add's
    // push_back) copies this pointer, not the (immutable once parsed) P/Q
    // structure it points to -- cheap, and safe precisely because nothing
    // ever mutates a CanonicalRational after from_canonical builds it.
    std::shared_ptr<const CanonicalRational> pq_;
    // cr.parameters[0] if from_canonical's `cr` had one; empty otherwise.
    // Cached here rather than re-read from pq_->parameters on every call,
    // since every P/Q method needs it.
    std::string pq_param_;
    // d/dz of pq_->P and pq_->Q, computed ONCE (set_pq_backing) rather than
    // rebuilt from scratch on every deriv()/critical_points() call -- a
    // purely symbolic (ParamExpr-tree) computation each time otherwise,
    // for information that never changes after construction.
    PolyZ pq_dP_, pq_dQ_;
    // pole_locations(a) when Q does not reference the active parameter at
    // all (so the poles are the same for every `a`): solved once in
    // set_pq_backing and returned as-is. Valid iff pq_poles_constant_.
    bool pq_poles_constant_ = false;
    std::vector<Cplx> pq_constant_poles_;

    // Authored-form preservation (Stage 3) -- see from_expression's own
    // comment. Defaulted from cr's own source/parameters by set_pq_backing
    // (so a map built via the simpler from_canonical still has these
    // populated consistently: pq_original_source_ == pq_->source,
    // pq_active_param_ == pq_param_, pq_fixed_params_ empty), then
    // OVERWRITTEN by from_expression with the true pre-substitution
    // values once set_pq_backing returns.
    std::string pq_original_source_;
    std::string pq_active_param_;
    std::map<std::string, Cplx> pq_fixed_params_;
};

// -----------------------------------------------------------------------------
// A named collection of saved families, for the sandbox library.
// -----------------------------------------------------------------------------
class FamilyLibrary {
public:
    void add(const RationalMap& m);
    bool remove(const std::string& name);
    const RationalMap* find(const std::string& name) const;
    std::vector<std::string> names() const;
    std::size_t size() const { return maps_.size(); }

    // Whole-library round trip, for a single sandbox save file.
    std::string serialize() const;
    static bool deserialize(const std::string& text, FamilyLibrary& out,
                            std::string& error);

    // Populates with the classical families.
    static FamilyLibrary with_defaults();

private:
    std::vector<RationalMap> maps_;
};

}  // namespace cdx
