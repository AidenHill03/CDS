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
// =============================================================================
#pragma once

#include <complex>
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
    inline void step(double& zr, double& zi) const;

private:
    friend class RationalMap;
    std::vector<PolyOp> poly_;
    std::vector<PoleOp> pole_;
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

    // --- structure ----------------------------------------------------------
    // Degree as a rational map of the sphere: max(deg numerator, deg
    // denominator) after clearing denominators.
    int degree(Cplx a) const;

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

    std::string           name_ = "untitled";
    std::string           notes_;
    std::vector<PolyTerm> poly_;
    std::vector<PoleTerm> pole_;
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
