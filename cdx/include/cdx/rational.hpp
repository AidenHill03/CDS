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
    std::size_t add_poly(Cplx coeff, int exponent, int param_power = 0,
                         std::string label = {});
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

    // --- structure ----------------------------------------------------------
    // Degree as a rational map of the sphere: max(deg numerator, deg
    // denominator) after clearing denominators.
    int degree(Cplx a) const;

    // Locations of the finite poles (deduplicated), including any pole at the
    // origin implied by a negative polynomial exponent.
    std::vector<Cplx> pole_locations(Cplx a) const;

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
