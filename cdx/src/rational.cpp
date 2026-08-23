// =============================================================================
// cdx/rational.cpp -- term-based rational map builder.
// =============================================================================
#include "cdx/rational.hpp"

#include "cdx/roots.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cdx {

namespace {

// -----------------------------------------------------------------------------
// Polynomial-algebra helpers for critical_points(). Plain ascending-order
// coefficient vectors (same convention as cdx::Polynomial), independent of
// RationalMap's term types so they work equally on R's own terms or on a
// transformed term list (e.g. the derivative's).
// -----------------------------------------------------------------------------

std::vector<Cplx> poly_mul(const std::vector<Cplx>& a, const std::vector<Cplx>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<Cplx> out(a.size() + b.size() - 1, Cplx(0.0, 0.0));
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            out[i + j] += a[i] * b[j];
    return out;
}

std::vector<Cplx> poly_add(const std::vector<Cplx>& a, const std::vector<Cplx>& b) {
    std::vector<Cplx> out(std::max(a.size(), b.size()), Cplx(0.0, 0.0));
    for (std::size_t i = 0; i < a.size(); ++i) out[i] += a[i];
    for (std::size_t i = 0; i < b.size(); ++i) out[i] += b[i];
    return out;
}

std::vector<Cplx> poly_negate(std::vector<Cplx> a) {
    for (Cplx& v : a) v = -v;
    return a;
}

// Multiply by (z - p), growing the degree by one.
std::vector<Cplx> mul_linear(const std::vector<Cplx>& c, Cplx p) {
    std::vector<Cplx> out(c.size() + 1, Cplx(0.0, 0.0));
    for (std::size_t k = 0; k < c.size(); ++k) {
        out[k + 1] += c[k];
        out[k]     -= p * c[k];
    }
    return out;
}

std::vector<Cplx> linear_power(Cplx p, int m) {
    std::vector<Cplx> c = {Cplx(1.0, 0.0)};
    for (int i = 0; i < m; ++i) c = mul_linear(c, p);
    return c;
}

// Multiply by z^k: prepend k zero coefficients.
std::vector<Cplx> poly_shift(const std::vector<Cplx>& a, int k) {
    std::vector<Cplx> out(a.size() + static_cast<std::size_t>(k), Cplx(0.0, 0.0));
    for (std::size_t i = 0; i < a.size(); ++i) out[i + static_cast<std::size_t>(k)] = a[i];
    return out;
}

// A term c*z^e (already evaluated at the bound parameter, already filtered
// for `enabled`) contributing to a polynomial sum.
struct Monomial {
    Cplx coeff;
    int  exponent;   // may be negative
};

// A pole term s/(z-p)^m (already evaluated, already filtered), contributing
// to the same sum.
struct PoleFactor {
    Cplx location;
    int  order;
    Cplx strength;
};

struct ClearedFraction {
    std::vector<Cplx> numerator;
    std::vector<Cplx> denominator;
};

// N(z)/D(z) of SUM(monomials) + SUM(poles), where D(z) = z^origin_order *
// prod_j (z - poles[j].location)^order and origin_order absorbs any
// negative monomial exponents. This is a valid (not necessarily reduced)
// common denominator: if two pole factors share a location, D just carries
// that factor twice, which is still algebraically correct -- any spurious
// extra root that introduces at that location gets filtered out by
// critical_points()'s pole-exclusion pass regardless of the order computed
// there, and any redundant common factor affects N and D equally, so it
// cancels out of degree DIFFERENCES (which is all critical_points() needs
// D for) even without actually reducing the fraction.
ClearedFraction clear_denominators(const std::vector<Monomial>& monomials,
                                   const std::vector<PoleFactor>& poles) {
    int min_neg = 0;
    for (const auto& m : monomials)
        if (m.exponent < min_neg) min_neg = m.exponent;
    const int origin_order = -min_neg;

    std::vector<Cplx> pole_product = {Cplx(1.0, 0.0)};
    for (const auto& f : poles)
        pole_product = poly_mul(pole_product, linear_power(f.location, f.order));

    int max_shifted = origin_order;
    for (const auto& m : monomials)
        max_shifted = std::max(max_shifted, m.exponent + origin_order);
    std::vector<Cplx> shifted_monomials(static_cast<std::size_t>(max_shifted) + 1,
                                        Cplx(0.0, 0.0));
    for (const auto& m : monomials) {
        const int idx = m.exponent + origin_order;   // always >= 0 by construction
        shifted_monomials[static_cast<std::size_t>(idx)] += m.coeff;
    }
    std::vector<Cplx> numerator = poly_mul(shifted_monomials, pole_product);

    for (std::size_t j = 0; j < poles.size(); ++j) {
        std::vector<Cplx> excl = {Cplx(1.0, 0.0)};
        for (std::size_t l = 0; l < poles.size(); ++l) {
            if (l == j) continue;
            excl = poly_mul(excl, linear_power(poles[l].location, poles[l].order));
        }
        excl = poly_shift(excl, origin_order);
        for (Cplx& v : excl) v *= poles[j].strength;
        numerator = poly_add(numerator, excl);
    }

    std::vector<Cplx> denominator = poly_shift(pole_product, origin_order);
    return {numerator, denominator};
}

// R's own numerator/denominator (not the derivative's, not any other
// transform): the extraction critical_points(), fixed_points() and
// pole_orders() all need as their starting point.
ClearedFraction own_fraction(const std::vector<PolyTerm>& poly, const std::vector<PoleTerm>& pole,
                             Cplx a) {
    std::vector<Monomial> monomials;
    for (const auto& t : poly) {
        if (!t.enabled) continue;
        monomials.push_back({t.effective_coeff(a), t.exponent});
    }
    std::vector<PoleFactor> pole_factors;
    for (const auto& t : pole) {
        if (!t.enabled) continue;
        pole_factors.push_back({t.effective_location(a), t.order, t.effective_strength(a)});
    }
    return clear_denominators(monomials, pole_factors);
}

// Divides c(z) by (z - p): returns the quotient (degree one less than c)
// and the remainder c(p). Ascending-order synthetic division.
struct Division {
    std::vector<Cplx> quotient;
    Cplx remainder;
};

Division divide_by_linear(const std::vector<Cplx>& c, Cplx p) {
    const int n = static_cast<int>(c.size()) - 1;
    if (n < 0) return {{}, Cplx(0.0, 0.0)};
    if (n == 0) return {{}, c[0]};

    std::vector<Cplx> q(static_cast<std::size_t>(n), Cplx(0.0, 0.0));
    q[static_cast<std::size_t>(n - 1)] = c[static_cast<std::size_t>(n)];
    for (int k = n - 1; k >= 1; --k) {
        q[static_cast<std::size_t>(k - 1)] =
            c[static_cast<std::size_t>(k)] + p * q[static_cast<std::size_t>(k)];
    }
    const Cplx r = c[0] + p * q[0];
    return {q, r};
}

// Multiplicity of p as a root of c: repeated synthetic division by (z - p)
// until the remainder is no longer negligible relative to the (deflating)
// polynomial's own scale. Used to find the TRUE local order of a pole --
// summing each contributing term's nominal order would overcount when
// multiple terms share a location (their dominant orders do not simply
// add), and would not notice a numerator that happens to also vanish there
// (a removable singularity, or -- concretely -- a McMullen-style pole whose
// strength is exactly zero for the parameter in play).
constexpr double kVanishingRelTol = 1e-9;

int vanishing_order(std::vector<Cplx> c, Cplx p) {
    int order = 0;
    while (c.size() > 1) {
        double maxabs = 0.0;
        for (const Cplx& v : c) maxabs = std::max(maxabs, std::abs(v));
        if (maxabs == 0.0) break;   // identically zero: nothing further to extract

        const Division d = divide_by_linear(c, p);
        if (std::abs(d.remainder) > kVanishingRelTol * maxabs) break;
        ++order;
        c = d.quotient;
    }
    return order;
}

// Roots within this fraction of a pole's own magnitude are treated as
// artifacts of clearing denominators rather than genuine critical points.
// Matches RenderSettings::tol's default -- both are "how close counts as
// the same point" thresholds at the same natural scale for this project.
constexpr double kPoleExclRelTol = 1e-6;

Cplx ipow(Cplx b, int n) {
    if (n == 0) return Cplx(1.0, 0.0);
    const bool inv = n < 0;
    if (inv) n = -n;
    Cplx acc(1.0, 0.0);
    while (n) {
        if (n & 1) acc *= b;
        b *= b;
        n >>= 1;
    }
    return inv ? Cplx(1.0, 0.0) / acc : acc;
}

std::string fmt(Cplx v) {
    std::ostringstream o;
    o.precision(10);
    if (v.imag() == 0.0) { o << v.real(); }
    else {
        o << '(' << v.real() << (v.imag() < 0 ? "" : "+") << v.imag() << "i)";
    }
    return o.str();
}

// -----------------------------------------------------------------------------
// P/Q-backed helpers (Stage 2). A P/Q map's coefficients are ParamExpr
// trees (rational_parser.hpp) in AT MOST one parameter; resolving each one
// against a single (name -> value) binding turns a PolyZ into a plain
// ascending-order std::vector<Cplx> -- at that point it is EXACTLY the same
// shape clear_denominators' own numerator/denominator already are, so
// every polynomial op below this point (poly_mul, poly_add, poly_negate,
// poly_shift, vanishing_order, mul_linear, roots(), effective_degree())
// applies unchanged. A P/Q map needs no "clear denominators" step at all --
// P and Q already ARE the numerator and denominator, not a term list to
// combine into one.
// -----------------------------------------------------------------------------
std::map<std::string, Cplx> pq_binding(const std::string& param_name, Cplx a) {
    if (param_name.empty()) return {};
    return {{param_name, a}};
}

std::vector<Cplx> eval_coeffs(const PolyZ& p, const std::map<std::string, Cplx>& params) {
    std::vector<Cplx> out;
    out.reserve(p.coeffs.size());
    for (const auto& c : p.coeffs) out.push_back(c->eval(params));
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
Cplx PolyTerm::effective_coeff(Cplx a) const {
    return param_power == 0 ? coeff : coeff * ipow(a, param_power);
}

Cplx PoleTerm::effective_strength(Cplx a) const {
    return param_power == 0 ? strength : strength * ipow(a, param_power);
}

Cplx PoleTerm::effective_location(Cplx a) const {
    return location_is_param ? a : location;
}

// -----------------------------------------------------------------------------
// P/Q-backed construction (Stage 2 + Stage 3).
// -----------------------------------------------------------------------------
void RationalMap::set_pq_backing(CanonicalRational cr) {
    if (cr.parameters.size() > 1) {
        throw std::invalid_argument(
            "RationalMap::from_canonical: '" + cr.source + "' has " +
            std::to_string(cr.parameters.size()) + " parameters (" +
            [&] { std::string s; for (std::size_t i = 0; i < cr.parameters.size(); ++i) {
                      if (i) s += ", "; s += cr.parameters[i]; } return s; }() +
            "); this engine is single-active-parameter -- select which one plays that "
            "role and substitute the rest as constants before calling from_canonical "
            "(see its own doc comment)");
    }
    pq_param_ = cr.parameters.empty() ? std::string() : cr.parameters[0];
    pq_dP_ = poly_deriv(cr.P);
    pq_dQ_ = poly_deriv(cr.Q);
    // Defaults for the authored-form fields (Stage 3): correct as-is for a
    // map built directly through from_canonical (source and active
    // parameter both come straight from `cr`, nothing was substituted);
    // from_expression overwrites all three with the true pre-substitution
    // values once this call returns.
    pq_original_source_ = cr.source;
    pq_active_param_ = pq_param_;
    pq_fixed_params_.clear();
    pq_ = std::make_shared<const CanonicalRational>(std::move(cr));
}

RationalMap RationalMap::from_canonical(CanonicalRational cr, std::string name) {
    RationalMap m(std::move(name));
    m.set_pq_backing(std::move(cr));
    return m;
}

RationalMap RationalMap::from_expression(const std::string& source, const std::string& active_param,
                                         const std::map<std::string, Cplx>& fixed_values,
                                         std::string name) {
    CanonicalRational cr;
    std::string error;
    if (!parse_rational(source, cr, error)) {
        throw std::invalid_argument("RationalMap::from_expression: " + error);
    }

    std::string active = active_param;
    if (cr.parameters.empty()) {
        if (!active.empty()) {
            throw std::invalid_argument(
                "RationalMap::from_expression: '" + source + "' has no parameters, but "
                "active_param='" + active + "' was given");
        }
    } else if (cr.parameters.size() == 1) {
        if (active.empty()) {
            active = cr.parameters[0];   // exactly one parameter -> auto-active
        } else if (active != cr.parameters[0]) {
            throw std::invalid_argument(
                "RationalMap::from_expression: active_param='" + active + "' doesn't match "
                "'" + source + "'s own single parameter '" + cr.parameters[0] + "'");
        }
    } else {
        if (active.empty()) {
            throw std::invalid_argument(
                "RationalMap::from_expression: '" + source + "' has " +
                std::to_string(cr.parameters.size()) + " parameters; active_param must "
                "name which one is active (there is no default when the choice is "
                "genuinely ambiguous)");
        }
        if (std::find(cr.parameters.begin(), cr.parameters.end(), active) == cr.parameters.end()) {
            throw std::invalid_argument(
                "RationalMap::from_expression: active_param='" + active + "' is not one of "
                "'" + source + "'s own parameters");
        }
    }

    PolyZ reduced_P = cr.P, reduced_Q = cr.Q;
    std::map<std::string, Cplx> used_fixed;
    for (const auto& pname : cr.parameters) {
        if (pname == active) continue;
        auto it = fixed_values.find(pname);
        if (it == fixed_values.end()) {
            throw std::invalid_argument(
                "RationalMap::from_expression: '" + source + "' references parameter '" +
                pname + "', which is not the active parameter ('" + active + "') and has "
                "no fixed value in fixed_values");
        }
        reduced_P = substitute_param(reduced_P, pname, it->second);
        reduced_Q = substitute_param(reduced_Q, pname, it->second);
        used_fixed[pname] = it->second;
    }

    CanonicalRational reduced;
    reduced.P = std::move(reduced_P);
    reduced.Q = std::move(reduced_Q);
    reduced.parameters = active.empty() ? std::vector<std::string>{}
                                        : std::vector<std::string>{active};
    reduced.source = source;   // placeholder -- pq_original_source_ below is authoritative

    RationalMap m(std::move(name));
    m.set_pq_backing(std::move(reduced));
    m.pq_original_source_ = source;
    m.pq_active_param_ = active;
    m.pq_fixed_params_ = std::move(used_fixed);
    return m;
}

const std::string& RationalMap::pq_source() const {
    static const std::string kEmpty;
    return pq_ ? pq_original_source_ : kEmpty;
}

const std::string& RationalMap::pq_active_param() const {
    static const std::string kEmpty;
    return pq_ ? pq_active_param_ : kEmpty;
}

const std::map<std::string, Cplx>& RationalMap::pq_fixed_params() const {
    static const std::map<std::string, Cplx> kEmpty;
    return pq_ ? pq_fixed_params_ : kEmpty;
}

namespace {
// "(z - location)", or "(z + |location|)" when location is a negative
// real -- matches to_formula()'s own term-based sign convention (see its
// pole-factor printing below) so a P/Q map's formula reads the same way a
// term-based one's does.
std::string fmt_linear_factor(Cplx location) {
    if (location.imag() == 0.0 && location.real() < 0) return "(z+" + fmt(-location) + ")";
    return "(z-" + fmt(location) + ")";
}
}  // namespace

void RationalMap::add_pole_at(Cplx location) {
    if (!pq_) {
        throw std::logic_error(
            "RationalMap::add_pole_at: only valid for a P/Q-backed map (is_pq_backed()) -- "
            "use add_pole for a term-based one");
    }
    // Preserved explicitly: set_pq_backing's own defaults would otherwise
    // wipe the substitution record for a map built via from_expression --
    // multiplying Q by a plain (z-location) factor doesn't touch any
    // parameter, fixed or active, so there is nothing here that should
    // change about it.
    const std::map<std::string, Cplx> saved_fixed = pq_fixed_params_;

    CanonicalRational cr = *pq_;
    PolyZ factor;
    factor.coeffs = {param_const(-location), param_const(Cplx(1.0, 0.0))};   // z - location
    cr.Q = poly_mul(cr.Q, factor);
    cr.source = "(" + pq_original_source_ + ") / " + fmt_linear_factor(location);

    set_pq_backing(std::move(cr));
    pq_fixed_params_ = saved_fixed;
}

void RationalMap::add_zero_at(Cplx location) {
    if (!pq_) {
        throw std::logic_error(
            "RationalMap::add_zero_at: only valid for a P/Q-backed map (is_pq_backed()) -- "
            "use add_poly for a term-based one");
    }
    const std::map<std::string, Cplx> saved_fixed = pq_fixed_params_;

    CanonicalRational cr = *pq_;
    PolyZ factor;
    factor.coeffs = {param_const(-location), param_const(Cplx(1.0, 0.0))};   // z - location
    cr.P = poly_mul(cr.P, factor);
    cr.source = "(" + pq_original_source_ + ") * " + fmt_linear_factor(location);

    set_pq_backing(std::move(cr));
    pq_fixed_params_ = saved_fixed;
}

// -----------------------------------------------------------------------------
// 1e-12 absolute, matching pole_locations()'s own "same point" tolerance
// (rational.cpp's pole_locations, push_unique) -- this check exists to
// catch the SAME location told twice, not to reject a genuinely distinct
// nearby pole, so it stays exactly as tight as that existing convention.
constexpr double kPoleLocationSameTol = 1e-12;

bool RationalMap::pole_location_conflict(Cplx location) const {
    for (const auto& t : pole_) {
        if (!t.enabled || t.location_is_param) continue;
        if (std::abs(t.location - location) < kPoleLocationSameTol) return true;
    }
    if (std::abs(location) < kPoleLocationSameTol) {
        for (const auto& t : poly_) {
            if (t.enabled && t.exponent < 0) return true;
        }
    }
    return false;
}

std::size_t RationalMap::add_poly(Cplx coeff, int exponent, int param_power,
                                  std::string label) {
    if (exponent < 0) {
        throw std::invalid_argument(
            "poly terms cannot have a negative exponent (that represents a pole at the "
            "origin) -- use add_pole instead, so a pole has exactly one representation");
    }
    PolyTerm t;
    t.coeff = coeff;
    t.exponent = exponent;
    t.param_power = param_power;
    t.label = std::move(label);
    poly_.push_back(t);
    return poly_.size() - 1;
}

std::size_t RationalMap::add_pole(Cplx location, Cplx strength, int order,
                                  int param_power, std::string label) {
    if (pole_location_conflict(location)) {
        throw std::invalid_argument(
            "a pole already exists at " + fmt(location) + "; edit it instead of adding "
            "a duplicate");
    }
    PoleTerm t;
    t.location = location;
    t.strength = strength;
    t.order = order < 1 ? 1 : order;
    t.param_power = param_power;
    t.label = std::move(label);
    pole_.push_back(t);
    return pole_.size() - 1;
}

void RationalMap::remove_poly(std::size_t i) {
    if (i < poly_.size()) poly_.erase(poly_.begin() + static_cast<long>(i));
}
void RationalMap::remove_pole(std::size_t i) {
    if (i < pole_.size()) pole_.erase(pole_.begin() + static_cast<long>(i));
}
void RationalMap::clear() { poly_.clear(); pole_.clear(); }

// -----------------------------------------------------------------------------
Cplx RationalMap::eval(Cplx z, Cplx a) const {
    if (pq_) {
        const auto params = pq_binding(pq_param_, a);
        const Cplx num = pq_->P.eval(z, params);
        const Cplx den = pq_->Q.eval(z, params);
        if (den == Cplx(0.0, 0.0)) return Cplx(1e300, 0.0);
        return num / den;
    }

    Cplx sum(0.0, 0.0);

    for (const auto& t : poly_) {
        if (!t.enabled) continue;
        sum += t.effective_coeff(a) * ipow(z, t.exponent);
    }
    for (const auto& t : pole_) {
        if (!t.enabled) continue;
        const Cplx d = z - t.effective_location(a);
        // A point exactly at a pole maps to infinity; signal it with a huge
        // value so the caller's escape test fires rather than producing NaN.
        if (d == Cplx(0.0, 0.0)) return Cplx(1e300, 0.0);
        sum += t.effective_strength(a) / ipow(d, t.order);
    }
    return sum;
}

Cplx RationalMap::deriv(Cplx z, Cplx a) const {
    if (pq_) {
        // (P'Q - PQ') / Q^2, closed form -- pq_dP_/pq_dQ_ are precomputed
        // once (set_pq_backing), not rebuilt symbolically on every call.
        const auto params = pq_binding(pq_param_, a);
        const Cplx P = pq_->P.eval(z, params);
        const Cplx Q = pq_->Q.eval(z, params);
        if (Q == Cplx(0.0, 0.0)) return Cplx(1e300, 0.0);
        const Cplx Pp = pq_dP_.eval(z, params);
        const Cplx Qp = pq_dQ_.eval(z, params);
        return (Pp * Q - P * Qp) / (Q * Q);
    }

    Cplx sum(0.0, 0.0);

    // d/dz [c z^e] = c e z^(e-1)
    for (const auto& t : poly_) {
        if (!t.enabled || t.exponent == 0) continue;
        sum += t.effective_coeff(a) * static_cast<double>(t.exponent) *
               ipow(z, t.exponent - 1);
    }
    // d/dz [s (z-p)^-m] = -m s (z-p)^-(m+1)
    for (const auto& t : pole_) {
        if (!t.enabled) continue;
        const Cplx d = z - t.effective_location(a);
        if (d == Cplx(0.0, 0.0)) return Cplx(1e300, 0.0);
        sum -= static_cast<double>(t.order) * t.effective_strength(a) /
               ipow(d, t.order + 1);
    }
    return sum;
}

// -----------------------------------------------------------------------------
CompiledMap RationalMap::compile(Cplx a) const {
    if (pq_) {
        CompiledMap c;
        c.is_pq_ = true;
        const auto params = pq_binding(pq_param_, a);
        c.pr_.reserve(pq_->P.coeffs.size());
        c.pi_.reserve(pq_->P.coeffs.size());
        for (const auto& coeff : pq_->P.coeffs) {
            const Cplx v = coeff->eval(params);
            c.pr_.push_back(v.real());
            c.pi_.push_back(v.imag());
        }
        c.qr_.reserve(pq_->Q.coeffs.size());
        c.qi_.reserve(pq_->Q.coeffs.size());
        for (const auto& coeff : pq_->Q.coeffs) {
            const Cplx v = coeff->eval(params);
            c.qr_.push_back(v.real());
            c.qi_.push_back(v.imag());
        }
        return c;
    }

    CompiledMap c;
    c.poly_.reserve(poly_.size());
    for (const auto& t : poly_) {
        if (!t.enabled) continue;
        const Cplx coeff = t.effective_coeff(a);
        c.poly_.push_back({coeff.real(), coeff.imag(), t.exponent});
    }
    c.pole_.reserve(pole_.size());
    for (const auto& t : pole_) {
        if (!t.enabled) continue;
        const Cplx loc = t.effective_location(a);
        const Cplx str = t.effective_strength(a);
        c.pole_.push_back({loc.real(), loc.imag(), str.real(), str.imag(), t.order});
    }
    return c;
}

// -----------------------------------------------------------------------------
int RationalMap::degree(Cplx a) const {
    if (pq_) {
        const auto params = pq_binding(pq_param_, a);
        const int p_deg = effective_degree(Polynomial{eval_coeffs(pq_->P, params)});
        const int q_deg = effective_degree(Polynomial{eval_coeffs(pq_->Q, params)});
        return std::max(p_deg, q_deg);
    }

    // Clearing denominators: the denominator is the product of (z-p_j)^m_j
    // together with z^{|min negative exponent|}. The numerator degree is the
    // largest positive exponent plus the full denominator degree.
    int den_deg = 0;
    int min_neg = 0;
    for (const auto& t : poly_) {
        if (!t.enabled) continue;
        if (t.exponent < min_neg) min_neg = t.exponent;
    }
    den_deg += -min_neg;
    for (const auto& t : pole_) {
        if (t.enabled) den_deg += t.order;
    }

    int max_pos = 0;
    for (const auto& t : poly_) {
        if (t.enabled && t.exponent > max_pos) max_pos = t.exponent;
    }

    const int num_deg = max_pos + den_deg;
    (void)a;
    return std::max(num_deg, den_deg);
}

bool RationalMap::is_polynomial_structurally() const {
    if (pq_) {
        // Q is a nonzero constant (degree 0, i.e. at most one coefficient
        // -- PolyZ trims structurally-zero trailing entries, so an empty
        // or single-entry Q means no z-dependence at all) and P's
        // STRUCTURAL degree (coefficient count - 1) is >= 2. Both are
        // parameter-independent: which coefficient SLOTS exist doesn't
        // depend on evaluating any of them.
        if (pq_->Q.coeffs.size() > 1) return false;
        return pq_->P.coeffs.size() >= 3;
    }
    for (const auto& t : pole_) {
        if (t.enabled) return false;
    }
    for (const auto& t : poly_) {
        if (t.enabled && t.exponent < 0) return false;   // implies a pole at the origin
    }
    return degree(Cplx(0.0, 0.0)) >= 2;
}

std::vector<Cplx> RationalMap::pole_locations(Cplx a) const {
    if (pq_) {
        const auto params = pq_binding(pq_param_, a);
        const std::vector<Cplx> raw = roots(Polynomial{eval_coeffs(pq_->Q, params)});
        // roots() returns WITH multiplicity (a double pole comes back as
        // two numerically-close, not necessarily bit-identical, estimates
        // -- same Aberth-Ehrlich degradation distinct_critical_points()'s
        // own comment documents), but pole_locations() is DEDUPLICATED,
        // one entry per distinct location, with pole_orders() reporting
        // each one's own order (see its own contract). A RELATIVE
        // tolerance, matching distinct_critical_points()'s own default
        // (not the term-based path's 1e-12 absolute below, which exists
        // to catch an EXACT declared location told twice, not a
        // root-finder's own estimate spread).
        std::vector<Cplx> out;
        for (Cplx p : raw) {
            bool dup = false;
            for (Cplx q : out) {
                const double scale = std::max(1.0, std::max(std::abs(p), std::abs(q)));
                if (std::abs(p - q) < 1e-4 * scale) { dup = true; break; }
            }
            if (!dup) out.push_back(p);
        }
        return out;
    }

    std::vector<Cplx> out;
    auto push_unique = [&out](Cplx p) {
        for (const auto& q : out) if (std::abs(p - q) < 1e-12) return;
        out.push_back(p);
    };
    for (const auto& t : pole_) {
        if (t.enabled) push_unique(t.effective_location(a));
    }
    for (const auto& t : poly_) {
        if (t.enabled && t.exponent < 0) { push_unique(Cplx(0.0, 0.0)); break; }
    }
    return out;
}

std::vector<int> RationalMap::pole_orders(Cplx a) const {
    if (pq_) {
        const auto params = pq_binding(pq_param_, a);
        const std::vector<Cplx> P = eval_coeffs(pq_->P, params);
        const std::vector<Cplx> Q = eval_coeffs(pq_->Q, params);
        std::vector<int> out;
        for (Cplx p : pole_locations(a)) {
            const int d_order = vanishing_order(Q, p);
            const int n_order = vanishing_order(P, p);
            out.push_back(std::max(0, d_order - n_order));
        }
        return out;
    }

    const ClearedFraction frac = own_fraction(poly_, pole_, a);
    std::vector<int> out;
    for (Cplx p : pole_locations(a)) {
        const int d_order = vanishing_order(frac.denominator, p);
        const int n_order = vanishing_order(frac.numerator, p);
        out.push_back(std::max(0, d_order - n_order));
    }
    return out;
}

std::vector<Cplx> RationalMap::critical_points(Cplx a) const {
    if (pq_) {
        const auto params = pq_binding(pq_param_, a);
        const std::vector<Cplx> P = eval_coeffs(pq_->P, params);
        const std::vector<Cplx> Q = eval_coeffs(pq_->Q, params);
        const std::vector<Cplx> dP = eval_coeffs(pq_dP_, params);
        const std::vector<Cplx> dQ = eval_coeffs(pq_dQ_, params);
        const std::vector<Cplx> poles = pole_locations(a);
        std::vector<Cplx> out;

        // ---- ordinary critical points: zeros of P'Q - PQ', away from poles --
        // No "clear denominators" step needed -- P and Q already ARE the
        // numerator/denominator, not a term list to combine.
        {
            const std::vector<Cplx> deriv_num =
                poly_add(poly_mul(dP, Q), poly_negate(poly_mul(P, dQ)));
            const std::vector<Cplx> candidates = roots(Polynomial{deriv_num});
            for (Cplx z : candidates) {
                bool at_pole = false;
                for (Cplx p : poles) {
                    const double scale = std::max(1.0, std::abs(p));
                    if (std::abs(z - p) < kPoleExclRelTol * scale) { at_pole = true; break; }
                }
                if (!at_pole) out.push_back(z);
            }
        }

        // ---- each pole of TRUE local order m contributes multiplicity m-1 --
        for (Cplx p : poles) {
            const int d_order = vanishing_order(Q, p);
            const int n_order = vanishing_order(P, p);
            const int true_order = d_order - n_order;
            for (int k = 1; k < true_order; ++k) out.push_back(p);
        }

        // ---- infinity contributes multiplicity |p-q|-1 when |p-q| >= 2 -----
        const int p_deg = effective_degree(Polynomial{P});
        const int q_deg = effective_degree(Polynomial{Q});
        if (p_deg >= 0 && q_deg >= 0) {
            const int diff = std::abs(p_deg - q_deg);
            if (diff >= 2) {
                const Cplx infinity(std::numeric_limits<double>::infinity(), 0.0);
                for (int k = 1; k < diff; ++k) out.push_back(infinity);
            }
        }

        return out;
    }

    const std::vector<Cplx> poles = pole_locations(a);
    std::vector<Cplx> out;

    // ---- ordinary critical points: zeros of the derivative, away from any pole ----
    // Transform the term lists into the derivative's: d/dz[c z^e] = c e
    // z^(e-1); d/dz[s (z-p)^-m] = -m s (z-p)^-(m+1). Same shape deriv() uses.
    {
        std::vector<Monomial> dpolys;
        for (const auto& t : poly_) {
            if (!t.enabled || t.exponent == 0) continue;
            dpolys.push_back({t.effective_coeff(a) * static_cast<double>(t.exponent),
                              t.exponent - 1});
        }
        std::vector<PoleFactor> dpoles;
        for (const auto& t : pole_) {
            if (!t.enabled) continue;
            dpoles.push_back({t.effective_location(a), t.order + 1,
                              -static_cast<double>(t.order) * t.effective_strength(a)});
        }
        const ClearedFraction dfrac = clear_denominators(dpolys, dpoles);
        const std::vector<Cplx> candidates = roots(Polynomial{dfrac.numerator});

        for (Cplx z : candidates) {
            bool at_pole = false;
            for (Cplx p : poles) {
                const double scale = std::max(1.0, std::abs(p));
                if (std::abs(z - p) < kPoleExclRelTol * scale) { at_pole = true; break; }
            }
            if (!at_pole) out.push_back(z);
        }
    }

    // R's own numerator/denominator (not the derivative's) drive both
    // remaining sources: a pole is a critical point in its own right, and
    // whether infinity is critical depends on their degrees.
    const ClearedFraction frac = own_fraction(poly_, pole_, a);

    // ---- each pole of TRUE local order m contributes multiplicity m-1 --------
    // TRUE order, not the nominal order(s) of whichever term(s) happen to be
    // at that location: found from the actual constructed N(z)/D(z), so it
    // is correct even when multiple terms share a location (their orders do
    // not simply add) or when a pole's strength happens to vanish at this
    // particular `a` (no pole there at all, for this parameter value).
    for (Cplx p : poles) {
        const int d_order = vanishing_order(frac.denominator, p);
        const int n_order = vanishing_order(frac.numerator, p);
        const int true_order = d_order - n_order;
        for (int k = 1; k < true_order; ++k) out.push_back(p);
    }

    // ---- infinity contributes multiplicity |p-q|-1 when |p-q| >= 2 -----------
    // p, q = numerator/denominator degree after clearing denominators; a
    // shared spurious factor (as clear_denominators can introduce) affects
    // both equally, so the DIFFERENCE is correct even without reducing.
    const int p_deg = effective_degree(Polynomial{frac.numerator});
    const int q_deg = effective_degree(Polynomial{frac.denominator});
    if (p_deg >= 0 && q_deg >= 0) {
        const int diff = std::abs(p_deg - q_deg);
        if (diff >= 2) {
            const Cplx infinity(std::numeric_limits<double>::infinity(), 0.0);
            for (int k = 1; k < diff; ++k) out.push_back(infinity);
        }
    }

    return out;
}

std::vector<Cplx> RationalMap::distinct_critical_points(Cplx a, double rel_tol) const {
    const std::vector<Cplx> all = critical_points(a);
    std::vector<Cplx> out;
    for (Cplx z : all) {
        const bool z_inf = std::isinf(z.real()) || std::isinf(z.imag());
        bool found = false;
        for (Cplx existing : out) {
            const bool existing_inf = std::isinf(existing.real()) || std::isinf(existing.imag());
            if (z_inf != existing_inf) continue;   // one finite, one infinite: distinct
            if (z_inf) { found = true; break; }     // both infinite: the same point
            const double scale = std::max(1.0, std::max(std::abs(z), std::abs(existing)));
            if (std::abs(z - existing) < rel_tol * scale) { found = true; break; }
        }
        if (!found) out.push_back(z);
    }
    return out;
}

bool RationalMap::critical_points_constant() const {
    if (pq_) {
        if (pq_param_.empty()) return true;   // no free parameter at all
        // Q must not reference the parameter anywhere: a pole is a
        // critical point in its own right, so its location/order moving
        // with `a` disqualifies it outright (mirrors the term-based
        // path's own pole check below).
        for (const auto& c : pq_->Q.coeffs) {
            if (references_param(c, pq_param_)) return false;
        }
        // P must not reference the parameter EXCEPT possibly its degree-0
        // (constant) coefficient -- mirrors the term-based path's own
        // exponent==0 exemption exactly (see the header's own comment on
        // why that's safe: such a term never reaches the derivative).
        for (std::size_t k = 1; k < pq_->P.coeffs.size(); ++k) {
            if (references_param(pq_->P.coeffs[k], pq_param_)) return false;
        }
        return true;
    }
    // Only terms that survive into the derivative (exponent != 0) can move
    // an ordinary critical point; a param-dependent exponent==0 term (e.g.
    // mandelbrot()'s "+ a") never reaches it. See the header comment for the
    // (deliberate, narrow) degenerate case this doesn't cover.
    for (const auto& t : poly_) {
        if (!t.enabled) continue;
        if (t.exponent != 0 && t.param_power != 0) return false;
    }
    // A pole is a critical point in its own right (critical_points()'s
    // second source), so its location moving with `a` -- or its TRUE order
    // changing because its strength does -- both disqualify it outright.
    for (const auto& t : pole_) {
        if (!t.enabled) continue;
        if (t.param_power != 0 || t.location_is_param) return false;
    }
    return true;
}

std::vector<FixedPoint> RationalMap::fixed_points(Cplx a) const {
    if (pq_) {
        const auto params = pq_binding(pq_param_, a);
        const std::vector<Cplx> P = eval_coeffs(pq_->P, params);
        const std::vector<Cplx> Q = eval_coeffs(pq_->Q, params);

        // R(z) == z  <=>  P(z) - z*Q(z) == 0
        const std::vector<Cplx> fixed_poly = poly_add(P, poly_negate(poly_shift(Q, 1)));
        const std::vector<Cplx> candidates = roots(Polynomial{fixed_poly});

        const std::vector<Cplx> poles = pole_locations(a);
        std::vector<FixedPoint> out;
        for (Cplx z : candidates) {
            bool at_pole = false;
            for (Cplx p : poles) {
                const double scale = std::max(1.0, std::abs(p));
                if (std::abs(z - p) < kPoleExclRelTol * scale) { at_pole = true; break; }
            }
            if (!at_pole) out.push_back({z, deriv(z, a)});
        }

        const int p_deg = effective_degree(Polynomial{P});
        const int q_deg = effective_degree(Polynomial{Q});
        if (p_deg > q_deg) {
            const int diff = p_deg - q_deg;
            const Cplx multiplier = diff >= 2
                ? Cplx(0.0, 0.0)
                : Q[static_cast<std::size_t>(q_deg)] / P[static_cast<std::size_t>(p_deg)];
            out.push_back({Cplx(std::numeric_limits<double>::infinity(), 0.0), multiplier});
        }
        return out;
    }

    const ClearedFraction frac = own_fraction(poly_, pole_, a);

    // R(z) == z  <=>  N(z) - z*D(z) == 0
    const std::vector<Cplx> fixed_poly =
        poly_add(frac.numerator, poly_negate(poly_shift(frac.denominator, 1)));
    const std::vector<Cplx> candidates = roots(Polynomial{fixed_poly});

    // Same reasoning as critical_points()'s pole exclusion: an unreduced
    // denominator (redundant same-location terms) can make a genuine pole
    // look, algebraically, like it solves N(z)-zD(z)=0 too. A true pole is
    // never actually a fixed point (R is not defined there), so filter it
    // out the same way.
    const std::vector<Cplx> poles = pole_locations(a);
    std::vector<FixedPoint> out;
    for (Cplx z : candidates) {
        bool at_pole = false;
        for (Cplx p : poles) {
            const double scale = std::max(1.0, std::abs(p));
            if (std::abs(z - p) < kPoleExclRelTol * scale) { at_pole = true; break; }
        }
        if (!at_pole) out.push_back({z, deriv(z, a)});
    }

    // Infinity is fixed iff R(infinity) == infinity, i.e. the numerator
    // degree p exceeds the denominator degree q after clearing
    // denominators. Its multiplier comes from the w=1/z chart: writing
    // R(z) ~ (N_lead/D_lead) z^(p-q) for large z, R~(w) = 1/R(1/w) ~
    // (D_lead/N_lead) w^(p-q), so R~'(0) is 0 when p-q >= 2 (matching
    // critical_points()'s infinity rule -- consistent, since a fixed point
    // with multiplier 0 is by definition also critical) and D_lead/N_lead
    // when p-q == 1 (an ordinary, non-critical fixed point at infinity).
    const int p_deg = effective_degree(Polynomial{frac.numerator});
    const int q_deg = effective_degree(Polynomial{frac.denominator});
    if (p_deg > q_deg) {
        const int diff = p_deg - q_deg;
        const Cplx multiplier = diff >= 2
            ? Cplx(0.0, 0.0)
            : frac.denominator[static_cast<std::size_t>(q_deg)] /
              frac.numerator[static_cast<std::size_t>(p_deg)];
        out.push_back({Cplx(std::numeric_limits<double>::infinity(), 0.0), multiplier});
    }

    return out;
}

// -----------------------------------------------------------------------------
std::string RationalMap::to_formula() const {
    // The user's own ORIGINAL authored text, verbatim -- not a
    // reconstruction, and not the reduced single-parameter form
    // from_expression actually computes with (see pq_source's own
    // comment). A later stage may want a normalized/pretty form instead;
    // nothing is more correct than what the user actually typed.
    if (pq_) return pq_original_source_;

    std::ostringstream o;
    bool first = true;
    auto sep = [&]() { if (!first) o << " + "; first = false; };

    for (const auto& t : poly_) {
        if (!t.enabled) continue;
        sep();
        const bool unit = (t.coeff == Cplx(1.0, 0.0));
        if (!unit || (t.param_power == 0 && t.exponent == 0)) o << fmt(t.coeff);
        if (t.param_power != 0) {
            if (!unit) o << "*";
            o << "a";
            if (t.param_power != 1) o << "^" << t.param_power;
        }
        if (t.exponent != 0) {
            if (!unit || t.param_power != 0) o << "*";
            o << "z";
            if (t.exponent != 1) o << "^" << t.exponent;
        }
    }
    for (const auto& t : pole_) {
        if (!t.enabled) continue;
        sep();
        const bool unit = (t.strength == Cplx(1.0, 0.0));
        if (!unit) o << fmt(t.strength);
        if (!unit || t.param_power != 0) { /* separator added below */ }
        if (t.param_power != 0) {
            o << "a";
            if (t.param_power != 1) o << "^" << t.param_power;
        }
        const Cplx L = t.effective_location({0.0, 0.0});
        const bool at_origin = !t.location_is_param && L == Cplx(0.0, 0.0);
        const bool has_lead = (!unit || t.param_power != 0);
        if (at_origin) {
            // pole at the origin prints as a/z^m rather than a*1/(z)^m
            o << (has_lead ? "/z" : "1/z");
            if (t.order != 1) o << "^" << t.order;
        } else {
            o << (has_lead ? "/(z" : "1/(z");
            if (t.location_is_param) o << "-a";
            else if (L.imag() == 0.0 && L.real() < 0) o << "+" << fmt(-L);
            else o << "-" << fmt(L);
            o << ")";
            if (t.order != 1) o << "^" << t.order;
        }
    }
    if (first) o << "0";
    return o.str();
}

// -----------------------------------------------------------------------------
// Serialization: one directive per line. Deliberately plain text so saved
// families are diffable in git and editable by hand.
// -----------------------------------------------------------------------------
std::string RationalMap::serialize() const {
    std::ostringstream o;
    o.precision(17);
    o << "map " << name_ << "\n";
    if (!notes_.empty()) o << "notes " << notes_ << "\n";
    if (pq_) {
        // "pq2": full authored-form fidelity (Stage 3) -- the ORIGINAL,
        // possibly multi-parameter source text, which parsed parameter is
        // active ("-" for none), and one "pqfixed" line per substituted
        // parameter. deserialize() reconstructs via from_expression,
        // exactly reproducing the substitution this instance itself was
        // built with -- not just a map that happens to compute the same
        // way right now.
        o << "pq2 " << (pq_active_param_.empty() ? "-" : pq_active_param_) << ' '
          << pq_original_source_ << "\n";
        for (const auto& [pname, pval] : pq_fixed_params_) {
            o << "pqfixed " << pname << ' ' << pval.real() << ' ' << pval.imag() << "\n";
        }
        o << "end\n";
        return o.str();
    }
    for (const auto& t : poly_) {
        o << "poly " << t.coeff.real() << ' ' << t.coeff.imag() << ' '
          << t.exponent << ' ' << t.param_power << ' ' << (t.enabled ? 1 : 0)
          << '\n';
    }
    for (const auto& t : pole_) {
        o << "pole " << t.location.real() << ' ' << t.location.imag() << ' '
          << t.strength.real() << ' ' << t.strength.imag() << ' '
          << t.order << ' ' << t.param_power << ' ' << (t.enabled ? 1 : 0)
          << ' ' << (t.location_is_param ? 1 : 0) << '\n';
    }
    o << "end\n";
    return o.str();
}

bool RationalMap::deserialize(const std::string& text, RationalMap& out,
                              std::string& error) {
    std::istringstream in(text);
    std::string line;
    out = RationalMap();
    bool saw_map = false;
    // "pq2"/"pqfixed" (Stage 3): buffered until "end", since from_expression
    // needs the full picture (source, active parameter, EVERY fixed value)
    // at once, not incrementally the way poly/pole lines build up out.poly_/
    // pole_ directly.
    bool has_pq2 = false;
    std::string pq2_active_token, pq2_source;
    std::map<std::string, Cplx> pq2_fixed;

    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind;
        if (!(ls >> kind)) continue;

        if (kind == "map") {
            std::string n;
            std::getline(ls, n);
            if (!n.empty() && n[0] == ' ') n.erase(0, 1);
            out.set_name(n.empty() ? "untitled" : n);
            saw_map = true;
        } else if (kind == "notes") {
            std::string n;
            std::getline(ls, n);
            if (!n.empty() && n[0] == ' ') n.erase(0, 1);
            out.set_notes(n);
        } else if (kind == "poly") {
            PolyTerm t; double re, im; int en;
            if (!(ls >> re >> im >> t.exponent >> t.param_power >> en)) {
                error = "malformed poly line: " + line; return false;
            }
            t.coeff = Cplx(re, im);
            t.enabled = (en != 0);
            out.poly_.push_back(t);
        } else if (kind == "pole") {
            PoleTerm t; double lr, li, sr, si; int en, lp;
            if (!(ls >> lr >> li >> sr >> si >> t.order >> t.param_power >> en >> lp)) {
                error = "malformed pole line: " + line; return false;
            }
            t.location = Cplx(lr, li);
            t.strength = Cplx(sr, si);
            t.enabled = (en != 0);
            t.location_is_param = (lp != 0);
            out.pole_.push_back(t);
        } else if (kind == "pq") {
            std::string src;
            std::getline(ls, src);
            if (!src.empty() && src[0] == ' ') src.erase(0, 1);
            CanonicalRational cr;
            std::string parse_error;
            if (!parse_rational(src, cr, parse_error)) {
                error = "malformed pq line ('" + src + "'): " + parse_error;
                return false;
            }
            // set_pq_backing itself throws std::invalid_argument for >1
            // parameter -- shouldn't happen for a line THIS class's own
            // serialize() wrote (from_canonical already enforced it before
            // the map existed to serialize), but a hand-edited save file
            // is exactly the kind of input this should report as a clean
            // error rather than let escape as an exception.
            try {
                out.set_pq_backing(std::move(cr));
            } catch (const std::invalid_argument& e) {
                error = e.what();
                return false;
            }
        } else if (kind == "pq2") {
            // Full authored-form fidelity (Stage 3) -- see serialize()'s
            // own comment for the format. "-" (a token, not the empty
            // string, since >> can't read an empty token) means no active
            // parameter.
            if (!(ls >> pq2_active_token)) {
                error = "malformed pq2 line (missing active-parameter token): " + line;
                return false;
            }
            std::string src;
            std::getline(ls, src);
            if (!src.empty() && src[0] == ' ') src.erase(0, 1);
            has_pq2 = true;
            pq2_source = src;
        } else if (kind == "pqfixed") {
            std::string pname;
            double re, im;
            if (!(ls >> pname >> re >> im)) {
                error = "malformed pqfixed line: " + line; return false;
            }
            pq2_fixed[pname] = Cplx(re, im);
        } else if (kind == "end") {
            break;
        }
    }
    if (!saw_map) { error = "no 'map' header found"; return false; }
    if (has_pq2) {
        const std::string active = pq2_active_token == "-" ? std::string() : pq2_active_token;
        // from_expression returns a FRESH RationalMap -- reassigning `out`
        // to it would silently drop the name_/notes_ already set above by
        // the "map"/"notes" lines, so carry them across explicitly.
        const std::string saved_name = out.name_;
        const std::string saved_notes = out.notes_;
        try {
            out = RationalMap::from_expression(pq2_source, active, pq2_fixed, saved_name);
        } catch (const std::invalid_argument& e) {
            error = e.what();
            return false;
        }
        out.notes_ = saved_notes;
    }
    error.clear();
    return true;
}

// -----------------------------------------------------------------------------
// Presets -- every classical family expressed in the term representation.
// -----------------------------------------------------------------------------
RationalMap RationalMap::mandelbrot() {
    RationalMap m("mandelbrot");
    m.set_notes("z^2 + a  -- the quadratic family");
    m.add_poly({1, 0}, 2, 0, "z^2");
    m.add_poly({1, 0}, 0, 1, "a");     // coefficient a, exponent 0
    return m;
}

RationalMap RationalMap::multibrot(int n) {
    RationalMap m("multibrot" + std::to_string(n));
    m.set_notes("z^" + std::to_string(n) + " + a");
    m.add_poly({1, 0}, n, 0);
    m.add_poly({1, 0}, 0, 1, "a");
    return m;
}

RationalMap RationalMap::mcmullen(int n) {
    RationalMap m("mcmullen" + std::to_string(n));
    m.set_notes("z^" + std::to_string(n) + " + a/z^" + std::to_string(n) +
                "  -- pole of order " + std::to_string(n) + " at the origin");
    m.add_poly({1, 0}, n, 0);
    m.add_pole({0, 0}, {1, 0}, n, 1, "a/z^n");
    return m;
}

RationalMap RationalMap::newton_cubic() {
    // N(z) = z - (z^3-1)/(3z^2) = (2/3) z + (1/3) z^-2. The second term is a
    // pole of order 2 at the origin -- routed through add_pole, not a
    // negative-exponent add_poly (which would now throw; see add_poly's own
    // comment), so this preset has exactly one representation for it.
    RationalMap m("newton3");
    m.set_notes("Newton map of z^3-1, simplified to (2/3)z + (1/3)z^-2");
    m.add_poly({2.0 / 3.0, 0}, 1, 0, "(2/3)z");
    m.add_pole({0.0, 0.0}, {1.0 / 3.0, 0.0}, 2, 0, "1/(3z^2)");
    return m;
}

// -----------------------------------------------------------------------------
// FamilyLibrary
// -----------------------------------------------------------------------------
void FamilyLibrary::add(const RationalMap& m) {
    for (auto& e : maps_) {
        if (e.name() == m.name()) { e = m; return; }   // replace by name
    }
    maps_.push_back(m);
}

bool FamilyLibrary::remove(const std::string& name) {
    for (auto it = maps_.begin(); it != maps_.end(); ++it) {
        if (it->name() == name) { maps_.erase(it); return true; }
    }
    return false;
}

const RationalMap* FamilyLibrary::find(const std::string& name) const {
    for (const auto& m : maps_) if (m.name() == name) return &m;
    return nullptr;
}

std::vector<std::string> FamilyLibrary::names() const {
    std::vector<std::string> out;
    out.reserve(maps_.size());
    for (const auto& m : maps_) out.push_back(m.name());
    return out;
}

std::string FamilyLibrary::serialize() const {
    std::ostringstream o;
    o << "cdx-library 1\n";
    for (const auto& m : maps_) o << m.serialize();
    return o.str();
}

bool FamilyLibrary::deserialize(const std::string& text, FamilyLibrary& out,
                                std::string& error) {
    out = FamilyLibrary();
    std::istringstream in(text);
    std::string line, block;
    bool in_block = false;

    while (std::getline(in, line)) {
        if (line.rfind("map ", 0) == 0) {
            if (in_block) {
                RationalMap m;
                if (!RationalMap::deserialize(block, m, error)) return false;
                out.add(m);
            }
            block = line + "\n";
            in_block = true;
        } else if (in_block) {
            block += line + "\n";
            if (line.rfind("end", 0) == 0) {
                RationalMap m;
                if (!RationalMap::deserialize(block, m, error)) return false;
                out.add(m);
                in_block = false;
                block.clear();
            }
        }
    }
    if (in_block && !block.empty()) {
        RationalMap m;
        if (!RationalMap::deserialize(block, m, error)) return false;
        out.add(m);
    }
    error.clear();
    return true;
}

FamilyLibrary FamilyLibrary::with_defaults() {
    FamilyLibrary lib;
    lib.add(RationalMap::mandelbrot());
    lib.add(RationalMap::multibrot(3));
    lib.add(RationalMap::multibrot(5));
    lib.add(RationalMap::mcmullen(2));
    lib.add(RationalMap::mcmullen(3));
    lib.add(RationalMap::newton_cubic());
    return lib;
}

}  // namespace cdx
