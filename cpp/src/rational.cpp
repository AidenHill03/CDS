// =============================================================================
// cdx/rational.cpp -- term-based rational map builder.
// =============================================================================
#include "cdx/rational.hpp"

#include "cdx/roots.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

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

// Numerator N(z) of SUM(monomials) + SUM(poles), expressed over the common
// denominator D(z) = z^origin_order * prod_j (z - poles[j].location)^order,
// where origin_order absorbs any negative monomial exponents. This is a
// valid (not necessarily reduced) common denominator: if two pole factors
// share a location, D just carries that factor twice, which is still
// algebraically correct -- any spurious extra root that introduces at that
// location gets filtered out by critical_points()'s pole-exclusion pass
// regardless of the order computed there.
std::vector<Cplx> clear_denominators(const std::vector<Monomial>& monomials,
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
    return numerator;
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
std::size_t RationalMap::add_poly(Cplx coeff, int exponent, int param_power,
                                  std::string label) {
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
int RationalMap::degree(Cplx a) const {
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

std::vector<Cplx> RationalMap::pole_locations(Cplx a) const {
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

std::vector<Cplx> RationalMap::critical_points(Cplx a) const {
    // Transform the term lists into the derivative's: d/dz[c z^e] = c e
    // z^(e-1); d/dz[s (z-p)^-m] = -m s (z-p)^-(m+1). Same shape deriv() uses.
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

    const std::vector<Cplx> numerator = clear_denominators(dpolys, dpoles);
    const std::vector<Cplx> candidates = roots(Polynomial{numerator});

    const std::vector<Cplx> poles = pole_locations(a);
    std::vector<Cplx> out;
    for (Cplx z : candidates) {
        bool at_pole = false;
        for (Cplx p : poles) {
            const double scale = std::max(1.0, std::abs(p));
            if (std::abs(z - p) < kPoleExclRelTol * scale) { at_pole = true; break; }
        }
        if (!at_pole) out.push_back(z);
    }
    return out;
}

// -----------------------------------------------------------------------------
std::string RationalMap::to_formula() const {
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
        } else if (kind == "end") {
            break;
        }
    }
    if (!saw_map) { error = "no 'map' header found"; return false; }
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
    // N(z) = z - (z^3-1)/(3z^2) = (2/3) z + (1/3) z^-2
    RationalMap m("newton3");
    m.set_notes("Newton map of z^3-1, simplified to (2/3)z + (1/3)z^-2");
    m.add_poly({2.0 / 3.0, 0}, 1, 0, "(2/3)z");
    m.add_poly({1.0 / 3.0, 0}, -2, 0, "(1/3)z^-2");
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
