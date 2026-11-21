// =============================================================================
// cdx/composed.cpp -- see cdx/composed.hpp for the contract.
//
// Arithmetic is hand-rolled real/imag doubles in the per-point paths (the
// project convention for hot loops -- see CLAUDE.md); std::complex is used
// only in deriv(), which is a cold path.
// =============================================================================
#include "cdx/composed.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace cdx {

namespace {

constexpr double kInf  = std::numeric_limits<double>::infinity();
constexpr double kNaN  = std::numeric_limits<double>::quiet_NaN();
constexpr double kHuge = 1e300;   // |z|^2 above this counts as infinity (chordal_distance's rule)

// A point on the sphere: (zr, zi) when !inf.
struct SpherePt {
    double zr = 0.0, zi = 0.0;
    bool   inf = false;
    bool   nan = false;
};

inline SpherePt from_plane(double zr, double zi) {
    SpherePt p;
    if (zr != zr || zi != zi) { p.nan = true; p.zr = p.zi = kNaN; return p; }
    if (std::isinf(zr) || std::isinf(zi) || zr * zr + zi * zi > kHuge) { p.inf = true; return p; }
    p.zr = zr; p.zi = zi;
    return p;
}

inline SpherePt infinity() { SpherePt p; p.inf = true; return p; }

// z^e for integer e (may be negative), hand-rolled. Overflow gives inf
// components, which the callers treat as infinity.
inline void cpow(double zr, double zi, int e, double& outr, double& outi) {
    const int ae = e < 0 ? -e : e;
    double pr = 1.0, pi = 0.0, br = zr, bi = zi;
    for (int n = ae; n; n >>= 1) {
        if (n & 1) {
            const double nr = pr * br - pi * bi, ni = pr * bi + pi * br;
            pr = nr; pi = ni;
        }
        if (n > 1) {
            const double nbr = br * br - bi * bi, nbi = 2.0 * br * bi;
            br = nbr; bi = nbi;
        }
    }
    if (e >= 0) { outr = pr; outi = pi; return; }
    const double den = pr * pr + pi * pi;
    outr = pr / den;
    outi = -pi / den;
}

// a / b, hand-rolled; caller guarantees b != 0.
inline void cdiv(double ar, double ai, double br, double bi, double& outr, double& outi) {
    // Scale to avoid premature overflow/underflow of br^2 + bi^2.
    const double s = std::max(std::fabs(br), std::fabs(bi));
    const double xr = br / s, xi = bi / s;
    const double den = xr * xr + xi * xi;
    outr = ((ar * xr + ai * xi) / den) / s;
    outi = ((ai * xr - ar * xi) / den) / s;
}

// ---- term-backed stage ------------------------------------------------------
SpherePt eval_terms(const ComposedMap::Compiled& c, const SpherePt& z) {
    if (z.nan) return z;
    if (z.inf) {
        if (c.inf_to_inf) return infinity();
        SpherePt out; out.zr = c.inf_r; out.zi = c.inf_i;
        return out;
    }
    const double zr = z.zr, zi = z.zi;
    double sr = 0.0, si = 0.0;
    for (std::size_t k = 0; k < c.ce.size(); ++k) {
        const int e = c.ce[k];
        if (e < 0 && zr == 0.0 && zi == 0.0) return infinity();
        double pr, pi;
        cpow(zr, zi, e, pr, pi);
        sr += c.cr[k] * pr - c.ci[k] * pi;
        si += c.cr[k] * pi + c.ci[k] * pr;
    }
    // Simple poles: no branch in the loop (an exact hit shows up as a
    // non-finite sum and is resolved below), so the loop stays tight.
    const std::size_t m = c.lr.size();
    const double* LR = c.lr.data(); const double* LI = c.li.data();
    const double* SR = c.sr.data(); const double* SI = c.si.data();
    double ar = 0.0, ai = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        const double dx = zr - LR[k], dy = zi - LI[k];
        const double den = dx * dx + dy * dy;
        const double ix = dx / den, iy = -dy / den;
        ar += ix * SR[k] - iy * SI[k];
        ai += ix * SI[k] + iy * SR[k];
    }
    sr += ar; si += ai;
    for (std::size_t k = 0; k < c.hord.size(); ++k) {
        const double dx = zr - c.hlr[k], dy = zi - c.hli[k];
        if (dx == 0.0 && dy == 0.0) return infinity();
        double ir, ii;
        cpow(dx, dy, -c.hord[k], ir, ii);
        sr += c.hsr[k] * ir - c.hsi[k] * ii;
        si += c.hsr[k] * ii + c.hsi[k] * ir;
    }
    if (std::isfinite(sr) && std::isfinite(si)) {
        SpherePt out; out.zr = sr; out.zi = si;
        return out;
    }
    // Non-finite: an exact pole hit (0/0 in the branch-free loop), a pole
    // so close that 1/(z-p) overflowed, or a genuine overflow of the sum --
    // all three are a point at (or numerically indistinguishable from)
    // infinity on the sphere.
    return infinity();
}

// ---- P/Q-backed stage -------------------------------------------------------
// Horner for sum_k c_k x^k (ascending), and for the reversed polynomial.
inline void horner(const std::vector<double>& cr, const std::vector<double>& ci,
                   double xr, double xi, double& outr, double& outi) {
    double ar = cr.back(), ai = ci.back();
    for (std::size_t k = cr.size() - 1; k-- > 0;) {
        const double nr = ar * xr - ai * xi + cr[k];
        const double ni = ar * xi + ai * xr + ci[k];
        ar = nr; ai = ni;
    }
    outr = ar; outi = ai;
}
inline void horner_reversed(const std::vector<double>& cr, const std::vector<double>& ci,
                            double wr, double wi, double& outr, double& outi) {
    double ar = cr.front(), ai = ci.front();
    for (std::size_t k = 1; k < cr.size(); ++k) {
        const double nr = ar * wr - ai * wi + cr[k];
        const double ni = ar * wi + ai * wr + ci[k];
        ar = nr; ai = ni;
    }
    outr = ar; outi = ai;
}

SpherePt eval_pq(const ComposedMap::Compiled& c, const SpherePt& z) {
    if (z.nan) return z;
    const int p = static_cast<int>(c.pr.size()) - 1;
    const int q = static_cast<int>(c.qr.size()) - 1;
    if (z.inf) {
        if (p > q) return infinity();
        SpherePt out;
        if (p < q) return out;   // 0
        cdiv(c.pr.back(), c.pi.back(), c.qr.back(), c.qi.back(), out.zr, out.zi);
        return out;
    }
    const double zr = z.zr, zi = z.zi;
    double nr, ni, dr, di;
    if (zr * zr + zi * zi <= 1.0) {
        horner(c.pr, c.pi, zr, zi, nr, ni);
        horner(c.qr, c.qi, zr, zi, dr, di);
        if (dr == 0.0 && di == 0.0) return infinity();
        SpherePt out;
        cdiv(nr, ni, dr, di, out.zr, out.zi);
        if (!std::isfinite(out.zr) || !std::isfinite(out.zi)) return infinity();
        return out;
    }
    // Reciprocal chart: P(z) = z^p P~(w), Q(z) = z^q Q~(w), w = 1/z.
    double wr, wi;
    cdiv(1.0, 0.0, zr, zi, wr, wi);
    horner_reversed(c.pr, c.pi, wr, wi, nr, ni);
    horner_reversed(c.qr, c.qi, wr, wi, dr, di);
    if (dr == 0.0 && di == 0.0) return infinity();
    double tr, ti;
    cdiv(nr, ni, dr, di, tr, ti);
    double er, ei;
    cpow(zr, zi, p - q, er, ei);
    SpherePt out;
    out.zr = er * tr - ei * ti;
    out.zi = er * ti + ei * tr;
    if (!std::isfinite(out.zr) || !std::isfinite(out.zi)) return infinity();
    return out;
}

inline SpherePt eval_stage(const ComposedMap::Compiled& c, SpherePt z) {
    for (int k = 0; k < c.repeat; ++k) {
        z = c.pq ? eval_pq(c, z) : eval_terms(c, z);
        if (z.nan) break;
    }
    return z;
}

// ---- derivatives (cold path, std::complex) ----------------------------------
using C = std::complex<double>;

C deriv_terms(const ComposedMap::Compiled& c, C z) {
    C sum(0.0, 0.0);
    for (std::size_t k = 0; k < c.ce.size(); ++k) {
        const int e = c.ce[k];
        if (e == 0) continue;
        double pr, pi;
        cpow(z.real(), z.imag(), e - 1, pr, pi);
        sum += C(c.cr[k], c.ci[k]) * static_cast<double>(e) * C(pr, pi);
    }
    for (std::size_t k = 0; k < c.lr.size(); ++k) {
        const C d = z - C(c.lr[k], c.li[k]);
        sum -= C(c.sr[k], c.si[k]) / (d * d);
    }
    for (std::size_t k = 0; k < c.hord.size(); ++k) {
        const C d = z - C(c.hlr[k], c.hli[k]);
        double ir, ii;
        cpow(d.real(), d.imag(), -(c.hord[k] + 1), ir, ii);
        sum -= static_cast<double>(c.hord[k]) * C(c.hsr[k], c.hsi[k]) * C(ir, ii);
    }
    return sum;
}

// Value and derivative of sum_k a_k x^k (ascending) by Horner.
inline void horner_d(const std::vector<double>& cr, const std::vector<double>& ci, C x, C& v, C& dv) {
    v = C(cr.back(), ci.back());
    dv = C(0.0, 0.0);
    for (std::size_t k = cr.size() - 1; k-- > 0;) {
        dv = dv * x + v;
        v = v * x + C(cr[k], ci[k]);
    }
}

C deriv_pq(const ComposedMap::Compiled& c, C z) {
    const int p = static_cast<int>(c.pr.size()) - 1;
    const int q = static_cast<int>(c.qr.size()) - 1;
    if (std::norm(z) <= 1.0) {
        C P, dP, Q, dQ;
        horner_d(c.pr, c.pi, z, P, dP);
        horner_d(c.qr, c.qi, z, Q, dQ);
        return (dP * Q - P * dQ) / (Q * Q);
    }
    // R(z) = z^e T(w), w = 1/z, T = P~/Q~ with P~(w) = sum P_k w^(p-k).
    // R'(z) = e z^(e-1) T - z^(e-2) T'(w).
    const C w = C(1.0, 0.0) / z;
    std::vector<double> rpr(c.pr.rbegin(), c.pr.rend()), rpi(c.pi.rbegin(), c.pi.rend());
    std::vector<double> rqr(c.qr.rbegin(), c.qr.rend()), rqi(c.qi.rbegin(), c.qi.rend());
    C Pt, dPt, Qt, dQt;
    horner_d(rpr, rpi, w, Pt, dPt);
    horner_d(rqr, rqi, w, Qt, dQt);
    const C T = Pt / Qt;
    const C dT = (dPt * Qt - Pt * dQt) / (Qt * Qt);
    const int e = p - q;
    double a1r, a1i, a2r, a2i;
    cpow(z.real(), z.imag(), e - 1, a1r, a1i);
    cpow(z.real(), z.imag(), e - 2, a2r, a2i);
    return static_cast<double>(e) * C(a1r, a1i) * T - C(a2r, a2i) * dT;
}

}  // namespace

// -----------------------------------------------------------------------------
ComposedMap::ComposedMap(std::vector<ComposedStage> stages) : stages_(std::move(stages)) {
    if (stages_.empty()) throw std::invalid_argument("ComposedMap needs at least one stage");
    degree_ = 1.0;
    log10_degree_ = 0.0;
    compiled_.reserve(stages_.size());
    for (std::size_t i = 0; i < stages_.size(); ++i) {
        const ComposedStage& st = stages_[i];
        if (st.repeat < 1) throw std::invalid_argument("ComposedMap: a stage's repeat must be >= 1");
        const int deg = st.map.degree(st.param);
        if (deg < 1) throw std::invalid_argument("ComposedMap: a stage has degree < 1 (a constant map)");
        degree_ *= std::pow(static_cast<double>(deg), st.repeat);
        log10_degree_ += st.repeat * std::log10(static_cast<double>(deg));

        Compiled c;
        c.repeat = st.repeat;
        std::vector<Cplx> P, Q;
        if (st.map.pq_coefficients(st.param, P, Q)) {
            c.pq = true;
            while (P.size() > 1 && P.back() == Cplx(0.0, 0.0)) P.pop_back();
            while (Q.size() > 1 && Q.back() == Cplx(0.0, 0.0)) Q.pop_back();
            if (P.empty()) P.push_back(Cplx(0.0, 0.0));
            if (Q.empty() || (Q.size() == 1 && Q[0] == Cplx(0.0, 0.0)))
                throw std::invalid_argument("ComposedMap: a P/Q stage has an identically zero denominator");
            for (const Cplx& v : P) { c.pr.push_back(v.real()); c.pi.push_back(v.imag()); }
            for (const Cplx& v : Q) { c.qr.push_back(v.real()); c.qi.push_back(v.imag()); }
        } else {
            Cplx at_inf(0.0, 0.0);
            for (const PolyTerm& t : st.map.poly_terms()) {
                if (!t.enabled) continue;
                const Cplx coeff = t.effective_coeff(st.param);
                if (coeff == Cplx(0.0, 0.0)) continue;
                c.cr.push_back(coeff.real());
                c.ci.push_back(coeff.imag());
                c.ce.push_back(t.exponent);
                if (t.exponent > 0) c.inf_to_inf = true;
                if (t.exponent == 0) at_inf += coeff;
            }
            for (const PoleTerm& t : st.map.pole_terms()) {
                if (!t.enabled) continue;
                const Cplx loc = t.effective_location(st.param);
                const Cplx str = t.effective_strength(st.param);
                if (str == Cplx(0.0, 0.0)) continue;
                if (t.order == 1) {
                    c.lr.push_back(loc.real()); c.li.push_back(loc.imag());
                    c.sr.push_back(str.real()); c.si.push_back(str.imag());
                } else {
                    c.hlr.push_back(loc.real()); c.hli.push_back(loc.imag());
                    c.hsr.push_back(str.real()); c.hsi.push_back(str.imag());
                    c.hord.push_back(t.order);
                }
            }
            c.inf_r = at_inf.real();
            c.inf_i = at_inf.imag();
        }
        compiled_.push_back(std::move(c));
    }
}

int ComposedMap::applications() const {
    int n = 0;
    for (const auto& st : stages_) n += st.repeat;
    return n;
}

Cplx ComposedMap::eval(Cplx z) const {
    SpherePt p = from_plane(z.real(), z.imag());
    for (const Compiled& c : compiled_) {
        p = eval_stage(c, p);
        if (p.nan) return Cplx(kNaN, kNaN);
    }
    return p.inf ? Cplx(kInf, 0.0) : Cplx(p.zr, p.zi);
}

void ComposedMap::step(double& zr, double& zi) const {
    SpherePt p = from_plane(zr, zi);
    for (const Compiled& c : compiled_) {
        p = eval_stage(c, p);
        if (p.nan) break;
    }
    if (p.nan) { zr = kNaN; zi = kNaN; return; }
    if (p.inf) { zr = kInf; zi = 0.0; return; }
    zr = p.zr;
    zi = p.zi;
}

Cplx ComposedMap::deriv(Cplx z) const {
    SpherePt p = from_plane(z.real(), z.imag());
    C d(1.0, 0.0);
    for (const Compiled& c : compiled_) {
        for (int k = 0; k < c.repeat; ++k) {
            if (p.nan || p.inf) return Cplx(kNaN, kNaN);
            const C here(p.zr, p.zi);
            d *= c.pq ? deriv_pq(c, here) : deriv_terms(c, here);
            p = c.pq ? eval_pq(c, p) : eval_terms(c, p);
        }
    }
    if (p.nan || p.inf) return Cplx(kNaN, kNaN);
    return d;
}

std::string ComposedMap::describe() const {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%zu stage%s, %d application%s, degree %.4g",
                  stages_.size(), stages_.size() == 1 ? "" : "s",
                  applications(), applications() == 1 ? "" : "s", degree_);
    return buf;
}

}  // namespace cdx
