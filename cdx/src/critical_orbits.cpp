// critical_orbits.cpp -- see cdx/critical_orbits.hpp for the method.
#include "cdx/critical_orbits.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cdx {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// |z| at or past this is "at infinity" for stepping purposes: CompiledMap::step
// maps a pole to 1e300, and Horner overflows a little beyond it.
constexpr double kAtInfinity2 = 1e290;
// Closure is not tested within this chordal distance of infinity: two points
// that near infinity are chordally near each other whatever they are (see
// analysis.cpp's near_infinity_chordally).
constexpr double kNearInfinity = 1e-3;
// |lambda| below this is treated as superattracting (the log-log form).
constexpr double kSuperattracting = 1e-12;
// After refinement the orbit must actually be near the cycle it found.
constexpr double kNearCycle = 1e-2;
// z_ref is reset at n = 1, 2, 4, ..., kMaxRefAge, then every kMaxRefAge
// steps. Pure doubling lets z_ref fall up to n/2 steps behind, and a slowly
// attracting orbit is then still far from where it was that long ago: it
// never closes within budget (measured: ~0.5% of the Mandelbrot interior,
// near every component boundary, left undetermined). Capping the age keeps
// every window able to see a period up to kMaxRefAge (the reconciled path
// saw up to 64).
constexpr int kMaxRefAge = 128;

bool finite(Cplx z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

// Chordal distance; infinity (or anything past kAtInfinity2) is the north pole.
double chordal(Cplx z, Cplx w) {
    const double nz = std::norm(z), nw = std::norm(w);
    const bool zi = !finite(z) || !(nz < kAtInfinity2), wi = !finite(w) || !(nw < kAtInfinity2);
    if (zi && wi) return 0.0;
    if (zi) return 2.0 / std::sqrt(1.0 + nw);
    if (wi) return 2.0 / std::sqrt(1.0 + nz);
    return 2.0 * std::abs(z - w) / std::sqrt((1.0 + nz) * (1.0 + nw));
}

// Squared chordal distance for two finite points -- the per-step closure
// test, no square roots.
inline double chordal2_finite(double zr, double zi, double wr, double wi) {
    const double dr = zr - wr, di = zi - wi;
    return 4.0 * (dr * dr + di * di) / ((1.0 + zr * zr + zi * zi) * (1.0 + wr * wr + wi * wi));
}

// One step on the sphere: an orbit that reaches infinity (a pole, or past
// double range) continues from f(inf) when infinity is not fixed.
// Returns false when the orbit is AT a fixed infinity.
inline bool sphere_step(const CompiledMap& f, const InfinityStatus& inf, double& zr, double& zi) {
    f.step(zr, zi);
    const double m2 = zr * zr + zi * zi;
    if (m2 < kAtInfinity2) return true;   // also false for NaN, handled below
    if (inf.fixed) return false;
    zr = inf.image.real();
    zi = inf.image.imag();
    return true;
}

// Minimal period of a refined cycle: the smallest q | p with pts[q] back
// on pts[0] (and the rest repeating), to `tol` chordally.
int minimal_period(const std::vector<Cplx>& pts, double tol) {
    const int p = static_cast<int>(pts.size());
    for (int q = 1; q < p; ++q) {
        if (p % q != 0) continue;
        bool repeats = true;
        for (int i = q; i < p && repeats; ++i) {
            if (chordal(pts[static_cast<std::size_t>(i)], pts[static_cast<std::size_t>(i % q)]) >= tol)
                repeats = false;
        }
        if (repeats) return q;
    }
    return p;
}

// The last kRing iterates (z_{n-kRing+1} .. z_n), so the smooth value can be
// taken where the orbit actually crossed `tol`, not extrapolated from wherever
// the cycle happened to be detected.
constexpr int kRing = kMaxRefAge + 1;
struct OrbitRing {
    double re[kRing], im[kRing];
    int last_n = 0;                       // step of the newest entry
    void put(int n, double zr, double zi) { re[n % kRing] = zr; im[n % kRing] = zi; last_n = n; }
    int oldest() const { return std::max(1, last_n - kRing + 1); }
    Cplx at(int n) const { return {re[n % kRing], im[n % kRing]}; }
};

// Candidate closure at step n (z_n = zn) against the reference z_ref with
// period candidate p. Fills `out` and returns true for an attracting cycle.
bool try_cycle(const CompiledMap& f, Cplx ref, int p, Cplx zn, int n, const OrbitRing& ring,
               const CriticalOrbitOptions& opts, CriticalOrbitOutcome& out) {
    // 1. Minimal divisor q of p with f^q(z_ref) still within closure_eps.
    std::vector<Cplx> it;
    it.reserve(static_cast<std::size_t>(p) + 1);
    it.push_back(ref);
    for (int j = 1; j <= p; ++j) {
        const Cplx w = f.eval_c(it.back());
        if (!finite(w)) return false;
        it.push_back(w);
    }
    int q = p;
    for (int d = 1; d < p; ++d) {
        if (p % d == 0 && chordal(it[static_cast<std::size_t>(d)], ref) < opts.closure_eps) { q = d; break; }
    }

    // 2. Newton on g(x) = f^q(x) - x from z_ref.
    Cplx x = ref;
    bool converged = false;
    for (int k = 0; k < opts.newton_iterations; ++k) {
        Cplx y = x, D(1.0, 0.0);
        for (int i = 0; i < q; ++i) {
            D *= f.deriv_c(y);
            y = f.eval_c(y);
        }
        if (!finite(y) || !finite(D)) return false;
        const Cplx gp = D - Cplx(1.0, 0.0);
        if (std::abs(gp) < 1e-12) return false;   // parabolic: f^q - id has a double root
        const Cplx dx = (y - x) / gp;
        x -= dx;
        if (!finite(x)) return false;
        if (std::abs(dx) <= 1e-14 * std::max(1.0, std::abs(x))) { converged = true; break; }
    }
    (void)converged;   // the closure check below is the real acceptance test

    // 3. The refined cycle, reduced again to its minimal period.
    std::vector<Cplx> pts;
    pts.reserve(static_cast<std::size_t>(q));
    Cplx y = x;
    for (int i = 0; i < q; ++i) {
        pts.push_back(y);
        y = f.eval_c(y);
        if (!finite(y)) return false;
    }
    if (chordal(y, x) > 1e-8) return false;      // Newton did not land on a periodic point
    const int qm = minimal_period(pts, 1e-7);
    pts.resize(static_cast<std::size_t>(qm));

    // 4. Attracting?
    Cplx lambda(1.0, 0.0);
    for (const Cplx& c : pts) lambda *= f.deriv_c(c);
    if (!finite(lambda) || !(std::abs(lambda) < 1.0 - opts.attracting_margin)) return false;

    // 5. The orbit must actually be near the cycle it found.
    double d = kInf;
    for (const Cplx& c : pts) d = std::min(d, chordal(zn, c));
    if (!(d < kNearCycle)) return false;

    // 6. Smooth value (see the header), taken at m = the first retained step
    // with d_m < tol. The orbit crossed tol between steps m-1 and m, so the
    // fractional correction is clamped to [-1, 0]: extrapolating from deep
    // inside (at a component's centre the critical point IS on the cycle and
    // d is ~1e-300) otherwise put the crossing several steps before the orbit
    // even started converging. If even the oldest retained step is inside
    // tol, the crossing is older than the ring and only the >= -m clamp
    // applies; if no step is inside yet (closure_eps is looser than tol), the
    // value is extrapolated forward from n. Origin: exactly at tol at step m
    // gives m - 1 -- the reconciled path's interpolation convention, so the
    // colouring of existing renders keeps its meaning.
    const double abs_l = std::abs(lambda);
    auto dist = [&](Cplx z) {
        double dd = kInf;
        for (const Cplx& c : pts) dd = std::min(dd, chordal(z, c));
        return dd;
    };
    auto correction = [&](double dm) {   // <= 0 when dm <= tol
        if (!(dm > 0.0)) return -1.0;
        double c = 0.0;
        if (abs_l >= kSuperattracting) {
            c = static_cast<double>(qm) * std::log(opts.tol / dm) / std::log(abs_l);
        } else if (dm < 1.0 && opts.tol < 1.0) {
            c = static_cast<double>(qm) * std::log(std::log(opts.tol) / std::log(dm)) / std::log(2.0);
        }
        return std::isfinite(c) ? c : 0.0;
    };
    double nu;
    if (d >= opts.tol) {
        nu = static_cast<double>(n - 1) + std::max(0.0, correction(d));
    } else {
        int m = n;
        for (int k = ring.oldest(); k <= n; ++k) {
            if (dist(ring.at(k)) < opts.tol) { m = k; break; }
        }
        const double c = correction(dist(ring.at(m)));
        nu = static_cast<double>(m - 1) +
             (m > ring.oldest() ? std::clamp(c, -1.0, 0.0) : std::max(c, -static_cast<double>(m)));
    }
    if (!std::isfinite(nu)) nu = static_cast<double>(n - 1);

    out.kind = CriticalOrbitOutcome::Kind::Cycle;
    out.period = qm;
    out.cycle = std::move(pts);
    out.multiplier = lambda;
    out.nu = nu;
    out.steps = n;
    return true;
}

}  // namespace

InfinityStatus infinity_status(const CompiledMap& f) {
    InfinityStatus s;
    f.infinity_shape(s.fixed, s.local_degree, s.multiplier, s.image);
    return s;
}

Cplx compiled_eval(const CompiledMap& f, Cplx z) { return f.eval_c(z); }
Cplx compiled_deriv(const CompiledMap& f, Cplx z) { return f.deriv_c(z); }

CriticalOrbitOutcome track_critical_orbit(const CompiledMap& f, const InfinityStatus& inf, Cplx c0,
                                          const CriticalOrbitOptions& opts) {
    CriticalOrbitOutcome out;
    const bool poly = opts.poly_R2 > 0.0;
    const bool inf_attracting = !poly && inf.attracting();
    const double eps2 = opts.closure_eps * opts.closure_eps;
    const double near_inf_m2 = 4.0 / (kNearInfinity * kNearInfinity);   // chordal(z, inf) < kNearInfinity
    const double tol_inf_m2 = 4.0 / (opts.tol * opts.tol);              // chordal(z, inf) < tol (approx.)

    auto to_infinity = [&](int n, double m2) {
        out.kind = CriticalOrbitOutcome::Kind::Infinity;
        out.period = 1;
        out.cycle = {Cplx(kInf, 0.0)};
        out.multiplier = inf.multiplier;
        out.steps = n;
        // Same origin as the cycle value (see try_cycle, step 6): exactly at
        // R_inf (chordal distance tol) at step n gives n - 1.
        const double base = static_cast<double>(n - 1);
        double nu = base;
        if (!poly && std::isfinite(m2) && m2 > 1.0) {
            const double log_mod = 0.5 * std::log(m2);
            // n is the first step inside R_inf, so the crossing was in
            // (n-1, n]: the correction is clamped to [-1, 0] (see try_cycle).
            double c = 0.0;
            if (inf.local_degree >= 2) {
                const double log_R = std::log(2.0 / opts.tol);
                c = -std::log(log_mod / log_R) / std::log(static_cast<double>(inf.local_degree));
            } else {
                const double d = 2.0 / std::sqrt(1.0 + m2);
                const double al = std::abs(inf.multiplier);
                if (al > 0.0 && al < 1.0) c = std::log(opts.tol / d) / std::log(al);
            }
            nu = std::isfinite(c) ? base + std::clamp(c, -1.0, 0.0) : base;
        }
        out.nu = nu;
        return out;
    };

    double zr = c0.real(), zi = c0.imag();
    if (!finite(c0) || !(zr * zr + zi * zi < kAtInfinity2)) {
        // A critical point at infinity (free only when infinity is not
        // structurally pre-fixed): start from f(inf), or stay at a fixed inf.
        if (inf.fixed) {
            if (poly || inf_attracting) return to_infinity(0, kInf);
            out.steps = 0;
            return out;   // at a fixed, non-attracting infinity: undetermined
        }
        zr = inf.image.real();
        zi = inf.image.imag();
    }

    OrbitRing ring;
    double rr = zr, ri = zi;   // z_ref
    int n_ref = 0;
    int next_reset = 1;
    bool tried = false;        // a candidate at this z_ref already failed

    for (int n = 1; n <= opts.budget; ++n) {
        if (!sphere_step(f, inf, zr, zi)) {   // at a fixed infinity
            if (poly || inf_attracting) return to_infinity(n, kInf);
            out.steps = n;
            return out;
        }
        const double m2 = zr * zr + zi * zi;
        if (!(m2 == m2)) { out.steps = n; return out; }   // NaN: undetermined, never guessed
        ring.put(n, zr, zi);

        if (poly) {
            if (m2 > opts.poly_R2) return to_infinity(n, m2);
        } else if (inf_attracting && m2 > tol_inf_m2) {
            return to_infinity(n, m2);
        }

        if (!tried && m2 < near_inf_m2 && rr * rr + ri * ri < near_inf_m2 &&
            chordal2_finite(zr, zi, rr, ri) < eps2) {
            if (try_cycle(f, Cplx(rr, ri), n - n_ref, Cplx(zr, zi), n, ring, opts, out)) return out;
            tried = true;
        }
        if (n == next_reset) {
            rr = zr; ri = zi;
            n_ref = n;
            next_reset = n < kMaxRefAge ? 2 * n : n + kMaxRefAge;
            tried = false;
        }
    }
    out.steps = opts.budget;
    return out;   // out of budget: undetermined
}

bool same_attractor(const CriticalOrbitOutcome& a, const CriticalOrbitOutcome& b, double tol) {
    if (!a.resolved() || !b.resolved()) return false;
    if (a.kind != b.kind || a.period != b.period) return false;
    if (a.kind == CriticalOrbitOutcome::Kind::Infinity) return true;
    if (a.cycle.empty() || b.cycle.empty()) return false;
    for (const Cplx& p : a.cycle) {
        if (chordal(p, b.cycle.front()) < tol) return true;
    }
    return false;
}

}  // namespace cdx
