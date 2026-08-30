// =============================================================================
// cdx/renderer.cpp -- implementation of the Complex Dynamics core.
// =============================================================================
#include "cdx/renderer.hpp"

#include "cdx/analysis.hpp"   // polynomial_escape_certified -- see Map::escape_certified below

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace cdx {

namespace {
constexpr double kInf     = std::numeric_limits<double>::infinity();
constexpr double kHuge    = 1e300;
constexpr double kTinyDen = 1e-300;
inline bool is_bad(double v) { return !(v == v); }   // NaN test
}  // namespace

// -----------------------------------------------------------------------------
// Family names
// -----------------------------------------------------------------------------
std::string to_string(Family f) {
    switch (f) {
        case Family::Quadratic: return "quadratic";
        case Family::Cubic:     return "cubic";
        case Family::Quintic:   return "quintic";
        case Family::McMullen2: return "mcmullen2";
        case Family::McMullen3: return "mcmullen3";
        case Family::Newton3:   return "newton3";
        case Family::Custom:    return "custom";
    }
    return "unknown";
}

// Custom is deliberately not parseable here: it needs an actual RationalMap
// payload, which a bare name string cannot carry. Construct it via
// Map::custom() instead.
bool family_from_string(const std::string& s, Family& out) {
    if (s == "quadratic" || s == "mandelbrot") { out = Family::Quadratic; return true; }
    if (s == "cubic"     || s == "multibrot3") { out = Family::Cubic;     return true; }
    if (s == "quintic"   || s == "multibrot5") { out = Family::Quintic;   return true; }
    if (s == "mcmullen2")                      { out = Family::McMullen2; return true; }
    if (s == "mcmullen3")                      { out = Family::McMullen3; return true; }
    if (s == "newton3")                        { out = Family::Newton3;   return true; }
    return false;
}

namespace {
bool close_enough(Cplx a, Cplx b) {
    return std::abs(a - b) < 1e-9 * std::max(1.0, std::abs(b));
}
}  // namespace

// -----------------------------------------------------------------------------
// recognize_family -- see the doc comment in the header. Each shape below
// mirrors the exact terms its RationalMap:: preset factory constructs (see
// rational.cpp's mandelbrot/multibrot/mcmullen/newton_cubic), restricted to
// ENABLED terms only. A tolerance-based coefficient comparison (not exact
// ==) is deliberate: a map that round-tripped through
// RationalMap::serialize()/deserialize() loses some precision (fmt() prints
// 10 significant digits, not the ~17 a double needs to round-trip exactly),
// and a bit-exact requirement would silently stop recognizing a SAVED
// mandelbrot() after one save/load cycle. 1e-9 relative is tight enough
// that no plausible deliberately-DIFFERENT coefficient could pass it by
// accident, and loose enough to absorb that round-trip.
//
// P/Q-BACKED MAPS (Stage 2 of the P/Q milestone) are never recognized --
// this reads poly_terms()/pole_terms() directly, and a P/Q-backed map has
// no term list to read (is_pq_backed()). Correct either way: a P/Q map
// that happens to BE shaped like z^2+a still renders correctly, just
// through the general/compiled path instead of the hardcoded Family fast
// path (up to ~10% slower per the P5a.1 benchmark, never wrong) -- a
// representation-aware structural match against each of the shapes below,
// done directly on P and Q instead of terms, is future work if that gap
// ever matters in practice, not required for correctness now.
// -----------------------------------------------------------------------------
std::optional<Family> recognize_family(const RationalMap& m) {
    if (m.is_pq_backed()) return std::nullopt;

    std::vector<PolyTerm> polys;
    for (const auto& t : m.poly_terms()) if (t.enabled) polys.push_back(t);
    std::vector<PoleTerm> poles;
    for (const auto& t : m.pole_terms()) if (t.enabled) poles.push_back(t);

    // z^n + a: exactly one "+a" term (exponent 0, param_power 1, coeff 1)
    // and one z^n term (param_power 0, coeff 1), no poles -- see
    // RationalMap::multibrot. Order-independent: RationalMap does not
    // guarantee term insertion order survives a round trip.
    if (polys.size() == 2 && poles.empty()) {
        auto is_a_term = [](const PolyTerm& t) {
            return t.exponent == 0 && t.param_power == 1 && close_enough(t.coeff, Cplx(1, 0));
        };
        auto is_zn_term = [](const PolyTerm& t) {
            return t.param_power == 0 && close_enough(t.coeff, Cplx(1, 0));
        };
        const PolyTerm* zn = nullptr;
        if (is_a_term(polys[0]) && is_zn_term(polys[1])) zn = &polys[1];
        else if (is_a_term(polys[1]) && is_zn_term(polys[0])) zn = &polys[0];
        if (zn) {
            switch (zn->exponent) {
                case 2: return Family::Quadratic;
                case 3: return Family::Cubic;
                case 5: return Family::Quintic;
                default: break;
            }
        }
    }

    // z^n + a/z^n: one z^n poly term (param_power 0, coeff 1) plus one pole
    // of order n at the origin whose strength scales with a (coeff 1,
    // param_power 1) -- see RationalMap::mcmullen. Newton's method of
    // z^3-1, simplified to (2/3)z + (1/3)z^-2, has the SAME shape (one
    // poly term, one pole at the origin -- see RationalMap::newton_cubic,
    // whose z^-2 term is now a PoleTerm rather than a negative-exponent
    // PolyTerm, per add_poly's own restriction) but different values: a
    // linear (not degree-n) poly term with a FIXED coefficient (2/3, not
    // 1) and NO parameter dependence anywhere (param_power 0 throughout,
    // not the pole strength scaling as a^1 the way McMullen's does).
    if (polys.size() == 1 && poles.size() == 1) {
        const PolyTerm& zn = polys[0];
        const PoleTerm& p  = poles[0];
        if (zn.param_power == 0 && close_enough(zn.coeff, Cplx(1, 0)) &&
            !p.location_is_param && close_enough(p.location, Cplx(0, 0)) &&
            p.param_power == 1 && close_enough(p.strength, Cplx(1, 0)) &&
            p.order == zn.exponent) {
            if (zn.exponent == 2) return Family::McMullen2;
            if (zn.exponent == 3) return Family::McMullen3;
        }
        if (zn.exponent == 1 && zn.param_power == 0 &&
            close_enough(zn.coeff, Cplx(2.0 / 3.0, 0.0)) &&
            !p.location_is_param && close_enough(p.location, Cplx(0, 0)) &&
            p.order == 2 && p.param_power == 0 &&
            close_enough(p.strength, Cplx(1.0 / 3.0, 0.0))) {
            return Family::Newton3;
        }
    }

    return std::nullopt;
}

// -----------------------------------------------------------------------------
// Map
// -----------------------------------------------------------------------------
Map Map::custom(RationalMap m, Cplx param) {
    Map out;
    out.family_ = Family::Custom;
    out.param_ = param;
    out.custom_ = std::make_shared<const RationalMap>(std::move(m));
    return out;
}

int Map::degree() const {
    if (family_ == Family::Custom) return custom_->degree(param_);
    switch (family_) {
        case Family::Quadratic: return 2;
        case Family::Cubic:     return 3;
        case Family::Quintic:   return 5;
        case Family::McMullen2: return 4;   // z^2 + a/z^2 has degree 2n = 4
        case Family::McMullen3: return 6;   // z^3 + a/z^3 has degree 2n = 6
        case Family::Newton3:   return 3;
        case Family::Custom:    break;      // handled above
    }
    return 2;
}

bool Map::escape_certified() const {
    if (family_ == Family::Custom) return polynomial_escape_certified(*custom_);
    switch (family_) {
        case Family::Quadratic: return true;    // z^2+a
        case Family::Cubic:     return true;    // z^3+a
        case Family::Quintic:   return true;    // z^5+a
        case Family::McMullen2: return false;   // z^2+a/z^2 -- a pole at the origin
        case Family::McMullen3: return false;   // z^3+a/z^3 -- a pole at the origin
        case Family::Newton3:   return false;   // (2/3)z + (1/3)z^-2 -- a pole at the origin
        case Family::Custom:    break;          // handled above
    }
    return true;
}

void Map::step_with(Family f, double pr, double pi, double& zr, double& zi) {
    const double a = zr, b = zi;
    const double a2 = a * a, b2 = b * b;

    switch (f) {
        case Family::Quadratic:
            zr = a2 - b2 + pr;
            zi = 2.0 * a * b + pi;
            return;

        case Family::Cubic:
            zr = a * (a2 - 3.0 * b2) + pr;
            zi = b * (3.0 * a2 - b2) + pi;
            return;

        case Family::Quintic:
            zr = a * (a2 * a2 - 10.0 * a2 * b2 + 5.0 * b2 * b2) + pr;
            zi = b * (5.0 * a2 * a2 - 10.0 * a2 * b2 + b2 * b2) + pi;
            return;

        case Family::McMullen2: {
            const double sr = a2 - b2, si = 2.0 * a * b;      // z^2
            const double den = sr * sr + si * si;
            if (den < kTinyDen) { zr = kHuge; zi = 0.0; return; }
            zr = sr + (pr * sr + pi * si) / den;
            zi = si + (pi * sr - pr * si) / den;
            return;
        }

        case Family::McMullen3: {
            const double cr = a * (a2 - 3.0 * b2);            // z^3
            const double ci = b * (3.0 * a2 - b2);
            const double den = cr * cr + ci * ci;
            if (den < kTinyDen) { zr = kHuge; zi = 0.0; return; }
            zr = cr + (pr * cr + pi * ci) / den;
            zi = ci + (pi * cr - pr * ci) / den;
            return;
        }

        case Family::Newton3: {
            const double cr = a * (a2 - 3.0 * b2);            // z^3
            const double ci = b * (3.0 * a2 - b2);
            const double nr = cr - 1.0, ni = ci;              // z^3 - 1
            const double sr = 3.0 * (a2 - b2), si = 6.0 * a * b;  // 3 z^2
            const double den = sr * sr + si * si;
            if (den < kTinyDen) { zr = kHuge; zi = 0.0; return; }
            zr = a - (nr * sr + ni * si) / den;
            zi = b - (ni * sr - nr * si) / den;
            return;
        }

        // Unreachable in normal use: Map::step()/step_with_param() intercept
        // Custom before it ever reaches this static, family-keyed function,
        // since a Custom map's formula lives on the RationalMap instance,
        // not in the enum value. No-op rather than corrupt zr/zi.
        case Family::Custom:
            return;
    }
}

void Map::step(double& zr, double& zi) const {
    if (family_ == Family::Custom) {
        const Cplx z = custom_->eval(Cplx(zr, zi), param_);
        zr = z.real();
        zi = z.imag();
        return;
    }
    step_with(family_, param_.real(), param_.imag(), zr, zi);
}

Cplx Map::critical_point(Family f, Cplx param) {
    switch (f) {
        case Family::Quadratic:
        case Family::Cubic:
        case Family::Quintic:
            return {0.0, 0.0};                 // only critical point of z^n + a

        case Family::McMullen2: {              // z^4 = a
            const double m = std::pow(std::abs(param), 0.25);
            const double t = std::arg(param) * 0.25;
            return {m * std::cos(t), m * std::sin(t)};
        }
        case Family::McMullen3: {              // z^6 = a
            const double m = std::pow(std::abs(param), 1.0 / 6.0);
            const double t = std::arg(param) / 6.0;
            return {m * std::cos(t), m * std::sin(t)};
        }
        case Family::Newton3:
            return {0.0, 0.0};                 // parameter-independent

        // Unreachable in normal use; see the comment on the Custom case in
        // step_with. Use critical_point_at for a Custom map.
        case Family::Custom:
            return {0.0, 0.0};
    }
    return {0.0, 0.0};
}

Cplx Map::critical_point_at(Cplx p) const {
    if (family_ == Family::Custom) {
        const std::vector<Cplx> cps = custom_->critical_points(p);
        return cps.empty() ? Cplx(0.0, 0.0) : cps.front();
    }
    return critical_point(family_, p);
}

void Map::step_with_param(Cplx p, double& zr, double& zi) const {
    if (family_ == Family::Custom) {
        const Cplx z = custom_->eval(Cplx(zr, zi), p);
        zr = z.real();
        zi = z.imag();
        return;
    }
    step_with(family_, p.real(), p.imag(), zr, zi);
}

// -----------------------------------------------------------------------------
// Chordal metric
// -----------------------------------------------------------------------------
double chordal_distance(double zr, double zi, double wr, double wi) {
    const bool z_inf = std::isinf(zr) || std::isinf(zi) ||
                       is_bad(zr) || is_bad(zi) || (zr * zr + zi * zi) > kHuge;
    const bool w_inf = std::isinf(wr) || std::isinf(wi);

    if (z_inf && w_inf) return 0.0;
    if (w_inf) return 2.0 / std::sqrt(1.0 + zr * zr + zi * zi);
    if (z_inf) return 2.0 / std::sqrt(1.0 + wr * wr + wi * wi);

    const double dr = zr - wr, di = zi - wi;
    const double num = 2.0 * std::sqrt(dr * dr + di * di);
    const double den = std::sqrt((1.0 + zr * zr + zi * zi) *
                                 (1.0 + wr * wr + wi * wi));
    return num / den;
}

// -----------------------------------------------------------------------------
// Renderer helpers
// -----------------------------------------------------------------------------
void Renderer::zoom(Cplx target, double factor) {
    if (factor <= 0.0) return;
    view_.center = target;
    view_.scale /= factor;
}

double Renderer::precision_floor() const {
    // Neighbouring grid points must remain distinct doubles. The spacing is
    // 2*scale/(res-1); it must exceed the ULP at the centre's magnitude.
    const double mag = std::max(1.0, std::abs(view_.center));
    return mag * std::numeric_limits<double>::epsilon() * view_.resolution;
}

template <typename F>
void Renderer::parallel_columns(F body, const std::atomic<bool>* cancel) const {
    const int cols = view_.resolution;
    unsigned n = settings_.threads > 0
                     ? static_cast<unsigned>(settings_.threads)
                     : std::thread::hardware_concurrency();
    if (n == 0) n = 1;
    n = std::min<unsigned>(n, static_cast<unsigned>(std::max(1, cols)));

    // relaxed: this is a plain "has someone asked us to stop" poll, not
    // synchronizing access to any other data -- the caller only ever reads
    // the (discarded, on cancellation) partial Image after every worker
    // thread has been joined below, which is itself a full synchronization
    // point regardless of this load's memory order.
    if (n == 1) {
        for (int c = 0; c < cols; ++c) {
            if (cancel && cancel->load(std::memory_order_relaxed)) return;
            body(c);
        }
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(n);
    const int chunk = (cols + static_cast<int>(n) - 1) / static_cast<int>(n);
    for (unsigned t = 0; t < n; ++t) {
        const int lo = static_cast<int>(t) * chunk;
        const int hi = std::min(cols, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&body, lo, hi, cancel] {
            for (int c = lo; c < hi; ++c) {
                if (cancel && cancel->load(std::memory_order_relaxed)) return;
                body(c);
            }
        });
    }
    for (auto& th : pool) th.join();
}

namespace {
// Resolves, ONCE PER RENDER, how render_julia/render_basin/render_greens
// should advance a pixel's orbit for the CURRENT map -- shared so the
// three-way "not Custom / Custom-but-recognized / Custom-and-compiled"
// dispatch (see recognize_family/CompiledMap) is written once instead of
// once per render mode. render_parameter does NOT use this: there `a` is
// the PIXEL, not map_.param(), so its plan has to be resolved per pixel
// instead of per render -- see its own comment.
struct StepPlan {
    bool use_compiled = false;
    Family family = Family::Quadratic;   // valid when !use_compiled
    double pr = 0.0, pi = 0.0;           // valid when !use_compiled
    CompiledMap compiled;                // valid when use_compiled

    inline void step(double& zr, double& zi) const {
        if (use_compiled) compiled.step(zr, zi);
        else Map::step_with(family, pr, pi, zr, zi);
    }
};

StepPlan make_step_plan(const Map& m) {
    const RationalMap* custom = m.custom_map();
    const std::optional<Family> recognized = custom ? recognize_family(*custom) : std::nullopt;
    StepPlan plan;
    plan.use_compiled = custom && !recognized;
    plan.family = recognized.value_or(m.family());
    plan.pr = m.param().real();
    plan.pi = m.param().imag();
    if (plan.use_compiled) plan.compiled = custom->compile(m.param());
    return plan;
}

// -----------------------------------------------------------------------------
// Shared sphere-aware orbit classifier -- the SAME chordal-distance-against-
// found-attractors loop render_julia_rational (Stage 2) and render_greens'/
// render_parameter_greens' rational paths (Stage 3) all need, written once.
// `step` is templated (not std::function) so it inlines exactly like a
// direct StepPlan::step call would -- render_julia/render_greens pass a
// per-render-fixed StepPlan, render_parameter_greens passes a per-PIXEL
// closure (its step formula depends on the pixel's own parameter value; see
// its own comment), and both cost nothing extra here.
// -----------------------------------------------------------------------------
struct OrbitFate {
    bool   resolved  = false;
    int    label     = 0;
    double nu        = 0.0;    // pragmatic smooth chordal approach-rate (0 if unresolved)
    int    n         = 0;      // iterations taken to resolve (1-based; 0 if unresolved)
    double d_curr    = 0.0;    // chordal distance to the reached attractor at iteration n
    double d_prev    = 0.0;    // ...at iteration n-1, valid iff have_prev
    bool   have_prev = false;
};

template <typename Step>
OrbitFate classify_rational_orbit(double zr, double zi, Step&& step,
                                  const std::vector<double>& ar, const std::vector<double>& ai,
                                  const std::vector<int>& aid, double tol, int max_iter) {
    OrbitFate fate;
    const double log_tol = std::log(tol);
    const int nattr = static_cast<int>(ar.size());

    int prev_attractor = -1;
    double prev_dist = 0.0;
    bool have_prev = false;

    for (int n = 0; n < max_iter; ++n) {
        step(zr, zi);
        if (is_bad(zr) || is_bad(zi)) break;   // genuine overflow -- stays unresolved

        int closest = -1;
        double closest_dist = kHuge;
        for (int k = 0; k < nattr; ++k) {
            const double d = chordal_distance(zr, zi, ar[k], ai[k]);
            if (d < closest_dist) { closest_dist = d; closest = k; }
        }

        if (closest_dist < tol) {
            fate.resolved = true;
            fate.label = aid[closest];
            fate.n = n + 1;
            fate.d_curr = closest_dist;

            double nu = static_cast<double>(n + 1);
            if (have_prev && prev_attractor == closest && prev_dist > closest_dist &&
                closest_dist > 0.0) {
                fate.d_prev = prev_dist;
                fate.have_prev = true;
                const double log_prev = std::log(prev_dist);
                const double log_curr = std::log(closest_dist);
                const double denom = log_curr - log_prev;
                if (denom < 0.0 && std::isfinite(denom)) {
                    const double interpolated =
                        static_cast<double>(n - 1) + (log_tol - log_prev) / denom;
                    if (std::isfinite(interpolated)) nu = interpolated;
                }
            }
            fate.nu = nu;
            return fate;
        }

        prev_attractor = closest;
        prev_dist = closest_dist;
        have_prev = true;
    }
    return fate;   // unresolved: default-constructed (resolved=false)
}

// A finite critical-point seed that IS a pole of `map` (a pole of order >= 2
// is itself a sphere-critical point -- see RationalMap::critical_points)
// can't be STEPPED meaningfully starting exactly at the pole: the very
// first step lands on CompiledMap::step's own escape sentinel (see its
// comment), and the step after THAT overflows to NaN -- classify_rational_
// orbit then reports "unresolved" for an orbit that may well have been
// genuinely, cleanly attracted to infinity, exactly the same failure mode
// find_attractors_from_seeds' own identically-named nudge (analysis.cpp)
// exists to avoid for its map.eval()-based iteration. Mirrors that nudge
// exactly (same tolerance, same offset) so a Custom map's critical point
// that happens to sit on its own pole -- a structurally common case for
// Newton-like families, not a freak edge hit -- classifies the same way
// discovery already does, rather than poisoning every pixel that seeds
// from it (see render_parameter_rational's own comment for a measured
// case: Nova's z=0 critical point is also its own order-2 pole).
Cplx nudge_seed_off_pole(Cplx z, const RationalMap& map, Cplx a) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) return z;
    for (const Cplx& p : map.pole_locations(a)) {
        if (std::abs(z - p) < 1e-9) return z + Cplx(1e-4, 3.7e-5);
    }
    return z;
}

// -----------------------------------------------------------------------------
// Stage 3's Conformal potential, log|phi(z)| for the Boettcher (super-
// attracting) or Koenigs (geometrically attracting) coordinate at the
// attractor an OrbitFate resolved to, estimated numerically from ONLY the
// last two chordal distances straddling tol (fate.d_prev, fate.d_curr) and
// the integer step count (fate.n) -- no extra iteration, reusing exactly
// what classify_rational_orbit already computed.
//
// SUPERATTRACTING (incl. polynomial infinity, where this literally reduces
// to the certified-polynomial path's own log|z_n|/d^n -- see render_greens'
// own comment): near a superattracting point p of local degree k, the
// Boettcher coordinate is a genuine local biholomorphism with phi_p(p) = 0,
// phi_p(f(z)) = phi_p(z)^k, so phi_p(z) ~ C*(z-p) to leading order and
// d(z_n, p) ~ C' * d(z_{n-1}, p)^k locally -- k is estimated from the
// log-log ratio log(d_curr)/log(d_prev) (both logs negative for d<1),
// accepted only when it lands close to an integer >= 2 AND d_prev is
// already small enough (log_prev < -1.2, i.e. d_prev < ~0.3) for the
// leading-order approximation to be trustworthy. G_p(z) = -log(d_curr) /
// k^n follows from the functional equation G_p(f^n(z)) = k^n * G_p(z) with
// G_p(z_n) ~= -log(C*d(z_n,p)) ~= -log(d(z_n,p)) for z_n already close to p
// -- vanishes at the basin boundary (n -> large), diverges at p itself
// (n -> small), same sign convention as the certified-polynomial formula.
//
// GEOMETRIC (0 < |lambda| < 1): the Koenigs coordinate psi(z) with
// psi(f(z)) = lambda*psi(z) linearizes f near p; |lambda| is estimated
// directly as d_curr/d_prev (a genuine linear ratio, no log-log needed).
// log|psi(z)| ~= log(d_curr) - n*log(|lambda|) follows the same way. UNLIKE
// the superattracting case, this is NOT sign-determinate against the same
// "0 at boundary, diverges at p" convention -- psi is only defined up to an
// arbitrary nonzero multiplicative normalization (no k-power self-map to
// pin it down the way Boettcher's does), so this value can run either
// direction. That is a genuine, acknowledged modeling limitation, not a
// bug: it is still a perfectly good scalar for palette/equipotential-band
// coloring (color_scalar_field only needs a scalar field, not a
// particular sign), just not one that reduces to the Boettcher convention.
//
// PARABOLIC / AMBIGUOUS (|lambda| ~= 1, or no reliable local-rate data at
// all): the true Fatou-coordinate potential is hard and NOT attempted here
// -- returns exact=false so the caller falls back to the Pragmatic value.
// -----------------------------------------------------------------------------
struct ConformalPotential {
    double value = 0.0;
    bool   exact = false;
};

ConformalPotential conformal_potential(const OrbitFate& fate) {
    if (!fate.resolved || !fate.have_prev || fate.d_prev <= 0.0 || fate.d_curr <= 0.0)
        return {};

    const double log_prev = std::log(fate.d_prev);
    const double log_curr = std::log(fate.d_curr);
    if (!std::isfinite(log_prev) || !std::isfinite(log_curr) || log_prev >= 0.0) return {};
    const double n = static_cast<double>(fate.n);

    const double ratio_log = log_curr / log_prev;
    const double k_round = std::round(ratio_log);
    if (log_prev < -1.2 && k_round >= 2.0 && std::abs(ratio_log - k_round) < 0.2) {
        const double g = -log_curr / std::pow(k_round, n);
        if (std::isfinite(g)) return {g, true};
    }

    const double lambda_est = fate.d_curr / fate.d_prev;
    if (lambda_est > 1e-6 && lambda_est < 0.98) {
        const double g = log_curr - n * std::log(lambda_est);
        if (std::isfinite(g)) return {g, true};
    }

    return {};   // parabolic or too ambiguous to classify -- caller falls back to Pragmatic
}
}  // namespace

// -----------------------------------------------------------------------------
// Julia
// -----------------------------------------------------------------------------
// The pre-Stage-2 escape-time path, UNCHANGED, extracted so
// Renderer::render_julia's own dispatch reads as "certified -> this;
// rational -> the chordal path below" rather than one function trying to
// be both. escape_radius-based; correct and fast for a CERTIFIED
// polynomial (see Map::escape_certified/polynomial_escape_certified's own
// doc comments for why that certification is what makes it correct, not
// just conventional).
Image Renderer::render_julia_polynomial(const std::atomic<bool>* cancel) const {
    const int  res  = view_.resolution;
    const double R2 = settings_.escape_radius * settings_.escape_radius;
    const double inv_log2 = 1.0 / std::log(2.0);
    Image img(res, res);

    // `a` = map_.param() is fixed for the WHOLE render, so the step plan --
    // native step_with formula for a built-in family or a recognized
    // Custom shape, else compile(a) once -- is resolved once here rather
    // than per pixel or per iteration. See StepPlan/make_step_plan above.
    const StepPlan plan = make_step_plan(map_);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            double zr = c.real(), zi = c.imag();
            double out = 0.0;

            for (int n = 0; n < settings_.max_iter; ++n) {
                plan.step(zr, zi);
                const double m2 = zr * zr + zi * zi;

                if (m2 > R2) {                       // per-pixel early exit
                    const double nu = static_cast<double>(n + 1) + 1.0 -
                                      std::log(std::log(std::sqrt(m2))) * inv_log2;
                    out = is_bad(nu) ? static_cast<double>(n + 1) : nu;
                    break;
                }
                if (is_bad(m2) || m2 > kHuge) { out = static_cast<double>(n + 1); break; }
            }
            img.at(col, row) = out;
        }
    }, cancel);
    return img;
}

// The Stage-2 sphere-aware path for a map WITH poles -- structured like
// Renderer::render_basin (chordal_distance to each found attractor,
// including infinity as an ordinary point when it's one of `cycles`), but
// producing a SMOOTH value per pixel instead of just a discrete iteration
// count, the rational analog of render_julia_polynomial's own smooth
// escape-time nu. NO escape_radius anywhere here -- see render_julia's
// own header doc comment.
//
// SMOOTHING: once a pixel's chordal distance to its eventual attractor
// drops below tol at iteration n, its LOCAL contraction rate is estimated
// from the ratio of the last two distances straddling tol (d at n-1, d at
// n) via the same log-log interpolation idea render_julia_polynomial's
// own double-log formula uses for escape, just in "distance shrinking
// toward a point" space instead of "modulus growing away from one":
// treating d_k ~= C * |lambda|^k locally, log|lambda| is estimated as
// log(d_n) - log(d_{n-1}), and the fractional iteration n* where d
// crosses tol solves out to (n-1) + (log(tol) - log(d_{n-1})) /
// (log(d_n) - log(d_{n-1})). This needs the PREVIOUS iteration's distance
// to the SAME attractor the pixel eventually reaches -- if the closest
// attractor changed between n-1 and n (rare, near a boundary between two
// basins), there is no valid local rate to interpolate from, so the pixel
// falls back to the plain integer count n+1 (matching
// render_julia_polynomial's own is_bad/degenerate-case fallback
// philosophy: punt to something reasonable rather than propagate a NaN
// from a near-zero or negative log-ratio).
Image Renderer::render_julia_rational(const std::vector<Cycle>& cycles, Image* labels,
                                      const std::atomic<bool>* cancel) const {
    const int res = view_.resolution;
    Image img(res, res);
    if (labels) *labels = Image(res, res);
    if (cycles.empty()) return img;   // nothing to classify against -- every pixel unresolved

    // Flatten to parallel arrays, same reasoning as render_basin's own.
    std::vector<double> ar, ai;
    std::vector<int>    aid;
    for (const auto& cyc : cycles) {
        for (const auto& pt : cyc.points) {
            ar.push_back(pt.real());
            ai.push_back(pt.imag());
            aid.push_back(cyc.id);
        }
    }
    const double tol = settings_.tol;

    const StepPlan plan = make_step_plan(map_);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            const OrbitFate fate = classify_rational_orbit(
                c.real(), c.imag(), [&](double& zr, double& zi) { plan.step(zr, zi); },
                ar, ai, aid, tol, settings_.max_iter);

            img.at(col, row) = fate.resolved ? fate.nu : 0.0;
            if (labels) labels->at(col, row) = fate.resolved ? static_cast<double>(fate.label) : 0.0;
        }
    }, cancel);
    return img;
}

Image Renderer::render_julia(const std::atomic<bool>* cancel, const std::vector<Cycle>& cycles,
                             Image* labels) const {
    if (map_.escape_certified()) return render_julia_polynomial(cancel);
    return render_julia_rational(cycles, labels, cancel);
}

// -----------------------------------------------------------------------------
// Parameter plane
// -----------------------------------------------------------------------------
// escape_radius-based escape-TIME -- see render_parameter's own header doc
// comment for why this mode stays escape_radius-based (a visualization
// knob, not a set-membership invariant) even after Stage 2/3's sphere-
// aware, escape-radius-free treatment of Julia/Green's, and for why a
// brief Stage 4 detour into a multi-critical, escape-radius-free rational
// path here was retired in favour of Parameter_basin as its own dedicated
// mode.
// Pre-Stage-1 form, UNCHANGED -- extracted verbatim so render_parameter's own
// dispatch reads the same "certified -> this; rational -> the sphere-aware
// path below" shape every other mode already has. escape_radius-based;
// correct here because a CERTIFIED polynomial's infinity is ALWAYS
// superattracting, so a fixed |z| > escape_radius test is a provably
// forward-invariant trap (same reasoning as render_julia_polynomial's own).
Image Renderer::render_parameter_polynomial(const std::atomic<bool>* cancel) const {
    const int    res = view_.resolution;
    const double R2  = settings_.escape_radius * settings_.escape_radius;
    const double inv_log2 = 1.0 / std::log(2.0);
    Image img(res, res);

    // The parameter here is the PIXEL, not map_.param() -- everything below
    // is computed once per PIXEL, not once per render, unlike the other
    // three modes. Three paths, cheapest first:
    //   * not Custom: unchanged, step_with_param's existing behaviour.
    //   * recognized (see recognize_family): the map IS structurally a
    //     built-in shape, so its critical point is the O(1) native formula
    //     (Map::critical_point -- exact for all six shapes, including
    //     McMullen's genuinely parameter-dependent z^(2n)=a) and each
    //     iteration is step_with's native formula. No compile() at all.
    //   * otherwise: critical_points_constant() decides whether the
    //     critical point can be computed once for the whole render instead
    //     of once per pixel (see that method's doc comment -- normally the
    //     dominant cost of rendering a Custom map's parameter plane), and
    //     compile(p) still happens once per pixel, since `a` = p varies
    //     pixel to pixel here unlike the other three render_* modes.
    const RationalMap* custom = map_.custom_map();
    const std::optional<Family> recognized = custom ? recognize_family(*custom) : std::nullopt;

    const bool cp_fixed = !recognized && custom && custom->critical_points_constant();
    const Cplx fixed_c0 = cp_fixed ? map_.critical_point_at(Cplx(1.0, 0.0)) : Cplx(0.0, 0.0);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx p = view_.coord(col, row);      // the PIXEL is the parameter
            Cplx c0;
            if (recognized) c0 = Map::critical_point(*recognized, p);
            else if (cp_fixed) c0 = fixed_c0;
            else c0 = map_.critical_point_at(p);
            double zr = c0.real(), zi = c0.imag();
            double out = 0.0;

            const CompiledMap compiled = (custom && !recognized) ? custom->compile(p) : CompiledMap{};

            for (int n = 0; n < settings_.max_iter; ++n) {
                if (recognized) {
                    Map::step_with(*recognized, p.real(), p.imag(), zr, zi);
                } else if (custom) {
                    compiled.step(zr, zi);
                } else {
                    map_.step_with_param(p, zr, zi);
                }
                const double m2 = zr * zr + zi * zi;

                if (m2 > R2) {
                    const double nu = static_cast<double>(n + 1) + 1.0 -
                                      std::log(std::log(std::sqrt(m2))) * inv_log2;
                    out = is_bad(nu) ? static_cast<double>(n + 1) : nu;
                    break;
                }
                if (is_bad(m2) || m2 > kHuge) { out = static_cast<double>(n + 1); break; }
            }
            img.at(col, row) = out;
        }
    }, cancel);
    return img;
}

// -----------------------------------------------------------------------------
// Sphere-aware, multi-critical, escape-radius-free rational path -- see
// render_parameter's own header doc comment for the full derivation and the
// git-history context (the reverted "AllCaptured/FastestCapture/PerCritical"
// attempt this generalizes correctly).
//
// Custom map: every free critical point (RationalMap::distinct_critical_
// points(a), UNFILTERED) is classified against complete_attractors_from_
// seeds' own discovery at that SAME seed list -- one extra find_attractors
// pass per pixel beyond what render_parameter_basin already pays, since the
// two methods answer different questions (a count vs a per-critical smooth
// rate) and neither can be derived from the other's own return shape alone.
//
// Non-Custom built-in rational family (McMullen2/McMullen3/Newton3 -- the
// only Family values that ever reach this method, since Quadratic/Cubic/
// Quintic are escape_certified and dispatch to render_parameter_polynomial
// instead): there is no RationalMap for complete_attractors_from_seeds to
// call eval()/deriv()/fixed_points() on, so ONE is synthesized once per
// render (RationalMap::mcmullen/newton_cubic -- the exact same closed-form
// shape Map::step_with already hard-codes, just also available as a
// RationalMap for discovery's sake) purely as that discovery target. The
// TRACKED seed stays Map::critical_point's single representative (per
// ParameterStrategy's own doc comment: every finite critical point of one
// of these three specific families shares escape/capture behaviour by
// symmetry, so nothing is lost by not also calling the synthesized map's
// own distinct_critical_points here).
Image Renderer::render_parameter_rational(ParameterStrategy strategy, int critical_index,
                                          const std::atomic<bool>* cancel) const {
    const int res = view_.resolution;
    Image img(res, res);

    const RationalMap* custom = map_.custom_map();
    const std::optional<Family> recognized = custom ? recognize_family(*custom) : std::nullopt;
    const bool cp_fixed = !recognized && custom && custom->critical_points_constant();
    const std::vector<Cplx> fixed_crit_pts =
        cp_fixed ? custom->distinct_critical_points(Cplx(1.0, 0.0)) : std::vector<Cplx>{};

    std::optional<RationalMap> synthesized;
    const RationalMap* discovery_map = custom;
    if (!custom) {
        switch (map_.family()) {
            case Family::McMullen2: synthesized = RationalMap::mcmullen(2); break;
            case Family::McMullen3: synthesized = RationalMap::mcmullen(3); break;
            case Family::Newton3:   synthesized = RationalMap::newton_cubic(); break;
            default: break;   // unreachable: those three are escape_certified
        }
        discovery_map = synthesized ? &*synthesized : nullptr;
    }
    if (!discovery_map) return img;   // unreachable in practice; honest empty fallback

    // Same "correctness knob, not a speed knob" reasoning as render_
    // parameter_basin's own identical choice -- see that method's comment.
    const FindAttractorsOptions opts;

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx p = view_.coord(col, row);      // the PIXEL is the parameter

            std::vector<Cplx> crit_pts;
            if (custom) crit_pts = cp_fixed ? fixed_crit_pts : custom->distinct_critical_points(p);
            else        crit_pts.push_back(map_.critical_point_at(p));

            const std::vector<Cycle> cycles =
                complete_attractors_from_seeds(crit_pts, *discovery_map, p, opts);

            std::vector<double> ar, ai;
            std::vector<int>    aid;
            for (const Cycle& cyc : cycles) {
                for (const Cplx& pt : cyc.points) {
                    ar.push_back(pt.real());
                    ai.push_back(pt.imag());
                    aid.push_back(cyc.id);
                }
            }

            const CompiledMap compiled = (custom && !recognized) ? custom->compile(p) : CompiledMap{};
            auto step = [&](double& zr, double& zi) {
                if (recognized) Map::step_with(*recognized, p.real(), p.imag(), zr, zi);
                else if (custom) compiled.step(zr, zi);
                else map_.step_with_param(p, zr, zi);
            };

            std::vector<OrbitFate> fates;
            fates.reserve(crit_pts.size());
            for (const Cplx& c0 : crit_pts) {
                const Cplx seed = nudge_seed_off_pole(c0, *discovery_map, p);
                fates.push_back(classify_rational_orbit(seed.real(), seed.imag(), step, ar, ai, aid,
                                                         settings_.tol, settings_.max_iter));
            }

            double out = 0.0;
            switch (strategy) {
                case ParameterStrategy::Slowest: {
                    bool all_resolved = !fates.empty();
                    double worst = 0.0;
                    for (const OrbitFate& f : fates) {
                        if (!f.resolved) { all_resolved = false; break; }
                        worst = std::max(worst, f.nu);
                    }
                    out = all_resolved ? worst : 0.0;
                    break;
                }
                case ParameterStrategy::Fastest: {
                    double best = kHuge;
                    bool any_resolved = false;
                    for (const OrbitFate& f : fates) {
                        if (f.resolved) { any_resolved = true; best = std::min(best, f.nu); }
                    }
                    out = any_resolved ? best : 0.0;
                    break;
                }
                case ParameterStrategy::PerCritical: {
                    if (!fates.empty()) {
                        const int idx =
                            std::clamp(critical_index, 0, static_cast<int>(fates.size()) - 1);
                        out = fates[idx].resolved ? fates[idx].nu : 0.0;
                    }
                    break;
                }
            }
            img.at(col, row) = out;
        }
    }, cancel);
    return img;
}

// Public dispatcher -- same Map::escape_certified() split every other mode
// already uses; see render_parameter's own header doc comment for what each
// path means.
Image Renderer::render_parameter(const std::atomic<bool>* cancel, ParameterStrategy strategy,
                                 int critical_index) const {
    if (map_.escape_certified()) return render_parameter_polynomial(cancel);
    return render_parameter_rational(strategy, critical_index, cancel);
}

// -----------------------------------------------------------------------------
// Parameter_basin -- see the header doc comment for the full method and the
// honestly-tracked residual (Siegel/Herman/parabolic land in `unresolved`,
// never in the attracting-cycle count). No native/compiled step dispatch
// here at all -- unlike every OTHER render_* method, this one's inner loop
// is complete_attractors_from_seeds' own map.eval()/map.deriv() calls (the
// general RationalMap evaluator), not a per-render StepPlan/CompiledMap;
// a recognized built-in shape gets no fast-path benefit here, since the
// discovery machinery being reused doesn't take a custom step function.
//
// complete_attractors_from_seeds, not find_attractors_from_seeds directly
// (see analysis.hpp's own doc comment): the count this method reports must
// agree with the fact sheet's attracting_cycles and with basin-mode's own
// classification for the SAME (map, a) -- using the plain critical-seeded
// count here would silently under-count at exactly the parameter pixels
// where those other two consumers' union recovers an extra, algebraically-
// attracting fixed point.
// -----------------------------------------------------------------------------
Image Renderer::render_parameter_basin(const std::atomic<bool>* cancel, Image* unresolved,
                                       Image* certain) const {
    const int res = view_.resolution;
    Image img(res, res);
    if (unresolved) *unresolved = Image(res, res);
    if (certain) *certain = Image(res, res);

    const RationalMap* custom = map_.custom_map();
    if (!custom) return img;   // see header doc comment -- no RationalMap to discover cycles against

    // See render_parameter/render_parameter_greens for the SAME critical_
    // points_constant() optimization -- a Custom map whose critical points
    // don't depend on `a` at all gets them computed ONCE for the whole
    // render, not once per pixel (the dominant cost here otherwise).
    const bool cp_fixed = custom->critical_points_constant();
    const std::vector<Cplx> fixed_crit_pts =
        cp_fixed ? custom->distinct_critical_points(Cplx(1.0, 0.0)) : std::vector<Cplx>{};

    // Deliberately NOT settings_.tol: that governs a DIFFERENT notion --
    // "is a pixel's own orbit close enough to a KNOWN attractor" (basin
    // membership, where a coarser user-tunable tolerance is fine, since
    // the target is a single fixed point/cycle already in hand). Cycle-
    // CLOSURE detection (does THIS orbit return to itself) is a stricter
    // question: a looser tolerance there risks a FALSE POSITIVE -- a
    // merely slowly-converging (near-parabolic) orbit that never actually
    // closes can look like it "closed" under a loose enough tol, exactly
    // the false-attractor failure mode this mode's own `unresolved`
    // tracking exists to avoid. find_attractors' own established default
    // (1e-9) is what every other trusted caller already relies on; every
    // FindAttractorsOptions field stays at its default here for the same
    // reason burn_in/max_period do -- these govern CORRECTNESS of
    // discovery, not a rendering-speed knob to retune per mode.
    const FindAttractorsOptions opts;

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx p = view_.coord(col, row);      // the PIXEL is the parameter
            const std::vector<Cplx> crit_pts =
                cp_fixed ? fixed_crit_pts : custom->distinct_critical_points(p);

            int unresolved_n = 0;
            int certain_n = 0;
            const std::vector<Cycle> cycles = complete_attractors_from_seeds(
                crit_pts, *custom, p, opts, &unresolved_n, &certain_n);

            img.at(col, row) = static_cast<double>(cycles.size());
            if (unresolved) unresolved->at(col, row) = static_cast<double>(unresolved_n);
            if (certain) certain->at(col, row) = static_cast<double>(certain_n);
        }
    }, cancel);
    return img;
}

// -----------------------------------------------------------------------------
// Per-pixel tracked-seed generation for render_parameter_period, one case per
// CriticalPointFamily enumerator -- see that enum's own doc comment for why
// this is a small, explicit dispatch rather than a plugin mechanism. Returns
// an EMPTY vector at a degenerate parameter (e.g. RelaxedNewtonPower's own
// formula divides by (n-a), which vanishes at a=n exactly) rather than
// guessing -- render_parameter_period's own caller then correctly reports
// that single pixel as undetermined, the same honest treatment every other
// non-convergent case gets.
namespace {

std::vector<Cplx> tracked_seeds(CriticalPointFamily seed_family, int n, Cplx a) {
    switch (seed_family) {
        case CriticalPointFamily::RelaxedNewtonPower: {
            // z_c(a) = (a(n-1)/(n-a))^(1/n) * omega^j, j=0..n-1 -- see
            // CriticalPointFamily::RelaxedNewtonPower's own doc comment.
            const Cplx denom = Cplx(static_cast<double>(n), 0.0) - a;
            if (std::abs(denom) < 1e-12) return {};
            const Cplx ratio = a * static_cast<double>(n - 1) / denom;
            const double two_pi = 2.0 * std::acos(-1.0);
            const Cplx base = std::pow(ratio, 1.0 / static_cast<double>(n));
            const Cplx omega = std::polar(1.0, two_pi / static_cast<double>(n));
            std::vector<Cplx> seeds;
            seeds.reserve(static_cast<std::size_t>(n));
            Cplx w(1.0, 0.0);
            for (int j = 0; j < n; ++j) {
                seeds.push_back(base * w);
                w *= omega;
            }
            return seeds;
        }
    }
    return {};   // unreachable for a valid enumerator; never guess a seed list
}

}  // namespace

Image Renderer::render_parameter_period(CriticalPointFamily seed_family, int n,
                                        const std::atomic<bool>* cancel, Image* undetermined,
                                        Image* is_infinity) const {
    const int res = view_.resolution;
    Image img(res, res);
    if (undetermined) *undetermined = Image(res, res);
    if (is_infinity) *is_infinity = Image(res, res);

    const RationalMap* custom = map_.custom_map();
    // An all-undetermined degenerate image (never guessed) matches
    // render_parameter_basin's own treatment of the same missing-
    // RationalMap case -- see its own doc comment.
    if (!custom) {
        if (undetermined) {
            for (double& v : undetermined->data) v = 1.0;
        }
        return img;
    }

    const FindAttractorsOptions opts;

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx a = view_.coord(col, row);      // the PIXEL is the parameter
            const std::vector<Cplx> seeds = tracked_seeds(seed_family, n, a);

            int period = 0;
            bool inf_flag = false;
            bool resolved_any = false;
            if (!seeds.empty()) {
                // First tracked seed (in `seeds`' own generation order) that
                // resolves wins -- see render_parameter_period's own doc
                // comment for why this is a tie-break for RelaxedNewtonPower
                // specifically (its n seeds are equivariant-guaranteed to
                // agree), not a general disagreement policy.
                for (const SeedOutcome& outcome : per_seed_outcomes(seeds, *custom, a, opts)) {
                    if (outcome.resolved) {
                        period = outcome.period;
                        inf_flag = outcome.is_infinity;
                        resolved_any = true;
                        break;
                    }
                }
            }

            img.at(col, row) = inf_flag ? 0.0 : static_cast<double>(period);
            if (undetermined) undetermined->at(col, row) = resolved_any ? 0.0 : 1.0;
            if (is_infinity) is_infinity->at(col, row) = inf_flag ? 1.0 : 0.0;
        }
    }, cancel);
    return img;
}

// -----------------------------------------------------------------------------
// Parameter-plane Green's function (family escape-rate function, G_M(c) for
// the quadratic family) -- see the header comment for why this is a
// DIFFERENT function on a DIFFERENT space from render_greens, not a
// reparameterization of it. Structurally: render_parameter's per-pixel
// critical-point/step dispatch, with render_greens' accumulate-and-
// normalize body substituted for render_parameter's escape-time test.
// -----------------------------------------------------------------------------
// The pre-Stage-3 escape-radius path, UNCHANGED, extracted the same way as
// render_greens_polynomial.
Image Renderer::render_parameter_greens_polynomial(const std::atomic<bool>* cancel) const {
    const int    res  = view_.resolution;
    const double escR = settings_.escape_radius;
    Image img(res, res);

    // See render_parameter for the three-path critical-point/step
    // dispatch this mirrors exactly.
    const RationalMap* custom = map_.custom_map();
    const std::optional<Family> recognized = custom ? recognize_family(*custom) : std::nullopt;

    const bool cp_fixed = !recognized && custom && custom->critical_points_constant();
    const Cplx fixed_c0 = cp_fixed ? map_.critical_point_at(Cplx(1.0, 0.0)) : Cplx(0.0, 0.0);

    // Unlike the critical point above, degree is ALWAYS parameter-
    // independent (RationalMap::degree(Cplx a) is purely structural and
    // never reads `a` -- see its own implementation), so it is computed
    // ONCE for the whole render, the same as render_greens does for its
    // own fixed-`a` case, never per-pixel.
    int degree;
    if (recognized) degree = Map(*recognized, Cplx(0.0, 0.0)).degree();
    else if (custom) degree = custom->degree(Cplx(0.0, 0.0));
    else degree = map_.degree();
    const double ddeg = static_cast<double>(degree);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx p = view_.coord(col, row);      // the PIXEL is the parameter
            Cplx c0;
            if (recognized) c0 = Map::critical_point(*recognized, p);
            else if (cp_fixed) c0 = fixed_c0;
            else c0 = map_.critical_point_at(p);
            double zr = c0.real(), zi = c0.imag();

            const CompiledMap compiled = (custom && !recognized) ? custom->compile(p) : CompiledMap{};

            // Escape-rate potential G(z) = lim d^-n log+|f^n(z)|: normalize
            // AT the pixel's own escape iteration, not against a single
            // global max_iter^degree (see render_greens' own comment for
            // why the old accumulate-then-divide-by-degree^max_iter form
            // was simply wrong, not just overflow-prone).
            double g = 0.0;
            for (int n = 0; n < settings_.max_iter; ++n) {
                if (recognized) {
                    Map::step_with(*recognized, p.real(), p.imag(), zr, zi);
                } else if (custom) {
                    compiled.step(zr, zi);
                } else {
                    map_.step_with_param(p, zr, zi);
                }
                const double mag = std::sqrt(zr * zr + zi * zi);
                if (is_bad(mag)) break;
                if (mag > escR) {
                    g = std::log(mag) / std::pow(ddeg, static_cast<double>(n + 1));
                    if (is_bad(g)) g = 0.0;
                    break;
                }
            }
            img.at(col, row) = g;
        }
    }, cancel);
    return img;
}

// Stage 3's sphere-aware rational path -- see render_parameter_greens' own
// header doc comment for why this tracks chordal distance to a FIXED,
// trivial one-point "infinity" attractor rather than a per-parameter-pixel
// find_attractors call. render_parameter_basin (Parameter_basin) solves
// the harder multi-critical version of this same "which critical point(s),
// and against what" question; this function's own question stays the
// narrower "does THE critical orbit escape to infinity," which never
// needed find_attractors either. Same classify_rational_orbit/
// conformal_potential pair render_greens_rational uses, just with a
// per-PIXEL step closure (the parameter varies per pixel here) instead of
// a per-render-fixed StepPlan.
Image Renderer::render_parameter_greens_rational(GreensPotential potential, Image* exact,
                                                  const std::atomic<bool>* cancel) const {
    const int res = view_.resolution;
    Image img(res, res);
    if (exact) *exact = Image(res, res);

    const std::vector<double> ar{kInf};
    const std::vector<double> ai{0.0};
    const std::vector<int>    aid{1};
    const double tol = settings_.tol;

    // See render_parameter for the three-path critical-point/step dispatch
    // this mirrors exactly.
    const RationalMap* custom = map_.custom_map();
    const std::optional<Family> recognized = custom ? recognize_family(*custom) : std::nullopt;

    const bool cp_fixed = !recognized && custom && custom->critical_points_constant();
    const Cplx fixed_c0 = cp_fixed ? map_.critical_point_at(Cplx(1.0, 0.0)) : Cplx(0.0, 0.0);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx p = view_.coord(col, row);      // the PIXEL is the parameter
            Cplx c0;
            if (recognized) c0 = Map::critical_point(*recognized, p);
            else if (cp_fixed) c0 = fixed_c0;
            else c0 = map_.critical_point_at(p);

            const CompiledMap compiled = (custom && !recognized) ? custom->compile(p) : CompiledMap{};
            auto step = [&](double& zr, double& zi) {
                if (recognized) Map::step_with(*recognized, p.real(), p.imag(), zr, zi);
                else if (custom) compiled.step(zr, zi);
                else map_.step_with_param(p, zr, zi);
            };

            const OrbitFate fate = classify_rational_orbit(c0.real(), c0.imag(), step, ar, ai, aid,
                                                            tol, settings_.max_iter);

            double g = 0.0;
            bool is_exact = false;
            if (fate.resolved) {
                if (potential == GreensPotential::Pragmatic) {
                    g = fate.nu;
                } else {
                    const ConformalPotential cp = conformal_potential(fate);
                    g = cp.exact ? cp.value : fate.nu;
                    is_exact = cp.exact;
                }
            }
            img.at(col, row) = g;
            if (exact) exact->at(col, row) = is_exact ? 1.0 : 0.0;
        }
    }, cancel);
    return img;
}

Image Renderer::render_parameter_greens(const std::atomic<bool>* cancel, GreensPotential potential,
                                        Image* exact) const {
    if (map_.escape_certified()) return render_parameter_greens_polynomial(cancel);
    return render_parameter_greens_rational(potential, exact, cancel);
}

// -----------------------------------------------------------------------------
// Basins
// -----------------------------------------------------------------------------
Image Renderer::render_basin(const std::vector<Cycle>& cycles, Image* iterations,
                             const std::atomic<bool>* cancel) const {
    const int res = view_.resolution;
    Image img(res, res);
    if (iterations) *iterations = Image(res, res);
    if (cycles.empty()) return img;

    // Flatten to parallel arrays: better locality in the innermost test.
    std::vector<double> ar, ai;
    std::vector<int>    aid;
    for (const auto& cyc : cycles) {
        for (const auto& pt : cyc.points) {
            ar.push_back(pt.real());
            ai.push_back(pt.imag());
            aid.push_back(cyc.id);
        }
    }
    const int nattr = static_cast<int>(ar.size());
    const double tol = settings_.tol;

    // See render_julia: `a` = map_.param() is fixed for the whole render.
    const StepPlan plan = make_step_plan(map_);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            double zr = c.real(), zi = c.imag();
            double label = 0.0;
            int steps = 0;

            for (int n = 0; n < settings_.max_iter; ++n) {
                plan.step(zr, zi);
                steps = n + 1;

                for (int k = 0; k < nattr; ++k) {
                    if (chordal_distance(zr, zi, ar[k], ai[k]) < tol) {
                        label = static_cast<double>(aid[k]);
                        break;
                    }
                }
                if (label > 0.0) break;
                if (is_bad(zr) || is_bad(zi)) break;
            }
            img.at(col, row) = label;
            if (iterations) iterations->at(col, row) = static_cast<double>(steps);
        }
    }, cancel);
    return img;
}

// -----------------------------------------------------------------------------
// Green's function
// -----------------------------------------------------------------------------
// The pre-Stage-3 escape-radius path, UNCHANGED, extracted the same way
// render_julia_polynomial was in Stage 2 -- correct and fast for a
// CERTIFIED polynomial, where PRAGMATIC and CONFORMAL-Boettcher already
// coincide exactly (see GreensPotential's own doc comment), so there is
// nothing a `potential` selector would even distinguish here.
Image Renderer::render_greens_polynomial(const std::atomic<bool>* cancel) const {
    const int    res  = view_.resolution;
    const double escR = settings_.escape_radius;
    const double ddeg = static_cast<double>(map_.degree());
    Image img(res, res);

    // See render_julia: `a` = map_.param() is fixed for the whole render.
    const StepPlan plan = make_step_plan(map_);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            double zr = c.real(), zi = c.imag();

            // Escape-rate potential: G(z) = lim_{n->inf} d^-n log+|f^n(z)|.
            // Normalize AT the pixel's own escape iteration n, not by
            // accumulating log(max(|z|,1)) over the WHOLE orbit and
            // dividing by one global degree^max_iter. That old form was
            // wrong, not just overflow-prone: measured on z^2+c, the
            // accumulated sum saturates near 58 while degree^max_iter
            // (2^200) is ~1.6e60, so G ~ 1e-59 everywhere -- a constant
            // field that renders flat gray at any color scaling, not a
            // dead pixel here and there. Per-pixel normalization at each
            // pixel's own (usually small) escape step keeps values
            // well-scaled: ~83% of pixels nonzero with genuine ~1e56
            // dynamic range on the same test case. Non-escaping pixels
            // get exactly 0, matching G's own definition on the filled
            // set. This also removes the overflow case this function
            // used to guard against (`normalized`/kHuge/is_bad on the
            // accumulator): std::pow(degree, n+1) is only ever evaluated
            // at ONE pixel's own escape step, essentially never near
            // max_iter, and even if it were, IEEE division by +inf
            // degrades gracefully to +0.0 for that single pixel rather
            // than requiring a whole-image escape hatch.
            double g = 0.0;
            for (int n = 0; n < settings_.max_iter; ++n) {
                plan.step(zr, zi);
                const double mag = std::sqrt(zr * zr + zi * zi);
                if (is_bad(mag)) break;
                if (mag > escR) {
                    g = std::log(mag) / std::pow(ddeg, static_cast<double>(n + 1));
                    if (is_bad(g)) g = 0.0;
                    break;
                }
            }
            img.at(col, row) = g;
        }
    }, cancel);
    return img;
}

// Stage 3's sphere-aware rational path -- same classify_rational_orbit
// (shared with render_julia_rational) supplying BOTH the resolved
// attractor's chordal approach data and the PRAGMATIC value directly;
// CONFORMAL is derived from that same data by conformal_potential (see its
// own doc comment for the Boettcher/Koenigs derivations). escape_radius
// plays no role, same reasoning as render_julia_rational's own.
Image Renderer::render_greens_rational(const std::vector<Cycle>& cycles, GreensPotential potential,
                                       Image* exact, const std::atomic<bool>* cancel) const {
    const int res = view_.resolution;
    Image img(res, res);
    if (exact) *exact = Image(res, res);
    if (cycles.empty()) return img;

    std::vector<double> ar, ai;
    std::vector<int>    aid;
    for (const auto& cyc : cycles) {
        for (const auto& pt : cyc.points) {
            ar.push_back(pt.real());
            ai.push_back(pt.imag());
            aid.push_back(cyc.id);
        }
    }
    const double tol = settings_.tol;
    const StepPlan plan = make_step_plan(map_);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            const OrbitFate fate = classify_rational_orbit(
                c.real(), c.imag(), [&](double& zr, double& zi) { plan.step(zr, zi); },
                ar, ai, aid, tol, settings_.max_iter);

            double g = 0.0;
            bool is_exact = false;
            if (fate.resolved) {
                if (potential == GreensPotential::Pragmatic) {
                    g = fate.nu;
                } else {
                    const ConformalPotential cp = conformal_potential(fate);
                    g = cp.exact ? cp.value : fate.nu;
                    is_exact = cp.exact;
                }
            }
            img.at(col, row) = g;
            if (exact) exact->at(col, row) = is_exact ? 1.0 : 0.0;
        }
    }, cancel);
    return img;
}

Image Renderer::render_greens(const std::atomic<bool>* cancel, const std::vector<Cycle>& cycles,
                              GreensPotential potential, Image* exact) const {
    if (map_.escape_certified()) return render_greens_polynomial(cancel);
    return render_greens_rational(cycles, potential, exact, cancel);
}

}  // namespace cdx
