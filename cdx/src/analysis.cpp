// =============================================================================
// cdx/analysis.cpp -- attractor discovery and diagnostics.
// =============================================================================
#include "cdx/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cdx {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

bool is_finite_cplx(Cplx z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

double chordal(Cplx z, Cplx w) {
    return chordal_distance(z.real(), z.imag(), w.real(), w.imag());
}

}  // namespace

// =============================================================================
// 1. find_attractors
// =============================================================================
namespace {

bool counts_as_infinite(Cplx z, double inf_cutoff) {
    return !is_finite_cplx(z) || std::abs(z) > inf_cutoff;
}

// Appends new_cycle unless it matches an existing one up to cyclic rotation
// (chordally, within tol).
void add_cycle(std::vector<Cycle>& cycles, std::vector<Cplx> new_cycle, double tol) {
    for (const auto& existing : cycles) {
        if (existing.points.size() != new_cycle.size()) continue;
        const std::size_t n = existing.points.size();
        for (std::size_t shift = 0; shift < n; ++shift) {
            bool all_close = true;
            for (std::size_t i = 0; i < n; ++i) {
                if (chordal(new_cycle[(i + shift) % n], existing.points[i]) >= tol) {
                    all_close = false;
                    break;
                }
            }
            if (all_close) return;   // duplicate, up to rotation
        }
    }
    Cycle c;
    c.points = std::move(new_cycle);
    c.id = static_cast<int>(cycles.size()) + 1;
    cycles.push_back(std::move(c));
}

}  // namespace

std::vector<Cycle> find_attractors(const RationalMap& map, Cplx a,
                                   const FindAttractorsOptions& opts) {
    return find_attractors_from_seeds(map.distinct_critical_points(a), map, a, opts);
}

std::vector<Cycle> find_attractors_from_seeds(const std::vector<Cplx>& seeds,
                                              const RationalMap& map, Cplx a,
                                              const FindAttractorsOptions& opts,
                                              int* unresolved_count) {
    std::vector<Cycle> cycles;
    if (unresolved_count) *unresolved_count = 0;

    for (Cplx seed : seeds) {
        // A seed at infinity can't be evaluated through eval() directly (that
        // formula is for finite z); use a large finite proxy instead, same
        // trick FindAttractors.m uses for its own belt-and-suspenders Inf
        // seed. Ordinary float arithmetic and the inf_cutoff check below take
        // it from there -- no 1/z-chart handling needed for this case.
        Cplx z = is_finite_cplx(seed) ? seed : Cplx(opts.inf_cutoff, 0.0);

        // A finite critical-point seed that IS a pole (a pole of order >= 2
        // is itself a critical point -- see RationalMap::critical_points)
        // can't be evaluated meaningfully AT the pole: eval() returns the
        // escape sentinel there by definition, which would make every such
        // seed look like it converges straight to infinity regardless of
        // whether infinity is actually attracting for this map. Nudge it to
        // a nearby point instead, so the orbit reflects the map's actual
        // (large but finite, and then free to evolve either way) behaviour
        // near the pole rather than the singular value exactly at it.
        if (is_finite_cplx(z)) {
            for (Cplx p : map.pole_locations(a)) {
                if (std::abs(z - p) < 1e-9) { z += Cplx(1e-4, 3.7e-5); break; }
            }
        }

        bool at_inf = false;

        // ---- burn-in: skip the transient ----------------------------------
        for (int n = 0; n < opts.burn_in; ++n) {
            z = map.eval(z, a);
            if (counts_as_infinite(z, opts.inf_cutoff)) { at_inf = true; break; }
        }

        if (at_inf) {
            // A critical orbit exceeding inf_cutoff during burn-in does NOT
            // by itself prove infinity is attracting -- only that this
            // particular finite-length numerical orbit got large. A
            // rational map's behaviour AT infinity can be repelling (e.g.
            // Newton's method: RationalMap::fixed_points reports infinity
            // there with multiplier 3/2) or not even a fixed point at all,
            // in which case a transient excursion past inf_cutoff is a
            // NUMERICAL ARTIFACT (the true orbit would come back down),
            // not genuine convergence. Verify algebraically via
            // fixed_points()' own w=1/z-chart multiplier -- the SAME
            // exact-degree-comparison computation dynamical_facts() already
            // trusts for its own infinity multiplier (see its own comment
            // on why it reuses this instead of re-deriving it) -- rather
            // than assuming. For a genuine polynomial (or any rational map
            // whose numerator degree exceeds its denominator degree by
            // >=2 after clearing denominators) infinity is ALWAYS
            // superattracting there, so this never rejects a real case;
            // it only rejects a spurious one.
            if (opts.verify_multiplier) {
                bool infinity_attracting = false;
                for (const FixedPoint& fp : map.fixed_points(a)) {
                    if (!is_finite_cplx(fp.point) && std::abs(fp.multiplier) < 1.0) {
                        infinity_attracting = true;
                        break;
                    }
                }
                if (!infinity_attracting) {
                    if (unresolved_count) ++*unresolved_count;   // spurious excursion, not a real attractor
                    continue;
                }
            }
            add_cycle(cycles, {Cplx(kInf, 0.0)}, opts.tol);
            continue;
        }

        // ---- detect period: smallest k with f^k(z) ~ z chordally ----------
        std::vector<Cplx> orbit;
        orbit.reserve(static_cast<std::size_t>(opts.max_period) + 1);
        orbit.push_back(z);

        int found = 0;
        bool hit_inf_mid_orbit = false;
        for (int k = 0; k < opts.max_period; ++k) {
            Cplx zn = map.eval(orbit.back(), a);
            const bool zn_inf = counts_as_infinite(zn, opts.inf_cutoff);
            if (zn_inf) zn = Cplx(kInf, 0.0);
            orbit.push_back(zn);

            if (chordal(zn, orbit.front()) < opts.tol) { found = k + 1; break; }
            if (zn_inf) {
                // Passed through Inf without closing. A genuine attracting
                // cycle running through a pole would need the 1/z chart to
                // continue iterating numerically; rare enough to skip rather
                // than silently mishandle (matches FindAttractors.m's own
                // documented scope).
                hit_inf_mid_orbit = true;
                break;
            }
        }

        if (found <= 0 || hit_inf_mid_orbit) {
            if (unresolved_count) ++*unresolved_count;   // no cycle detected, or punted
            continue;
        }

        std::vector<Cplx> cyc(orbit.begin(), orbit.begin() + found);

        // ---- verify it is actually attracting ------------------------------
        if (opts.verify_multiplier) {
            const bool has_inf = std::any_of(cyc.begin(), cyc.end(),
                                             [](Cplx z) { return !is_finite_cplx(z); });
            if (!has_inf) {
                Cplx multiplier(1.0, 0.0);
                for (Cplx zc : cyc) multiplier *= map.deriv(zc, a);
                if (!(std::abs(multiplier) < 1.0)) {
                    if (unresolved_count) ++*unresolved_count;   // not attracting after all
                    continue;
                }
            }
        }

        add_cycle(cycles, std::move(cyc), opts.tol * 1e3);   // looser dedupe tolerance
    }

    return cycles;
}

bool polynomial_escape_certified(const RationalMap& map) {
    for (const auto& t : map.pole_terms()) {
        if (t.enabled) return false;
    }
    for (const auto& t : map.poly_terms()) {
        if (t.enabled && t.exponent < 0) return false;   // implies a pole at the origin too
    }
    return map.degree(Cplx(0.0, 0.0)) >= 2;   // degree() itself never reads its `a` argument
}

// =============================================================================
// 2. wada_diagnostic
// =============================================================================
namespace {

// Binary dilation by a (2r+1)x(2r+1) box, via two separable 1D passes.
std::vector<std::uint8_t> box_dilate(const std::vector<std::uint8_t>& src,
                                     int width, int height, int r) {
    auto idx = [width](int col, int row) { return static_cast<std::size_t>(row) * width + col; };

    std::vector<std::uint8_t> tmp(src.size(), 0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            std::uint8_t v = 0;
            for (int dc = -r; dc <= r && !v; ++dc) {
                const int c = col + dc;
                if (c >= 0 && c < width) v = src[idx(c, row)];
            }
            tmp[idx(col, row)] = v;
        }
    }
    std::vector<std::uint8_t> out(src.size(), 0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            std::uint8_t v = 0;
            for (int dr = -r; dr <= r && !v; ++dr) {
                const int rr = row + dr;
                if (rr >= 0 && rr < height) v = tmp[idx(col, rr)];
            }
            out[idx(col, row)] = v;
        }
    }
    return out;
}

}  // namespace

WadaStats wada_diagnostic(const Image& labels, double radius_fraction) {
    WadaStats stats;
    const int w = labels.width, h = labels.height;
    stats.radius_px = std::max(1, static_cast<int>(std::lround(radius_fraction * std::min(w, h))));

    std::vector<int> distinct_labels;
    for (double v : labels.data) {
        const int lab = static_cast<int>(std::lround(v));
        if (lab > 0 &&
            std::find(distinct_labels.begin(), distinct_labels.end(), lab) == distinct_labels.end()) {
            distinct_labels.push_back(lab);
        }
    }
    std::sort(distinct_labels.begin(), distinct_labels.end());
    const int d = static_cast<int>(distinct_labels.size());
    stats.n_basins = d;

    long n_unresolved = 0;
    for (double v : labels.data)
        if (static_cast<int>(std::lround(v)) == 0) ++n_unresolved;
    stats.unresolved_fraction =
        static_cast<double>(n_unresolved) / static_cast<double>(labels.data.size());

    if (d < 2) {
        stats.boundary_fraction = 0.0;
        stats.wada_fraction = std::numeric_limits<double>::quiet_NaN();
        return stats;
    }

    std::vector<std::vector<std::uint8_t>> near_label(static_cast<std::size_t>(d));
    for (int k = 0; k < d; ++k) {
        std::vector<std::uint8_t> mask(labels.data.size(), 0);
        for (std::size_t i = 0; i < labels.data.size(); ++i) {
            mask[i] = static_cast<int>(std::lround(labels.data[i])) == distinct_labels[static_cast<std::size_t>(k)]
                          ? 1 : 0;
        }
        near_label[static_cast<std::size_t>(k)] = box_dilate(mask, w, h, stats.radius_px);
    }

    long n_boundary = 0, n_wada = 0;
    for (std::size_t i = 0; i < labels.data.size(); ++i) {
        int labels_nearby = 0;
        for (int k = 0; k < d; ++k) labels_nearby += near_label[static_cast<std::size_t>(k)][i];
        if (labels_nearby >= 2) {
            ++n_boundary;
            if (labels_nearby == d) ++n_wada;
        }
    }
    stats.boundary_fraction = static_cast<double>(n_boundary) / static_cast<double>(labels.data.size());
    stats.wada_fraction = n_boundary == 0
                              ? std::numeric_limits<double>::quiet_NaN()
                              : static_cast<double>(n_wada) / static_cast<double>(n_boundary);
    return stats;
}

// =============================================================================
// 3. hausdorff_distance
// =============================================================================
namespace {

std::vector<Cplx> subsample(const std::vector<Cplx>& v, std::size_t max_points) {
    if (v.size() <= max_points || max_points == 0) return v;
    std::vector<Cplx> out;
    out.reserve(max_points);
    for (std::size_t i = 0; i < max_points; ++i) {
        const std::size_t idx = max_points <= 1
            ? 0
            : static_cast<std::size_t>(std::llround(
                  static_cast<double>(i) * static_cast<double>(v.size() - 1) /
                  static_cast<double>(max_points - 1)));
        out.push_back(v[idx]);
    }
    return out;
}

struct Directed { double chordal; double euclidean; };

// sup_{p in a} inf_{q in b} d(p,q), both metrics computed together since
// each point's nearest neighbour can differ between the two metrics.
Directed directed_hausdorff(const std::vector<Cplx>& a, const std::vector<Cplx>& b) {
    double dc = 0.0, de = 0.0;
    for (Cplx p : a) {
        double min_c = std::numeric_limits<double>::infinity();
        double min_e = std::numeric_limits<double>::infinity();
        for (Cplx q : b) {
            min_c = std::min(min_c, chordal(p, q));
            min_e = std::min(min_e, std::abs(p - q));
        }
        dc = std::max(dc, min_c);
        de = std::max(de, min_e);
    }
    return {dc, de};
}

}  // namespace

HausdorffResult hausdorff_distance(const std::vector<Cplx>& julia_points,
                                   const std::vector<Cplx>& target_points,
                                   std::size_t max_points) {
    const std::vector<Cplx> j = subsample(julia_points, max_points);
    const std::vector<Cplx> t = subsample(target_points, max_points);

    HausdorffResult r;
    if (j.empty() || t.empty()) {
        // An empty side makes the Hausdorff distance undefined (there is
        // nothing to be close to); report infinity rather than 0, which
        // would silently read as "a perfect match".
        r.chordal = r.euclidean = kInf;
        r.chordal_julia_to_target = r.chordal_target_to_julia = kInf;
        r.euclidean_julia_to_target = r.euclidean_target_to_julia = kInf;
        return r;
    }

    const Directed j2t = directed_hausdorff(j, t);
    const Directed t2j = directed_hausdorff(t, j);

    r.chordal_julia_to_target = j2t.chordal;
    r.chordal_target_to_julia = t2j.chordal;
    r.euclidean_julia_to_target = j2t.euclidean;
    r.euclidean_target_to_julia = t2j.euclidean;
    r.chordal = std::max(j2t.chordal, t2j.chordal);
    r.euclidean = std::max(j2t.euclidean, t2j.euclidean);
    return r;
}

std::vector<Cplx> extract_boundary_points(const Image& labels, const Viewport& view) {
    std::vector<Cplx> out;
    const int w = labels.width, h = labels.height;
    auto lab = [&](int col, int row) { return static_cast<int>(std::lround(labels.at(col, row))); };

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            const int L = lab(col, row);
            if (L <= 0) continue;
            bool boundary = false;
            if (col > 0)         { const int n = lab(col - 1, row); if (n > 0 && n != L) boundary = true; }
            if (!boundary && col < w - 1) { const int n = lab(col + 1, row); if (n > 0 && n != L) boundary = true; }
            if (!boundary && row > 0)     { const int n = lab(col, row - 1); if (n > 0 && n != L) boundary = true; }
            if (!boundary && row < h - 1) { const int n = lab(col, row + 1); if (n > 0 && n != L) boundary = true; }
            if (boundary) out.push_back(view.coord(col, row));
        }
    }
    return out;
}

// =============================================================================
// 4. dynamical_facts
// =============================================================================
DynamicalFacts dynamical_facts(const RationalMap& map, Cplx a, const FindAttractorsOptions& opts) {
    DynamicalFacts facts;
    facts.degree = map.degree(a);
    facts.critical_points = map.critical_points(a);
    facts.pole_locations = map.pole_locations(a);
    facts.pole_orders = map.pole_orders(a);
    facts.fixed_points = map.fixed_points(a);

    for (const auto& cyc : find_attractors(map, a, opts)) {
        DynamicalFacts::AttractingCycle ac;
        ac.points = cyc.points;
        ac.period = static_cast<int>(cyc.points.size());

        // find_attractors only ever returns an infinity-containing cycle as
        // the sole point of a period-1 {Inf} cycle (an orbit passing
        // through infinity MID-cycle is detected and skipped there, not
        // returned) -- so the only Inf case to handle is this one, and its
        // multiplier is already computed correctly by fixed_points() via
        // the w=1/z chart. Reuse it instead of re-deriving it.
        if (ac.points.size() == 1 && !is_finite_cplx(ac.points[0])) {
            for (const auto& fp : facts.fixed_points) {
                if (!is_finite_cplx(fp.point)) { ac.multiplier = fp.multiplier; break; }
            }
        } else {
            Cplx multiplier(1.0, 0.0);
            for (Cplx z : ac.points) multiplier *= map.deriv(z, a);
            ac.multiplier = multiplier;
        }

        facts.attracting_cycles.push_back(ac);
    }

    return facts;
}

}  // namespace cdx
