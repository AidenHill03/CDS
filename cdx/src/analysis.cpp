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

// TRUE if z is already chordally indistinguishable from infinity at
// (a safety margin above) the given tol -- i.e. its chordal distance to the
// north pole is itself comparable to the tolerance a closure test would
// use. Two points that are BOTH this close to infinity are automatically
// chordally close to EACH OTHER too (triangle inequality), no matter how
// different they are as raw numbers or whether the orbit is remotely
// periodic -- the sphere compactifies arbitrarily large values together.
// A transient excursion near a pole can pass through this zone (large but
// still under inf_cutoff) while still very much diverging, not settling;
// the OLD, always-500-steps burn-in silently outlasted it before ever
// attempting a closure comparison. The interleaved settle-and-detect loop
// checks closure far earlier, so it must exclude this zone explicitly
// instead of relying on brute iteration count to avoid it by accident.
// margin_factor gives headroom above the raw tolerance so the guard itself
// can't become the source of a missed (or falsely accepted) detection.
bool near_infinity_chordally(Cplx z, double tol) {
    return chordal(z, Cplx(std::numeric_limits<double>::infinity(), 0.0)) < tol * 100.0;
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

// Attempts to confirm a WEAKLY ATTRACTING cycle after the strict closure
// pass has already failed for this orbit -- see find_attractors_from_seeds'
// own doc comment for the full reasoning. `orbit` is the still-live,
// post-burn-in orbit accumulated so far (orbit.front() is z_0; this
// function CONTINUES iterating it, extending `orbit` in place, rather than
// restarting). Returns the refined cycle's points on success, or an empty
// vector if no weakly-attracting cycle could be confirmed within budget --
// callers must not mistake an empty return for "period zero", only for
// "nothing to add here".
std::vector<Cplx> confirm_weakly_attracting(std::vector<Cplx>& orbit, const RationalMap& map,
                                            Cplx a, const FindAttractorsOptions& opts) {
    if (!opts.verify_multiplier || !opts.confirm_weakly_attracting) return {};

    // ---- settle further, re-anchoring progressively instead of waiting for --
    // ---- the full extended_max_period budget before ever looking ------------
    // Same mechanism as the strict pass' own settle loop below (see its own
    // comment for the full reasoning): try establishing an anchor and
    // searching up to `max_period` steps forward from it -- EXACTLY the old
    // algorithm's own single-anchor, fixed-budget search, just possibly
    // triggered from an earlier point. If a trial fails, its own steps
    // become the settling progress toward the NEXT, later anchor -- no
    // wasted work -- with the FINAL trial always landing exactly at
    // `extended_max_period`, replicating the old algorithm's own one-shot
    // trial exactly as a safety net. This has IDENTICAL detection power to
    // the old fixed-budget search at every trial (same single-anchor
    // comparison, same tolerance, same per-trial budget): an earlier trial
    // can only succeed if the orbit is ALREADY close enough to a genuine
    // periodic point for that same strict test to pass, so it can never
    // find something the old algorithm's own final trial wouldn't also
    // confirm -- unlike comparing against a WINDOW of several different
    // prior points, which is a strictly more permissive (and therefore
    // WRONG for a pure speed change) test.
    std::size_t anchor_idx = 0;
    bool anchor_established = false;
    int current_anchor_step = 0;
    const int first_anchor_step = std::min(opts.max_period, opts.extended_max_period);
    int loose_found = 0;

    const int total_budget = opts.extended_max_period + opts.max_period;
    for (int n = 0; n < total_budget; ++n) {
        Cplx zn = map.eval(orbit.back(), a);
        if (counts_as_infinite(zn, opts.inf_cutoff)) return {};
        orbit.push_back(zn);
        const int steps_done = n + 1;

        if (!anchor_established && steps_done >= first_anchor_step) {
            anchor_idx = orbit.size() - 1;
            current_anchor_step = steps_done;
            anchor_established = true;
            continue;
        }

        if (anchor_established && !near_infinity_chordally(zn, opts.loose_tol) &&
            !near_infinity_chordally(orbit[anchor_idx], opts.loose_tol) &&
            chordal(zn, orbit[anchor_idx]) < opts.loose_tol) {
            const int k = static_cast<int>(orbit.size() - 1 - anchor_idx);
            bool accept = true;
            if (current_anchor_step < opts.extended_max_period) {
                // Same speculative-anchor restriction as the strict pass
                // (see its own, fuller comment): a slowly-converging
                // MULTI-POINT cycle's residual doesn't shrink monotonically
                // as it settles, so an early anchor can land on a lucky
                // phase and spuriously "confirm" a period the old
                // algorithm's own single, fully-converged anchor genuinely
                // does not -- not just find the SAME thing faster, which
                // is the one thing this optimization must never do. k>1 is
                // therefore trusted ONLY at the FINAL trial, matching old
                // code's own single-check acceptance there exactly. k==1
                // remains safe with one extra confirmation step.
                if (k == 1) {
                    Cplx zc = map.eval(zn, a);
                    accept = !counts_as_infinite(zc, opts.inf_cutoff) && chordal(zc, zn) < opts.loose_tol;
                } else {
                    accept = false;
                }
            }
            if (accept) {
                loose_found = k;
                break;
            }
            // else: not (yet) trustworthy -- keep scanning larger k
            // against the SAME anchor.
        }
        if (anchor_established) {
            // Same overshoot guard as the strict pass (see its own
            // comment): cap a speculative trial's window so the final
            // (extended_max_period) trial is always reached exactly, never
            // skipped past.
            const int trial_end = current_anchor_step >= opts.extended_max_period
                ? current_anchor_step + opts.max_period
                : std::min(current_anchor_step + opts.max_period, opts.extended_max_period);
            if (steps_done >= trial_end) {
                if (current_anchor_step >= opts.extended_max_period) break;   // final trial exhausted
                // Re-anchor IMMEDIATELY at the current point -- see the
                // strict pass' own comment for why deferring to the next
                // iteration's establishment check drifts the final anchor
                // away from `extended_max_period` by one step per trial.
                anchor_idx = orbit.size() - 1;
                current_anchor_step = steps_done;
            }
        }
    }
    if (loose_found <= 0) return {};
    Cplx z0 = orbit[anchor_idx];

    // ---- Newton-polish the candidate toward the TRUE periodic point ---------
    // g(z) = f^p(z) - z; g'(z) = (f^p)'(z) - 1, the chain-rule product of
    // deriv() over the p points -- the SAME quantity the multiplier check
    // itself needs, computed once per Newton step rather than derived
    // separately.
    for (int iter = 0; iter < opts.newton_iterations; ++iter) {
        Cplx zk = z0;
        Cplx deriv_prod(1.0, 0.0);
        bool ok = true;
        for (int i = 0; i < loose_found; ++i) {
            if (!is_finite_cplx(zk)) { ok = false; break; }
            deriv_prod *= map.deriv(zk, a);
            zk = map.eval(zk, a);
        }
        if (!ok || !is_finite_cplx(zk)) return {};
        const Cplx gprime = deriv_prod - Cplx(1.0, 0.0);
        // A degenerate (near-zero) derivative here means the candidate is
        // near-PARABOLIC (multiplier near 1) -- Newton's own method is
        // unreliable there, and a parabolic cycle isn't attracting anyway,
        // so bailing out (leaving it correctly unresolved) is the right
        // answer, not a missed case.
        if (std::abs(gprime) < 1e-9) return {};
        z0 -= (zk - z0) / gprime;
        if (!is_finite_cplx(z0)) return {};
    }

    // ---- recompute the EXACT cycle points at the refined point ---------------
    std::vector<Cplx> refined;
    refined.reserve(static_cast<std::size_t>(loose_found));
    Cplx zk = z0;
    for (int i = 0; i < loose_found; ++i) {
        if (!is_finite_cplx(zk)) return {};
        refined.push_back(zk);
        zk = map.eval(zk, a);
    }
    // The refined point must genuinely close (f^p(z0) ~= z0) at a tolerance
    // much tighter than the LOOSE one that found the candidate in the first
    // place -- confirms Newton's polish actually converged to a real
    // periodic point, not a spurious nearby root of the linearization.
    if (chordal(zk, z0) >= opts.tol * 1e2) return {};

    // See attracting_margin's own doc comment for why this needs a safety
    // margin below 1.0 (not just < 1.0 outright): Newton's method's own
    // linear (not quadratic) convergence at a genuinely parabolic
    // candidate leaves a small floating-point residual that would
    // otherwise read as "just barely attracting".
    Cplx multiplier(1.0, 0.0);
    for (Cplx zc : refined) multiplier *= map.deriv(zc, a);
    if (!(std::abs(multiplier) < 1.0 - opts.attracting_margin)) return {};

    return refined;
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
        bool hit_inf_mid_orbit = false;

        // ---- settle onto the cycle, re-anchoring progressively -------------
        // Old approach: burn `burn_in` steps unconditionally to eliminate
        // the transient, THEN search up to `max_period` more steps for
        // closure against the single point burn-in happened to leave off
        // at. That means every non-escaping seed pays the full `burn_in`
        // cost even when the orbit actually settles in a fraction of that
        // -- the dominant per-pixel cost for an ordinary (not weakly
        // attracting) interior parameter.
        //
        // This tries establishing the anchor EARLIER, at `max_period`,
        // `2*max_period`, ... and running the OLD algorithm's own single-
        // anchor, fixed-budget forward search from each -- stopping the
        // instant one succeeds. A failed trial's own steps become the
        // settling progress toward the next, later anchor (no wasted
        // work), and the FINAL trial always lands exactly at `burn_in`,
        // replicating the old algorithm's own one-shot check exactly as a
        // safety net. This has IDENTICAL detection power to the old
        // fixed-budget search at every trial (same single-anchor
        // comparison, same tolerance, same per-trial budget): an earlier
        // trial can only succeed if the orbit is ALREADY close enough to a
        // genuine periodic point for that SAME strict test to pass, so it
        // can never find something the old algorithm's own final trial
        // wouldn't also confirm -- unlike comparing against a WINDOW of
        // several different prior points (an earlier version of this
        // change did exactly that), which is a strictly more permissive
        // test and therefore WRONG for a pure speed change: it could
        // resolve pixels the old algorithm's own single check genuinely
        // couldn't, not just resolve the SAME ones faster.
        //
        // The escape-to-infinity classification below preserves the exact
        // same step-count boundary the old two-phase split had: an
        // excursion within the first `burn_in` evals is still treated as a
        // possibly-spurious transient (verified against the algebraic
        // oracle just below), a LATER excursion is still
        // `hit_inf_mid_orbit` -- unresolved, no oracle check -- since by
        // that point the orbit was presumed already close to a cycle, same
        // as before. Only WHEN closure can be detected has changed, not
        // this classification.
        std::vector<Cplx> orbit;
        orbit.reserve(static_cast<std::size_t>(opts.burn_in + opts.max_period) + 1);
        orbit.push_back(z);

        std::size_t anchor_idx = 0;
        bool anchor_established = false;
        int current_anchor_step = 0;
        const int first_anchor_step = std::min(opts.max_period, opts.burn_in);
        int found = 0;

        const int settle_budget = opts.burn_in + opts.max_period;
        for (int n = 0; n < settle_budget; ++n) {
            Cplx zn = map.eval(orbit.back(), a);
            const bool zn_inf = counts_as_infinite(zn, opts.inf_cutoff);
            if (zn_inf) {
                if (n < opts.burn_in) at_inf = true;
                else hit_inf_mid_orbit = true;
                break;
            }
            orbit.push_back(zn);
            const int steps_done = n + 1;

            if (!anchor_established && steps_done >= first_anchor_step) {
                anchor_idx = orbit.size() - 1;
                current_anchor_step = steps_done;
                anchor_established = true;
                continue;   // don't compare the anchor to itself
            }

            if (anchor_established && !near_infinity_chordally(zn, opts.tol) &&
                !near_infinity_chordally(orbit[anchor_idx], opts.tol) &&
                chordal(zn, orbit[anchor_idx]) < opts.tol) {
                const int k = static_cast<int>(orbit.size() - 1 - anchor_idx);
                bool accept = true;
                if (current_anchor_step < opts.burn_in) {
                    // A SPECULATIVE (earlier-than-burn_in) anchor may not
                    // have fully settled yet. For a genuine FIXED point
                    // (k==1) this is safe with one extra check: CONFIRM
                    // f(zn)~zn too, guarding against a still-spiralling
                    // orbit (a multiplier with a rotational component)
                    // passing transiently near an under-converged anchor
                    // without being genuinely periodic -- a risk old
                    // code's own generous, always-`burn_in` anchor never
                    // runs into (by then the spiral radius is negligible).
                    //
                    // For k>1, confirmation is NOT enough to trust: a
                    // slowly-converging MULTI-POINT cycle's residual
                    // distance to itself does not shrink monotonically as
                    // it settles (it oscillates while the overall envelope
                    // shrinks), so an early anchor can land on a lucky
                    // phase where ITS OWN residual dips under `tol` even
                    // though the SAME point, compared at the exact
                    // burn_in anchor, would not -- and that luck can
                    // recur for several periods, defeating even repeated
                    // confirmation (found and root-caused with an actual
                    // period-9 Nova cycle: the old algorithm's single
                    // burn_in-anchor trial genuinely fails to resolve it,
                    // while an earlier speculative anchor spuriously
                    // "succeeds"). So k>1 is trusted ONLY at the FINAL
                    // (burn_in) trial, replicating old code exactly there
                    // -- no confirmation needed at that trial either,
                    // matching old code's own unconfirmed acceptance.
                    if (k == 1) {
                        Cplx zc = map.eval(zn, a);
                        accept = !counts_as_infinite(zc, opts.inf_cutoff) && chordal(zc, zn) < opts.tol;
                    } else {
                        accept = false;
                    }
                }
                if (accept) {
                    found = k;
                    break;
                }
                // else: not (yet) trustworthy -- keep scanning larger k
                // against the SAME anchor rather than re-anchoring
                // immediately.
            }
            if (anchor_established) {
                // A speculative trial's own max_period-sized search window
                // must never overshoot burn_in: if it did, giving up on it
                // would land the NEXT anchor somewhere PAST burn_in
                // instead of exactly there, drifting away from old code's
                // own exact final-anchor position (the guarantee this
                // whole design depends on). Cap this trial's window so the
                // final (burn_in) trial is always reached cleanly, with no
                // overshoot.
                const int trial_end = current_anchor_step >= opts.burn_in
                    ? current_anchor_step + opts.max_period
                    : std::min(current_anchor_step + opts.max_period, opts.burn_in);
                if (steps_done >= trial_end) {
                    if (current_anchor_step >= opts.burn_in) break;   // final trial exhausted
                    // Re-anchor IMMEDIATELY at the CURRENT point (steps_done
                    // == trial_end exactly here) rather than deferring to
                    // the next iteration's own establishment check -- that
                    // check only fires on evals AFTER this one, landing the
                    // new anchor one step late every single time, which
                    // compounds across trials into a final anchor that has
                    // drifted away from `burn_in` entirely (a real bug this
                    // design hit and had to be fixed: the drift caused a
                    // genuine period-11 cycle to be missed at anchor 501
                    // when old code's own exact anchor at 500 finds it).
                    anchor_idx = orbit.size() - 1;
                    current_anchor_step = steps_done;
                }
            }
        }
        if (found > 0) {
            std::vector<Cplx> cyc(orbit.begin() + static_cast<std::ptrdiff_t>(anchor_idx),
                                  orbit.begin() + static_cast<std::ptrdiff_t>(anchor_idx) + found);
            orbit = std::move(cyc);
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

        if (found <= 0 || hit_inf_mid_orbit) {
            // The strict pass failed to close -- before giving up, see
            // whether this orbit is actually converging (slowly) toward a
            // genuinely attracting cycle the strict tol/period budget just
            // didn't catch in time (see confirm_weakly_attracting's own
            // doc comment). Not attempted after hit_inf_mid_orbit: the
            // orbit has already run off past inf_cutoff there, so there is
            // no live finite state left to continue iterating.
            std::vector<Cplx> refined =
                !hit_inf_mid_orbit ? confirm_weakly_attracting(orbit, map, a, opts)
                                   : std::vector<Cplx>{};
            if (!refined.empty()) {
                add_cycle(cycles, std::move(refined), opts.tol * 1e3);
                continue;
            }
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
    // Representation-agnostic (Stage 2 of the P/Q milestone): works
    // identically whether `map` is term-based or P/Q-backed, since it goes
    // through RationalMap::is_polynomial_structurally() rather than
    // reading poly_terms()/pole_terms() directly -- a P/Q-backed map has
    // no term list to read. See that method's own doc comment for why the
    // check has to stay structural (no poles for ANY parameter value, not
    // just the one degree() happens to be evaluated at).
    return map.is_polynomial_structurally();
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
