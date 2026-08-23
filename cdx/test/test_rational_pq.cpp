// =============================================================================
// test_rational_pq.cpp -- P/Q-backed RationalMap satisfies the FULL
// RationalMap interface: equivalence against the term-built map for the
// SAME family, through every method AND every render/analysis consumer.
//
// Each case below builds the SAME mathematical map two ways -- once via
// RationalMap::from_canonical(parse_rational(...)) (P/Q-backed), once via
// the existing term-based preset factory or hand-built terms -- and checks
// that every consumer this codebase has cannot tell the difference.
// =============================================================================
#include "cdx/analysis.hpp"
#include "cdx/rational_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

using namespace cdx;

static int failures = 0;

static void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++failures;
}

static bool close(Cplx a, Cplx b, double tol = 1e-6) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) < tol * scale;
}

static RationalMap build_pq(const std::string& src) {
    CanonicalRational cr;
    std::string error;
    if (!parse_rational(src, cr, error)) {
        std::printf("  [FATAL] parse_rational('%s') failed: %s\n", src.c_str(), error.c_str());
        std::exit(1);
    }
    return RationalMap::from_canonical(std::move(cr), "pq");
}

// Matches a set of Cplx points to another set, order-independent, within
// tol -- used for critical/fixed points, where the two representations are
// not guaranteed to return them in the same order.
static bool same_point_set(std::vector<Cplx> a, std::vector<Cplx> b, double tol = 1e-4) {
    if (a.size() != b.size()) return false;
    std::vector<bool> used(b.size(), false);
    for (Cplx pa : a) {
        bool found = false;
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (used[j]) continue;
            const bool a_inf = std::isinf(pa.real()) || std::isinf(pa.imag());
            const bool b_inf = std::isinf(b[j].real()) || std::isinf(b[j].imag());
            if (a_inf != b_inf) continue;
            if (a_inf ? true : close(pa, b[j], tol)) { used[j] = true; found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Full method-by-method + render-by-render equivalence between a P/Q-backed
// map and its term-built counterpart, at the SAME parameter `a`.
// -----------------------------------------------------------------------------
static void check_equivalent(const char* label, const RationalMap& pq, const RationalMap& term,
                             Cplx a, const Viewport& v) {
    std::printf("\n%s (a=%.4f%+.4fi):\n", label, a.real(), a.imag());

    check(pq.is_pq_backed(), "sanity: the P/Q map is actually P/Q-backed");
    check(!term.is_pq_backed(), "sanity: the reference map is actually term-based");

    // ---- eval/deriv at many random z -----------------------------------------
    {
        std::mt19937 rng(777);
        std::uniform_real_distribution<double> unif(-2.5, 2.5);
        int n_checked = 0, n_eval_mismatch = 0, n_deriv_mismatch = 0;
        for (int i = 0; i < 100; ++i) {
            const Cplx z{unif(rng), unif(rng)};
            const Cplx e1 = pq.eval(z, a), e2 = term.eval(z, a);
            const Cplx d1 = pq.deriv(z, a), d2 = term.deriv(z, a);
            if (std::abs(e1) > 1e8 || std::abs(e2) > 1e8) continue;   // near a pole
            ++n_checked;
            if (!close(e1, e2)) ++n_eval_mismatch;
            if (!close(d1, d2, 1e-4)) ++n_deriv_mismatch;
        }
        check(n_checked > 50, "sanity: most sample points were usable");
        check(n_eval_mismatch == 0, "eval(z,a) matches at every usable sample point");
        check(n_deriv_mismatch == 0, "deriv(z,a) matches at every usable sample point");
    }

    // ---- compile()+step() matches eval() for the SAME orbit ------------------
    {
        CompiledMap c = pq.compile(a);
        double zr = 0.3, zi = 0.2;
        Cplx z{0.3, 0.2};
        bool all_match = true;
        for (int i = 0; i < 200; ++i) {
            c.step(zr, zi);
            z = pq.eval(z, a);
            if (std::abs(z) > 1e8) break;   // escaped -- both sides sentinel from here
            if (!close(Cplx(zr, zi), z, 1e-6)) { all_match = false; break; }
        }
        check(all_match, "compile()+step() computes the SAME orbit as eval() (P/Q's own Horner "
              "fast path)");
    }

    // ---- structure: degree, pole locations/orders -----------------------------
    check(pq.degree(a) == term.degree(a), "degree(a) matches");
    {
        auto pl_pq = pq.pole_locations(a), pl_term = term.pole_locations(a);
        check(same_point_set(pl_pq, pl_term), "pole_locations(a) matches (as a set)");
        // pole_orders' contract is index-parallel to pole_locations' OWN
        // ordering, which can differ between the two representations --
        // compare orders by MATCHING location, not by index.
        auto po_pq = pq.pole_orders(a), po_term = term.pole_orders(a);
        bool orders_match = pl_pq.size() == po_pq.size() && pl_term.size() == po_term.size();
        if (orders_match) {
            for (std::size_t i = 0; i < pl_pq.size() && orders_match; ++i) {
                bool found = false;
                for (std::size_t j = 0; j < pl_term.size(); ++j) {
                    if (close(pl_pq[i], pl_term[j], 1e-4)) {
                        if (po_pq[i] != po_term[j]) orders_match = false;
                        found = true;
                        break;
                    }
                }
                if (!found) orders_match = false;
            }
        }
        check(orders_match, "pole_orders(a) matches at each shared pole location");
    }

    // ---- critical points, fixed points -----------------------------------------
    {
        auto cp_pq = pq.distinct_critical_points(a), cp_term = term.distinct_critical_points(a);
        check(same_point_set(cp_pq, cp_term), "distinct_critical_points(a) matches (as a set)");

        auto fp_pq = pq.fixed_points(a), fp_term = term.fixed_points(a);
        check(fp_pq.size() == fp_term.size(), "fixed_points(a) count matches");
        bool fp_match = fp_pq.size() == fp_term.size();
        if (fp_match) {
            std::vector<bool> used(fp_term.size(), false);
            for (const auto& p : fp_pq) {
                bool found = false;
                for (std::size_t j = 0; j < fp_term.size(); ++j) {
                    if (used[j]) continue;
                    const bool p_inf = std::isinf(p.point.real());
                    const bool q_inf = std::isinf(fp_term[j].point.real());
                    if (p_inf != q_inf) continue;
                    const bool point_ok = p_inf || close(p.point, fp_term[j].point, 1e-4);
                    const bool mult_ok = close(p.multiplier, fp_term[j].multiplier, 1e-4);
                    if (point_ok && mult_ok) { used[j] = true; found = true; break; }
                }
                if (!found) { fp_match = false; break; }
            }
        }
        check(fp_match, "fixed_points(a) matches (points AND multipliers, as a set)");
    }

    // ---- find_attractors_from_seeds --------------------------------------------
    {
        auto seeds_pq = pq.distinct_critical_points(a);
        auto seeds_term = term.distinct_critical_points(a);
        auto cyc_pq = find_attractors_from_seeds(seeds_pq, pq, a);
        auto cyc_term = find_attractors_from_seeds(seeds_term, term, a);
        check(cyc_pq.size() == cyc_term.size(),
              "find_attractors_from_seeds: same attracting-cycle count");
    }

    // ---- render consumers: basin, Julia, Green's, Parameter_basin --------------
    {
        RenderSettings settings{100, 2.0, 1e-6, 1};
        Renderer r_pq(Map::custom(pq, a), v, settings);
        Renderer r_term(Map::custom(term, a), v, settings);

        auto cyc_pq = find_attractors(pq, a);
        auto cyc_term = find_attractors(term, a);

        Image labels_pq, labels_term;
        const Image jvals_pq = r_pq.render_julia(nullptr, cyc_pq, &labels_pq);
        const Image jvals_term = r_term.render_julia(nullptr, cyc_term, &labels_term);
        bool julia_close = jvals_pq.data.size() == jvals_term.data.size();
        int n_julia_diff = 0;
        if (julia_close) {
            for (std::size_t i = 0; i < jvals_pq.data.size(); ++i) {
                if (std::abs(jvals_pq.data[i] - jvals_term.data[i]) > 1e-3) ++n_julia_diff;
            }
        }
        check(julia_close && n_julia_diff < static_cast<int>(jvals_pq.data.size()) / 20,
              "render_julia: values match (allowing a thin boundary-pixel margin, "
              "<5% differing, for classification noise right at a Julia-set boundary)");

        Image basin_pq = r_pq.render_basin(cyc_pq);
        Image basin_term = r_term.render_basin(cyc_term);
        int n_resolved_pq = 0, n_resolved_term = 0;
        for (double v2 : basin_pq.data) if (v2 > 0.0) ++n_resolved_pq;
        for (double v2 : basin_term.data) if (v2 > 0.0) ++n_resolved_term;
        check(std::abs(n_resolved_pq - n_resolved_term) < static_cast<int>(basin_pq.data.size()) / 20,
              "render_basin: nearly the same number of resolved pixels");

        const Image g_pq = r_pq.render_greens(nullptr, cyc_pq);
        const Image g_term = r_term.render_greens(nullptr, cyc_term);
        int n_greens_diff = 0;
        for (std::size_t i = 0; i < g_pq.data.size(); ++i) {
            if (std::abs(g_pq.data[i] - g_term.data[i]) > 1e-3) ++n_greens_diff;
        }
        check(n_greens_diff < static_cast<int>(g_pq.data.size()) / 20,
              "render_greens: values match (same boundary-noise allowance)");

        if (pq.distinct_critical_points(a).size() >= 1) {
            const Image pb_pq = r_pq.render_parameter_basin();
            const Image pb_term = r_term.render_parameter_basin();
            int n_pb_diff = 0;
            for (std::size_t i = 0; i < pb_pq.data.size(); ++i) {
                if (pb_pq.data[i] != pb_term.data[i]) ++n_pb_diff;
            }
            check(n_pb_diff < static_cast<int>(pb_pq.data.size()) / 20,
                  "render_parameter_basin: per-pixel attracting-cycle counts match "
                  "(same boundary-noise allowance)");
        }
    }

    // ---- escape-radius invariance for the P/Q rational path --------------------
    if (!pq.is_polynomial_structurally()) {
        auto cyc_pq = find_attractors(pq, a);
        Renderer r2(Map::custom(pq, a), v, RenderSettings{100, 2.0, 1e-6, 1});
        Renderer r10(Map::custom(pq, a), v, RenderSettings{100, 10.0, 1e-6, 1});
        Image labels2, labels10;
        const Image vals2 = r2.render_julia(nullptr, cyc_pq, &labels2);
        const Image vals10 = r10.render_julia(nullptr, cyc_pq, &labels10);
        bool identical = true;
        for (std::size_t i = 0; i < vals2.data.size(); ++i) {
            if (vals2.data[i] != vals10.data[i] || labels2.data[i] != labels10.data[i]) {
                identical = false;
                break;
            }
        }
        check(identical, "ESCAPE-RADIUS INVARIANCE: a P/Q-backed RATIONAL map's render_julia is "
              "BYTE-IDENTICAL at escape_radius=2 vs 10 (the rational path never reads it)");
    } else {
        check(pq.is_polynomial_structurally() == term.is_polynomial_structurally(),
              "sanity: both representations agree this map is structurally polynomial");
    }
}

int main() {
    std::printf("=== cdx P/Q-backed RationalMap equivalence tests ===\n");

    // ---- z^2 + a: certified polynomial, single parameter ------------------------
    {
        RationalMap pq = build_pq("z^2 + a");
        check_equivalent("z^2 + a  (mandelbrot)", pq, RationalMap::mandelbrot(),
                         Cplx{-0.75, 0.1}, Viewport{{0.0, 0.0}, 2.0, 61});
        check(pq.is_polynomial_structurally(), "z^2+a: recognized as structurally polynomial "
              "(degree_certified >= 2, no poles) via the SAME predicate the term-based path "
              "uses, from the P/Q side");
        // ACCEPTANCE: a P/Q polynomial takes the certified fast path and
        // matches the term-built polynomial's own render exactly.
        Renderer r_pq(Map::custom(pq, Cplx{-0.75, 0.1}),
                     Viewport{{0.0, 0.0}, 1.5, 61}, RenderSettings{100, 2.0, 1e-6, 1});
        Renderer r_term(Map::custom(RationalMap::mandelbrot(), Cplx{-0.75, 0.1}),
                        Viewport{{0.0, 0.0}, 1.5, 61}, RenderSettings{100, 2.0, 1e-6, 1});
        const Image jp = r_pq.render_julia();
        const Image jt = r_term.render_julia();
        bool byte_identical = jp.data.size() == jt.data.size();
        if (byte_identical) {
            for (std::size_t i = 0; i < jp.data.size(); ++i) {
                if (jp.data[i] != jt.data[i]) { byte_identical = false; break; }
            }
        }
        check(byte_identical, "ACCEPTANCE: the P/Q-built z^2+a's render_julia is BYTE-IDENTICAL "
              "to the term-built mandelbrot()'s -- the certified polynomial fast path (which "
              "reads only degree()/eval(), never terms) genuinely doesn't care which "
              "representation produced them");
    }

    // ---- z^3 + a/z^3: rational, single parameter --------------------------------
    {
        RationalMap pq = build_pq("z^3 + a/z^3");
        check(!pq.is_polynomial_structurally(), "z^3+a/z^3: correctly NOT structurally "
              "polynomial (has a pole) via the P/Q predicate");
        check_equivalent("z^3 + a/z^3  (mcmullen3)", pq, RationalMap::mcmullen(3),
                         Cplx{0.3, -0.2}, Viewport{{0.0, 0.0}, 2.0, 61});
    }

    // ---- serialize()/deserialize() basic round trip (full authored-form -----
    // fidelity -- multiple parameters, which is active -- is Stage 3's job;
    // this just confirms a single-parameter P/Q map survives at all) ---------
    std::printf("\nserialize()/deserialize(): basic round trip:\n");
    {
        RationalMap pq = build_pq("z^3 + a/z^3");
        const std::string blob = pq.serialize();
        RationalMap loaded;
        std::string error;
        check(RationalMap::deserialize(blob, loaded, error), "a P/Q-backed map's serialize() "
              "output deserializes without error");
        check(loaded.is_pq_backed(), "...and comes back P/Q-backed, not term-based");
        check(loaded.pq_source() == pq.pq_source(), "...with the SAME authored source text");
        const Cplx a{0.4, -0.1};
        check(close(loaded.eval(Cplx{0.7, 0.3}, a), pq.eval(Cplx{0.7, 0.3}, a)),
              "...and computes identically to the original after the round trip");
    }

    // ---- Newton's method for z^3-1: rational, NO free parameter -----------------
    {
        RationalMap pq = build_pq("z - (z^3-1)/(3*z^2)");
        check(pq.pq_source() == "z - (z^3-1)/(3*z^2)", "pq_source() returns the authored text "
              "verbatim");
        check(pq.to_formula() == pq.pq_source(), "to_formula() IS pq_source() for a P/Q-backed "
              "map -- the user's own text, not a reconstruction");
        check_equivalent("z - (z^3-1)/(3*z^2)  (newton_cubic)", pq, RationalMap::newton_cubic(),
                         Cplx{0.0, 0.0}, Viewport{{0.0, 0.0}, 2.0, 61});
    }

    // ---- add_pole_at/add_zero_at: forward root->factor on the canonical P/Q ----
    std::printf("\nadd_pole_at/add_zero_at: forward root->factor on P/Q:\n");
    {
        RationalMap m = build_pq("z^2 + a");
        const Cplx z{0.7, 0.3}, a{0.1, -0.2};
        const Cplx before = m.eval(z, a);

        m.add_pole_at(Cplx{2.0, 0.0});
        const Cplx after_pole = m.eval(z, a);
        check(close(after_pole, before / (z - Cplx{2.0, 0.0})),
              "add_pole_at(2): eval matches dividing the PREVIOUS value by (z-2) exactly");
        check(m.to_formula() == "(z^2 + a) / (z-2)", "to_formula() wraps the previous authored "
              "text in the new factor -- mathematically exact, not just labelled");

        m.add_zero_at(Cplx{-1.0, 0.0});
        const Cplx after_zero = m.eval(z, a);
        check(close(after_zero, after_pole * (z - Cplx{-1.0, 0.0})),
              "add_zero_at(-1): eval matches multiplying the PREVIOUS value by (z+1) exactly");
        check(m.to_formula() == "((z^2 + a) / (z-2)) * (z+1)", "to_formula() wraps again, "
              "nesting correctly");

        // Re-parsing the updated formula reproduces the identical map --
        // the wrapped text is a faithful, re-parseable description, not
        // just a display-only label detached from the actual P/Q.
        RationalMap reparsed = build_pq(m.to_formula());
        check(close(reparsed.eval(z, a), m.eval(z, a)), "re-parsing to_formula()'s own output "
              "reproduces the identical map (eval matches)");

        // A repeated add_pole_at at the SAME location is NOT rejected as a
        // collision the way the term-based add_pole is (see add_pole_at's
        // own doc comment for why) -- verified via the SAME "divides the
        // previous value by the new factor" contract as the first
        // add_pole_at check above, not via pole_orders()'s numerically-
        // estimated order at that location: a genuine double root's
        // root-finder estimate isn't precise enough for vanishing_order's
        // own tolerance to reliably see past the first order of vanishing
        // (the same Aberth-Ehrlich multiple-root precision limit
        // documented elsewhere in this codebase -- e.g. cdx::roots' own
        // docs on repeated-root convergence degrading from cubic to
        // linear), a pre-existing characteristic of numerically-found
        // poles, not something add_pole_at introduces or needs to work
        // around for its OWN correctness (P and Q are exact either way).
        const Cplx before_repeat = m.eval(z, a);
        m.add_pole_at(Cplx{2.0, 0.0});
        check(close(m.eval(z, a), before_repeat / (z - Cplx{2.0, 0.0})),
              "a repeated add_pole_at at the same location still divides correctly -- it is "
              "not rejected as a collision");

        // Substituted-parameter bookkeeping survives a pole/zero addition
        // untouched -- neither operation touches any parameter.
        RationalMap m2 = RationalMap::from_expression("a*z^2 + b", "a", {{"b", Cplx{0.5, 0.0}}},
                                                       "sub_then_pole");
        m2.add_pole_at(Cplx{3.0, 0.0});
        check(m2.pq_fixed_params().size() == 1 && close(m2.pq_fixed_params().at("b"), Cplx{0.5, 0.0}),
              "add_pole_at preserves the substituted-parameter record (pq_fixed_params) "
              "untouched");
        check(m2.pq_active_param() == "a", "...and the active parameter too");

        // Throws for a term-based map.
        bool threw = false;
        try {
            RationalMap term = RationalMap::mandelbrot();
            term.add_pole_at(Cplx{1.0, 0.0});
        } catch (const std::logic_error&) { threw = true; }
        check(threw, "add_pole_at on a TERM-BASED map throws (not is_pq_backed())");
    }

    // ---- from_canonical rejects more than one parameter -------------------------
    std::printf("\nfrom_canonical: single-active-parameter enforcement:\n");
    {
        CanonicalRational cr;
        std::string error;
        check(parse_rational("a*z^2 + b*z + c", cr, error), "sanity: a genuinely "
              "multi-parameter expression parses fine at Stage 1");
        bool threw = false;
        try {
            RationalMap::from_canonical(cr, "multi");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "from_canonical throws for a CanonicalRational with more than one "
              "parameter -- selecting which one is active is a later stage's job, not this "
              "constructor's");
    }
    {
        CanonicalRational cr;
        std::string error;
        check(parse_rational("z - (z^3-1)/(3*z^2)", cr, error) && cr.parameters.empty(),
              "sanity: this map genuinely has zero free parameters");
        RationalMap pq = RationalMap::from_canonical(cr, "no-param");
        check(pq.is_pq_backed(), "from_canonical succeeds for a ZERO-parameter "
              "CanonicalRational too (the identity parameter binding case)");
    }

    // ---- is_polynomial_structurally is representation-agnostic ------------------
    std::printf("\nis_polynomial_structurally: same verdict from both representations:\n");
    {
        // z^5+a/z (has a pole -- NOT certified) -- hand-built term map vs P/Q,
        // a shape with no matching preset factory, so this is a genuinely
        // independent construction on each side.
        RationalMap term("custom5");
        term.add_poly({1.0, 0.0}, 5, 0, "z^5");
        term.add_pole({0.0, 0.0}, {1.0, 0.0}, 1, 1, "a/z");
        RationalMap pq = build_pq("z^5 + a/z");
        check(term.is_polynomial_structurally() == false &&
              pq.is_polynomial_structurally() == false,
              "both representations agree z^5+a/z is NOT structurally polynomial");

        RationalMap term2("custom_poly");
        term2.add_poly({1.0, 0.0}, 4, 0, "z^4");
        term2.add_poly({1.0, 0.0}, 1, 1, "a");
        RationalMap pq2 = build_pq("z^4 + a*z");
        check(term2.is_polynomial_structurally() && pq2.is_polynomial_structurally(),
              "both representations agree z^4+a*z IS structurally polynomial");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
