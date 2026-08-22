"""test_session.py -- property-based checks for app/session.Session.

Mirrors the C++ suite's style (PASS/FAIL per check, properties rather than
golden output) rather than pulling in a Python test framework this project
does not otherwise use. Run with:

    PYTHONPATH=cdx/build python -m app.test_session

(from the repository root -- app.session now imports app.render_cache, so
this must run package-qualified with the repo root on sys.path, not
standalone from inside this directory).
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

import cdx
from app.orbit_tracker import OrbitTracker
from app.session import PARAMETER_PLANE_MODES, PRESET_FAMILY_NAMES, Session, render_map

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    print("=== app.session tests ===")

    # ---- defaults --------------------------------------------------------------
    print("\ndefaults:")
    s = Session()
    check(s.map.name == "mandelbrot", "starts on the mandelbrot preset")
    check(s.param == complex(-0.7269, 0.1889),
          "starts with the dendritic Julia parameter, for whenever the user switches to it")
    check(len(s.library) >= 6, "library starts populated (with_defaults)")
    # Session no longer owns a render_mode or a viewport at all (see
    # app.pane.Pane -- both moved there in the coupled-viewer refactor);
    # RENDER_MODES/PARAMETER_PLANE_MODES/mode-string validation are now
    # exercised against Pane directly, in app/test_pane.py.

    # ---- render mode dispatch: render() takes viewport/mode explicitly ----------
    print("\nrender mode (Session.render(viewport, mode) dispatch):")
    vp = cdx.Viewport(complex(0, 0), 1.5, 41)
    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)

    julia = s.render(vp, "julia")
    check(julia.shape == (41, 41), "julia render matches viewport resolution")

    s.param = -1 + 0j   # the basilica
    basin = s.render(vp, "basin")
    check(basin.shape == (2, 41, 41),
          "basin render is STACKED: (2, height, width) -- labels, then iterations")
    labels, iterations = basin[0], basin[1]
    check(len(set(labels.flatten().tolist())) >= 2,
          "basin render for the basilica finds more than one region")
    check(iterations.min() >= 0, "the iterations layer is always non-negative")
    check(iterations[labels > 0].min() >= 1,
          "every RESOLVED pixel took at least one iteration to get there")

    param_plane = s.render(vp, "parameter")
    check(param_plane.shape == (41, 41), "parameter-plane render matches viewport resolution")

    greens_array = s.render(vp, "greens")
    check(greens_array.shape == (41, 41), "greens render matches viewport resolution")

    pgreens_array = s.render(vp, "parameter_greens")
    check(pgreens_array.shape == (41, 41),
          "parameter_greens render also matches viewport resolution")

    pbasin_array = s.render(vp, "parameter_basin")
    check(pbasin_array.shape == (2, 41, 41),
          "parameter_basin render is ALWAYS STACKED: (2, height, width) -- counts, "
          "then unresolved -- there is no certified-polynomial plain-2D fast path here")

    # ---- Stage 2: rational Julia is STACKED, escape_radius-invariant ------------
    print("\nrender mode dispatch: rational Julia (Stage 2 sphere-aware classification):")
    newton4 = cdx.RationalMap("newton4")   # (3/4)z + (1/4)z^-3 -- Newton's method for z^4-1
    newton4.add_poly(complex(0.75, 0.0), 1, 0, "(3/4)z")
    newton4.add_pole(complex(0.0, 0.0), complex(0.25, 0.0), 3, 0, "1/(4z^3)")
    check(cdx.polynomial_escape_certified(newton4) is False,
          "sanity: Newton z^4-1 genuinely takes the rational path")

    s_rational = Session()
    s_rational.map = newton4
    s_rational.param = 0j
    rational_vp = cdx.Viewport(complex(0, 0), 2.0, 41)
    s_rational.render_settings = cdx.RenderSettings(80, 2.0, 1e-6, 1)
    rational_julia = s_rational.render(rational_vp, "julia")
    check(rational_julia.shape == (2, 41, 41),
          "a RATIONAL map's julia render is STACKED: (2, height, width) -- smooth values, "
          "then basin labels -- the SAME shape basin's own render already has")
    values, labels = rational_julia[0], rational_julia[1]
    check(len(np.unique(labels[labels > 0])) >= 2,
          "Newton z^4-1's Julia render finds more than one basin (more than one root "
          "captures pixels)")

    # ACCEPTANCE TEST at the full Session.render level: escape_radius must
    # not affect a rational map's classification AT ALL.
    s_rational.render_settings = cdx.RenderSettings(80, 10.0, 1e-6, 1)
    rational_julia_r10 = s_rational.render(rational_vp, "julia")
    check(np.array_equal(rational_julia, rational_julia_r10),
          "ACCEPTANCE: Session.render(..., 'julia') for a rational map is BYTE-IDENTICAL "
          "at escape_radius=2 vs escape_radius=10 -- through the FULL app dispatch, not "
          "just the bare cdx.Renderer call")

    # A certified polynomial's OWN Julia render must still be plain (unchanged
    # shape/values) -- the rational path's introduction must not leak into it.
    s_rational.map = cdx.RationalMap.mandelbrot()
    s_rational.param = complex(-0.7269, 0.1889)
    s_rational.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)
    poly_julia = s_rational.render(rational_vp, "julia")
    check(poly_julia.shape == (41, 41),
          "ACCEPTANCE: a certified polynomial's julia render is STILL a plain 2D array, "
          "not stacked -- Stage 2 didn't change its shape or meaning")

    # ---- Parameter_basin: counts distinct attracting cycles per parameter ----------
    print("\nrender mode dispatch: Parameter_basin (number of attracting cycles):")
    # z^2 + a: at c=0 (deep in the cardioid) BOTH the origin's own fixed
    # point AND infinity are independently confirmed attracting -- count=2.
    # Far outside (c=5+5i), both critical-point seeds (0 and infinity)
    # converge to the SAME attractor and dedupe to count=1. Cross-checked
    # against cdx.find_attractors directly in cdx/test/test_analysis.cpp;
    # this is the app-dispatch-level counterpart, not a re-derivation.
    quad = cdx.RationalMap("quad")
    quad.add_poly(1 + 0j, 2, 0, "z^2")
    quad.add_poly(1 + 0j, 0, 1, "a")

    s_pb = Session()
    s_pb.map = quad
    s_pb.render_settings = cdx.RenderSettings(100, 2.0, 1e-6, 1)

    tiny_vp_inside = cdx.Viewport(complex(0, 0), 0.001, 3)
    counts_inside, unresolved_inside = s_pb.render(tiny_vp_inside, "parameter_basin")
    check(counts_inside[1, 1] == 2.0 and unresolved_inside[1, 1] == 0.0,
          "c=0: count=2 (origin's own fixed point + infinity), unresolved=0")

    tiny_vp_far = cdx.Viewport(complex(5, 5), 0.001, 3)
    counts_far, unresolved_far = s_pb.render(tiny_vp_far, "parameter_basin")
    check(counts_far[1, 1] == 1.0 and unresolved_far[1, 1] == 0.0,
          "c=5+5i: count=1 (both seeds dedupe onto infinity), unresolved=0")

    # A real render spanning the cusp shows count genuinely CHANGING across
    # the plane -- this is what makes colour_parameter_basin's sharp
    # colour edges meaningful at all, not a uniform/degenerate image.
    wide_vp = cdx.Viewport(complex(-0.5, 0), 1.5, 61)
    counts_wide, _unresolved_wide = s_pb.render(wide_vp, "parameter_basin")
    unique_counts = set(counts_wide.flatten().tolist())
    check(len(unique_counts & {1.0, 2.0}) == 2,
          "a real render shows BOTH count=1 and count=2 pixels -- a genuine "
          "classification, not a flat plane")

    # No-effect-parameter guard (dynamical-plane concept generalized):
    # newton_cubic() has no term depending on `a` at all, so ImageView's
    # own no_effect_parameter_message (app/sandbox.py) should treat
    # "parameter_basin" the same as "parameter"/"parameter_greens" --
    # verified here at the describe_parameter_role level this session
    # already tests elsewhere, confirming PARAMETER_PLANE_MODES
    # membership (not a hardcoded per-mode name list) is what drives it.
    check("parameter_basin" in PARAMETER_PLANE_MODES,
          "parameter_basin is a genuine PARAMETER-plane mode -- gets the marker/"
          "arrow-nudging/no-effect-guard machinery every other one already has, "
          "purely through set membership")

    # ---- render cache -----------------------------------------------------------
    print("\nrender cache:")
    s.map = cdx.RationalMap.mandelbrot()
    s.param = 0j
    vp = cdx.Viewport(complex(0, 0), 1.5, 41)
    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)
    s.cache.clear()

    before = s.cache.stats
    first = s.render(vp, "julia")
    after_first = s.cache.stats
    check(after_first.misses == before.misses + 1 and after_first.entry_count == 1,
          "Session.render() populates the cache on a miss")

    second = s.render(vp, "julia")
    after_second = s.cache.stats
    check(after_second.hits == after_first.hits + 1 and after_second.misses == after_first.misses,
          "an identical Session.render() call is a cache hit, not a second render")
    check(np.array_equal(first, second), "a cache hit returns the same pixels as the original render")

    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 2)   # only threads differs
    after_threads = s.cache.stats
    s.render(vp, "julia")
    check(s.cache.stats.hits == after_threads.hits + 1,
          "threads is not part of the cache key -- changing only thread count still hits")

    vp = cdx.Viewport(complex(0, 0), 1.5, 43)   # a real, output-affecting change
    misses_before_res_change = s.cache.stats.misses
    s.render(vp, "julia")
    check(s.cache.stats.misses == misses_before_res_change + 1,
          "changing resolution is a cache miss (a different key), not a stale hit")

    # P5b integration requirement: term edits and parameter changes must
    # invalidate the right cache keys and nothing else. make_key() folds in
    # map.serialize() (the FULL term content, not just object identity) and
    # param as separate components (see render_cache.py's own docstring),
    # so this should already hold "by construction" -- confirmed directly
    # here rather than just asserted, since a cache key silently missing a
    # real input is exactly the kind of bug that only shows up as a stale
    # image, not an exception.
    s.render(vp, "julia")   # re-establish a hit baseline at the CURRENT (43-res) key
    hits_before_edit = s.cache.stats.hits
    misses_before_edit = s.cache.stats.misses
    entries_before_edit = s.cache.stats.entry_count
    s.map.poly_terms()[0].coeff = 3 + 0j   # mandelbrot's z^2 term, live-mutated in place
    s.render(vp, "julia")
    check(s.cache.stats.misses == misses_before_edit + 1 and s.cache.stats.hits == hits_before_edit,
          "editing a term is a cache miss, not a stale hit -- even though viewport/settings/mode "
          "are all unchanged and s.map is the SAME Python object (mutated in place, not reassigned)")
    check(s.cache.stats.entry_count == entries_before_edit + 1,
          "the edit adds exactly one new entry -- the old (now-unreachable) one isn't touched, "
          "not silently overwritten or duplicated")

    misses_before_param = s.cache.stats.misses
    s.param = s.param + 0.01
    s.render(vp, "julia")
    check(s.cache.stats.misses == misses_before_param + 1,
          "changing the parameter is also a cache miss, not a stale hit -- param is part of "
          "the key even though it isn't part of map.serialize() at all")

    # A cancelled render must not poison the cache with a partial result --
    # cancelled BEFORE the render call (not raced against a slow render) so
    # this is deterministic: render_map still checks `cancel` after calling
    # into the engine, so a token that was already cancelled on entry never
    # gets to populate the cache at all.
    cancel_key_viewport = cdx.Viewport(complex(0.4, 0.4), 1.2, 41)
    cancel_token = cdx.CancelToken()
    cancel_token.cancel()
    entries_before_cancel = s.cache.stats.entry_count
    render_map(s.map, s.param, cancel_key_viewport, s.render_settings, "julia",
              cancel_token, s.cache)
    check(s.cache.stats.entry_count == entries_before_cancel,
          "a cancelled render's partial result is never stored in the cache")

    misses_before_retry = s.cache.stats.misses
    render_map(s.map, s.param, cancel_key_viewport, s.render_settings, "julia", None, s.cache)
    check(s.cache.stats.misses == misses_before_retry + 1,
          "a subsequent lookup for that same key still misses -- nothing was cached under it")

    # ---- term editing ----------------------------------------------------------
    print("\nterm editing:")
    s2 = Session()
    s2.map = cdx.RationalMap("scratch")
    idx = s2.add_poly_term(2 + 0j, 3)
    check(idx == 0, "add_poly_term returns the new term's index")
    check(s2.map.eval(2 + 0j, 0j) == 16 + 0j, "added term (2z^3) evaluates correctly: 2*2^3=16")

    s2.edit_poly_term(0, coeff=3 + 0j)
    check(s2.map.eval(2 + 0j, 0j) == 24 + 0j, "edit_poly_term mutated the live term (3*2^3=24)")

    try:
        s2.edit_poly_term(5, coeff=1 + 0j)
        check(False, "out-of-range edit_poly_term raises")
    except IndexError:
        check(True, "out-of-range edit_poly_term raises IndexError")

    pole_idx = s2.add_pole_term(1 + 0j, 1 + 0j, 1)
    check(pole_idx == 0, "add_pole_term returns the new term's index")
    s2.remove_poly_term(0)
    check(len(s2.map.poly_terms()) == 0, "remove_poly_term removes the poly term")
    s2.remove_pole_term(0)
    check(len(s2.map.pole_terms()) == 0, "remove_pole_term removes the pole term")

    # ---- term editing: reorder --------------------------------------------------
    print("\nterm editing (reorder):")
    s2r = Session()
    s2r.map = cdx.RationalMap("scratch-reorder")
    s2r.add_poly_term(1 + 0j, 2, 0, "first")
    s2r.add_poly_term(2 + 0j, 3, 0, "second")
    s2r.add_poly_term(3 + 0j, 4, 0, "third")

    ok = s2r.move_poly_term(0, +1)
    labels = [t.label for t in s2r.map.poly_terms()]
    check(ok and labels == ["second", "first", "third"],
          "move_poly_term(0, +1) swaps with its neighbor, not a silent duplicate "
          "(the bind_vector aliasing trap this is written to avoid)")

    ok = s2r.move_poly_term(0, -1)
    check(not ok and [t.label for t in s2r.map.poly_terms()] == ["second", "first", "third"],
          "move_poly_term at index 0 with direction -1 is a no-op (no neighbor there)")

    ok = s2r.move_poly_term(2, +1)
    check(not ok, "move_poly_term at the last index with direction +1 is a no-op")

    s2r.add_pole_term(0j, 1 + 0j, 1, 0, "polefirst")
    s2r.add_pole_term(5 + 0j, 2 + 0j, 1, 0, "polesecond")
    before_eval = s2r.map.eval(3 + 0j, 0j)
    ok = s2r.move_pole_term(0, +1)
    after_eval = s2r.map.eval(3 + 0j, 0j)
    check(ok and [t.label for t in s2r.map.pole_terms()] == ["polesecond", "polefirst"],
          "move_pole_term swaps pole terms the same way")
    check(after_eval == before_eval,
          "reordering terms never changes what the map evaluates to -- it's a sum either way")

    # ---- term editing: pole-location uniqueness / negative-exponent rejection ----
    print("\nterm editing (validation):")
    s2b = Session()
    s2b.map = cdx.RationalMap("scratch2")
    try:
        s2b.add_poly_term(1 + 0j, -2)
        check(False, "add_poly_term with a negative exponent raises")
    except ValueError as e:
        check("negative exponent" in str(e), "add_poly_term raises ValueError, message explains why")
    check(len(s2b.map.poly_terms()) == 0, "the rejected add_poly_term added nothing")

    s2b.add_pole_term(1 + 0j, 1 + 0j, 1)
    try:
        s2b.add_pole_term(1 + 0j, 2 + 0j, 3)
        check(False, "add_pole_term at an already-occupied location raises")
    except ValueError as e:
        check("already exists" in str(e), "add_pole_term raises ValueError, message names the conflict")
    check(len(s2b.map.pole_terms()) == 1, "the rejected add_pole_term added nothing")

    # ---- library -----------------------------------------------------------------
    print("\nlibrary:")
    s3 = Session()
    s3.map = cdx.RationalMap.mandelbrot()
    s3.save_to_library("my-mandelbrot")
    check(s3.library.find("my-mandelbrot") is not None, "save_to_library adds under the given name")

    s3.map = cdx.RationalMap.newton_cubic()
    s3.load_from_library("my-mandelbrot")
    check(s3.map.name == "my-mandelbrot" and s3.map.to_formula() == "z^2 + a",
          "load_from_library restores the saved map")

    try:
        s3.load_from_library("does-not-exist")
        check(False, "loading a missing name raises")
    except KeyError:
        check(True, "loading a missing name raises KeyError")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "library.txt")
        s3.save_library_file(path)
        names_before = set(s3.library.names())
        s4 = Session()
        s4.load_library_file(path)
        check(set(s4.library.names()) == names_before,
              "save_library_file/load_library_file round-trips the same names")

    # ---- library: presets are read-only --------------------------------------------
    print("\nlibrary (preset read-only guards):")
    s3b = Session()
    try:
        s3b.save_to_library("mandelbrot")
        check(False, "save_to_library over a preset name raises")
    except ValueError as e:
        check("preset" in str(e), "save_to_library raises ValueError naming the preset")
    s3b.map.name = "mandelbrot"
    try:
        s3b.save_to_library()   # name=None -- uses self.map.name, still a preset
        check(False, "save_to_library() with self.map.name already a preset also raises")
    except ValueError:
        check(True, "the guard applies whether the name comes from the argument or self.map.name")

    try:
        s3b.rename_in_library("mandelbrot", "not-mandelbrot")
        check(False, "renaming a preset raises")
    except ValueError:
        check(True, "rename_in_library raises ValueError for a preset source name")
    try:
        s3b.save_to_library("scratch-for-rename")
        s3b.rename_in_library("scratch-for-rename", "mandelbrot")
        check(False, "renaming ONTO a preset name raises")
    except ValueError:
        check(True, "rename_in_library raises ValueError for a preset target name")

    try:
        s3b.delete_from_library("mandelbrot")
        check(False, "deleting a preset raises")
    except ValueError:
        check(True, "delete_from_library raises ValueError for a preset name")

    try:
        s3b.set_library_notes("mandelbrot", "nope")
        check(False, "editing a preset's notes raises")
    except ValueError:
        check(True, "set_library_notes raises ValueError for a preset name")

    # ---- library: rename / delete / notes on a real user entry ---------------------
    print("\nlibrary (rename/delete/notes):")
    s3c = Session()
    s3c.map = cdx.RationalMap("scratch")
    s3c.save_to_library("fam-a")
    s3c.set_library_notes("fam-a", "first attempt")
    check(s3c.library.find("fam-a").notes == "first attempt", "set_library_notes updates the entry")

    try:
        s3c.set_library_notes("does-not-exist", "x")
        check(False, "editing notes on a missing name raises")
    except KeyError:
        check(True, "set_library_notes raises KeyError for a missing name")

    s3c.rename_in_library("fam-a", "fam-b")
    check("fam-b" in s3c.library.names() and "fam-a" not in s3c.library.names(),
          "rename_in_library moves the entry to the new name")
    check(s3c.library.find("fam-b").notes == "first attempt", "the renamed entry keeps its notes")

    s3c.save_to_library("fam-c")
    try:
        s3c.rename_in_library("fam-b", "fam-c")
        check(False, "renaming onto an existing (non-preset) name raises")
    except ValueError as e:
        check("already exists" in str(e), "rename_in_library names the collision")

    try:
        s3c.rename_in_library("does-not-exist", "whatever")
        check(False, "renaming a missing source name raises")
    except KeyError:
        check(True, "rename_in_library raises KeyError for a missing source name")

    check(s3c.delete_from_library("fam-b") is None, "delete_from_library succeeds silently")
    check("fam-b" not in s3c.library.names(), "the deleted entry is gone")
    try:
        s3c.delete_from_library("fam-b")
        check(False, "deleting an already-gone name raises")
    except KeyError:
        check(True, "delete_from_library raises KeyError for a missing name")

    # ---- library: user-only persistence (save_user_library/load_user_library) ------
    print("\nlibrary (user-only persistence):")
    with tempfile.TemporaryDirectory() as tmp:
        lib_path = os.path.join(tmp, "library.txt")

        s5a = Session()
        s5a.map = cdx.RationalMap("scratch")
        s5a.save_to_library("only-user-family")
        s5a.save_user_library(lib_path)
        saved_text = open(lib_path).read()
        check("only-user-family" in saved_text and "mandelbrot" not in saved_text,
              "save_user_library writes the user entry but none of the six presets")

        s5b = Session()
        check("only-user-family" not in s5b.library.names(),
              "sanity: a fresh Session doesn't have the user family yet")
        s5b.load_user_library(lib_path)
        check("only-user-family" in s5b.library.names(),
              "load_user_library merges the saved user family in")
        check(set(PRESET_FAMILY_NAMES) <= set(s5b.library.names()),
              "load_user_library never removes any of the six presets")

        s5c = Session()
        s5c.load_user_library(os.path.join(tmp, "does-not-exist.txt"))
        check(set(s5c.library.names()) == PRESET_FAMILY_NAMES,
              "load_user_library on a missing file is a silent no-op, not an error")

        malformed_path = os.path.join(tmp, "malformed.txt")
        with open(malformed_path, "w") as f:
            f.write("this is not a valid library file\n")
        s5d = Session()
        s5d.load_user_library(malformed_path)   # must not raise
        check(set(s5d.library.names()) == PRESET_FAMILY_NAMES,
              "load_user_library on a malformed file is also a silent no-op, not an error")

    # ---- dynamical_facts -----------------------------------------------------------
    print("\ndynamical_facts:")
    s5 = Session()
    s5.map = cdx.RationalMap.newton_cubic()
    s5.param = 0j
    facts = s5.dynamical_facts()
    check(facts.degree == 3, "newton_cubic: degree 3")
    check(len(facts.critical_points) == 4, "newton_cubic: 4 critical points")
    check(len(facts.attracting_cycles) == 3, "newton_cubic: 3 attracting cycles")
    check(all(ac.period == 1 for ac in facts.attracting_cycles),
          "newton_cubic: all 3 attracting cycles are fixed points")
    check(len(facts.fixed_points) == 4, "newton_cubic: 4 fixed points (3 roots + infinity)")
    check(facts.pole_locations == [0j] and facts.pole_orders == [2],
          "newton_cubic: one pole at the origin, order 2")

    opts = cdx.FindAttractorsOptions()
    opts.burn_in = 5   # deliberately too small to reliably converge
    facts_custom = s5.dynamical_facts(opts)
    check(facts_custom.degree == facts.degree,
          "dynamical_facts(opts): facts unrelated to discovery are unaffected")

    s6 = Session()
    s6.map = cdx.RationalMap.mandelbrot()
    s6.param = -1 + 0j
    facts_basilica = s6.dynamical_facts()
    n_2cycle = sum(1 for ac in facts_basilica.attracting_cycles if ac.period == 2)
    n_finite_fixed_attracting = sum(
        1 for fp in facts_basilica.fixed_points
        if not math.isinf(fp.point.real) and abs(fp.multiplier) < 1.0
    )
    check(n_2cycle == 1, "basilica: one attracting 2-cycle")
    check(n_finite_fixed_attracting == 0, "basilica: zero finite attracting fixed points")

    # ---- experiment snapshots: round-trip (map/param/layout/settings/orbit) --------
    print("\nexperiment snapshots (snapshot_to_dict/restore_from_snapshot):")

    # A non-trivial, custom (not-a-preset) map -- to_formula() is the
    # cleanest thing to compare on, since RationalMap has no __eq__.
    # Neither a viewport/render_mode NOR a notion of "panes" is Session's
    # own (see app.pane.Pane) -- snapshot_to_dict/restore_from_snapshot
    # now take/return a full LAYOUT (one (viewport, render_mode) pair per
    # pane, which pane is focused, coupled/single) explicitly, the same
    # way a real SandboxWindow would supply/receive it (Stage 4).
    src = Session()
    custom_map = cdx.RationalMap("scratch")
    custom_map.add_poly(complex(1, 0), 3, 0, "z^3")
    custom_map.add_pole(complex(0.5, -0.5), complex(1, 0), 2, 1, "pole")
    src.map = custom_map
    src.param = complex(0.123, -0.456)
    src_pane_a = (cdx.Viewport(complex(0.01, -0.02), 1e-4, 333), "basin")   # a genuinely zoomed view
    src_pane_b = (cdx.Viewport(complex(-0.5, 0.0), 1.5, 400), "julia")
    src_panes = [src_pane_a, src_pane_b]
    src_focused_index = 1
    src_coupled = False
    src.render_settings = cdx.RenderSettings(321, 3.5, 1e-8, 2)

    z0 = complex(0.3, -0.1)
    d = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled, orbit=(1, z0, 4))
    dst = Session()
    dst_panes, dst_focused_index, dst_coupled, orbit, _preview = dst.restore_from_snapshot(d)

    check(dst.map.to_formula() == src.map.to_formula(), "round-trip: map formula matches")
    check(dst.param == src.param, "round-trip: param matches")
    check(len(dst_panes) == 2, "round-trip: both panes survive, not just one")
    for i, (src_vp, src_mode) in enumerate(src_panes):
        dst_vp, dst_mode = dst_panes[i]
        check(dst_vp.center == src_vp.center and dst_vp.scale == src_vp.scale and
              dst_vp.resolution == src_vp.resolution,
              f"round-trip: pane {i}'s viewport (center/scale/resolution) matches")
        check(dst_mode == src_mode, f"round-trip: pane {i}'s render_mode matches")
    check(dst_focused_index == src_focused_index, "round-trip: which pane is focused matches")
    check(dst_coupled == src_coupled, "round-trip: coupled/single layout flag matches")
    check(dst.render_settings.max_iter == src.render_settings.max_iter and
          dst.render_settings.escape_radius == src.render_settings.escape_radius and
          dst.render_settings.tol == src.render_settings.tol and
          dst.render_settings.threads == src.render_settings.threads,
          "round-trip: render_settings (all four fields) matches")
    check(orbit is not None and orbit[0] == 1 and orbit[1] == z0 and orbit[2] == 4,
          "round-trip: orbit's owning pane index, z0, and n all survive")

    # z/history are DERIVED, not stored -- reconstruct via the tracker's
    # REAL API (exactly what app/sandbox.py's Open Experiment does) and
    # confirm the regenerated orbit matches what stepping the ORIGINAL
    # map/param the same way would have produced.
    restored_tracker = OrbitTracker()
    restored_tracker.seed(dst.map, dst.param, orbit[1])
    restored_tracker.step(dst.map, dst.param, orbit[2])
    reference_tracker = OrbitTracker()
    reference_tracker.seed(src.map, src.param, z0)
    reference_tracker.step(src.map, src.param, 4)
    check(restored_tracker.state.z == reference_tracker.state.z,
          "round-trip: the regenerated orbit's current z matches stepping the ORIGINAL "
          "map/param the same number of times from the same z0")
    check(restored_tracker.state.history == reference_tracker.state.history,
          "round-trip: the regenerated orbit's full history matches too, not just the "
          "final point")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.cdsx")
        src.save_snapshot(path, src_panes, src_focused_index, src_coupled, orbit=(1, z0, 4))
        dst_file = Session()
        (panes_file, focused_index_file, coupled_file, orbit_file,
         _preview_file) = dst_file.load_snapshot(path)
        check(dst_file.map.to_formula() == src.map.to_formula() and dst_file.param == src.param
              and focused_index_file == src_focused_index and coupled_file == src_coupled
              and panes_file[0][1] == src_pane_a[1] and panes_file[1][1] == src_pane_b[1]
              and orbit_file == (1, z0, 4),
              "save_snapshot/load_snapshot round-trips through an ACTUAL JSON file, not just "
              "the in-memory dict -- the float encode/decode step is lossless too")

    # ---- orbit-null round-trip -------------------------------------------------------
    d_null = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled, orbit=None)
    dst_null = Session()
    (_panes_null, _focus_null, _coupled_null,
     orbit_null, _preview_null) = dst_null.restore_from_snapshot(d_null)
    check(orbit_null is None, "orbit-null round-trip: no active orbit survives as no orbit")

    # ---- parameter-plane mode with no orbit -------------------------------------------
    param_plane_session = Session()
    param_plane_session.map = cdx.RationalMap.mandelbrot()
    d_pp = param_plane_session.snapshot_to_dict(
        [(src_pane_a[0], "parameter"), src_pane_b], 0, True, orbit=None)
    dst_pp = Session()
    panes_pp, _focus_pp, _coupled_pp, orbit_pp, _preview_pp = dst_pp.restore_from_snapshot(d_pp)
    check(orbit_pp is None and panes_pp[0][1] == "parameter",
          "a snapshot taken with a parameter-plane pane and no orbit restores cleanly, with "
          "nothing to seed")

    # ---- embedded preview (Stage C): round-trips, and is optional on read ------------
    print("\nexperiment snapshots (embedded preview thumbnail):")
    d_preview = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled,
                                     orbit=None, preview_png_base64="not-really-a-png-just-a-marker")
    dst_preview = Session()
    (_panes_pv, _focus_pv, _coupled_pv,
     _orbit_pv, preview_pv) = dst_preview.restore_from_snapshot(d_preview)
    check(preview_pv == "not-really-a-png-just-a-marker",
          "a snapshot saved WITH a preview restores that exact preview string")

    d_no_preview = src.snapshot_to_dict(src_panes, src_focused_index, src_coupled, orbit=None)
    dst_no_preview = Session()
    (_panes_np, _focus_np, _coupled_np,
     _orbit_np, preview_np) = dst_no_preview.restore_from_snapshot(d_no_preview)
    check(preview_np is None,
          "a snapshot saved WITHOUT a preview restores preview as None, not an error")

    # A hand-built version-2 dict -- the actual pre-Stage-C shape, with NO
    # "preview" key at all (not even present, unlike d_no_preview above
    # which is a version-3 dict whose preview happens to be None) -- must
    # still load cleanly: see MIN_SNAPSHOT_SCHEMA_VERSION's own comment for
    # why this, unlike the version-1 case below, is accepted rather than
    # rejected.
    v2_snapshot = {
        "schema_version": 2, "app_version": "0.0.0-test",
        "map": src.map.serialize(), "param": [src.param.real, src.param.imag],
        "render_settings": {"max_iter": src.render_settings.max_iter,
                            "escape_radius": src.render_settings.escape_radius,
                            "tol": src.render_settings.tol,
                            "threads": src.render_settings.threads},
        "layout": {"coupled": src_coupled, "focused_index": src_focused_index,
                   "panes": [{"render_mode": mode,
                              "viewport": {"center": [vp.center.real, vp.center.imag],
                                          "scale": vp.scale, "resolution": vp.resolution}}
                             for vp, mode in src_panes]},
        "orbit": None,
    }
    dst_v2 = Session()
    (panes_v2, focus_v2, coupled_v2,
     orbit_v2, preview_v2) = dst_v2.restore_from_snapshot(v2_snapshot)
    check(dst_v2.map.to_formula() == src.map.to_formula() and focus_v2 == src_focused_index
          and coupled_v2 == src_coupled and len(panes_v2) == 2 and orbit_v2 is None,
          "an older (version-2, pre-Stage-C) snapshot with no 'preview' key at all still "
          "loads fine -- not just rejected the way a version-1 snapshot is")
    check(preview_v2 is None,
          "...with preview correctly reported as None, not a KeyError")

    # ---- malformed / wrong-version input: raises, changes NOTHING --------------------
    print("\nexperiment snapshots (validation -- malformed input never mutates state):")
    guard_session = Session()
    guard_session.map = cdx.RationalMap.mandelbrot()
    guard_session.param = complex(-0.7269, 0.1889)
    before_map = guard_session.map.serialize()   # pre-captured, compared byte-for-byte after
    before_param = guard_session.param
    before_max_iter = guard_session.render_settings.max_iter

    valid_map_text = guard_session.map.serialize()
    valid_layout = {"coupled": True, "focused_index": 0,
                    "panes": [{"render_mode": "julia",
                               "viewport": {"center": [0, 0], "scale": 1, "resolution": 10}}]}
    bad_snapshots = [
        {},                                   # missing schema_version entirely
        {"schema_version": 999},              # wrong version
        {"schema_version": 1, "map": valid_map_text, "param": [0, 0],   # the OLD (Stage 1) shape,
         "viewport": {"center": [0, 0], "scale": 1, "resolution": 10},  # rejected by version, not
         "render_mode": "julia",                                        # migrated -- see
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0}},  # SNAPSHOT_SCHEMA_VERSION's own comment
        "not even a dict",
        None,
        {"schema_version": 2, "map": "not a real serialized map", "param": [0, 0],
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": valid_layout},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": {"coupled": True, "focused_index": 0,
                    "panes": [{"render_mode": "not_a_real_mode",
                               "viewport": {"center": [0, 0], "scale": 1, "resolution": 10}}]}},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],   # focused_index out of range
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": {"coupled": True, "focused_index": 5,
                    "panes": [{"render_mode": "julia",
                               "viewport": {"center": [0, 0], "scale": 1, "resolution": 10}}]}},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],   # layout.panes empty
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": {"coupled": True, "focused_index": 0, "panes": []}},
        {"schema_version": 2, "map": valid_map_text, "param": [0, 0],   # orbit pane_index out of range
         "render_settings": {"max_iter": 1, "escape_radius": 1, "tol": 1, "threads": 0},
         "layout": valid_layout,
         "orbit": {"pane_index": 9, "z0": [0, 0], "n": 0}},
    ]
    for bad in bad_snapshots:
        try:
            guard_session.restore_from_snapshot(bad)
            check(False, f"malformed input {bad!r} should have raised ValueError")
        except ValueError:
            check(True, f"malformed/wrong-version input ({type(bad).__name__}) raises ValueError")

    check(guard_session.map.serialize() == before_map,
          "the live session's map is BYTE-FOR-BYTE unchanged after every rejected load")
    check(guard_session.param == before_param and
          guard_session.render_settings.max_iter == before_max_iter,
          "...and so is everything else Session still owns -- no half-applied load")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
