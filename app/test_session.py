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
from app.session import PRESET_FAMILY_NAMES, Session, render_map

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
    check(s.render_mode == "parameter", "starts in parameter-plane render mode (the Mandelbrot set)")
    check(len(s.library) >= 6, "library starts populated (with_defaults)")

    # ---- render mode ------------------------------------------------------------
    print("\nrender mode:")
    try:
        s.set_render_mode("nonsense")
        check(False, "invalid render mode raises")
    except ValueError:
        check(True, "invalid render mode raises ValueError")

    s.viewport = cdx.Viewport(complex(0, 0), 1.5, 41)
    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)

    s.set_render_mode("julia")
    julia = s.render()
    check(julia.shape == (41, 41), "julia render matches viewport resolution")

    s.param = -1 + 0j   # the basilica
    s.set_render_mode("basin")
    basin = s.render()
    labels = set(basin.flatten().tolist())
    check(len(labels) >= 2, "basin render for the basilica finds more than one region")

    s.set_render_mode("parameter")
    param_plane = s.render()
    check(param_plane.shape == (41, 41), "parameter-plane render matches viewport resolution")

    s.set_render_mode("greens")
    greens = s.render()
    check(greens.shape == (41, 41), "greens render matches viewport resolution")

    # ---- render cache -----------------------------------------------------------
    print("\nrender cache:")
    s.map = cdx.RationalMap.mandelbrot()
    s.param = 0j
    s.viewport = cdx.Viewport(complex(0, 0), 1.5, 41)
    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 0)
    s.set_render_mode("julia")
    s.cache.clear()

    before = s.cache.stats
    first = s.render()
    after_first = s.cache.stats
    check(after_first.misses == before.misses + 1 and after_first.entry_count == 1,
          "Session.render() populates the cache on a miss")

    second = s.render()
    after_second = s.cache.stats
    check(after_second.hits == after_first.hits + 1 and after_second.misses == after_first.misses,
          "an identical Session.render() call is a cache hit, not a second render")
    check(np.array_equal(first, second), "a cache hit returns the same pixels as the original render")

    s.render_settings = cdx.RenderSettings(50, 2.0, 1e-6, 2)   # only threads differs
    after_threads = s.cache.stats
    s.render()
    check(s.cache.stats.hits == after_threads.hits + 1,
          "threads is not part of the cache key -- changing only thread count still hits")

    s.viewport = cdx.Viewport(complex(0, 0), 1.5, 43)   # a real, output-affecting change
    misses_before_res_change = s.cache.stats.misses
    s.render()
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
    s.render()   # re-establish a hit baseline at the CURRENT (43-res) key
    hits_before_edit = s.cache.stats.hits
    misses_before_edit = s.cache.stats.misses
    entries_before_edit = s.cache.stats.entry_count
    s.map.poly_terms()[0].coeff = 3 + 0j   # mandelbrot's z^2 term, live-mutated in place
    s.render()
    check(s.cache.stats.misses == misses_before_edit + 1 and s.cache.stats.hits == hits_before_edit,
          "editing a term is a cache miss, not a stale hit -- even though viewport/settings/mode "
          "are all unchanged and s.map is the SAME Python object (mutated in place, not reassigned)")
    check(s.cache.stats.entry_count == entries_before_edit + 1,
          "the edit adds exactly one new entry -- the old (now-unreachable) one isn't touched, "
          "not silently overwritten or duplicated")

    misses_before_param = s.cache.stats.misses
    s.param = s.param + 0.01
    s.render()
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

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
