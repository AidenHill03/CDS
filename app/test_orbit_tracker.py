"""test_orbit_tracker.py -- property-based checks for app/orbit_tracker.

Run with:

    PYTHONPATH=cdx/build python -m app.test_orbit_tracker

(from the repository root -- pure Python + cdx, no Qt dependency.)
"""

from __future__ import annotations

import cdx
from app.orbit_tracker import OrbitTracker

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    print("=== app.orbit_tracker tests ===")

    mandelbrot = cdx.RationalMap.mandelbrot()

    # ---- seed / step / clear ------------------------------------------------------
    print("\nseed/step/clear:")
    t = OrbitTracker()
    check(t.state is None, "a fresh tracker has no orbit")

    t.seed(mandelbrot, 0j, 0.3 + 0j)
    check(t.state is not None and t.state.z0 == 0.3 + 0j, "seed() starts an orbit at z0")
    check(t.state.n == 0 and t.state.z == 0.3 + 0j,
          "immediately after seeding, n=0 and z is still z0 -- no step has happened yet")
    check(t.state.history == [0.3 + 0j], "history starts as just [z0]")

    t.step(mandelbrot, 0j, 1)
    check(t.state.n == 1, "one step advances n by 1")
    expected_z1 = mandelbrot.eval(0.3 + 0j, 0j)
    check(t.state.z == expected_z1, "one step applies map.eval(z, param) exactly once")
    check(t.state.history == [0.3 + 0j, expected_z1],
          "history grows by exactly one point per step")

    t.step(mandelbrot, 0j, 5)
    check(t.state.n == 6, "stepping with count=5 advances n by 5 more (6 total)")
    check(len(t.state.history) == 7, "history has one entry per iteration plus the seed")

    t.clear()
    check(t.state is None, "clear() removes the orbit")
    t.step(mandelbrot, 0j, 1)   # stepping with no orbit must not crash
    check(t.state is None, "stepping with no active orbit is a harmless no-op, not a crash")

    # ---- reset_if_stale: clears on map/param change, survives nothing else --------
    print("\nreset on map/param change:")
    t2 = OrbitTracker()
    t2.seed(mandelbrot, -1 + 0j, 0.1 + 0j)
    t2.step(mandelbrot, -1 + 0j, 3)
    check(t2.state.n == 3, "sanity: orbit has advanced before the staleness checks below")

    t2.reset_if_stale(mandelbrot, -1 + 0j)   # SAME map, SAME param
    check(t2.state is not None and t2.state.n == 3,
          "reset_if_stale is a no-op when neither the map nor the parameter actually changed")

    t2.reset_if_stale(mandelbrot, -0.5 + 0j)   # same map, DIFFERENT param
    check(t2.state is None, "reset_if_stale clears the orbit when the parameter changes")

    t3 = OrbitTracker()
    t3.seed(mandelbrot, 0j, 0.1 + 0j)
    newton = cdx.RationalMap.newton_cubic()
    t3.reset_if_stale(newton, 0j)   # different map, same param value
    check(t3.state is None, "reset_if_stale clears the orbit when the MAP changes")

    # Explicitly NOT tied to viewport in any way -- there is no viewport
    # parameter to reset_if_stale at all, which is the point: nothing
    # about panning/zooming can even be expressed as an input to this
    # class, so it structurally cannot clear the orbit for that reason.

    # ---- recompute_current: keeps z0, restarts under the new map/param ------------
    print("\nrecompute_current (P6: a persistent, independently-chosen seed):")
    t5 = OrbitTracker()
    check(t5.state is None, "sanity: no orbit seeded yet")
    t5.recompute_current(mandelbrot, -1 + 0j)
    check(t5.state is None, "a no-op with no active orbit -- nothing to replay")

    t5.seed(mandelbrot, -1 + 0j, 0.2 + 0j)
    t5.step(mandelbrot, -1 + 0j, 4)
    check(t5.state.n == 4, "sanity: orbit has advanced before recomputing")
    t5.recompute_current(mandelbrot, -0.5 + 0j)   # a NEW param, same map
    check(t5.state is not None, "recompute_current does NOT clear the orbit, unlike reset_if_stale")
    check(t5.state.z0 == 0.2 + 0j, "the z0 itself survives the recompute unchanged")
    check(t5.state.n == 0, "the orbit restarts from n=0 under the new param, not left at n=4")
    check(t5.state.z == 0.2 + 0j, "z is back at z0 -- this is a fresh orbit, not a continuation")

    t5.recompute_current(newton, -0.5 + 0j)   # a NEW map entirely, same z0
    check(t5.state.z0 == 0.2 + 0j, "recompute_current also survives a MAP change, not just param")

    t5.clear()
    t5.recompute_current(mandelbrot, 0j)
    check(t5.state is None,
          "once cleared, recompute_current stays a no-op -- Clear means cleared, a param "
          "change afterward must not resurrect the orbit")

    # ---- classification: converges to a period-2 cycle (the basilica) -------------
    print("\nclassification (basilica, period-2 cycle):")
    t4 = OrbitTracker()
    t4.seed(mandelbrot, -1 + 0j, 0.1 + 0j)   # near the basin of the 0<->-1 cycle
    for _ in range(200):
        t4.step(mandelbrot, -1 + 0j, 1)
    result = t4.classify(mandelbrot, -1 + 0j, max_iter=200, tol=1e-6)
    check(result.kind == "converged_cycle", "the basilica orbit is classified as a converged cycle")
    check(result.period == 2, "the reported period is 2, matching the basilica's known cycle")
    check(result.multiplier is not None, "a multiplier is reported for the converged cycle")
    check("period-2" in result.text, "the human-readable text names the period")

    # ---- classification: converges to infinity (sphere-first, chordal) ------------
    print("\nclassification (escaping orbit -> infinity):")
    t5 = OrbitTracker()
    t5.seed(mandelbrot, 10 + 0j, 0.5 + 0j)   # c=10 is far outside the Mandelbrot set
    for _ in range(100):
        t5.step(mandelbrot, 10 + 0j, 1)
    result5 = t5.classify(mandelbrot, 10 + 0j, max_iter=200, tol=1e-6)
    check(result5.kind == "converged_fixed",
          "an escaping orbit converges to a period-1 (fixed point) classification")
    check(result5.period == 1, "infinity is reported as a period-1 attractor")
    check("infinity" in result5.text,
          "infinity is named explicitly, not shown as some huge finite-looking number -- it IS "
          "an ordinary attractor here, per the chordal-metric house convention")

    # ---- classification: running, then undetermined within budget -----------------
    print("\nclassification (running / undetermined):")
    t6 = OrbitTracker()
    t6.seed(mandelbrot, -1 + 0j, 0.05 + 0j)
    t6.step(mandelbrot, -1 + 0j, 1)
    running = t6.classify(mandelbrot, -1 + 0j, max_iter=200, tol=1e-6)
    check(running.kind == "running", "a freshly-stepped, unconverged orbit is still 'running'")
    check(running.period is None, "no period is reported while still running")

    # A budget so tight (max_iter=1) that the orbit cannot possibly have
    # settled yet -- one step in is nowhere near the basilica's own
    # attracting cycle -- forces the genuine "undetermined" verdict
    # deterministically, rather than hoping a longer run happens not to
    # have converged yet.
    t7 = OrbitTracker()
    t7.seed(mandelbrot, -1 + 0j, 0.05 + 0j)
    t7.step(mandelbrot, -1 + 0j, 1)
    undetermined = t7.classify(mandelbrot, -1 + 0j, max_iter=1, tol=1e-6)
    check(undetermined.kind == "undetermined",
          "once n reaches max_iter with no match found, the verdict becomes a FINAL "
          "'undetermined', not 'running'")
    check("iteration budget" in undetermined.text,
          "the undetermined verdict explains it's a budget limit, not silence")
    check("1 iteration " in undetermined.text and "1 iterations " not in undetermined.text,
          "the message is grammatically singular for exactly 1 iteration")

    # ---- facts caching: memoized per (map, param) ----------------------------------
    print("\nfacts caching:")
    t8 = OrbitTracker()
    t8.seed(mandelbrot, -1 + 0j, 0.1 + 0j)
    t8.step(mandelbrot, -1 + 0j, 200)
    t8.classify(mandelbrot, -1 + 0j, max_iter=200, tol=1e-6)
    facts_first = t8._facts
    t8.classify(mandelbrot, -1 + 0j, max_iter=200, tol=1e-6)   # same (map, param) again
    check(t8._facts is facts_first,
          "classify() reuses the cached DynamicalFacts for an unchanged (map, param) -- "
          "same object, not a fresh find_attractors call every time")

    t8.classify(mandelbrot, -0.5 + 0j, max_iter=200, tol=1e-6)   # different param
    check(t8._facts is not facts_first,
          "a genuinely different parameter recomputes (and re-caches) the facts")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
