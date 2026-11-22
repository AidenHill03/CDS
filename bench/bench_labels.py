"""bench/bench_labels.py -- the cost of the critical-point labelling convention.

    PYTHONPATH=cdx/build python -m bench.bench_labels [--res 800] [--rounds 3]

PerCritical on z^3 + a z + 1 continues the labels a* -> a per pixel (see
cdx CriticalLabelBase); Slowest does not need labels. The brief's gate:
PerCritical must not render more than 15% slower than Slowest at 800 px.
Renders are interleaved (S, P, S, P, ...) and the minimum per strategy is
reported, because this machine's load varies run to run.
"""

from __future__ import annotations

import argparse
import time

import cdx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    m = cdx.RationalMap.from_expression("z^3 + a*z + 1", "a", {}, "cubic")
    r = cdx.Renderer(map=cdx.Map.custom(m, 0j), viewport=cdx.Viewport(0j, 1.5, args.res),
                     settings=cdx.RenderSettings(200, 2.0, 1e-6, args.threads))
    best = {}
    for _ in range(args.rounds):
        for name, strat in (("slowest", cdx.ParameterStrategy.Slowest),
                            ("per_critical", cdx.ParameterStrategy.PerCritical)):
            t0 = time.perf_counter()
            r.render_parameter(None, strat, 0)
            dt = time.perf_counter() - t0
            best[name] = min(best.get(name, dt), dt)
    ratio = best["per_critical"] / best["slowest"]
    print(f"z^3+az+1, {args.res} px: slowest {best['slowest']:.3f} s, "
          f"per_critical {best['per_critical']:.3f} s, ratio {ratio:.3f} "
          f"({'within' if ratio <= 1.15 else 'OVER'} the 1.15 gate)")


if __name__ == "__main__":
    main()
