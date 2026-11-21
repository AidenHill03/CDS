"""bench/bench_approx.py -- headless timing of the Approximation tab's Basins view.

Builds r = M o F^n o S for the test images (construction.image_to_basins,
certification off: it does not affect the Basins render) and times exactly
what the panel's wheel zoom runs: scoring.render_basins at the panel's
BASIN_RESOLUTION on the default viewport and on zoomed-in ones.

    PYTHONPATH=cdx/build python -m bench.bench_approx
    ... --save bench/out/before        # keep label arrays for bench/compare.py
    ... --only sectors64

`terms/s` is resolution^2 * deg S / wall: the rate at which the first
application of S -- the pole sum -- is evaluated, if it were all the cost.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import cdx
from modules.approximation import construction, scoring
from modules.approximation.panel import ApproximationPanel

REF = os.path.join(os.path.dirname(construction.__file__), "test_reference")
CASES = (("sectors64", "sectors.png", 64), ("sectors128", "sectors.png", 128), ("spiral128", "spiral.png", 128))
# (name, wheel notches in, about which point of the default view as a fraction of its half-width)
VIEWS = (("default", 0, 0j), ("zoom4", 4, 0.3 + 0.2j), ("zoom12", 12, 0.3 + 0.2j))


def timed(fn):
    t0, c0 = time.perf_counter(), os.times()
    out = fn()
    t1, c1 = time.perf_counter(), os.times()
    return out, t1 - t0, (c1.user - c0.user) + (c1.system - c0.system)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--res", type=int, default=ApproximationPanel.BASIN_RESOLUTION)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    res = args.res
    print(f"# Basins view at {res}^2, cpus={os.cpu_count()}")
    print(f"{'case':22s} {'deg S':>7s} {'d':>2s} {'n':>2s} {'wall s':>8s} {'cores':>5s} {'terms/s':>10s} "
          f"{'mean it':>7s} {'unres %':>7s}")
    for name, image, size in CASES:
        if args.only and not any(t in name for t in args.only.split(",")):
            continue
        (r, info), build_s, _ = timed(lambda: construction.image_to_basins(
            os.path.join(REF, image), image_size=size, certify=False, grid_size=256))
        base = scoring.default_viewport(info, res)
        for vname, notches, frac in VIEWS:
            s = base.scale / 1.25 ** notches
            z = base.center + frac * base.scale          # the cursor point the zoom keeps fixed
            center = z + (base.center - z) / 1.25 ** notches
            vp = cdx.Viewport(center, s, res)
            best = None
            for _ in range(args.repeat):
                (labels, its), wall, cpu = timed(lambda: scoring.render_basins(r, info, vp))
                if best is None or wall < best[0]:
                    best = (wall, cpu)
            wall, cpu = best
            labels, its = np.asarray(labels), np.asarray(its)
            print(f"{name + '/' + vname:22s} {r.degree_S:7d} {r.d:2d} {r.n:2d} {wall:8.2f} {cpu / wall:5.2f} "
                  f"{res * res * r.degree_S / wall:10.3g} {its.mean():7.2f} {100 * (labels == 0).mean():7.3f}",
                  flush=True)
            if args.save:
                np.savez_compressed(os.path.join(args.save, f"approx__{name}__{vname}__res{res}.npz"),
                                    array=np.stack([labels, its]))
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(dict(case=f"{name}/{vname}", res=res, deg_S=r.degree_S, wall_s=wall,
                                            cores=cpu / wall, build_s=build_s)) + "\n")


if __name__ == "__main__":
    main()
