"""bench/bench_newton.py -- headless timing of the relaxed Newton family
N_a(z) = z - a (z^n - 1) / (n z^(n-1)) through the SAME code path the main
panes use: app.session.render_map on a P/Q-backed cdx.RationalMap built by
RationalMap.from_expression (what typing the formula into the app produces),
with RenderTask's two stages (quarter-resolution preview at 2x overscan, then
the full render at 1.3x overscan).

Not a test suite: it prints timings and (with --save) writes every rendered
array to an .npz so a later run can be compared pixel for pixel with
bench/compare.py.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m bench.bench_newton --res 800
    ... --res 200 --only n5            # quick look
    ... --save bench/out/before        # keep arrays for the faithfulness check

Settings default to the ones saved in ~/.complexdynamics/settings.json
(--app-defaults uses app.settings.Settings() instead). The render cache is
off, so every number is compute time.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import os
import time

import numpy as np

import cdx
from app.sandbox import FULL_OVERSCAN_FACTOR, PREVIEW_OVERSCAN_FACTOR, PREVIEW_RESOLUTION_DIVISOR, _overscanned
from app.session import render_map
from app.settings import Settings, load_settings


def relaxed_newton(n):
    return cdx.RationalMap.from_expression(f"z - a*(z^{n}-1)/({n}*z^{n - 1})", "a", {}, f"relaxed_newton{n}")


def centre(n, k):
    """Superattracting centre a_k = n - (n-1) w_k (T4)."""
    return n - (n - 1) * cmath.exp(2j * math.pi * k / n)


def cases(n):
    """(name, mode, bound parameter, view centre, view half-width)."""
    a1 = centre(n, 1)
    out = []
    for mode in ("parameter", "parameter_basin", "parameter_period"):
        # both circles |a-n| = n-1 and |a-n| = n-2, and every centre
        out.append((f"n{n}/{mode}/wide", mode, 1 + 0j, complex(n), float(n)))
        # one centre and the cusps beside it, on |a-n| = n-1
        out.append((f"n{n}/{mode}/centre1", mode, 1 + 0j, a1, 0.3))
    for label, a in (("a=1", 1 + 0j),                 # plain Newton
                     ("a=centre1", a1),               # a superattracting star cycle (T4)
                     ("a=1.98", 1.98 + 0j)):          # roots' multiplier 1-a = -0.98: near-parabolic
        for mode in ("julia", "basin", "greens"):
            out.append((f"n{n}/{mode}/{label}", mode, a, 0j, 2.0))
    return out


def timed(fn):
    t0, c0 = time.perf_counter(), os.times()
    out = fn()
    t1, c1 = time.perf_counter(), os.times()
    wall = t1 - t0
    cpu = (c1.user - c0.user) + (c1.system - c0.system)
    return out, wall, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=None, help="pane resolution (default: the saved setting)")
    ap.add_argument("--only", default="", help="substring filter on case names, comma-separated")
    ap.add_argument("--skip", default="", help="substring filter on case names to skip, comma-separated")
    ap.add_argument("--n", default="5,8")
    ap.add_argument("--save", default="", help="directory for the rendered arrays (.npz)")
    ap.add_argument("--app-defaults", action="store_true")
    ap.add_argument("--no-preview", action="store_true", help="time the full stage only")
    ap.add_argument("--json", default="", help="append one JSON line per case to this file")
    args = ap.parse_args()

    s = Settings() if args.app_defaults else load_settings()
    res = args.res or s.resolution
    rs = cdx.RenderSettings(s.max_iter, s.escape_radius, s.tol, s.threads)
    print(f"# res={res} (full buffer {round(res * FULL_OVERSCAN_FACTOR)}^2, preview "
          f"{round(max(1, res // PREVIEW_RESOLUTION_DIVISOR) * PREVIEW_OVERSCAN_FACTOR)}^2)  "
          f"max_iter={s.max_iter} tol={s.tol:g} threads={s.threads} "
          f"strategy={s.parameter_strategy}/{s.parameter_critical_index} potential={s.greens_potential}  "
          f"cpus={os.cpu_count()}")
    print(f"{'case':38s} {'preview s':>9s} {'full s':>9s} {'cores':>5s} {'us/px':>8s}")
    only = [t for t in args.only.split(",") if t]
    skip = [t for t in args.skip.split(",") if t]
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    for n in [int(v) for v in args.n.split(",")]:
        m = relaxed_newton(n)
        for name, mode, a, c, half in cases(n):
            if only and not any(t in name for t in only):
                continue
            if any(t in name for t in skip):
                continue
            vp = cdx.Viewport(c, half, res)
            kw = dict(cancel=None, cache=None, potential=s.greens_potential,
                      strategy=s.parameter_strategy, critical_index=s.parameter_critical_index)
            pre_wall = 0.0
            if not args.no_preview:
                pvp = _overscanned(cdx.Viewport(c, half, max(1, res // PREVIEW_RESOLUTION_DIVISOR)),
                                   PREVIEW_OVERSCAN_FACTOR)
                _, pre_wall, _ = timed(lambda: render_map(m, a, pvp, rs, mode, **kw))
            fvp = _overscanned(vp, FULL_OVERSCAN_FACTOR)
            arr, wall, cpu = timed(lambda: render_map(m, a, fvp, rs, mode, **kw))
            px = fvp.resolution ** 2
            print(f"{name:38s} {pre_wall:9.2f} {wall:9.2f} {cpu / wall:5.2f} {1e6 * wall / px:8.2f}", flush=True)
            if args.save:
                np.savez_compressed(os.path.join(args.save, name.replace("/", "__") + f"__res{res}.npz"),
                                    array=np.asarray(arr))
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(dict(case=name, res=res, buffer=fvp.resolution, preview_s=pre_wall,
                                            full_s=wall, cpu_s=cpu, cores=cpu / wall)) + "\n")


if __name__ == "__main__":
    main()
