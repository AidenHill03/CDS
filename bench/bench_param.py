"""bench/bench_param.py -- parameter-plane timing cases, cdx-only (no app
import), so the same command runs against any engine build. Use with
bench/ab.py for interleaved A/B:

    python -m bench.ab <engineA> <engineB> --rounds 3 -- -m bench.bench_param --only typed

Each case prints (and, with --json PATH, appends) {"case", "wall_s"}: the
best of --repeat runs inside this process.
"""

from __future__ import annotations

import argparse
import json
import time

import cdx


def _r(m, center, scale, res, max_iter=200):
    return cdx.Renderer(map=cdx.Map.custom(m, 0j), viewport=cdx.Viewport(center, scale, res),
                        settings=cdx.RenderSettings(max_iter, 2.0, 1e-6, 0))


def cases(res):
    typed = cdx.RationalMap.from_expression("z^2+a", "a", {}, "typed")
    cubic = cdx.RationalMap.from_expression("z^3+a*z+1", "a", {}, "cubic")
    lib = cdx.FamilyLibrary.with_defaults()
    rn5 = cdx.RationalMap.from_expression("z - a*(z^5-1)/(5*z^4)", "a", {}, "rn5")
    out = {
        "parameter/z2+a typed": lambda: _r(typed, complex(-0.5, 0), 1.5, res).render_parameter(),
        "parameter/z3+az+1": lambda: _r(cubic, 0j, 1.5, res).render_parameter(),
        "parameter_basin/mcmullen2": lambda: _r(lib.find("mcmullen2"), 0j, 0.3, res).render_parameter_basin(),
    }
    ff = getattr(cdx, "CriticalPointFamily", None)
    if ff is not None and hasattr(ff, "FromMap"):
        out["parameter_period/mandelbrot"] = lambda: _r(
            lib.find("mandelbrot"), complex(-0.75, 0), 1.5, res).render_parameter_period(ff.FromMap, 0)
    if ff is not None:
        out["parameter_period/relaxed newton n=5"] = lambda: _r(
            rn5, complex(5, 0), 6.0, res).render_parameter_period(ff.RelaxedNewtonPower, 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    for name, fn in cases(args.res).items():
        if args.only and not any(s in name for s in args.only.split(",")):
            continue
        best = float("inf")
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        print(f"{name:40s} {best:8.3f} s")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps({"case": name, "wall_s": best}) + "\n")


if __name__ == "__main__":
    main()
