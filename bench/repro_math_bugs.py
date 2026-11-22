"""bench/repro_math_bugs.py -- reproduce (and later, confirm fixed) the two
parameter-plane bugs written up in notes/2026-11-21-math-bugs.md, and time
the renders they touch.

    PYTHONPATH=cdx/build python -m bench.repro_math_bugs [--save TAG] [--res N]

Bug A: single-pixel "parameter_period" on the Mandelbrot family.
Bug B: "parameter" on z^3 + a z + 1 (centre 0, scale 1.5) -- do the three
       strategies differ, and is the image conjugation-symmetric across
       Im a = 0 (the map has real coefficients, so f_conj(a) is conjugate to
       f_a and every strategy symmetric in the critical points must be too)?

--save TAG writes PNGs to bench/out/<TAG>_*.png (bench/out is git-ignored).
Everything goes through app.session.render_map, the app's own dispatch.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import cdx
from app.session import render_map

MANDEL_CASES = [
    (0j, "1"),
    (-1 + 0j, "2"),
    (-1.31 + 0j, "4"),
    (-1.7549 + 0j, "3"),
    (-1.1 + 0.2j, "2"),
    (-0.1226 + 0.7449j, "3"),
    # 0.25+0.5i is the root of the period-4 bulb (fixed point i/2, multiplier
    # exactly i): parabolic, so the honest answer is "undetermined". The
    # original write-up listed it as "escapes" -- what the wrong seeds said.
    (0.25 + 0.5j, "undetermined"),
    (0.5 + 0.5j, "escapes"),
]


def settings(threads=0):
    return cdx.RenderSettings(200, 2.0, 1e-6, threads)


def period_label(stack, r=1, c=1):
    periods, undetermined, is_inf = stack[0], stack[1], stack[2]
    if is_inf[r, c] == 1.0:
        return "escapes"
    if undetermined[r, c] == 1.0:
        return "undetermined"
    return str(int(periods[r, c]))


def timed(fn, repeat=3):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def bug_a():
    print("Bug A -- Mandelbrot parameter_period, single pixel")
    print(f"  {'a':>20}  {'true':>8}  {'engine':>12}")
    m = cdx.RationalMap.mandelbrot()
    bad = 0
    for a, want in MANDEL_CASES:
        vp = cdx.Viewport(a, 1e-9, 3)
        got = period_label(render_map(m, 0j, vp, settings(1), "parameter_period"))
        flag = "" if got == want else "   <-- WRONG"
        bad += got != want
        print(f"  {str(a):>20}  {want:>8}  {got:>12}{flag}")
    return bad


def cubic():
    return cdx.RationalMap.from_expression("z^3 + a*z + 1", "a", {}, "cubic_az1")


def seam_fraction(img, view):
    """Fraction of pixels with Re a in [-1.5, -0.45] whose bounded/escaped
    status differs from the conjugate pixel's (row r <-> row res-1-r)."""
    res = img.shape[0]
    re = view.center.real - view.scale + 2 * view.scale / (res - 1) * np.arange(res)
    cols = (re >= -1.5) & (re <= -0.45)
    a = img[:, cols] > 0
    b = np.flipud(img)[:, cols] > 0
    return float(np.mean(a != b))


def bug_b(res, save_tag):
    print(f"\nBug B -- z^3 + a z + 1 parameter plane, centre 0, scale 1.5, {res} px")
    m = cubic()
    vp = cdx.Viewport(0j, 1.5, res)
    imgs = {}
    for strat in ("slowest", "fastest", "per_critical"):
        t, img = timed(lambda: render_map(m, 0j, vp, settings(), "parameter", strategy=strat),
                       repeat=1)
        imgs[strat] = img
        print(f"  {strat:>12}: {t:6.2f} s, black fraction {np.mean(img == 0):.3f}, "
              f"conjugation-asymmetric fraction (Re a in [-1.5,-0.45]) {seam_fraction(img, vp):.4f}")
    if "per_critical" in imgs:
        t1, img1 = timed(lambda: render_map(m, 0j, vp, settings(), "parameter",
                                            strategy="per_critical", critical_index=1), repeat=1)
        imgs["per_critical_1"] = img1
        print(f"  per_critical[1]: {t1:6.2f} s")
    same = (np.array_equal(imgs["slowest"], imgs["fastest"]) and
            np.array_equal(imgs["slowest"], imgs["per_critical"]))
    print(f"  slowest/fastest/per_critical byte-identical: {same}")
    if save_tag:
        save_param_pngs(imgs, vp, f"{save_tag}_bugB")
    return same


def timings(res):
    print(f"\nTimings ({res} px, default settings, all threads, best of 3)")
    vp = cdx.Viewport(complex(-0.5, 0.0), 1.5, res)
    t, _ = timed(lambda: render_map(cdx.RationalMap.mandelbrot(), 0j, vp, settings(),
                                    "parameter"))
    print(f"  parameter        z^2 + a        : {t * 1000:8.1f} ms")
    pvp = cdx.Viewport(complex(-0.75, 0.0), 1.5, 200)
    t, _ = timed(lambda: render_map(cdx.RationalMap.mandelbrot(), 0j, pvp, settings(),
                                    "parameter_period"), repeat=1)
    print(f"  parameter_period z^2 + a, 200 px: {t * 1000:8.1f} ms")
    n = 5
    rn = cdx.RationalMap.from_expression(f"z - a*(z^{n}-1)/({n}*z^{n - 1})", "a", {}, "rn5")
    rvp = cdx.Viewport(complex(n, 0.0), n + 1.0, 200)
    t, _ = timed(lambda: render_map(rn, 0j, rvp, settings(), "parameter_period"), repeat=1)
    print(f"  parameter_period relaxed N n=5, 200 px: {t * 1000:8.1f} ms")


def save_param_pngs(imgs, vp, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("bench/out", exist_ok=True)
    ext = [vp.center.real - vp.scale, vp.center.real + vp.scale,
           vp.center.imag - vp.scale, vp.center.imag + vp.scale]
    fig, axes = plt.subplots(1, len(imgs), figsize=(4.2 * len(imgs), 4.4))
    for ax, (name, img) in zip(np.atleast_1d(axes), imgs.items()):
        shade = np.where(img > 0, np.log1p(np.abs(img)), np.nan)
        ax.imshow(shade, origin="lower", extent=ext, cmap="magma")
        ax.set_facecolor("black")
        ax.set_title(name)
    fig.tight_layout()
    path = f"bench/out/{prefix}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved {path}")


def save_period_png(save_tag, res=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("bench/out", exist_ok=True)
    vp = cdx.Viewport(complex(-0.75, 0.0), 1.5, res)
    stack = render_map(cdx.RationalMap.mandelbrot(), 0j, vp, settings(), "parameter_period")
    periods = stack[0].copy()
    periods[stack[2] == 1.0] = np.nan          # escapes
    ext = [-2.25, 0.75, -1.5, 1.5]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_facecolor("black")
    im = ax.imshow(np.where(periods > 0, periods, np.nan), origin="lower", extent=ext,
                   cmap="tab20", vmin=0.5, vmax=20.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="period")
    ax.set_title(f"Mandelbrot parameter_period ({save_tag})")
    path = f"bench/out/{save_tag}_bugA.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--skip-timings", action="store_true")
    args = ap.parse_args()
    bug_a()
    if args.save:
        save_period_png(args.save)
    bug_b(min(args.res, 400), args.save)
    if not args.skip_timings:
        timings(args.res)


if __name__ == "__main__":
    main()
