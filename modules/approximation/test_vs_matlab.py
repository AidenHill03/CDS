"""test_vs_matlab.py -- the construction against matlab-reference/image_to_basins.m.

Reference output was produced in GNU Octave 8.4 by test_reference/dump.m
(regenerate with: cd modules/approximation/test_reference && octave --path
../../../matlab-reference runall.m). Run with:

    PYTHONPATH=cdx/build python -m modules.approximation.test_vs_matlab

from the repository root.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.io

from modules.approximation import construction

REF = os.path.join(os.path.dirname(__file__), "test_reference")
CASES = {
    "ref1": ("sectors.png", dict(grid_size=120, image_size=64)),
    "ref2": ("disk.png", dict(grid_size=100, rational_degree=150, anchor_degree=12)),
    "ref3": ("sectors_p.png", dict(GridSize=[70, 90], ImageSize=48, InnerIterations=8,
                                   RungeShift=0.01j, BoundaryColors=[0, 0, 0])),
    "ref4": ("sectors.png", dict(grid_size=110, image_size=50, rational_degree=900,
                                 plot_bounds=[-2, 2, -1.5, 1.5], max_iterations=40, num_basins=3)),
}

failures = 0


def check(cond, what):
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if not (np.array_equal(np.isfinite(a), np.isfinite(b)) and np.array_equal(np.isnan(a), np.isnan(b))):
        return np.inf
    fin = np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        return 0.0
    return float(np.max(np.abs(a[fin] - b[fin])) / max(1.0, np.max(np.abs(b[fin]))))


def main():
    print("=== modules.approximation vs image_to_basins.m (Octave) ===")
    for name in sorted(CASES):
        img, kw = CASES[name]
        print(f"\n{name} ({img}, {kw}):")
        m = scipy.io.loadmat(os.path.join(REF, name + ".mat"), squeeze_me=True)
        r, info = construction.image_to_basins(os.path.join(REF, img), critical_method="pencil", **kw)
        check(np.array_equal(info["target_labels"], m["targetLabels"]), "target labels identical")
        check(np.allclose(info["palette"], m["palette"], atol=1e-12), "palette identical")
        check(info["outside_label"] == m["outside"] and info["inner_iterations"] == m["n"],
              "exterior label and n identical")
        check(info["poles"].size == m["poles"].size, "same number of poles")
        check(rel(info["poles"], m["poles"]) < 1e-13 and rel(info["residues"], m["residues"]) < 1e-13,
              "poles and residues agree to 1e-13")
        check(rel(info["trap_radii"], m["trapRadii"]) < 1e-10, "trap radii agree to 1e-10")
        check(rel(info["fixed_points"], m["fixedPoints"]) < 1e-12, "fixed points agree to 1e-12")
        zt = np.atleast_1d(m["zt"]).astype(complex)
        check(rel(info["runge"](zt), m["sz"]) < 1e-11, "S agrees at test points (incl. Inf, NaN, a pole)")
        check(rel(r(zt), m["rz"]) < 1e-9, "r (engine ComposedMap) agrees at the same points")
        agree = float(np.mean(info["basin_labels"] == m["basinLabels"]))
        check(agree > 0.999, f"basin labels agree on {100 * agree:.3f}% of the grid "
                             f"(differences only where ulps are amplified on the Julia set)")
        if np.isfinite(m["coreErr"]):
            check(abs(info["core_max_runge_error"] - m["coreErr"]) <= 1e-10 * max(1, abs(m["coreErr"]))
                  and info["core_sample_count"] == m["coreCount"], "interior-core diagnostics agree")
        crit = info["critical_check"]
        check(crit["performed"] == bool(m["critPerformed"]), "critical check performed iff MATLAB's was")
        if m["critPerformed"]:
            ours = crit["runge_critical_points"]
            theirs = np.atleast_1d(m["critPoints"]).astype(complex)
            check(ours.size == theirs.size and crit["near_infinity"] == m["critInf"],
                  "same number of finite critical points and of near-infinite ones")
            d = np.abs(ours[:, None] - theirs[None, :]).min(axis=1)
            check(float(np.max(d)) < 1e-7, "critical points agree to 1e-7")
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
