"""test_construction.py -- properties of the construction and its dynamical data.

Run with (from the repository root):

    PYTHONPATH=cdx/build python -m modules.approximation.test_construction
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np

import cdx
from modules.approximation import _approx_kernel as kernel
from modules.approximation import construction
from modules.approximation.construction import post_bound

REF = os.path.join(os.path.dirname(__file__), "test_reference")
failures = 0


def check(cond, what):
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


CASES = [
    ("disk.png", dict(rational_degree=150, anchor_degree=12)),                          # d = 2
    ("sectors.png", dict(image_size=50, rational_degree=900, num_basins=3)),            # d = 3
    ("sectors_p.png", dict(image_size=48, runge_shift=0.01j, boundary_colors=[0, 0, 0])),  # d = 4
]


def main():
    print("=== modules.approximation construction tests ===")

    print("\nmodel map F and the trapping bound:")
    for d in (2, 3, 4, 6):
        t = np.concatenate([[0], (d - 1) ** (1 / (d - 1)) * np.exp(1j * (2 * np.arange(d - 1) + 1) * np.pi / (d - 1))])
        F = cdx.RationalMap.from_expression(kernel.model_map_source(d), "", {})
        ok_fixed = all(abs(F.eval(complex(v), 0) - v) < 1e-12 for v in t)
        ok_crit = np.all(np.abs(kernel.post_map(d, 1, t + 1e-5) - t) < 1e-8)
        check(ok_fixed and ok_crit, f"d={d}: F fixes 0 and the roots of u^(d-1) = -(d-1), superattracting")
    check(np.isinf(post_bound(np.array([0j]), np.array([0.6]), 2)[0]),
          "postBound is Inf (not negative) when the disk reaches the pole of F")
    worst = 0.0
    for d in range(2, 8):
        t = np.concatenate([[0], (d - 1) ** (1 / (d - 1)) * np.exp(1j * (2 * np.arange(d - 1) + 1) * np.pi / (d - 1))])
        for rad in (1e-3, 0.05, 0.2):
            b = post_bound(t, np.full(d, rad), d)
            ring = t[:, None] + rad * np.exp(2j * np.pi * np.arange(2000) / 2000)[None, :]
            act = np.abs(kernel.post_map(d, 1, ring) - t[:, None]).max(axis=1)
            worst = max(worst, float(np.max(act / b)))
    check(worst <= 1 + 1e-9, f"postBound bounds |F(u)-t| on the disk (max ratio {worst:.12f}; it is sharp)")

    print("\nTaylor enclosure of S:")
    rng = np.random.default_rng(1)
    p = rng.normal(size=60) + 1j * rng.normal(size=60)
    q = rng.normal(size=60) + 1j * rng.normal(size=60)
    S = kernel.Runge(p, q, 0.2j)
    z = rng.normal(size=500) + 1j * rng.normal(size=500)
    s, B = S.taylor_bound(z, 0.03)
    ring = np.exp(2j * np.pi * np.arange(96) / 96)
    actual = np.abs(S(z[:, None] + 0.03 * ring[None, :]) - s[:, None]).max(axis=1)
    ok = np.isfinite(B)
    check(ok.sum() > 400 and np.all(actual[ok] <= B[ok] * (1 + 1e-9)), "|S(w)-S(z)| <= B on the disk")
    check(np.all(np.isinf(B[np.abs(z[:, None] - p[None, :]).min(axis=1) <= 0.03])),
          "B is Inf when a pole is inside the disk")

    print("\nfixed-degree allocation is deterministic:")
    starts = np.array([0, 1, 2, 3, 4], complex)
    lengths = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    base = construction.edge_quadrature(starts, starts + lengths, np.ones(5, complex), 0.5, 4, 17, 0)[0]
    same = True
    for eps in (1e-15, -1e-15, 3e-16):
        jit = lengths * (1 + eps * np.array([1, -1, 2, -2, 1]))
        again = construction.edge_quadrature(starts, starts + jit, np.ones(5, complex), 0.5, 4, 17, 0)[0]
        same = same and again.size == base.size and float(np.max(np.abs(again - base))) < 1e-12
    check(same, "ulp-level jitter of equal edge lengths does not change the pole set")

    for img, kw in CASES:
        print(f"\n{img} {kw}:")
        r, info = construction.image_to_basins(os.path.join(REF, img), grid_size=80, **kw)
        dyn = info["dynamics"]
        d = info["polynomial_degree"]

        # the engine's ComposedMap is the map
        cm = r.cdx_composed()
        zt = np.concatenate([rng.normal(size=200) * 0.8 + 1j * rng.normal(size=200) * 0.8,
                             [complex("inf"), info["poles"][3]]])
        check(abs(cm.degree - dyn["degree"]["r"]) <= 1e-9 * dyn["degree"]["r"],
              "cdx.ComposedMap degree = deg S * d^n")
        check(np.allclose(cm.eval_array(zt), r(zt), rtol=0, atol=0, equal_nan=True),
              "the kernel's r and cdx.ComposedMap give bit-identical values (same engine code)")
        check(np.isinf(r(info["poles"][0])) and np.isnan(r(complex("nan"))), "exact pole -> Inf, NaN -> NaN")

        # certificates
        c = dyn["certificate"]
        xs, ys, lab = c["x_edges"], c["y_edges"], c["cell_labels"]
        rows, cols = np.nonzero(lab > 0)
        pick = rng.choice(rows.size, 3000, replace=False)
        zc = (xs[cols[pick]] + (xs[1] - xs[0]) * rng.random(3000)) + 1j * (ys[rows[pick]] - (ys[0] - ys[1]) * rng.random(3000))
        got, entry = r.classify(zc, info["trap_centers"], info["trap_radii"], 3)
        check(np.array_equal(got, lab[rows[pick], cols[pick]]) and np.all(entry <= 1),
              "random points of certified cells reach the stated trap in one step")
        far = 50 * np.exp(2j * np.pi * rng.random(200)) * (1 + rng.random(200))
        check(c["exterior_certified"] and np.all(r.classify(far, info["trap_centers"], info["trap_radii"], 2)[0]
                                                 == info["outside_label"]),
              "outside the window everything is in the exterior basin (certified, and observed)")

        # critical points
        crit = dyn["critical"]
        m = info["rational_degree"]
        check(crit["count_confirmed"] and crit["all_captured"] and dyn["fatou_set_verified"],
              "critical-point count confirmed, every critical value captured, Fatou set verified")
        check(crit["layer_count_found"] + crit["in_anchor_disks"] + crit["outside_layer"] == 2 * m - 2,
              "layer + anchor disks + elsewhere = 2 deg S - 2")
        pts = crit["critical_points"]
        s1 = info["runge"].derivs(pts)[1]
        scale = np.abs(info["residues"][None, :] / (pts[:, None] - info["poles"][None, :]) ** 2).sum(axis=1)
        check(float(np.max(np.abs(s1) / scale)) < 1e-9 and crit["min_separation"] > 1e-7,
              "the computed critical points are distinct zeros of S'")
        if m == 150:
            _, ref = construction.image_to_basins(os.path.join(REF, img), grid_size=20, critical_method="pencil",
                                                  critical_check_max_degree=2000, **kw)
            dd = np.abs(pts[:, None] - ref["critical_check"]["runge_critical_points"][None, :]).min(axis=1)
            check(float(np.max(dd)) < 1e-7, "they agree with the O(m^3) eigenvalue method")

        # fixed points and multipliers
        ok = all(b["fixed_point_residual"] < 1e-12 and abs(b["fixed_point"] - b["trap_center"]) < b["trap_radius"] / 2
                 and b["trap_image_radius_bound"] <= b["trap_radius"] / 2 and b["log10_abs_multiplier"] < -10
                 for b in dyn["basins"])
        check(ok, "fixed points inside their traps, residual < 1e-12, attracting (|r'| tiny)")
        inf = dyn["infinity"]
        check(np.all(inf["relative_moments"][: inf["min_vanishing_order"] - 1] < 1e-12)
              and abs(inf["sum_residues"]) < 1e-14,
              f"S - S(Inf) vanishes to order >= {inf['min_vanishing_order']} at infinity")

        # orientation seam and summaries
        flipped = construction.to_cdx_orientation(info["basin_labels"])
        check(np.array_equal(flipped[0], info["basin_labels"][-1])
              and np.array_equal(construction.to_cdx_orientation(flipped), info["basin_labels"]),
              "to_cdx_orientation flips rows and is its own inverse")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            construction.save_summary(info, path)
            data = json.loads(open(path).read())
            check(data["fatou_set_verified"] is True and len(data["basins"]) == d
                  and "Fatou set" in open(path + ".txt").read(), "summary JSON and text report written")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
