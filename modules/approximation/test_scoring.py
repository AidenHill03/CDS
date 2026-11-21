"""test_scoring.py -- Wada/Hausdorff scoring through the engine's analyses.

Run with (from the repository root):

    PYTHONPATH=cdx/build python -m modules.approximation.test_scoring
"""

from __future__ import annotations

import os

import numpy as np

import cdx
from modules.approximation import construction, scoring

REF = os.path.join(os.path.dirname(__file__), "test_reference")
failures = 0


def check(cond, what):
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main():
    print("=== modules.approximation scoring tests ===")

    print("\norientation at the seam:")
    labels = np.ones((40, 60), int)          # image order: row 0 = top
    labels[:20] = 2                          # top half is label 2
    info = dict(target_labels=labels, image_bounds=np.array([-1.0, 1.0, -2 / 3, 2 / 3]),
                outside_label=1, palette=np.eye(3)[:2])
    vp = cdx.Viewport(0j, 1.0, 101)
    t = scoring.target_on_viewport(info, vp)
    # cdx order: row 0 = BOTTOM. A point with Im z = +0.5 is in the top half -> label 2.
    row_top = int(round((0.5 - (-1.0)) / vp.pixel_size))
    row_bot = int(round((-0.5 - (-1.0)) / vp.pixel_size))
    check(t[row_top, 50] == 2 and t[row_bot, 50] == 1, "top of the image is positive Im z on the cdx grid")
    check(t[100, 50] == 1 and t[0, 50] == 1, "outside the image: the exterior label")

    print("\nper-basin Hausdorff distance:")
    a = np.zeros((50, 50))
    a[:, :25] = 1
    a[:, 25:] = 2
    h0 = scoring.per_basin_hausdorff(a, a.copy(), 2, 0.1)
    check(all(x["hausdorff_px"] == 0 for x in h0), "identical label images: 0 for every basin")
    b = np.zeros((50, 50))
    b[:, :28] = 1
    b[:, 28:] = 2
    h3 = scoring.per_basin_hausdorff(a, b, 2, 0.1)
    check(h3[0]["basin_to_region_px"] == 3 and h3[0]["region_to_basin_px"] == 0
          and h3[1]["region_to_basin_px"] == 3 and abs(h3[1]["hausdorff"] - 0.3) < 1e-12,
          "an edge moved by 3 pixels: directed distances 3 and 0, in pixels and in plane units")
    bead = a.copy()
    bead[10, 40] = 1                          # one pixel of basin 1 deep inside region 2
    hb = scoring.per_basin_hausdorff(a, bead, 2, 1.0)
    check(hb[0]["basin_to_region_px"] == 16, "a single stray bead pixel sets basin->region (16 px here)")

    print("\nscoring a built map (sectors.png, d = 4):")
    r, info = construction.image_to_basins(os.path.join(REF, "sectors.png"), image_size=64, grid_size=60)
    sc = scoring.score(r, info, resolutions=(128, 256, 512))
    labels512 = sc["basin_labels"]
    check(labels512.shape == (512, 512) and set(np.unique(labels512)) <= {0, 1, 2, 3, 4},
          "engine render on the default viewport")
    # Every label render_basin assigns comes from inside a certified trap.
    vp = scoring.default_viewport(info, 512)
    rr, cc = np.nonzero(labels512 > 0)
    pick = np.random.default_rng(0).choice(rr.size, 2000, replace=False)
    z = np.array([complex(vp.center.real - vp.scale + vp.pixel_size * c,
                          vp.center.imag - vp.scale + vp.pixel_size * row) for row, c in zip(rr[pick], cc[pick])])
    got, _ = r.classify(z, info["trap_centers"], info["trap_radii"], 60)
    check(np.array_equal(got, labels512[rr[pick], cc[pick]]),
          "engine render labels = trap classification (tolerance only fires inside a trap)")
    wb = [w["wada_fraction"] for w in sc["wada_basins"]]
    check(wb[0] <= wb[1] <= wb[2], f"basins' Wada fraction rises with resolution ({wb})")
    wt = [w["wada_fraction"] for w in sc["wada_target"]]
    check(max(wt) < 0.05, "the target (sectors meeting along arcs) is not Wada")
    h = sc["hausdorff"]
    check(h["euclidean"] / sc["pixel_size"] <= 5, "J(r) follows the target edges within 5 pixels")
    check(all(b["region_to_basin_px"] <= 4 for b in sc["per_basin"]),
          "every region is within 4 px of its basin (region -> basin)")
    check(all(b["basin_to_region_px"] > 20 for b in sc["per_basin"]),
          "every basin has beads far from its own region (basin -> region): no common boundary in the target")
    check("Hausdorff" in scoring.score_report(sc), "score_report renders")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
