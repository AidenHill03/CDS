"""bench/compare.py -- faithfulness check between two benchmark runs.

Compares every .npz the two --save directories have in common, layer by layer
and pixel by pixel. "identical" means bit-for-bit equal (NaNs in the same
places count as equal). Anything else is listed: how many pixels differ, the
largest absolute difference, and the first few (layer, row, col, before,
after).

    python -m bench.compare bench/out/before bench/out/after
Exit status 1 if any array differs.
"""

from __future__ import annotations

import os
import sys

import numpy as np


def main():
    before, after = sys.argv[1], sys.argv[2]
    names = sorted(set(os.listdir(before)) & set(os.listdir(after)))
    only = sorted(set(os.listdir(before)) ^ set(os.listdir(after)))
    bad = 0
    total_px = 0
    for name in names:
        if not name.endswith(".npz"):
            continue
        a = np.load(os.path.join(before, name))["array"]
        b = np.load(os.path.join(after, name))["array"]
        if a.shape != b.shape:
            print(f"SHAPE   {name}: {a.shape} vs {b.shape}")
            bad += 1
            continue
        total_px += a.size
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if same.all():
            print(f"identical  {name}  {a.shape}")
            continue
        bad += 1
        idx = np.argwhere(~same)
        diff = np.abs(np.where(same, 0.0, a - b))
        print(f"DIFFERS    {name}: {len(idx)} of {a.size} values ({100 * len(idx) / a.size:.4f}%), "
              f"max |diff| {np.nanmax(diff):.3g}")
        for i in idx[:10]:
            print(f"             at {tuple(int(v) for v in i)}: {a[tuple(i)]!r} -> {b[tuple(i)]!r}")
    print(f"\n{len(names)} arrays compared, {total_px} values, {bad} differing"
          + (f"; {len(only)} files present on one side only" if only else ""))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
