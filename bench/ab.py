"""bench/ab.py -- interleaved A/B timing of two engine builds.

This machine has two cores and other processes take a varying share of them,
so one run before and one run after can differ by 2x from load alone. This
runs the same benchmark command alternately against two engine directories
(A, B, A, B, ...), each in its own process (an extension module can only be
loaded once), and reports the MINIMUM wall time per case for each -- the run
least disturbed -- and the ratio.

    python -m bench.ab bench/out/baseline_engine cdx/build --rounds 3 -- \
        -m bench.bench_newton --res 100 --no-preview --only n5/parameter_basin/wide

Everything after `--` is the python command line; `--json` is added for it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


def main():
    argv = sys.argv[1:]
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args(argv[:split])
    cmd = argv[split + 1:]

    best = {"A": {}, "B": {}}
    order = []
    with tempfile.TemporaryDirectory() as tmp:
        for rnd in range(args.rounds):
            for tag, engine in (("A", args.a), ("B", args.b)):
                out = os.path.join(tmp, f"{tag}{rnd}.jsonl")
                env = dict(os.environ, PYTHONPATH=engine, QT_QPA_PLATFORM="offscreen")
                subprocess.run([sys.executable, *cmd, "--json", out], check=True, env=env,
                               stdout=subprocess.DEVNULL)
                for line in open(out):
                    rec = json.loads(line)
                    wall = rec.get("full_s", rec.get("wall_s"))
                    if rec["case"] not in order:
                        order.append(rec["case"])
                    best[tag][rec["case"]] = min(wall, best[tag].get(rec["case"], float("inf")))
    print(f"A = {args.a}\nB = {args.b}\nminimum of {args.rounds} interleaved runs each")
    print(f"{'case':38s} {'A s':>9s} {'B s':>9s} {'A/B':>7s}")
    for case in order:
        a, b = best["A"][case], best["B"][case]
        print(f"{case:38s} {a:9.2f} {b:9.2f} {a / b:7.2f}x")


if __name__ == "__main__":
    main()
