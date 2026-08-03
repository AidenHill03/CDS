"""
demo.py -- exercise the cdx Python bindings.

Build the module first:

    pip install pybind11 numpy matplotlib
    cmake -B build -DCDX_BUILD_PYTHON=ON
    cmake --build build

then run this from the directory containing the built module (or add it to
PYTHONPATH):

    python python/demo.py

Note on orientation: render arrays have row 0 at the BOTTOM of the view, so
plot with origin="lower" together with viewport.extent().
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import cdx


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label:<28} {dt:6.3f} s")
    return out


def main() -> None:
    print("cdx Python bindings demo\n" + "=" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # ---- 1. Mandelbrot parameter plane -------------------------------------
    r = cdx.Renderer(
        family="mandelbrot",
        center=complex(-0.5, 0.0),
        scale=1.5,
        resolution=900,
        max_iter=300,
        escape_radius=2.0,
    )
    M = timed("mandelbrot parameter", r.render_parameter)
    axes[0, 0].imshow(M, origin="lower", extent=r.viewport.extent(), cmap="hot")
    axes[0, 0].set_title("Mandelbrot parameter plane")

    # ---- 2. Julia set at an interesting parameter ---------------------------
    r2 = cdx.Renderer(
        family="mandelbrot",
        param=complex(-0.7269, 0.1889),
        scale=1.5,
        resolution=900,
        max_iter=300,
    )
    J = timed("julia (-0.7269+0.1889i)", r2.render_julia)
    axes[0, 1].imshow(J, origin="lower", extent=r2.viewport.extent(), cmap="hot")
    axes[0, 1].set_title("Julia set  a = -0.7269+0.1889i")

    # ---- 3. Newton basins (the classic Wada example) ------------------------
    # Three superattracting fixed points: the cube roots of unity. Each is a
    # cycle of length one. Their basins share a single common boundary, which
    # is the d = 3 Lakes-of-Wada configuration the approximation theorem is
    # about.
    roots = [
        complex(1.0, 0.0),
        complex(-0.5, 0.8660254037844386),
        complex(-0.5, -0.8660254037844386),
    ]
    cycles = [cdx.Cycle([z], i + 1) for i, z in enumerate(roots)]

    r3 = cdx.Renderer(family="newton3", scale=2.0, resolution=900, max_iter=200)
    B = timed("newton basins", lambda: r3.render_basin(cycles))
    axes[1, 0].imshow(B, origin="lower", extent=r3.viewport.extent(),
                      cmap="hsv", vmin=0, vmax=3)
    axes[1, 0].set_title("Newton $z^3-1$ basins (Wada)")

    unresolved = float(np.mean(B == 0))
    print(f"  unresolved fraction: {unresolved:.5f}")

    # ---- 4. McMullenbrot -----------------------------------------------------
    r4 = cdx.Renderer(
        family="mcmullen3",
        center=complex(0.0, 0.0),
        scale=0.3,
        resolution=900,
        max_iter=300,
        escape_radius=4.0,
    )
    P = timed("mcmullenbrot", r4.render_parameter)
    axes[1, 1].imshow(P, origin="lower", extent=r4.viewport.extent(), cmap="hot")
    axes[1, 1].set_title(r"McMullenbrot  $z^3 + \lambda/z^3$")

    for ax in axes.flat:
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")

    # ---- extras --------------------------------------------------------------
    print()
    print(f"chordal d(0, inf)  = {cdx.chordal_distance(0, complex(np.inf, 0)):.6f}"
          "   (antipodal, should be 2)")
    print(f"precision floor    = {r2.precision_floor:.3e}"
          "   (deepest resolvable half-width)")
    print(f"map degree         = {r4.map.degree}")

    # zoom demonstration: the renderer is stateful, so zooming is one call
    r2.zoom(complex(0.3, 0.4), 50.0)
    Z = timed("julia after 50x zoom", r2.render_julia)
    print(f"  new scale = {r2.viewport.scale:.5f}")

    plt.tight_layout()
    plt.savefig("cdx_demo.png", dpi=120)
    print("\nwrote cdx_demo.png")


if __name__ == "__main__":
    main()
