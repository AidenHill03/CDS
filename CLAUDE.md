# Project Phoenix — project instructions

Rational-map dynamics sandbox. C++ numerics core, Python bindings, migrating
off a MATLAB prototype. Goal: a distributable app for exploring rational maps
and, eventually, solving the simultaneous-approximation problem.

## Non-obvious decisions (do not "fix" these)

**Sphere-first, always.** Infinity is an ordinary point. Basin membership uses
the *chordal* metric, not Euclidean distance plus an escape radius. This is not
stylistic — the research problem (Fisher–Hill–Lazebnik–Thompson) is stated on
the Riemann sphere, where the basin of infinity is one Fatou component among
others, not a failure mode. Any code treating "escaped" as a separate outcome
from "converged to an attractor" is wrong.

**Attractors are discovered, never assumed.** Attracting *cycles*, not just
fixed points. The basilica (z²−1) has **zero** attracting fixed points but a
perfectly good attracting 2-cycle; fixed-point-only logic silently renders it
as empty. Discovery goes through critical orbits (Fatou: every attracting cycle
attracts a critical point).

**The model polynomial coefficient is d/(d−1), not d/(d+1).** In
P(z) = z^d + (d/(d−1))z, that coefficient is exactly what makes every critical
point *fixed*, which is what the paper's Proposition 2.13 depends on. With
d/(d+1) the critical points map to multiples of themselves and the whole
guarantee evaporates. This cost real debugging time.

**No `-ffast-math`.** The escape tests and NaN guards depend on IEEE
semantics.

**Hand-rolled real/imag arithmetic in hot loops, not `std::complex`.**
`operator*` carries inf/nan branch handling that measurably costs. `std::complex`
is fine in the API and cold paths.

**`imagesc`/imshow orientation.** Render arrays have row 0 at the **bottom**
(matching `Viewport::coord`). Plot with `origin="lower"`. In the MATLAB
prototype, forgetting this per-call caused a visible flip during progressive
rendering.

## Pitfalls hit before

- **Progressive rendering:** set the axis orientation immediately after every
  draw call, not once at the end, or the preview flips relative to the full
  render.
- **Marker/overlay plotting expands axes limits**, silently breaking
  drag-zoom because the axes limits no longer match the rendered window. Pin
  limits after drawing any overlay.
- **Green's function must normalize at each pixel's own escape iteration,
  not against one global `degree^max_iter`.** The original MATLAB-ported
  form accumulated `log(max(|z|,1))` over the whole orbit and divided by
  `degree^max_iter` once at the end. That wasn't just overflow-prone
  (`deg^max_iter` blows past double range for even moderate `max_iter` —
  2^200 is already out of range) — it was mathematically wrong: on
  z²+c the accumulated sum saturates around 58 while `degree^max_iter`
  is astronomically larger, so `G` came out ~1e-59 everywhere, a
  constant field that renders flat gray at any color scaling. The
  correct escape-rate potential is `G(z) = log|z_n| / degree^n`, using
  the pixel's OWN escape iteration `n`, not `max_iter` — non-escaping
  pixels are exactly 0. This also removes the overflow case entirely
  (`degree^n` is only ever evaluated at one pixel's own, usually small,
  escape step) rather than needing to detect and report it. Verify any
  future change here against the defining identity `G(f(z)) = degree ·
  G(z)` for an escaping point — a much stronger check than any golden
  image, and the one that would have caught the original formula
  immediately.
- **Click-vs-drag thresholds belong in screen pixels**, not data units;
  otherwise deep zoom reinterprets ordinary drags as clicks.
- **Zoom has a real double-precision floor** (~1e-14 relative). This is
  arithmetic, not state that can be cleared. Beating it needs extended
  precision or perturbation theory.
- **A superseded background render is still running**, not cancelled --
  `cdx`'s render calls have no interrupt mechanism, so discarding a stale
  result does not stop the thread computing it. In `app/sandbox.py` this
  means anything that tears down Qt objects (closing the window) must drain
  the thread pool first, or a late-arriving signal fires into already-
  destroyed objects and crashes.
- **PySide6's plugin path is not always on Qt's default search path** (seen
  with the anaconda-installed wheel on macOS: `Could not find the Qt
  platform plugin "cocoa"/"offscreen" in ""`). Fix by pointing
  `QT_QPA_PLATFORM_PLUGIN_PATH` at
  `<PySide6 install dir>/Qt/plugins/platforms` explicitly.

## Conventions

- Measure before optimizing. Several confident performance hypotheses in this
  project were wrong; profiling settled them in one run each. Prefer adding an
  instrument to guessing.
- Every renderer consumes the same map representation. Adding a family should
  not require touching renderers.
- Tests assert *properties* (symmetries, known membership facts, metric
  identities), not golden images. This catches geometric bugs that a diff
  against a reference implementation would miss.

## Build

```bash
cd cdx && cmake -B build -DCDX_BUILD_PYTHON=ON && cmake --build build
./build/cdx_test                     # C++ tests
PYTHONPATH=build python python/demo.py
```

Interactive sandbox (from the repository root, requires PySide6):

```bash
PYTHONPATH=cdx/build python -m app.sandbox
```

## Current state

Done: C++ core (`renderer.hpp/cpp`) with Julia/parameter/basin/Green's modes,
threaded, ~22× the original MATLAB. pybind11 bindings with zero-copy NumPy and
GIL released during render. Expression evaluator (`expr.hpp/cpp`) for
user-typed formulas. Term-based sandbox (`rational.hpp/cpp`) with poles as
first-class editable objects, plus a serializable family library.

Open: wiring `RationalMap` into `Renderer`; a complex polynomial root-finder
(Durand–Kerner or Aberth) so custom maps get critical points and therefore
parameter-plane rendering; dynamical data extraction.

See ARCHITECTURE.md for layering rules
