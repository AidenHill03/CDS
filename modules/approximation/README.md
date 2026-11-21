# modules/approximation

The Fisher–Hill–Lazebnik–Thompson construction (*Simultaneous Approximation
by Attracting Basins*, arXiv:2605.09903, Section 2) from an image: a rational
map whose d attracting basins approximate the d colored regions, with what is
proved about it and how well it matches the target.

```
S(z) = c + Σ res_k/(z − p_k)        Runge approximant: Cauchy quadrature on the region edges
F(u) = u^d / (1 + d/(d−1) u^(d−1))   the model polynomial P(z) = z^d + d/(d−1) z, reciprocal chart
r    = M ∘ F^n ∘ S                   M(u) = origin + scale·u puts the attracting points in a small cluster
```

`r` has degree deg S · dⁿ (10⁶–10⁷ for ordinary images). It is never expanded:
the engine's `cdx.ComposedMap` evaluates it stage by stage.

## Build and run

```bash
cmake -B modules/approximation/build -S modules/approximation   # builds _approx_kernel next to __init__.py
cmake --build modules/approximation/build --config Release
PYTHONPATH=cdx/build python phoenix.py                            # the app, with the Approximation tab
```

From Python:

```python
from modules import approximation as ap
r, info = ap.image_to_basins("regions.png", image_size=128, progress=print)
print(ap.report(info))                       # the dynamical data, every line says how it was obtained
sc = ap.score(r, info, resolutions=(256, 512))
cm = r.cdx_composed()                        # the same map as a cdx.ComposedMap
```

Options are those of `matlab-reference/image_to_basins.m`, in snake_case or in
the MATLAB spelling, plus `certify`, `certificate_subdivision`,
`certificate_refinements`, `certificate_margin`, `critical_method`
(`"auto"`, `"aberth"`, `"pencil"`, `"none"`).

## What is computed, and what "certified" means

| `info["dynamics"]` | Contents | How |
|---|---|---|
| `degree` | deg S, d, n, exact deg r | counting |
| `basins[j]` | fixed point, residual, log10 \|r′\|, trap radius, capture radius | trap: explicit bound r(D) ⊂ D/2 (Lemma 2.10); multiplier: chain rule in the log domain |
| `infinity` | S(∞), minimum vanishing order of S − S(∞) | from the quadrature: ≥ min(2 q_min, AnchorDegree − 1) + 1 |
| `certificate` | basin label of every cell of a fine grid, area fractions, exterior | order-10 Taylor enclosure of S per cell; maximum principle outside the window |
| `critical` | critical points of S in the undecided layer, their values and capture steps | Ehrlich–Aberth, count confirmed by the argument principle on the layer boundary, tile-by-tile search if short |
| `fatou_set_verified` | every critical value of r is captured | all of the above (Remark 2.12) |

"Certified" and "proved" mean an explicit inequality evaluated in double
precision with safety factors — not interval arithmetic.

## Scoring (`scoring.score`)

* `cdx.hausdorff_distance` between J(r) (boundary pixels of an engine
  `render_basin` image) and the target's edges, chordal and Euclidean, both
  directions.
* Per basin, the Hausdorff distance between basin j and region j as pixel
  sets — the quantity simultaneous approximation is about.
* `cdx.wada_diagnostic` of target and basins at several resolutions. The
  basins always trend toward the Wada signature (J(r) is the boundary of every
  basin); the target's value says whether the paper's common-boundary
  hypothesis holds for it. When it does not, every basin still reaches every
  edge (the beads), so basin → region distances stay large however fine the
  construction: that is the theorem's hypothesis showing, not an error.

## Orientation

`info`'s rasters are in image order (row 0 = top). The engine's are row 0 =
bottom. `construction.to_cdx_orientation` is the only place they are flipped.

## Tests

```bash
PYTHONPATH=cdx/build python -m modules.approximation.test_construction
PYTHONPATH=cdx/build python -m modules.approximation.test_vs_matlab     # vs matlab-reference/image_to_basins.m (Octave output)
PYTHONPATH=cdx/build python -m modules.approximation.test_scoring
QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m modules.approximation.test_panel
```

Reference data in `test_reference/` comes from GNU Octave 8.4 running
`matlab-reference/image_to_basins.m` (`runall.m`, `bench.m`).

## Cost

Every merged raster edge gets at least `quadrature_order` poles, so fractal
targets reach deg S of 10⁴–5·10⁴ at `image_size=128`. The critical-point stage
dominates: about 2 min at deg S = 13,748 and 13 min at 46,288 on 2 cores.
Turn `certify` off for quick looks; the panel's Cancel stops a build within a
fraction of a second.
