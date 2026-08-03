# Architecture

The project is a **rational dynamics sandbox** with research modules layered on
top. This document fixes the layering so that adding a research capability
means adding files, not editing the engine.

## Layers

```
  modules/          optional, domain-specific research code
      |             (approximation, pole-geometry inversion, ...)
      v
  app/              the sandbox application: session state, UI, export
      |
      v
  cdx/              the engine: maps, rendering, roots, analysis
```

**Dependencies point inward only.** `cdx/` knows nothing about `app/`, and
neither knows anything about `modules/`. A module may depend on the engine and
the app; nothing may depend on a module. If a module needs an engine change,
that change must be justifiable as a *general* dynamics capability — otherwise
it belongs in the module.

## What lives where

### `cdx/` — the engine

Everything that is true of rational maps in general, independent of what you
are studying.

- `rational` — `RationalMap` as editable terms; poles as first-class objects
- `expr` — parser/evaluator for user-typed formulas
- `roots` — Aberth–Ehrlich polynomial root finder
- `renderer` — Julia / parameter / basin / Green's rendering
- `analysis` — `find_attractors`, `wada_diagnostic`, `hausdorff_distance`

Test by mathematical property (symmetries, Riemann–Hurwitz counts, metric
identities), never by golden image.

### `app/` — the sandbox

The application a user actually operates. Owns mutable session state; the
engine stays stateless apart from `Renderer`'s configuration.

- current map, parameter, viewport, render mode
- the family library (save / name / load)
- term editing operations and undo
- data extraction and export
- animation export

### `modules/` — research code

Self-contained investigations. Each owns its own data types, its own UI panel
if it has one, and its own tests.

## The seam

Research modules attach at exactly one place: an **analysis** is a function
from *(map, parameter, and optionally a rendered image)* to structured data.
That is the shape `wada_diagnostic` and `hausdorff_distance` already have, and
it is enough for everything currently foreseen.

Keep it a plain function signature. **Do not build a plugin loader, a
registry, or an abstract base class hierarchy** until there are at least three
modules that demonstrably need one. The project convention is to measure
before optimizing; the same applies to abstraction. Premature extension
machinery is harder to remove than to add.

## Deferred: the approximation module

`verify_conditions` for the Fisher–Hill–Lazebnik–Thompson construction is
**not** engine code, and this is worth recording because it is easy to
misfile.

The construction is
$$S_n = P_\lambda^n \circ R$$
where $P_\lambda(z) = z^d + \tfrac{d}{d-1}z$ is the model polynomial (all $d$
critical points fixed) and $R$ is a Runge approximant. Conditions (2.4) and
(2.5), from Notation 2.5 and Proposition 2.8 of the paper:

$$A_j^\varepsilon := \{z \in A_j : \operatorname{dist}(z,J) > \varepsilon\}
  \;\cup\; B_j^\varepsilon \;\cup\; D(\lambda\xi_j, \delta)$$

$$\textbf{(2.4)}\quad R(A_j^\varepsilon) \subset D(\lambda\xi_j, \delta)
\qquad\qquad
\textbf{(2.5)}\quad \mathrm{CV}(R) \cap \mathcal{J}(P_\lambda) = \varnothing$$

Two points that caused a wrong implementation once and should not again:

1. **$A_j$ is the TARGET region** supplied by the user (typically from an
   image), *not* a computed basin of the map being rendered. $A_j^\varepsilon$
   is that target eroded away from the common boundary $J$, plus a small disc
   around the model's $j$-th fixed critical point.
2. **$R$ is the preprocessing map, not the iterated map.** Applying $R$ once
   to points of a rendered basin is meaningless — those points reach their
   attractor only after a full orbit under $S_n$, not after one application of
   $R$.

So the module needs three inputs the sandbox has no notion of: target regions,
a candidate $R$, and the model map $P_\lambda$. That is precisely why it is a
module and not engine code.

Implement it when the approximation research resumes, in
`modules/approximation/`, alongside the Runge/contraction map construction it
exists to score.

## Adding a module: checklist

- [ ] Does it need engine changes? If so, is the change general enough to be
      true of all rational maps? If not, keep it in the module.
- [ ] Does it introduce new mathematical objects (target regions, reference
      orbits, ...)? Those types belong to the module.
- [ ] Can it be tested by property rather than by golden image?
- [ ] Does anything in `cdx/` or `app/` now `#include` the module? That is a
      layering violation.
