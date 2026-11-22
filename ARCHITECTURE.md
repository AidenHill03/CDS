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
- `composed` — `ComposedMap`: a composition of `RationalMap` stages (each with
  a repeat count) kept unexpanded, evaluated stage by stage in both charts of
  the sphere; `Map::composed` makes it renderable. For maps whose degree rules
  out a P/Q form (iterates, conjugations, M∘P^n∘R constructions)
- `expr` — parser/evaluator for user-typed formulas
- `roots` — Aberth–Ehrlich polynomial root finder
- `renderer` — Julia / parameter / basin / Green's rendering
- `analysis` — `find_attractors`, `wada_diagnostic`, `hausdorff_distance`
- `critical_orbits` — `track_critical_orbit`: one free critical orbit, followed
  with inline period detection (the parameter planes' engine)

Test by mathematical property (symmetries, Riemann–Hurwitz counts, metric
identities), never by golden image.

#### Parameter-plane conventions

Every parameter-plane mode agrees on five things. The definitions live in
`cdx/rational.hpp` (`FreeCriticalStructure`, `CriticalLabelBase`) and
`cdx/analysis.hpp` (`polynomial_escape_radius`); this is the summary.

- **Free critical points.** A mode iterates only the *free* critical points.
  A critical point is structurally pre-fixed, and dropped, when at three
  generic parameters it sits at the same place and is fixed, or maps onto
  the same fixed point. Examples: ∞ for every polynomial, and a pole of order
  ≥ 2 when ∞ is fixed, such as z = 0 for relaxed Newton. The split is computed
  once per render.
- **`critical_index`: one labelling everywhere.** The index means the same
  free critical point in `parameter` (PerCritical, polynomial or rational),
  in `parameter_period` (FromMap), and in
  `RationalMap.critical_point_labels(a)` (for the Facts panel). The free
  critical points are sorted by (Re, Im) at a fixed base parameter
  a* = 0.618 + 0.382i (nudged off any collision), then continued along the
  straight segment a* → a. Colliding critical points swap when you go round
  the collision, so some cut is unavoidable. This convention puts the cuts on
  the rays from each collision point pointing directly away from a*. They do
  not move when you pan or zoom.
- **Parameter planes follow the free critical orbits; full attractor
  reconciliation is single-parameter only.** Every attracting cycle attracts a
  critical point (Fatou), and a pre-fixed critical point's attractor is known
  in closed form, so following the free orbits finds every attracting cycle
  that matters. `parameter` (rational), `parameter_basin` and
  `parameter_period` run `track_critical_orbit` per pixel and per free
  critical point (`cdx/critical_orbits.hpp`): escape to ∞ (certified radius
  for a polynomial; ∞'s multiplier read off the compiled coefficients for a
  rational map), otherwise Brent periodicity checking, reduction to the
  minimal period, Newton refinement of f^q(z) − z, and the multiplier test
  |λ| < 1 − 10⁻⁶. Neutral, parabolic, Siegel, or out of budget is reported
  undetermined, never guessed. No fixed-point solve, attractor discovery,
  algebraic union or reconciliation sweep runs per pixel.
  `complete_attractors` and the reconciliation code serve single-parameter
  uses: Facts, dynamical `basin`, overlays. The old per-pixel path survives
  only behind `Renderer::set_reconciled_parameter_path`, as a test oracle and
  for A/B timing, not as a user setting.
- **Escape status is a mask.** `julia` and `parameter` return an explicit
  escaped/resolved mask next to the smooth value. The value is only a shade,
  and is legitimately ≤ 0 for fast escapes.
- **Strategies.** Several free orbits are combined the same way for a
  polynomial and a rational map. Slowest is unescaped iff some orbit does not
  settle; otherwise it takes the largest smooth value. Fastest is unescaped
  iff none settles (for a polynomial, that is the connectedness locus);
  otherwise it takes the smallest. PerCritical uses only the `critical_index` orbit. A unicritical
  polynomial (z^d + a) has one orbit and ignores the strategy, and its
  single-orbit path is pinned bit-for-bit in the tests.
- **"Certified" escape.** `polynomial_escape_certified` means "no poles, so ∞
  is superattracting". The *radius* comes from the coefficients: every
  polynomial escape-time path escapes at max(ρ, `escape_radius`). Here ρ is
  the positive root of |a_d| r^d − Σ_{k<d} |a_k| r^k − r, recomputed per pixel
  in the parameter plane. Past ρ every orbit provably escapes. A user radius
  on its own is not a proof: 0.1 z² + a has bounded orbits through |z| = 10.
  For z^d + a at the default radius 2, the result is unchanged.

`parameter_period` seeds from the map's own free critical points
(`CriticalPointFamily::FromMap`). The closed-form relaxed-Newton seeds are
used only when `relaxed_newton_power_n` recognises the map structurally.
`find_attractors` reduces every detected cycle to its minimal period.

### `app/` — the sandbox

The application a user actually operates. Owns mutable session state; the
engine stays stateless apart from `Renderer`'s configuration.

- current map, parameter, viewport, render mode
- the family library (save / name / load)
- term editing operations and undo
- data extraction and export
- animation export

### The window

`SandboxWindow` (app/sandbox.py) is a `QMainWindow`:

- **Top bar** (a fixed toolbar holding `app.widgets.TopBar`): wordmark, map
  chip (family name, or `custom` once the formula is edited away from a
  library entry; opens the Library), the workspace switch, Coupled | Single,
  render status, Reset view, Export.
- **Central widget**: a `QStackedWidget` of workspaces. *Explore* is the two
  planes side by side in a splitter, each an `app.plane_frame.PlaneFrame`
  (caps title, segmented mode control, the `ImageView`, overlay chips).
  Modules can add more workspaces.
- **Sidebar** (right `TabbedDock`): Info | Library | Facts.
- **Equations & Settings** (bottom `TabbedDock`): Equation | Rendering |
  Coloring | Orbit. Rendering and Coloring are live `SettingsPanel`s (one
  shared debounce, one coalesced re-render); there is no Settings dialog.

Both docks collapse to their tab strip and can be closed and reopened from
View; their state, the window geometry and the splitter persist in
`~/.complexdynamics/layout.ini` (`QSettings` + `saveState()`), and View ▸
Reset Layout restores the defaults. Below 1250 px wide the sidebar collapses
on its own. The overlay chips and the View-menu overlay items are both built
from `overlay_toggles()` (the `OVERLAY_LAYERS` registry plus the seeded
orbit, the parameter marker and its crosshair), so they cannot disagree.

### Theme

`app/theme.py` holds every color (a `Tokens` set per theme; only `dark`
exists today, structured so a light set can be added), the three bundled
OFL fonts (`app/resources/fonts/`, registered at startup, silently falling
back to system fonts) and the sizes. It generates the Qt style sheet and the
palette; the app runs on the Fusion style on every platform so macOS and
Windows look the same. Widgets opt into a look with dynamic properties
(`role`, `plane`, `invalid`, `mono`), never with their own style sheets.
`app/test_theme.py` fails if any other UI source contains a hex color,
`rgb()`, a `QColor(r, g, b)` literal or a named Qt color.

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

## Attaching a module's UI

`app/` never imports a module. The **composition root**, `phoenix.py` at the
repository root, is the one file that knows every layer: it builds each
module's panel and hands it to `SandboxWindow.add_panel(widget, title,
placement)`, a plain method (it also calls the widget's `shutdown()`, if it
has one, when the window closes). `placement` is one of:

- `"workspace"`: a page of the central stack next to Explore; the top bar's
  Explore | … switch appears. While it shows, the plane-only controls
  (Coupled | Single, overlay chips, the Rendering/Coloring/Orbit tabs) hide.
- `"sidebar"` (the default): a sidebar tab after Info, Library, Facts.
- `"dock"`: a tab of the Equations & Settings dock.

When `placement` is omitted, the widget's own `placement` attribute is used
(the Approximation panel sets `placement = "workspace"`), then `"sidebar"`.
`python -m app.sandbox` still starts the plain sandbox. Packaging
(`complexdynamics.spec`) starts from `phoenix.py`.

## Controls registry

`app/controls.py` is the single description of every input the app responds
to: mouse/trackpad gestures, keys handled by a focused plot pane, and
application shortcuts. `CONTROLS` is a tuple of `ControlEntry` (id, section,
human action text, context such as "Dynamical plane", a structured binding,
and an optional status-bar hint). Bindings are `KeyBinding` (a QAction
shortcut: a `QKeySequence.StandardKey` where Qt has one, extra portable
sequences, a fallback), `MouseBinding` (gesture + button + modifiers) or
`PlotKey` (Qt keys + modifiers, handled in `ImageView.keyPressEvent` so they
never fire while typing). An entry with no binding is a prose note.

It is the same single-source pattern as `OVERLAY_LAYERS` for the View menu
(a data table that code reads, not plugin machinery):

- `SandboxWindow`'s QActions take their shortcuts from `key_sequences(id)`;
  `ImageView` takes its key sets from `plot_keys(id)` / `plot_modifiers(id)`.
- The Info tab (`app/info_panel.py`, in the sidebar) is generated by
  `info_html()` (followed by a "This map" section), and the
  status-bar pane hint by `hint_parts()`, so documentation cannot drift.
- The mouse handlers are a state machine and cannot simply look bindings
  up, so `app/test_sandbox.py` drives the real handler for every bound entry
  and checks the documented effect; that test fails if an entry is added
  without a check.
- Labels are platform-aware (`binding_label`): Qt's NativeText for key
  sequences ("Ctrl" in a QKeySequence is Command on macOS), "⇧ Drag" /
  "⌥ ←" on macOS and "Shift+Drag" / "Alt+←" elsewhere. The platform is
  injectable so `app/test_controls.py` checks both on any host.

A module documents its own tab's controls by giving its panel widget a
`controls` attribute (a sequence of `ControlEntry`); `add_panel` adds them to
the Info tab. `modules/approximation/panel.py` does this. To add a control:
add an entry, read its binding from the registry where the handler can, and
add its agreement check.

## Orientation at the seam

Engine and app rasters have row 0 at the **bottom** (`cdx.Viewport.coord`).
A module that works in image order (row 0 at the top) converts in exactly one
named function at its seam — for the approximation module,
`modules.approximation.construction.to_cdx_orientation` — and nowhere else.

## The approximation module

`modules/approximation/` implements the Fisher–Hill–Lazebnik–Thompson
construction from an image: Runge approximant `S` by Cauchy quadrature on the
region edges, `r = M ∘ F^n ∘ S` with `F` the model polynomial in reciprocal
coordinates, certified trapping disks, certified basin cells, a complete
critical-point count (argument principle + Ehrlich–Aberth), and scoring
against the target with `cdx.hausdorff_distance` / `cdx.wada_diagnostic`.
`r` is evaluated by the engine's `ComposedMap`; the module's own C++ kernel
(`_approx_kernel`, built by `modules/approximation/CMakeLists.txt`) holds only
what is specific to the construction. Its panel is the "Approximation"
workspace.

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

In the module: (2.4) on the target's interior cores is `construction.
check_image_cores` (sampled) and `dynamics.certify_cells` (every cell,
proved); (2.5) is replaced by its consequence that actually matters, Remark
2.12 — every critical value of `r` lies in a basin — checked directly by
`dynamics.critical_analysis` for the map that was built.

## Adding a module: checklist

- [ ] Does it need engine changes? If so, is the change general enough to be
      true of all rational maps? If not, keep it in the module.
- [ ] Does it introduce new mathematical objects (target regions, reference
      orbits, ...)? Those types belong to the module.
- [ ] Can it be tested by property rather than by golden image?
- [ ] Does anything in `cdx/` or `app/` now `#include` the module? That is a
      layering violation.
