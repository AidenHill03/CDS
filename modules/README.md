# modules/

Self-contained research investigations. Empty for now; see
[`ARCHITECTURE.md`](../ARCHITECTURE.md) for the full layering rationale. This
file states the rule that governs everything placed here.

## The rule

A module attaches to the sandbox at exactly one seam: an **analysis** is a
plain function from *(map, parameter, and optionally a rendered image)* to
structured data — the shape `cdx::find_attractors`, `cdx::wada_diagnostic`,
and `cdx::hausdorff_distance` already have.

- **Dependencies point inward only.** A module may depend on `cdx/` and
  `app/`. Nothing in `cdx/` or `app/` may depend on a module.
- **No engine changes for module-specific needs.** If a module needs a change
  to `cdx/`, that change must be justifiable as a *general* rational-dynamics
  capability, independent of the module's own research question — otherwise
  the change belongs inside the module.
- **New mathematical objects belong to the module**, not the engine. Target
  regions, reference orbits, and anything else specific to one research
  question are defined and owned where they are used.
- **No plugin registry, loader, or abstract base class**, here or anywhere
  else in the project, until at least three modules demonstrably need one.
  The seam is a plain function signature; that is deliberate, not a
  placeholder for infrastructure to be added later. Premature extension
  machinery is harder to remove than to add.
- **Test by property**, not golden image, same as the rest of the project.

## The first module: `modules/approximation/`

Not yet implemented. It scores candidates for the Fisher–Hill–Lazebnik–Thompson
simultaneous-approximation construction — conditions (2.4) and (2.5) from the
paper. See the "Deferred: the approximation module" section of
`ARCHITECTURE.md` for the construction, the exact statement of both
conditions, and two specific mistakes already made once and worth not
repeating.
