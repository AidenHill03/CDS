"""modules.approximation -- the Fisher-Hill-Lazebnik-Thompson simultaneous-
approximation construction (arXiv:2605.09903, Section 2) for Project Phoenix.

From an image of d colored regions it builds a rational map

    r = M o F^(n) o S

(S a Runge approximant from Cauchy quadrature on the region edges, F the
model polynomial P(z) = z^d + d/(d-1) z in reciprocal coordinates, M an
affine map), whose d attracting basins approximate the regions, and reports
what is proved about it: certified trapping disks, certified basin cells,
and a complete count of the critical points of S with the orbits of their
values -- which together verify the paper's Remark 2.12 (the Fatou set is
exactly the d basins).

    construction  image -> map + dynamical data       (image_to_basins)
    dynamics      certificates and critical points
    scoring       Hausdorff distance and Wada statistics against the target,
                  via cdx.hausdorff_distance / cdx.wada_diagnostic
    panel         the "Approximation" tab of the app

The map is evaluated by the engine's general cdx.ComposedMap; the module's
own C++ kernel (_approx_kernel, built by this directory's CMakeLists.txt)
holds only what is specific to this construction.
"""

from .construction import (DEFAULTS, BasinMap, Cancelled, image_to_basins, report, save_summary,
                           summary, to_cdx_orientation)
from .scoring import score

__all__ = ["image_to_basins", "BasinMap", "Cancelled", "DEFAULTS", "report", "summary", "save_summary",
           "score", "to_cdx_orientation"]
