"""modules.approximation.scoring -- how well the basins of r match the target.

This is the use the engine's two image analyses were written for:

* ``cdx.hausdorff_distance`` between the Julia set of r (the boundary pixels
  of a ``cdx.Renderer.render_basin`` image, via ``cdx.extract_boundary_points``)
  and the boundary of the target regions, in the chordal metric the paper
  measures approximation in (Notation 1.1; Theorem 2.15 is a statement about
  exactly this distance), and in the Euclidean metric. The two directed
  distances are kept: a large julia_to_target means structure of J(r) far
  from every target edge, a large target_to_julia means a target edge that
  J(r) does not follow.

* Per basin, the Hausdorff distance between basin j and target region j as
  pixel sets (exact on the grid, by Euclidean distance transforms). This is
  the quantity the paper's simultaneous approximation is about: every basin
  close to its own region at once.

* ``cdx.wada_diagnostic`` on both label images. For the construction's maps
  the Julia set is the boundary of EVERY basin (it is for any rational map
  with attracting basins), so the basin image should show the Wada signature
  -- wada_fraction rising toward 1 with resolution -- whatever the target
  looks like. The target's own wada_fraction says whether the paper's
  common-boundary hypothesis holds for it; when it does not, the Hausdorff
  distance cannot go to 0 (the beads of the other basins along each edge are
  forced), and the basins approximate the regions only in the weaker
  one-sided sense the certificates measure. Read wada_fraction as a trend
  over the resolutions computed, not as one number (see cdx.WadaStats).

Everything here works on the engine's grid: cdx.Viewport, row 0 at the
BOTTOM. The target raster arrives in image order and is flipped exactly once,
by construction.to_cdx_orientation.
"""

from __future__ import annotations

import math

import numpy as np

from .construction import fill_boundary_pixels, to_cdx_orientation


def default_viewport(info, resolution):
    """Square cdx.Viewport covering the image (and nothing of the anchor cluster)."""
    import cdx
    b = info["image_bounds"]
    center = complex((b[0] + b[1]) / 2, (b[2] + b[3]) / 2)
    scale = max(b[1] - b[0], b[3] - b[2]) / 2
    return cdx.Viewport(center, scale, int(resolution))


def target_on_viewport(info, viewport):
    """The (boundary-filled) target labels sampled at the viewport's pixels, cdx orientation.

    Outside the image the exterior label is used: it is the region that
    contains infinity in the construction.
    """
    labels = fill_boundary_pixels(info["target_labels"])          # image order, row 0 = top
    H, W = labels.shape
    x0, x1, y0, y1 = info["image_bounds"]                          # y0 bottom, y1 top
    res = viewport.resolution
    s = viewport.pixel_size
    xs = viewport.center.real - viewport.scale + s * np.arange(res)
    ys = viewport.center.imag - viewport.scale + s * np.arange(res)   # row 0 = bottom
    col = np.floor((xs - x0) / (x1 - x0) * W).astype(int)
    row_top = np.floor((y1 - ys) / (y1 - y0) * H).astype(int)
    inside = ((row_top >= 0) & (row_top < H))[:, None] & ((col >= 0) & (col < W))[None, :]
    out = np.full((res, res), float(info["outside_label"]))
    rr = np.clip(row_top, 0, H - 1)[:, None]
    cc = np.clip(col, 0, W - 1)[None, :]
    sampled = labels[rr, cc].astype(float)
    out[inside] = sampled[inside]
    return out                                                      # row 0 = bottom already


def classification_tolerance(info):
    """Chordal tolerance for cdx.Renderer.render_basin that only fires inside a certified trap.

    A point within chordal distance tol of the fixed point zeta_j is within
    Euclidean distance tol * (1 + |zeta_j|^2 + ...) / 2 of it; this keeps that
    below a fifth of the trap radius, so every label render_basin assigns is a
    certified one.
    """
    zeta = np.abs(np.asarray(info["fixed_points"]))
    rho = np.asarray(info["trap_radii"])
    return float(np.min(0.2 * rho * 2 / (1 + (zeta + rho) ** 2)))


def render_basins(r, info, viewport, max_iter=60, threads=0, cancel=None):
    """Basin labels of r on the viewport via the engine (cdx orientation): (labels, iterations)."""
    import cdx
    cycles = [cdx.Cycle([complex(z)], j + 1) for j, z in enumerate(info["fixed_points"])]
    settings = cdx.RenderSettings(max_iter=int(max_iter), tol=classification_tolerance(info),
                                  threads=int(threads))
    renderer = cdx.Renderer(map=r.cdx_map(), viewport=viewport, settings=settings)
    labels, iterations = renderer.render_basin(cycles, cancel)
    return labels, iterations


def per_basin_hausdorff(target, basins, n_labels, pixel_size):
    """Hausdorff distance between basin j and region j as pixel sets, for each label j (Euclidean).

    Returns a list of dicts with the two directed distances in pixels and in
    plane units. Unresolved pixels (label 0) belong to neither set.
    """
    from scipy.ndimage import distance_transform_edt
    out = []
    for j in range(1, n_labels + 1):
        A = target == j
        B = basins == j
        if not A.any() or not B.any():
            out.append(dict(label=j, basin_to_region_px=math.inf, region_to_basin_px=math.inf,
                            hausdorff_px=math.inf, hausdorff=math.inf))
            continue
        to_A = distance_transform_edt(~A)       # distance of every pixel to the nearest pixel of A
        to_B = distance_transform_edt(~B)
        b2a = float(to_A[B].max())
        a2b = float(to_B[A].max())
        h = max(b2a, a2b)
        out.append(dict(label=j, basin_to_region_px=b2a, region_to_basin_px=a2b,
                        hausdorff_px=h, hausdorff=h * pixel_size))
    return out


def _wada_dict(stats):
    return dict(n_basins=int(stats.n_basins), unresolved_fraction=float(stats.unresolved_fraction),
                boundary_fraction=float(stats.boundary_fraction), wada_fraction=float(stats.wada_fraction),
                radius_px=int(stats.radius_px))


def score(r, info, resolutions=(256, 512), radius_fraction=0.004, max_iter=60, max_points=4000,
          viewport=None, threads=0, cancel=None, progress=None):
    """Wada statistics at each resolution and the Hausdorff distance at the finest one.

    Returns a dict:
      viewport                  (center, scale) used
      resolutions               as given
      wada_target, wada_basins  one cdx.WadaStats-like dict per resolution
      hausdorff                 cdx.HausdorffResult as a dict (finest resolution): J(r) vs target edges
      per_basin                 Hausdorff distance basin j vs region j, per label (finest resolution)
      boundary_points           (number on J(r), number on the target boundary)
      target_labels, basin_labels   label images at the finest resolution, cdx orientation
    """
    import cdx
    resolutions = [int(x) for x in resolutions]
    wada_t, wada_b = [], []
    target = basins = vp = None
    for res in resolutions:
        if progress is not None:
            progress(f"Scoring at {res}x{res}")
        vp = default_viewport(info, res) if viewport is None else cdx.Viewport(viewport.center, viewport.scale, res)
        target = target_on_viewport(info, vp)
        basins, _ = render_basins(r, info, vp, max_iter=max_iter, threads=threads, cancel=cancel)
        if cancel is not None and cancel.is_cancelled:
            return None
        wada_t.append(_wada_dict(cdx.wada_diagnostic(target, radius_fraction)))
        wada_b.append(_wada_dict(cdx.wada_diagnostic(basins, radius_fraction)))
    julia = cdx.extract_boundary_points(basins, vp)
    edges = cdx.extract_boundary_points(target, vp)
    h = cdx.hausdorff_distance(julia, edges, max_points)
    n_labels = int(np.asarray(info["palette"]).shape[0])
    per_basin = per_basin_hausdorff(target, basins, n_labels, float(vp.pixel_size))
    hausdorff = {k: float(getattr(h, k)) for k in (
        "chordal", "euclidean", "chordal_julia_to_target", "chordal_target_to_julia",
        "euclidean_julia_to_target", "euclidean_target_to_julia")}
    return dict(viewport=(complex(vp.center), float(vp.scale)), resolutions=resolutions,
                radius_fraction=float(radius_fraction),
                wada_target=wada_t, wada_basins=wada_b, hausdorff=hausdorff, per_basin=per_basin,
                pixel_size=float(vp.pixel_size), boundary_points=(len(julia), len(edges)),
                target_labels=target, basin_labels=basins)


def score_report(sc):
    """Plain-text summary of score()."""
    if sc is None:
        return "Scoring cancelled."
    h = sc["hausdorff"]
    px = sc["pixel_size"]
    L = ["Scoring  (engine: cdx.render_basin, cdx.wada_diagnostic, cdx.hausdorff_distance)"]
    L.append(f"Hausdorff  J(r) vs target boundary at {sc['resolutions'][-1]}x{sc['resolutions'][-1]} "
             f"(pixel {px:.2e}; {sc['boundary_points'][0]} and {sc['boundary_points'][1]} boundary points)")
    L.append(f"           chordal {h['chordal']:.4g}   (J->target {h['chordal_julia_to_target']:.4g}, "
             f"target->J {h['chordal_target_to_julia']:.4g})")
    L.append(f"           euclidean {h['euclidean']:.4g} = {h['euclidean'] / px:.1f} pixels   "
             f"(J->target {h['euclidean_julia_to_target'] / px:.1f} px, target->J "
             f"{h['euclidean_target_to_julia'] / px:.1f} px)")
    L.append("Per basin  Hausdorff distance between basin j and region j (pixel sets, Euclidean):")
    for b in sc["per_basin"]:
        L.append(f"           label {b['label']}: {b['hausdorff_px']:.1f} px   (basin->region "
                 f"{b['basin_to_region_px']:.1f} px, region->basin {b['region_to_basin_px']:.1f} px)")
    L.append("Wada       fraction of boundary pixels whose neighbourhood holds every label (trend matters)")
    L.append("           resolution   target    basins of r")
    for res, t, b in zip(sc["resolutions"], sc["wada_target"], sc["wada_basins"]):
        tf = "n/a" if math.isnan(t["wada_fraction"]) else f"{t['wada_fraction']:.3f}"
        bf = "n/a" if math.isnan(b["wada_fraction"]) else f"{b['wada_fraction']:.3f}"
        L.append(f"           {res:<10d}   {tf:<8s}  {bf}")
    t_last = sc["wada_target"][-1]["wada_fraction"]
    if not math.isnan(t_last) and t_last < 0.5:
        L.append("           The target's regions mostly meet two at a time (no common boundary), so the")
        L.append("           paper's Hausdorff theorem does not apply. J(r) can still follow the edges")
        L.append("           closely, but every basin reaches every edge (beads), so a basin is not")
        L.append("           Hausdorff-close to its own region where that region is far from an edge it")
        L.append("           does not touch -- see the per-basin distances above.")
    return "\n".join(L)
