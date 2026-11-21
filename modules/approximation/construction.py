"""modules.approximation.construction -- the Fisher-Hill-Lazebnik-Thompson
construction from an image (a port of matlab-reference/image_to_basins.m).

Construction from Section 2 of Fisher, Hill, Lazebnik and Thompson,
"Simultaneous Approximation by Attracting Basins" (2026),
https://arxiv.org/abs/2605.09903

    S(z) = constant + sum(residues[k] / (z - poles[k]))      Runge approximant
    F(u) = u^d / (1 + d/(d-1) * u^(d-1))                       the paper's P in
                                                               reciprocal coordinates
    r(z) = anchor_origin + anchor_scale * F^(n)(S(z))

Labels are 1-based exactly as in the MATLAB code (0 = unresolved), so that
``info`` arrays line up with MATLAB output.  Option names may be given in
snake_case (``num_basins``) or in the MATLAB spelling (``NumBasins``).

ORIENTATION. Every raster in ``info`` (target_labels, basin_labels,
certificate cell_labels) is in IMAGE order: row 0 is the TOP (largest Im z).
The engine and the app use row 0 at the BOTTOM (cdx.Viewport.coord). Arrays
cross that seam in exactly one place, ``to_cdx_orientation`` below, used by
scoring.py and panel.py; nothing else flips.

The map r is evaluated by the engine's general cdx::ComposedMap (inside the
module kernel); ``BasinMap.cdx_composed()`` / ``cdx_map()`` give the same map
as cdx Python objects, for rendering through cdx.Renderer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _approx_kernel as _kernel
from . import dynamics
from .dynamics import Cancelled


def to_cdx_orientation(raster):
    """Image order (row 0 = top) <-> cdx order (row 0 = bottom). Its own inverse."""
    return np.ascontiguousarray(np.asarray(raster)[::-1])

EPS = np.finfo(float).eps

# --------------------------------------------------------------------------
# Options

_DEFAULTS: dict[str, Any] = dict(
    num_basins=None,
    palette=None,
    outside_label=None,
    boundary_colors=None,
    color_tolerance=1 / 255,
    alpha_background=(1.0, 1.0, 1.0),
    image_size=128,
    boundary_pixels=2,
    panel_pixels=None,
    quadrature_order=4,
    rational_degree=None,
    anchor_degree=16,
    anchor_scale=0.10,
    runge_shift=0.0,
    inner_iterations=None,
    grid_size=360,
    plot_bounds=None,
    max_iterations=25,
    critical_check_max_degree=180,
    certify=True,
    certificate_subdivision=2,
    certificate_refinements=2,
    certificate_margin=4,
    critical_method="auto",
    unresolved_color=(0.30, 0.30, 0.30),
    output_image="",
    verbose=False,
)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _is_int(v, minimum):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return np.isscalar(v) and not isinstance(v, (bool, complex)) and math.isfinite(f) and f == int(f) and f >= minimum


def _rgb_rows(v):
    a = np.asarray(v, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 3 or not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 255):
        return None
    return a


def parse_options(**kwargs) -> dict[str, Any]:
    opts = dict(_DEFAULTS)
    for key, value in kwargs.items():
        k = _snake(key) if any(c.isupper() for c in key) else key
        if k not in opts:
            raise TypeError(f"Unknown option {key!r}")
        opts[k] = value

    def need(cond, name):
        if not cond:
            raise ValueError(f"Invalid value for {name}: {opts[name]!r}")

    o = opts
    need(o["num_basins"] is None or _is_int(o["num_basins"], 2), "num_basins")
    if o["palette"] is not None:
        p = _rgb_rows(o["palette"])
        need(p is not None and p.shape[0] >= 2, "palette")
        o["palette"] = p
    need(o["outside_label"] is None or _is_int(o["outside_label"], 1), "outside_label")
    if o["boundary_colors"] is not None:
        b = _rgb_rows(o["boundary_colors"])
        need(b is not None, "boundary_colors")
        o["boundary_colors"] = b
    need(np.isscalar(o["color_tolerance"]) and math.isfinite(o["color_tolerance"]) and o["color_tolerance"] >= 0,
         "color_tolerance")
    for name in ("alpha_background", "unresolved_color"):
        a = _rgb_rows(o[name])
        need(a is not None and a.shape[0] == 1, name)
        o[name] = a[0]
    need(_is_int(o["image_size"], 2), "image_size")
    need(_is_int(o["boundary_pixels"], 0), "boundary_pixels")
    need(o["panel_pixels"] is None or (np.isscalar(o["panel_pixels"]) and math.isfinite(o["panel_pixels"])
                                       and o["panel_pixels"] > 0), "panel_pixels")
    need(_is_int(o["quadrature_order"], 1), "quadrature_order")
    need(o["rational_degree"] is None or _is_int(o["rational_degree"], 1), "rational_degree")
    need(_is_int(o["anchor_degree"], 2), "anchor_degree")
    need(np.isscalar(o["anchor_scale"]) and math.isfinite(o["anchor_scale"]) and o["anchor_scale"] > 0,
         "anchor_scale")
    need(np.isscalar(o["runge_shift"]) and np.isfinite(o["runge_shift"]), "runge_shift")
    need(o["inner_iterations"] is None or _is_int(o["inner_iterations"], 1), "inner_iterations")
    g = np.atleast_1d(np.asarray(o["grid_size"], dtype=float))
    need(g.ndim == 1 and 1 <= g.size <= 2 and np.all(np.isfinite(g)) and np.all(g >= 2) and np.all(g == np.trunc(g)),
         "grid_size")
    o["grid_size"] = g.astype(int)
    if o["plot_bounds"] is not None:
        pb = np.asarray(o["plot_bounds"], dtype=float).ravel()
        need(pb.size == 4 and np.all(np.isfinite(pb)) and pb[0] < pb[1] and pb[2] < pb[3], "plot_bounds")
        o["plot_bounds"] = pb
    need(_is_int(o["max_iterations"], 0), "max_iterations")
    need(_is_int(o["critical_check_max_degree"], 0), "critical_check_max_degree")
    need(_is_int(o["certificate_subdivision"], 1), "certificate_subdivision")
    need(_is_int(o["certificate_refinements"], 0) and o["certificate_refinements"] <= 6, "certificate_refinements")
    need(_is_int(o["certificate_margin"], 2), "certificate_margin")
    need(o["critical_method"] in ("auto", "aberth", "pencil", "none"), "critical_method")
    o["certify"] = bool(o["certify"])
    for name in ("image_size", "boundary_pixels", "quadrature_order", "anchor_degree",
                 "max_iterations", "critical_check_max_degree", "certificate_subdivision",
                 "certificate_refinements", "certificate_margin"):
        o[name] = int(o[name])
    for name in ("num_basins", "outside_label", "rational_degree", "inner_iterations"):
        if o[name] is not None:
            o[name] = int(o[name])
    o["runge_shift"] = complex(o["runge_shift"])
    o["verbose"] = bool(o["verbose"])
    o["output_image"] = str(o["output_image"] or "")
    return o


# --------------------------------------------------------------------------
# Image reading (readRegions)

def _as_rgb(a):
    a = np.asarray(a, dtype=float)
    return a / 255 if np.any(a > 1) else a


def _unit(a: np.ndarray, bits: int | None = None) -> np.ndarray:
    if bits is not None:
        return a.astype(float) / (2 ** bits - 1)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(float) / float(np.iinfo(a.dtype).max)
    return a.astype(float)


def _load_image(image):
    """Return (rgb HxWx3 in [0,1], alpha HxW in [0,1] or None)."""
    if isinstance(image, np.ndarray):
        a = image
        if a.ndim == 2:
            a = a[:, :, None]
        if a.ndim != 3 or a.shape[2] not in (1, 2, 3, 4):
            raise ValueError("Array images must be HxW, HxWx1, HxWx2, HxWx3 or HxWx4.")
        c = a.shape[2]
        alpha = _unit(a[:, :, c - 1]) if c in (2, 4) else None
        base = a[:, :, : (1 if c <= 2 else 3)]
        rgb = _unit(base)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        return rgb, alpha

    from PIL import Image
    with Image.open(image) as im:
        im.load()
        mode = im.mode
        info = dict(im.info)
        if mode == "P" or mode == "PA":
            idx = np.asarray(im.getchannel(0) if mode == "PA" else im, dtype=np.intp)
            pal = np.asarray(im.getpalette() or [], dtype=float).reshape(-1, 3) / 255
            rgb = pal[idx]
            alpha = None
            if mode == "PA":
                alpha = _unit(np.asarray(im.getchannel(1)))
            elif "transparency" in info:
                t = info["transparency"]
                table = np.ones(len(pal))
                if isinstance(t, (bytes, bytearray)):
                    vals = np.frombuffer(bytes(t), dtype=np.uint8)[: len(pal)] / 255
                    table[: len(vals)] = vals
                else:
                    table[int(t)] = 0.0
                alpha = table[idx]
            return rgb, alpha
        if mode == "1":
            g = np.asarray(im, dtype=float)
            return np.repeat(g[:, :, None], 3, axis=2), None
        if mode in ("L", "LA"):
            g = _unit(np.asarray(im.getchannel(0)))
            alpha = _unit(np.asarray(im.getchannel(1))) if mode == "LA" else None
            if mode == "L" and "transparency" in info and isinstance(info["transparency"], int):
                alpha = (np.asarray(im) != info["transparency"]).astype(float)
            return np.repeat(g[:, :, None], 3, axis=2), alpha
        if mode.startswith("I;16") or mode == "I":
            a = np.asarray(im).astype(np.int64)
            bits = 16 if (mode.startswith("I;16") or (mode == "I" and a.max(initial=0) <= 65535)) else 32
            g = _unit(a, bits)
            return np.repeat(g[:, :, None], 3, axis=2), None
        if mode == "F":
            g = np.asarray(im, dtype=float)
            return np.repeat(g[:, :, None], 3, axis=2), None
        if mode in ("RGB", "RGBA"):
            a = np.asarray(im)
            rgb = _unit(a[:, :, :3])
            alpha = _unit(a[:, :, 3]) if mode == "RGBA" else None
            if mode == "RGB" and isinstance(info.get("transparency"), tuple):
                alpha = np.any(a != np.asarray(info["transparency"]), axis=2).astype(float)
            return rgb, alpha
        if mode == "CMYK" or mode == "YCbCr" or mode == "LAB" or mode == "HSV":
            raise ValueError("image_to_basins:ImageFormat - Use an RGB, grayscale, or indexed image.")
        conv = im.convert("RGBA")
        a = np.asarray(conv)
        return _unit(a[:, :, :3]), _unit(a[:, :, 3])


def _nearest_color(pixels, palette):
    best = np.full(pixels.shape[0], np.inf)
    labels = np.zeros(pixels.shape[0], dtype=np.int64)
    for j in range(palette.shape[0]):
        dist = np.sum((pixels - palette[j]) ** 2, axis=1)
        closer = dist < best
        labels[closer] = j + 1
        best[closer] = dist[closer]
    return labels, best


def _cluster_colors(colors, weights, d):
    if colors.shape[0] == d:
        return colors.copy()
    palette = np.zeros((d, 3))
    seed = int(np.argmax(weights))
    gap = np.full(weights.shape, np.inf)
    for j in range(d):
        palette[j] = colors[seed]
        gap = np.minimum(gap, np.sum((colors - palette[j]) ** 2, axis=1))
        seed = int(np.argmax(weights * gap))
    previous = None
    for _ in range(30):
        groups, _ = _nearest_color(colors, palette)
        if previous is not None and np.array_equal(groups, previous):
            break
        previous = groups
        for j in range(d):
            sel = groups == j + 1
            if np.any(sel):
                palette[j] = weights[sel] @ colors[sel] / np.sum(weights[sel])
    return palette


def _matlab_round(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(int)


def _linspace(a, b, n):
    """linspace with the rounding of GNU Octave 7+ (symmetric about the midpoint).

    Pixel-edge coordinates feed edge lengths, and in RationalDegree mode exact
    ties between equal-length edges are broken by those lengths' last bits.
    """
    n1 = n - 1
    k = np.arange(n)
    delta = (b - a) / n1
    out = np.where(k < n / 2, a + k * delta, b - (n1 - k) * delta)
    out[0], out[-1] = a, b
    return out


def read_regions(image, opts):
    rgb_full, alpha_full = _load_image(image)
    height, width = rgb_full.shape[:2]
    extent = max(height, width)
    resized = np.maximum(1, _matlab_round(np.array([height, width]) * min(1.0, opts["image_size"] / extent)))
    row = np.floor((np.arange(1, resized[0] + 1) - 0.5) * height / resized[0]).astype(int)
    col = np.floor((np.arange(1, resized[1] + 1) - 0.5) * width / resized[1]).astype(int)
    rgb = rgb_full[np.ix_(row, col)].reshape(-1, 3)
    if alpha_full is not None:
        opacity = alpha_full[np.ix_(row, col)].reshape(-1, 1)
        rgb = opacity * rgb + (1 - opacity) * _as_rgb(opts["alpha_background"])

    known = np.ones(rgb.shape[0], dtype=bool)
    if opts["boundary_colors"] is not None:
        _, gap = _nearest_color(rgb, _as_rgb(opts["boundary_colors"]))
        known = gap > opts["color_tolerance"] ** 2
    if not np.any(known):
        raise ValueError("image_to_basins:NoColors - Every pixel was excluded by BoundaryColors.")

    palette = None if opts["palette"] is None else _as_rgb(opts["palette"])
    if palette is None:
        colors, inverse = np.unique(rgb[known], axis=0, return_inverse=True)
        counts = np.bincount(inverse.ravel(), minlength=colors.shape[0]).astype(float)
        d = opts["num_basins"]
        if d is None:
            d = colors.shape[0]
            if d > 8:
                d = 3
        if d < 2 or d > colors.shape[0]:
            raise ValueError("image_to_basins:ColorCount - Choose at least two available colors, "
                             "or supply a Palette with an exterior color.")
        palette = _cluster_colors(colors, counts, d)
    elif opts["num_basins"] is not None and opts["num_basins"] != palette.shape[0]:
        raise ValueError("image_to_basins:PaletteSize - NumBasins must match the number of Palette rows.")
    if np.unique(palette, axis=0).shape[0] != palette.shape[0]:
        raise ValueError("image_to_basins:DuplicateColor - Palette rows must be distinct.")

    labels = np.zeros(rgb.shape[0], dtype=np.int64)
    labels[known] = _nearest_color(rgb[known], palette)[0]
    labels = labels.reshape(int(resized[0]), int(resized[1]))
    outside = opts["outside_label"]
    if outside is None:
        border = np.ones(labels.shape, dtype=bool)
        border[1:-1, 1:-1] = False
        border_labels = labels[border & (labels > 0)]
        if border_labels.size == 0:
            border_labels = labels[labels > 0]
        outside = int(np.argmax(np.bincount(border_labels)))  # smallest mode, like MATLAB
    if outside > palette.shape[0]:
        raise ValueError("image_to_basins:OutsideLabel - OutsideLabel exceeds the number of colors.")
    used = np.zeros(palette.shape[0], dtype=bool)
    used[labels[labels > 0] - 1] = True
    used[outside - 1] = True
    if not np.all(used):
        raise ValueError(f"image_to_basins:UnusedColor - Palette label(s) {list(np.flatnonzero(~used) + 1)} "
                         "have no pixels. Check Palette, NumBasins, or ImageSize.")
    x_edges = _linspace(-width / extent, width / extent, int(resized[1]) + 1)
    y_edges = _linspace(height / extent, -height / extent, int(resized[0]) + 1)
    return labels, palette, outside, x_edges, y_edges


def fill_boundary_pixels(labels):
    """Extend known labels by four-neighbor distance (up, down, left, right priority)."""
    labels = labels.copy()
    rows, cols = labels.shape
    while np.any(labels == 0):
        z_row = np.zeros((1, cols), labels.dtype)
        z_col = np.zeros((rows, 1), labels.dtype)
        up = np.vstack([z_row, labels[:-1]])
        down = np.vstack([labels[1:], z_row])
        left = np.hstack([z_col, labels[:, :-1]])
        right = np.hstack([labels[:, 1:], z_col])
        for nb in (up, down, left, right):
            fill = (labels == 0) & (nb > 0)
            labels[fill] = nb[fill]
    return labels


# --------------------------------------------------------------------------
# Cauchy quadrature (rasterEdges, edgeQuadrature)

def _nonzero_runs(row):
    n = row.size
    changes = np.concatenate([[0], np.flatnonzero(row[1:] != row[:-1]) + 1, [n]])
    first = changes[:-1]
    last = changes[1:] - 1
    w = row[first]
    keep = w != 0
    return first[keep], last[keep], w[keep]


def raster_edges(values, x, y):
    """Oriented contour edges carrying (left value - right value)."""
    rows, cols = values.shape
    h = np.vstack([np.zeros((1, cols)), values]) - np.vstack([values, np.zeros((1, cols))])
    v = np.hstack([np.zeros((rows, 1)), values]) - np.hstack([values, np.zeros((rows, 1))])
    starts, ends, jumps = [], [], []
    for j in range(rows + 1):
        f, l, w = _nonzero_runs(h[j])
        starts.append(x[f] + 1j * y[j])
        ends.append(x[l + 1] + 1j * y[j])
        jumps.append(w)
    for j in range(cols + 1):
        f, l, w = _nonzero_runs(v[:, j])
        starts.append(x[j] + 1j * y[l + 1])
        ends.append(x[j] + 1j * y[f])
        jumps.append(w)
    return (np.concatenate(starts).astype(complex), np.concatenate(ends).astype(complex),
            np.concatenate(jumps).astype(complex))


_GAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def gauss_rule(n):
    """Gauss-Legendre nodes and weights on [0, 1]."""
    if n not in _GAUSS_CACHE:
        if n == 1:
            _GAUSS_CACHE[n] = (np.array([0.5]), np.array([1.0]))
        else:
            k = np.arange(1, n)
            beta = k / np.sqrt(4.0 * k ** 2 - 1)
            evals, vecs = np.linalg.eigh(np.diag(beta, 1) + np.diag(beta, -1))
            order = np.argsort((evals + 1) / 2)
            _GAUSS_CACHE[n] = ((evals[order] + 1) / 2, vecs[0, order] ** 2)
    return _GAUSS_CACHE[n]


def edge_quadrature(starts, ends, jumps, panel_length, order, budget, anchor_count):
    lengths = np.abs(ends - starts)
    if budget is None:
        ratio = lengths / panel_length
        panels = np.maximum(1, np.ceil(ratio - 32 * np.spacing(ratio))).astype(np.int64)
        counts = order * panels
    else:
        minimum = lengths.size
        if budget < minimum:
            raise ValueError(f"image_to_basins:DegreeBudget - RationalDegree must be at least "
                             f"{minimum + anchor_count} for this raster and AnchorDegree.")
        # Many edges have identical length, so their shares tie.  Shares are
        # rounded to 1e-9 before the stable sort so that ties are broken by
        # edge order, not by the last bits of linspace and sum.
        share = np.floor((budget - minimum) * lengths / np.sum(lengths) * 1e9 + 0.5) / 1e9
        counts = 1 + np.floor(share).astype(np.int64)
        priority = np.argsort(-np.floor((share - np.floor(share)) * 1e9 + 0.5), kind="stable")
        extra = priority[: int(budget - counts.sum())]
        counts[extra] += 1
        panels = np.maximum(1, counts // order)
    total = int(counts.sum())
    poles = np.zeros(total, dtype=complex)
    residues = np.zeros(total, dtype=complex)
    nxt = 0
    min_nodes = np.inf
    two_pi_i = 2 * np.pi * 1j
    for j in range(lengths.size):
        base, remainder = divmod(int(counts[j]), int(panels[j]))
        step = (ends[j] - starts[j]) / panels[j]
        for k in range(int(panels[j])):
            q = base + (k < remainder)
            nodes, weights = gauss_rule(q)
            min_nodes = min(min_nodes, q)
            poles[nxt:nxt + q] = starts[j] + step * (k + nodes)
            residues[nxt:nxt + q] = -jumps[j] * step * weights / two_pi_i
            nxt += q
    return poles, residues, min_nodes


# --------------------------------------------------------------------------
# Trapping disks

def post_bound(targets, radius, d):
    """Bound |F(u)-target| on |u-target| <= radius (vectorized over targets)."""
    targets = np.asarray(targets, dtype=complex)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), targets.shape)
    bound = np.full(targets.shape, np.inf)
    zero = targets == 0
    r0 = radius[zero]
    den0 = 1 - d / (d - 1) * r0 ** (d - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = r0 ** d / den0
    b0[den0 <= 0] = np.inf           # the pole of F at |u|^(d-1) = (d-1)/d is inside the disk
    bound[zero] = b0
    nz = ~zero
    ratio = radius[nz] / np.abs(targets[nz])
    den = d - 1 - d * np.expm1((d - 1) * np.log1p(ratio))
    term = np.ones_like(ratio)
    numerator = np.zeros_like(ratio)
    for k in range(1, d + 1):
        term = term * ratio * ((d - k + 1) / k)
        numerator = numerator + (k - 1) * term
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.abs(targets[nz]) * numerator / den
    b[den <= 0] = np.inf
    bound[nz] = b
    bound[~np.isfinite(radius)] = np.inf
    return bound


def trapping_disks(S, poles, residues, targets, centers, scale, spacing, requested):
    d = targets.size
    base_radius = min(0.25, spacing / 3) * np.ones(d)
    while np.any(post_bound(targets, base_radius, d) > base_radius / 2):
        base_radius = base_radius / 2
    dist = np.abs(centers[:, None] - poles[None, :])
    max_radius = np.minimum(0.3 * dist.min(axis=1), 0.22 * scale * spacing)
    abs_res = np.abs(residues)
    slope_bound = (1.0 / (dist * (dist - max_radius[:, None]))) @ abs_res
    slack = 32 * EPS * (1 + (1.0 / dist) @ abs_res)
    center_error = np.abs(S(centers) - targets) + slack
    if np.any(center_error >= base_radius):
        raise RuntimeError("image_to_basins:AnchorFit - Runge fit misses an anchor disk. Increase "
                           "RationalDegree, QuadratureOrder, or AnchorScale, or reduce RungeShift.")
    radii = np.minimum(max_radius, 0.85 * (base_radius - center_error) / slope_bound)
    n = 6 if requested is None else requested
    bound = center_error + radii * slope_bound
    for _ in range(n):
        bound = post_bound(targets, bound, d)
    if requested is None:
        while np.any(scale * bound > radii / 2):
            n += 1
            bound = post_bound(targets, bound, d)
    else:
        for _ in range(20):
            shrink = scale * bound > radii / 2
            if not np.any(shrink):
                break
            radii[shrink] = radii[shrink] / 2
            bound = center_error + radii * slope_bound
            for _ in range(n):
                bound = post_bound(targets, bound, d)
        if np.any(scale * bound > radii / 2):
            raise RuntimeError("image_to_basins:InnerIterations - This InnerIterations value did not "
                               "establish trapping disks. Increase it or use None.")
    details = dict(base_radius=base_radius, center_error=center_error, slope_bound=slope_bound,
                   image_bound=scale * bound, noise=slack)
    return radii, n, details


# --------------------------------------------------------------------------
# Diagnostics

def plot_grid(bounds, requested):
    spans = np.array([bounds[3] - bounds[2], bounds[1] - bounds[0]])
    requested = np.asarray(requested)
    if requested.size == 1:
        requested = np.maximum(2, _matlab_round(requested[0] * spans / spans.max()))
    rows, cols = int(requested[0]), int(requested[1])
    x = bounds[0] + (np.arange(1, cols + 1) - 0.5) * spans[1] / cols
    y = bounds[3] - (np.arange(1, rows + 1) - 0.5) * spans[0] / rows
    return x, y


def _box_count(mask, margin):
    """conv2(kernel, kernel.', mask, 'same') with a (2m+1)^2 box of ones."""
    k = 2 * margin + 1
    p = np.pad(mask.astype(np.int64), margin)
    c = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.int64)
    c[1:, 1:] = p.cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def check_image_cores(labels, targets, x, y, margin, S, d, n, scale, radii):
    if margin == 0:
        core = labels > 0
    elif 2 * margin + 1 <= min(labels.shape):
        core = np.zeros(labels.shape, dtype=bool)
        for j in range(1, d + 1):
            core |= _box_count(labels == j, margin) == (2 * margin + 1) ** 2
    else:
        core = np.zeros(labels.shape, dtype=bool)
    row, col = np.nonzero(core)
    count = row.size
    if count == 0:
        return math.nan, math.nan, 0
    z = (x[col] + x[col + 1]) / 2 + 1j * (y[row] + y[row + 1]) / 2
    u = S(z)
    lab = labels[row, col]
    ideal = targets[lab - 1]
    max_error = float(np.max(np.abs(u - ideal)))
    in_trap = scale * np.abs(_kernel.post_map(d, n, u) - ideal) <= radii[lab - 1]
    return max_error, float(np.count_nonzero(in_trap) / count), int(count)


def check_critical_values(poles, residues, r, centers, radii, opts):
    """MATLAB's check: all critical points of S from a (2m+1) generalized eigenproblem, O(m^3)."""
    m = poles.size
    check = dict(performed=False, method="pencil", status="skipped", runge_critical_points=None,
                 relative_derivative_residual=None, values_tested=None, basin_labels=None,
                 all_computed_captured=None, near_infinity=None)
    if m > opts["critical_check_max_degree"]:
        check["status"] = f"skipped (degree {m} exceeds check limit {opts['critical_check_max_degree']})."
        return check
    # Finite eigenvalues are the zeros of sum residues/(z-poles)^2: an arrowhead
    # pencil with a double-pole Jordan block per pole.
    import scipy.linalg
    D = np.diag(poles)
    A = np.block([
        [D, np.eye(m), np.zeros((m, 1))],
        [np.zeros((m, m)), D, np.ones((m, 1))],
        [residues[None, :] / np.max(np.abs(residues)), np.zeros((1, m + 1))],
    ]).astype(complex)
    B = np.diag(np.concatenate([np.ones(2 * m), [0.0]])).astype(complex)
    try:
        ev = scipy.linalg.eig(A, B, right=False)
    except Exception as err:  # pragma: no cover
        check["status"] = f"eigenvalue calculation failed ({err})."
        return check
    points = ev[np.isfinite(ev)]
    # Infinity is a multiple critical point of S by design (see dynamics.infinity_data).
    # QZ returns part of it as huge "finite" eigenvalues; they are counted with infinity.
    extent = np.max(np.abs(poles))
    far = np.abs(points) > 1e3 * extent
    near_infinity = int(2 * m - 2 - np.count_nonzero(~far))
    points = points[~far]
    terms = residues[None, :] / (points[:, None] - poles[None, :]) ** 2
    residual = np.abs(terms.sum(axis=1)) / np.abs(terms).sum(axis=1)
    values = np.concatenate([r(points), r(np.array([complex(np.inf, 0)])), centers])
    labels, _ = r.classify(values, centers, radii, opts["max_iterations"])
    check.update(performed=True, runge_critical_points=points, relative_derivative_residual=residual,
                 values_tested=values, basin_labels=labels, all_computed_captured=bool(np.all(labels > 0)),
                 near_infinity=near_infinity)
    check["status"] = (f"{np.count_nonzero(labels)}/{labels.size} tested values captured. "
                       f"{np.count_nonzero(residual <= 1e-6)}/{residual.size} finite roots have residual <= 1e-6. "
                       f"{near_infinity} of {2 * m - 2} critical points at or near infinity (value S(Inf) tested).")
    return check


def color_raster(labels, palette):
    return palette[labels]


# --------------------------------------------------------------------------
# Public map object

class BasinMap:
    """Callable r(z) = origin + scale*F^(n)(S(z)) backed by the C++ kernel.

    Accepts scalars or arrays of complex points (including complex infinity).
    """

    def __init__(self, poles, residues, shift, d, n, origin, scale):
        self.poles = np.ascontiguousarray(poles, dtype=complex)
        self.residues = np.ascontiguousarray(residues, dtype=complex)
        self.shift = complex(shift)
        self._k = _kernel.BasinMap(self.poles, self.residues, self.shift, int(d), int(n),
                                   complex(origin), float(scale))
        self.d, self.n, self.origin, self.scale = int(d), int(n), complex(origin), float(scale)
        self.degree_S = int(self.poles.size)

    def cdx_composed(self):
        """The same map as an engine object: cdx.ComposedMap([S, F^n, M])."""
        import cdx
        S = cdx.RationalMap.from_partial_fractions(self.shift, self.poles, self.residues)
        F = cdx.RationalMap.from_expression(_kernel.model_map_source(self.d), "", {})
        M = cdx.RationalMap()
        M.add_poly(self.origin, 0)
        M.add_poly(complex(self.scale), 1)
        return cdx.ComposedMap([(S, 0j, 1), (F, 0j, self.n), (M, 0j, 1)])

    def cdx_map(self):
        """cdx.Map.composed(...) of this map, ready for cdx.Renderer."""
        import cdx
        return cdx.Map.composed(self.cdx_composed())

    @staticmethod
    def _wrap(fn, z):
        arr = np.asarray(z, dtype=complex)
        out = fn(arr.reshape(-1) if arr.ndim == 0 else arr)
        return out.reshape(()).item() if arr.ndim == 0 else out

    def __call__(self, z):
        return self._wrap(self._k, z)

    def runge(self, z):
        return self._wrap(self._k.runge, z)

    def classify(self, z, centers, radii, max_iterations):
        arr = np.asarray(z, dtype=complex)
        return self._k.classify(arr, np.asarray(centers, complex), np.asarray(radii, float), int(max_iterations))

    @property
    def degree(self):
        """Exact integer degree, deg S * d^n."""
        return self.degree_S * self.d ** self.n

    def __repr__(self):
        return f"BasinMap(degree(S)={self.degree_S}, d={self.d}, n={self.n}, degree(r)={self.degree:.5g})"


class _RungeFn:
    def __init__(self, poles, residues, shift):
        self._k = _kernel.Runge(poles, residues, complex(shift))

    def derivs(self, z):
        """(S, S', S'', sum 1/(z-p)) at an array of points."""
        return self._k.derivs(np.asarray(z, complex))

    def __call__(self, z):
        return BasinMap._wrap(self._k, z)


# --------------------------------------------------------------------------
# Main entry point

def image_to_basins(image, progress=None, should_cancel=None, **options):
    """Construct a rational map whose attracting basins approximate the regions in an image.

    Parameters
    ----------
    image : str, Path, or ndarray
        Image file, or an HxW / HxWx{1,2,3,4} array (uint8, uint16 or float in [0,1]).
        The image is centered at 0 with its longer side of length 2; the top of the
        image is positive Im(z).
    progress : callable(str), optional
        Called with a short description as each stage starts.
    should_cancel : callable() -> bool, optional
        Polled between stages and inside the long loops; when it returns True
        the construction stops by raising ``Cancelled``.
    **options
        Same options as the MATLAB function, in snake_case or MATLAB spelling.
        See ``DEFAULTS``.

    Returns
    -------
    r : BasinMap
        Callable rational map.
    info : dict
        Same fields as the MATLAB INFO struct, in snake_case.
    """
    opts = parse_options(**options)

    def stage(text):
        if should_cancel is not None and should_cancel():
            raise Cancelled()
        if progress is not None:
            progress(text)

    stage("Reading the image")
    target_labels, palette, outside, x_edges, y_edges = read_regions(image, opts)
    labels = fill_boundary_pixels(target_labels)
    d = palette.shape[0]

    # F has fixed critical points 0 and the roots of u^(d-1) = -(d-1).
    other = np.array([j for j in range(1, d + 1) if j != outside])
    targets = np.zeros(d, dtype=complex)
    angles = (2 * np.arange(d - 1) + 1) * np.pi / (d - 1)
    targets[other - 1] = (d - 1) ** (1 / (d - 1)) * np.exp(1j * angles)
    distances = np.abs(targets[:, None] - targets[None, :])
    np.fill_diagonal(distances, np.inf)
    spacing = float(distances.min())
    pixel_size = min(x_edges[1] - x_edges[0], y_edges[0] - y_edges[1])
    scale = float(opts["anchor_scale"])
    origin = x_edges[-1] + scale * (1 + np.max(np.abs(targets))) + 2 * pixel_size
    centers = origin + scale * targets
    anchor_radius = 0.22 * scale * spacing

    values = targets[labels - 1]
    starts, ends, jumps = raster_edges(values, x_edges, y_edges)
    panel_pixels = opts["panel_pixels"]
    if panel_pixels is None:
        panel_pixels = 2 * max(1, opts["boundary_pixels"])
    anchor_count = other.size * opts["anchor_degree"]
    budget = opts["rational_degree"]
    if budget is not None:
        budget = budget - anchor_count
    stage("Cauchy quadrature on the region edges")
    poles, residues, min_nodes = edge_quadrature(starts, ends, jumps, panel_pixels * pixel_size,
                                      opts["quadrature_order"], budget, anchor_count)

    # On an anchor circle the trapezoid rule gives 1/(1-((z-c)/h)^m).
    m = opts["anchor_degree"]
    offsets = anchor_radius * np.exp(2j * np.pi * np.arange(m) / m)
    anchor_poles = offsets[:, None] + centers[other - 1][None, :]
    anchor_residues = -np.outer(offsets, targets[other - 1]) / m
    poles = np.concatenate([poles, anchor_poles.ravel(order="F")])
    residues = np.concatenate([residues, anchor_residues.ravel(order="F")])
    if np.unique(poles).size != poles.size:
        raise RuntimeError("image_to_basins:PoleCollision - Some poles coincide in floating point. "
                           "Increase AnchorScale or lower RationalDegree.")
    shift = opts["runge_shift"]
    if np.any(residues == 0):
        raise RuntimeError("image_to_basins:ZeroResidue - A residue vanished, so degree(S) would be "
                           "smaller than the number of poles.")
    S = _RungeFn(poles, residues, shift)
    stage(f"Trapping disks (deg S = {poles.size})")
    trap_radii, n, trap = trapping_disks(S, poles, residues, targets, centers, scale, spacing,
                                         opts["inner_iterations"])
    r = BasinMap(poles, residues, shift, d, n, origin, scale)

    fixed_points = centers.copy()
    for _ in range(40):
        nxt = r(fixed_points)
        converged = np.all(np.abs(nxt - fixed_points) <= 16 * EPS * (1 + np.abs(nxt)))
        fixed_points = nxt
        if converged:
            break

    image_bounds = np.array([x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]])
    bounds = image_bounds if opts["plot_bounds"] is None else opts["plot_bounds"]
    x, y = plot_grid(bounds, opts["grid_size"])
    stage(f"Classifying a {x.size}x{y.size} grid")
    Z = x[None, :] + 1j * y[:, None]
    basins, iterations = r.classify(Z, centers, trap_radii, opts["max_iterations"])
    core_error, core_capture, core_count = check_image_cores(
        target_labels, targets, x_edges, y_edges, opts["boundary_pixels"], S, d, n, scale, trap_radii)

    # ---- certificates and dynamical data --------------------------------
    caps = dynamics.capture_radii(post_bound, targets, trap["base_radius"], trap_radii, scale, n)
    fp = dynamics.fixed_point_data(S._k, fixed_points, targets, scale, d, n, trap["noise"])
    fp_res = np.abs(r(fixed_points) - fixed_points)
    basin_table = []
    for j in range(d):
        basin_table.append(dict(
            label=j + 1, color=palette[j].copy(), is_exterior=bool(j + 1 == outside),
            target=complex(targets[j]), trap_center=complex(centers[j]), trap_radius=float(trap_radii[j]),
            trap_image_radius_bound=float(trap["image_bound"][j]), capture_radius=float(caps[j]),
            fixed_point=complex(fixed_points[j]), fixed_point_residual=float(fp_res[j]), **fp[j]))

    certificate = None
    if opts["certify"]:
        stage("Certifying basin cells (Taylor enclosures of S)")
        k = opts["certificate_subdivision"]
        xs, ys, (row0, col0) = dynamics.certification_grid(x_edges, y_edges, poles, opts["certificate_margin"], k)
        cell_labels, cell_bound, xs, ys = dynamics.certify_cells(S._k, xs, ys, targets, caps,
                                                                opts["certificate_refinements"],
                                                                should_cancel=should_cancel)
        k *= 2 ** opts["certificate_refinements"]
        row0, col0 = row0 * 2 ** opts["certificate_refinements"], col0 * 2 ** opts["certificate_refinements"]
        ext_ok, ext_sup = dynamics.exterior_certificate(cell_labels, cell_bound, outside, caps[outside - 1], shift)
        H, W = target_labels.shape
        block = cell_labels[row0:row0 + H * k, col0:col0 + W * k]
        intended = np.repeat(np.repeat(target_labels, k, axis=0), k, axis=1)
        known = intended > 0
        tiles = block.reshape(H, k, W, k).transpose(0, 2, 1, 3).reshape(H, W, k * k)
        pixel_labels = np.where(np.all(tiles == tiles[:, :, :1], axis=2), tiles[:, :, 0], 0)
        edge_length = float(np.sum(np.abs(ends - starts)))
        cell_area = (xs[1] - xs[0]) * (ys[0] - ys[1])
        layer_area = float(np.count_nonzero(block == 0) * cell_area)
        certificate = dict(
            x_edges=xs, y_edges=ys, cell_labels=cell_labels, cell_bound=cell_bound,
            cells_per_pixel=k,
            window=np.array([xs[0], xs[-1], ys[-1], ys[0]]), image_offset=(row0, col0),
            pixel_labels=pixel_labels,
            area_certified_intended=float(np.count_nonzero((block == intended) & known) / known.sum()),
            area_certified_other=float(np.count_nonzero((block != intended) & (block > 0) & known) / known.sum()),
            area_uncertified=float(np.count_nonzero((block == 0) & known) / known.sum()),
            layer_mean_width_pixels=layer_area / edge_length / pixel_size if edge_length > 0 else 0.0,
            exterior_certified=ext_ok, exterior_sup_bound=ext_sup)

    method = opts["critical_method"]
    if method == "auto":
        method = "aberth" if opts["certify"] else "pencil"
    if opts["critical_check_max_degree"] == 0:
        method = "none"
    if method == "aberth" and certificate is None:
        raise ValueError("critical_method='aberth' needs certify=True.")
    if method == "aberth":
        stage(f"Critical points of S ({2 * poles.size - 2} on the sphere)")
        critical = dynamics.critical_analysis(
            r, S._k, poles, residues, certificate["x_edges"], certificate["y_edges"],
            certificate["cell_labels"], centers, trap_radii, certificate["exterior_certified"],
            max(opts["max_iterations"], 60),
            anchors=[(centers[j - 1], anchor_radius, targets[j - 1], caps[j - 1]) for j in other],
            cells_per_pixel=certificate["cells_per_pixel"],
            should_cancel=should_cancel, progress=progress)
        critical["status"] = (
            f"{critical['layer_count_found']} critical points in the uncertified layer "
            f"(argument principle: {critical['layer_count_argument_principle']}), "
            f"{int(np.count_nonzero(critical['basin_labels']))} of their values captured. "
            f"Fatou set = the {d} basins: {critical['verdict']}.")
    elif method == "pencil":
        stage("Critical points of S (eigenvalue pencil)")
        critical = check_critical_values(poles, residues, r, centers, trap_radii, opts)
        critical["fatou_set_verified"] = False
        critical["verdict"] = ("not verified: the eigenvalue check has no completeness count"
                               if critical["performed"] else "not checked")
    else:
        critical = dict(performed=False, method="none", status="skipped (not requested).",
                        fatou_set_verified=False, verdict="not checked")

    dyn = dict(
        degree=dict(S=int(poles.size), F=int(d), inner_iterations=int(n), r=int(poles.size) * int(d) ** int(n)),
        basins=basin_table,
        infinity=dynamics.infinity_data(poles, residues, shift, min_nodes, opts["anchor_degree"]),
        certificate=certificate, critical=critical,
        fatou_set_verified=bool(critical.get("fatou_set_verified", False)),
        fatou_verdict=critical.get("verdict", "not checked"),
    )

    info = dict(
        runge=S, poles=poles, residues=residues, constant=shift, rational_degree=poles.size,
        polynomial_degree=d, map_degree=poles.size * float(d) ** n, inner_iterations=n,
        anchor_origin=origin, anchor_scale=scale, runge_targets=targets, trap_centers=centers,
        trap_radii=trap_radii, fixed_points=fixed_points, fixed_point_residuals=fp_res,
        palette=palette, outside_label=outside, target_labels=target_labels,
        image_bounds=image_bounds, plot_bounds=bounds, x=x, y=y,
        basin_labels=basins, entry_iterations=iterations,
        unresolved_fraction=float(np.count_nonzero(basins == 0) / basins.size),
        core_max_runge_error=core_error, core_capture_fraction=core_capture,
        core_sample_count=core_count, critical_check=critical, dynamics=dyn,
        max_iterations=opts["max_iterations"], unresolved_color=_as_rgb(opts["unresolved_color"]),
    )

    if opts["verbose"]:
        print(report(info))

    if opts["output_image"]:
        from PIL import Image
        plot_palette = np.vstack([_as_rgb(opts["unresolved_color"]), palette])
        basin_rgb = color_raster(basins, plot_palette)
        Image.fromarray(np.floor(np.clip(basin_rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            opts["output_image"])
    if progress is not None:
        progress("Done")
    return r, info


# --------------------------------------------------------------------------
# Readable output

def _cfmt(z):
    return f"{z.real:+.6f}{z.imag:+.6f}i"


def report(info):
    """Plain-text summary of the map and its dynamics. Every line says how it was obtained."""
    dyn = info["dynamics"]
    deg = dyn["degree"]
    L = []
    L.append(f"Map        r = M o F^n o S,  deg S = {deg['S']},  deg F = {deg['F']},  n = {deg['inner_iterations']},  "
             f"deg r = {deg['S']}*{deg['F']}^{deg['inner_iterations']} = {deg['r']:.5g}")
    L.append("Basins     one attracting fixed point per label; trap = disk with r(disk) inside the disk of half the radius")
    L.append("           label  role      fixed point zeta          |r(zeta)-zeta|  log10|r'(zeta)|  trap radius  capture radius")
    for b in dyn["basins"]:
        mult = f"{'<= ' if b['multiplier_is_upper_bound'] else ''}{b['log10_abs_multiplier']:.1f}"
        L.append(f"           {b['label']:<5d}  {'exterior' if b['is_exterior'] else 'interior'}  "
                 f"{_cfmt(b['fixed_point']):<24s}  {b['fixed_point_residual']:<14.1e}  {mult:<15s}  "
                 f"{b['trap_radius']:<11.3e}  {b['capture_radius']:.3e}")
    L.append("           Traps are certified by an explicit bound (floating point), so |r'(zeta)| <= 1/2 by the Schwarz lemma.")
    L.append("           The computed multipliers are nonzero (attracting, not superattracting) but far below double range.")
    inf = dyn["infinity"]
    L.append(f"Infinity   S(Inf) = {inf['value']:.3g} (exterior target 0 plus RungeShift). By construction S - S(Inf) "
             f"vanishes there to order >= {inf['min_vanishing_order']},")
    L.append(f"           so Inf is a critical point of S of multiplicity >= {inf['min_critical_multiplicity']}; "
             f"its critical value r(Inf) lies in the exterior trap")
    cert = dyn["certificate"]
    if cert is not None:
        L.append(f"Certified  {100 * cert['area_certified_intended']:.2f}% of the image area lies in cells proved to be in the "
                 f"intended basin; {100 * cert['area_certified_other']:.2f}% proved to be in another basin;")
        L.append(f"           {100 * cert['area_uncertified']:.2f}% undecided (a layer about {cert['layer_mean_width_pixels']:.2f} "
                 f"input pixels wide along region edges; it contains the Julia set; cells are 1/{cert['cells_per_pixel']} pixel)")
        L.append(f"           everything outside the window [{cert['window'][0]:.3f}, {cert['window'][1]:.3f}] x "
                 f"[{cert['window'][2]:.3f}, {cert['window'][3]:.3f}] is in the exterior basin: "
                 f"{'proved' if cert['exterior_certified'] else 'NOT proved'}")
    crit = dyn["critical"]
    if crit.get("method") == "aberth":
        L.append(f"Critical   S has {crit['total_on_sphere']} critical points on the sphere (with multiplicity). In the undecided layer: "
                 f"{crit['layer_count_found']} computed, {crit['layer_count_argument_principle']} by the argument principle")
        if len(crit["multiple_points"]):
            L.append("           degenerate critical points (multiplicity from a local winding number): " +
                     ", ".join(f"{_cfmt(z)} (order {int(k)})" for z, k in
                               zip(crit["multiple_points"][:6], crit["multiple_point_orders"][:6])))
        steps = ", ".join((f"{v} with r(c) already in a trap" if k == 0 else
                           f"{v} after {k} more application{'s' if k != 1 else ''} of r")
                          for k, v in sorted(crit["entry_histogram"].items()))
        L.append(f"           their critical values are captured: {int(np.count_nonzero(crit['basin_labels']))} of "
                 f"{crit['distinct_found']} distinct points ({steps})")
        if crit["anchor_disks"]:
            L.append(f"           {crit['in_anchor_disks']} more sit inside the {len(crit['anchor_disks'])} anchor disks "
                     f"(counted by the argument principle; each disk is proved to map into its trap)")
        if crit["outside_layer"] is not None:
            L.append(f"           the other {crit['outside_layer']} lie in certified cells or outside the window, "
                     f"so their values are captured as well")
    else:
        L.append(f"Critical   {crit['status']}")
    L.append(f"Fatou set  equal to the union of the {deg['F']} basins (paper, Remark 2.12): {dyn['fatou_verdict']}")
    L.append(f"Render     {info['basin_labels'].shape[1]}x{info['basin_labels'].shape[0]} grid, "
             f"{100 * info['unresolved_fraction']:.2f}% not captured within {info['max_iterations']} iterations")
    if info["core_sample_count"] > 0:
        L.append(f"           sampled pixel centers sent to the intended trap in one step: "
                 f"{100 * info['core_capture_fraction']:.2f}% of {info['core_sample_count']}")
    return "\n".join(L)


def summary(info):
    """JSON-serializable dictionary with the dynamical data (no large arrays)."""
    def conv(v):
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()
                    if not (isinstance(x, np.ndarray) and x.size > 64)}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, np.ndarray):
            return [conv(x) for x in v.tolist()]
        if isinstance(v, (complex, np.complexfloating)):
            return {"re": float(v.real), "im": float(v.imag)}
        if isinstance(v, (np.floating, float)):
            return float(v) if math.isfinite(v) else str(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    out = conv(info["dynamics"])
    out["render"] = dict(unresolved_fraction=info["unresolved_fraction"], max_iterations=info["max_iterations"],
                         core_capture_fraction=conv(info["core_capture_fraction"]),
                         core_sample_count=info["core_sample_count"])
    return out


def save_summary(info, path):
    """Write ``summary(info)`` as JSON and the text report next to it (path + '.txt')."""
    import json
    with open(path, "w") as fh:
        json.dump(summary(info), fh, indent=2)
    with open(str(path) + ".txt", "w") as fh:
        fh.write(report(info) + "\n")


DEFAULTS = dict(_DEFAULTS)
