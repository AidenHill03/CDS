"""app/colour.py -- turns a raw render array into an RGB image.

Three input kinds, matching the three things cdx.Renderer actually produces:
  - ESCAPE-TIME (julia/parameter modes): smooth escape value, 0 = never
    escaped (see cdx::Renderer::render_julia/render_parameter's own doc
    comment -- that convention is trusted here, not re-derived).
  - BASIN (categorical + shading): a label array (0 = unresolved, k = basin
    k, per cdx::Renderer::render_basin), optionally paired with a
    convergence-iteration array for shading.
  - SCALAR FIELD (Green's-function-shaped data): a non-negative potential,
    every pixel meaningful -- no "never escaped" sentinel.

SCALING is the part that determines whether images look right. Measured on
the Mandelbrot parameter plane at max_iter=200: 90% of escaped pixels have
smooth escape value <= 10 out of a range to 200. A LINEAR map to the palette
therefore puts 90% of the image in the first 5% of the colour range and
looks broken.

DEFAULT is log1p scaling (scale_log1p), anchored to a caller-supplied
`reference` (e.g. max_iter) rather than to this array's own realized
min/max -- that is what makes it COMPARABLE: the same raw escape value
always lands at the same point in [0,1], in any render that shares the same
reference/period, regardless of what other pixels happen to be in this
particular image. An adjustable cyclic `period` gives the classic banded
look (wraps the VALUE, not the scaled position, before taking log1p) and is
still monotonic within each band.

HISTOGRAM EQUALISATION (scale_histogram_eq) is offered as an explicit
enhancement/density mode: it rank-transforms each image against its OWN
empirical distribution, using the full palette range evenly, at the direct
cost of the comparability property above -- the same raw value maps to a
DIFFERENT colour depending on what else is in the image. Never the default
for exactly that reason; this module never picks it silently.

PALETTES: grayscale, a classic blue/orange two-tone, and two perceptually
uniform ones (viridis, magma) for when the image IS the data, not
decoration. The two 256-entry perceptual LUTs are baked in below as
base64-encoded raw RGB bytes, generated ONCE via matplotlib as a dev-time
tool (see the generation snippet in this module's commit message) -- this
module has NO runtime dependency on matplotlib, which matters for
PyInstaller packaging (P5c section 7): bundling all of matplotlib just for
a 768-byte lookup table would be pure bloat.

NEVER-ESCAPED PIXELS (escape-time) and UNRESOLVED PIXELS (basin) always get
a flat, separate colour -- NOT palette index 0. Palette index 0 is a real
(if extreme) data colour; conflating "no data" with "the fastest-escaping
pixel in the image" would be a real correctness problem for a research
tool, not just an aesthetic one.
"""

from __future__ import annotations

import base64
import colorsys
import math

import numpy as np

# ---- palettes -----------------------------------------------------------------------

_VIRIDIS_B64 = (
    "RAFURAJWRQRXRQVZRgdaRghcRgpdRgteRw1gRw5hRxBjRxFkRxNlSBRnSBZoSBdpSBhqSBpsSBttSBxu"
    "SB1vSB9wSCBxSCFzSCN0SCR1SCV2SCZ3SCh4SCl5Ryp6Ryx6Ry17Ry58Ry99RjB+RjJ+RjN/RjSARTWB"
    "RTeBRTiCRDmDRDqDRDuEQz2EQz6FQj+FQkCGQkGGQUKHQUSHQEWIQEaIP0eIP0iJPkmJPkqJPkyKPU2K"
    "PU6KPE+KPFCLO1GLO1KLOlOLOlSMOVWMOVaMOFiMOFmMN1qMN1uNNlyNNl2NNV6NNV+NNGCNNGGNM2KN"
    "M2ONMmSOMmWOMWaOMWeOMWiOMGmOMGqOL2uOL2yOLm2OLm6OLm+OLXCOLXGOLHGOLHKOLHOOK3SOK3WO"
    "KnaOKneOKniOKXmOKXqOKXuOKHyOKH2OJ36OJ3+OJ4COJoGOJoKOJoKOJYOOJYSOJYWOJIaOJIeOI4iO"
    "I4mOI4qNIouNIoyNIo2NIY6NIY+NIZCNIZGMIJKMIJKMIJOMH5SMH5WLH5aLH5eLH5iLH5mKH5qKHpuK"
    "HpyJHp2JH56JH5+IH6CIH6GIH6GHH6KHIKOGIKSGIaWFIaaFIqeFIqiEI6mDJKqDJauCJayCJq2BJ62B"
    "KK6AKa9/KrB/LLF+LbJ9LrN8L7R8MbV7MrZ6NLZ5Nbd5N7h4OLl3Orp2O7t1Pbx0P7xzQL1yQr5xRL9w"
    "RsBvSMFuSsFtTMJsTsNrUMRqUsVpVMVoVsZnWMdlWshkXMhjXsliYMpgY8tfZcteZ8xcac1bbM1abs5Y"
    "cM9Xc9BWddBUd9FTetFRfNJQf9NOgdNNhNRLhtVJidVIi9ZGjtZFkNdDk9dBldhAmNg+m9k8ndk7oNo5"
    "oto3pds2qNs0qtwyrdwwsN0vst0ttd4ruN4put4ovd8mwN8lwt8jxeAhyOAgyuEfzeEd0OEc0uIb1eIa"
    "2OIZ2uMZ3eMY3+MY4uQY5eQZ5+QZ6uUa7OUb7+Uc8eUd9OYe9uYg+OYh++cj/ecl"
)

_MAGMA_B64 = (
    "AAAEAQAFAQEGAQEIAgEJAgILAgINAwMPAwMSBAQUBQQWBgUYBgUaBwYcCAceCQcgCggiCwkkDAkmDQop"
    "DgsrEAstEQwvEg0xEw00FA42FQ44Fg87GA89GRA/GhBCHBBEHRFHHhFJIBFLIRFOIhFQJBJTJRJVJxJY"
    "KRFaKhFcLBFfLRFhLxFjMRFlMxBnNBBpNhBrOBBsOQ9uOw9wPQ9xPw9yQA90Qg91RA92RRB3RxB4SRB4"
    "ShB5TBF6ThF7TxJ7URJ8UhN8VBN9VhR9VxV+WRV+WhZ+XBZ/XRd/Xxh/YBiAYhmAZBqAZRqAZxuAaByB"
    "ahyBax2BbR2Bbh6BcB+Bch+BcyCBdSGBdiGBeCKBeSKCeyOCfCOCfiSCgCWCgSWBgyaBhCaBhieBiCeB"
    "iSiBiymBjCmBjiqBkCqBkSuBkyuAlCyAliyAmC2AmS2Amy5/nC5/ni9/oC9/oTB+ozB+pTF+pjF9qDJ9"
    "qjN9qzN8rTR8rjR7sDV7sjV7szZ6tTZ6tzd5uDd5ujh4vDl4vTl3vzp3wDp2wjt1xDx1xTx0xz1zyD5z"
    "yj5yzD9xzUBxz0Bw0EFv0kJv00Nu1URt1kVs2EVs2UZr20dq3Ehp3klo30po4Exn4k1m405l5E9k5VBk"
    "51Jj6FNi6VRi6lZh61dg7Fhg7Vpf7lte711e8F9e8WBd8mJd8mRc82Vc9Gdc9Glc9Wtc9mxc9m5c93Bc"
    "93Jc+HRc+HZc+Xhd+Xld+Xtd+n1e+n9e+oFf+4Nf+4Vg+4dh/Ilh/Ipi/Ixj/I5k/JBl/ZJm/ZRn/ZZo"
    "/Zhp/Zpq/Ztr/p1s/p9t/qFu/qNv/qVx/qdy/qlz/qp0/qx2/q53/rB4/rJ6/rR7/rZ8/rd+/rl//ruB"
    "/r2C/r+E/sGF/sKH/sSI/saK/siM/sqN/syP/s2Q/s+S/tGU/tOV/tWX/teZ/tia/dqc/dye/d6g/eCh"
    "/eKj/eOl/eWn/eep/emq/eus/Oyu/O6w/PCy/PK0/PS2/Pa4/Pe5/Pm7/Pu9/P2/"
)


def _decode_lut(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(256, 3).copy()


def _linear_gray() -> np.ndarray:
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp, ramp, ramp], axis=1)


def _interpolate_stops(stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    xs = np.array([s[0] for s in stops])
    colours = np.array([s[1] for s in stops], dtype=float)
    t = np.linspace(0.0, 1.0, 256)
    out = np.empty((256, 3))
    for ch in range(3):
        out[:, ch] = np.interp(t, xs, colours[:, ch])
    return out.round().clip(0, 255).astype(np.uint8)


def _classic_blue_orange() -> np.ndarray:
    # Dark blue -> warm white -> orange: the familiar Mandelbrot-renderer
    # look, deliberately passing through a bright midpoint rather than a
    # straight two-colour blend (which would look muddy through the
    # middle of the range).
    return _interpolate_stops([
        (0.0, (0, 7, 100)),
        (0.5, (255, 255, 245)),
        (1.0, (255, 140, 0)),
    ])


PALETTES: dict[str, np.ndarray] = {
    "grayscale": _linear_gray(),
    "classic": _classic_blue_orange(),
    "viridis": _decode_lut(_VIRIDIS_B64),
    "magma": _decode_lut(_MAGMA_B64),
}
PALETTE_NAMES: tuple[str, ...] = tuple(PALETTES.keys())

# Deliberately NOT palette index 0 -- see module docstring.
NEVER_ESCAPED_RGB: tuple[int, int, int] = (0, 0, 0)
UNRESOLVED_BASIN_RGB: tuple[int, int, int] = (0, 0, 0)
# Same flat-black "nothing resolved here" convention as the two above,
# for colour_parameter_basin's own unresolved-dominant pixels -- a
# DIFFERENT question (a whole PARAMETER's worth of critical orbits mostly
# failing to settle, not one pixel's own orbit never reaching a known
# attractor), but visually the same "black = nothing here" language this
# app already speaks everywhere else.
PARAMETER_BASIN_UNRESOLVED_RGB: tuple[int, int, int] = (0, 0, 0)

SCALING_MODES: tuple[str, ...] = ("log1p", "histogram")


def _apply_palette(t: np.ndarray, mask: np.ndarray, palette: np.ndarray,
                   masked_rgb: tuple[int, int, int]) -> np.ndarray:
    n = len(palette)
    idx = np.clip((t * (n - 1)).round().astype(np.int64), 0, n - 1)
    rgb = palette[idx].copy()
    rgb[~mask] = masked_rgb
    return rgb


# ---- scaling ------------------------------------------------------------------------

def scale_log1p(values: np.ndarray, reference: float, period: float | None = None) -> np.ndarray:
    """[0,1]-normalized colour position, monotonic in `values` (within a
    period-band, if `period` is given) and independent of THIS array's own
    realized min/max -- the same raw value always maps to the same output
    given the same (reference, period), which is what keeps two renders
    (different viewport, same settings) quantitatively comparable. See
    scale_histogram_eq for the contrasting, non-comparable alternative.

    `reference` anchors the log1p denominator (max_iter for escape-time,
    or a settings-derived scale for a scalar field) instead of deriving it
    from the data. `period`, if given, wraps VALUES modulo period BEFORE
    taking log1p -- the classic "a band every N iterations" look.
    """
    working = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if period is not None and period > 0:
        working = np.mod(working, period)
        denom = math.log1p(period)
    else:
        denom = math.log1p(reference) if reference > 0 else 1.0
    if denom <= 0:
        return np.zeros_like(working)
    return np.clip(np.log1p(working) / denom, 0.0, 1.0)


def scale_histogram_eq(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Rank-transforms `values` (restricted to `mask`, if given) into
    [0,1] via their own empirical CDF -- uses the full palette range for
    THIS image, at the cost of the comparability scale_log1p has: the
    same raw value maps to a DIFFERENT position depending on what other
    pixels happen to be in this particular render. An explicit
    enhancement/density mode; callers must opt in.
    """
    values = np.asarray(values, dtype=float)
    selected = values.ravel() if mask is None else values[mask]
    if selected.size == 0:
        return np.zeros_like(values)
    order = np.argsort(selected)
    ranks = np.empty(selected.size, dtype=float)
    denom = max(selected.size - 1, 1)
    ranks[order] = np.arange(selected.size) / denom
    if mask is None:
        return ranks.reshape(values.shape)
    out = np.zeros_like(values)
    out[mask] = ranks
    return out


def scale_scalar_field(values: np.ndarray, band_width: float = 1.0,
                       eps: float = 1e-9) -> np.ndarray:
    """log(value + eps), divided by a FIXED band width (natural-log
    units) -- not normalized to this array's own min/max, so two renders
    of the same map/settings but a different viewport stay comparable,
    the same principle as scale_log1p above. Returns a value in roughly
    [log(eps)/band_width, inf); callers wanting a cyclic banded look wrap
    this with `np.mod(result, n_bands)` themselves (see colour_scalar_field
    and, for the discrete equipotential-band UI, app/greens_view.py).
    """
    return np.log(np.clip(np.asarray(values, dtype=float), 0.0, None) + eps) / band_width


# ---- escape-time ----------------------------------------------------------------------

def colour_escape_time(values: np.ndarray, max_iter: int, palette: str = "viridis",
                       scaling: str = "log1p", period: float | None = None) -> np.ndarray:
    """The full escape-time colouring pipeline: raw smooth-escape values
    -> an (H, W, 3) uint8 RGB image. 0 always means "never escaped" (see
    module docstring) and always gets NEVER_ESCAPED_RGB, never a palette
    colour -- including palette index 0, which is a real escape-value
    colour, not a stand-in for "no data."
    """
    if scaling not in SCALING_MODES:
        raise ValueError(f"unknown scaling {scaling!r}; must be one of {SCALING_MODES}")
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}; must be one of {PALETTE_NAMES}")
    values = np.asarray(values, dtype=float)
    escaped = values > 0.0
    if scaling == "log1p":
        t = scale_log1p(values, max_iter, period)
    else:
        t = scale_histogram_eq(values, escaped)
    return _apply_palette(t, escaped, PALETTES[palette], NEVER_ESCAPED_RGB)


# ---- basin: categorical hue + optional convergence-speed shading ----------------------

_GOLDEN_ANGLE = 0.6180339887498949   # 1/phi -- successive hues land far apart on the wheel


def _golden_hue(index: int) -> float:
    """A deterministic, well-separated hue per basin index -- golden-angle
    spacing, the standard trick for generating N visually distinct colours
    without knowing N (the number of basins) in advance.
    """
    return math.fmod(index * _GOLDEN_ANGLE, 1.0)


def _hue_to_rgb_255(h: float, s: float = 0.75, v: float = 1.0) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r * 255.0, g * 255.0, b * 255.0)


def colour_basin(labels: np.ndarray, iterations: np.ndarray | None = None,
                 max_iter: int = 1, period: float | None = None,
                 scaling: str = "log1p") -> np.ndarray:
    """Hue = basin index (golden-angle rotation, see _golden_hue), brightness
    = convergence speed when `iterations` is given -- the SAME scale_log1p/
    scale_histogram_eq machinery escape-time uses (colour_escape_time),
    since basin iteration counts have the identical skew CLAUDE.md's
    measurement describes for escape counts. Unresolved pixels (label == 0,
    per cdx::Renderer::render_basin's own "0 = unresolved" convention) are
    always flat UNRESOLVED_BASIN_RGB, regardless of shading.

    `scaling` picks between the two exactly as colour_escape_time does:
    "log1p" (default, comparable across renders, `period` gives the cyclic
    banded look) or "histogram" (scale_histogram_eq, ranked against just
    THIS image's own resolved pixels via `mask=resolved` -- unresolved
    pixels take no part in the ranking, the same "restricted to real data"
    treatment colour_escape_time's own histogram branch gives escaped
    pixels). `period` only has meaning for "log1p" -- scale_histogram_eq
    has no periodic concept, exactly as colour_escape_time's own histogram
    branch already ignores it.

    Brighter = converged FASTER (a small iteration count): points deep in
    a basin, far from the boundary, typically resolve almost immediately;
    points near the Julia-set boundary take many more iterations to
    settle. Mapping SLOW convergence to dark makes that boundary structure
    the visually prominent feature, which is the point of basin shading. A
    shaded pixel never goes fully black (see the 0.15 floor below) so it
    stays visually distinct from an UNRESOLVED pixel, which is exactly and
    only black.
    """
    if scaling not in SCALING_MODES:
        raise ValueError(f"unknown scaling {scaling!r}; must be one of {SCALING_MODES}")
    labels = np.asarray(labels)
    resolved = labels > 0
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    if not np.any(resolved):
        return rgb

    base = np.zeros(labels.shape + (3,), dtype=float)
    for basin_id in np.unique(labels[resolved]).astype(np.int64):
        base[labels == basin_id] = _hue_to_rgb_255(_golden_hue(int(basin_id)))

    if iterations is None:
        shaded = base
    else:
        if scaling == "log1p":
            t = scale_log1p(iterations, max_iter, period)
        else:
            t = scale_histogram_eq(iterations, resolved)
        brightness = 0.15 + 0.85 * (1.0 - t)   # fast convergence (t near 0) -> bright
        shaded = base * brightness[..., None]

    rgb = shaded.round().clip(0, 255).astype(np.uint8)
    rgb[~resolved] = UNRESOLVED_BASIN_RGB
    return rgb


def colour_parameter_basin(counts: np.ndarray, unresolved: np.ndarray | None = None) -> np.ndarray:
    """Categorical colouring for Parameter_basin: hue = the COUNT itself
    (golden-angle spacing, the SAME _golden_hue/_hue_to_rgb_255 primitive
    colour_basin already uses for basin ids -- N distinct, well-separated
    colours without needing to know N, the largest count this render will
    ever show, in advance). Deliberately NOT colour_escape_time's palette-
    gradient machinery: a count is a CATEGORY (2 is not "a bit more than
    1" the way a larger escape iteration is), so a gradient would imply an
    ordering relationship between adjacent counts that isn't actually
    there -- count=1 and count=2 should read as two DIFFERENT things, not
    two shades of the same thing. This is what makes a bifurcation
    boundary (where the count changes) read as a sharp colour EDGE rather
    than a subtle gradient shift.

    A pixel is UNRESOLVED-DOMINANT -- flat PARAMETER_BASIN_UNRESOLVED_RGB,
    never a count colour -- when count itself is 0 (nothing was ever
    confirmed, so there is no real category to show) OR when `unresolved`
    (given) exceeds `counts` at that pixel (more of this parameter's
    critical orbits failed to resolve than succeeded, so the confirmed
    count is more noise than signal). A pixel with count >= 1 and SOME,
    but not a majority, of its orbits unresolved still gets its real
    count's colour -- a confirmed attractor is shown as confirmed
    regardless of what else, separately, didn't resolve.

    Shading WITHIN a count region by slowest convergence rate (finer
    structure inside one colour band) is an intentionally deferred,
    optional refinement -- see cdx::Renderer::render_parameter_basin's own
    doc comment; it does not currently expose a convergence-rate channel
    for this to shade by at all, so there is nothing to wire up yet, not
    a corner cut.
    """
    counts = np.asarray(counts)
    unresolved_arr = np.zeros_like(counts) if unresolved is None else np.asarray(unresolved)

    dominant_unresolved = (counts == 0) | (unresolved_arr > counts)
    resolved = ~dominant_unresolved

    base = np.zeros(counts.shape + (3,), dtype=float)
    if np.any(resolved):
        for count_value in np.unique(counts[resolved]).astype(np.int64):
            base[counts == count_value] = _hue_to_rgb_255(_golden_hue(int(count_value)))

    rgb = base.round().clip(0, 255).astype(np.uint8)
    rgb[dominant_unresolved] = PARAMETER_BASIN_UNRESOLVED_RGB
    return rgb


# ---- Julia, RATIONAL classification (Stage 2 of the sphere-aware milestone) ------------
#
# Deliberately NOT a bespoke colourer here. A rational map's "julia" render
# colours through colour_escape_time -- the SAME function a certified
# polynomial's does -- using only the smooth chordal approach-rate (the
# stacked payload's own values layer; see cdx::Renderer::render_julia's own
# doc comment). The basin LABEL a pixel reached is still computed and kept
# available (for the cursor readout -- see ImageView._sample_at_pixel), just
# never used for colouring: which basin a pixel is in is BASIN mode's own
# question, not Julia's. A per-attractor hue scheme was tried here first and
# reverted -- it changed Julia's own COLOURING PHILOSOPHY (palette + scaling,
# honoured by every other escape-time-shaped mode) into something bespoke,
# which is exactly what this milestone's own governing principle rules out:
# the sphere-aware revamp changes CLASSIFICATION and the SCALAR a pixel
# produces, never a mode's colouring philosophy.


# ---- scalar field (Green's-function-shaped data) ---------------------------------------

def colour_scalar_field(values: np.ndarray, palette: str = "viridis", band_width: float = 1.0,
                        period_bands: float = 12.0, eps: float = 1e-9,
                        contour: bool = False,
                        contour_rgb: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Full scalar-field colouring pipeline: quantises log(value+eps) into
    `period_bands` cyclic steps (via the same palette machinery as
    escape-time), giving the classic equipotential-banded look. Every
    pixel gets a real colour -- there is no "never escaped"-style mask
    here, since 0 is a legitimate potential value, not a sentinel.

    CONTOUR LINES, when `contour` is set: drawn directly into the pixel
    grid (not a separate QPainter overlay -- an equipotential boundary is
    exactly "where the integer band index differs between adjacent
    pixels," a property of the rendered array itself, not of screen/
    viewport geometry the way a critical-point marker is). A pixel is on
    a contour if EITHER its left or its top neighbour landed in a
    different (unwrapped) band -- checked on the integer band index
    BEFORE the cyclic period wrap, so a contour line marks every true
    equipotential boundary, not just the ones that happen to also cross
    a period-wrap seam.
    """
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}; must be one of {PALETTE_NAMES}")
    bands = scale_scalar_field(values, band_width, eps)
    t = np.mod(bands, period_bands) / period_bands if period_bands > 0 else np.zeros_like(bands)
    mask = np.ones(np.asarray(values).shape, dtype=bool)
    rgb = _apply_palette(t, mask, PALETTES[palette], (0, 0, 0))
    if contour:
        band_index = np.floor(bands).astype(np.int64)
        on_contour = np.zeros(band_index.shape, dtype=bool)
        on_contour[:, 1:] |= band_index[:, 1:] != band_index[:, :-1]
        on_contour[1:, :] |= band_index[1:, :] != band_index[:-1, :]
        rgb = rgb.copy()
        rgb[on_contour] = contour_rgb
    return rgb
