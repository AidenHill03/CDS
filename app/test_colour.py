"""test_colour.py -- property-based checks for app/colour.

Mirrors the rest of this project's test style (PASS/FAIL per check,
properties rather than golden images -- a colour LUT is exactly the kind of
thing a pixel-diff test would rot on for no functional reason). Run with:

    PYTHONPATH=cdx/build python -m app.test_colour

(from the repository root -- no PySide6/Qt dependency at all; this module
is pure numpy.)
"""

from __future__ import annotations

import numpy as np

from app.colour import (NEVER_ESCAPED_RGB, PALETTE_NAMES, PALETTES,
                        PARAMETER_BASIN_UNRESOLVED_RGB, UNRESOLVED_BASIN_RGB, colour_basin,
                        colour_escape_time, colour_parameter_basin, colour_scalar_field,
                        scale_histogram_eq, scale_log1p, scale_scalar_field)

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    print("=== app.colour tests ===")

    # ---- palettes -----------------------------------------------------------------
    print("\npalettes:")
    for name, lut in PALETTES.items():
        check(lut.shape == (256, 3), f"{name}: shape is (256, 3)")
        check(lut.dtype == np.uint8, f"{name}: dtype is uint8")
        check(0 <= lut.min() and lut.max() <= 255, f"{name}: all values in [0, 255]")

    gray = PALETTES["grayscale"]
    check(np.array_equal(gray[:, 0], gray[:, 1]) and np.array_equal(gray[:, 1], gray[:, 2]),
          "grayscale: R == G == B at every index")
    check(np.all(np.diff(gray[:, 0].astype(int)) >= 0), "grayscale: strictly non-decreasing ramp")

    classic = PALETTES["classic"]
    check(tuple(classic[0]) == (0, 7, 100), "classic: starts at the specified dark blue")
    check(tuple(classic[255]) == (255, 140, 0), "classic: ends at the specified orange")

    # Known canonical anchor values for matplotlib's viridis/magma at their
    # first/last stops -- confirms the baked-in LUT is the real thing, not
    # a placeholder or a corrupted decode.
    check(tuple(PALETTES["viridis"][0]) == (68, 1, 84), "viridis[0] matches the canonical anchor")
    check(tuple(PALETTES["viridis"][255]) == (253, 231, 37),
          "viridis[255] matches the canonical anchor")
    check(tuple(PALETTES["magma"][0]) == (0, 0, 4), "magma[0] matches the canonical anchor")
    check(tuple(PALETTES["magma"][255]) == (252, 253, 191),
          "magma[255] matches the canonical anchor")

    check(set(PALETTE_NAMES) == set(PALETTES.keys()), "PALETTE_NAMES matches PALETTES' keys")

    # ---- scale_log1p: monotonic, reference-anchored, comparable across images -----
    print("\nscale_log1p:")
    values = np.array([0.0, 1.0, 5.0, 20.0, 100.0, 199.0])
    t = scale_log1p(values, reference=200)
    check(np.all(np.diff(t) >= 0), "output is non-decreasing as input increases (no period)")
    check(0.0 <= t.min() and t.max() <= 1.0, "output stays within [0, 1]")

    # THE comparability property: the SAME value, in TWO DIFFERENT arrays
    # with different realized max/min, maps to the SAME output as long as
    # `reference` is the same -- this is what distinguishes it from
    # histogram equalisation below, and is the entire reason it's the
    # default.
    a1 = scale_log1p(np.array([5.0, 500.0]), reference=200)[0]
    a2 = scale_log1p(np.array([5.0, 6.0, 7.0]), reference=200)[0]
    check(a1 == a2,
          "the value 5.0 scales identically regardless of what else is in the array -- "
          "comparability across renders")

    # log1p(0) with a max_iter=0 image (degenerate, but must not divide by zero)
    degenerate = scale_log1p(np.array([0.0]), reference=0)
    check(np.all(np.isfinite(degenerate)), "reference=0 degrades to zeros, not NaN/Inf")

    periodic = scale_log1p(np.array([0.0, 10.0, 20.0]), reference=200, period=10.0)
    check(abs(periodic[0]) < 1e-9,
          "with period=10, value 0.0 wraps to the same band position as value 20.0 (0 mod 10 == "
          "20 mod 10 == 0)")
    check(abs(periodic[0] - periodic[2]) < 1e-9,
          "values one full period apart land at the identical scaled position -- the banding wrap")

    # ---- scale_histogram_eq: uses the full range, NOT comparable across images ----
    print("\nscale_histogram_eq:")
    h1 = scale_histogram_eq(np.array([5.0, 10.0, 15.0]))
    check(h1.min() == 0.0 and h1.max() == 1.0,
          "histogram-eq uses the FULL [0,1] range for this image regardless of the raw values")

    b1 = scale_histogram_eq(np.array([5.0, 500.0]))[0]                       # 5.0 is the min: rank 0
    b2 = scale_histogram_eq(np.array([1.0, 2.0, 5.0, 100.0, 200.0]))[2]      # 5.0 is the middle: rank 0.5
    check(b1 != b2,
          "the SAME value (5.0) scales DIFFERENTLY depending on the rest of the array -- the "
          "exact non-comparability scale_log1p's reference-anchoring avoids, and why this mode "
          "is opt-in rather than the default")

    masked = scale_histogram_eq(np.array([1.0, 999.0, 2.0]), mask=np.array([True, False, True]))
    check(masked[1] == 0.0, "a masked-out position is not included in the ranking")
    check(masked[0] != masked[2], "masked-in positions still rank against each other")

    # ---- colour_escape_time: never-escaped gets a flat colour, not palette index 0 --
    print("\ncolour_escape_time:")
    vals = np.array([0.0, 1.0, 50.0, 199.0])
    img = colour_escape_time(vals, max_iter=200, palette="viridis")
    check(img.shape == (4, 3) and img.dtype == np.uint8, "output shape/dtype")
    check(tuple(img[0]) == NEVER_ESCAPED_RGB, "value 0.0 (never escaped) gets NEVER_ESCAPED_RGB")
    check(tuple(img[0]) != tuple(PALETTES["viridis"][0]),
          "never-escaped is NOT palette index 0 -- distinguishable from 'escaped almost instantly'")
    check(np.array_equal(img[3], PALETTES["viridis"][255]),
          "the highest escape value (near max_iter) lands at (or very near) the palette's top end")

    try:
        colour_escape_time(vals, max_iter=200, palette="not-a-real-palette")
        check(False, "an unknown palette name raises")
    except ValueError as e:
        check("palette" in str(e), "ValueError names the bad palette")

    try:
        colour_escape_time(vals, max_iter=200, scaling="not-a-real-scaling")
        check(False, "an unknown scaling name raises")
    except ValueError as e:
        check("scaling" in str(e), "ValueError names the bad scaling mode")

    # ---- colour_basin: unresolved is flat black; hue distinct; shading monotonic ----
    print("\ncolour_basin:")
    labels = np.array([0, 1, 1, 2, 2])
    iters = np.array([0, 3, 190, 3, 190])
    bimg = colour_basin(labels, iters, max_iter=200)
    check(tuple(bimg[0]) == UNRESOLVED_BASIN_RGB, "an unresolved (label 0) pixel is flat black")

    check(tuple(bimg[1]) != tuple(bimg[3]),
          "two different basin ids (1 vs 2) get visually distinct hues")

    fast, slow = bimg[1], bimg[2]   # same basin id (1), different convergence speed
    check(sum(int(c) for c in fast) > sum(int(c) for c in slow),
          "within the SAME basin, faster convergence (iters=3) is brighter than slower "
          "convergence (iters=190)")
    check(tuple(slow) != UNRESOLVED_BASIN_RGB,
          "even the slowest-converging RESOLVED pixel stays visually distinct from unresolved "
          "black (the 0.15 brightness floor)")

    flat = colour_basin(labels, iterations=None)
    check(tuple(flat[1]) == tuple(flat[2]),
          "with no iterations array, basin colouring is flat -- same basin id, same colour "
          "regardless of position")

    all_unresolved = colour_basin(np.zeros(5, dtype=int))
    check(np.all(all_unresolved == 0), "an all-unresolved label array is entirely black, no crash")

    # ---- colour_basin: gradient scaling modes (Stage 0) ------------------------------
    print("\ncolour_basin: gradient scaling modes:")
    try:
        colour_basin(labels, iters, max_iter=200, scaling="not-a-real-scaling")
        check(False, "an unknown scaling name raises")
    except ValueError as e:
        check("scaling" in str(e), "ValueError names the bad scaling mode, same as "
              "colour_escape_time's own validation")

    # One basin, six pixels spanning a REAL range of convergence speed -- not just
    # the two-sample fast/slow check above -- to confirm the gradient is a genuine
    # spread across the basin, not a near-flat wash.
    rich_labels = np.ones(6, dtype=int)
    rich_iters = np.array([1, 2, 5, 20, 80, 190])
    for scaling in ("log1p", "histogram"):
        rimg = colour_basin(rich_labels, rich_iters, max_iter=200, scaling=scaling)
        brightness = rimg.astype(float).sum(axis=1)   # proportional to the 0.15..1.0 factor
        check(brightness.max() - brightness.min() > brightness.max() * 0.3,
              f"{scaling} scaling produces a REAL spread of brightness across iterations "
              f"1..190 within a single basin, not a near-flat gradient")
        check(all(brightness[i] >= brightness[i + 1] for i in range(len(brightness) - 1)),
              f"{scaling} scaling is monotonic -- brightness never increases with slower "
              f"convergence")

    # A cyclic period reaches basin the same way it already does for escape-time --
    # values one period apart land at the same brightness.
    period_labels = np.ones(2, dtype=int)
    period_iters = np.array([1.0, 11.0])   # one period (10) apart
    pimg = colour_basin(period_labels, period_iters, max_iter=200, period=10.0)
    check(tuple(pimg[0]) == tuple(pimg[1]),
          "a cyclic period reaches basin's log1p scaling too -- values one period apart "
          "land at the same brightness")

    # Switching scaling mode changes ONLY the shading, never which basin gets which
    # hue -- distinct-hue and unresolved/flat behavior hold under histogram scaling too.
    hist_hue_img = colour_basin(labels, iters, max_iter=200, scaling="histogram")
    check(tuple(hist_hue_img[0]) == UNRESOLVED_BASIN_RGB,
          "unresolved stays flat black under histogram scaling too")
    check(tuple(hist_hue_img[1]) != tuple(hist_hue_img[3]),
          "two different basin ids stay visually distinct under histogram scaling too")
    hist_flat = colour_basin(labels, iterations=None, scaling="histogram")
    check(tuple(hist_flat[1]) == tuple(flat[1]),
          "with no iterations array, the scaling mode has no effect at all -- same flat "
          "colour regardless of 'scaling'")

    # ---- colour_parameter_basin: categorical by count, unresolved-dominant flat -----
    print("\ncolour_parameter_basin:")
    counts = np.array([0, 1, 1, 2, 3])
    pbimg = colour_parameter_basin(counts)
    check(tuple(pbimg[0]) == PARAMETER_BASIN_UNRESOLVED_RGB,
          "count == 0 gets the flat unresolved colour, not a 'count 0' hue of its own")
    check(tuple(pbimg[1]) == tuple(pbimg[2]),
          "the SAME count gets the SAME colour, deterministically")
    check(tuple(pbimg[1]) != tuple(pbimg[3]) and tuple(pbimg[3]) != tuple(pbimg[4]) and
         tuple(pbimg[1]) != tuple(pbimg[4]),
          "three DIFFERENT counts (1, 2, 3) get three visually distinct colours")

    # Categorical, not a gradient: count=1 and count=2 aren't required to be
    # "close" in colour space the way adjacent escape-time values would be
    # under a palette gradient -- confirmed structurally by the distinctness
    # checks above already using the golden-angle-separated hue wheel, not a
    # linear ramp (no ordering assumption asserted here on purpose).

    no_unresolved = colour_parameter_basin(np.array([1, 2]))
    check(tuple(no_unresolved[0]) == tuple(colour_parameter_basin(np.array([1, 2]),
                                                                  np.array([0, 0]))[0]),
          "omitting `unresolved` entirely behaves the same as passing an all-zero array")

    dominant = colour_parameter_basin(np.array([2, 2]), np.array([1, 3]))
    check(tuple(dominant[0]) != PARAMETER_BASIN_UNRESOLVED_RGB,
          "count=2 with unresolved=1 (a MINORITY of orbits unresolved) still shows the "
          "real count's colour -- a confirmed attractor is shown as confirmed")
    check(tuple(dominant[1]) == PARAMETER_BASIN_UNRESOLVED_RGB,
          "count=2 with unresolved=3 (unresolved EXCEEDS count -- a majority) is "
          "UNRESOLVED-DOMINANT and gets the flat neutral colour instead, not conflated "
          "with a real count")

    all_unresolved_pb = colour_parameter_basin(np.zeros(4, dtype=int))
    check(np.all(all_unresolved_pb == 0), "an all-zero count array is entirely the flat "
          "unresolved colour, no crash")

    # Rational Julia (Stage 2) has NO bespoke colourer at this layer any more
    # -- it colours through colour_escape_time directly (see app/sandbox.py's
    # array_to_qimage, "GOVERNING PRINCIPLE"), so there is nothing new to
    # unit-test here beyond colour_escape_time's own existing coverage above;
    # see app/test_sandbox.py for the integration-level check that a
    # rational map's julia render actually goes through it (palette-
    # sensitive, label-blind).

    # ---- colour_scalar_field: every pixel coloured, cyclic banding ------------------
    print("\ncolour_scalar_field:")
    field = np.array([1e-9, 1.0, np.e ** 12, np.e ** 24])
    fimg = colour_scalar_field(field, band_width=1.0, period_bands=12.0)
    check(fimg.shape == (4, 3) and fimg.dtype == np.uint8, "output shape/dtype")
    check(np.array_equal(fimg[1], fimg[3]),
          "values one full 12-band period apart (e^0 vs e^24, band_width=1) land at the same "
          "colour -- the cyclic equipotential-band wrap")

    try:
        colour_scalar_field(field, palette="not-a-real-palette")
        check(False, "an unknown palette name raises")
    except ValueError:
        check(True, "colour_scalar_field validates its palette argument too")

    # ---- colour_scalar_field: contour lines ------------------------------------------
    print("\ncolour_scalar_field (contour lines):")
    # A row that crosses several equipotential bands (e^0, e^1, ..., e^5,
    # band_width=1) so there's a real boundary to detect, plus a flat
    # region (all e^0) so there's also a real NO-boundary case to check.
    varying = np.array([[np.e ** 0, np.e ** 1, np.e ** 2, np.e ** 3, np.e ** 4, np.e ** 5]])
    flat = np.full((1, 6), np.e ** 0)

    without_contour = colour_scalar_field(varying, band_width=1.0, period_bands=12.0)
    with_contour = colour_scalar_field(varying, band_width=1.0, period_bands=12.0, contour=True)
    check(not np.array_equal(without_contour, with_contour),
          "contour=True changes the output when the field actually crosses band boundaries")
    check(tuple(with_contour[0, 1]) == (0, 0, 0),
          "a pixel on a real band boundary is drawn in the contour colour")

    flat_with_contour = colour_scalar_field(flat, band_width=1.0, period_bands=12.0, contour=True)
    flat_without = colour_scalar_field(flat, band_width=1.0, period_bands=12.0)
    check(np.array_equal(flat_with_contour, flat_without),
          "a perfectly flat field (no band boundaries anywhere) is unchanged by contour=True")

    custom_colour = colour_scalar_field(varying, band_width=1.0, period_bands=12.0, contour=True,
                                        contour_rgb=(255, 0, 0))
    check(tuple(custom_colour[0, 1]) == (255, 0, 0),
          "contour_rgb controls the actual drawn colour, not just an on/off flag")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
