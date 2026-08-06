"""test_settings.py -- property-based checks for app/settings.

Run with:

    PYTHONPATH=cdx/build python -m app.test_settings

(from the repository root -- no cdx import is actually needed by this
module, but this keeps one invocation pattern for the whole app/ suite).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.settings import (
    FIELD_SPECS,
    Settings,
    estimated_full_render_seconds,
    load_settings,
    save_settings,
    slow_render_warning,
    validate_field,
)

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    print("=== app.settings tests ===")

    # ---- defaults sit inside their own valid range ---------------------------------
    print("\ndefaults are self-consistent:")
    defaults = Settings()
    for name, spec in FIELD_SPECS.items():
        ok, _value, error = validate_field(name, getattr(defaults, name))
        check(ok, f"Settings()'s default for {name!r} passes its own FieldSpec ({error})")

    # ---- validate_field: strict, per-field --------------------------------------
    print("\nvalidate_field:")
    ok, value, _err = validate_field("resolution", 500)
    check(ok and value == 500, "an in-range int passes and is returned unchanged")

    ok, value, err = validate_field("resolution", 50)
    check(not ok and value is None and err, "below the minimum fails with a non-empty message")

    ok, _value, _err = validate_field("resolution", 4000)
    check(ok, "the maximum itself is inclusive and valid")

    ok, _value, _err = validate_field("resolution", 4001)
    check(not ok, "just above the maximum fails")

    ok, _value, _err = validate_field("escape_radius", 0.0)
    check(not ok, "escape_radius=0 fails -- exclusive minimum, not inclusive")

    ok, _value, _err = validate_field("escape_radius", 1e-12)
    check(ok, "a tiny but positive escape_radius passes")

    ok, _value, _err = validate_field("tol", 2.0)
    check(ok, "tol=2.0 (the chordal metric's own diameter) is valid, inclusive maximum")

    ok, _value, _err = validate_field("tol", 2.5)
    check(not ok, "tol above the chordal metric's diameter is rejected")

    ok, value, err = validate_field("max_iter", "not a number")
    check(not ok and value is None and "must be a" in err,
          "a non-numeric string fails with a type-explaining message, not a crash")

    ok, value, _err = validate_field("threads", "8")
    check(ok and value == 8, "a numeric STRING coerces to the right type (UI widgets may hand one)")

    ok, _value, _err = validate_field("escape_radius", float("inf"))
    check(not ok, "infinity fails -- not a usable escape radius")

    ok, _value, _err = validate_field("escape_radius", float("nan"))
    check(not ok, "NaN fails")

    # ---- validate_field: choices-based (enum) fields -------------------------------
    print("\nvalidate_field (choices):")
    ok, value, _err = validate_field("colour_palette", "viridis")
    check(ok and value == "viridis", "a listed palette name passes and is returned unchanged")

    ok, value, err = validate_field("colour_palette", "not-a-real-palette")
    check(not ok and value is None and "must be one of" in err,
          "an unlisted palette name fails with a message naming the valid choices")

    ok, value, _err = validate_field("colour_scaling", "log1p")
    check(ok and value == "log1p", "a listed scaling mode passes")

    ok, _value, _err = validate_field("colour_scaling", "histogram")
    check(ok, "the other listed scaling mode also passes")

    ok, _value, _err = validate_field("colour_period", 0.0)
    check(ok, "colour_period=0.0 (no cyclic banding) is valid -- NOT an exclusive minimum, "
              "unlike escape_radius/tol")

    ok, _value, _err = validate_field("colour_period", -1.0)
    check(not ok, "a negative colour_period is rejected")

    # ---- validate_field: bool fields -------------------------------------------------
    print("\nvalidate_field (bool):")
    ok, value, _err = validate_field("greens_contour", True)
    check(ok and value is True, "an actual bool True passes and is returned unchanged")
    ok, value, _err = validate_field("greens_contour", False)
    check(ok and value is False, "an actual bool False passes and is returned unchanged")

    ok, value, _err = validate_field("greens_contour", 1)
    check(ok and value is True, "int 1 coerces to True")
    ok, value, _err = validate_field("greens_contour", 0)
    check(ok and value is False, "int 0 coerces to False")

    ok, value, err = validate_field("greens_contour", "false")
    check(not ok and value is None,
          "the STRING 'false' is rejected, not silently truthy-coerced to True -- "
          "bool('false') would be True in plain Python, which is exactly the footgun this "
          "field deliberately avoids")
    ok, _value, _err = validate_field("greens_contour", "true")
    check(not ok, "the string 'true' is also rejected -- only real bools/0/1 are accepted")
    ok, _value, _err = validate_field("greens_contour", 2)
    check(not ok, "an int other than 0/1 is rejected")
    ok, _value, _err = validate_field("greens_contour", None)
    check(not ok, "None is rejected")

    ok, value, _err = validate_field("greens_band_width", 0.0)
    check(not ok, "greens_band_width=0 is rejected -- exclusive minimum (would divide by zero "
                  "downstream in scale_scalar_field)")
    ok, value, _err = validate_field("greens_band_width", 2.5)
    check(ok and value == 2.5, "a positive greens_band_width passes")

    # ---- Settings.sanitized(): tolerant, per-field fallback to defaults -----------
    print("\nSettings.sanitized():")
    broken = Settings(resolution=99999, max_iter=-5, escape_radius=2.0, tol=1e-6,
                      threads=0, cache_budget_bytes=100, colour_palette="bogus",
                      colour_period=-5.0, greens_band_width=0.0)
    fixed = broken.sanitized()
    check(fixed.resolution == FIELD_SPECS["resolution"].default,
          "an out-of-range resolution falls back to the field's own default")
    check(fixed.max_iter == FIELD_SPECS["max_iter"].default,
          "a negative max_iter falls back to the field's own default")
    check(fixed.escape_radius == 2.0 and fixed.tol == 1e-6,
          "fields that WERE already valid are left exactly as they were, not also reset")
    check(fixed.cache_budget_bytes == 100,
          "a valid (if small) cache_budget_bytes is kept, not silently bumped up")
    check(fixed.colour_palette == FIELD_SPECS["colour_palette"].default,
          "an unlisted colour_palette falls back to its own default, same as a numeric field")
    check(fixed.colour_period == FIELD_SPECS["colour_period"].default,
          "a negative colour_period falls back to its own default")
    check(fixed.greens_band_width == FIELD_SPECS["greens_band_width"].default,
          "greens_band_width=0 (excluded) falls back to its own default")
    check(fixed.greens_contour == FIELD_SPECS["greens_contour"].default,
          "a bool field that WAS already valid (default False) survives sanitized() unchanged")

    # ---- persistence: round-trip, missing file, malformed file --------------------
    print("\npersistence:")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "settings.json"

        loaded_missing = load_settings(path)
        check(loaded_missing == Settings(),
              "loading a file that does not exist yet returns plain defaults")

        custom = Settings(resolution=800, max_iter=500, escape_radius=4.0, tol=1e-4,
                          threads=4, cache_budget_bytes=64 * 1024 * 1024,
                          colour_palette="magma", colour_scaling="histogram", colour_period=20.0,
                          greens_band_width=2.0, greens_period_bands=8.0, greens_contour=True)
        save_settings(custom, path)
        roundtripped = load_settings(path)
        check(roundtripped == custom, "save_settings/load_settings round-trips every field exactly")

        # Forward compatibility: an unknown key (as if written by a NEWER
        # version of this app) must not break loading.
        raw = json.loads(path.read_text())
        raw["some_future_setting"] = "unrecognized"
        path.write_text(json.dumps(raw))
        check(load_settings(path) == custom,
              "an unknown key in the file is ignored, not a load failure")

        # Backward compatibility: a file missing a field (as if written by
        # an OLDER version, before that field existed) falls back to that
        # field's default rather than failing to load at all.
        del raw["threads"]
        raw.pop("some_future_setting")
        path.write_text(json.dumps(raw))
        partial = load_settings(path)
        check(partial.threads == FIELD_SPECS["threads"].default,
              "a field missing from an older config file falls back to its default")
        check(partial.resolution == 800 and partial.max_iter == 500,
              "fields that WERE present in the older file are still honored")

        # A value out of range in the persisted file (hand-edited, or from
        # a build with wider bounds) degrades to that field's default
        # rather than refusing to start.
        raw["threads"] = -7
        path.write_text(json.dumps(raw))
        check(load_settings(path).threads == FIELD_SPECS["threads"].default,
              "an out-of-range value in the file falls back to that field's default")

        # Malformed JSON and a non-object JSON value both degrade to
        # plain defaults rather than raising.
        path.write_text("{not valid json")
        check(load_settings(path) == Settings(), "malformed JSON falls back to plain defaults")

        path.write_text(json.dumps([1, 2, 3]))
        check(load_settings(path) == Settings(),
              "a JSON value that isn't an object falls back to plain defaults")

    # ---- slow-render warning ----------------------------------------------------
    print("\nslow-render warning:")
    fast = Settings(resolution=200, max_iter=50)
    check(slow_render_warning(fast) is None,
          "a small resolution/iteration combination gets no warning")

    slow = Settings(resolution=4000, max_iter=100_000)
    warning = slow_render_warning(slow)
    check(warning is not None and "⚠" in warning,
          "a large resolution/iteration combination gets a one-line warning")
    check(warning is not None and "\n" not in warning,
          "the warning is genuinely one line, not a multi-line block (it's a status hint, not a modal)")

    check(estimated_full_render_seconds(slow) > estimated_full_render_seconds(fast),
          "the estimate actually grows with resolution/max_iter, not a constant")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
