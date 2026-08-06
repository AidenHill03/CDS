"""app/settings.py -- persisted, user-editable render/cache settings.

The single source of truth for every app-level default that used to live
scattered in app/session.py (DEFAULT_RESOLUTION, DEFAULT_CACHE_BUDGET_BYTES)
-- session.py now imports them from here instead. Adding a new setting
means adding one FieldSpec entry and one Settings field; nothing else needs
touching (see app/sandbox.py's SettingsPanel, built generically off
FIELD_SPECS for exactly this reason).

Deliberately NOT a Session method or attribute of Session itself: Settings
is about the RENDER PIPELINE'S configuration (resolution, iteration/escape/
tolerance parameters, thread count, cache budget), independent of which map
or viewport a Session happens to be looking at right now, and it persists
across sessions/restarts (see load_settings/save_settings) while a
Session's map/param/viewport do not.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

# Pure-numpy, no Qt/cdx dependency (see app/colour.py's own docstring) --
# importing it here doesn't compromise this module's own "must stay
# importable without the cdx extension" property.
from app.colour import PALETTE_NAMES, SCALING_MODES

# ---- defaults ------------------------------------------------------------------
# Moved here from app/session.py. Measured three times, each superseding the
# last as the underlying engine cost changed:
#   1. P5a-final's first pass measured ~1100, but against the WRONG map --
#      the hardcoded Family.Quadratic fast path, not what the app actually
#      renders through (Map.custom(self.map, ...), even for the built-in
#      "mandelbrot" preset).
#   2. Re-measured against the real Map.custom path, P5a-final landed on
#      120 -- correct for the engine AS IT STOOD THEN, but that path turned
#      out to be paying two avoidable costs: a full per-pixel critical-point
#      root-find in parameter-plane mode, and per-iteration std::complex
#      arithmetic with no compiled/hand-rolled fast path at all.
#   3. P5a.1 fixed both (see cdx/include/cdx/rational.hpp's CompiledMap and
#      cdx::recognize_family, and the P5a.1 commit's own benchmark numbers).
#      A built-in-shaped Custom map like "mandelbrot" is now recognized and
#      dispatches to the SAME native formula a built-in Family::Quadratic
#      Map would use -- this measurement is what THAT is worth: the largest
#      resolution whose full render (already at the 1.3x overscan factor
#      every full render uses) stayed under ~0.4s (median of 9 runs,
#      parameter mode -- the startup default) on the development machine,
#      post-fix. It landed back at 800 -- coincidentally cdx.Viewport()'s
#      own incidental C++-side default, now actually earned rather than
#      just inherited.
DEFAULT_RESOLUTION = 800
# The rest match cdx.RenderSettings()'s own C++-side defaults (see
# cdx/include/cdx/renderer.hpp) -- duplicated here rather than read from a
# live cdx.RenderSettings() instance so this module has no dependency on the
# cdx extension being importable (settings persistence/validation should
# work even in a context that hasn't loaded it, e.g. a future non-GUI tool).
DEFAULT_MAX_ITER = 200
DEFAULT_ESCAPE_RADIUS = 2.0
DEFAULT_TOL = 1e-6
DEFAULT_THREADS = 0
# Comfortably many overscanned full renders at DEFAULT_RESOLUTION (each
# roughly resolution^2 * 8 bytes, a float64 array) on an otherwise-modest
# desktop, without needing to think about it.
DEFAULT_CACHE_BUDGET_BYTES = 256 * 1024 * 1024

# "classic" (the familiar blue/white/orange Mandelbrot look), not viridis --
# P5c's own spec frames viridis/magma as being for "when the image is data,
# not decoration," which is a choice a user makes deliberately, not the
# right first impression on a sandbox meant to be explored. log1p is the
# explicitly-specified default scaling (see app/colour.py's own docstring
# for why: a linear map puts 90% of a typical escaped image in the first 5%
# of the palette). Period 0 means no cyclic banding -- the plain smooth
# escape-time look; banding is an adjustable enhancement, not a forced
# default.
DEFAULT_COLOUR_PALETTE = "classic"
DEFAULT_COLOUR_SCALING = "log1p"
DEFAULT_COLOUR_PERIOD = 0.0

# Green's-function display (both G_f(z) on the dynamical plane and G_M(c)
# on the parameter plane -- see app.colour.colour_scalar_field, which both
# share). band_width=1.0 and 12 cyclic bands are a reasonable, legible
# starting point (roughly matching the classic equipotential-band look
# fractal software defaults to); NOT tied to colour_period above -- that's
# an escape-time-specific concept (wraps a raw iteration VALUE), this
# wraps an already-log-scaled BAND INDEX, a different unit entirely.
# Contour lines default OFF: the spec calls them "optional."
DEFAULT_GREENS_BAND_WIDTH = 1.0
DEFAULT_GREENS_PERIOD_BANDS = 12.0
DEFAULT_GREENS_CONTOUR = False


@dataclass(frozen=True)
class FieldSpec:
    """Describes one setting for both validation and (in
    app/sandbox.py's SettingsPanel) UI generation. `minimum`/`maximum` are
    inclusive unless `exclusive_minimum` is set (needed for fields like
    escape_radius/tol where exactly 0 is meaningless, not just small).

    `choices`, when set, makes this an ENUM field instead of a numeric
    range: `kind` must be `str`, `minimum`/`maximum`/`exclusive_minimum`
    are ignored, and validate_field checks membership in `choices` rather
    than a numeric bound. SettingsPanel builds a QComboBox for these
    instead of a QSpinBox/QDoubleSpinBox (see _widget_for).

    `kind is bool` is its own third kind (independent of `choices`):
    `minimum`/`maximum`/`exclusive_minimum` are ignored the same way, and
    validate_field accepts True/False or anything Python's own bool()
    treats as boolean-shaped input, coerced via bool(). SettingsPanel
    builds a QCheckBox for these (see _widget_for).
    """
    kind: type            # int, float, str (with choices), or bool
    minimum: float
    maximum: float
    default: float | str
    label: str
    exclusive_minimum: bool = False
    choices: tuple[str, ...] | None = None


# 200-4000, as originally specified. An earlier engine-cost measurement put
# DEFAULT_RESOLUTION (see above) at 120, briefly below this floor -- widened
# to 100 to accommodate it rather than clamping the default up and
# contradicting "default = whatever was measured". P5a.1's engine fix moved
# the honest default back to 800, comfortably inside 200-4000 again, so the
# floor is restored to its originally-specified value.
RESOLUTION_RANGE = (200, 4000)

FIELD_SPECS: dict[str, FieldSpec] = {
    "resolution": FieldSpec(int, RESOLUTION_RANGE[0], RESOLUTION_RANGE[1],
                            DEFAULT_RESOLUTION, "Resolution (render pixels per side)"),
    "max_iter": FieldSpec(int, 1, 100_000, DEFAULT_MAX_ITER, "Max iterations"),
    "escape_radius": FieldSpec(float, 0.0, 1000.0, DEFAULT_ESCAPE_RADIUS, "Escape radius",
                               exclusive_minimum=True),
    # Chordal distance is bounded by the sphere's diameter in this
    # normalization (d(z,w) = 2|z-w|/sqrt(...), maximal at antipodal
    # points) -- see cdx::chordal_distance -- so a tolerance above 2 can
    # never mean anything different from exactly 2 (everything matches).
    "tol": FieldSpec(float, 0.0, 2.0, DEFAULT_TOL, "Basin tolerance (chordal)",
                     exclusive_minimum=True),
    "threads": FieldSpec(int, 0, 256, DEFAULT_THREADS, "Thread count (0 = hardware concurrency)"),
    "cache_budget_bytes": FieldSpec(int, 0, 16 * 1024 ** 3, DEFAULT_CACHE_BUDGET_BYTES,
                                    "Cache byte budget"),
    # minimum/maximum/exclusive_minimum are unused for choices-based
    # fields (see FieldSpec's own docstring) -- 0/0 rather than omitted,
    # since FieldSpec has no default for either.
    "colour_palette": FieldSpec(str, 0, 0, DEFAULT_COLOUR_PALETTE, "Colour palette",
                                choices=PALETTE_NAMES),
    "colour_scaling": FieldSpec(str, 0, 0, DEFAULT_COLOUR_SCALING, "Colour scaling",
                                choices=SCALING_MODES),
    # 0 (no cyclic banding) is a legitimate, meaningful value here, not
    # excluded the way escape_radius/tol exclude 0 -- so NOT
    # exclusive_minimum. Ceiling is arbitrary headroom past any max_iter a
    # user would realistically set (see FIELD_SPECS["max_iter"]'s own
    # ceiling of 100_000); a period longer than max_iter just never wraps.
    "colour_period": FieldSpec(float, 0.0, 100_000.0, DEFAULT_COLOUR_PERIOD,
                               "Colour cycle period (0 = off)"),
    # log(value)/band_width must stay meaningfully sized -- 0 would make
    # every value fall in the same band (division by zero, in fact), so
    # this one IS exclusive_minimum, unlike colour_period above.
    "greens_band_width": FieldSpec(float, 0.0, 1000.0, DEFAULT_GREENS_BAND_WIDTH,
                                   "Green's function band width", exclusive_minimum=True),
    "greens_period_bands": FieldSpec(float, 1.0, 1000.0, DEFAULT_GREENS_PERIOD_BANDS,
                                     "Green's function bands per cycle"),
    # minimum/maximum/exclusive_minimum are unused for a bool field (see
    # FieldSpec's own docstring) -- 0/0 rather than omitted, matching the
    # same convention the choices-based fields above already use.
    "greens_contour": FieldSpec(bool, 0, 0, DEFAULT_GREENS_CONTOUR,
                                "Green's function contour lines"),
}


@dataclass
class Settings:
    resolution: int = DEFAULT_RESOLUTION
    max_iter: int = DEFAULT_MAX_ITER
    escape_radius: float = DEFAULT_ESCAPE_RADIUS
    tol: float = DEFAULT_TOL
    threads: int = DEFAULT_THREADS
    cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES
    colour_palette: str = DEFAULT_COLOUR_PALETTE
    colour_scaling: str = DEFAULT_COLOUR_SCALING
    colour_period: float = DEFAULT_COLOUR_PERIOD
    greens_band_width: float = DEFAULT_GREENS_BAND_WIDTH
    greens_period_bands: float = DEFAULT_GREENS_PERIOD_BANDS
    greens_contour: bool = DEFAULT_GREENS_CONTOUR

    def sanitized(self) -> Settings:
        """A copy with every field individually validated, out-of-range or
        wrongly-typed ones silently replaced by that field's default. Used
        when loading persisted settings -- a corrupted or hand-edited
        config file should degrade gracefully field-by-field, not refuse to
        start the app or poison every OTHER, perfectly fine setting.
        """
        fixed = {}
        for name, spec in FIELD_SPECS.items():
            ok, value, _error = validate_field(name, getattr(self, name))
            fixed[name] = value if ok else spec.default
        return Settings(**fixed)


def validate_field(name: str, raw_value) -> tuple[bool, object, str]:
    """Strict validation for one field, used by the Settings panel's Apply
    handler: invalid input is reported, not coerced -- the caller reverts
    the widget to its last-good value rather than silently accepting a
    clamped one (see SettingsPanel._apply). Returns (ok, parsed_value,
    error_message); error_message is "" when ok is True.
    """
    spec = FIELD_SPECS[name]
    if spec.kind is bool:
        # Deliberately NOT bool(raw_value) -- Python's bool() truthy-casts
        # any non-empty string, so bool("false") is True, a real footgun
        # for a hand-edited settings.json. Only an actual bool (the normal
        # case: JSON true/false, or a QCheckBox's .isChecked()) or an int
        # 0/1 (JSON has no separate bool literal some hand-editors might
        # reach for) are accepted; anything else -- including strings --
        # is rejected outright rather than silently coerced.
        if isinstance(raw_value, bool):
            return True, raw_value, ""
        if isinstance(raw_value, int) and raw_value in (0, 1):
            return True, bool(raw_value), ""
        return False, None, f"{spec.label} must be true or false"
    try:
        value = spec.kind(raw_value)
    except (TypeError, ValueError):
        return False, None, f"{spec.label} must be a {spec.kind.__name__}"
    if spec.choices is not None:
        if value not in spec.choices:
            return False, None, f"{spec.label} must be one of {spec.choices}"
        return True, value, ""
    if spec.kind is float and (value != value or value in (float("inf"), float("-inf"))):
        return False, None, f"{spec.label} must be a finite number"
    if spec.exclusive_minimum and value <= spec.minimum:
        return False, None, f"{spec.label} must be greater than {spec.minimum}"
    if not spec.exclusive_minimum and value < spec.minimum:
        return False, None, f"{spec.label} must be at least {spec.minimum}"
    if value > spec.maximum:
        return False, None, f"{spec.label} must be at most {spec.maximum}"
    return True, value, ""


# ---- persistence -----------------------------------------------------------------
def config_dir() -> Path:
    """Where settings.json (and, in future, a default family-library file --
    see app.session's save_library_file/load_library_file, which today
    always take an explicit path with no default location of their own)
    live. Created on first use.
    """
    directory = Path.home() / ".complexdynamics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def settings_path() -> Path:
    return config_dir() / "settings.json"


def library_path() -> Path:
    """Where a user's saved family library lives -- see app.session's
    save_user_library/load_user_library, which (unlike save_library_file/
    load_library_file, always explicit-path exports) use this as their
    default location, the "beside settings.json" this module's own
    config_dir() docstring already anticipated.
    """
    return config_dir() / "library.txt"


def load_settings(path: Path | None = None) -> Settings:
    """Never raises: a missing file, unreadable file, malformed JSON, or a
    JSON value that is not an object all just fall back to Settings()'s
    defaults (or, for a partially-valid file, defaults for only the
    individual fields that do not parse -- see Settings.sanitized).
    """
    path = path or settings_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return Settings()
    if not isinstance(raw, dict):
        return Settings()
    # Unknown keys (e.g. from a NEWER version of this file) are ignored
    # rather than rejected; missing keys (from an OLDER one) fall back to
    # Settings()'s own defaults for that field -- this is what keeps adding
    # a setting later from being a breaking change to old config files.
    known = {name: raw[name] for name in FIELD_SPECS if name in raw}
    return Settings(**{**dataclasses.asdict(Settings()), **known}).sanitized()


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or settings_path()
    path.write_text(json.dumps(dataclasses.asdict(settings), indent=2) + "\n")


# ---- slow-render warning ----------------------------------------------------------
# Calibrated from the SAME measurement DEFAULT_RESOLUTION itself came from
# (see that constant's comment): res=800, overscanned to round(800*1.3) =
# 1040, max_iter=200, ~0.30s median through the actual Map.custom render
# path in "parameter" mode, for the "mandelbrot" preset -- which, since
# P5a.1's fast-path recognition (see cdx::recognize_family), now dispatches
# to the SAME native formula a built-in Family::Quadratic Map would use.
# 1040*1040*200 / 0.30 =~ 720M pixel-iterations/second at THAT speed --
# but Settings has no idea what map will actually be rendered, and a
# general custom map with even a couple of poles measured roughly 9x that
# cost in the P5a.1 acceptance benchmark (two required reciprocal
# divisions per pole -- see CompiledMap::step's own comment). A heuristic
# that only warns at the BEST case's speed would stay silent through
# exactly the maps most likely to actually feel slow, so this divides the
# measured recognized-path throughput by that same ~9x -- the same
# "over-warn rather than under-warn" choice the previous calibration made
# by picking the slower of two render modes, applied here to the slower of
# two MAP SHAPES instead.
MEASURED_PIXEL_ITERATIONS_PER_SECOND = 80_000_000
SLOW_RENDER_WARNING_SECONDS = 1.0
# Duplicated from app.sandbox.FULL_OVERSCAN_FACTOR rather than imported, so
# this module has no dependency on PySide6 being installed -- only the
# default value for callers (the Settings panel) that don't pass their own.
DEFAULT_OVERSCAN_FACTOR_ESTIMATE = 1.3


def estimated_full_render_seconds(settings: Settings,
                                  overscan_factor: float = DEFAULT_OVERSCAN_FACTOR_ESTIMATE) -> float:
    overscanned_side = settings.resolution * overscan_factor
    return (overscanned_side ** 2) * settings.max_iter / MEASURED_PIXEL_ITERATIONS_PER_SECOND


def slow_render_warning(settings: Settings,
                        overscan_factor: float = DEFAULT_OVERSCAN_FACTOR_ESTIMATE) -> str | None:
    """A one-line hint, not a modal -- these settings may make a full
    render slow enough to notice. None if the estimate stays comfortable.
    A rough heuristic (see MEASURED_PIXEL_ITERATIONS_PER_SECOND above), not
    a promise: actual time depends heavily on the map and how much of the
    view escapes quickly versus burns the full iteration budget.
    """
    estimated = estimated_full_render_seconds(settings, overscan_factor)
    if estimated < SLOW_RENDER_WARNING_SECONDS:
        return None
    return f"⚠ these settings may make a full render take roughly {estimated:.1f}s"
