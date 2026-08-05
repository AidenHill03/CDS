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

# ---- defaults ------------------------------------------------------------------
# Moved here from app/session.py -- see the P5a-final commit message for how
# this number was measured (the largest resolution whose full render, via
# the ACTUAL Map.custom code path the app always uses, stayed under ~0.4s on
# the development machine). Note this is below the Settings panel's own
# RESOLUTION_RANGE minimum of 200 -- see that constant's comment.
DEFAULT_RESOLUTION = 120
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


@dataclass(frozen=True)
class FieldSpec:
    """Describes one setting for both validation and (in
    app/sandbox.py's SettingsPanel) UI generation. `minimum`/`maximum` are
    inclusive unless `exclusive_minimum` is set (needed for fields like
    escape_radius/tol where exactly 0 is meaningless, not just small).
    """
    kind: type            # int or float
    minimum: float
    maximum: float
    default: float
    label: str
    exclusive_minimum: bool = False


# The widget's floor (100) sits below the 200 a resolution slider would
# naturally start from, specifically so DEFAULT_RESOLUTION (measured at 120,
# see above) is itself a valid, enterable value -- a default the range
# excludes would be a contradiction the first time the panel opens.
RESOLUTION_RANGE = (100, 4000)

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
}


@dataclass
class Settings:
    resolution: int = DEFAULT_RESOLUTION
    max_iter: int = DEFAULT_MAX_ITER
    escape_radius: float = DEFAULT_ESCAPE_RADIUS
    tol: float = DEFAULT_TOL
    threads: int = DEFAULT_THREADS
    cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES

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
    try:
        value = spec.kind(raw_value)
    except (TypeError, ValueError):
        return False, None, f"{spec.label} must be a {spec.kind.__name__}"
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
# (see the P5a-final commit message): res=120, overscanned to round(120*1.3)
# = 156, max_iter=200, ~0.33s median through the actual Map.custom render
# path in "parameter" mode -- the slower of the two render modes measured,
# and so the more conservative (over-warns rather than under-warns) anchor
# for a heuristic meant to catch "this will feel slow", not predict it
# precisely. 156*156*200 / 0.33 =~ 14.7M pixel-iterations/second.
MEASURED_PIXEL_ITERATIONS_PER_SECOND = 14_700_000
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
