"""app/family_views.py -- each family's stored PARAMETER-PLANE framing.

A family is characterised by its parameter plane, so each family carries a
framing (center, half-width) for it: loading the family applies it to the
parameter pane, and its Library thumbnail is rendered there (see
app.sandbox.render_family_thumbnail).

- Presets: PRESET_PARAMETER_VIEWS below.
- Saved families: the parameter pane's viewport at the moment of
  "Save current as...", stored app-side in a JSON sidecar beside the
  family's preview PNG (app.settings.family_meta_path_for) -- FamilyLibrary's
  serialized format cannot carry extra fields without changing cdx.
- Anything without a stored view -- including every entry of a library
  file written before this existed -- falls back to
  app.session.DEFAULT_PARAMETER_VIEW_CENTER/SCALE. Old library files load
  unchanged: nothing here is read from library.txt.

The same sidecar records the parameter strategy a family was saved with,
if it was not the default (stored_parameter_strategy), and the key a saved family's thumbnail PNG was
rendered under, so a palette or strategy change re-renders it (see
app.sandbox.FamilyThumbnailer).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.session import (DEFAULT_PARAMETER_VIEW_CENTER, DEFAULT_PARAMETER_VIEW_SCALE,
                         PRESET_FAMILY_NAMES)
from app.settings import family_meta_path_for


def _builder_views() -> dict[str, tuple[complex, float]]:
    from app.family_builder import preset_parameter_views
    return preset_parameter_views()


PRESET_PARAMETER_VIEWS: dict[str, tuple[complex, float]] = {
    "mandelbrot": (complex(-0.75, 0.0), 1.5),
    "multibrot3": (complex(0.0, 0.0), 1.35),
    "multibrot5": (complex(0.0, 0.0), 1.25),
    "mcmullen2": (complex(0.0, 0.0), 0.3),
    "mcmullen3": (complex(0.0, 0.0), 0.3),
}
# The family builder's presets that have a parameter (app.family_builder.PRESETS).
PRESET_PARAMETER_VIEWS.update(_builder_views())

DEFAULT_VIEW: tuple[complex, float] = (DEFAULT_PARAMETER_VIEW_CENTER, DEFAULT_PARAMETER_VIEW_SCALE)


def _read_meta(name: str) -> dict:
    """The sidecar's contents, or {} -- never raises (missing, unreadable or
    malformed files all read as "nothing stored")."""
    try:
        data = json.loads(family_meta_path_for(name).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_meta(name: str, data: dict) -> None:
    path = family_meta_path_for(name)
    path.write_text(json.dumps(data, indent=1, sort_keys=True))


def stored_parameter_view(name: str) -> tuple[complex, float] | None:
    """The family's own framing (preset table, else its sidecar), or None."""
    if name in PRESET_PARAMETER_VIEWS:
        return PRESET_PARAMETER_VIEWS[name]
    view = _read_meta(name).get("parameter_view")
    try:
        re_, im_ = view["center"]
        scale = float(view["scale"])
        center = complex(float(re_), float(im_))
    except (TypeError, KeyError, ValueError):
        return None
    if not (scale > 0.0) or scale != scale or center != center:
        return None
    return center, scale


def parameter_view_for(name: str) -> tuple[complex, float]:
    """The framing a parameter-plane pane gets on loading `name` (and on
    Reset View): its stored view, else DEFAULT_PARAMETER_VIEW_CENTER/SCALE."""
    return stored_parameter_view(name) or DEFAULT_VIEW


def save_parameter_view(name: str, center: complex, scale: float) -> None:
    """Records a saved family's parameter-plane framing (presets are fixed)."""
    if name in PRESET_PARAMETER_VIEWS:
        return
    data = _read_meta(name)
    data["parameter_view"] = {"center": [center.real, center.imag], "scale": float(scale)}
    data.pop("thumbnail_key", None)   # the framing is part of what the thumbnail shows
    _write_meta(name, data)


def stored_parameter_strategy(name: str) -> tuple[str, int] | None:
    """The (parameter_strategy, parameter_critical_index) saved WITH the
    family, or None -- loading a family without one resets the strategy to
    the default (a non-default strategy never leaks from one family into
    the next). Presets never carry one."""
    if name in PRESET_FAMILY_NAMES:
        return None
    entry = _read_meta(name).get("parameter_strategy")
    try:
        strategy, index = str(entry["strategy"]), int(entry["critical_index"])
    except (TypeError, KeyError, ValueError):
        return None
    from app.settings import PARAMETER_STRATEGY_CHOICES
    if strategy not in PARAMETER_STRATEGY_CHOICES or not 0 <= index <= 63:
        return None
    return strategy, index


def save_parameter_strategy(name: str, strategy: str | None, critical_index: int = 0) -> None:
    """Records the strategy a family was saved with; None (or the default
    strategy) removes it. Presets are fixed."""
    if name in PRESET_FAMILY_NAMES:
        return
    from app.settings import DEFAULT_PARAMETER_STRATEGY
    data = _read_meta(name)
    if strategy is None or strategy == DEFAULT_PARAMETER_STRATEGY:
        if data.pop("parameter_strategy", None) is None:
            return
    else:
        data["parameter_strategy"] = {"strategy": strategy, "critical_index": int(critical_index)}
    _write_meta(name, data)


def thumbnail_key(name: str) -> str | None:
    return _read_meta(name).get("thumbnail_key")


def set_thumbnail_key(name: str, key: str) -> None:
    data = _read_meta(name)
    data["thumbnail_key"] = key
    _write_meta(name, data)


def rename_family_meta(old: str, new: str) -> None:
    """Moves a family's sidecar with it on rename (the PNG is re-rendered)."""
    src, dst = family_meta_path_for(old), family_meta_path_for(new)
    if src.exists():
        dst.write_text(src.read_text())
        src.unlink()


def meta_paths(names) -> set[Path]:
    return {family_meta_path_for(n) for n in names}
