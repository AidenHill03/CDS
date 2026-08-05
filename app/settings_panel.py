"""app/settings_panel.py -- the Settings tab.

A QWidget, not folded into app/sandbox.py's SandboxWindow, so the render/
cache configuration UI stays a separate concern from window/image-view
orchestration -- matching app/render_cache.py and app/settings.py already
being their own files.

APPLY ON DEMAND. Every field here edits a WIDGET value only; nothing is
read back into a live Settings object, let alone triggers a re-render,
until the Apply button is clicked (see _apply). A resolution slider that
re-rendered per tick at 2000px would feel like a freeze -- see the
P5a-final commit message for just how real "feels like a freeze" is on
this project's own development machine.

Structured for growth: every plain numeric field in app.settings.FIELD_SPECS
gets a widget generically (see _build_group) -- adding a new setting means
adding one FieldSpec entry in app/settings.py and one line in FIELD_GROUPS
below, not touching _build_ui's layout code.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from app.settings import FIELD_SPECS, Settings, slow_render_warning, validate_field

# Which group box each field's row appears in, and in what order -- purely
# a UI grouping concern, so it lives here rather than in app/settings.py
# (which knows nothing about widgets or layout).
FIELD_GROUPS = {
    "Rendering": ["resolution", "max_iter", "escape_radius", "tol", "threads"],
    "Cache": ["cache_budget_bytes"],
}

_BYTES_PER_MB = 1024 * 1024

# How often the live cache readout (hits/misses/bytes used) refreshes while
# this tab is visible. Not tied to render completion signals -- polling is
# simpler and plenty responsive for a stat display nobody is timing.
CACHE_READOUT_REFRESH_MS = 500

_ERROR_STYLE = "border: 1px solid #cc4444;"


def _spinbox_for(name: str) -> QSpinBox | QDoubleSpinBox:
    """Builds (but does not populate) the editing widget for one
    FIELD_SPECS entry. cache_budget_bytes is the one field shown in
    different units than it's stored in (MB, not raw bytes) -- see
    _mb_widget_range/_to_mb/_from_mb.
    """
    spec = FIELD_SPECS[name]
    if name == "cache_budget_bytes":
        box = QSpinBox()
        box.setRange(*_mb_widget_range(spec))
        box.setSuffix(" MB")
        return box
    if spec.kind is int:
        box = QSpinBox()
        box.setRange(int(spec.minimum), int(spec.maximum))
        return box
    box = QDoubleSpinBox()
    # tol needs enough decimals to show a meaningful chordal tolerance like
    # 1e-6 (range 0..2); escape_radius (range 0..1000) does not, and 9
    # decimals of trailing zeros there is just clutter.
    box.setDecimals(9 if spec.maximum <= 10 else 3)
    # QDoubleSpinBox's own range is inclusive at both ends; an exclusive
    # minimum (escape_radius/tol must be > 0, not >= 0) is enforced by
    # validate_field on Apply, not by the widget itself -- the widget's
    # floor stays at the spec's stated minimum so a user can still type
    # something close to it and get a clear rejection message, rather than
    # the widget silently refusing the keystroke.
    box.setRange(spec.minimum, spec.maximum)
    box.setSingleStep(_sensible_step(spec))
    return box


def _sensible_step(spec) -> float:
    # A step of 1.0 is silly for tol (range 0..2, meaningful values like
    # 1e-6) and fine for escape_radius (range 0..1000) -- scale the default
    # step to the field's own range instead of hardcoding one number.
    span = spec.maximum - spec.minimum
    if span <= 0:
        return 1.0
    if span <= 10:
        return max(span / 1000.0, 1e-6)
    return max(span / 100.0, 0.01)


def _mb_widget_range(spec) -> tuple[int, int]:
    return (int(spec.minimum // _BYTES_PER_MB), int(spec.maximum // _BYTES_PER_MB))


def _to_widget_value(name: str, value):
    if name == "cache_budget_bytes":
        return round(value / _BYTES_PER_MB)
    return value


def _from_widget_value(name: str, raw):
    if name == "cache_budget_bytes":
        return raw * _BYTES_PER_MB
    return raw


class SettingsPanel(QWidget):
    """Render/cache settings, editable and applied on demand. Reads its
    initial values from `session.settings` and, on a successful Apply,
    calls back through `on_apply` (rather than mutating `session` directly)
    so SandboxWindow decides what applying a Settings actually entails --
    updating the session, persisting to disk, and triggering an immediate
    (not debounced) progressive re-render are all its concern, not this
    widget's.
    """

    def __init__(self, session, on_apply, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self._on_apply = on_apply
        self._last_good: Settings = session.settings
        self._widgets: dict[str, QSpinBox | QDoubleSpinBox] = {}

        self._build_ui()
        self._load_into_widgets(self._last_good)
        self._update_slow_render_hint()

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(CACHE_READOUT_REFRESH_MS)
        self._stats_timer.timeout.connect(self._refresh_cache_readout)
        self._stats_timer.start()
        self._refresh_cache_readout()

    # ---- construction ----------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        for group_title, field_names in FIELD_GROUPS.items():
            layout.addWidget(self._build_group(group_title, field_names))

        cache_row = QHBoxLayout()
        self._cache_readout_label = QLabel()
        cache_row.addWidget(self._cache_readout_label)
        cache_row.addStretch(1)
        clear_button = QPushButton("Clear Cache")
        clear_button.clicked.connect(self._clear_cache)
        cache_row.addWidget(clear_button)
        layout.addLayout(cache_row)

        self._slow_render_label = QLabel()
        self._slow_render_label.setWordWrap(True)
        layout.addWidget(self._slow_render_label)

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #cc4444;")
        layout.addWidget(self._error_label)

        apply_row = QHBoxLayout()
        apply_row.addStretch(1)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._apply)
        apply_row.addWidget(apply_button)
        layout.addLayout(apply_row)

        layout.addStretch(1)

    def _build_group(self, title: str, field_names: list[str]) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for name in field_names:
            spec = FIELD_SPECS[name]
            widget = _spinbox_for(name)
            widget.valueChanged.connect(self._update_slow_render_hint)
            self._widgets[name] = widget
            form.addRow(spec.label + ":", widget)
        return box

    # ---- widget <-> Settings -----------------------------------------------------------
    def _load_into_widgets(self, settings: Settings) -> None:
        for name, widget in self._widgets.items():
            widget.blockSignals(True)
            widget.setValue(_to_widget_value(name, getattr(settings, name)))
            widget.blockSignals(False)

    def _draft_settings_from_widgets(self) -> dict:
        return {name: _from_widget_value(name, widget.value())
               for name, widget in self._widgets.items()}

    # ---- Apply: validate every field, revert invalid ones, apply the rest together ----
    def _apply(self) -> None:
        draft = self._draft_settings_from_widgets()
        errors: list[str] = []
        parsed: dict = {}
        for name, raw_value in draft.items():
            ok, value, error = validate_field(name, raw_value)
            if ok:
                parsed[name] = value
            else:
                errors.append(error)
                parsed[name] = getattr(self._last_good, name)   # revert -- see class docstring

        if errors:
            self._error_label.setText("  •  ".join(errors))
            self._highlight_invalid_fields(draft)
            # Reload every field (not just the invalid ones) from the
            # about-to-be-restored last-good values, so a field that WAS
            # valid but never got applied (because a SIBLING field failed)
            # doesn't end up visually different from what's about to
            # actually take effect -- Apply is all-or-nothing per click.
            self._load_into_widgets(self._last_good)
            return

        self._error_label.setText("")
        self._clear_invalid_highlights()
        new_settings = Settings(**parsed)
        self._last_good = new_settings
        self._on_apply(new_settings)

    def _highlight_invalid_fields(self, draft: dict) -> None:
        for name, widget in self._widgets.items():
            ok, _value, _error = validate_field(name, draft[name])
            widget.setStyleSheet("" if ok else _ERROR_STYLE)

    def _clear_invalid_highlights(self) -> None:
        for widget in self._widgets.values():
            widget.setStyleSheet("")

    # ---- cache readout / clear ---------------------------------------------------------
    def _refresh_cache_readout(self) -> None:
        stats = self.session.cache.stats
        used_mb = stats.current_bytes / _BYTES_PER_MB
        budget_mb = stats.budget_bytes / _BYTES_PER_MB
        self._cache_readout_label.setText(
            f"{used_mb:.1f} / {budget_mb:.1f} MB used  ·  {stats.entry_count} entries  ·  "
            f"{stats.hits} hits  ·  {stats.misses} misses")

    def _clear_cache(self) -> None:
        self.session.cache.clear()
        self._refresh_cache_readout()

    # ---- slow-render hint --------------------------------------------------------------
    def _update_slow_render_hint(self) -> None:
        # A DRAFT estimate from whatever is currently typed, not from the
        # last-applied Settings -- the whole point is warning BEFORE
        # Apply, not after. Uses a throwaway Settings built straight from
        # the widgets; invalid values here just feed the estimate (a
        # ballpark, not a promise) rather than being validated -- Apply is
        # what actually rejects bad input.
        draft = self._draft_settings_from_widgets()
        try:
            probe = Settings(**draft)
        except (TypeError, ValueError):
            self._slow_render_label.setText("")
            return
        warning = slow_render_warning(probe)
        self._slow_render_label.setText(warning or "")
