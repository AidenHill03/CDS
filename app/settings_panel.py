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

import dataclasses

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from app import theme
from app.settings import FIELD_SPECS, Settings, slow_render_warning, validate_field
from app.widgets import FlowLayout

# Which group box each field's row appears in, and in what order -- purely
# a UI grouping concern, so it lives here rather than in app/settings.py
# (which knows nothing about widgets or layout).
FIELD_GROUPS = {
    "Rendering": ["resolution", "max_iter", "escape_radius", "tol", "threads"],
    "Coloring": ["color_palette", "color_scaling", "color_period"],
    "Green's function": ["greens_potential", "greens_band_width", "greens_period_bands",
                        "greens_contour"],
    "Parameter plane": ["parameter_strategy", "parameter_critical_index"],
    "Navigation": ["param_marker_step", "param_marker_rate"],
    "Cache": ["cache_budget_bytes"],
}

# A one-line note shown under a field's widget.
FIELD_NOTES = {
    "parameter_strategy": "Per-critical: dashed rays mark where critical-point labels "
                          "switch — a labelling convention, not dynamics.",
}

# The two dock tabs (UI redesign): which groups each shows.
RENDERING_GROUPS = {k: FIELD_GROUPS[k] for k in ("Rendering", "Parameter plane", "Navigation", "Cache")}
COLORING_GROUPS = {k: FIELD_GROUPS[k] for k in ("Coloring", "Green's function")}

# Live apply waits for a pause in editing (typing "800" is three edits),
# then applies once; the window renders immediately on apply.
SETTINGS_APPLY_DEBOUNCE_MS = 300
COLUMN_ROWS = 3

_BYTES_PER_MB = 1024 * 1024

# How often the live cache readout (hits/misses/bytes used) refreshes while
# this tab is visible. Not tied to render completion signals -- polling is
# simpler and plenty responsive for a stat display nobody is timing.
CACHE_READOUT_REFRESH_MS = 500



def _widget_for(name: str) -> QSpinBox | QDoubleSpinBox | QComboBox | QCheckBox:
    """Builds (but does not populate) the editing widget for one
    FIELD_SPECS entry. cache_budget_bytes is the one NUMERIC field shown
    in different units than it's stored in (MB, not raw bytes) -- see
    _mb_widget_range/_to_mb/_from_mb. A choices-based field (color_palette,
    color_scaling) gets a QComboBox instead of a spinbox entirely, and a
    bool field (greens_contour) gets a QCheckBox -- see FieldSpec's own
    docstring for both.
    """
    spec = FIELD_SPECS[name]
    if spec.kind is bool:
        return QCheckBox()
    if spec.choices is not None:
        box = QComboBox()
        box.addItems(spec.choices)
        return box
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


def _set_widget(widget: QSpinBox | QDoubleSpinBox | QComboBox | QCheckBox, value) -> None:
    if isinstance(widget, QComboBox):
        widget.setCurrentText(value)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(value)
    else:
        widget.setValue(value)


def _get_widget(widget: QSpinBox | QDoubleSpinBox | QComboBox | QCheckBox):
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    return widget.value()


def _connect_change(widget: QSpinBox | QDoubleSpinBox | QComboBox | QCheckBox, slot) -> None:
    if isinstance(widget, QComboBox):
        widget.currentTextChanged.connect(slot)
    elif isinstance(widget, QCheckBox):
        widget.stateChanged.connect(slot)
    else:
        widget.valueChanged.connect(slot)


class SettingsPanel(QWidget):
    """Render/cache/coloring settings for the fields in `groups`, laid out
    as wrapping columns. Reads initial values from `session.settings`.

    LIVE (the default): every edit restarts a short debounce; when it fires,
    this panel's fields are validated and, if all valid, merged into the
    CURRENT session.settings (so two panels over disjoint fields never undo
    each other) and handed to `on_apply` -- SandboxWindow updates the
    session, persists, and re-renders. An invalid field keeps what was typed,
    gets an error border and its own error text under it, and nothing is
    applied until it's fixed.
    """

    def __init__(self, session, on_apply, parent: QWidget | None = None,
                 groups: dict[str, list[str]] | None = None, live: bool = True,
                 schedule_apply=None):
        """`schedule_apply`: if given, called instead of this panel's own
        debounce, so several panels can share one pause-then-apply (the
        window's Rendering + Coloring tabs do)."""
        super().__init__(parent)
        self.session = session
        self._on_apply = on_apply
        self._groups = groups if groups is not None else FIELD_GROUPS
        self._live = live
        self._schedule_apply = schedule_apply
        self._last_good: Settings = session.settings
        self._widgets: dict[str, QSpinBox | QDoubleSpinBox | QComboBox | QCheckBox] = {}
        self._field_errors: dict[str, QLabel] = {}
        self._field_notes: dict[str, QLabel] = {}
        self._cache_readout_label: QLabel | None = None
        self._slow_render_label: QLabel | None = None

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(SETTINGS_APPLY_DEBOUNCE_MS)
        self._apply_timer.timeout.connect(self._apply)

        self._build_ui()
        self._load_into_widgets(self._last_good)
        self._update_slow_render_hint()

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(CACHE_READOUT_REFRESH_MS)
        self._stats_timer.timeout.connect(self._refresh_cache_readout)
        if self._cache_readout_label is not None:
            self._stats_timer.start()
        self._refresh_cache_readout()

    # ---- construction ----------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 10, 18, 10)
        outer.setSpacing(6)
        columns = QWidget(self)
        flow = FlowLayout(columns, spacing=20)
        for group_title, field_names in self._groups.items():
            # A long group splits into columns of at most COLUMN_ROWS fields,
            # so the dock's few rows of height hold every field without scrolling.
            chunks = [field_names[i:i + COLUMN_ROWS] for i in range(0, len(field_names), COLUMN_ROWS)]
            for i, chunk in enumerate(chunks):
                flow.addWidget(self._build_group(group_title if i == 0 else "", chunk,
                                                 group_title))
        outer.addWidget(columns)

        if any(n in self._widgets for n in ("resolution", "max_iter")):
            self._slow_render_label = QLabel()
            self._slow_render_label.setWordWrap(True)
            self._slow_render_label.setProperty("role", "caption")
            outer.addWidget(self._slow_render_label)
        outer.addStretch(1)

    def _build_group(self, title: str, field_names: list[str], group: str = "") -> QWidget:
        column = QWidget(self)
        v = QVBoxLayout(column)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        heading = QLabel(title.upper() or " ", column)
        heading.setProperty("role", "section")
        v.addWidget(heading)
        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(4)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for name in field_names:
            spec = FIELD_SPECS[name]
            widget = _widget_for(name)
            widget.setToolTip(spec.label)
            widget.setAccessibleName(spec.label)
            widget.setMinimumWidth(120)
            _connect_change(widget, self._on_widget_changed)
            self._widgets[name] = widget
            cell = QWidget(column)
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(2)
            cv.addWidget(widget)
            error = QLabel("", cell)
            error.setProperty("role", "error")
            error.setWordWrap(True)
            error.hide()
            cv.addWidget(error)
            self._field_errors[name] = error
            if name in FIELD_NOTES:
                note = QLabel(FIELD_NOTES[name], cell)
                note.setProperty("role", "caption")
                note.setWordWrap(True)
                cv.addWidget(note)
                self._field_notes[name] = note
            label = QLabel(_short_label(spec.label, group or title) + ":", column)
            label.setToolTip(spec.label)
            label.setBuddy(widget)
            form.addRow(label, cell)
        v.addLayout(form)
        if "cache_budget_bytes" in field_names:
            self._cache_readout_label = QLabel(column)
            self._cache_readout_label.setProperty("role", "caption")
            self._cache_readout_label.setWordWrap(True)
            v.addWidget(self._cache_readout_label)
            clear_button = QPushButton("Clear Cache", column)
            clear_button.setToolTip("Drop every cached render")
            clear_button.clicked.connect(self._clear_cache)
            v.addWidget(clear_button, 0, Qt.AlignmentFlag.AlignLeft)
            self._clear_cache_button = clear_button
        v.addStretch(1)
        return column

    def _on_widget_changed(self, *_args) -> None:
        self._update_slow_render_hint()
        if self._live:
            if self._schedule_apply is not None:
                self._schedule_apply()
            else:
                self._apply_timer.start()   # restarts: one apply per pause in editing

    # ---- widget <-> Settings -----------------------------------------------------------
    def _load_into_widgets(self, settings: Settings) -> None:
        for name, widget in self._widgets.items():
            widget.blockSignals(True)
            _set_widget(widget, _to_widget_value(name, getattr(settings, name)))
            widget.blockSignals(False)

    def refresh_from_session(self) -> None:
        """Re-reads every field from session.settings (e.g. after an
        experiment was opened), clearing any pending edit and error."""
        self._apply_timer.stop()
        self._last_good = self.session.settings
        self._load_into_widgets(self._last_good)
        self._show_errors({})
        self._update_slow_render_hint()

    def _draft_settings_from_widgets(self) -> dict:
        """Every Settings field: this panel's widgets over the CURRENT session values."""
        base = dataclasses.asdict(self.session.settings)
        base.update({name: _from_widget_value(name, _get_widget(widget))
                     for name, widget in self._widgets.items()})
        return base

    # ---- apply: validate this panel's fields; all valid -> apply together ------------------
    def apply_pending(self) -> None:
        """Apply now if anything is pending (the shared debounce calls this)."""
        self._apply()

    def _apply(self) -> None:
        self._apply_timer.stop()
        draft = self._draft_settings_from_widgets()
        errors: dict[str, str] = {}
        parsed: dict = dict(draft)
        for name in self._widgets:
            ok, value, error = validate_field(name, draft[name])
            if ok:
                parsed[name] = value
            else:
                errors[name] = error
        self._show_errors(errors)
        if errors:
            return   # keep what was typed; nothing applies until it's fixed
        new_settings = Settings(**parsed)
        if new_settings == self.session.settings or new_settings == self._last_good:
            return   # nothing actually changed (e.g. a value typed back) -- no re-render
        self._last_good = new_settings
        self._on_apply(new_settings)

    def _show_errors(self, errors: dict[str, str]) -> None:
        for name, widget in self._widgets.items():
            message = errors.get(name, "")
            theme.set_invalid(widget, bool(message))
            label = self._field_errors[name]
            label.setText(message)
            label.setVisible(bool(message))

    def field_note(self, name: str) -> str:
        """The note shown under a field ("" if none)."""
        note = self._field_notes.get(name)
        return note.text() if note is not None else ""

    def errors(self) -> dict[str, str]:
        return {n: l.text() for n, l in self._field_errors.items() if l.text()}

    # ---- cache readout / clear ---------------------------------------------------------
    def _refresh_cache_readout(self) -> None:
        if self._cache_readout_label is None:
            return
        stats = self.session.cache.stats
        used_mb = stats.current_bytes / _BYTES_PER_MB
        budget_mb = stats.budget_bytes / _BYTES_PER_MB
        self._cache_readout_label.setText(
            f"{used_mb:.1f} / {budget_mb:.1f} MB used · {stats.entry_count} entries · "
            f"{stats.hits} hits · {stats.misses} misses")

    def _clear_cache(self) -> None:
        self.session.cache.clear()
        self._refresh_cache_readout()

    # ---- slow-render hint --------------------------------------------------------------
    def _update_slow_render_hint(self) -> None:
        # A DRAFT estimate from whatever is currently typed: the warning
        # shows before the (debounced) apply, not after.
        if self._slow_render_label is None:
            return
        draft = self._draft_settings_from_widgets()
        try:
            probe = Settings(**draft)
        except (TypeError, ValueError):
            self._slow_render_label.setText("")
            return
        warning = slow_render_warning(probe)
        self._slow_render_label.setText(warning or "")


def _short_label(label: str, group: str = "") -> str:
    """The form label as shown under its group heading: no parenthesised
    explanation (the tooltip has the full label) and no prefix the heading
    already says ("Green's function band width" under GREEN'S FUNCTION ->
    "Band width")."""
    text = label.split(" (")[0]
    for prefix in (group, "Parameter plane", "Green's function", "Parameter"):
        if prefix and text.lower().startswith(prefix.lower() + " "):
            text = text[len(prefix) + 1:]
            return text[:1].upper() + text[1:]
    return text
