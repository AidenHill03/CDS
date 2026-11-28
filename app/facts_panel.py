"""app/facts_panel.py -- the Facts tab: read-only dynamical-data extraction.

Everything here comes from cdx.dynamical_facts(map, param) (plus a bit of
Python-side grouping/classification -- see below), displayed in four
read-only tables plus a degree/Riemann-Hurwitz summary. Nothing here
mutates session.map or session.param; clicking a row either re-centres the
viewport (poles -- see on_center_view) or SEEDS AN ORBIT at that point on
the dynamical plane (fixed/critical points, Stage 5 -- see on_seed_orbit
and _on_row_clicked's own routing for exactly which table does which).

CACHED PER (map, param). dynamical_facts() runs a root-finder and an
attracting-cycle search -- real work, not a field lookup -- so refresh()
memoizes on (session.map.serialize(), session.param) and only recomputes
when that key actually changes. SandboxWindow calls refresh() after every
term edit and whenever this tab becomes current; both calls are cheap
no-ops when nothing has changed since the last one.

RIEMANN-HURWITZ CHECK, inline rather than test-only. critical_points()
is documented to return exactly 2*degree-2 points (with multiplicity) on
a well-formed map; this mismatching has caught two real completeness bugs
during development (see CLAUDE.md's pitfalls). Surfacing it here means a
user who builds a custom map that breaks the invariant sees it immediately
next to the numbers that are wrong, not only in a test run.

MULTIPLICITY GROUPING. critical_points() returns each critical point
WITH multiplicity -- a literal duplicate entry per multiplicity, not a
count field -- so grouping them for display reuses
RationalMap.distinct_critical_points() (the same tolerance-and-infinity-
aware dedup the engine itself uses to seed one critical orbit per point,
see rational.cpp) for the representative list, then counts each
representative's occurrences in critical_points() under the same rule,
rather than reimplementing that grouping logic from scratch here.

CLASSIFICATION. FixedPoint only carries (point, multiplier); attracting/
neutral/repelling/superattracting is derived from |multiplier| here, the
same classification test_session.py already checks by hand.
"""

from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

import cdx
from app import theme

CRITICAL_COLUMNS = ("Point", "Multiplicity")
FIXED_COLUMNS = ("Point", "Multiplier", "|Multiplier|", "Classification")
CYCLE_COLUMNS = ("Period", "Multiplier", "|Multiplier|", "Points")
POLE_COLUMNS = ("Location", "Order")

# |multiplier - 1| within this counts as neutral rather than "just barely"
# attracting/repelling -- a display-only grouping choice, not a dynamical
# claim; the raw multiplier is always shown alongside it anyway.
_NEUTRAL_TOL = 1e-6



# Row-click hints (controls redesign, Step 3): each table's caption, and the
# tooltip on every one of its rows.
SEED_ROW_HINT = "Click to seed an orbit here"
CENTER_ROW_HINT = "Click to center the view here"
INFINITE_ROW_HINT = "At infinity -- clicking does nothing"

def _format_complex(z: complex) -> str:
    # str(complex) always round-trips through complex() exactly, but wraps
    # anything with a nonzero imaginary part in parens ("(1+2j)") -- drop
    # them so the cell shows exactly what a user would type back in.
    s = str(z)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s


# The full-precision value of a cell (what its tooltip shows and Copy copies).
_FULL_ROLE = Qt.ItemDataRole.UserRole


def format_short(z: complex, digits: int = 6) -> str:
    """6 significant digits per part, i for the imaginary unit, ∞ for infinity."""
    if _is_inf(z):
        return "∞"
    if z.imag == 0:
        return f"{z.real:.{digits}g}"
    sign = "−" if z.imag < 0 else "+"
    return f"{z.real:.{digits}g} {sign} {abs(z.imag):.{digits}g}i"


def format_full(z: complex) -> str:
    return "∞" if _is_inf(z) else _format_complex(z)


def critical_orbit_fates(rational_map: cdx.RationalMap, a: complex, facts, max_iter: int,
                         tol: float) -> list[tuple[complex, str]]:
    """Where each distinct critical point's orbit goes, for the Info tab's
    "This map" section: the attracting cycle it lands on (by chordal
    distance -- infinity is an ordinary point of the sphere here), or that
    none was found within `max_iter` iterations."""
    out = []
    cycles = list(facts.attracting_cycles)
    for c in rational_map.distinct_critical_points(a):
        z, fate = c, None
        for n in range(max_iter + 1):
            for cyc in cycles:
                if any(cdx.chordal_distance(z, p) < tol for p in cyc.points):
                    at_inf = any(_is_inf(p) for p in cyc.points)
                    if cyc.period == 1:
                        fate = "→ ∞ (attracting fixed point)" if at_inf else \
                            f"→ fixed point {format_short(cyc.points[0], 4)}"
                    else:
                        fate = f"→ {cyc.period}-cycle"
                    break
            if fate is not None or n == max_iter:
                break
            z = rational_map.eval(z, a)
            if math.isnan(z.real) or math.isnan(z.imag):
                break
        out.append((c, fate or f"no attracting cycle within {max_iter} iterations"))
    return out


def _is_inf(z: complex) -> bool:
    return math.isinf(z.real) or math.isinf(z.imag)


def _format_magnitude(z: complex) -> str:
    # A plain, real, non-negative number -- never routed through
    # _format_complex (which always prints a "+0j" suffix, correct for an
    # actual complex value but wrong for a magnitude). ".4g" matches the
    # convention other magnitude-like readouts already use (see
    # app.orbit_tracker's own cycle-multiplier text, app.sandbox's status-
    # bar readouts).
    return f"{abs(z):.6g}"


def _classify(multiplier: complex) -> str:
    m = abs(multiplier)
    if m == 0.0:
        return "superattracting"
    if abs(m - 1.0) < _NEUTRAL_TOL:
        return "neutral"
    return "attracting" if m < 1.0 else "repelling"


def _critical_points_with_multiplicity(rational_map: cdx.RationalMap, a: complex,
                                       rel_tol: float = 1e-4) -> list[tuple[complex, int]]:
    """[(representative, multiplicity), ...], one entry per DISTINCT
    critical point. Mirrors RationalMap::distinct_critical_points' own
    grouping rule (scale-relative tolerance; any two infinities are the
    same point) for the counting pass, so a point's reported multiplicity
    agrees with which bucket distinct_critical_points put it in.
    """
    representatives = rational_map.distinct_critical_points(a, rel_tol)
    all_points = rational_map.critical_points(a)
    groups = []
    for rep in representatives:
        if _is_inf(rep):
            count = sum(1 for z in all_points if _is_inf(z))
        else:
            scale = max(1.0, abs(rep))
            count = sum(1 for z in all_points if not _is_inf(z) and abs(z - rep) < rel_tol * scale)
        groups.append((rep, count))
    return groups


def facts_to_dict(session, render_mode: str, viewport: cdx.Viewport) -> dict:
    """A plain, JSON-serializable snapshot of session.dynamical_facts() for
    the CURRENT (session.map, session.param) -- File > Export Facts (JSON)'s
    own logic (see app/sandbox.py's _do_export_facts). Reuses
    _critical_points_with_multiplicity/_classify, the SAME grouping and
    classification FactsPanel's own tables use, so the exported JSON always
    matches what is actually on screen -- one source of truth for both,
    not a second copy of the grouping logic that could drift out of sync.

    `render_mode` and `viewport` are the CALLER's own -- typically the
    FOCUSED pane's (see app.pane.Pane) -- included purely for PROVENANCE
    (what was being looked at when this was exported), the same reason
    Session.snapshot_to_dict takes them explicitly rather than reading a
    render_mode/viewport this class doesn't own.

    Complex numbers serialize as [real, imag] pairs throughout, matching
    Session.snapshot_to_dict's own convention. A point at infinity
    serializes as literal Infinity (Python's json module emits this by
    default, and its own json.load reads it back) rather than being
    excluded or special-cased -- infinity is an ordinary point here (see
    CLAUDE.md's "sphere-first" convention), not an error case to hide.
    """
    facts = session.dynamical_facts()
    expected = 2 * facts.degree - 2
    actual = len(facts.critical_points)
    matches = actual == expected
    if matches:
        rh_summary = (f"{actual} critical points with multiplicity, matches 2d-2 = {expected}")
    else:
        rh_summary = (f"MISMATCH: {actual} critical points counted, expected 2d-2 = {expected} "
                      "-- critical_points() is incomplete for this map")

    def c(z: complex) -> list[float]:
        return [z.real, z.imag]

    critical_groups = _critical_points_with_multiplicity(session.map, session.param)
    return {
        "map": {"name": session.map.name, "formula": session.map.to_formula()},
        "param": c(session.param),
        "render_mode": render_mode,
        "viewport": {
            "center": c(viewport.center),
            "scale": viewport.scale,
            "resolution": viewport.resolution,
        },
        "degree": facts.degree,
        "riemann_hurwitz": {
            "critical_points_with_multiplicity": actual,
            "expected_2d_minus_2": expected,
            "matches": matches,
            "summary": rh_summary,
        },
        "critical_points": [{"point": c(point), "multiplicity": mult}
                            for point, mult in critical_groups],
        "fixed_points": [{"point": c(fp.point), "multiplier": c(fp.multiplier),
                          "multiplier_magnitude": abs(fp.multiplier),
                          "classification": _classify(fp.multiplier)}
                         for fp in facts.fixed_points],
        "attracting_cycles": [{"period": cyc.period, "multiplier": c(cyc.multiplier),
                               "multiplier_magnitude": abs(cyc.multiplier),
                               "points": [c(p) for p in cyc.points]}
                              for cyc in facts.attracting_cycles],
        "poles": [{"location": c(loc), "order": order}
                 for loc, order in zip(facts.pole_locations, facts.pole_orders)],
    }


def _cell(z: complex) -> QTableWidgetItem:
    """A table cell showing z to 6 significant digits, holding the full value."""
    item = QTableWidgetItem(format_short(z))
    item.setData(_FULL_ROLE, format_full(z))
    return item


class FactsPanel(QWidget):
    def __init__(self, session, on_center_view: Callable[[complex], None],
                on_seed_orbit: Callable[[complex], None], parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self._on_center_view = on_center_view
        # Stage 5: a fixed-point or critical-point row SEEDS an orbit there
        # instead of recentring -- seeding is the action; the orbit marker
        # (on whichever pane it lands on -- see SandboxWindow's own
        # routing) shows where it is. Pole rows keep the EXISTING
        # centre-the-view behavior unchanged (see _on_row_clicked below for
        # which table routes to which callback).
        self._on_seed_orbit = on_seed_orbit
        self._cache_key = None
        self._facts: cdx.DynamicalFacts | None = None
        # Keyed by id(table), not the table itself (QTableWidget isn't
        # hashable in a way worth relying on) -- which complex point
        # clicking a given row should centre the view on.
        self._row_points: dict[int, list[complex | None]] = {}
        # What clicking a row of each table does, as a tooltip/caption.
        self._row_hints: dict[int, str] = {}
        self._section_toggles: dict[int, QToolButton] = {}

        self._build_ui()
        self.refresh()

    # ---- construction ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Compact summary on top: degree, critical points, attracting cycles.
        summary = QFormLayout()
        summary.setHorizontalSpacing(12)
        summary.setVerticalSpacing(6)
        self._degree_label = QLabel()
        self._critical_summary = QLabel()
        self._cycles_summary = QLabel()
        for label in (self._degree_label, self._critical_summary, self._cycles_summary):
            label.setProperty("mono", True)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for caption, value in (("Degree", self._degree_label),
                               ("Critical points", self._critical_summary),
                               ("Attracting cycles", self._cycles_summary)):
            name = QLabel(caption)
            name.setProperty("role", "muted")
            summary.addRow(name, value)
        layout.addLayout(summary)

        self._rh_label = QLabel()
        self._rh_label.setWordWrap(True)
        self._rh_label.setProperty("role", "caption")
        layout.addWidget(self._rh_label)

        self._critical_table = self._add_table_group(layout, "Critical Points", CRITICAL_COLUMNS,
                                                     SEED_ROW_HINT)
        self._fixed_table = self._add_table_group(layout, "Fixed Points", FIXED_COLUMNS,
                                                  SEED_ROW_HINT)
        self._cycle_table = self._add_table_group(layout, "Attracting Cycles", CYCLE_COLUMNS,
                                                  CENTER_ROW_HINT)
        self._pole_table = self._add_table_group(layout, "Poles", POLE_COLUMNS, CENTER_ROW_HINT)

        row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Recompute the facts for the current map and a")
        refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        row.addWidget(refresh_btn)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addStretch(1)

    def _add_table_group(self, layout: QVBoxLayout, title: str,
                         columns: tuple[str, ...], row_hint: str) -> QTableWidget:
        """A collapsible section: a header button, the click caption, the table."""
        group = QWidget()
        v = QVBoxLayout(group)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        toggle = QToolButton(group)
        toggle.setProperty("role", "sectionToggle")
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.DownArrow)
        toggle.setAccessibleName(f"{title}: show or hide")
        v.addWidget(toggle)
        caption = QLabel(row_hint)
        caption.setProperty("role", "caption")
        v.addWidget(caption)
        table = QTableWidget(0, len(columns))
        self._row_hints[id(table)] = row_hint
        table.setHorizontalHeaderLabels(columns)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setProperty("mono", True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.cellClicked.connect(lambda row, _column, t=table: self._on_row_clicked(t, row))
        # Copy copies the FULL values of the selected rows (cells show 6 digits).
        copy = QAction("Copy", table)
        copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        copy.triggered.connect(lambda _checked=False, t=table: self.copy_rows(t))
        table.addAction(copy)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        v.addWidget(table)
        toggle.toggled.connect(lambda on, c=caption, t=table, b=toggle: (
            c.setVisible(on), t.setVisible(on),
            b.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)))
        self._section_toggles[id(table)] = toggle
        layout.addWidget(group)
        return table

    def section_toggle(self, table: QTableWidget) -> QToolButton:
        return self._section_toggles[id(table)]

    def copy_rows(self, table: QTableWidget) -> str:
        rows = sorted({i.row() for i in table.selectedItems()}) or (
            [table.currentRow()] if table.currentRow() >= 0 else [])
        lines = []
        for r in rows:
            cells = []
            for c in range(table.columnCount()):
                item = table.item(r, c)
                cells.append("" if item is None else (item.data(_FULL_ROLE) or item.text()))
            lines.append("\t".join(cells))
        text = "\n".join(lines)
        if text:
            QGuiApplication.clipboard().setText(text)
        return text

    def _fit_table_height(self, table: QTableWidget) -> None:
        """Tables show all their rows; the sidebar scrolls, not each table."""
        height = table.horizontalHeader().height() + 2 * table.frameWidth()
        for r in range(table.rowCount()):
            height += table.rowHeight(r)
        if table.horizontalScrollBar().isVisible() or table.horizontalHeader().length() > table.viewport().width():
            height += table.horizontalScrollBar().sizeHint().height()
        table.setFixedHeight(height)

    # ---- refresh: recompute only when (map, param) actually changed --------------------
    def refresh(self, force: bool = False) -> None:
        key = (self.session.map.serialize(), self.session.param)
        if not force and key == self._cache_key:
            return
        self._cache_key = key
        self._facts = self.session.dynamical_facts()
        self._render_facts()

    def _render_facts(self) -> None:
        facts = self._facts
        self._degree_label.setText(str(facts.degree))
        self._degree_label.setAccessibleName(f"Degree: {facts.degree}")
        groups = _critical_points_with_multiplicity(self.session.map, self.session.param)
        self._critical_summary.setText(", ".join(
            format_short(p, 4) + (f" (×{m})" if m > 1 else "") for p, m in groups) or "none")
        cycles = list(facts.attracting_cycles)
        self._cycles_summary.setText(", ".join(
            ("∞ (fixed)" if cyc.period == 1 and any(_is_inf(p) for p in cyc.points)
             else "fixed" if cyc.period == 1 else f"period {cyc.period}") for cyc in cycles)
            or "none found")

        expected = 2 * facts.degree - 2
        actual = len(facts.critical_points)
        if actual == expected:
            theme.set_property(self._rh_label, "role", "muted")
            self._rh_label.setText(
                f"Riemann-Hurwitz: {actual} critical points with multiplicity, "
                f"matches 2d-2 = {expected} ✓")
        else:
            theme.set_property(self._rh_label, "role", "error")
            self._rh_label.setText(
                f"Riemann-Hurwitz MISMATCH: {actual} critical points counted, expected "
                f"2d-2 = {expected} -- critical_points() is incomplete for this map")

        self._fill_critical_table()
        self._fill_fixed_table()
        self._fill_cycle_table()
        self._fill_pole_table()

    def _fill_critical_table(self) -> None:
        table = self._critical_table
        groups = _critical_points_with_multiplicity(self.session.map, self.session.param)
        table.setRowCount(len(groups))
        points: list[complex | None] = []
        for row, (point, mult) in enumerate(groups):
            table.setItem(row, 0, _cell(point))
            table.setItem(row, 1, QTableWidgetItem(str(mult)))
            points.append(point)
        self._row_points[id(table)] = points
        self._apply_row_tooltips(table)

    def _fill_fixed_table(self) -> None:
        table = self._fixed_table
        pts = self._facts.fixed_points
        table.setRowCount(len(pts))
        points: list[complex | None] = []
        for row, fp in enumerate(pts):
            table.setItem(row, 0, _cell(fp.point))
            table.setItem(row, 1, _cell(fp.multiplier))
            table.setItem(row, 2, QTableWidgetItem(_format_magnitude(fp.multiplier)))
            table.setItem(row, 3, QTableWidgetItem(_classify(fp.multiplier)))
            points.append(fp.point)
        self._row_points[id(table)] = points
        self._apply_row_tooltips(table)

    def _fill_cycle_table(self) -> None:
        table = self._cycle_table
        cycles = self._facts.attracting_cycles
        table.setRowCount(len(cycles))
        points: list[complex | None] = []
        for row, cyc in enumerate(cycles):
            table.setItem(row, 0, QTableWidgetItem(str(cyc.period)))
            table.setItem(row, 1, _cell(cyc.multiplier))
            table.setItem(row, 2, QTableWidgetItem(_format_magnitude(cyc.multiplier)))
            pts = QTableWidgetItem(", ".join(format_short(p) for p in cyc.points))
            pts.setData(_FULL_ROLE, ", ".join(format_full(p) for p in cyc.points))
            table.setItem(row, 3, pts)
            # A cycle row centres on its FIRST point -- some single choice
            # has to be made, and points[0] is the critical point's own
            # orbit landing spot that discovered this cycle (see
            # find_attractors), not an arbitrary pick.
            points.append(cyc.points[0] if cyc.points else None)
        self._row_points[id(table)] = points
        self._apply_row_tooltips(table)

    def _fill_pole_table(self) -> None:
        table = self._pole_table
        locations = self._facts.pole_locations
        orders = self._facts.pole_orders
        table.setRowCount(len(locations))
        points: list[complex | None] = []
        for row, (loc, order) in enumerate(zip(locations, orders)):
            table.setItem(row, 0, _cell(loc))
            table.setItem(row, 1, QTableWidgetItem(str(order)))
            points.append(loc)
        self._row_points[id(table)] = points
        self._apply_row_tooltips(table)

    def _apply_row_tooltips(self, table: QTableWidget) -> None:
        """Each cell's tooltip: its full-precision value (if shortened), then
        what clicking the row does."""
        hint = self._row_hints[id(table)]
        for row, point in enumerate(self._row_points[id(table)]):
            text = hint if point is not None and not _is_inf(point) else INFINITE_ROW_HINT
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is not None:
                    full = item.data(_FULL_ROLE)
                    item.setToolTip(f"{full}\n{text}" if full and full != item.text() else text)
        self._fit_table_height(table)

    # ---- click a row -> seeds an orbit (critical/fixed) or centres the view (pole) ------
    def _on_row_clicked(self, table: QTableWidget, row: int) -> None:
        points = self._row_points.get(id(table), [])
        if not 0 <= row < len(points):
            return
        point = points[row]
        if point is None or _is_inf(point):
            return   # infinity (or an empty cycle, defensively) has nowhere to seed/centre on
        if table is self._critical_table or table is self._fixed_table:
            # Stage 5: NOT also a recentre -- seeding is the action, the
            # orbit marker itself shows where the point is.
            self._on_seed_orbit(point)
        else:
            # Pole rows (and cycle rows, unmentioned by Stage 5's own spec
            # -- kept at their EXISTING, unchanged behavior rather than
            # guessed at) still just centre the view.
            self._on_center_view(point)
