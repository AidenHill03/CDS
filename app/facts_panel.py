"""app/facts_panel.py -- the Facts tab: read-only dynamical-data extraction.

Everything here comes from cdx.dynamical_facts(map, param) (plus a bit of
Python-side grouping/classification -- see below), displayed in four
read-only tables plus a degree/Riemann-Hurwitz summary. Nothing here
mutates session.map or session.param; the only session-affecting action is
clicking a row, which re-centres the viewport (see _on_center_view).

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

from PySide6.QtWidgets import (QAbstractItemView, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

import cdx
from app.term_editor_panel import _format_complex

CRITICAL_COLUMNS = ("Point", "Multiplicity")
FIXED_COLUMNS = ("Point", "Multiplier", "Classification")
CYCLE_COLUMNS = ("Period", "Multiplier", "Points")
POLE_COLUMNS = ("Location", "Order")

# |multiplier - 1| within this counts as neutral rather than "just barely"
# attracting/repelling -- a display-only grouping choice, not a dynamical
# claim; the raw multiplier is always shown alongside it anyway.
_NEUTRAL_TOL = 1e-6


def _is_inf(z: complex) -> bool:
    return math.isinf(z.real) or math.isinf(z.imag)


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


class FactsPanel(QWidget):
    def __init__(self, session, on_center_view: Callable[[complex], None],
                parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self._on_center_view = on_center_view
        self._cache_key = None
        self._facts: cdx.DynamicalFacts | None = None
        # Keyed by id(table), not the table itself (QTableWidget isn't
        # hashable in a way worth relying on) -- which complex point
        # clicking a given row should centre the view on.
        self._row_points: dict[int, list[complex | None]] = {}

        self._build_ui()
        self.refresh()

    # ---- construction ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._degree_label = QLabel()
        self._degree_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._degree_label)

        self._rh_label = QLabel()
        self._rh_label.setWordWrap(True)
        layout.addWidget(self._rh_label)

        self._critical_table = self._add_table_group(layout, "Critical Points", CRITICAL_COLUMNS)
        self._fixed_table = self._add_table_group(layout, "Fixed Points", FIXED_COLUMNS)
        self._cycle_table = self._add_table_group(layout, "Attracting Cycles", CYCLE_COLUMNS)
        self._pole_table = self._add_table_group(layout, "Poles", POLE_COLUMNS)

        row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        row.addWidget(refresh_btn)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addStretch(1)

    def _add_table_group(self, layout: QVBoxLayout, title: str,
                         columns: tuple[str, ...]) -> QTableWidget:
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.cellClicked.connect(lambda row, _column, t=table: self._on_row_clicked(t, row))
        v.addWidget(table)
        layout.addWidget(box)
        return table

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
        self._degree_label.setText(f"Degree: {facts.degree}")

        expected = 2 * facts.degree - 2
        actual = len(facts.critical_points)
        if actual == expected:
            self._rh_label.setStyleSheet("")
            self._rh_label.setText(
                f"Riemann-Hurwitz: {actual} critical points with multiplicity, "
                f"matches 2d-2 = {expected} ✓")
        else:
            self._rh_label.setStyleSheet("color: #cc4444; font-weight: bold;")
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
            table.setItem(row, 0, QTableWidgetItem(_format_complex(point)))
            table.setItem(row, 1, QTableWidgetItem(str(mult)))
            points.append(point)
        self._row_points[id(table)] = points

    def _fill_fixed_table(self) -> None:
        table = self._fixed_table
        pts = self._facts.fixed_points
        table.setRowCount(len(pts))
        points: list[complex | None] = []
        for row, fp in enumerate(pts):
            table.setItem(row, 0, QTableWidgetItem(_format_complex(fp.point)))
            table.setItem(row, 1, QTableWidgetItem(_format_complex(fp.multiplier)))
            table.setItem(row, 2, QTableWidgetItem(_classify(fp.multiplier)))
            points.append(fp.point)
        self._row_points[id(table)] = points

    def _fill_cycle_table(self) -> None:
        table = self._cycle_table
        cycles = self._facts.attracting_cycles
        table.setRowCount(len(cycles))
        points: list[complex | None] = []
        for row, cyc in enumerate(cycles):
            table.setItem(row, 0, QTableWidgetItem(str(cyc.period)))
            table.setItem(row, 1, QTableWidgetItem(_format_complex(cyc.multiplier)))
            table.setItem(row, 2, QTableWidgetItem(", ".join(_format_complex(p)
                                                              for p in cyc.points)))
            # A cycle row centres on its FIRST point -- some single choice
            # has to be made, and points[0] is the critical point's own
            # orbit landing spot that discovered this cycle (see
            # find_attractors), not an arbitrary pick.
            points.append(cyc.points[0] if cyc.points else None)
        self._row_points[id(table)] = points

    def _fill_pole_table(self) -> None:
        table = self._pole_table
        locations = self._facts.pole_locations
        orders = self._facts.pole_orders
        table.setRowCount(len(locations))
        points: list[complex | None] = []
        for row, (loc, order) in enumerate(zip(locations, orders)):
            table.setItem(row, 0, QTableWidgetItem(_format_complex(loc)))
            table.setItem(row, 1, QTableWidgetItem(str(order)))
            points.append(loc)
        self._row_points[id(table)] = points

    # ---- click a row -> centre the view on its point ------------------------------------
    def _on_row_clicked(self, table: QTableWidget, row: int) -> None:
        points = self._row_points.get(id(table), [])
        if not 0 <= row < len(points):
            return
        point = points[row]
        if point is None or _is_inf(point):
            return   # infinity (or an empty cycle, defensively) has nowhere to centre on
        self._on_center_view(point)
