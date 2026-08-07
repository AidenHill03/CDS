"""app/orbit_tracker.py -- orbit seeding, stepping, and classification for
the dynamical-plane click-to-seed feature (P5c section 4).

Pure Python, no Qt dependency -- painting the traced orbit and wiring
mouse clicks are app/sandbox.py's ImageView's job (it already owns the
pixel<->complex screen mapping this needs); this module only owns the
orbit's own state and the reused-not-reimplemented classification logic.

CLASSIFICATION reuses cdx.dynamical_facts (which itself wraps
find_attractors) rather than re-deriving attracting behaviour independently
-- the current orbit point is checked against every already-discovered
attracting cycle's points via cdx.chordal_distance, the SAME sphere-first
metric (infinity is an ordinary point, never a special case) the rest of
this codebase already uses for basin classification. `tol` is meant to be
the caller's actual render_settings.tol -- "close enough to a known
attractor" should mean the same thing here as it means for basin
membership, not a second, independently-chosen notion of "close."

facts are cached per (map, param), same principle (and same cost -- a real
find_attractors call) as every other per-(map,param) cache in this app
(FactsPanel, ImageView's critical-point overlay).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cdx


def _is_finite(z: complex) -> bool:
    return math.isfinite(z.real) and math.isfinite(z.imag)


def _format_point(z: complex) -> str:
    if not _is_finite(z):
        return "infinity"
    return f"{z.real:.6g}{'+' if z.imag >= 0 else ''}{z.imag:.6g}j"


@dataclass
class OrbitClassification:
    """kind is one of "running" (still short of the iteration budget, no
    match yet), "undetermined" (budget exhausted, still no match),
    "converged_fixed" (period 1), or "converged_cycle" (period > 1).
    `text` is always a ready-to-display summary; period/multiplier are
    populated only for the two "converged_*" kinds.
    """
    kind: str
    text: str
    period: int | None = None
    multiplier: complex | None = None


@dataclass
class OrbitState:
    z0: complex
    n: int = 0
    z: complex = field(init=False)
    history: list[complex] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.z = self.z0
        self.history = [self.z0]


class OrbitTracker:
    """Owns at most one traced orbit at a time, for one (map, param) pair.

    CLEARS when the map or parameter changes (see reset_if_stale) --
    an orbit traced under a different map/parameter describes different
    dynamics entirely, so keeping it around would be actively misleading,
    not just stale. Deliberately does NOT clear on a viewport (pan/zoom)
    change -- that's the same dynamics, seen differently, which is
    exactly why nothing in this class ever touches a viewport at all.
    """

    def __init__(self) -> None:
        self._key: tuple[str, complex] | None = None
        self.state: OrbitState | None = None
        self._facts_key: tuple[str, complex] | None = None
        self._facts: cdx.DynamicalFacts | None = None

    def reset_if_stale(self, rational_map: cdx.RationalMap, param: complex) -> None:
        key = (rational_map.serialize(), param)
        if key != self._key:
            self._key = key
            self.state = None

    def seed(self, rational_map: cdx.RationalMap, param: complex, z0: complex) -> None:
        self.reset_if_stale(rational_map, param)
        self.state = OrbitState(z0)

    def clear(self) -> None:
        self.state = None

    def recompute_current(self, rational_map: cdx.RationalMap, param: complex) -> None:
        """Keeps the CURRENT z0 (if any) but restarts its orbit under a
        NEW map/param -- the counterpart to reset_if_stale's own CLEAR
        behaviour, used specifically when the user explicitly changes `a`
        via the parameter field or a parameter-plane click (P6's own
        wording: z0 is a persistent, independently-chosen seed, so its
        orbit under the new map is exactly what should be shown, not a
        blanked overlay). A no-op if no orbit is currently seeded -- there
        is no z0 to replay, and Clear should stay cleared until the user
        picks a new seed.
        """
        if self.state is None:
            return
        self.seed(rational_map, param, self.state.z0)

    def step(self, rational_map: cdx.RationalMap, param: complex, count: int = 1) -> None:
        if self.state is None:
            return
        for _ in range(max(count, 0)):
            if not _is_finite(self.state.z):
                break   # already non-finite -- nothing further to evaluate there
            self.state.z = rational_map.eval(self.state.z, param)
            self.state.n += 1
            self.state.history.append(self.state.z)

    def _facts_for(self, rational_map: cdx.RationalMap, param: complex) -> cdx.DynamicalFacts:
        key = (rational_map.serialize(), param)
        if key != self._facts_key:
            self._facts_key = key
            self._facts = cdx.dynamical_facts(rational_map, param)
        return self._facts

    def classify(self, rational_map: cdx.RationalMap, param: complex, max_iter: int,
                tol: float) -> OrbitClassification:
        if self.state is None:
            return OrbitClassification("none", "")
        z = self.state.z
        facts = self._facts_for(rational_map, param)
        for ac in facts.attracting_cycles:
            for pt in ac.points:
                if cdx.chordal_distance(z, pt) < tol:
                    if ac.period == 1:
                        return OrbitClassification(
                            "converged_fixed",
                            f"Converged to the attracting fixed point at {_format_point(pt)} "
                            f"(multiplier {ac.multiplier:.4g})",
                            period=1, multiplier=ac.multiplier)
                    return OrbitClassification(
                        "converged_cycle",
                        f"Converged to a period-{ac.period} cycle (multiplier "
                        f"{ac.multiplier:.4g})",
                        period=ac.period, multiplier=ac.multiplier)
        if self.state.n >= max_iter:
            plural = "" if self.state.n == 1 else "s"
            return OrbitClassification(
                "undetermined",
                f"Undetermined after {self.state.n} iteration{plural} (within the current "
                "iteration budget)")
        return OrbitClassification("running", f"Running -- {self.state.n} iteration"
                                              f"{'' if self.state.n == 1 else 's'} so far")
