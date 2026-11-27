"""app/label_cuts.py -- where the Per-critical labelling convention cuts.

The parameter plane's Per-critical strategy renders ONE labelled free
critical orbit (critical_index k). Labels are fixed at a base parameter a*
and continued along the straight segment a* -> a (see the C++
CriticalLabelBase comment), so the label k jumps exactly where that segment
crosses a parameter at which two free critical points collide: on the ray
from each collision point pointing directly away from a*. Those rays are a
LABELLING CONVENTION, not dynamics -- a different a* would move them -- so
the app draws them as a dashed overlay whenever Per-critical is active.

Finding the collisions. Let M(z; a) be the factor of the derivative
numerator whose roots are the TRACKED free critical points (finite, not at
a pole, not structurally pre-fixed -- exactly the set the C++ continues).
Its discriminant

    D(a) = lc(a)^(2k-2) * prod_{i<j} (r_i(a) - r_j(a))^2

is a polynomial in a (lc = the derivative numerator's leading coefficient,
which M shares: the factors split off are monic in z), and its zeros are
the collisions. D is sampled on |a| = 1 from the tracked roots, fitted by a
DFT, accepted only if the fit reproduces D at parameters off the circle,
and rooted. A zero of lc itself (a free critical point running off to
infinity, where the tracked-root count changes and the continuation stops)
cuts the same way, so those are found the same way and included.

Pure numerics plus viewport geometry in complex coordinates (no Qt), so
tests can check the rays against the collisions directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import cdx

# Two tracked points closer than this (relative) count as the same point --
# distinct_critical_points' own dedup radius.
_SAME_POINT = 1e-4
_FIT_SIZES = (16, 32, 64, 128, 256, 512)
_FIT_CHECK = (0.37 + 0.21j, -1.6 + 0.9j, 2.3 - 1.1j, -0.2 - 3.1j)
_FIT_REL_TOL = 1e-7
_MAX_ABS = 1e8          # collisions further out than this are not drawn
_CLUSTER = 1e-3         # roots this close (relative) are one collision

TOOLTIP = ("Labelling cut (Per-critical): critical points are numbered at a* and the "
           "numbering is carried along the straight segment a* → a; it swaps when that "
           "segment crosses a collision of two critical points. These rays are a "
           "labelling convention, not dynamics.")


@dataclass
class LabelCuts:
    a_star: complex
    collisions: list[complex] = field(default_factory=list)   # rays start here

    def directions(self) -> list[complex]:
        """Unit direction of each ray: from its collision, directly away from a*."""
        out = []
        for c in self.collisions:
            d = c - self.a_star
            out.append(d / abs(d) if abs(d) > 0 else 1.0 + 0j)
        return out


def _tracked_points(rmap: cdx.RationalMap, a: complex) -> list[complex]:
    """The free critical points the label continuation tracks at `a`."""
    poles = list(rmap.pole_locations(a))
    out = []
    for z in rmap.free_critical_points(a):
        if not np.isfinite(z.real) or not np.isfinite(z.imag):
            continue
        scale = max(1.0, abs(z))
        if any(abs(z - p) <= _SAME_POINT * scale for p in poles):
            continue
        out.append(complex(z))
    return out


def _leading(rmap: cdx.RationalMap, a: complex, degree: int) -> complex:
    coeffs = rmap.deriv_numerator_coefficients(a)
    return complex(coeffs[degree]) if degree < len(coeffs) else 0j


def _discriminant(rmap: cdx.RationalMap, a: complex, degree: int, k: int) -> complex | None:
    pts = _tracked_points(rmap, a)
    if len(pts) != k:
        return None     # a sample on (or extremely near) a collision: unusable
    prod = complex(_leading(rmap, a, degree)) ** (2 * k - 2)
    for i in range(k):
        for j in range(i + 1, k):
            prod *= (pts[i] - pts[j]) ** 2
    return prod


def _fit_polynomial(sample, radius: float = 1.0) -> np.ndarray | None:
    """Ascending coefficients of the polynomial `sample` evaluates (None
    if no fit up to the largest size reproduces it off the circle)."""
    for n in _FIT_SIZES:
        # A half-step rotation keeps the samples off the real axis, where
        # symmetric families put their collisions.
        nodes = radius * np.exp(2j * np.pi * (np.arange(n) + 0.5) / n)
        vals = [sample(complex(a)) for a in nodes]
        if any(v is None for v in vals):
            continue
        vals = np.array(vals, dtype=complex)
        # vals[m] = sum_j c_j nodes[m]^j  ->  c = DFT with the rotation undone.
        coeffs = np.fft.fft(vals) / n
        coeffs *= np.exp(-1j * np.pi * np.arange(n) / n) / radius ** np.arange(n)
        top = np.max(np.abs(coeffs))
        if top == 0.0:
            return np.zeros(1, dtype=complex)
        coeffs[np.abs(coeffs) < 1e-13 * top] = 0.0
        if np.any(coeffs[n // 2:] != 0.0):
            continue    # aliasing: the polynomial needs more samples
        ok = True
        for a in _FIT_CHECK:
            want = sample(a)
            if want is None:
                continue
            got = np.polyval(coeffs[::-1], a)
            if abs(got - want) > _FIT_REL_TOL * max(abs(want), np.sum(np.abs(coeffs)) * 1e-6):
                ok = False
                break
        if ok:
            nz = np.nonzero(coeffs)[0]
            return coeffs[: nz[-1] + 1] if len(nz) else coeffs[:1]
    return None


def _roots(coeffs: np.ndarray) -> list[complex]:
    if coeffs is None or len(coeffs) < 2:
        return []
    return [complex(r) for r in np.roots(coeffs[::-1]) if abs(r) < _MAX_ABS]


def compute_label_cuts(rmap: cdx.RationalMap) -> LabelCuts | None:
    """The cuts of `rmap`'s labelling (None when there is no labelling to
    cut: fewer than two tracked free critical points, critical points that
    do not move with a, or a discriminant that is not a polynomial we can
    fit)."""
    try:
        base = rmap.critical_label_base()
    except Exception:
        return None
    a_star = complex(base["a_star"])
    k = len(base["points"])
    degree = int(base["expected_degree"])
    if base["constant"] or k < 2:
        return None     # nothing moves, or nothing for a label to swap with

    def disc(a):
        return _discriminant(rmap, a, degree, k)
    coeffs = _fit_polynomial(disc)
    if coeffs is None:
        return None
    collisions: list[complex] = _roots(coeffs)

    def lead(a):
        return _leading(rmap, a, degree)
    lc = _fit_polynomial(lead)
    if lc is not None:
        collisions += _roots(lc)
    # A multiple zero of D (and the same point found in both fits) comes
    # back as a tight cluster split by the fit's rounding (~sqrt of it):
    # one collision, at the cluster's mean -- which is accurate where the
    # individual split roots are not.
    clusters: list[list[complex]] = []
    for c in collisions:
        for cl in clusters:
            if abs(c - cl[0]) <= _CLUSTER * max(1.0, abs(cl[0])):
                cl.append(c)
                break
        else:
            clusters.append([c])
    unique = [complex(sum(cl) / len(cl)) for cl in clusters]
    unique = [c for c in unique if abs(c - a_star) > 0]
    unique.sort(key=lambda c: (round(c.real, 9), round(c.imag, 9)))
    return LabelCuts(a_star, unique)


_cache: dict[str, LabelCuts | None] = {}


def label_cuts(rmap: cdx.RationalMap) -> LabelCuts | None:
    """compute_label_cuts, memoised on the map's serialized form (the cuts
    depend on the map only -- never on the viewport or the strategy)."""
    try:
        key = rmap.serialize()
    except Exception:
        return compute_label_cuts(rmap)
    if key not in _cache:
        if len(_cache) > 32:
            _cache.clear()
        _cache[key] = compute_label_cuts(rmap)
    return _cache[key]


# ---- viewport geometry ---------------------------------------------------------------
def clip_ray(origin: complex, direction: complex, lo: complex, hi: complex
             ) -> tuple[complex, complex] | None:
    """The part of the ray origin + t*direction (t >= 0) inside the box
    [lo.real, hi.real] x [lo.imag, hi.imag], or None (Liang-Barsky)."""
    t0, t1 = 0.0, float("inf")
    for p, q_lo, q_hi in ((direction.real, lo.real - origin.real, hi.real - origin.real),
                          (direction.imag, lo.imag - origin.imag, hi.imag - origin.imag)):
        if p == 0.0:
            if q_lo > 0.0 or q_hi < 0.0:
                return None
            continue
        a, b = q_lo / p, q_hi / p
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    return origin + t0 * direction, origin + t1 * direction


def visible_segments(cuts: LabelCuts, center: complex, scale: float, margin: float = 0.05
                     ) -> list[tuple[complex, complex, complex]]:
    """(collision, start, end) for every ray crossing the square viewport
    (center, half-width scale), slightly inflated so a ray along the edge
    still draws. Complex coordinates."""
    pad = scale * (1.0 + margin)
    lo, hi = center - complex(pad, pad), center + complex(pad, pad)
    out = []
    for c, d in zip(cuts.collisions, cuts.directions()):
        seg = clip_ray(c, d, lo, hi)
        if seg is not None:
            out.append((c, seg[0], seg[1]))
    return out


def distance_to_segment(p: complex, s0: complex, s1: complex) -> float:
    d = s1 - s0
    if d == 0:
        return abs(p - s0)
    t = ((p - s0) * d.conjugate()).real / abs(d) ** 2
    t = min(1.0, max(0.0, t))
    return abs(p - (s0 + t * d))
