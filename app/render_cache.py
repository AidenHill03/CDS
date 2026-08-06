"""app/render_cache.py -- byte-budgeted cache for rendered arrays.

No caching exists anywhere in cdx/ (the engine stays stateless apart from
Renderer's own configuration -- see ARCHITECTURE.md's layering rules), so
this lives here instead: a plain in-memory, thread-safe, LRU-by-access
cache keyed on everything that affects a render's PIXELS, not everything
that affects its speed.

Renders happen on a QThreadPool worker thread (see app/sandbox.py's
RenderTask) while the GUI thread may read cache stats for a live readout
(see the Settings panel) or, in principle, render synchronously (e.g.
Session.render() called directly, as app/test_session.py does) -- so every
public method here takes a lock.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

# `threads` is deliberately NOT part of the cache key: it changes how fast
# a render computes, never what it computes. Keying on it would turn a pure
# performance knob into a cache-defeating one -- flip thread count, lose
# every hit, for pixels that are byte-identical either way.
CacheKey = tuple


def make_key(map_serialized: str, param: complex, mode: str,
            viewport_center: complex, viewport_scale: float, viewport_resolution: int,
            max_iter: int, escape_radius: float, tol: float) -> CacheKey:
    """Builds a cache key from exactly the inputs that affect a render's
    pixels. Callers pass viewport fields (not a cdx.Viewport) and
    map.serialize() (not a RationalMap) so this stays a hashable tuple of
    plain values -- cdx objects are not guaranteed hashable/comparable.
    """
    return (map_serialized, param, mode, viewport_center, viewport_scale,
            viewport_resolution, max_iter, escape_radius, tol)


@dataclass
class CacheStats:
    hits: int
    misses: int
    entry_count: int
    current_bytes: int
    budget_bytes: int


# A cached value is usually a plain ndarray, but a render mode that carries
# extra per-render metadata alongside its pixels (e.g. "greens"/
# "parameter_greens" pairing the array with the normalized-vs-overflowed
# bool -- see app.session.render_map) stores a (ndarray, metadata) tuple
# instead. Only the array actually costs bytes worth budgeting against; a
# bool (or any other small metadata payload) is negligible, so this is the
# ONE place that distinction matters -- every other RenderCache method
# already treats the stored value as opaque.
CachedValue = np.ndarray | tuple


def _nbytes(value: CachedValue) -> int:
    array = value[0] if isinstance(value, tuple) else value
    return array.nbytes


class RenderCache:
    """Maps CacheKey -> a rendered NumPy array (or, for a mode with extra
    metadata, an (array, metadata) tuple -- see CachedValue), evicting
    least-recently-USED entries (not least-recently-inserted: a re-fetched
    entry is moved to the front, same as a normal LRU) once `current_bytes`
    would exceed `budget_bytes`. Changing the budget down evicts
    immediately; changing it up just raises the ceiling for future inserts
    -- existing entries are never touched by a budget change alone.
    """

    def __init__(self, budget_bytes: int) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[CacheKey, CachedValue] = OrderedDict()
        self._current_bytes = 0
        self._budget_bytes = budget_bytes
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> CachedValue | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self._misses += 1
                return None
            self._hits += 1
            self._entries.move_to_end(key)
            return value

    def put(self, key: CacheKey, value: CachedValue) -> None:
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._current_bytes -= _nbytes(existing)
            self._entries[key] = value
            self._current_bytes += _nbytes(value)
            self._evict_to_budget()

    def _evict_to_budget(self) -> None:
        # Caller holds self._lock. A single entry larger than the whole
        # budget is still stored -- it just evicts everything else and the
        # budget check below leaves current_bytes over budget for it alone,
        # rather than refusing to cache anything at that resolution at all.
        while self._current_bytes > self._budget_bytes and len(self._entries) > 1:
            _stale_key, stale_value = self._entries.popitem(last=False)
            self._current_bytes -= _nbytes(stale_value)

    def set_budget(self, budget_bytes: int) -> None:
        with self._lock:
            self._budget_bytes = budget_bytes
            self._evict_to_budget()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._current_bytes = 0
            # Deliberately NOT resetting hits/misses -- those describe this
            # cache instance's lifetime effectiveness, not what happens to
            # be resident right now; a clear shouldn't erase that history.

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(hits=self._hits, misses=self._misses,
                              entry_count=len(self._entries),
                              current_bytes=self._current_bytes,
                              budget_bytes=self._budget_bytes)
