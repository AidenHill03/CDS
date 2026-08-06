"""test_render_cache.py -- property-based checks for app/render_cache.

Mirrors the rest of this project's test style (PASS/FAIL per check,
properties rather than golden output). Run with:

    PYTHONPATH=cdx/build python -m app.test_render_cache

(from the repository root) -- no cdx import is actually needed by this
module, but running it the same package-qualified way as the rest of the
app/ suite keeps one invocation pattern for all of them.
"""

from __future__ import annotations

import threading

import numpy as np

from app.render_cache import RenderCache, make_key

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    print("=== app.render_cache tests ===")

    # ---- basic get/put/miss --------------------------------------------------------
    print("\nget/put/miss:")
    cache = RenderCache(budget_bytes=10_000_000)
    key_a = make_key("map-a", 0j, "julia", 0j, 1.5, 100, 200, 2.0, 1e-6)
    key_b = make_key("map-b", 0j, "julia", 0j, 1.5, 100, 200, 2.0, 1e-6)

    check(cache.get(key_a) is None, "an empty cache misses on any key")
    check(cache.stats.misses == 1, "a miss increments the miss counter")

    array_a = np.arange(9, dtype=np.float64).reshape(3, 3)
    cache.put(key_a, array_a)
    fetched = cache.get(key_a)
    check(fetched is not None and np.array_equal(fetched, array_a),
          "a put key is fetched back byte-identical")
    check(cache.stats.hits == 1, "a hit increments the hit counter")

    check(cache.get(key_b) is None, "a different key still misses")
    check(cache.stats.misses == 2, "misses accumulate across distinct keys")

    # ---- key composition: threads excluded, everything else included -------------
    print("\ncache key composition:")
    same_shape_diff_threads = make_key("map-a", 0j, "julia", 0j, 1.5, 100, 200, 2.0, 1e-6)
    check(key_a == same_shape_diff_threads,
          "threads is not part of the key -- two identical requests differing "
          "only in thread count share a cache entry")

    key_diff_resolution = make_key("map-a", 0j, "julia", 0j, 1.5, 200, 200, 2.0, 1e-6)
    check(key_a != key_diff_resolution,
          "a different resolution is a different key (old entries become "
          "unreachable, not wrong, on a resolution change)")

    # ---- LRU eviction under a byte budget ------------------------------------------
    print("\nLRU eviction:")
    # Each array is 8*8*8 = 512 bytes (float64). Budget for ~2.5 entries so
    # the 3rd put must evict exactly one.
    evict_cache = RenderCache(budget_bytes=1300)
    k1, k2, k3 = (make_key(f"m{i}", 0j, "julia", 0j, 1.5, 100, 200, 2.0, 1e-6)
                 for i in range(3))
    arr = np.zeros((8, 8), dtype=np.float64)   # 512 bytes each
    check(arr.nbytes == 512, "sanity: test array is 512 bytes")

    evict_cache.put(k1, arr)
    evict_cache.put(k2, arr)
    check(evict_cache.stats.entry_count == 2 and evict_cache.stats.current_bytes == 1024,
          "two 512-byte entries fit under a 1300-byte budget")

    evict_cache.put(k3, arr)   # 1536 > 1300 -- must evict the LEAST RECENTLY USED
    check(evict_cache.stats.entry_count == 2 and evict_cache.stats.current_bytes == 1024,
          "a third entry evicts exactly one to stay under budget")
    check(evict_cache.get(k1) is None and evict_cache.get(k2) is not None
          and evict_cache.get(k3) is not None,
          "k1 (never re-fetched, oldest) was evicted; k2 and k3 remain")

    # Access order, not insertion order, decides who's "least recently
    # used": fetch k2 to promote it, then force another eviction and
    # confirm k3 (now the stale one) goes, not k2.
    access_cache = RenderCache(budget_bytes=1300)
    access_cache.put(k1, arr)
    access_cache.put(k2, arr)
    access_cache.get(k1)          # promote k1 -- k2 is now the stale one
    access_cache.put(k3, arr)     # must evict k2, not k1
    check(access_cache.get(k2) is None and access_cache.get(k1) is not None,
          "get() promotes an entry -- eviction is by LAST ACCESS, not insertion order")

    # A single entry bigger than the whole budget is still stored, not refused.
    big_cache = RenderCache(budget_bytes=100)
    big_arr = np.zeros((8, 8), dtype=np.float64)   # 512 bytes > 100-byte budget
    big_cache.put(k1, big_arr)
    check(big_cache.get(k1) is not None,
          "an entry larger than the whole budget is still cached, not refused")

    # ---- set_budget: lowering evicts immediately; raising evicts nothing ---------
    print("\nset_budget:")
    budget_cache = RenderCache(budget_bytes=2000)
    budget_cache.put(k1, arr)
    budget_cache.put(k2, arr)
    budget_cache.put(k3, arr)
    check(budget_cache.stats.entry_count == 3, "three 512-byte entries fit under 2000")

    budget_cache.set_budget(600)
    check(budget_cache.stats.entry_count == 1 and budget_cache.stats.current_bytes <= 600,
          "lowering the budget evicts immediately down to the new ceiling")

    budget_cache.set_budget(10_000)
    check(budget_cache.stats.entry_count == 1,
          "raising the budget does not resurrect already-evicted entries")

    # ---- clear() ---------------------------------------------------------------------
    # ---- tuple payloads: a mode with extra metadata (e.g. greens' normalized flag) ---
    print("\ntuple payloads:")
    tuple_cache = RenderCache(budget_bytes=10_000_000)
    key_g = make_key("map-g", 0j, "greens", 0j, 1.5, 100, 200, 2.0, 1e-6)
    payload = (np.arange(9, dtype=np.float64).reshape(3, 3), False)
    tuple_cache.put(key_g, payload)
    fetched_payload = tuple_cache.get(key_g)
    check(fetched_payload is not None, "a tuple payload is fetched back on a hit")
    check(np.array_equal(fetched_payload[0], payload[0]) and fetched_payload[1] is False,
          "both the array and the metadata (here, normalized=False) round-trip exactly")
    check(tuple_cache.stats.current_bytes == payload[0].nbytes,
          "byte accounting for a tuple payload counts only the array, not the metadata")

    replaced = (np.arange(4, dtype=np.float64).reshape(2, 2), True)
    tuple_cache.put(key_g, replaced)
    check(tuple_cache.stats.current_bytes == replaced[0].nbytes,
          "replacing a tuple entry updates current_bytes to the NEW array's size, not a "
          "stale leftover from the old one")

    print("\nclear:")
    clear_cache = RenderCache(budget_bytes=10_000)
    clear_cache.put(k1, arr)
    clear_cache.get(k1)     # one hit
    clear_cache.get(k2)     # one miss
    clear_cache.clear()
    stats = clear_cache.stats
    check(stats.entry_count == 0 and stats.current_bytes == 0,
          "clear() empties every entry and zeroes current_bytes")
    check(stats.hits == 1 and stats.misses == 1,
          "clear() does not reset the hit/miss counters -- they describe this "
          "cache's lifetime effectiveness, not what's resident right now")

    # ---- thread safety: concurrent put/get does not corrupt state -----------------
    print("\nconcurrent access:")
    stress_cache = RenderCache(budget_bytes=200_000)
    keys = [make_key(f"stress{i}", 0j, "julia", 0j, 1.5, 100, 200, 2.0, 1e-6)
           for i in range(20)]
    small = np.zeros((4, 4), dtype=np.float64)

    def worker() -> None:
        for _ in range(200):
            for k in keys:
                stress_cache.put(k, small)
                stress_cache.get(k)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final_stats = stress_cache.stats
    check(final_stats.current_bytes == sum(e.nbytes for e in stress_cache._entries.values()),
          "current_bytes stays consistent with actual resident entries after concurrent access")
    check(final_stats.hits + final_stats.misses == 6 * 200 * len(keys),
          "no get() call was lost or double-counted under concurrent access")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
