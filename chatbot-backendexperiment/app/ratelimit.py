"""
Fixed-window, per-client rate limiting.

Deliberately kept free of any FastAPI/Starlette import so the logic can be
unit-tested without the web framework installed (see
tests/test_core_pipeline.py). `main.py` owns the HTTP concerns — reading
the client IP and turning a refusal into a 429 response.

NOTE: state is per-process and in-memory. That is correct for a single
backend instance, which is how this project is deployed. Running multiple
workers or instances would give each its own counters, so a shared store
(e.g. Redis) should replace this class at that point — the interface is
small enough that only this file would change.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        prune_interval_seconds: int = 300,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.prune_interval_seconds = prune_interval_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._last_prune = 0.0

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a request from `key` and report whether it is allowed.

        `now` is injectable so tests can advance time deterministically
        instead of sleeping."""
        now = time.time() if now is None else now
        window_start = now - self.window_seconds

        bucket = self._hits[key]
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            self._maybe_prune(now, window_start)
            return False

        bucket.append(now)
        self._maybe_prune(now, window_start)
        return True

    def _maybe_prune(self, now: float, window_start: float) -> None:
        """Drop buckets for clients not seen within the window.

        Without this, one deque per client IP would be retained forever —
        a slow memory leak on a public site, where every scanner and
        crawler earns a permanent entry. Pruning is amortised (it runs at
        most once per prune_interval_seconds) so `allow()` stays O(1) in
        the common case."""
        if now - self._last_prune < self.prune_interval_seconds:
            return
        self._last_prune = now
        stale = [k for k, times in self._hits.items() if not times or times[-1] < window_start]
        for k in stale:
            del self._hits[k]

    @property
    def tracked_clients(self) -> int:
        """Number of clients currently held in memory (used by tests)."""
        return len(self._hits)
