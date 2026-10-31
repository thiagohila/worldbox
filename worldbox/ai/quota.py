"""Stop asking the LLM while its quota is spent.

A 429 means the provider refuses for now: the per-minute rate, or (on free
plans) a daily quota that can stay spent for hours. Retrying every couple of
minutes only piles up failures, so every caller that shares a key shares
one gate: after a 429 they wait, longer after each one in a row (1 minute,
then 5, 15, 30, an hour). Any answer reopens it. Calls that were already in
flight when the gate closed come back with 429 too; those do not count again,
so a burst of callers does not jump straight to the longest wait.

The tribe minds keep one on ``MindDirector.quota``; code built on the game
(its own minds, a chat) can use the same object.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict

WAITS = (60.0, 300.0, 900.0, 1800.0, 3600.0)  # seconds, by 429s in a row


class QuotaGate:
    """Shared by every caller of the same LLM source."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self.strikes = 0
        self.until = 0.0
        self.error = ""

    def open(self) -> bool:
        with self._lock:
            return self._clock() >= self.until

    def retry_in(self) -> int:
        """Seconds until the next attempt is allowed (0 if open)."""
        with self._lock:
            return max(0, int(round(self.until - self._clock())))

    def record(self, result: Any) -> None:
        """Note how a call went (a ``ChatResult``)."""
        with self._lock:
            if getattr(result, "status", 0) == 429:
                if self._clock() < self.until:
                    return  # Already waiting: the same refusal, seen by another caller.
                self.until = self._clock() + WAITS[min(self.strikes, len(WAITS) - 1)]
                self.strikes += 1
                self.error = str(getattr(result, "error", "") or "API returned 429")[:200]
            elif getattr(result, "ok", False):
                self.strikes, self.until, self.error = 0, 0.0, ""

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            wait = max(0, int(round(self.until - self._clock())))
            return {"limited": wait > 0, "retryIn": wait, "strikes": self.strikes}
