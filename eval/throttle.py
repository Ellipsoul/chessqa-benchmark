"""Rate limiting and retry classification for the ChessQA eval runner.

Kept separate from run_openrouter.py so it is unit-testable without importing chess/tqdm,
and because it has no persistence concerns. Two responsibilities:

1. ``RateLimiter``: a thread-shared token bucket with a global penalty gate. All worker
   threads call ``acquire()`` before each HTTP attempt, so request *starts* are paced at a
   configured rate regardless of worker count — concurrency covers in-flight latency, the
   bucket governs throughput. A 429 (or Retry-After on any status) calls ``penalize()``,
   which holds *every* thread until the window reopens, then the bucket drains them through
   at ``requests_per_second`` instead of releasing a thundering herd.

2. Retry classification: not all failures deserve the same treatment. The key distinction
   is ``billed_risk`` — whether the request may have reached the provider and been billed
   before failing. Long non-streaming generations that die late (read timeouts, gateway
   500/504/524 after upstream completion) were the source of a measured +63% billing
   premium: each blind retry regenerated an already-billed response. Those classes are
   capped at EXPENSIVE_MAX_ATTEMPTS total tries; cheap rejections (429/502/503, connection
   failures before a response) retry freely; permanent 4xx errors never retry.
"""

import random
import threading
import time
from email.utils import parsedate_to_datetime

import requests

# Status codes where the request was rejected before generation — retrying costs nothing.
RETRIABLE_CHEAP = {429, 502, 503}
# Status codes (and read timeouts) where the provider may have generated — and billed —
# a response before the failure reached us. Retry at most EXPENSIVE_MAX_ATTEMPTS total.
RETRIABLE_EXPENSIVE = {500, 504, 524}
# Permanent client errors: retrying can never succeed.
FATAL_4XX = {400, 401, 402, 403, 404, 405, 413, 422}

EXPENSIVE_MAX_ATTEMPTS = 2
BACKOFF_CAP_SECONDS = 60.0


class RateLimiter:
    """Thread-shared token bucket with a global penalty gate.

    ``acquire()`` blocks until a token is available AND any penalty window has passed.
    ``penalize(seconds)`` pushes the shared reopen time forward (monotonic max, so
    concurrent penalties don't shorten each other).
    """

    def __init__(self, requests_per_second: float, burst: int):
        self.requests_per_second = max(requests_per_second, 0.01)
        self.burst = max(burst, 1)
        self._lock = threading.Lock()
        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._hold_until = 0.0

    def acquire(self) -> None:
        """Block until a request may start."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Refill the bucket for elapsed time, capped at burst size.
                self._tokens = min(self.burst, self._tokens + (now - self._last_refill) * self.requests_per_second)
                self._last_refill = now

                if now >= self._hold_until and self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Compute how long to sleep outside the lock.
                penalty_wait = max(0.0, self._hold_until - now)
                token_wait = (1.0 - self._tokens) / self.requests_per_second if self._tokens < 1.0 else 0.0
                wait = max(penalty_wait, token_wait, 0.01)
            time.sleep(min(wait, 0.5))

    def penalize(self, seconds: float) -> None:
        """Hold all threads for ``seconds`` (e.g. from a 429's Retry-After)."""
        with self._lock:
            self._hold_until = max(self._hold_until, time.monotonic() + max(seconds, 0.0))


def parse_retry_after(headers) -> float | None:
    """Parse an HTTP Retry-After header: delta-seconds or HTTP-date. None if absent/invalid."""
    value = None
    if headers is not None:
        value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_time = parsedate_to_datetime(value)
        return max(0.0, retry_time.timestamp() - time.time())
    except (ValueError, TypeError):
        return None


def classify(response: requests.Response | None, exception: Exception | None) -> tuple[str, bool, bool]:
    """Classify one attempt outcome.

    Exactly one of ``response`` (a completed HTTP response with non-2xx status, or a 2xx
    that later failed parsing — pass None here and the exception instead) / ``exception``
    should be provided.

    Returns ``(error_class, retriable, billed_risk)``:
    - error_class: ok | http_429 | http_4xx | http_5xx | timeout | connection | parse | unknown
    - retriable: whether another attempt may help
    - billed_risk: whether the provider may have billed this attempt
    """
    if response is not None:
        status = response.status_code
        if 200 <= status < 300:
            return "ok", False, True
        if status == 429:
            return "http_429", True, False
        if status in FATAL_4XX or (400 <= status < 500 and status not in RETRIABLE_CHEAP):
            return "http_4xx", False, False
        if status in RETRIABLE_EXPENSIVE:
            return "http_5xx", True, True
        if status in RETRIABLE_CHEAP:
            return "http_5xx" if status != 429 else "http_429", True, False
        # Unknown 5xx and anything else server-side: treat as expensive-retriable.
        return "http_5xx", True, True

    if isinstance(exception, requests.Timeout):
        # Read timeout: the provider may have finished (and billed) the generation.
        return "timeout", True, True
    if isinstance(exception, requests.ConnectionError):
        return "connection", True, False
    if isinstance(exception, (ValueError, KeyError, TypeError, AttributeError)):
        # Parse/shape failure on a 2xx body: the response was billed and received;
        # retrying regenerates it. Do not retry — surface for debugging instead.
        return "parse", False, True
    return "unknown", True, True


def compute_backoff(attempt_number: int, retry_after_seconds: float | None) -> float:
    """Seconds to sleep before the next attempt.

    Honors Retry-After exactly when the server provided one; otherwise full-jitter
    exponential backoff capped at BACKOFF_CAP_SECONDS.
    """
    if retry_after_seconds is not None:
        return min(retry_after_seconds, 5 * BACKOFF_CAP_SECONDS)
    return random.uniform(0, min(BACKOFF_CAP_SECONDS, 2**attempt_number))
