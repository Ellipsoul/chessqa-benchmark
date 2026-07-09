"""Unit tests for eval/throttle.py — no network, no sleeps beyond fractions of a second."""

import threading
import time

import pytest
import requests

import throttle


class FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


# --- classify -----------------------------------------------------------------------------


def test_classify_success():
    assert throttle.classify(FakeResponse(200), None) == ("ok", False, True)


def test_classify_429_is_cheap_retriable():
    error_class, retriable, billed_risk = throttle.classify(FakeResponse(429), None)
    assert (error_class, retriable, billed_risk) == ("http_429", True, False)


@pytest.mark.parametrize("status", sorted(throttle.FATAL_4XX))
def test_classify_fatal_4xx_never_retries(status):
    error_class, retriable, billed_risk = throttle.classify(FakeResponse(status), None)
    assert (error_class, retriable, billed_risk) == ("http_4xx", False, False)


@pytest.mark.parametrize("status", [500, 504, 524])
def test_classify_expensive_5xx_has_billed_risk(status):
    error_class, retriable, billed_risk = throttle.classify(FakeResponse(status), None)
    assert (error_class, retriable, billed_risk) == ("http_5xx", True, True)


@pytest.mark.parametrize("status", [502, 503])
def test_classify_cheap_5xx_no_billed_risk(status):
    error_class, retriable, billed_risk = throttle.classify(FakeResponse(status), None)
    assert (error_class, retriable, billed_risk) == ("http_5xx", True, False)


def test_classify_read_timeout_is_expensive():
    error_class, retriable, billed_risk = throttle.classify(None, requests.Timeout("read timed out"))
    assert (error_class, retriable, billed_risk) == ("timeout", True, True)


def test_classify_connection_error_is_cheap():
    error_class, retriable, billed_risk = throttle.classify(None, requests.ConnectionError("refused"))
    assert (error_class, retriable, billed_risk) == ("connection", True, False)


def test_classify_parse_error_is_fatal_and_billed():
    error_class, retriable, billed_risk = throttle.classify(None, KeyError("choices"))
    assert (error_class, retriable, billed_risk) == ("parse", False, True)


# --- parse_retry_after --------------------------------------------------------------------


def test_retry_after_seconds():
    assert throttle.parse_retry_after({"Retry-After": "7"}) == 7.0


def test_retry_after_lowercase_header():
    assert throttle.parse_retry_after({"retry-after": "2.5"}) == 2.5


def test_retry_after_http_date():
    from email.utils import formatdate

    value = formatdate(time.time() + 30, usegmt=True)
    parsed = throttle.parse_retry_after({"Retry-After": value})
    assert parsed is not None and 25 <= parsed <= 31


def test_retry_after_absent_or_garbage():
    assert throttle.parse_retry_after({}) is None
    assert throttle.parse_retry_after(None) is None
    assert throttle.parse_retry_after({"Retry-After": "soon"}) is None


# --- compute_backoff ----------------------------------------------------------------------


def test_backoff_honors_retry_after_exactly():
    assert throttle.compute_backoff(1, 12.0) == 12.0


def test_backoff_jitter_bounded():
    for attempt_number in range(1, 12):
        wait = throttle.compute_backoff(attempt_number, None)
        assert 0 <= wait <= min(throttle.BACKOFF_CAP_SECONDS, 2**attempt_number)


# --- RateLimiter --------------------------------------------------------------------------


def test_limiter_paces_burst_exhaustion():
    """With burst=2 and rps=20, 6 acquires need ~4 refills => >= ~0.2s wall time."""
    limiter = throttle.RateLimiter(requests_per_second=20, burst=2)
    start = time.monotonic()
    for _ in range(6):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.15, f"limiter failed to pace: {elapsed:.3f}s"


def test_limiter_penalize_blocks_all_threads():
    limiter = throttle.RateLimiter(requests_per_second=100, burst=10)
    limiter.penalize(0.5)
    release_times = []

    def worker():
        limiter.acquire()
        release_times.append(time.monotonic())

    start = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(release_times) == 4
    assert min(release_times) - start >= 0.45, "acquire returned during penalty window"


def test_limiter_penalize_is_monotonic_max():
    limiter = throttle.RateLimiter(requests_per_second=100, burst=10)
    limiter.penalize(0.5)
    limiter.penalize(0.1)  # must NOT shorten the existing hold
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start >= 0.45
