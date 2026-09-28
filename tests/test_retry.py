from __future__ import annotations

import pytest

from nexusagent.core.retry import is_transient_error, retry_transient


class FakeRateLimitError(Exception):
    pass


class FakeServiceUnavailableError(Exception):
    pass


def test_is_transient_error_matches_network_and_rate_limit_failures() -> None:
    assert is_transient_error(FakeRateLimitError("Rate limit reached")) is True
    assert is_transient_error(TimeoutError("request timed out")) is True
    assert is_transient_error(ConnectionError("connection reset by peer")) is True
    assert is_transient_error(FakeServiceUnavailableError("503")) is True
    assert is_transient_error(Exception("503 Service Unavailable")) is True


def test_is_transient_error_ignores_permanent_errors() -> None:
    assert is_transient_error(ValueError("invalid regex")) is False
    assert is_transient_error(FileNotFoundError("missing.txt")) is False
    assert is_transient_error(RuntimeError("missing required .env setting(s)")) is False


def test_is_transient_error_walks_exception_chain() -> None:
    cause = ConnectionError("connection closed")
    wrapped = RuntimeError("search failed")
    wrapped.__cause__ = cause

    assert is_transient_error(wrapped) is True


def test_retry_transient_succeeds_after_transient_failures() -> None:
    calls = {"count": 0}

    @retry_transient(attempts=3, initial_wait=0.01, max_wait=0.02)
    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("timed out")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3


def test_retry_transient_reraises_after_exhaustion() -> None:
    calls = {"count": 0}

    @retry_transient(attempts=2, initial_wait=0.01, max_wait=0.02)
    def always_transient() -> None:
        calls["count"] += 1
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        always_transient()
    assert calls["count"] == 2


def test_retry_transient_does_not_retry_permanent_errors() -> None:
    calls = {"count": 0}

    @retry_transient(attempts=3, initial_wait=0.01, max_wait=0.02)
    def broken() -> None:
        calls["count"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        broken()
    assert calls["count"] == 1
