"""Unit tests for the pure worker auth helper.

Kept deliberately light: no node loading, no server start, no env mutation —
just the pure :func:`authorize` decision function.
"""

from nodetool.worker.worker_auth import authorize


def test_no_token_allows_any():
    # Worker is open when no token is configured, regardless of the header.
    assert authorize(None, None) is True
    assert authorize(None, "") is True
    assert authorize("Bearer whatever", None) is True
    assert authorize("Bearer whatever", "") is True


def test_correct_bearer_allowed():
    assert authorize("Bearer s3cret", "s3cret") is True


def test_wrong_token_denied():
    assert authorize("Bearer nope", "s3cret") is False


def test_missing_header_denied():
    assert authorize(None, "s3cret") is False
    assert authorize("", "s3cret") is False


def test_malformed_header_denied():
    # No "Bearer " prefix.
    assert authorize("s3cret", "s3cret") is False
    # Wrong scheme.
    assert authorize("Basic s3cret", "s3cret") is False
    # Right scheme, wrong case on the scheme is not accepted.
    assert authorize("bearer s3cret", "s3cret") is False
    # Extra whitespace / no value.
    assert authorize("Bearer ", "s3cret") is False
    assert authorize("Bearer", "s3cret") is False
