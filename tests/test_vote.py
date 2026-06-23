"""Tests for the functional voting API in llm_multi_vote.vote.

These tests use ``pytest`` features (``pytest.approx``, ``pytest.raises``).
When the suite is collected by the standard-library ``unittest`` runner in
an environment without pytest installed, the import below would otherwise
abort discovery; ``SkipTest`` makes the module skip cleanly instead. The
equivalent assertions are covered by ``test_unittest_suite.py`` using only
the standard library, and the full pytest run remains unaffected.
"""

import asyncio
import os
import sys

# Make the src-layout package importable without an editable install (see
# tests/test_unittest_suite.py for the rationale).
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import pytest
except ImportError as exc:  # pragma: no cover - only hit without pytest
    import unittest

    raise unittest.SkipTest("pytest not installed") from exc

from llm_multi_vote.vote import (  # noqa: E402  (import after sys.path bootstrap)
    Strategy,
    VoteResult,
    default_normalizer,
    vote,
    vote_async,
)


def _const(value):
    """Build a voter callable that ignores the prompt and returns `value`."""
    return lambda _prompt: value


def test_default_normalizer():
    assert default_normalizer("  Yes  ") == "yes"
    assert default_normalizer("NO") == "no"


def test_majority_winner():
    voters = [("a", _const("yes")), ("b", _const("yes")), ("c", _const("no"))]
    result = vote(voters, "q")
    assert result.winner == "yes"
    assert result.consensus is False
    assert result.confidence == pytest.approx(2 / 3)


def test_majority_no_winner_on_tie():
    voters = [("a", _const("yes")), ("b", _const("no"))]
    result = vote(voters, "q")
    assert result.winner is None
    assert result.confidence == 0.0


def test_majority_normalizes_before_counting():
    voters = [("a", _const("Yes")), ("b", _const("YES ")), ("c", _const("no"))]
    result = vote(voters, "q")
    assert result.winner == "yes"


def test_plurality_picks_most_common():
    voters = [
        ("a", _const("x")),
        ("b", _const("x")),
        ("c", _const("y")),
        ("d", _const("z")),
    ]
    result = vote(voters, "q", strategy=Strategy.PLURALITY)
    assert result.winner == "x"


def test_plurality_none_on_tie_for_first():
    voters = [("a", _const("x")), ("b", _const("y"))]
    result = vote(voters, "q", strategy=Strategy.PLURALITY)
    assert result.winner is None


def test_unanimous_all_agree():
    voters = [("a", _const("yes")), ("b", _const("yes"))]
    result = vote(voters, "q", strategy=Strategy.UNANIMOUS)
    assert result.winner == "yes"
    assert result.consensus is True


def test_unanimous_fails_with_disagreement():
    voters = [("a", _const("yes")), ("b", _const("no"))]
    result = vote(voters, "q", strategy=Strategy.UNANIMOUS)
    assert result.winner is None


def test_min_threshold_met():
    voters = [("a", _const("yes")), ("b", _const("yes")), ("c", _const("no"))]
    result = vote(voters, "q", strategy=Strategy.MIN_THRESHOLD, min_votes_required=2)
    assert result.winner == "yes"


def test_min_threshold_not_met():
    voters = [("a", _const("yes")), ("b", _const("no"))]
    result = vote(voters, "q", strategy=Strategy.MIN_THRESHOLD, min_votes_required=2)
    assert result.winner is None


def test_min_threshold_requires_kwarg():
    voters = [("a", _const("yes"))]
    with pytest.raises(ValueError):
        vote(voters, "q", strategy=Strategy.MIN_THRESHOLD)


def test_failing_voter_is_captured_not_raised():
    def boom(_prompt):
        raise RuntimeError("backend down")

    voters = [("a", _const("yes")), ("b", _const("yes")), ("c", boom)]
    result = vote(voters, "q")
    assert result.winner == "yes"
    assert "c" in result.failures
    assert isinstance(result.failures["c"], RuntimeError)
    # failures count in the denominator: 2 of 3 -> 0.667
    assert result.confidence == pytest.approx(2 / 3)
    assert result.consensus is False


def test_no_voters_returns_empty_result():
    result = vote([], "q")
    assert result.winner is None
    assert result.votes == {}
    assert result.confidence == 0.0
    assert isinstance(result, VoteResult)


def test_all_responses_preserves_raw_text():
    voters = [("a", _const("Yes!")), ("b", _const("yes!"))]
    result = vote(voters, "q", strategy=Strategy.UNANIMOUS)
    assert result.all_responses == {"a": "Yes!", "b": "yes!"}


def test_custom_normalizer():
    voters = [("a", _const("AB")), ("b", _const("ba"))]
    # collapse to a sorted set of characters so "AB" and "ba" agree
    result = vote(
        voters,
        "q",
        strategy=Strategy.UNANIMOUS,
        normalizer=lambda s: "".join(sorted(s.lower())),
    )
    assert result.winner == "ab"


def test_vote_async_mixes_sync_and_async_voters():
    async def async_voter(_prompt):
        return "yes"

    voters = [("a", async_voter), ("b", _const("yes")), ("c", _const("no"))]
    result = asyncio.run(vote_async(voters, "q"))
    assert result.winner == "yes"


def test_vote_async_sequential_mode():
    voters = [("a", _const("yes")), ("b", _const("yes"))]
    result = asyncio.run(vote_async(voters, "q", parallel=False))
    assert result.winner == "yes"


def test_vote_async_captures_failures():
    async def boom(_prompt):
        raise RuntimeError("oops")

    voters = [("a", _const("yes")), ("b", boom)]
    result = asyncio.run(vote_async(voters, "q", strategy=Strategy.PLURALITY))
    assert result.winner == "yes"
    assert "b" in result.failures
