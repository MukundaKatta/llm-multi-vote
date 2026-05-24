"""Tests for llm-multi-vote."""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_multi_vote import (
    Strategy,
    VoteResult,
    default_normalizer,
    vote,
    vote_async,
)


def _const(answer: str):
    """Build a voter that always returns `answer`."""

    def _fn(_prompt: str) -> str:
        return answer

    return _fn


# ---------- MAJORITY ----------


def test_majority_three_for_yes_two_for_no_returns_yes():
    voters = [
        ("a", _const("yes")),
        ("b", _const("yes")),
        ("c", _const("yes")),
        ("d", _const("no")),
        ("e", _const("no")),
    ]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY)
    assert r.winner == "yes"
    assert r.confidence == pytest.approx(3 / 5)
    assert r.votes == {"yes": ["a", "b", "c"], "no": ["d", "e"]}
    assert r.consensus is False


def test_majority_two_two_split_returns_none():
    voters = [
        ("a", _const("yes")),
        ("b", _const("yes")),
        ("c", _const("no")),
        ("d", _const("no")),
    ]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY)
    assert r.winner is None
    assert r.confidence == 0.0


def test_majority_needs_strictly_more_than_half():
    # 1 of 2 is not majority
    voters = [("a", _const("yes")), ("b", _const("no"))]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY)
    assert r.winner is None


def test_majority_single_voter_wins_trivially():
    r = vote([("a", _const("yes"))], "Q", strategy=Strategy.MAJORITY)
    assert r.winner == "yes"
    assert r.confidence == 1.0
    assert r.consensus is True


# ---------- PLURALITY ----------


def test_plurality_picks_most_common_from_three_way():
    voters = [
        ("a", _const("red")),
        ("b", _const("red")),
        ("c", _const("blue")),
        ("d", _const("green")),
    ]
    r = vote(voters, "Q", strategy=Strategy.PLURALITY)
    assert r.winner == "red"
    assert r.confidence == pytest.approx(2 / 4)


def test_plurality_ties_return_none():
    voters = [
        ("a", _const("red")),
        ("b", _const("blue")),
    ]
    r = vote(voters, "Q", strategy=Strategy.PLURALITY)
    assert r.winner is None


def test_plurality_single_bucket_wins():
    voters = [("a", _const("x")), ("b", _const("x"))]
    r = vote(voters, "Q", strategy=Strategy.PLURALITY)
    assert r.winner == "x"
    assert r.consensus is True


# ---------- UNANIMOUS ----------


def test_unanimous_all_agree_returns_winner():
    voters = [("a", _const("ok")), ("b", _const("ok")), ("c", _const("ok"))]
    r = vote(voters, "Q", strategy=Strategy.UNANIMOUS)
    assert r.winner == "ok"
    assert r.consensus is True
    assert r.confidence == 1.0


def test_unanimous_one_disagreement_returns_none():
    voters = [
        ("a", _const("ok")),
        ("b", _const("ok")),
        ("c", _const("nope")),
    ]
    r = vote(voters, "Q", strategy=Strategy.UNANIMOUS)
    assert r.winner is None


def test_unanimous_with_failure_returns_none():
    def boom(_):
        raise RuntimeError("provider down")

    voters = [("a", _const("ok")), ("b", _const("ok")), ("c", boom)]
    r = vote(voters, "Q", strategy=Strategy.UNANIMOUS)
    assert r.winner is None
    assert "c" in r.failures


# ---------- MIN_THRESHOLD ----------


def test_min_threshold_enforces_floor():
    voters = [
        ("a", _const("approve")),
        ("b", _const("approve")),
        ("c", _const("deny")),
    ]
    r = vote(voters, "Q", strategy=Strategy.MIN_THRESHOLD, min_votes_required=2)
    assert r.winner == "approve"

    r2 = vote(voters, "Q", strategy=Strategy.MIN_THRESHOLD, min_votes_required=3)
    assert r2.winner is None


def test_min_threshold_requires_kwarg():
    with pytest.raises(ValueError):
        vote([("a", _const("x"))], "Q", strategy=Strategy.MIN_THRESHOLD)


# ---------- normalizer ----------


def test_default_normalizer_collapses_variants():
    voters = [
        ("a", _const("Yes.")),
        ("b", _const("yes")),
        ("c", _const("YES")),
    ]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY)
    # default normalizer strips/lowercases but does NOT strip punctuation
    # "Yes." -> "yes." and "yes" -> "yes" are different. Show that with the
    # custom normalizer test below they collapse.
    # Confirm here that the default normalizer at least handles case+whitespace.
    assert "yes" in r.votes  # b and c collapse
    assert len(r.votes["yes"]) == 2


def test_custom_normalizer_strips_punctuation():
    def strip_punct(s: str) -> str:
        return s.strip().lower().rstrip(".!?")

    voters = [
        ("a", _const("Yes.")),
        ("b", _const("yes")),
        ("c", _const("YES")),
    ]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY, normalizer=strip_punct)
    assert r.winner == "yes"
    assert r.votes == {"yes": ["a", "b", "c"]}
    assert r.consensus is True
    assert r.confidence == 1.0


def test_default_normalizer_exposed():
    assert default_normalizer("  HELLO  ") == "hello"


# ---------- failures ----------


def test_voter_exception_captured_and_excluded():
    def boom(_):
        raise ValueError("api error")

    voters = [
        ("a", _const("yes")),
        ("b", _const("yes")),
        ("c", boom),
    ]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY)
    assert r.winner == "yes"
    # c was excluded from the tally
    assert "c" not in [n for names in r.votes.values() for n in names]
    # but recorded in failures
    assert "c" in r.failures
    assert isinstance(r.failures["c"], ValueError)
    # raw responses only include successful voters
    assert "c" not in r.all_responses
    assert r.all_responses == {"a": "yes", "b": "yes"}
    # confidence uses total voters in denom (2/3)
    assert r.confidence == pytest.approx(2 / 3)


def test_all_voters_fail_returns_no_winner():
    def boom(_):
        raise RuntimeError("nope")

    voters = [("a", boom), ("b", boom)]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY)
    assert r.winner is None
    assert r.confidence == 0.0
    assert r.all_responses == {}
    assert len(r.failures) == 2


# ---------- VoteResult shape ----------


def test_vote_result_is_frozen():
    from dataclasses import FrozenInstanceError

    r = vote([("a", _const("x"))], "Q", strategy=Strategy.MAJORITY)
    assert isinstance(r, VoteResult)
    with pytest.raises(FrozenInstanceError):
        r.winner = "nope"  # type: ignore[misc]


def test_all_responses_preserves_raw_strings():
    voters = [("a", _const("  Yes!  ")), ("b", _const("YES"))]
    r = vote(voters, "Q", strategy=Strategy.MAJORITY)
    # raw response kept untouched
    assert r.all_responses["a"] == "  Yes!  "
    assert r.all_responses["b"] == "YES"


def test_empty_voters_returns_no_winner():
    r = vote([], "Q", strategy=Strategy.MAJORITY)
    assert r.winner is None
    assert r.confidence == 0.0
    assert r.votes == {}


# ---------- async ----------


async def test_async_basic_majority():
    async def yes_voter(_p):
        return "yes"

    async def no_voter(_p):
        return "no"

    r = await vote_async(
        [("a", yes_voter), ("b", yes_voter), ("c", no_voter)],
        "Q",
        strategy=Strategy.MAJORITY,
    )
    assert r.winner == "yes"
    assert r.confidence == pytest.approx(2 / 3)


async def test_async_runs_voters_in_parallel():
    async def slow(_p: str) -> str:
        await asyncio.sleep(0.1)
        return "ok"

    voters = [("a", slow), ("b", slow), ("c", slow)]
    start = time.monotonic()
    r = await vote_async(voters, "Q", strategy=Strategy.UNANIMOUS)
    elapsed = time.monotonic() - start
    assert r.winner == "ok"
    # parallel: three 0.1s sleeps complete in well under 0.3s.
    # Allow some scheduling jitter but the upper bound rules out sequential.
    assert elapsed < 0.25, f"voters ran sequentially? elapsed={elapsed:.3f}s"


async def test_async_sequential_takes_longer():
    async def slow(_p: str) -> str:
        await asyncio.sleep(0.05)
        return "ok"

    voters = [("a", slow), ("b", slow), ("c", slow)]
    start = time.monotonic()
    await vote_async(voters, "Q", strategy=Strategy.UNANIMOUS, parallel=False)
    elapsed = time.monotonic() - start
    # Sequential 3 * 0.05 = 0.15. Allow jitter.
    assert elapsed >= 0.13


async def test_async_mixes_sync_and_async_voters():
    async def async_yes(_p):
        return "yes"

    def sync_yes(_p):
        return "yes"

    r = await vote_async(
        [("a", async_yes), ("b", sync_yes)],
        "Q",
        strategy=Strategy.UNANIMOUS,
    )
    assert r.winner == "yes"
    assert r.consensus is True


async def test_async_voter_exception_captured():
    async def boom(_p):
        raise RuntimeError("oops")

    async def ok(_p):
        return "yes"

    r = await vote_async(
        [("a", ok), ("b", ok), ("c", boom)],
        "Q",
        strategy=Strategy.MAJORITY,
    )
    assert r.winner == "yes"
    assert "c" in r.failures
    assert r.confidence == pytest.approx(2 / 3)


async def test_async_min_threshold_requires_kwarg():
    async def yes(_p):
        return "yes"

    with pytest.raises(ValueError):
        await vote_async([("a", yes)], "Q", strategy=Strategy.MIN_THRESHOLD)
