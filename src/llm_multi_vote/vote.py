"""Core voting implementation."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

# Each voter is a (name, callable) pair. The callable takes the prompt and
# returns a string answer (sync or coroutine for the async path).
SyncVoterFn = Callable[[str], str]
AsyncVoterFn = Callable[[str], Awaitable[str]]
Voter = tuple[str, SyncVoterFn | AsyncVoterFn]

# Normalizer canonicalizes raw answers so "Yes."/"yes"/"YES" collapse to one
# bucket before counting.
Normalizer = Callable[[str], str]


def default_normalizer(s: str) -> str:
    """Default canonicalizer: strip whitespace and lower-case."""
    return s.strip().lower()


class Strategy(Enum):
    """Aggregation rule applied to the normalized voter responses.

    Values:
      * MAJORITY: winner needs strictly more than half of the non-failing
        votes. Returns None on ties (50/50, three-way split, etc).
      * PLURALITY: winner is whichever bucket has the most votes. Returns
        None only when there are zero successful voters or there is a tie
        for first place.
      * UNANIMOUS: every non-failing voter must produce the same canonical
        answer. Otherwise returns None.
      * MIN_THRESHOLD: winner needs at least `min_votes_required` votes in
        a single bucket. Pair with `min_votes_required=` kwarg.
    """

    MAJORITY = "majority"
    PLURALITY = "plurality"
    UNANIMOUS = "unanimous"
    MIN_THRESHOLD = "min_threshold"


@dataclass(frozen=True)
class VoteResult:
    """Outcome of a vote() call.

    Attributes:
      * winner: canonical answer that won under the chosen strategy, or
        None if no winner met the bar.
      * votes: canonical answer -> list of voter names that produced it.
        Excludes failed voters.
      * confidence: winner_count / total_voters (counts FAILED voters in
        the denominator so a 2/3 majority of three voters is 0.667 even
        if the third one crashed). 0.0 when winner is None.
      * consensus: True when every non-failing voter agreed (single bucket
        and zero failures). Cheaper than checking len(votes) == 1.
      * all_responses: voter name -> raw (un-normalized) response string,
        for callers that want to inspect actual outputs.
      * failures: voter name -> Exception that voter raised. Failing voters
        are excluded from the vote tally but still appear here so callers
        can log / retry.
    """

    winner: str | None
    votes: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0
    consensus: bool = False
    all_responses: dict[str, str] = field(default_factory=dict)
    failures: dict[str, Exception] = field(default_factory=dict)


# ---- sync ----


def vote(
    voters: Sequence[Voter],
    prompt: str,
    strategy: Strategy = Strategy.MAJORITY,
    *,
    normalizer: Normalizer | None = None,
    min_votes_required: int | None = None,
) -> VoteResult:
    """Run `prompt` through each voter sequentially and aggregate.

    Args:
      voters: list of (name, callable) tuples. Names should be unique;
        duplicates last-write-wins.
      prompt: the prompt string passed to every voter unchanged.
      strategy: one of Strategy.MAJORITY (default), PLURALITY, UNANIMOUS,
        MIN_THRESHOLD.
      normalizer: optional callable str -> str to canonicalize answers
        before counting. Defaults to strip + lower.
      min_votes_required: required when strategy is MIN_THRESHOLD; ignored
        otherwise.

    Returns:
      VoteResult.

    A voter that raises is captured in `failures` and excluded from the
    tally. This function itself does not raise on voter failures; it only
    raises on caller-side mistakes such as missing min_votes_required.
    """
    if strategy is Strategy.MIN_THRESHOLD and min_votes_required is None:
        raise ValueError("Strategy.MIN_THRESHOLD requires min_votes_required")

    norm = normalizer or default_normalizer
    all_responses: dict[str, str] = {}
    failures: dict[str, Exception] = {}
    buckets: dict[str, list[str]] = {}

    for name, fn in voters:
        try:
            raw = fn(prompt)
        except Exception as exc:  # noqa: BLE001 - voter failures must be captured
            failures[name] = exc
            continue
        all_responses[name] = raw
        canonical = norm(raw)
        buckets.setdefault(canonical, []).append(name)

    return _aggregate(
        buckets=buckets,
        all_responses=all_responses,
        failures=failures,
        total_voters=len(voters),
        strategy=strategy,
        min_votes_required=min_votes_required,
    )


# ---- async ----


async def vote_async(
    voters: Sequence[Voter],
    prompt: str,
    strategy: Strategy = Strategy.MAJORITY,
    *,
    normalizer: Normalizer | None = None,
    min_votes_required: int | None = None,
    parallel: bool = True,
) -> VoteResult:
    """Async version of `vote`. Each voter callable may be sync OR async.

    By default voters run concurrently via `asyncio.gather`. Pass
    `parallel=False` to run them sequentially (useful for rate-limit
    sensitive backends).

    Sync voter callables are awaited via `asyncio.to_thread` so they do
    not block the loop. Voters that raise are captured in `failures`.
    """
    if strategy is Strategy.MIN_THRESHOLD and min_votes_required is None:
        raise ValueError("Strategy.MIN_THRESHOLD requires min_votes_required")

    norm = normalizer or default_normalizer

    async def _invoke(
        name: str, fn: SyncVoterFn | AsyncVoterFn
    ) -> tuple[str, str | Exception]:
        try:
            if inspect.iscoroutinefunction(fn):
                raw = await fn(prompt)
            else:
                # offload sync voters so a blocking voter does not stall the loop
                raw = await asyncio.to_thread(fn, prompt)
            return name, raw
        except Exception as exc:  # noqa: BLE001 - voter failures must be captured
            return name, exc

    if parallel:
        results = await asyncio.gather(*(_invoke(name, fn) for name, fn in voters))
    else:
        results = [await _invoke(name, fn) for name, fn in voters]

    all_responses: dict[str, str] = {}
    failures: dict[str, Exception] = {}
    buckets: dict[str, list[str]] = {}

    for name, outcome in results:
        if isinstance(outcome, Exception):
            failures[name] = outcome
            continue
        all_responses[name] = outcome
        canonical = norm(outcome)
        buckets.setdefault(canonical, []).append(name)

    return _aggregate(
        buckets=buckets,
        all_responses=all_responses,
        failures=failures,
        total_voters=len(voters),
        strategy=strategy,
        min_votes_required=min_votes_required,
    )


# ---- internal: shared aggregation ----


def _aggregate(
    *,
    buckets: dict[str, list[str]],
    all_responses: dict[str, str],
    failures: dict[str, Exception],
    total_voters: int,
    strategy: Strategy,
    min_votes_required: int | None,
) -> VoteResult:
    if not buckets:
        return VoteResult(
            winner=None,
            votes={},
            confidence=0.0,
            consensus=False,
            all_responses=all_responses,
            failures=failures,
        )

    # Build sorted view: highest count first, name asc for deterministic ties.
    ranked = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    top_answer, top_voters = ranked[0]
    top_count = len(top_voters)
    successful_votes = sum(len(v) for v in buckets.values())

    winner: str | None = None

    if strategy is Strategy.MAJORITY:
        # strictly more than half of the SUCCESSFUL votes
        if successful_votes > 0 and top_count * 2 > successful_votes:
            winner = top_answer

    elif strategy is Strategy.PLURALITY:
        # most common; None on tie for first
        if len(ranked) == 1 or len(ranked[1][1]) < top_count:
            winner = top_answer

    elif strategy is Strategy.UNANIMOUS:
        # exactly one bucket and zero failures
        if len(buckets) == 1 and not failures:
            winner = top_answer

    elif strategy is Strategy.MIN_THRESHOLD:
        assert min_votes_required is not None  # validated at entry
        if top_count >= min_votes_required:
            winner = top_answer

    # confidence is winner_count / total_voters (failures count in denom).
    # If no winner, confidence is 0.0.
    confidence = (
        top_count / total_voters if winner is not None and total_voters > 0 else 0.0
    )

    # consensus: every voter ran AND they all agreed
    consensus = len(buckets) == 1 and not failures and successful_votes == total_voters

    return VoteResult(
        winner=winner,
        votes=buckets,
        confidence=confidence,
        consensus=consensus,
        all_responses=all_responses,
        failures=failures,
    )
