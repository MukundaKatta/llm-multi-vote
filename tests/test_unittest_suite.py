"""Standard-library ``unittest`` suite for llm-multi-vote.

This mirrors the pytest-based tests but uses only the standard library so
the suite can run anywhere with::

    python3 -m unittest discover -s tests

It exercises both public surfaces: the :class:`MultiVote` builder API and
the functional :func:`vote` / :func:`vote_async` API.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

# Make the ``src/`` layout package importable when the suite is run with the
# bare ``python3 -m unittest discover -s tests`` command from the repo root
# (which sets top-level-dir to ``tests`` and so skips ``tests/__init__``).
# An installed/editable package already on ``sys.path`` takes precedence.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_multi_vote import (  # noqa: E402  (import after sys.path bootstrap)
    Ballot,
    MultiVote,
    Strategy,
    VotingStrategy,
    default_normalizer,
    vote,
    vote_async,
)


def _const(value):
    """Build a voter callable that ignores the prompt and returns ``value``."""
    return lambda _prompt: value


class MultiVoteBuilderTests(unittest.TestCase):
    def test_majority_winner(self) -> None:
        mv = MultiVote()
        mv.add("a", "Paris").add("b", "Paris").add("c", "Lyon")
        result = mv.vote()
        self.assertEqual(result.winner, "Paris")
        self.assertEqual(result.strategy, VotingStrategy.MAJORITY.value)
        self.assertFalse(result.tied)

    def test_majority_tied(self) -> None:
        mv = MultiVote()
        mv.add("a", "Paris").add("b", "Lyon")
        self.assertTrue(mv.vote().tied)

    def test_majority_unanimous(self) -> None:
        mv = MultiVote()
        mv.add("a", "Paris").add("b", "Paris").add("c", "Paris")
        result = mv.vote()
        self.assertTrue(result.unanimous)
        self.assertEqual(result.winner, "Paris")

    def test_majority_case_insensitive(self) -> None:
        mv = MultiVote(normalize=True)
        mv.add("a", "Paris").add("b", "paris").add("c", "Lyon")
        result = mv.vote()
        self.assertIsNotNone(result.winner)
        assert result.winner is not None  # narrow for type checkers
        self.assertEqual(result.winner.lower(), "paris")

    def test_normalize_disabled_keeps_variants_distinct(self) -> None:
        mv = MultiVote(normalize=False)
        mv.add("a", "Paris").add("b", "paris").add("c", "Lyon")
        # With normalization off, no response repeats, so it is a 3-way tie.
        self.assertTrue(mv.vote().tied)

    def test_longest_and_shortest(self) -> None:
        mv = MultiVote(strategy=VotingStrategy.LONGEST)
        mv.add("a", "short").add("b", "this is much longer")
        self.assertEqual(mv.vote().winner, "this is much longer")
        self.assertEqual(
            mv.vote(strategy=VotingStrategy.SHORTEST).winner, "short"
        )

    def test_first_strategy(self) -> None:
        mv = MultiVote(strategy=VotingStrategy.FIRST)
        mv.add("a", "first").add("b", "second")
        self.assertEqual(mv.vote().winner, "first")

    def test_scored_with_explicit_scores(self) -> None:
        mv = MultiVote(strategy=VotingStrategy.SCORED)
        mv.add("a", "ok", score=0.5)
        mv.add("b", "great", score=0.9)
        result = mv.vote()
        self.assertEqual(result.winner, "great")
        self.assertEqual(result.scores["b"], 0.9)

    def test_scored_with_scorer(self) -> None:
        mv = MultiVote(strategy=VotingStrategy.SCORED, scorer=lambda r: len(r))
        mv.add("a", "hi").add("b", "hello world")
        self.assertEqual(mv.vote().winner, "hello world")

    def test_scored_duplicate_model_names(self) -> None:
        # The higher-scored response must win even when model names collide.
        mv = MultiVote(strategy=VotingStrategy.SCORED)
        mv.add("dup", "low", score=0.1)
        mv.add("dup", "high", score=0.9)
        self.assertEqual(mv.vote().winner, "high")

    def test_scored_explicit_zero_beats_scorer(self) -> None:
        mv = MultiVote(strategy=VotingStrategy.SCORED, scorer=lambda r: 100.0)
        mv.add("a", "explicit-zero", score=0.0)
        mv.add("b", "scored-default")
        result = mv.vote()
        self.assertEqual(result.winner, "scored-default")
        self.assertEqual(result.scores["a"], 0.0)

    def test_consensus_agreement_and_disagreement(self) -> None:
        agree = MultiVote(strategy=VotingStrategy.CONSENSUS)
        agree.add("a", "answer").add("b", "answer")
        agreed = agree.vote()
        self.assertEqual(agreed.winner, "answer")
        self.assertTrue(agreed.unanimous)

        disagree = MultiVote(strategy=VotingStrategy.CONSENSUS)
        disagree.add("a", "answer1").add("b", "answer2")
        disagreed = disagree.vote()
        self.assertIsNone(disagreed.winner)
        self.assertTrue(disagreed.tied)

    def test_empty_vote_for_every_strategy(self) -> None:
        for strategy in VotingStrategy:
            with self.subTest(strategy=strategy):
                self.assertIsNone(MultiVote(strategy=strategy).vote().winner)

    def test_ballot_count_and_clear(self) -> None:
        mv = MultiVote()
        mv.add("a", "x").add("b", "y")
        self.assertEqual(mv.ballot_count, 2)
        mv.clear()
        self.assertEqual(mv.ballot_count, 0)

    def test_result_models_and_ballot_count(self) -> None:
        mv = MultiVote()
        mv.add("gpt4", "x").add("claude", "y")
        result = mv.vote()
        self.assertEqual(result.models, ["gpt4", "claude"])
        self.assertEqual(result.ballot_count, 2)

    def test_from_responses(self) -> None:
        result = MultiVote.from_responses({"a": "yes", "b": "yes", "c": "no"})
        self.assertEqual(result.winner, "yes")

    def test_strategy_override_at_vote_time(self) -> None:
        mv = MultiVote(strategy=VotingStrategy.MAJORITY)
        mv.add("a", "short").add("b", "this is very long indeed")
        result = mv.vote(strategy=VotingStrategy.LONGEST)
        self.assertEqual(result.winner, "this is very long indeed")
        # Overriding must not mutate the instance default.
        self.assertEqual(mv.vote().strategy, VotingStrategy.MAJORITY.value)

    def test_ballot_normalized(self) -> None:
        self.assertEqual(
            Ballot(model="x", response="  Hello   World  ").normalized(),
            "hello world",
        )


class FunctionalVoteTests(unittest.TestCase):
    def test_default_normalizer(self) -> None:
        self.assertEqual(default_normalizer("  Yes  "), "yes")
        self.assertEqual(default_normalizer("NO"), "no")

    def test_majority_winner_and_confidence(self) -> None:
        voters = [("a", _const("yes")), ("b", _const("yes")), ("c", _const("no"))]
        result = vote(voters, "q")
        self.assertEqual(result.winner, "yes")
        self.assertFalse(result.consensus)
        self.assertAlmostEqual(result.confidence, 2 / 3)

    def test_majority_no_winner_on_tie(self) -> None:
        result = vote([("a", _const("yes")), ("b", _const("no"))], "q")
        self.assertIsNone(result.winner)
        self.assertEqual(result.confidence, 0.0)

    def test_majority_normalizes_before_counting(self) -> None:
        voters = [("a", _const("Yes")), ("b", _const("YES ")), ("c", _const("no"))]
        self.assertEqual(vote(voters, "q").winner, "yes")

    def test_plurality(self) -> None:
        voters = [
            ("a", _const("x")),
            ("b", _const("x")),
            ("c", _const("y")),
            ("d", _const("z")),
        ]
        self.assertEqual(vote(voters, "q", strategy=Strategy.PLURALITY).winner, "x")

    def test_plurality_none_on_tie_for_first(self) -> None:
        voters = [("a", _const("x")), ("b", _const("y"))]
        self.assertIsNone(vote(voters, "q", strategy=Strategy.PLURALITY).winner)

    def test_unanimous(self) -> None:
        agree = [("a", _const("yes")), ("b", _const("yes"))]
        result = vote(agree, "q", strategy=Strategy.UNANIMOUS)
        self.assertEqual(result.winner, "yes")
        self.assertTrue(result.consensus)

        disagree = [("a", _const("yes")), ("b", _const("no"))]
        self.assertIsNone(vote(disagree, "q", strategy=Strategy.UNANIMOUS).winner)

    def test_min_threshold(self) -> None:
        met = [("a", _const("yes")), ("b", _const("yes")), ("c", _const("no"))]
        self.assertEqual(
            vote(
                met, "q", strategy=Strategy.MIN_THRESHOLD, min_votes_required=2
            ).winner,
            "yes",
        )
        not_met = [("a", _const("yes")), ("b", _const("no"))]
        self.assertIsNone(
            vote(
                not_met, "q", strategy=Strategy.MIN_THRESHOLD, min_votes_required=2
            ).winner
        )

    def test_min_threshold_requires_kwarg(self) -> None:
        with self.assertRaises(ValueError):
            vote([("a", _const("yes"))], "q", strategy=Strategy.MIN_THRESHOLD)

    def test_failing_voter_is_captured_not_raised(self) -> None:
        def boom(_prompt):
            raise RuntimeError("backend down")

        voters = [("a", _const("yes")), ("b", _const("yes")), ("c", boom)]
        result = vote(voters, "q")
        self.assertEqual(result.winner, "yes")
        self.assertIn("c", result.failures)
        self.assertIsInstance(result.failures["c"], RuntimeError)
        # Failures count in the denominator: 2 of 3 -> 0.667.
        self.assertAlmostEqual(result.confidence, 2 / 3)
        self.assertFalse(result.consensus)

    def test_no_voters_returns_empty_result(self) -> None:
        result = vote([], "q")
        self.assertIsNone(result.winner)
        self.assertEqual(result.votes, {})
        self.assertEqual(result.confidence, 0.0)

    def test_all_responses_preserves_raw_text(self) -> None:
        voters = [("a", _const("Yes!")), ("b", _const("yes!"))]
        result = vote(voters, "q", strategy=Strategy.UNANIMOUS)
        self.assertEqual(result.all_responses, {"a": "Yes!", "b": "yes!"})

    def test_custom_normalizer(self) -> None:
        voters = [("a", _const("AB")), ("b", _const("ba"))]
        result = vote(
            voters,
            "q",
            strategy=Strategy.UNANIMOUS,
            normalizer=lambda s: "".join(sorted(s.lower())),
        )
        self.assertEqual(result.winner, "ab")

    def test_prompt_is_forwarded_to_voters(self) -> None:
        seen: list[str] = []

        def recorder(prompt):
            seen.append(prompt)
            return "ok"

        vote([("a", recorder), ("b", recorder)], "the-prompt")
        self.assertEqual(seen, ["the-prompt", "the-prompt"])


class FunctionalVoteAsyncTests(unittest.TestCase):
    def test_mixes_sync_and_async_voters(self) -> None:
        async def async_voter(_prompt):
            return "yes"

        voters = [("a", async_voter), ("b", _const("yes")), ("c", _const("no"))]
        result = asyncio.run(vote_async(voters, "q"))
        self.assertEqual(result.winner, "yes")

    def test_sequential_mode(self) -> None:
        voters = [("a", _const("yes")), ("b", _const("yes"))]
        result = asyncio.run(vote_async(voters, "q", parallel=False))
        self.assertEqual(result.winner, "yes")

    def test_captures_failures(self) -> None:
        async def boom(_prompt):
            raise RuntimeError("oops")

        voters = [("a", _const("yes")), ("b", boom)]
        result = asyncio.run(
            vote_async(voters, "q", strategy=Strategy.PLURALITY)
        )
        self.assertEqual(result.winner, "yes")
        self.assertIn("b", result.failures)

    def test_async_matches_sync_for_same_inputs(self) -> None:
        voters = [("a", _const("yes")), ("b", _const("yes")), ("c", _const("no"))]
        sync_result = vote(voters, "q")
        async_result = asyncio.run(vote_async(voters, "q"))
        self.assertEqual(sync_result.winner, async_result.winner)
        self.assertAlmostEqual(sync_result.confidence, async_result.confidence)


if __name__ == "__main__":
    unittest.main()
