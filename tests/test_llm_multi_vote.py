"""Tests for llm-multi-vote."""

import os
import sys

# Make the src-layout package importable without an editable install (see
# tests/test_unittest_suite.py for the rationale).
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_multi_vote import MultiVote, Ballot, VotingStrategy  # noqa: E402


def test_majority_winner():
    mv = MultiVote()
    mv.add("a", "Paris").add("b", "Paris").add("c", "Lyon")
    result = mv.vote()
    assert result.winner == "Paris"
    assert result.strategy == VotingStrategy.MAJORITY.value


def test_majority_tied():
    mv = MultiVote()
    mv.add("a", "Paris").add("b", "Lyon")
    result = mv.vote()
    assert result.tied is True


def test_majority_unanimous():
    mv = MultiVote()
    mv.add("a", "Paris").add("b", "Paris").add("c", "Paris")
    result = mv.vote()
    assert result.unanimous is True


def test_majority_case_insensitive():
    mv = MultiVote(normalize=True)
    mv.add("a", "Paris").add("b", "paris").add("c", "Lyon")
    result = mv.vote()
    # Both normalize to "paris", so they tie unless one has more
    # 2 "paris" variants vs 1 "lyon" — majority wins
    assert result.winner is not None
    assert result.winner.lower() == "paris"


def test_longest_strategy():
    mv = MultiVote(strategy=VotingStrategy.LONGEST)
    mv.add("a", "short").add("b", "this is much longer")
    result = mv.vote()
    assert result.winner == "this is much longer"


def test_shortest_strategy():
    mv = MultiVote(strategy=VotingStrategy.SHORTEST)
    mv.add("a", "short").add("b", "this is much longer")
    result = mv.vote()
    assert result.winner == "short"


def test_first_strategy():
    mv = MultiVote(strategy=VotingStrategy.FIRST)
    mv.add("a", "first").add("b", "second")
    result = mv.vote()
    assert result.winner == "first"


def test_scored_strategy_with_scores():
    mv = MultiVote(strategy=VotingStrategy.SCORED)
    mv.add("a", "ok", score=0.5)
    mv.add("b", "great", score=0.9)
    result = mv.vote()
    assert result.winner == "great"
    assert result.scores["b"] == 0.9


def test_scored_strategy_with_scorer():
    mv = MultiVote(strategy=VotingStrategy.SCORED, scorer=lambda r: len(r))
    mv.add("a", "hi")
    mv.add("b", "hello world")
    result = mv.vote()
    assert result.winner == "hello world"


def test_scored_strategy_duplicate_model_names():
    # Two ballots can share a model name; the higher-scored response must
    # still win even though scores are reported keyed by model.
    mv = MultiVote(strategy=VotingStrategy.SCORED)
    mv.add("dup", "low", score=0.1)
    mv.add("dup", "high", score=0.9)
    result = mv.vote()
    assert result.winner == "high"


def test_scored_strategy_explicit_zero_score_beats_scorer():
    # An explicit score of 0.0 must be honored over the fallback scorer.
    mv = MultiVote(strategy=VotingStrategy.SCORED, scorer=lambda r: 100.0)
    mv.add("a", "explicit-zero", score=0.0)
    mv.add("b", "scored-default")
    result = mv.vote()
    assert result.winner == "scored-default"
    assert result.scores["a"] == 0.0


def test_consensus_unanimous():
    mv = MultiVote(strategy=VotingStrategy.CONSENSUS)
    mv.add("a", "answer").add("b", "answer")
    result = mv.vote()
    assert result.winner == "answer"
    assert result.unanimous is True


def test_consensus_disagreement():
    mv = MultiVote(strategy=VotingStrategy.CONSENSUS)
    mv.add("a", "answer1").add("b", "answer2")
    result = mv.vote()
    assert result.winner is None
    assert result.tied is True


def test_empty_vote():
    mv = MultiVote()
    result = mv.vote()
    assert result.winner is None


def test_ballot_count():
    mv = MultiVote()
    mv.add("a", "x").add("b", "y")
    assert mv.ballot_count == 2


def test_clear():
    mv = MultiVote()
    mv.add("a", "x")
    mv.clear()
    assert mv.ballot_count == 0


def test_vote_result_models():
    mv = MultiVote()
    mv.add("gpt4", "x").add("claude", "y")
    result = mv.vote()
    assert "gpt4" in result.models
    assert "claude" in result.models


def test_from_responses():
    result = MultiVote.from_responses({"a": "yes", "b": "yes", "c": "no"})
    assert result.winner == "yes"


def test_strategy_override_at_vote_time():
    mv = MultiVote(strategy=VotingStrategy.MAJORITY)
    mv.add("a", "short").add("b", "this is very long indeed")
    result = mv.vote(strategy=VotingStrategy.LONGEST)
    assert result.winner == "this is very long indeed"


def test_ballot_normalized():
    b = Ballot(model="x", response="  Hello World  ")
    assert b.normalized() == "hello world"
