"""llm-multi-vote: Multi-LLM jury voting to select the best response.

This package exposes two complementary APIs:

* The :class:`MultiVote` builder (in this module) collects pre-computed
  responses and aggregates them with a :class:`VotingStrategy`. Use it when
  you already have the model outputs in hand.
* The functional :func:`vote` / :func:`vote_async` API (in
  :mod:`llm_multi_vote.vote`) *calls* a sequence of voter callables with a
  prompt, captures per-voter failures, and reports confidence/consensus.
  Use it when you want the library to invoke the models for you.

The functional API's result type is re-exported here as
``FunctionalVoteResult`` to avoid colliding with the builder's
:class:`VoteResult`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .vote import (
    Strategy,
    VoteResult as FunctionalVoteResult,
    default_normalizer,
    vote,
    vote_async,
)


class VotingStrategy(str, Enum):
    """How :class:`MultiVote` selects a winner from collected ballots.

    Members:
      * ``MAJORITY``: the response shared by the most ballots wins. Ties are
        reported via ``VoteResult.tied`` and the first such response is
        returned.
      * ``LONGEST`` / ``SHORTEST``: pick the ballot with the longest /
        shortest response string.
      * ``FIRST``: pick the first ballot added (useful as a baseline).
      * ``SCORED``: pick the highest-scoring ballot, using each ballot's
        explicit ``score`` or, when absent, the ``scorer`` callable.
      * ``CONSENSUS``: a winner is returned only when every ballot agrees;
        otherwise ``winner`` is ``None`` and ``tied`` is ``True``.
    """

    MAJORITY = "majority"
    LONGEST = "longest"
    SHORTEST = "shortest"
    FIRST = "first"
    SCORED = "scored"
    CONSENSUS = "consensus"


@dataclass
class Ballot:
    """A single model's response in a vote.

    Attributes:
      model: Identifier for the model that produced the response.
      response: The raw response text.
      score: Optional explicit quality score, used by ``SCORED``.
      metadata: Arbitrary extra data attached by the caller.
    """

    model: str
    response: str
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> str:
        """Return the response lower-cased with whitespace collapsed.

        Used as the grouping key for ``MAJORITY``/``CONSENSUS`` so that
        responses differing only in case or spacing count as equal.
        """
        return re.sub(r"\s+", " ", self.response.lower().strip())


@dataclass
class VoteResult:
    """Outcome of a :meth:`MultiVote.vote` call.

    Attributes:
      winner: The winning response text, or ``None`` when no winner could
        be determined (e.g. empty input or ``CONSENSUS`` disagreement).
      strategy: The string value of the strategy that produced this result.
      ballots: A copy of the ballots that were considered.
      tied: ``True`` when multiple responses tied for first place.
      unanimous: ``True`` when every ballot shared the winning response.
      scores: For ``SCORED``, maps each model name to its computed score.
    """

    winner: Optional[str]
    strategy: str
    ballots: list[Ballot]
    tied: bool = False
    unanimous: bool = False
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def ballot_count(self) -> int:
        """Number of ballots considered in this result."""
        return len(self.ballots)

    @property
    def models(self) -> list[str]:
        """The model names of all considered ballots, in ballot order."""
        return [b.model for b in self.ballots]


class MultiVote:
    """Aggregate multiple LLM responses using a chosen voting strategy."""

    def __init__(
        self,
        strategy: VotingStrategy = VotingStrategy.MAJORITY,
        scorer: Optional[Callable[[str], float]] = None,
        normalize: bool = True,
    ) -> None:
        self._strategy = strategy
        self._scorer = scorer
        self._normalize = normalize
        self._ballots: list[Ballot] = []

    def add(
        self, model: str, response: str, score: Optional[float] = None, **metadata: Any
    ) -> "MultiVote":
        """Record one model's response and return ``self`` for chaining.

        Args:
          model: Identifier for the model.
          response: The model's response text.
          score: Optional explicit score used by the ``SCORED`` strategy.
          **metadata: Arbitrary extra fields stored on the ballot.
        """
        self._ballots.append(
            Ballot(model=model, response=response, score=score, metadata=metadata)
        )
        return self

    def clear(self) -> "MultiVote":
        """Remove all recorded ballots and return ``self`` for chaining."""
        self._ballots.clear()
        return self

    @property
    def ballot_count(self) -> int:
        """Number of ballots recorded so far."""
        return len(self._ballots)

    def _key(self, b: Ballot) -> str:
        return b.normalized() if self._normalize else b.response

    def _vote_majority(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(
                winner=None, strategy=VotingStrategy.MAJORITY.value, ballots=[]
            )
        counts: dict[str, list[Ballot]] = {}
        for b in self._ballots:
            counts.setdefault(self._key(b), []).append(b)
        max_count = max(len(v) for v in counts.values())
        winners = [k for k, v in counts.items() if len(v) == max_count]
        tied = len(winners) > 1
        winner_response = counts[winners[0]][0].response
        unanimous = max_count == len(self._ballots)
        return VoteResult(
            winner=winner_response,
            strategy=VotingStrategy.MAJORITY.value,
            ballots=list(self._ballots),
            tied=tied,
            unanimous=unanimous,
        )

    def _vote_longest(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(
                winner=None, strategy=VotingStrategy.LONGEST.value, ballots=[]
            )
        best = max(self._ballots, key=lambda b: len(b.response))
        return VoteResult(
            winner=best.response,
            strategy=VotingStrategy.LONGEST.value,
            ballots=list(self._ballots),
        )

    def _vote_shortest(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(
                winner=None, strategy=VotingStrategy.SHORTEST.value, ballots=[]
            )
        best = min(self._ballots, key=lambda b: len(b.response))
        return VoteResult(
            winner=best.response,
            strategy=VotingStrategy.SHORTEST.value,
            ballots=list(self._ballots),
        )

    def _vote_first(self) -> VoteResult:
        winner = self._ballots[0].response if self._ballots else None
        return VoteResult(
            winner=winner,
            strategy=VotingStrategy.FIRST.value,
            ballots=list(self._ballots),
        )

    def _vote_scored(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(
                winner=None, strategy=VotingStrategy.SCORED.value, ballots=[]
            )
        scorer = self._scorer

        def score_of(b: Ballot) -> float:
            if b.score is not None:
                return b.score
            if scorer is not None:
                return scorer(b.response)
            return 0.0

        # Pair each ballot with its score so selection is independent of
        # model-name collisions (duplicate model names must not clobber
        # each other's score during winner selection).
        scored = [(b, score_of(b)) for b in self._ballots]
        best = max(scored, key=lambda pair: pair[1])[0]
        scores: dict[str, float] = {b.model: s for b, s in scored}
        return VoteResult(
            winner=best.response,
            strategy=VotingStrategy.SCORED.value,
            ballots=list(self._ballots),
            scores=scores,
        )

    def _vote_consensus(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(
                winner=None, strategy=VotingStrategy.CONSENSUS.value, ballots=[]
            )
        keys = {self._key(b) for b in self._ballots}
        if len(keys) == 1:
            return VoteResult(
                winner=self._ballots[0].response,
                strategy=VotingStrategy.CONSENSUS.value,
                ballots=list(self._ballots),
                unanimous=True,
            )
        return VoteResult(
            winner=None,
            strategy=VotingStrategy.CONSENSUS.value,
            ballots=list(self._ballots),
            tied=True,
        )

    def vote(self, strategy: Optional[VotingStrategy] = None) -> VoteResult:
        """Aggregate the recorded ballots and return a :class:`VoteResult`.

        Args:
          strategy: Override the strategy chosen at construction time for
            this call only. The instance's default is used when ``None``.
        """
        s = strategy or self._strategy
        dispatch = {
            VotingStrategy.MAJORITY: self._vote_majority,
            VotingStrategy.LONGEST: self._vote_longest,
            VotingStrategy.SHORTEST: self._vote_shortest,
            VotingStrategy.FIRST: self._vote_first,
            VotingStrategy.SCORED: self._vote_scored,
            VotingStrategy.CONSENSUS: self._vote_consensus,
        }
        return dispatch[s]()

    @staticmethod
    def from_responses(
        responses: dict[str, str],
        strategy: VotingStrategy = VotingStrategy.MAJORITY,
        **kwargs: Any,
    ) -> VoteResult:
        """One-shot helper: build a vote from a ``{model: response}`` map.

        Args:
          responses: Mapping of model name to response text. Note that a
            dict cannot hold duplicate model names; use :meth:`add` if you
            need several ballots from the same model.
          strategy: Strategy to aggregate with (defaults to ``MAJORITY``).
          **kwargs: Forwarded to the :class:`MultiVote` constructor
            (e.g. ``scorer=`` or ``normalize=``).
        """
        mv = MultiVote(strategy=strategy, **kwargs)
        for model, resp in responses.items():
            mv.add(model, resp)
        return mv.vote()


__all__ = [
    # Builder API
    "MultiVote",
    "Ballot",
    "VoteResult",
    "VotingStrategy",
    # Functional API (re-exported from llm_multi_vote.vote)
    "vote",
    "vote_async",
    "Strategy",
    "default_normalizer",
    "FunctionalVoteResult",
]
