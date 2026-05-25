"""
llm-multi-vote: Multi-LLM jury voting to select the best response.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class VotingStrategy(str, Enum):
    MAJORITY = "majority"
    LONGEST = "longest"
    SHORTEST = "shortest"
    FIRST = "first"
    SCORED = "scored"
    CONSENSUS = "consensus"


@dataclass
class Ballot:
    model: str
    response: str
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> str:
        return re.sub(r"\s+", " ", self.response.lower().strip())


@dataclass
class VoteResult:
    winner: Optional[str]
    strategy: str
    ballots: list[Ballot]
    tied: bool = False
    unanimous: bool = False
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def ballot_count(self) -> int:
        return len(self.ballots)

    @property
    def models(self) -> list[str]:
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

    def add(self, model: str, response: str, score: Optional[float] = None, **metadata: Any) -> "MultiVote":
        self._ballots.append(Ballot(model=model, response=response, score=score, metadata=metadata))
        return self

    def clear(self) -> "MultiVote":
        self._ballots.clear()
        return self

    @property
    def ballot_count(self) -> int:
        return len(self._ballots)

    def _key(self, b: Ballot) -> str:
        return b.normalized() if self._normalize else b.response

    def _vote_majority(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(winner=None, strategy=VotingStrategy.MAJORITY.value, ballots=[])
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
            return VoteResult(winner=None, strategy=VotingStrategy.LONGEST.value, ballots=[])
        best = max(self._ballots, key=lambda b: len(b.response))
        return VoteResult(winner=best.response, strategy=VotingStrategy.LONGEST.value, ballots=list(self._ballots))

    def _vote_shortest(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(winner=None, strategy=VotingStrategy.SHORTEST.value, ballots=[])
        best = min(self._ballots, key=lambda b: len(b.response))
        return VoteResult(winner=best.response, strategy=VotingStrategy.SHORTEST.value, ballots=list(self._ballots))

    def _vote_first(self) -> VoteResult:
        winner = self._ballots[0].response if self._ballots else None
        return VoteResult(winner=winner, strategy=VotingStrategy.FIRST.value, ballots=list(self._ballots))

    def _vote_scored(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(winner=None, strategy=VotingStrategy.SCORED.value, ballots=[])
        scorer = self._scorer
        scores: dict[str, float] = {}
        for b in self._ballots:
            if b.score is not None:
                scores[b.model] = b.score
            elif scorer is not None:
                scores[b.model] = scorer(b.response)
            else:
                scores[b.model] = 0.0
        best = max(self._ballots, key=lambda b: scores.get(b.model, 0.0))
        return VoteResult(
            winner=best.response,
            strategy=VotingStrategy.SCORED.value,
            ballots=list(self._ballots),
            scores=scores,
        )

    def _vote_consensus(self) -> VoteResult:
        if not self._ballots:
            return VoteResult(winner=None, strategy=VotingStrategy.CONSENSUS.value, ballots=[])
        keys = {self._key(b) for b in self._ballots}
        if len(keys) == 1:
            return VoteResult(
                winner=self._ballots[0].response,
                strategy=VotingStrategy.CONSENSUS.value,
                ballots=list(self._ballots),
                unanimous=True,
            )
        return VoteResult(winner=None, strategy=VotingStrategy.CONSENSUS.value, ballots=list(self._ballots), tied=True)

    def vote(self, strategy: Optional[VotingStrategy] = None) -> VoteResult:
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
        mv = MultiVote(strategy=strategy, **kwargs)
        for model, resp in responses.items():
            mv.add(model, resp)
        return mv.vote()


__all__ = ["MultiVote", "Ballot", "VoteResult", "VotingStrategy"]
