"""llm-multi-vote - run a prompt through N model callables and combine the
results via majority, plurality, unanimous, or threshold rules.

The jury pattern for high-stakes decisions: instead of trusting one model,
ask several and aggregate. Useful for spam/abuse classification, content
moderation, safety gates, and any binary or small-cardinality decision
where one model's hallucination should not be the final answer.

    from llm_multi_vote import vote, Strategy

    result = vote(
        voters=[("claude", claude), ("gpt", gpt), ("gemini", gemini)],
        prompt="Is this email spam? Answer yes or no.",
        strategy=Strategy.MAJORITY,
    )

    if result.winner == "yes":
        ...

Sibling to `llm-fallback-chain` (failover, picks the first OK response),
`agentcast` (single-model JSON validate-and-retry), and `agentsnap`
(snapshot eval).
"""

from llm_multi_vote.vote import (
    Strategy,
    Voter,
    VoteResult,
    default_normalizer,
    vote,
    vote_async,
)

__version__ = "0.1.0"

__all__ = [
    "Strategy",
    "VoteResult",
    "Voter",
    "__version__",
    "default_normalizer",
    "vote",
    "vote_async",
]
