# llm-multi-vote

[![PyPI](https://img.shields.io/pypi/v/llm-multi-vote.svg)](https://pypi.org/project/llm-multi-vote/)
[![Python](https://img.shields.io/pypi/pyversions/llm-multi-vote.svg)](https://pypi.org/project/llm-multi-vote/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Multi-LLM voting for high-stakes decisions.**

Run a prompt through N model callables, combine the answers with a
voting rule (majority, plurality, unanimous, threshold), and trust the
quorum rather than any single model. Zero runtime deps.

This is the jury pattern. Useful for spam/abuse labels, content moderation
gates, safety checks, and any low-cardinality decision where you would
rather not stake the outcome on one model's hallucination.

## Install

```bash
pip install llm-multi-vote
```

## Sync example

```python
from llm_multi_vote import vote, Strategy

def claude(prompt): ...
def gpt(prompt): ...
def gemini(prompt): ...

result = vote(
    voters=[("claude", claude), ("gpt", gpt), ("gemini", gemini)],
    prompt="Is this email spam? Answer yes or no.",
    strategy=Strategy.MAJORITY,
)

result.winner         # "no"  (canonical, case-insensitive trimmed)
result.confidence     # 0.666...  (winner_count / total_voters)
result.consensus      # False  (not all three agreed)
result.votes          # {"no": ["claude", "gpt"], "yes": ["gemini"]}
result.all_responses  # {"claude": "No.", "gpt": "no", "gemini": "Yes"}
result.failures       # {} unless a voter raised
```

If a voter raises, the exception is captured in `result.failures` and
that voter is excluded from the tally. The vote still produces a
`VoteResult`.

## Async example

Each voter callable can be sync or async. The async path runs all voters
concurrently with `asyncio.gather` by default.

```python
import asyncio
from llm_multi_vote import vote_async, Strategy

async def claude(prompt): ...
async def gpt(prompt): ...
async def gemini(prompt): ...

async def main():
    result = await vote_async(
        voters=[("claude", claude), ("gpt", gpt), ("gemini", gemini)],
        prompt="Classify the tone: positive, negative, or neutral.",
        strategy=Strategy.PLURALITY,
    )
    print(result.winner)

asyncio.run(main())
```

Pass `parallel=False` to run them one at a time if a backend is rate-limit
sensitive.

## Custom normalizer

By default answers are canonicalized with `s.strip().lower()` so `"Yes"`,
`"yes"`, and `"  YES  "` collapse into one bucket. Override for stricter
canonicalization:

```python
def strip_punct(s: str) -> str:
    return s.strip().lower().rstrip(".!?")

result = vote(
    voters=[...],
    prompt="...",
    strategy=Strategy.MAJORITY,
    normalizer=strip_punct,
)
# now "Yes.", "yes", "YES" all collapse to "yes"
```

## Strategies

| Strategy        | Winner rule                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `MAJORITY`      | A single bucket has strictly more than half of the non-failing votes. Ties return `None`.      |
| `PLURALITY`     | The bucket with the most votes wins. Ties for first return `None`.                             |
| `UNANIMOUS`     | Every non-failing voter agrees AND there are zero failures. Otherwise `None`.                  |
| `MIN_THRESHOLD` | A single bucket has at least `min_votes_required` votes. Pair with `min_votes_required=N`.     |

```python
vote(voters, prompt, strategy=Strategy.MIN_THRESHOLD, min_votes_required=3)
```

## What it does NOT do

- No HTTP. You bring the voter callables. Wrap whatever provider clients
  you want (Anthropic, OpenAI, Bedrock, local models).
- No retries inside a voter. If you want retry-on-failure for one model,
  compose with [`agentcast`](https://pypi.org/project/agentcast/)
  (validate-and-retry for a single LLM).
- No failover. If you want first-OK-wins instead of vote, use
  [`llm-fallback-chain`](https://pypi.org/project/llm-fallback-chain/).
- No persistence. `VoteResult` is a plain frozen dataclass; persist it
  yourself if you want an audit trail. Pairs well with
  [`agentsnap`](https://pypi.org/project/agentsnap/) for snapshot eval.
- No cost / token accounting. Use one of the sibling cost calculators if
  you need that.

## License

MIT
