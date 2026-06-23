# llm-multi-vote

Multi-LLM jury voting — run a prompt through several models (or collect their
answers however you like) and aggregate the responses into a single winner.

Useful for **self-consistency / ensembling**: instead of trusting one model's
answer, ask several, then take the majority, the consensus, the highest-scored
response, and so on. Voter failures are captured rather than raised, so one
flaky backend does not sink the whole vote.

- Zero runtime dependencies (pure standard library).
- Two complementary APIs — bring your own answers, or let the library call your
  models for you (sync **or** async).
- Fully type-hinted and ships a `py.typed` marker.

## Installation

```bash
pip install llm-multi-vote
```

Or from a clone:

```bash
git clone https://github.com/MukundaKatta/llm-multi-vote
cd llm-multi-vote
pip install -e ".[dev]"   # dev extra adds pytest + ruff
```

Requires Python 3.10+.

## Two APIs

### 1. `MultiVote` — aggregate answers you already have

Use this when you have already collected the model outputs and just want to
pick a winner.

```python
from llm_multi_vote import MultiVote, VotingStrategy

mv = MultiVote(strategy=VotingStrategy.MAJORITY)
mv.add("claude", "Paris")
mv.add("gpt4", "Paris")
mv.add("gemini", "Lyon")

result = mv.vote()
print(result.winner)      # "Paris"
print(result.unanimous)   # False
print(result.tied)        # False
print(result.models)      # ["claude", "gpt4", "gemini"]

# One-shot helper from a {model: response} mapping:
result = MultiVote.from_responses({"a": "yes", "b": "yes", "c": "no"})
print(result.winner)      # "yes"
```

Pick the best answer by a score (explicit per-ballot scores or a scorer fn):

```python
mv = MultiVote(strategy=VotingStrategy.SCORED)
mv.add("a", "an okay answer", score=0.5)
mv.add("b", "a great answer", score=0.9)
print(mv.vote().winner)   # "a great answer"

# Or compute scores on the fly (here: prefer the longest answer):
mv = MultiVote(strategy=VotingStrategy.SCORED, scorer=lambda r: len(r))
mv.add("a", "hi").add("b", "hello world")
print(mv.vote().winner)   # "hello world"
```

**Strategies** (`VotingStrategy`): `MAJORITY`, `LONGEST`, `SHORTEST`, `FIRST`,
`SCORED` (per-ballot `score=` or a `scorer` fn), `CONSENSUS` (every ballot must
agree). By default responses are normalized (lower-cased, whitespace collapsed)
before `MAJORITY`/`CONSENSUS` grouping; pass `normalize=False` to compare the
raw strings.

### 2. `vote` / `vote_async` — let the library call your models

Use this when you want the library to *invoke* each model. A voter is a
`(name, callable)` pair where the callable takes the prompt and returns a
string. Failing voters are captured in `result.failures` instead of raising.

```python
from llm_multi_vote import vote, Strategy

def ask_claude(prompt: str) -> str:
    ...   # call your model, return its answer
    return "yes"

def ask_gpt4(prompt: str) -> str:
    return "yes"

def ask_gemini(prompt: str) -> str:
    return "no"

voters = [("claude", ask_claude), ("gpt4", ask_gpt4), ("gemini", ask_gemini)]

result = vote(voters, "Is the sky blue?", strategy=Strategy.MAJORITY)
print(result.winner)        # "yes"
print(result.confidence)    # 0.666...  (winning votes / total voters)
print(result.consensus)     # False
print(result.votes)         # {"yes": ["claude", "gpt4"], "no": ["gemini"]}
print(result.failures)      # {} (any voter that raised would appear here)
```

Run voters concurrently with `vote_async`. Voter callables may be sync **or**
async; sync ones are offloaded to a thread so they do not block the event loop.

```python
import asyncio
from llm_multi_vote import vote_async, Strategy

async def ask_claude(prompt: str) -> str:
    return "yes"

def ask_gpt4(prompt: str) -> str:   # plain sync callable also works
    return "yes"

voters = [("claude", ask_claude), ("gpt4", ask_gpt4)]
result = asyncio.run(vote_async(voters, "q", strategy=Strategy.PLURALITY))
print(result.winner)   # "yes"
```

**Strategies** (`Strategy`):

| Strategy        | A winner is returned when…                                            |
| --------------- | -------------------------------------------------------------------- |
| `MAJORITY`      | one answer has strictly more than half of the **successful** votes   |
| `PLURALITY`     | one answer has the most votes (no tie for first place)               |
| `UNANIMOUS`     | every voter ran without failing and produced the same answer         |
| `MIN_THRESHOLD` | one answer reaches `min_votes_required=` votes                       |

When no winner meets the bar, `winner` is `None` and `confidence` is `0.0`.

## API reference

### Builder API (`from llm_multi_vote import ...`)

- **`MultiVote(strategy=VotingStrategy.MAJORITY, scorer=None, normalize=True)`** —
  collector for ballots.
  - `.add(model, response, score=None, **metadata) -> MultiVote` — record a
    ballot (chainable).
  - `.clear() -> MultiVote` — drop all ballots.
  - `.ballot_count -> int` — number of recorded ballots.
  - `.vote(strategy=None) -> VoteResult` — aggregate; `strategy` overrides the
    default for this call only.
  - `MultiVote.from_responses(responses, strategy=MAJORITY, **kwargs) -> VoteResult` —
    one-shot from a `{model: response}` mapping.
- **`VotingStrategy`** — `MAJORITY`, `LONGEST`, `SHORTEST`, `FIRST`, `SCORED`,
  `CONSENSUS`.
- **`Ballot(model, response, score=None, metadata={})`** — `.normalized()`
  returns the lower-cased, whitespace-collapsed response.
- **`VoteResult`** — `winner`, `strategy`, `ballots`, `tied`, `unanimous`,
  `scores`, plus `.ballot_count` and `.models`.

### Functional API (`from llm_multi_vote import ...`, defined in `llm_multi_vote.vote`)

- **`vote(voters, prompt, strategy=Strategy.MAJORITY, *, normalizer=None, min_votes_required=None) -> FunctionalVoteResult`**
- **`vote_async(voters, prompt, strategy=Strategy.MAJORITY, *, normalizer=None, min_votes_required=None, parallel=True) -> FunctionalVoteResult`**
- **`Strategy`** — `MAJORITY`, `PLURALITY`, `UNANIMOUS`, `MIN_THRESHOLD`.
- **`default_normalizer(s) -> str`** — strip + lower-case.
- **`FunctionalVoteResult`** (the `vote` module's `VoteResult`) — `winner`,
  `votes`, `confidence`, `consensus`, `all_responses`, `failures`.

> The functional API's result type is re-exported as `FunctionalVoteResult`
> so it does not collide with the builder API's `VoteResult`.

## Development

Run the tests. The suite works under both `pytest` and the standard-library
`unittest` runner:

```bash
pytest -v                                 # full suite (needs the dev extra)
python3 -m unittest discover -s tests     # standard library only, no deps
```

Lint with ruff:

```bash
ruff check src/ tests/
```

CI runs lint plus both test runners across Python 3.10–3.13
(see `.github/workflows/ci.yml`).

## License

MIT — see [LICENSE](LICENSE).
