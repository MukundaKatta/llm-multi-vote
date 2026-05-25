# llm-multi-vote

Multi-LLM jury voting — aggregate responses from multiple models and pick a winner.

```python
from llm_multi_vote import MultiVote, VotingStrategy

mv = MultiVote(strategy=VotingStrategy.MAJORITY)
mv.add("claude", "Paris")
mv.add("gpt4", "Paris")
mv.add("gemini", "Lyon")
result = mv.vote()
print(result.winner)   # "Paris"
print(result.unanimous)  # False

# one-shot helper
result = MultiVote.from_responses({"a": "yes", "b": "yes", "c": "no"})
```

Strategies: `MAJORITY`, `LONGEST`, `SHORTEST`, `FIRST`, `SCORED` (scorer fn or per-ballot score), `CONSENSUS` (all must agree).
