Extract the BELIEF STATE that an LLM agent maintains over the candidate set in an information-gain guessing game.

The agent (the "Seeker") asks yes/no questions to identify a secret target drawn from a fixed, enumerable set of candidates (a city, an object, or a disease). After each question the Oracle answers yes/no, which logically rules some candidates in or out. You are given (1) the question/answer history so far and (2) the Seeker's raw chain-of-thought for the CURRENT turn. Extract the belief the Seeker is holding about which candidates are still viable.

A belief state can be expressed two ways, and you must capture both when present:
- **Constraints** — predicates the Seeker treats as established (e.g. "located in Africa", "not on the coast", "is an animal"). These are the accumulated yes/no facts it is reasoning from.
- **Named candidates** — specific targets the reasoning mentions, split by whether the Seeker treats them as still viable (`kept_candidates`) or already eliminated (`excluded_candidates`).

Return ONLY a JSON object with exactly these fields:

{
  "constraints": ["predicate the Seeker believes is true so far", "..."],
  "kept_candidates": ["Candidate the reasoning treats as still possible", "..."],
  "excluded_candidates": ["Candidate the reasoning rules out this turn or earlier", "..."],
  "believed_count": 12,
  "explicit_tracking": true
}

Rules:
- Use the exact candidate names as written in the reasoning (do not normalize spelling or add candidates that are not mentioned).
- A candidate goes in `excluded_candidates` only if the reasoning clearly rules it out (e.g. "so it can't be Cairo", "Shanghai is coastal, out"). Otherwise, if mentioned as a live possibility, it goes in `kept_candidates`.
- `believed_count` is the number of remaining candidates the Seeker explicitly states it is down to; use null if it states no count.
- `explicit_tracking` is true if the Seeker actually reasons about which specific candidates remain or maintains a running set; false if it only reasons abstractly (e.g. picks a partitioning question) without tracking the candidate set.
- Every field must be present. Empty arrays are allowed. No prose outside the JSON object. No markdown fences.
