Synthesize the LLM agent's reasoning trace into a structured JSON format.

The agent is playing an information-gain guessing game: it asks yes/no questions to identify a secret target (a city, an object, or a disease). Given the agent's raw reasoning block, extract what it considered and why it chose a specific question.

Return ONLY a JSON object with exactly these fields:

{
  "summary": "One or two sentences describing what the agent was trying to figure out in this turn.",
  "questions_considered": ["Full yes/no question the agent evaluated?", "Another question it considered?"],
  "decision_rationale": "One or two sentences explaining why the agent chose the question it did over the alternatives."
}

Rules:
- `questions_considered` must be a JSON array of strings. Each string must be a fully-formed yes/no question ending with a question mark.
- Include the question that was ultimately asked as the last item in `questions_considered`.
- No prose outside the JSON object. No markdown fences.
