"""HumanSeekerAgent — interactive CLI seeker for human baseline experiments.

Presents game state to a human player via stdout and reads questions from stdin.
Implements the same interface as SeekerAgent so it is a drop-in replacement in
Orchestrator.from_target when ``seeker_config.model == "human"``.
"""

from __future__ import annotations

from typing import List, Optional

from ..data_types import ObservabilityMode, Question, Answer
from ..candidates import Candidate
from ..domain.types import DomainConfig, GEO_DOMAIN
from ..agents.llm_config import LLMConfig


class _MockAdapterConfig:
    """Minimal stand-in for LLMConfig accessed during export_conversation."""

    def __init__(self) -> None:
        self.model = "human"
        self.temperature = None
        self.max_tokens = None
        self.base_url = None


class _MockLLMAdapter:
    """Minimal mock of LLMAdapter used solely for export_conversation compatibility.

    Records question/answer exchanges in the same history format as LLMAdapter so
    that orchestrator.export_conversation() writes a readable seeker.json.
    """

    def __init__(self) -> None:
        self._save_history = True
        self._history: list[dict] = []
        self._reasoning_history: list[dict] = []
        self.config = _MockAdapterConfig()

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def reasoning_history(self) -> list[dict]:
        return list(self._reasoning_history)

    def append(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})


_DIVIDER = "─" * 60


class HumanSeekerAgent:
    """Human player as the Seeker in a benchmark game.

    Prompts the user for yes/no questions via stdin and displays game state
    (oracle answers, active candidates in FO mode) via stdout.
    """

    def __init__(
        self,
        observability_mode: ObservabilityMode,
        domain_config: DomainConfig | None = None,
        max_turns: int = 25,
    ) -> None:
        if not isinstance(observability_mode, ObservabilityMode):
            raise ValueError("Invalid observability mode")

        self._observability_mode = observability_mode
        self._domain_config = domain_config or GEO_DOMAIN
        self._max_turns = max_turns
        self._questions_asked = 0
        self._llm_adapter = _MockLLMAdapter()

    # --- Properties matching SeekerAgent ---

    @property
    def model(self) -> str:
        return "human"

    @property
    def observability_mode(self) -> ObservabilityMode:
        return self._observability_mode

    @property
    def questions_asked(self) -> int:
        return self._questions_asked

    # --- Game interface ---

    def add_initial_candidates(self, candidates_text: str, turn: int) -> None:
        """Display the full candidate list at the start (FO and IO modes)."""
        if self._observability_mode == ObservabilityMode.PARTIALLY_OBSERVABLE:
            return
        if not candidates_text:
            return
        print(f"\n{_DIVIDER}")
        print(f"  {self._domain_config.target_noun.upper()} POOL — {candidates_text.count(',') + 1} candidates")
        print(_DIVIDER)
        print(candidates_text)
        print(_DIVIDER)
        self._llm_adapter.append("user", f"[Computer] Initial candidates:\n{candidates_text}")

    def question_to_oracle(
        self,
        active_candidates: List[Candidate],
        turn: int,
    ) -> Question:
        """Prompt the human for a yes/no question."""
        print(f"\n{'═' * 60}")
        print(f"  TURN {turn}/{self._max_turns}  |  Active candidates: {len(active_candidates)}")
        if self._observability_mode == ObservabilityMode.FULLY_OBSERVABLE:
            labels = ", ".join(c.label for c in active_candidates)
            print(f"  Remaining: {labels}")
        print(f"{'═' * 60}")

        while True:
            try:
                text = input("  Your question > ").strip()
            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt("Game interrupted by user.")
            if text:
                break
            print("  [!] Question cannot be empty — try again.")

        self._questions_asked += 1
        self._llm_adapter.append("assistant", text)
        return Question(text=text)

    def add_oracle_answer_and_pruning(
        self,
        answer: Answer,
        candidates_text: Optional[str],
        turn: int,
    ) -> None:
        """Display the oracle's answer (and updated pool in FO mode)."""
        print(f"\n  Oracle → {answer.text}")
        if answer.game_over:
            print("  *** Target identified! Game over. ***")
        elif candidates_text and self._observability_mode == ObservabilityMode.FULLY_OBSERVABLE:
            print(f"\n  Remaining candidates:")
            print(f"  {candidates_text}")

        self._llm_adapter.append(
            "user",
            f"[Turn {turn}/{self._max_turns}] [Oracle] {answer.text}"
            + (f"\n[Computer] {candidates_text}" if candidates_text else ""),
        )
