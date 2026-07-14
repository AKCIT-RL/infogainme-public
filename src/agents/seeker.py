"""SeekerAgent implementation for proactive information gathering.

The SeekerAgent asks strategic questions to reduce uncertainty in the candidate pool,
operating under different observability modes (FULLY_OBSERVABLE, PARTIALLY_OBSERVABLE).
"""

from __future__ import annotations

from typing import List, Optional
from os import getenv

from ..data_types import ObservabilityMode, Question, Answer
from ..candidates import Candidate
from ..domain.types import DomainConfig, GEO_DOMAIN
from ..prompts import get_seeker_system_prompt
from .llm_adapter import LLMAdapter
from dotenv import load_dotenv

load_dotenv()


class SeekerAgent:
    """Agent that seeks information by asking strategic questions."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        observability_mode: ObservabilityMode,
        domain_config: DomainConfig | None = None,
        max_turns: int = 25,
    ) -> None:
        """Initialize the SeekerAgent.

        Args:
            llm_adapter: LLMAdapter instance for generating questions.
            observability_mode: Mode controlling how much candidate info is visible.
            domain_config: Domain configuration. Defaults to GEO_DOMAIN.

        Raises:
            ValueError: If llm_adapter is None or observability_mode is invalid.
        """
        if llm_adapter is None:
            raise ValueError("LLMAdapter cannot be None")
        if not isinstance(observability_mode, ObservabilityMode):
            raise ValueError("Invalid observability mode")

        self._model = llm_adapter.config.model
        self._llm_adapter = llm_adapter
        self._observability_mode = observability_mode
        self._domain_config = domain_config or GEO_DOMAIN
        self._max_turns = max_turns
        self._questions_asked = 0
        self._initial_candidates_injected: bool = False

        # Initialize conversation with system prompt
        self._llm_adapter.append_history(
            "system",
            get_seeker_system_prompt(
                target_noun=self._domain_config.target_noun,
                domain_description=self._domain_config.domain_description,
                max_turns=self._max_turns,
                observability_mode=self._observability_mode.value,
                pool_description=self._domain_config.seeker_pool_description,
            ),
        )

        # Thinking-model chat templates only open <think> after a user turn;
        # FO and IO supply one via add_initial_candidates (the candidate list
        # itself is the kickoff), PO needs this synthetic kickoff.
        if self._observability_mode == ObservabilityMode.PARTIALLY_OBSERVABLE:
            self._llm_adapter.append_history(
                "user",
                "Start the game. Ask your first question.",
            )

    @property
    def model(self) -> str:
        return self._model

    @property
    def observability_mode(self) -> ObservabilityMode:
        return self._observability_mode

    @property
    def questions_asked(self) -> int:
        return self._questions_asked

    def question_to_oracle(
        self,
        active_candidates: List[Candidate],
        turn: int,
    ) -> Question:
        """Generate a strategic question to ask the Oracle.

        Args:
            active_candidates: List of candidates not yet pruned.
            turn: Current turn number.

        Returns:
            A Question object containing the generated question text.
        """
        question_text = self._llm_adapter.generate()
        self._questions_asked += 1
        return Question(text=question_text.strip())

    def add_oracle_answer_and_pruning(
        self,
        answer: Answer,
        candidates_text: Optional[str],
        turn: int,
    ) -> None:
        """Add Oracle's answer to the conversation history.

        Args:
            answer: The Oracle's answer to add to history.
            candidates_text: Text representation of current active candidates (or None).
            turn: Current turn number.
        """
        user_answer = f"[Turn {turn}/{self._max_turns}] [Oracle] - {answer.text}"

        context = self._build_context(candidates_text, turn)
        if context:
            user_answer += f"\n[Computer] - {context}"

        self._llm_adapter.append_history("user", user_answer)

    def add_initial_candidates(self, candidates_text: str, turn: int) -> None:
        """Inject the initial candidate list into the conversation once.

        Fires for FULLY_OBSERVABLE and INITIALLY_OBSERVABLE; no-op for PO.

        Args:
            candidates_text: Textual representation of all candidates.
            turn: Current turn number.
        """
        if self._observability_mode == ObservabilityMode.PARTIALLY_OBSERVABLE:
            return
        if not candidates_text or self._initial_candidates_injected:
            return

        self._llm_adapter.append_history(
            "user",
            f"[Turn {turn}/{self._max_turns}] [Computer] - {candidates_text}",
        )
        self._initial_candidates_injected = True

    def _build_context(self, candidates_text: Optional[str], turn: int) -> Optional[str]:
        """Build context prompt based on observability mode."""
        if self.observability_mode == ObservabilityMode.FULLY_OBSERVABLE:
            assert candidates_text is not None, (
                "candidates_text cannot be None when observability mode is FULLY_OBSERVABLE"
            )
            return candidates_text

        elif self.observability_mode in (
            ObservabilityMode.PARTIALLY_OBSERVABLE,
            ObservabilityMode.INITIALLY_OBSERVABLE,
        ):
            # IO behaves like PO after the initial reveal — no candidate list
            # is appended on subsequent turns. The orchestrator passes None.
            return None

        else:
            raise ValueError(f"Unknown observability mode: {self.observability_mode}")
