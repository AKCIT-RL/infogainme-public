"""PrunerAgent implementation for LLM-driven candidate pruning.

The PrunerAgent analyzes question-answer pairs and determines which candidates
should be eliminated from the CandidatePool.

Strategy: the LLM returns `keep_labels` (survivors). Pruned = active - keep_labels.
This keeps the output small when many candidates are eliminated. Input uses to_text()
(compact labels only) to keep context usage low.
"""

from __future__ import annotations

import logging

from ..data_types import Answer, PruningResult, Question, PrunerResponse
from ..candidates import CandidatePool
from ..domain.types import DomainConfig, GEO_DOMAIN
from .llm_adapter import LLMAdapter
from ..prompts import get_pruner_system_prompt
from ..utils.utils import llm_final_content

logger = logging.getLogger(__name__)


class PrunerAgent:
    """Agent responsible for LLM-driven pruning of the candidate pool."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        domain_config: DomainConfig | None = None,
        max_retries: int = 15,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.domain_config = domain_config or GEO_DOMAIN
        self.pruning_count = 0
        self.max_retries = max_retries

        if self.llm_adapter._save_history:
            system_prompt = get_pruner_system_prompt(
                target_noun=self.domain_config.target_noun,
            )
            self.llm_adapter.append_history("system", system_prompt)

    def analyze_and_prune(
        self,
        candidate_pool: CandidatePool,
        turn_index: int,
        question: Question,
        answer: Answer,
        *,
        target_label: str | None = None,
    ) -> PruningResult:
        """Delegate pruning decision to the LLM.

        The LLM returns the smaller of keep/prune sets (mode field).
        Falls back to no pruning on any parse error to avoid crashing the game.

        Args:
            candidate_pool: CandidatePool with current active candidates.
            turn_index: Current turn number (1-based).
            question: Seeker's question.
            answer: Oracle's answer.
            target_label: Label that must NEVER be pruned.

        Returns:
            PruningResult with pruned_labels and rationale.
        """
        system_prompt = get_pruner_system_prompt(
            target_noun=self.domain_config.target_noun,
        )

        candidates_text = candidate_pool.to_text()

        user_prompt = (
            "CANDIDATES:\n" + candidates_text + "\n\n" +
            f"TURN: {turn_index}\n" +
            f"QUESTION: {question.text}\n" +
            f"ANSWER: {answer.text}\n\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.llm_adapter._save_history:
            self.llm_adapter.append_history("user", user_prompt)

        active_labels = {c.label for c in candidate_pool.get_active()}

        last_error: str = "unknown"
        for attempt in range(1, self.max_retries + 1):
            reply = self.llm_adapter.generate(messages=messages, stateless=True)
            reply = llm_final_content(reply)

            try:
                pruning_response = PrunerResponse.model_validate_json(reply)
            except Exception as e:
                last_error = f"parse error: {e}"
                logger.warning(
                    "Pruner JSON parse error (turn %d, attempt %d/%d): %s",
                    turn_index, attempt, self.max_retries, e,
                )
                continue

            keep_labels_raw = pruning_response.keep_labels
            rationale = pruning_response.rationale

            keep_labels = {l for l in keep_labels_raw if l in active_labels}
            unknown = set(keep_labels_raw) - active_labels
            if unknown:
                logger.debug("Pruner returned unknown keep_labels (ignored): %s", unknown)

            if not keep_labels:
                last_error = f"empty keep_labels: {rationale}"
                logger.warning(
                    "Pruner returned empty keep_labels (turn %d, attempt %d/%d) — retrying.",
                    turn_index, attempt, self.max_retries,
                )
                continue

            pruned_labels = active_labels - keep_labels

            # CRITICAL: never prune the target
            if target_label:
                pruned_labels.discard(target_label)

            self.pruning_count += len(pruned_labels)
            return PruningResult(pruned_labels=pruned_labels, rationale=rationale)

        logger.error(
            "Pruner failed after %d attempts (turn %d) — skipping pruning: %s",
            self.max_retries, turn_index, last_error,
        )
        return PruningResult(pruned_labels=set(), rationale=f"failed after {self.max_retries} attempts: {last_error}")
