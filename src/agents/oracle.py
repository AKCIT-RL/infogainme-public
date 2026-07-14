"""OracleAgent implementation for answering questions about the target.

The OracleAgent knows the target candidate and answers questions truthfully
without revealing the target directly.
"""

from __future__ import annotations

from pydantic import ValidationError

from ..data_types import Answer, Question, OracleResponse
from ..candidates import Candidate
from ..domain.types import DomainConfig, GEO_DOMAIN
from ..prompts import get_oracle_system_prompt
from ..utils import ClaryLogger
from .llm_adapter import LLMAdapter
from ..utils.utils import llm_final_content

logger = ClaryLogger.get_logger(__name__)


class OracleAgent:
    """Agent that answers questions about a known target candidate."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        target: Candidate,
        *,
        domain_config: DomainConfig | None = None,
    ) -> None:
        """Initialize the OracleAgent.

        Args:
            llm_adapter: LLMAdapter instance for generating answers.
            target: The Candidate the Oracle knows as the target.
            domain_config: Domain configuration. Defaults to GEO_DOMAIN.

        Raises:
            ValueError: If llm_adapter or target is None.
        """
        if llm_adapter is None:
            raise ValueError("LLMAdapter cannot be None")
        if target is None:
            raise ValueError("Target candidate cannot be None")

        self._model = llm_adapter.config.model
        self._llm_adapter = llm_adapter
        self._target = target
        self._answers_given = 0

        self._domain_config = domain_config or GEO_DOMAIN
        system_prompt = self._build_system_prompt_with_target(target)
        self._llm_adapter.append_history("system", system_prompt)

    @property
    def model(self) -> str:
        return self._model

    @property
    def target(self) -> Candidate:
        return self._target

    @property
    def answers_given(self) -> int:
        return self._answers_given

    def add_seeker_question(self, question: Question) -> None:
        """Add Seeker's question to conversation history."""
        self._llm_adapter.append_history("user", f"[Seeker] - {question.text}")

    _RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "OracleResponse",
            "schema": OracleResponse.model_json_schema(),
            "strict": True,
        },
    }

    _VALIDATION_RETRIES = 3

    def answer_seeker(self) -> Answer:
        """Generate an answer to the Seeker's question."""
        last_error: ValidationError | None = None
        for attempt in range(self._VALIDATION_RETRIES):
            response = self._llm_adapter.generate(response_format=self._RESPONSE_FORMAT)
            response = llm_final_content(response)
            try:
                oracle_response = OracleResponse.model_validate_json(response)
                break
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "OracleResponse validation failed (attempt %d/%d): %s | raw=%r",
                    attempt + 1, self._VALIDATION_RETRIES, exc, response[:200],
                )
                self._llm_adapter.pop_last_if_assistant()
        else:
            assert last_error is not None
            raise last_error

        is_compliant = self._check_compliance(oracle_response.answer)
        self._answers_given += 1

        return Answer(
            text=oracle_response.answer,
            compliant=is_compliant,
            game_over=oracle_response.game_over,
            rationale=oracle_response.rationale,
        )

    def _build_system_prompt_with_target(self, target: Candidate) -> str:
        """Build system prompt with target information included."""
        base_prompt = get_oracle_system_prompt(
            target_noun=self._domain_config.target_noun,
            domain_description=self._domain_config.domain_description,
        )

        attrs_parts = []
        for k, v in target.attrs.items():
            if k == "aliases" and isinstance(v, (list, tuple)):
                attrs_parts.append(f"aliases={list(v)}")
            else:
                attrs_parts.append(f"{k}={v}")
        attrs_str = f", {', '.join(attrs_parts)}" if attrs_parts else ""

        aliases_note = ""
        if target.attrs.get("aliases"):
            aliases = target.attrs["aliases"]
            if isinstance(aliases, (list, tuple)):
                aliases_note = (
                    f"\nAlso known as (accept these as correct): "
                    f"{', '.join(str(a) for a in aliases)}"
                )

        target_info = (
            f"\n\n## Your Target\n\n"
            f"ID: {target.id}\n"
            f"Label: {target.label}{attrs_str}{aliases_note}\n\n"
            f"This is the target you know about. Answer all questions truthfully based on "
            f"this target's properties. Set game_over=true when the Seeker correctly identifies "
            f"the target (by label or alias)."
        )

        return base_prompt + target_info

    def _check_compliance(self, answer_text: str) -> bool:
        """Check if the answer complies with Oracle rules."""
        answer_lower = answer_text.lower().strip()

        # Check for direct target revelation (basic heuristic)
        if self._target.label.lower() in answer_lower:
            return False

        for pattern in ("yes", "no"):
            if pattern in answer_lower:
                return True

        return True
