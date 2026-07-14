"""Orchestrator for a single benchmark game.

Coordinates turns between `SeekerAgent`, `OracleAgent`, and `PrunerAgent`,
computes entropy metrics with `Entropy`, and records `TurnState`.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

from .entropy import Entropy
from .candidates import Candidate, CandidatePool
from .data_types import TurnState, Question, Answer, ObservabilityMode
from .domain.types import DomainConfig, GEO_DOMAIN
from .agents.seeker import SeekerAgent
from .agents.human_seeker import HumanSeekerAgent
from .agents.oracle import OracleAgent
from .agents.pruner import PrunerAgent
from .agents.llm_adapter import LLMAdapter
from .agents.llm_config import LLMConfig
from .utils.git_info import get_git_info


class Orchestrator:
    """Orchestrates a single benchmark game.

    Args:
        pool: CandidatePool with all candidates and pruning state.
        seeker: SeekerAgent instance responsible for generating questions.
        oracle: OracleAgent instance responsible for generating answers.
        pruner: PrunerAgent for LLM-driven pruning.
        entropy: Entropy helper for computing entropy and information gain.
        max_turns: Maximum number of turns to execute.
    """

    def __init__(
        self,
        *,
        pool: CandidatePool,
        seeker: SeekerAgent,
        oracle: OracleAgent,
        pruner: PrunerAgent,
        entropy: Entropy,
        max_turns: int,
    ) -> None:
        if pool is None:
            raise ValueError("pool cannot be None")
        if seeker is None:
            raise ValueError("seeker cannot be None")
        if oracle is None:
            raise ValueError("oracle cannot be None")
        if entropy is None:
            raise ValueError("entropy cannot be None")
        if max_turns <= 0:
            raise ValueError("max_turns must be > 0")

        self._pool = pool
        self._seeker = seeker
        self._oracle = oracle
        self._entropy = entropy
        self._max_turns = max_turns
        self._pruner = pruner
        self._domain_config = getattr(pruner, "domain_config", None) or GEO_DOMAIN

        self._current_turn: int = 0
        self._turns: List[TurnState] = []

    @property
    def turns(self) -> List[TurnState]:
        return self._turns

    @property
    def current_turn(self) -> int:
        return self._current_turn

    @classmethod
    def from_target(
        cls,
        *,
        target: Candidate,
        pool: CandidatePool,
        seeker_config: LLMConfig,
        oracle_config: LLMConfig,
        pruner_config: LLMConfig,
        observability_mode: ObservabilityMode = ObservabilityMode.FULLY_OBSERVABLE,
        max_turns: int = 40,
        domain_config: DomainConfig | None = None,
    ) -> Orchestrator:
        """Factory method to create an Orchestrator with all agents configured.

        Args:
            target: The target Candidate that the Seeker must find.
            pool: CandidatePool for the game.
            seeker_config: LLM configuration for SeekerAgent.
            oracle_config: LLM configuration for OracleAgent.
            pruner_config: LLM configuration for PrunerAgent.
            observability_mode: How much candidate info the Seeker can see.
            max_turns: Maximum number of turns before game ends.
            domain_config: Domain configuration. Defaults to GEO_DOMAIN.

        Returns:
            Fully configured Orchestrator ready to run.
        """
        oracle_adapter = LLMAdapter(oracle_config, save_reasoning=True)
        pruner_adapter = LLMAdapter(pruner_config, save_reasoning=True)

        domain_config = domain_config or GEO_DOMAIN

        if seeker_config.model == "human":
            seeker = HumanSeekerAgent(
                observability_mode=observability_mode,
                domain_config=domain_config,
                max_turns=max_turns,
            )
        else:
            seeker_adapter = LLMAdapter(seeker_config, save_reasoning=True)
            seeker = SeekerAgent(
                llm_adapter=seeker_adapter,
                observability_mode=observability_mode,
                domain_config=domain_config,
                max_turns=max_turns,
            )

        oracle = OracleAgent(
            llm_adapter=oracle_adapter,
            target=target,
            domain_config=domain_config,
        )

        pruner = PrunerAgent(
            llm_adapter=pruner_adapter,
            domain_config=domain_config,
        )

        entropy = Entropy()

        orch = cls(
            pool=pool,
            seeker=seeker,
            oracle=oracle,
            pruner=pruner,
            entropy=entropy,
            max_turns=max_turns,
        )
        orch._domain_config = domain_config
        return orch

    def show_turn(self, turn: TurnState) -> None:
        """Log the turn state with detailed information."""
        progress = (turn.turn_index / self._max_turns) * 100
        pruning_info = ""
        if turn.pruning_result:
            pruning_info = (
                f" | pruned={turn.pruned_count}"
                f" | rationale={turn.pruning_result.rationale!r}"
            )
        logger.info(
            "Turn %d/%d (%.0f%%) | Q: %s | A: %s | H: %.4f→%.4f | IG: %.4f"
            " | active=%d%s",
            turn.turn_index, self._max_turns, progress,
            turn.question.text,
            turn.answer.text,
            turn.h_before, turn.h_after, turn.info_gain,
            turn.active_candidates_after or 0,
            pruning_info,
        )
        if turn.pruning_result and turn.pruning_result.pruned_labels:
            logger.debug("Pruned labels: %s", turn.pruning_result.pruned_labels)

    def run(self, debug: bool = False, save_plots: bool = False, plots_dir: Optional[Path] = None) -> None:
        """Execute the benchmark loop.

        Args:
            debug: Unused — verbosity is controlled by the log level.
            save_plots: Ignored (kept for backward compatibility).
            plots_dir: Ignored (kept for backward compatibility).
        """
        for turn in range(1, self._max_turns + 1):
            self._current_turn = turn
            turn_start = datetime.now()

            active_candidates = self._pool.get_active()
            active_count_before = len(active_candidates)
            h_before = self._entropy.compute(active_count_before)

            # Inject candidate list on first turn for FO and IO modes.
            # FO will keep updating the pool every turn; IO sees it only here.
            candidates_text = self._pool.to_text()
            if turn == 1 and self._seeker.observability_mode.name in (
                "FULLY_OBSERVABLE",
                "INITIALLY_OBSERVABLE",
            ):
                self._seeker.add_initial_candidates(candidates_text, turn)

            # Seeker asks a question
            question: Question = self._seeker.question_to_oracle(active_candidates, turn)

            # Oracle receives and answers
            self._oracle.add_seeker_question(question)
            answer: Answer = self._oracle.answer_seeker()

            # Pruner eliminates candidates
            pruned_count = 0
            pruning_result = self._pruner.analyze_and_prune(
                candidate_pool=self._pool,
                turn_index=turn,
                question=question,
                answer=answer,
                target_label=self._oracle._target.label,
            )
            if pruning_result.pruned_labels:
                pruned_count = self._pool.prune(pruning_result.pruned_labels)

            # Seeker integrates oracle's answer
            self._seeker.add_oracle_answer_and_pruning(
                answer,
                candidates_text=self._pool.to_text() if self._seeker.observability_mode.name == "FULLY_OBSERVABLE" else None,
                turn=turn,
            )

            # Compute entropy after pruning
            active_after = self._pool.get_active()
            active_count_after = len(active_after)

            # Validate Oracle's game_over against post-pruning state.
            # Oracle (especially Qwen3-8B with strict JSON schema) can hallucinate
            # game_over=true on off-task/non-yes-no questions, copying the
            # "Yes! You found the target!" example from the prompt. The most robust
            # signal of a real win is that the Pruner converged the active set to
            # exactly the target — anything else means the win is spurious.
            if answer.game_over:
                target_label = self._oracle._target.label
                converged = (
                    active_count_after == 1
                    and active_after[0].label == target_label
                )
                if not converged:
                    logger.warning(
                        "Oracle set game_over=true but pruning did not converge to "
                        "target (active=%d, target_in_active=%s). Overriding to "
                        "False. Q=%r",
                        active_count_after,
                        any(c.label == target_label for c in active_after),
                        question.text[:120],
                    )
                    answer = Answer(
                        rationale=answer.rationale,
                        text=answer.text,
                        compliant=False,
                        game_over=False,
                    )

            if answer.game_over:
                h_after = 0.0
            else:
                h_after = self._entropy.compute(active_count_after)

            info_gain = self._entropy.info_gain(h_before, h_after)

            turn_end = datetime.now()
            duration = (turn_end - turn_start).total_seconds()

            self._turns.append(
                TurnState(
                    turn_index=turn,
                    h_before=h_before,
                    h_after=h_after,
                    info_gain=info_gain,
                    pruned_count=pruned_count,
                    question=question,
                    answer=answer,
                    pruning_result=pruning_result,
                    active_candidates_before=active_count_before,
                    active_candidates_after=active_count_after,
                    timestamp_start=turn_start.isoformat(),
                    timestamp_end=turn_end.isoformat(),
                    duration_seconds=round(duration, 6),
                    candidates_snapshot=[c.label for c in active_candidates],
                )
            )

            if debug:
                self.show_turn(self._turns[-1])

            if answer.game_over:
                logger.info("Game over! Target found in %d turns.", turn)
                break

        if self._turns:
            summary = self.get_summary()
            logger.info(
                "Benchmark complete | turns=%d | H: %.4f→%.4f"
                " | total_IG=%.4f | avg_IG/turn=%.4f",
                summary["turns"],
                summary["h_start"],
                summary["h_end"],
                summary["total_info_gain"],
                summary["avg_info_gain_per_turn"],
            )

    def get_summary(self) -> dict:
        """Return a simple summary of the run."""
        total_info_gain = sum(t.info_gain for t in self._turns)
        num_turns = len(self._turns)

        return {
            "turns": num_turns,
            "current_turn": self._current_turn,
            "h_start": self._turns[0].h_before if self._turns else None,
            "h_end": self._turns[-1].h_after if self._turns else None,
            "total_info_gain": total_info_gain,
            "avg_info_gain_per_turn": total_info_gain / num_turns if num_turns > 0 else 0.0,
        }

    def export_conversation(self, output_dir: Path) -> None:
        """Export complete game conversation for all agents.

        Args:
            output_dir: Directory to save conversation files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Seeker history
        if self._seeker._llm_adapter._save_history:
            seeker_data = {
                "agent_type": "seeker",
                "config": {
                    "model": self._seeker._llm_adapter.config.model,
                    "temperature": self._seeker._llm_adapter.config.temperature,
                    "max_tokens": self._seeker._llm_adapter.config.max_tokens,
                    "base_url": self._seeker._llm_adapter.config.base_url,
                },
                "observability_mode": self._seeker.observability_mode.name,
                "total_messages": len(self._seeker._llm_adapter.history),
                "history": self._seeker._llm_adapter.history,
                "reasoning_history": self._seeker._llm_adapter.reasoning_history,
            }
            with (output_dir / "seeker.json").open("w", encoding="utf-8") as f:
                json.dump(seeker_data, f, indent=2, ensure_ascii=False)

        # 2. Save Oracle history
        if self._oracle._llm_adapter._save_history:
            oracle_data = {
                "agent_type": "oracle",
                "config": {
                    "model": self._oracle._llm_adapter.config.model,
                    "temperature": self._oracle._llm_adapter.config.temperature,
                    "max_tokens": self._oracle._llm_adapter.config.max_tokens,
                    "base_url": self._oracle._llm_adapter.config.base_url,
                },
                "target": {
                    "id": self._oracle._target.id,
                    "label": self._oracle._target.label,
                    "attrs": dict(self._oracle._target.attrs),
                },
                "total_messages": len(self._oracle._llm_adapter.history),
                "history": self._oracle._llm_adapter.history,
                "reasoning_history": self._oracle._llm_adapter.reasoning_history,
            }
            with (output_dir / "oracle.json").open("w", encoding="utf-8") as f:
                json.dump(oracle_data, f, indent=2, ensure_ascii=False)

        # 3. Save Pruner history
        pruner_data = {
            "agent_type": "pruner",
            "config": {
                "model": self._pruner.llm_adapter.config.model,
                "temperature": self._pruner.llm_adapter.config.temperature,
                "max_tokens": self._pruner.llm_adapter.config.max_tokens,
                "base_url": self._pruner.llm_adapter.config.base_url,
            },
            "save_history": self._pruner.llm_adapter._save_history,
            "total_calls": len(self.turns),
        }

        if self._pruner.llm_adapter._save_history:
            pruner_data["total_messages"] = len(self._pruner.llm_adapter.history)
            pruner_data["history"] = self._pruner.llm_adapter.history
            pruner_data["reasoning_history"] = self._pruner.llm_adapter.reasoning_history
        else:
            pruner_data["note"] = "Pruner was configured with save_history=False."
            pruner_data["history"] = []
            pruner_data["reasoning_history"] = []

        with (output_dir / "pruner.json").open("w", encoding="utf-8") as f:
            json.dump(pruner_data, f, indent=2, ensure_ascii=False)

        # 4. Save metadata
        summary = self.get_summary()
        win = any(t.answer.game_over for t in self.turns)
        compliance_rate = (
            sum(1 for t in self.turns if t.answer.compliant) / len(self.turns)
        ) if self.turns else 0.0

        total_pruned = sum(t.pruned_count for t in self._turns)
        initial_count = len(self._pool.candidates)
        final_active = len(self._pool.get_active())

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "git": get_git_info(),
            "target": {
                "id": self._oracle._target.id,
                "label": self._oracle._target.label,
                "attrs": dict(self._oracle._target.attrs),
            },
            "config": {
                "experiment_name": None,
                "observability_mode": self._seeker.observability_mode.name,
                "max_turns": self._max_turns,
                "models": {
                    "seeker": self._seeker._llm_adapter.config.model,
                    "oracle": self._oracle._llm_adapter.config.model,
                    "pruner": self._pruner.llm_adapter.config.model,
                },
            },
            "results": {
                "turns_played": len(self.turns),
                "win": win,
                "h_start": summary["h_start"],
                "h_end": summary["h_end"],
                "total_info_gain": summary["total_info_gain"],
                "avg_info_gain_per_turn": summary["avg_info_gain_per_turn"],
                "compliance_rate": round(compliance_rate, 4),
                "final_active_candidates": final_active,
            },
            "pool_stats": {
                "initial_candidates": initial_count,
                "final_candidates": final_active,
                "total_pruned": total_pruned,
                "pruning_efficiency": round(total_pruned / initial_count, 4) if initial_count > 0 else 0,
            },
        }

        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 5. Save turn-by-turn details
        with (output_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
            for turn_state in self.turns:
                turn_data = turn_state.to_export_dict()
                f.write(json.dumps(turn_data, ensure_ascii=False) + "\n")
