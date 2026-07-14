from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.data_types import ObservabilityMode
from src.agents.llm_config import LLMConfig
from src.domain.types import DomainConfig, GEO_DOMAIN


@dataclass
class BenchmarkConfig:
    """Configuration for a single game run in the benchmark.

    This groups all agent LLM configs and orchestrator options.
    """

    seeker_config: LLMConfig
    oracle_config: LLMConfig
    pruner_config: LLMConfig

    observability_mode: ObservabilityMode
    max_turns: int

    # Domain configuration (geo vs flat objects)
    domain_config: Optional[DomainConfig] = None

    # Optional experiment metadata
    experiment_name: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    # Conversation and visualization saving options
    save_conversations: bool = True
    save_graph_plots: bool = False  # Save graph visualizations for each turn

    