"""Simple configuration loader for benchmark experiments."""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from ..data_types import ObservabilityMode
from ..agents.llm_config import LLMConfig
from ..benchmark_config import BenchmarkConfig
import dataclasses

from ..domain.types import GEO_DOMAIN, OBJECTS_DOMAIN, DISEASES_DOMAIN


def _load_servers(config_path: Path, override_path: Optional[Path] = None) -> Dict[str, str]:
    """Load servers.yaml by walking up from config_path to find the configs/ root.

    If override_path is provided, merge it with servers.yaml (override takes precedence).
    """
    p = config_path.resolve().parent
    while p.name != "configs" and p != p.parent:
        p = p.parent
    servers_path = p / "servers.yaml"
    servers = {}

    if servers_path.exists():
        with servers_path.open("r") as f:
            data = yaml.safe_load(f)
        servers = data.get("servers", {}) if data else {}

    # Merge with override file if provided
    if override_path and override_path.exists():
        with override_path.open("r") as f:
            override_data = yaml.safe_load(f)
        override_servers = override_data.get("servers", {}) if override_data else {}
        servers.update(override_servers)  # override takes precedence

    return servers


def load_benchmark_config(config_path: Path, api_key: str, servers_override_path: Optional[Path] = None) -> tuple[BenchmarkConfig, Dict[str, Any]]:
    """Load benchmark configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
        api_key: OpenAI API key.
        servers_override_path: Optional path to servers override YAML file.

    Returns:
        Tuple of (BenchmarkConfig, full_config_dict).
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    servers = _load_servers(config_path, servers_override_path)

    # Create LLM configs
    def create_llm_config(model_config: Dict[str, Any]) -> LLMConfig:
        model_name = model_config["model"]

        # Human seeker needs no LLM endpoint
        if model_name == "human":
            return LLMConfig(model="human", api_key="n/a")

        # Extract extra parameters (all non-standard parameters)
        extra_params = {}
        standard_params = {
            "model", "base_url", "api_key", "timeout", "temperature",
            "max_tokens", "use_reasoning", "user", "response_format"
        }

        for key, value in model_config.items():
            if key not in standard_params:
                extra_params[key] = value

        env_key = "VLLM_" + re.sub(r"[^A-Z0-9]", "_", model_name.upper())
        base_url: Optional[str] = model_config.get("base_url") or os.environ.get(env_key) or servers.get(model_name)

        return LLMConfig(
            model=model_name,
            api_key=model_config.get("api_key", api_key),
            base_url=base_url,
            timeout=model_config.get("timeout"),
            temperature=model_config.get("temperature"),  # Allow None to use model default
            max_tokens=model_config.get("max_tokens"),
            user=model_config.get("user"),
            response_format=model_config.get("response_format"),
            use_reasoning=model_config.get("use_reasoning", False),
            extra=extra_params
        )
    
    # Parse observability mode
    mode_str = config["game"]["observability_mode"].upper()
    if mode_str in ["FULLY_OBSERVABLE", "FO"]:
        observability_mode = ObservabilityMode.FULLY_OBSERVABLE
    elif mode_str in ["PARTIALLY_OBSERVABLE", "PO"]:
        observability_mode = ObservabilityMode.PARTIALLY_OBSERVABLE
    elif mode_str in ["INITIALLY_OBSERVABLE", "IO"]:
        observability_mode = ObservabilityMode.INITIALLY_OBSERVABLE
    else:
        raise ValueError(f"Unknown observability mode: {mode_str}")
    
    models = config["models"]
    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "geo")
    if dataset_type == "objects":
        domain_config = OBJECTS_DOMAIN
    elif dataset_type == "diseases":
        domain_config = DISEASES_DOMAIN
    else:
        domain_config = GEO_DOMAIN

    pool_description = dataset_cfg.get("pool_description", "")
    if pool_description:
        domain_config = dataclasses.replace(
            domain_config, seeker_pool_description=pool_description
        )

    benchmark_config = BenchmarkConfig(
        seeker_config=create_llm_config(models["seeker"]),
        oracle_config=create_llm_config(models["oracle"]),
        pruner_config=create_llm_config(models["pruner"]),
        observability_mode=observability_mode,
        max_turns=config["game"]["max_turns"],
        domain_config=domain_config,
        experiment_name=config["experiment"]["name"],
        tags=config["experiment"].get("tags", {}),
        save_conversations=config["output"]["save_conversations"],
        save_graph_plots=config["output"]["save_graph_plots"],
    )

    return benchmark_config, config