"""Configuration loading utilities."""

import yaml
import os
import re
from pathlib import Path
from typing import Type, TypeVar

T = TypeVar('T', bound='BaseExperimentConfig')


def substitute_env_vars(config_dict: dict) -> dict:
    """Recursively substitute ${VAR} patterns with environment variables."""
    pattern = re.compile(r'\$\{(\w+)\}')

    def replace(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
        elif isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace(item) for item in value]
        return value

    return replace(config_dict)


def load_experiment_config(config_path: Path, config_class: Type[T]) -> T:
    """Load and validate experiment configuration from YAML."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Substitute environment variables
    config_dict = substitute_env_vars(config_dict)

    # Validate with Pydantic
    return config_class(**config_dict)


def save_experiment_config(config, output_path: Path) -> None:
    """Save experiment configuration to YAML."""
    with open(output_path, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
