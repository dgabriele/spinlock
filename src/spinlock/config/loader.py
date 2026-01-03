"""
YAML configuration loader for Spinlock.

Provides utilities to load and validate YAML configurations using Pydantic schemas.
Supports parameter substitution for runtime values (e.g., ${grid_size}).

Design principles:
- Single source of truth: YAML configs validated against Pydantic schemas
- Runtime flexibility: Parameter substitution for dynamic values
- Clear error messages: Validation errors point to exact config issues
"""

import yaml
import re
import os
from pathlib import Path
from typing import Any, Dict, Union
from .schema import SpinlockConfig


def substitute_params(obj: Any, params: Dict[str, Any]) -> Any:
    """
    Recursively substitute ${param} placeholders with actual values.

    Substitution priority:
    1. Runtime params (passed via params argument)
    2. Environment variables (from os.environ)

    Args:
        obj: Configuration object (dict, list, str, or scalar)
        params: Dictionary of parameter name -> value mappings

    Returns:
        Configuration object with substitutions applied

    Example:
        >>> config = {"grid_size": "${size}", "channels": [16, "${size}"]}
        >>> substitute_params(config, {"size": 64})
        {'grid_size': 64, 'channels': [16, 64]}

        >>> # Environment variable substitution
        >>> os.environ['API_KEY'] = 'secret123'
        >>> config = {"key": "${API_KEY}"}
        >>> substitute_params(config, {})
        {'key': 'secret123'}
    """
    if isinstance(obj, str):
        # Check for ${param} pattern
        match = re.fullmatch(r'\$\{(\w+)\}', obj)
        if match:
            param_name = match.group(1)

            # Priority 1: Runtime params
            if param_name in params:
                return params[param_name]

            # Priority 2: Environment variables
            env_value = os.getenv(param_name)
            if env_value is not None:
                return env_value

            # Not found in either source
            raise ValueError(
                f"Missing parameter: {param_name}. "
                f"Not found in runtime parameters {list(params.keys())} "
                f"or environment variables."
            )
        return obj
    elif isinstance(obj, dict):
        return {k: substitute_params(v, params) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_params(item, params) for item in obj]
    else:
        return obj


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML dict, got {type(config)}")

    return config


def load_config(
    path: Path,
    runtime_params: Union[Dict[str, Any], None] = None
) -> SpinlockConfig:
    """
    Load and validate Spinlock configuration from YAML file.

    Args:
        path: Path to YAML configuration file
        runtime_params: Optional runtime parameters for substitution
                       (e.g., {"grid_size": 512, "input_channels": 3})

    Returns:
        Validated SpinlockConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is malformed or validation fails
        pydantic.ValidationError: If config doesn't match schema

    Example:
        ```python
        # Load basic config
        config = load_config(Path("configs/experiment.yaml"))

        # Load with runtime parameters
        config = load_config(
            Path("configs/experiment.yaml"),
            runtime_params={"grid_size": 1024, "input_channels": 4}
        )
        ```
    """
    # Load raw YAML
    raw_config = load_yaml(path)

    # Apply parameter substitution (always run to pick up environment variables)
    raw_config = substitute_params(raw_config, runtime_params or {})

    # Validate with Pydantic
    try:
        config = SpinlockConfig(**raw_config)
    except Exception as e:
        raise ValueError(
            f"Configuration validation failed for {path}:\n{e}\n\n"
            f"Check your YAML against the schema in spinlock/config/schema.py"
        )

    return config


def save_config(config: SpinlockConfig, path: Path) -> None:
    """
    Save SpinlockConfig to YAML file.

    Useful for saving runtime-generated or modified configurations.

    Args:
        config: SpinlockConfig object to save
        path: Output path for YAML file

    Example:
        ```python
        config = load_config(Path("base.yaml"))
        config.sampling.total_samples = 20000
        save_config(config, Path("modified.yaml"))
        ```
    """
    # Convert Pydantic model to dict
    config_dict = config.model_dump(mode='json')

    # Convert Path objects to strings for YAML serialization
    def path_to_str(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: path_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [path_to_str(item) for item in obj]
        return obj

    config_dict = path_to_str(config_dict)

    # Write to YAML
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def validate_config_file(path: Path, runtime_params: Union[Dict[str, Any], None] = None) -> bool:
    """
    Validate a YAML configuration file without loading it.

    Args:
        path: Path to YAML configuration file
        runtime_params: Optional runtime parameters for substitution

    Returns:
        True if valid, False otherwise (prints errors)

    Example:
        ```python
        if validate_config_file(Path("config.yaml")):
            config = load_config(Path("config.yaml"))
        else:
            print("Fix configuration errors before proceeding")
        ```
    """
    try:
        load_config(path, runtime_params)
        print(f"✓ Configuration valid: {path}")
        return True
    except Exception as e:
        print(f"✗ Configuration invalid: {path}")
        print(f"  Error: {e}")
        return False


# =============================================================================
# Utility Functions
# =============================================================================


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries (deep merge).

    Args:
        base: Base configuration
        override: Configuration to override base with

    Returns:
        Merged configuration dict

    Example:
        ```python
        base = {"sampling": {"total_samples": 1000}}
        override = {"sampling": {"batch_size": 100}}
        merged = merge_configs(base, override)
        # Result: {"sampling": {"total_samples": 1000, "batch_size": 100}}
        ```
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_config_with_overrides(
    base_path: Path,
    override_path: Union[Path, None] = None,
    runtime_params: Union[Dict[str, Any], None] = None
) -> SpinlockConfig:
    """
    Load configuration with optional overrides.

    Useful for inheriting from a base configuration and applying
    experiment-specific changes.

    Args:
        base_path: Path to base configuration
        override_path: Optional path to override configuration
        runtime_params: Optional runtime parameters

    Returns:
        Merged and validated SpinlockConfig

    Example:
        ```python
        config = load_config_with_overrides(
            base_path=Path("configs/base.yaml"),
            override_path=Path("configs/large_scale.yaml")
        )
        ```
    """
    base_config = load_yaml(base_path)

    if override_path:
        override_config = load_yaml(override_path)
        base_config = merge_configs(base_config, override_config)

    if runtime_params:
        base_config = substitute_params(base_config, runtime_params)

    return SpinlockConfig(**base_config)
