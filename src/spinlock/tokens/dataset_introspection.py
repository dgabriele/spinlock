"""
Dataset introspection utilities for VQTokenizer.

Automatically detect dataset structure and feature dimensions at runtime,
eliminating hardcoded assumptions about channel counts, feature dimensions, etc.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import h5py
import logging

logger = logging.getLogger(__name__)


class DatasetIntrospector:
    """
    Introspect dataset structure to automatically determine encoder dimensions.

    As a framework, VQTokenizer should adapt to any dataset structure automatically
    without requiring manual configuration of dimensions that can be inferred from data.
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize introspector.

        Args:
            dataset_path: Path to HDF5 dataset
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    def introspect_all(self) -> Dict[str, any]:
        """
        Introspect complete dataset structure.

        Returns:
            Dict with inferred dimensions:
            - num_samples: Number of samples
            - initial_manual_dim: Initial feature dimension (if exists)
            - temporal_feature_dim: Temporal feature dimension (if exists)
            - theta_param_dim: Parameter dimension (if exists)
            - initial_raw_channels: Number of input channels (if exists)
            - initial_raw_shape: Full shape of raw inputs
            - temporal_timesteps: Number of timesteps
        """
        with h5py.File(self.dataset_path, 'r') as f:
            result = {}

            # Number of samples (from any dataset)
            if 'parameters/params' in f:
                result['num_samples'] = f['parameters/params'].shape[0]
            elif 'features/temporal/features' in f:
                result['num_samples'] = f['features/temporal/features'].shape[0]
            elif 'inputs/fields' in f:
                result['num_samples'] = f['inputs/fields'].shape[0]
            else:
                raise ValueError("Cannot determine dataset size - no standard arrays found")

            # Initial manual features (if available)
            if 'features/initial/aggregated/features' in f:
                initial_shape = f['features/initial/aggregated/features'].shape
                result['initial_manual_dim'] = initial_shape[1]
                logger.info(f"Initial manual features: {initial_shape} -> {initial_shape[1]}D")
            else:
                result['initial_manual_dim'] = None
                logger.warning("No initial manual features found at /features/initial/aggregated/features")

            # Temporal features (if available)
            if 'features/temporal/features' in f:
                temporal_shape = f['features/temporal/features'].shape  # [N, T, D]
                result['temporal_feature_dim'] = temporal_shape[2]
                result['temporal_timesteps'] = temporal_shape[1]
                logger.info(
                    f"Temporal features: {temporal_shape} -> "
                    f"{temporal_shape[1]} timesteps × {temporal_shape[2]}D"
                )
            else:
                result['temporal_feature_dim'] = None
                result['temporal_timesteps'] = None
                logger.warning("No temporal features found at /features/temporal/features")

            # Parameters/theta (if available)
            if 'parameters/params' in f:
                params_shape = f['parameters/params'].shape  # [N, P]
                result['theta_param_dim'] = params_shape[1]
                logger.info(f"Parameters: {params_shape} -> {params_shape[1]}D")
            else:
                result['theta_param_dim'] = None
                logger.warning("No parameters found at /parameters/params")

            # Raw initial conditions (for CNN encoder)
            if 'inputs/fields' in f:
                raw_shape = f['inputs/fields'].shape
                result['initial_raw_shape'] = raw_shape

                # Infer number of channels
                # Common shapes: [N, C, H, W], [N, M, C, H, W], [N, C, S, H, W]
                if len(raw_shape) == 4:
                    # [N, C, H, W]
                    result['initial_raw_channels'] = raw_shape[1]
                elif len(raw_shape) == 5:
                    # Could be [N, M, C, H, W] or [N, C, S, H, W]
                    # If dim1 is small (< 10), likely channels; otherwise realizations
                    if raw_shape[1] < 10:
                        # [N, C, S, H, W] - flatten channels × species
                        result['initial_raw_channels'] = raw_shape[1] * raw_shape[2]
                        logger.info(
                            f"Raw inputs: {raw_shape} -> "
                            f"{raw_shape[1]} channels × {raw_shape[2]} species = "
                            f"{result['initial_raw_channels']} effective channels"
                        )
                    else:
                        # [N, M, C, H, W] - take channel count
                        result['initial_raw_channels'] = raw_shape[2]
                        logger.info(
                            f"Raw inputs: {raw_shape} -> "
                            f"{raw_shape[2]} channels (from realizations)"
                        )
                else:
                    result['initial_raw_channels'] = None
                    logger.warning(f"Unexpected raw input shape: {raw_shape}")

                logger.info(f"Raw initial conditions: {raw_shape}")
            else:
                result['initial_raw_shape'] = None
                result['initial_raw_channels'] = None
                logger.warning("No raw inputs found at /inputs/fields")

            return result

    def get_encoder_config_overrides(self) -> Dict[str, any]:
        """
        Get configuration overrides for encoder based on dataset introspection.

        Returns dict suitable for updating TokenizerConfig.encoder fields.

        Example:
            >>> introspector = DatasetIntrospector("datasets/qbm_50k.h5")
            >>> overrides = introspector.get_encoder_config_overrides()
            >>> config_dict.update(overrides)
        """
        info = self.introspect_all()
        overrides = {}

        # Initial encoder overrides
        if info['initial_manual_dim'] is not None:
            overrides.setdefault('encoder', {}).setdefault('initial', {})['manual_dim'] = info['initial_manual_dim']

        if info['initial_raw_channels'] is not None:
            overrides.setdefault('encoder', {}).setdefault('initial', {})['in_channels'] = info['initial_raw_channels']

        # Theta encoder overrides
        if info['theta_param_dim'] is not None:
            overrides.setdefault('encoder', {}).setdefault('theta', {})['param_dim'] = info['theta_param_dim']

        # Temporal encoder overrides (for validation)
        if info['temporal_timesteps'] is not None:
            overrides.setdefault('encoder', {}).setdefault('temporal', {})['max_timesteps'] = max(
                info['temporal_timesteps'],
                overrides.get('encoder', {}).get('temporal', {}).get('max_timesteps', 256)
            )

        return overrides

    def validate_config(self, config_dict: Dict) -> Tuple[bool, list]:
        """
        Validate that config is compatible with dataset.

        Args:
            config_dict: Configuration dictionary

        Returns:
            (is_valid, warnings): Tuple of validation result and list of warning messages
        """
        info = self.introspect_all()
        warnings = []

        # Check manual_dim if specified
        if 'encoder' in config_dict and 'initial' in config_dict['encoder']:
            initial_cfg = config_dict['encoder']['initial']

            if 'manual_dim' in initial_cfg:
                configured_dim = initial_cfg['manual_dim']
                actual_dim = info['initial_manual_dim']

                if actual_dim is not None and configured_dim != actual_dim:
                    warnings.append(
                        f"Config specifies manual_dim={configured_dim} but dataset has "
                        f"{actual_dim}D features. Will use dataset dimension."
                    )

            if 'in_channels' in initial_cfg:
                configured_channels = initial_cfg['in_channels']
                actual_channels = info['initial_raw_channels']

                if actual_channels is not None and configured_channels != actual_channels:
                    warnings.append(
                        f"Config specifies in_channels={configured_channels} but dataset has "
                        f"{actual_channels} channels. Will use dataset channels."
                    )

        # Check theta param_dim if specified
        if 'encoder' in config_dict and 'theta' in config_dict['encoder']:
            theta_cfg = config_dict['encoder']['theta']

            if 'param_dim' in theta_cfg:
                configured_dim = theta_cfg['param_dim']
                actual_dim = info['theta_param_dim']

                if actual_dim is not None and configured_dim != actual_dim:
                    warnings.append(
                        f"Config specifies param_dim={configured_dim} but dataset has "
                        f"{actual_dim} parameters. Will use dataset dimension."
                    )

        return (len(warnings) == 0, warnings)


def introspect_and_update_config(
    config_dict: Dict,
    dataset_path: Path,
    verbose: bool = True
) -> Dict:
    """
    Automatically introspect dataset and update config with detected dimensions.

    This ensures the config is consistent with the actual dataset structure,
    overriding any manually-specified dimensions that don't match.

    Args:
        config_dict: Configuration dictionary (will be modified in-place)
        dataset_path: Path to HDF5 dataset
        verbose: Print introspection results

    Returns:
        Updated config dict

    Example:
        >>> config_dict = yaml.safe_load(open('config.yaml'))
        >>> config_dict = introspect_and_update_config(config_dict, 'datasets/qbm_50k.h5')
        >>> config = TokenizerConfig(**config_dict)
    """
    introspector = DatasetIntrospector(dataset_path)

    if verbose:
        logger.info("Introspecting dataset structure...")

    # Get detected dimensions
    overrides = introspector.get_encoder_config_overrides()

    # Validate config if dimensions were manually specified
    is_valid, warnings = introspector.validate_config(config_dict)

    if not is_valid and verbose:
        logger.warning("Configuration mismatches detected:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Apply overrides (deep merge)
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config_dict = deep_update(config_dict, overrides)

    if verbose:
        logger.info("Applied dataset-detected dimensions to config")

    return config_dict
