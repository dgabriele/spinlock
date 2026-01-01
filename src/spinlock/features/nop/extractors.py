"""
NOP (Neural Operator Parameter) feature extractors.

Extracts parameter-derived features from [0,1]^P unit hypercube
and mapped operator configurations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from ..registry import FeatureRegistry, FeatureMetadata
from .config import NOPConfig


class NOPExtractor:
    """
    Neural Operator Parameter (NOP) feature extractor.

    Extracts parameter-derived features from [0,1]^P unit hypercube
    and mapped operator configurations. Features are per-operator only
    (no temporal or realization dimensions).

    Output shape: [N, D_nop] where D_nop depends on enabled categories.

    Future extensibility:
        - Learned embeddings via PCA/clustering of parameter manifold
        - Adaptive stratification refinement metadata
        - Cross-operator similarity metrics

    Example:
        >>> from spinlock.features.nop import NOPConfig, NOPExtractor
        >>> config = NOPConfig()
        >>> param_spec = {...}  # From dataset config
        >>> extractor = NOPExtractor(config, param_spec)
        >>>
        >>> # Extract from [N, P] parameters
        >>> parameters = np.random.rand(100, 14)  # 100 operators, 14-dim space
        >>> features_dict = extractor.extract(parameters)
        >>> # features_dict: {feature_name: [N] array}
    """

    def __init__(self, config: NOPConfig, param_spec: Dict[str, Dict[str, Any]]):
        """
        Args:
            config: NOP feature extraction configuration
            param_spec: Parameter space specification from config
                       (needed to decode [0,1]^P to actual parameters)
        """
        self.config = config
        self.param_spec = param_spec

        # Build parameter index mapping for efficient access
        self._param_indices = self._build_parameter_indices()

        # Build feature registry
        self.registry = self._build_registry()

        # Cache for efficiency
        self._num_strata_per_dim = 4  # Could be from config

    def _build_parameter_indices(self) -> Dict[str, int]:
        """
        Build mapping from parameter name to index in unit_params array.

        Returns:
            Dictionary mapping parameter_name -> index
        """
        indices = {}
        idx = 0

        # Follow the order in param_spec (architecture, stochastic, operator, evolution)
        for category in ['architecture', 'stochastic', 'operator', 'evolution']:
            if category in self.param_spec:
                for param_name in self.param_spec[category].keys():
                    indices[param_name] = idx
                    idx += 1

        return indices

    def _get_parameter_dimensionality(self) -> int:
        """Get total parameter space dimensionality from param_spec."""
        count = 0
        for category in ['architecture', 'stochastic', 'operator', 'evolution']:
            if category in self.param_spec:
                count += len(self.param_spec[category])
        return count

    def _build_registry(self) -> FeatureRegistry:
        """Build feature registry for NOP family."""
        registry = FeatureRegistry(family_name="nop")

        # Architecture features
        if self.config.architecture.enabled:
            arch_spec = self.param_spec.get('architecture', {})

            if self.config.architecture.include_depth and 'num_layers' in arch_spec:
                registry.register("arch_num_layers", "architecture",
                                description="Number of convolutional layers")
            if self.config.architecture.include_width and 'base_channels' in arch_spec:
                registry.register("arch_base_channels", "architecture",
                                description="Base channel width")
            if self.config.architecture.include_kernel_size and 'kernel_size' in arch_spec:
                registry.register("arch_kernel_size", "architecture",
                                description="Convolutional kernel size")

            if self.config.architecture.include_activation_encoding and 'activation' in arch_spec:
                # One-hot encoding for activation types
                activation_choices = arch_spec['activation'].get('choices', [])
                for act_type in activation_choices:
                    registry.register(f"arch_activation_{act_type}", "architecture",
                                    description=f"Activation is {act_type}")

            if self.config.architecture.include_dropout_rate and 'dropout_rate' in arch_spec:
                registry.register("arch_dropout_rate", "architecture",
                                description="Dropout probability")

            if self.config.architecture.include_total_parameters:
                # This requires both num_layers and base_channels
                if 'num_layers' in arch_spec and 'base_channels' in arch_spec:
                    registry.register("arch_total_params_log10", "architecture",
                                    description="Log10 of total parameter count estimate")

        # Stochastic features
        if self.config.stochastic.enabled:
            stoch_spec = self.param_spec.get('stochastic', {})

            if self.config.stochastic.include_noise_scale_log and 'noise_scale' in stoch_spec:
                registry.register("stoch_noise_scale_log", "stochastic",
                                description="Log10 of noise scale")

            if self.config.stochastic.include_noise_schedule_encoding and 'noise_schedule' in stoch_spec:
                schedule_choices = stoch_spec['noise_schedule'].get('choices', [])
                for schedule in schedule_choices:
                    registry.register(f"stoch_schedule_{schedule}", "stochastic",
                                    description=f"Noise schedule is {schedule}")

            if self.config.stochastic.include_spatial_correlation and 'spatial_correlation' in stoch_spec:
                registry.register("stoch_spatial_corr", "stochastic",
                                description="Spatial correlation length")

            if self.config.stochastic.include_noise_type_encoding and 'noise_type' in stoch_spec:
                noise_choices = stoch_spec['noise_type'].get('choices', [])
                for noise_type in noise_choices:
                    registry.register(f"stoch_noise_{noise_type}", "stochastic",
                                    description=f"Noise type is {noise_type}")

            if self.config.stochastic.include_stochasticity_score and 'noise_scale' in stoch_spec:
                registry.register("stoch_score", "stochastic",
                                description="Combined stochasticity metric")

        # Operator features
        if self.config.operator.enabled:
            op_spec = self.param_spec.get('operator', {})

            if self.config.operator.include_normalization_encoding and 'normalization' in op_spec:
                norm_choices = op_spec['normalization'].get('choices', [])
                for norm_type in norm_choices:
                    registry.register(f"op_norm_{norm_type}", "operator",
                                    description=f"Normalization is {norm_type}")

            if self.config.operator.include_grid_size and 'grid_size' in op_spec:
                registry.register("op_grid_size", "operator",
                                description="Grid resolution (64/128/256)")

            if self.config.operator.include_grid_size_class and 'grid_size' in op_spec:
                grid_choices = op_spec['grid_size'].get('choices', [])
                for size in grid_choices:
                    registry.register(f"op_grid_{size}", "operator",
                                    description=f"Grid size is {size}")

        # Evolution features
        if self.config.evolution.enabled:
            evol_spec = self.param_spec.get('evolution', {})

            if self.config.evolution.include_update_policy_encoding and 'update_policy' in evol_spec:
                policy_choices = evol_spec['update_policy'].get('choices', [])
                for policy in policy_choices:
                    registry.register(f"evol_policy_{policy}", "evolution",
                                    description=f"Update policy is {policy}")

            if self.config.evolution.include_dt_log and 'dt' in evol_spec:
                registry.register("evol_dt_log", "evolution",
                                description="Log10 of integration timestep")

            if self.config.evolution.include_alpha and 'alpha' in evol_spec:
                registry.register("evol_alpha", "evolution",
                                description="Mixing parameter alpha")

        # Stratification features
        if self.config.stratification.enabled:
            if self.config.stratification.include_stratum_ids:
                # P dimensions × stratum ID
                P = self._get_parameter_dimensionality()
                for dim in range(P):
                    registry.register(f"strat_dim{dim}_stratum", "stratification",
                                    description=f"Stratum ID for dimension {dim}")

            if self.config.stratification.include_stratum_hash:
                registry.register("strat_hash", "stratification",
                                description="Composite stratum hash")

            if self.config.stratification.include_distance_to_boundary:
                registry.register("strat_boundary_dist", "stratification",
                                description="Min distance to unit hypercube boundary")

            if self.config.stratification.include_extremeness_score:
                registry.register("strat_extremeness", "stratification",
                                description="Distance from hypercube center")

        return registry

    def extract(self, parameters: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract NOP features from parameter vectors.

        Args:
            parameters: [N, P] array in [0,1] unit hypercube

        Returns:
            Dictionary mapping feature_name → [N] array
        """
        N, P = parameters.shape
        features = {}

        # Extract raw parameter features
        if self.config.architecture.enabled:
            features.update(self._extract_architecture(parameters))

        if self.config.stochastic.enabled:
            features.update(self._extract_stochastic(parameters))

        if self.config.operator.enabled:
            features.update(self._extract_operator(parameters))

        if self.config.evolution.enabled:
            features.update(self._extract_evolution(parameters))

        if self.config.stratification.enabled:
            features.update(self._extract_stratification(parameters))

        return features

    def _decode_parameter(
        self,
        unit_value: np.ndarray,  # [N] array
        param_name: str
    ) -> np.ndarray:
        """
        Decode a single parameter from [0,1] to actual value.

        Args:
            unit_value: [N] array of values in [0,1]
            param_name: Name of parameter to decode

        Returns:
            [N] array of decoded values
        """
        # Find param_spec for this parameter
        spec = None
        for category in ['architecture', 'stochastic', 'operator', 'evolution']:
            if category in self.param_spec and param_name in self.param_spec[category]:
                spec = self.param_spec[category][param_name]
                break

        if spec is None:
            raise ValueError(f"Parameter {param_name} not found in param_spec")

        param_type = spec["type"]

        if param_type == "integer":
            low, high = spec["bounds"]
            return np.round(low + unit_value * (high - low)).astype(np.float32)

        elif param_type == "continuous":
            low, high = spec["bounds"]
            if spec.get("log_scale", False):
                # Log-uniform sampling
                log_low, log_high = np.log10(low), np.log10(high)
                return (10 ** (log_low + unit_value * (log_high - log_low))).astype(np.float32)
            else:
                # Linear sampling
                return (low + unit_value * (high - low)).astype(np.float32)

        elif param_type == "choice":
            # For choice parameters, return the unit value itself
            # (one-hot encoding will be done separately)
            return unit_value.astype(np.float32)

        elif param_type == "boolean":
            return (unit_value > 0.5).astype(np.float32)

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _one_hot_encode_choice(
        self,
        unit_values: np.ndarray,  # [N] array in [0,1]
        param_name: str
    ) -> Dict[str, np.ndarray]:
        """
        One-hot encode a choice parameter.

        Args:
            unit_values: [N] array of values in [0,1]
            param_name: Name of choice parameter

        Returns:
            Dictionary {choice_name: [N] binary array}
        """
        # Find param_spec for this parameter
        spec = None
        for category in ['architecture', 'stochastic', 'operator', 'evolution']:
            if category in self.param_spec and param_name in self.param_spec[category]:
                spec = self.param_spec[category][param_name]
                break

        if spec is None or spec["type"] != "choice":
            raise ValueError(f"Parameter {param_name} not found or not a choice parameter")

        choices = spec["choices"]
        weights = spec.get("weights", None)
        N = len(unit_values)

        # Decode to choice indices
        if weights is None:
            # Uniform distribution
            indices = np.floor(unit_values * len(choices)).astype(int)
            indices = np.clip(indices, 0, len(choices) - 1)  # Handle u=1.0
        else:
            # Weighted distribution using cumulative probabilities
            cumulative_weights = np.cumsum(weights)
            indices = np.searchsorted(cumulative_weights, unit_values, side='right')
            indices = np.clip(indices, 0, len(choices) - 1)  # Safety clamp

        # Create one-hot encoding
        one_hot = {}
        for i, choice in enumerate(choices):
            one_hot[choice] = (indices == i).astype(np.float32)

        return one_hot

    def _extract_architecture(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract architecture-related features."""
        features = {}
        arch_spec = self.param_spec.get('architecture', {})

        if self.config.architecture.include_depth and 'num_layers' in arch_spec:
            idx = self._param_indices['num_layers']
            features['arch_num_layers'] = self._decode_parameter(params[:, idx], 'num_layers')

        if self.config.architecture.include_width and 'base_channels' in arch_spec:
            idx = self._param_indices['base_channels']
            features['arch_base_channels'] = self._decode_parameter(params[:, idx], 'base_channels')

        if self.config.architecture.include_kernel_size and 'kernel_size' in arch_spec:
            idx = self._param_indices['kernel_size']
            decoded = self._decode_parameter(params[:, idx], 'kernel_size')
            # For choice parameters, get the actual choice value
            choices = arch_spec['kernel_size']['choices']
            choice_idx = np.floor(params[:, idx] * len(choices)).astype(int)
            choice_idx = np.clip(choice_idx, 0, len(choices) - 1)
            features['arch_kernel_size'] = np.array([choices[i] for i in choice_idx], dtype=np.float32)

        if self.config.architecture.include_activation_encoding and 'activation' in arch_spec:
            idx = self._param_indices['activation']
            one_hot = self._one_hot_encode_choice(params[:, idx], 'activation')
            for choice_name, binary_array in one_hot.items():
                features[f'arch_activation_{choice_name}'] = binary_array

        if self.config.architecture.include_dropout_rate and 'dropout_rate' in arch_spec:
            idx = self._param_indices['dropout_rate']
            features['arch_dropout_rate'] = self._decode_parameter(params[:, idx], 'dropout_rate')

        if self.config.architecture.include_total_parameters:
            if 'num_layers' in arch_spec and 'base_channels' in arch_spec:
                idx_layers = self._param_indices['num_layers']
                idx_channels = self._param_indices['base_channels']
                num_layers = self._decode_parameter(params[:, idx_layers], 'num_layers')
                base_channels = self._decode_parameter(params[:, idx_channels], 'base_channels')
                # Rough estimate: depth × width^2 (parameters ≈ channels^2 per layer)
                total_params = num_layers * (base_channels ** 2)
                features['arch_total_params_log10'] = np.log10(total_params + 1e-8).astype(np.float32)

        return features

    def _extract_stochastic(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract stochastic parameter features."""
        features = {}
        stoch_spec = self.param_spec.get('stochastic', {})

        if self.config.stochastic.include_noise_scale_log and 'noise_scale' in stoch_spec:
            idx = self._param_indices['noise_scale']
            decoded = self._decode_parameter(params[:, idx], 'noise_scale')
            features['stoch_noise_scale_log'] = np.log10(decoded + 1e-10).astype(np.float32)

        if self.config.stochastic.include_noise_schedule_encoding and 'noise_schedule' in stoch_spec:
            idx = self._param_indices['noise_schedule']
            one_hot = self._one_hot_encode_choice(params[:, idx], 'noise_schedule')
            for choice_name, binary_array in one_hot.items():
                features[f'stoch_schedule_{choice_name}'] = binary_array

        if self.config.stochastic.include_spatial_correlation and 'spatial_correlation' in stoch_spec:
            idx = self._param_indices['spatial_correlation']
            features['stoch_spatial_corr'] = self._decode_parameter(params[:, idx], 'spatial_correlation')

        if self.config.stochastic.include_noise_type_encoding and 'noise_type' in stoch_spec:
            idx = self._param_indices['noise_type']
            one_hot = self._one_hot_encode_choice(params[:, idx], 'noise_type')
            for choice_name, binary_array in one_hot.items():
                features[f'stoch_noise_{choice_name}'] = binary_array

        if self.config.stochastic.include_stochasticity_score and 'noise_scale' in stoch_spec:
            idx = self._param_indices['noise_scale']
            noise_scale = self._decode_parameter(params[:, idx], 'noise_scale')

            # Combined stochasticity metric (can be enhanced with more features)
            # For now: log(noise_scale) normalized to [0,1]
            spec = stoch_spec['noise_scale']
            low, high = spec['bounds']
            log_low, log_high = np.log10(low), np.log10(high)
            log_scale = np.log10(noise_scale + 1e-10)
            score = (log_scale - log_low) / (log_high - log_low)
            features['stoch_score'] = np.clip(score, 0, 1).astype(np.float32)

        return features

    def _extract_operator(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract operator configuration features."""
        features = {}
        op_spec = self.param_spec.get('operator', {})

        if self.config.operator.include_normalization_encoding and 'normalization' in op_spec:
            idx = self._param_indices['normalization']
            one_hot = self._one_hot_encode_choice(params[:, idx], 'normalization')
            for choice_name, binary_array in one_hot.items():
                features[f'op_norm_{choice_name}'] = binary_array

        if self.config.operator.include_grid_size and 'grid_size' in op_spec:
            idx = self._param_indices['grid_size']
            # Get actual grid size value
            choices = op_spec['grid_size']['choices']
            choice_idx = np.floor(params[:, idx] * len(choices)).astype(int)
            choice_idx = np.clip(choice_idx, 0, len(choices) - 1)
            features['op_grid_size'] = np.array([choices[i] for i in choice_idx], dtype=np.float32)

        if self.config.operator.include_grid_size_class and 'grid_size' in op_spec:
            idx = self._param_indices['grid_size']
            one_hot = self._one_hot_encode_choice(params[:, idx], 'grid_size')
            for choice_name, binary_array in one_hot.items():
                features[f'op_grid_{choice_name}'] = binary_array

        return features

    def _extract_evolution(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract evolution policy features."""
        features = {}
        evol_spec = self.param_spec.get('evolution', {})

        if self.config.evolution.include_update_policy_encoding and 'update_policy' in evol_spec:
            idx = self._param_indices['update_policy']
            one_hot = self._one_hot_encode_choice(params[:, idx], 'update_policy')
            for choice_name, binary_array in one_hot.items():
                features[f'evol_policy_{choice_name}'] = binary_array

        if self.config.evolution.include_dt_log and 'dt' in evol_spec:
            idx = self._param_indices['dt']
            decoded = self._decode_parameter(params[:, idx], 'dt')
            features['evol_dt_log'] = np.log10(decoded + 1e-10).astype(np.float32)

        if self.config.evolution.include_alpha and 'alpha' in evol_spec:
            idx = self._param_indices['alpha']
            features['evol_alpha'] = self._decode_parameter(params[:, idx], 'alpha')

        return features

    def _extract_stratification(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Sobol stratification metadata."""
        N, P = params.shape
        features = {}

        if self.config.stratification.include_stratum_ids:
            # Assign stratum per dimension
            bins = np.linspace(0, 1, self._num_strata_per_dim + 1)
            strata = np.digitize(params, bins) - 1
            strata = np.clip(strata, 0, self._num_strata_per_dim - 1)  # Safety clamp

            for dim in range(P):
                features[f"strat_dim{dim}_stratum"] = strata[:, dim].astype(np.float32)

        if self.config.stratification.include_stratum_hash:
            # Composite stratum ID
            bins = np.linspace(0, 1, self._num_strata_per_dim + 1)
            strata = np.digitize(params, bins) - 1
            strata = np.clip(strata, 0, self._num_strata_per_dim - 1)

            # Create unique hash for each stratum combination
            stratum_hash = np.zeros(N, dtype=np.float32)
            for dim in range(P):
                stratum_hash += strata[:, dim] * (self._num_strata_per_dim ** dim)

            features["strat_hash"] = stratum_hash

        if self.config.stratification.include_distance_to_boundary:
            # Min distance to any edge of [0,1]^P
            dist_to_lower = params.min(axis=1)
            dist_to_upper = (1 - params).min(axis=1)
            features["strat_boundary_dist"] = np.minimum(dist_to_lower, dist_to_upper).astype(np.float32)

        if self.config.stratification.include_extremeness_score:
            # L2 distance from center [0.5]^P
            center = np.full((P,), 0.5)
            extremeness = np.linalg.norm(params - center, axis=1)
            features["strat_extremeness"] = extremeness.astype(np.float32)

        return features

    def get_feature_registry(self) -> FeatureRegistry:
        """Get NOP feature registry."""
        return self.registry
