"""
Operator builder: Factory for creating neural operators from parameters.

Translates sampled parameter sets into PyTorch models using:
- Registry pattern for extensible block types
- Parameter mapping from [0,1]^d to actual values
- Modular composition of building blocks

Design principles:
- Factory pattern: parameters → operators
- Registry: easy to add new block types
- Composition: build complex from simple
"""

import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Type, List, Union
from .blocks import (
    ConvBlock,
    ResidualBlock,
    StochasticBlock,
    DownsampleBlock,
    UpsampleBlock,
    OutputLayer,
)
from .parameters import OperatorParameters


class OperatorBuilder:
    """
    Build neural operators from sampled parameters.

    Features:
    - Parameter mapping from [0,1] to actual ranges
    - Registry pattern for block types
    - Sequential or custom architecture construction

    Example:
        ```python
        builder = OperatorBuilder()

        # From parameter dict
        params = {
            "num_layers": 3,
            "base_channels": 32,
            "input_channels": 3,
            "output_channels": 3,
            "grid_size": 64,
            ...
        }
        operator = builder.build_simple_cnn(params)

        # Forward pass
        x = torch.randn(8, 3, 64, 64)
        out = operator(x)  # Shape: (8, 3, 64, 64)
        ```
    """

    # Registry of available block types (extensible)
    BLOCK_REGISTRY: Dict[str, Type[nn.Module]] = {
        "ConvBlock": ConvBlock,
        "ResidualBlock": ResidualBlock,
        "StochasticBlock": StochasticBlock,
        "DownsampleBlock": DownsampleBlock,
        "UpsampleBlock": UpsampleBlock,
        "OutputLayer": OutputLayer,
    }

    @classmethod
    def register_block(cls, name: str, block_class: Type[nn.Module]) -> None:
        """
        Register a custom block type.

        Args:
            name: Block type name
            block_class: Block class

        Example:
            ```python
            class CustomBlock(BaseBlock):
                ...

            OperatorBuilder.register_block("CustomBlock", CustomBlock)
            ```
        """
        cls.BLOCK_REGISTRY[name] = block_class

    def map_parameters(
        self, unit_params: NDArray[np.float64], param_spec: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Map parameters from [0,1]^d to actual values.

        Args:
            unit_params: Parameter array in [0,1]^d, shape (d,)
            param_spec: Parameter specifications from config

        Returns:
            Dictionary of parameter_name -> value

        Example:
            ```python
            spec = {
                "num_layers": {"type": "integer", "bounds": [2, 6]},
                "noise_scale": {"type": "continuous", "bounds": [0.001, 0.1], "log_scale": True}
            }
            params = builder.map_parameters(np.array([0.5, 0.3]), spec)
            # {"num_layers": 4, "noise_scale": ~0.01}
            ```
        """
        mapped = {}
        idx = 0

        for name, spec in param_spec.items():
            if idx >= len(unit_params):
                raise ValueError(f"Not enough unit parameters for {name}")

            u = unit_params[idx]
            param_type = spec["type"]

            if param_type == "integer":
                low, high = spec["bounds"]
                mapped[name] = int(np.round(low + u * (high - low)))

            elif param_type == "continuous":
                low, high = spec["bounds"]
                if spec.get("log_scale", False):
                    # Log-uniform sampling
                    log_low, log_high = np.log10(low), np.log10(high)
                    mapped[name] = float(10 ** (log_low + u * (log_high - log_low)))
                else:
                    # Linear sampling
                    mapped[name] = float(low + u * (high - low))

            elif param_type == "choice":
                choices = spec["choices"]
                weights = spec.get("weights", None)

                if weights is None:
                    # Uniform distribution (backward compatible)
                    idx_choice = int(np.floor(u * len(choices)))
                    idx_choice = min(idx_choice, len(choices) - 1)  # Handle u=1.0
                else:
                    # Weighted distribution using cumulative probabilities
                    cumulative_weights = np.cumsum(weights)
                    idx_choice = np.searchsorted(cumulative_weights, u, side='right')
                    idx_choice = min(idx_choice, len(choices) - 1)  # Safety clamp

                mapped[name] = choices[idx_choice]

            elif param_type == "boolean":
                mapped[name] = u > 0.5

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            idx += 1

        return mapped

    def map_parameters_safe(
        self, unit_params: NDArray[np.float64], param_spec: Dict[str, Dict[str, Any]]
    ) -> OperatorParameters:
        """
        Map unit parameters to actual values, returning type-safe dataclass.

        This is a type-safe wrapper around map_parameters() that returns
        an OperatorParameters dataclass instead of Dict[str, Any].

        Args:
            unit_params: Array of parameters in [0,1]
            param_spec: Parameter specifications

        Returns:
            OperatorParameters dataclass with type-safe fields

        Example:
            ```python
            spec = {
                "num_layers": {"type": "integer", "bounds": [2, 6]},
                "noise_scale": {"type": "continuous", "bounds": [0.001, 0.1], "log_scale": True}
            }
            params = builder.map_parameters_safe(np.array([0.5, 0.3]), spec)
            # params.num_layers is int (type-safe)
            # params.noise_scale is float (type-safe)
            ```
        """
        param_dict = self.map_parameters(unit_params, param_spec)
        return OperatorParameters.from_dict(param_dict)

    def build_simple_cnn(self, params: Union[Dict[str, Any], OperatorParameters]) -> nn.Module:
        """
        Build a simple CNN operator from parameters.

        Constructs a sequential model:
        1. Input convolution
        2. N hidden layers (conv or residual blocks)
        3. Optional stochastic block
        4. Output layer

        Args:
            params: OperatorParameters dataclass or dict with keys:
                - num_layers: int
                - base_channels: int
                - input_channels: int
                - output_channels: int
                - kernel_size: int (or list)
                - activation: str
                - normalization: str
                - dropout_rate: float
                - use_batch_norm: bool
                - noise_type: str (optional)
                - noise_scale: float (optional)

                Accepts both Dict[str, Any] (backward compat) and
                OperatorParameters (type-safe, recommended)

        Returns:
            Sequential neural operator

        Example:
            ```python
            params = {
                "num_layers": 3,
                "base_channels": 32,
                "input_channels": 3,
                "output_channels": 3,
                "kernel_size": 3,
                "activation": "gelu",
                "normalization": "instance",
                "dropout_rate": 0.1,
                "noise_type": "gaussian",
                "noise_scale": 0.05
            }
            model = builder.build_simple_cnn(params)
            ```
        """
        layers = []
        current_channels = params["input_channels"]
        base_channels = params["base_channels"]

        # Global settings
        global_kwargs = {
            "kernel_size": params.get("kernel_size", 3),
            "activation": params.get("activation", "gelu"),
            "normalization": params.get("normalization", "instance"),
            "dropout": params.get("dropout_rate", 0.0),
        }

        # Input layer
        layers.append(ConvBlock(current_channels, base_channels, **global_kwargs))
        current_channels = base_channels

        # Hidden layers
        num_layers = params["num_layers"]
        for i in range(num_layers):
            # Gradually increase channels
            out_channels = base_channels * (2 ** min(i, 2))  # Cap at 4x base

            # Use residual blocks for deeper networks
            if num_layers > 3:
                layers.append(
                    ResidualBlock(current_channels, out_channels, num_convs=2, **global_kwargs)
                )
            else:
                layers.append(ConvBlock(current_channels, out_channels, **global_kwargs))

            current_channels = out_channels

        # Optional stochastic block
        noise_type = params.get("noise_type") if isinstance(params, dict) else params.noise_type
        noise_scale = params.get("noise_scale") if isinstance(params, dict) else params.noise_scale
        if noise_type is not None and noise_scale is not None:
            layers.append(
                StochasticBlock(
                    current_channels,
                    noise_type=noise_type,
                    noise_scale=noise_scale,
                    always_active=True,  # For dataset generation
                )
            )

        # Output layer
        layers.append(
            OutputLayer(
                current_channels, params["output_channels"], activation="none"  # Raw outputs
            )
        )

        return nn.Sequential(*layers)

    def build_from_sampled_params(
        self, unit_params: NDArray[np.float64], parameter_space_config: Dict[str, Dict[str, Any]]
    ) -> nn.Module:
        """
        Complete pipeline: unit parameters → mapped parameters → operator.

        Args:
            unit_params: Sampled parameters in [0,1]^d
            parameter_space_config: Parameter specifications

        Returns:
            Neural operator model

        Example:
            ```python
            # Sample from parameter space
            samples = sampler.sample(1)
            unit_params = samples[0]

            # Build operator
            operator = builder.build_from_sampled_params(
                unit_params,
                config.parameter_space.architecture
            )
            ```
        """
        # Map to actual values
        mapped_params = self.map_parameters(unit_params, parameter_space_config)

        # Build operator
        return self.build_simple_cnn(mapped_params)


class NeuralOperator(nn.Module):
    """
    Wrapper for neural operators with stochastic forward passes.

    Provides:
    - Multiple stochastic realizations
    - Seed control for reproducibility
    - Metadata tracking

    Example:
        ```python
        base_model = builder.build_simple_cnn(params)
        operator = NeuralOperator(base_model, name="stochastic_cnn")

        # Single forward pass
        out = operator(x)

        # Multiple realizations
        realizations = operator.generate_realizations(x, num_realizations=10, seed=42)
        # Shape: (batch, 10, channels, height, width)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        name: str = "neural_operator",
        metadata: Union[Dict[str, Any], None] = None,
    ):
        super().__init__()
        self.model = model
        self.name = name
        self.metadata = metadata or {}

    def forward(self, x: torch.Tensor, seed: Union[int, None] = None) -> torch.Tensor:
        """
        Forward pass with optional seed for reproducibility.

        Args:
            x: Input tensor [B, C, H, W]
            seed: Optional random seed

        Returns:
            Output tensor [B, C_out, H, W]
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        return self.model(x)

    def generate_realizations(
        self, x: torch.Tensor, num_realizations: int, base_seed: Union[int, None] = None
    ) -> torch.Tensor:
        """
        Generate multiple stochastic realizations.

        Args:
            x: Input tensor [B, C, H, W]
            num_realizations: Number of realizations
            base_seed: Base seed for reproducibility

        Returns:
            Output tensor [B, M, C_out, H, W] where M = num_realizations

        Example:
            ```python
            x = torch.randn(4, 3, 64, 64)
            realizations = operator.generate_realizations(x, num_realizations=10, base_seed=42)
            print(realizations.shape)  # (4, 10, 3, 64, 64)
            ```
        """
        realizations = []

        for m in range(num_realizations):
            seed = None
            if base_seed is not None:
                seed = base_seed + m

            realization = self.forward(x, seed=seed)
            realizations.append(realization)

        return torch.stack(realizations, dim=1)  # [B, M, C, H, W]

    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def summary(self) -> str:
        """Return model summary string."""
        summary = [
            f"NeuralOperator: {self.name}",
            f"Parameters: {self.num_parameters:,}",
            f"Metadata: {self.metadata}",
        ]
        return "\n".join(summary)
