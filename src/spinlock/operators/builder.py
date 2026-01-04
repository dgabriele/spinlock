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
import torch._dynamo
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
from .afno import SpectralMixingBlock, AFNOBlock
from .u_afno import UAFNOOperator
from .simple_cnn import SimpleCNNOperator
from .parameters import OperatorParameters
from typing import Literal

# Module-level flag to prevent repetitive JIT compilation logging
_JIT_COMPILE_LOGGED = False


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
        # AFNO blocks
        "SpectralMixingBlock": SpectralMixingBlock,
        "AFNOBlock": AFNOBlock,
        # Operator classes
        "UAFNOOperator": UAFNOOperator,
        "SimpleCNNOperator": SimpleCNNOperator,
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

    def build_simple_cnn(self, params: Union[Dict[str, Any], OperatorParameters]) -> SimpleCNNOperator:
        """
        Build a simple CNN operator from parameters.

        Constructs a SimpleCNNOperator with:
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
            SimpleCNNOperator with get_intermediate_features() support

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

            # Extract intermediate features
            features = model.get_intermediate_features(x, extract_from="all")
            ```
        """
        conv_blocks = []
        current_channels = params["input_channels"]
        base_channels = params["base_channels"]

        # Global settings
        global_kwargs = {
            "kernel_size": params.get("kernel_size", 3),
            "activation": params.get("activation", "gelu"),
            "normalization": params.get("normalization", "instance"),
            "dropout": params.get("dropout_rate", 0.0),
        }

        # Input layer (first conv block)
        conv_blocks.append(ConvBlock(current_channels, base_channels, **global_kwargs))
        current_channels = base_channels

        # Hidden layers
        num_layers = params["num_layers"]
        for i in range(num_layers):
            # Gradually increase channels
            out_channels = base_channels * (2 ** min(i, 2))  # Cap at 4x base

            # Use residual blocks for deeper networks
            if num_layers > 3:
                conv_blocks.append(
                    ResidualBlock(current_channels, out_channels, num_convs=2, **global_kwargs)
                )
            else:
                conv_blocks.append(ConvBlock(current_channels, out_channels, **global_kwargs))

            current_channels = out_channels

        # Optional stochastic block (separate from conv blocks)
        noise_type = params.get("noise_type") if isinstance(params, dict) else params.noise_type
        noise_scale = params.get("noise_scale") if isinstance(params, dict) else params.noise_scale
        stochastic_block = None
        if noise_type is not None and noise_scale is not None:
            stochastic_block = StochasticBlock(
                current_channels,
                noise_type=noise_type,
                noise_scale=noise_scale,
                always_active=True,  # For dataset generation
            )

        # Output layer (separate from conv blocks)
        output_layer = OutputLayer(
            current_channels, params["output_channels"], activation="none"  # Raw outputs
        )

        return SimpleCNNOperator(
            conv_blocks=nn.ModuleList(conv_blocks),
            stochastic_block=stochastic_block,
            output_layer=output_layer,
        )

    def build_u_afno(self, params: Union[Dict[str, Any], OperatorParameters]) -> nn.Module:
        """
        Build a U-AFNO operator from parameters.

        Constructs a U-Net encoder → AFNO bottleneck → U-Net decoder architecture
        with optional stochastic noise injection.

        Args:
            params: OperatorParameters dataclass or dict with keys:
                - input_channels: int
                - output_channels: int
                - base_channels: int
                - encoder_levels: int (default: 3)
                - modes: int (default: 32)
                - afno_blocks: int (default: 4)
                - hidden_dim: int (optional, default: 2x bottleneck channels)
                - activation: str (default: "gelu")
                - normalization: str (default: "instance")
                - noise_type: str (optional, e.g., "gaussian")
                - noise_scale: float (optional)
                - noise_schedule: str (default: "constant")
                - spatial_correlation: float (default: 0.0)

        Returns:
            UAFNOOperator neural operator

        Example:
            ```python
            params = {
                "input_channels": 3,
                "output_channels": 3,
                "base_channels": 32,
                "encoder_levels": 3,
                "modes": 32,
                "afno_blocks": 4,
                "activation": "gelu",
                "noise_type": "gaussian",
                "noise_scale": 0.05
            }
            model = builder.build_u_afno(params)
            ```
        """
        # Handle both dict and OperatorParameters
        def get_param(key: str, default: Any = None) -> Any:
            if isinstance(params, dict):
                return params.get(key, default)
            return getattr(params, key, default)

        return UAFNOOperator(
            in_channels=get_param("input_channels", 3),
            out_channels=get_param("output_channels", 3),
            base_channels=get_param("base_channels", 32),
            encoder_levels=get_param("encoder_levels", 3),
            modes=get_param("modes", 32),
            afno_blocks=get_param("afno_blocks", 4),
            hidden_dim=get_param("hidden_dim", None),
            blocks_per_level=get_param("blocks_per_level", 2),
            normalization=get_param("normalization", "instance"),
            activation=get_param("activation", "gelu"),
            noise_type=get_param("noise_type", None),
            noise_scale=get_param("noise_scale", None),
            noise_schedule=get_param("noise_schedule", "constant"),
            spatial_correlation=get_param("spatial_correlation", 0.0),
        )

    def build_from_sampled_params(
        self,
        unit_params: NDArray[np.float64],
        parameter_space_config: Dict[str, Dict[str, Any]],
        operator_type: Literal["cnn", "u_afno"] = "cnn",
    ) -> nn.Module:
        """
        Complete pipeline: unit parameters → mapped parameters → operator.

        Args:
            unit_params: Sampled parameters in [0,1]^d
            parameter_space_config: Parameter specifications
            operator_type: Type of operator to build ("cnn" or "u_afno")

        Returns:
            Neural operator model

        Example:
            ```python
            # Sample from parameter space
            samples = sampler.sample(1)
            unit_params = samples[0]

            # Build CNN operator (default)
            operator = builder.build_from_sampled_params(
                unit_params,
                config.parameter_space.architecture
            )

            # Build U-AFNO operator
            operator = builder.build_from_sampled_params(
                unit_params,
                config.parameter_space.architecture,
                operator_type="u_afno"
            )
            ```
        """
        # Map to actual values
        mapped_params = self.map_parameters(unit_params, parameter_space_config)

        # Build operator based on type
        if operator_type == "u_afno":
            return self.build_u_afno(mapped_params)
        else:
            return self.build_simple_cnn(mapped_params)


class NeuralOperator(nn.Module):
    """
    Wrapper for neural operators with stochastic forward passes.

    Provides:
    - Multiple stochastic realizations
    - Seed control for reproducibility
    - Metadata tracking
    - torch.compile() optimization support

    Example:
        ```python
        base_model = builder.build_simple_cnn(params)
        operator = NeuralOperator(base_model, name="stochastic_cnn")

        # Enable torch.compile() for 2-3× speedup
        operator.enable_compile(mode="max-autotune")

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
        use_compile: bool = False,
        compile_mode: str = "max-autotune",
    ):
        super().__init__()
        self.model = model
        self.name = name
        self.metadata = metadata or {}
        self._compiled = False
        self._compile_mode = compile_mode

        # Optionally compile immediately
        if use_compile:
            self.enable_compile(mode=compile_mode)

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

    def enable_compile(self, mode: str = "max-autotune") -> None:
        """
        Enable torch.compile() JIT compilation for faster inference.

        Args:
            mode: Compilation mode
                - "default": Balanced compilation (fastest compile time)
                - "reduce-overhead": Minimize Python overhead
                - "max-autotune": Aggressive optimization (best performance, slow compile)

        Expected speedup: 1.5-2× for typical CNN operators

        Example:
            ```python
            operator = NeuralOperator(model)
            operator.enable_compile(mode="max-autotune")  # One-time compilation cost

            # Subsequent forward passes are faster
            out = operator(x)  # Compiled execution
            ```

        WARNING - GPU Memory Leak with Diverse Architectures:
            torch.compile() caches compiled kernels in GPU memory. When generating
            datasets with >64 unique operator architectures (varying channels, layers,
            kernel sizes), the compilation cache accumulates and causes OOM errors.

            Mitigation strategies:
            1. Disable compilation for diverse datasets (set performance.compile=False)
            2. Use periodic cache clearing (torch._dynamo.reset() every 50 operators)
            3. Reduce cache_size_limit below 64 if using fewer unique architectures

            Symptoms:
            - GPU memory usage jumps 50%+ and never decreases
            - OOM errors after processing 50-70 operators
            - Recompilation limit warnings

        Note:
            - Requires PyTorch >= 2.0
            - First forward pass triggers compilation (slow)
            - Subsequent passes are 1.5-2× faster
            - Compilation is cached per unique architecture
        """
        if self._compiled:
            # Already compiled, skip silently
            return

        # Check PyTorch version
        if not hasattr(torch, "compile"):
            print(f"[WARNING] torch.compile() requires PyTorch >= 2.0 (current: {torch.__version__})")
            print("          Skipping compilation, using eager mode")
            return

        # Compile the model
        try:
            global _JIT_COMPILE_LOGGED

            # Configure torch._dynamo for variable operator architectures
            # This allows operators with different parameter shapes to share compiled graphs
            # instead of hitting the recompilation limit (default: 8)
            torch._dynamo.config.force_parameter_static_shapes = False
            torch._dynamo.config.cache_size_limit = 64  # Increased from default 8

            self.model = torch.compile(
                self.model,
                mode=mode,
                fullgraph=False,  # Allow partial graphs to avoid excessive recompilation
                dynamic=False     # Static shapes for better optimization
            )
            self._compiled = True
            self._compile_mode = mode

            # Only log the first time (avoid repetitive logging for multiple operators)
            if not _JIT_COMPILE_LOGGED:
                print(f"[INFO] Enabled torch.compile(mode='{mode}') for all operators")
                print("       Configured for variable architectures (dynamic parameters)")
                print("       First forward pass will be slow (JIT compilation)")
                print("       Subsequent passes: 1.5-2× faster")
                _JIT_COMPILE_LOGGED = True
        except Exception as e:
            print(f"[ERROR] Failed to compile operator '{self.name}': {e}")
            print("        Falling back to eager mode")

    def disable_compile(self) -> None:
        """
        Disable torch.compile() and revert to eager mode.

        Useful for debugging or when torch.compile() causes issues.

        Example:
            ```python
            operator.enable_compile()
            # ... compilation issues ...
            operator.disable_compile()  # Revert to eager mode
            ```
        """
        if not self._compiled:
            print(f"[INFO] Operator '{self.name}' is not compiled")
            return

        # Need to recreate the model from original definition
        # This is a limitation of torch.compile - can't "undo" compilation
        print(f"[WARNING] Cannot directly undo compilation for '{self.name}'")
        print("          Please recreate the operator to use eager mode")
        print("          Or restart Python session")

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

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        extract_from: str = "all",
        skip_levels: list[int] | None = None,
        layer_indices: list[int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features from the underlying model.

        Delegates to the wrapped model's get_intermediate_features() method.
        Supports both SimpleCNNOperator and UAFNOOperator.

        Args:
            x: Input tensor [B, C, H, W]
            extract_from: Which latents to extract:
                - For U-AFNO: "bottleneck", "skips", or "all"
                - For CNN: "early", "mid", "pre_output", or "all"
            skip_levels: (U-AFNO only) Which encoder levels to extract
            layer_indices: (CNN only) Which layer indices to extract

        Returns:
            Dict of intermediate feature tensors

        Raises:
            AttributeError: If underlying model doesn't support this method
        """
        if not hasattr(self.model, "get_intermediate_features"):
            raise AttributeError(
                "Operator must have get_intermediate_features() method. "
                "Supported: SimpleCNNOperator, UAFNOOperator."
            )

        # Detect operator type and call with appropriate kwargs
        if hasattr(self.model, "conv_blocks"):
            # SimpleCNNOperator
            return self.model.get_intermediate_features(
                x, extract_from=extract_from, layer_indices=layer_indices
            )
        else:
            # UAFNOOperator
            return self.model.get_intermediate_features(
                x, extract_from=extract_from, skip_levels=skip_levels
            )
