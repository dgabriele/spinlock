"""
NOP (Neural Operator Parameter) feature family configuration.

Provides configuration schemas for extracting parameter-derived features
from [0,1]^P unit hypercube and mapped operator configurations.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class ArchitectureParamsConfig(BaseModel):
    """Architecture feature extraction configuration.

    Extracts features from neural network architecture parameters:
    - Layer depth and width
    - Kernel sizes
    - Activation function types
    - Regularization parameters
    """
    enabled: bool = True
    include_depth: bool = True          # num_layers
    include_width: bool = True          # base_channels
    include_kernel_size: bool = True    # Convolutional kernel size
    include_activation_encoding: bool = True  # One-hot encoding for activation types
    include_dropout_rate: bool = True   # Dropout probability
    include_total_parameters: bool = True  # Computed: depth × width (log scale)


class ArchitectureStochasticConfig(BaseModel):
    """Stochastic parameter feature extraction configuration.

    Extracts features from stochastic evolution parameters:
    - Noise scales and schedules
    - Spatial correlation lengths
    - Noise distribution types
    - Combined stochasticity metrics
    """
    enabled: bool = True
    include_noise_scale_log: bool = True      # Log10 of noise scale
    include_noise_schedule_encoding: bool = True  # One-hot: constant/annealing/periodic
    include_spatial_correlation: bool = True  # Spatial correlation length
    include_noise_type_encoding: bool = True  # One-hot: gaussian/laplace
    include_stochasticity_score: bool = True  # Combined metric


class ArchitectureOperatorConfig(BaseModel):
    """Operator configuration feature extraction.

    Extracts features from operator-level configuration:
    - Normalization strategies
    - Grid resolutions
    - Computational settings
    """
    enabled: bool = True
    include_normalization_encoding: bool = True  # One-hot: batch/instance/none
    include_grid_size: bool = True              # Grid resolution (64/128/256)
    include_grid_size_class: bool = True        # One-hot encoding of grid size


class ArchitectureEvolutionConfig(BaseModel):
    """Evolution policy feature extraction configuration.

    Extracts features from temporal evolution policies:
    - Update policy types (residual/convex/autoregressive)
    - Integration timesteps
    - Mixing parameters
    """
    enabled: bool = True
    include_update_policy_encoding: bool = True  # One-hot: residual/convex/autoregressive
    include_dt_log: bool = True                  # Log10 of integration timestep
    include_alpha: bool = True                   # Mixing parameter for convex policy


class ArchitectureStratificationConfig(BaseModel):
    """Sobol stratification metadata extraction configuration.

    Extracts features from Sobol sampling stratification:
    - Per-dimension stratum IDs
    - Composite stratum hashes
    - Boundary proximity metrics
    - Extremeness scores (distance from hypercube center)
    """
    enabled: bool = True
    include_stratum_ids: bool = True         # Per-dimension stratum ID
    include_stratum_hash: bool = True        # Composite stratum ID (unique per cell)
    include_distance_to_boundary: bool = True  # Min distance to [0,1]^P edges
    include_extremeness_score: bool = True   # L2 distance from center [0.5]^P


class ArchitectureConfig(BaseModel):
    """NOP (Neural Operator Parameter) feature family configuration.

    Extracts parameter-derived features from [0,1]^P unit hypercube
    and mapped operator configurations. Features are per-operator only
    (no temporal or realization dimensions).

    Output shape: [N, D_nop] where D_nop depends on enabled categories.

    Categories:
        - architecture: Network architecture parameters
        - stochastic: Stochastic evolution parameters
        - operator: Operator-level configuration
        - evolution: Temporal evolution policies
        - stratification: Sobol sampling metadata

    Future Extensibility:
        Future versions may include learned embeddings via PCA/clustering
        for unsupervised discovery of parameter manifold structure. This
        would enable:
        - Dimensionality reduction via PCA
        - Cluster-based categorical features via k-means
        - Autoencoder-based latent representations
        - Adaptive stratification refinement

    Example:
        >>> config = ArchitectureConfig()
        >>> config.architecture.enabled = True
        >>> config.stratification.include_stratum_ids = True
    """
    version: str = "1.0.0"

    # Feature categories
    architecture: ArchitectureParamsConfig = Field(
        default_factory=ArchitectureParamsConfig,
        description="Architecture parameter features"
    )
    stochastic: ArchitectureStochasticConfig = Field(
        default_factory=ArchitectureStochasticConfig,
        description="Stochastic parameter features"
    )
    operator: ArchitectureOperatorConfig = Field(
        default_factory=ArchitectureOperatorConfig,
        description="Operator configuration features"
    )
    evolution: ArchitectureEvolutionConfig = Field(
        default_factory=ArchitectureEvolutionConfig,
        description="Evolution policy features"
    )
    stratification: ArchitectureStratificationConfig = Field(
        default_factory=ArchitectureStratificationConfig,
        description="Sobol stratification metadata"
    )

    # Future extensibility (placeholder for documentation)
    include_learned_embeddings: bool = Field(
        default=False,
        description="Enable learned embeddings (PCA/clustering) - not yet implemented"
    )
