"""TD (Temporal Dynamics) feature family configuration."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class TDConfig(BaseModel):
    """Configuration for TD (Temporal Dynamics) feature extraction.

    TD features preserve full temporal trajectories for VQ-VAE tokenization,
    unlike SDF which aggregates to scalar summaries.

    Attributes:
        version: Feature family version (semantic versioning)
        enabled: Whether TD extraction is enabled
        include_per_timestep: Include SDF's per-timestep features [N, M, T, 46]
        per_timestep_categories: Which SDF categories to include
        include_derived_curves: Compute additional temporal curves
        derived_features: Which derived features to extract
        store_sequences: Store full time series [N, M, T, D_td]
        store_context: Store global operator-level metadata [N, D_context]
        store_aggregated_cache: Cache encoder output [N, M, D_out] (optional)

    Example:
        >>> config = TDConfig(
        ...     include_per_timestep=True,
        ...     include_derived_curves=True,
        ...     derived_features=["energy_trajectory", "variance_trajectory"]
        ... )
        >>> config.version
        '1.0.0'
    """

    version: str = "1.0.0"
    enabled: bool = True

    # Per-timestep features (loaded from SDF)
    include_per_timestep: bool = True
    per_timestep_categories: List[str] = Field(
        default_factory=lambda: ["spatial", "spectral", "cross_channel"],
        description="SDF categories to include in TD features",
    )

    # Derived temporal curves
    include_derived_curves: bool = True
    derived_features: List[
        Literal[
            "energy_trajectory",
            "variance_trajectory",
            "smoothness_trajectory",
            "channel_correlation_trajectory",
        ]
    ] = Field(
        default_factory=lambda: [
            "energy_trajectory",
            "variance_trajectory",
            "smoothness_trajectory",
        ],
        description="Temporal curves to compute from trajectories",
    )

    # Storage options
    store_sequences: bool = True
    store_context: bool = True
    store_aggregated_cache: bool = Field(
        default=False,
        description="Optionally cache encoder output to HDF5 (for repeated training runs)",
    )

    class Config:
        """Pydantic config."""

        frozen = False  # Allow modification
        extra = "forbid"  # Reject unknown fields
