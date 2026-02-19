"""Pydantic models for feature grouping configuration and results."""

from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field


class LinkageMethod(str, Enum):
    """Hierarchical clustering linkage."""
    WARD = "ward"
    AVERAGE = "average"
    COMPLETE = "complete"
    SINGLE = "single"


class DistanceMetric(str, Enum):
    """Distance metric for clustering."""
    CORRELATION = "correlation"
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class KSelectionMethod(str, Enum):
    """Method for automatic K selection."""
    SILHOUETTE = "silhouette"
    GAP_STATISTIC = "gap_statistic"
    ELBOW = "elbow"
    MANUAL = "manual"


class ClusteringParams(BaseModel):
    """Parameters for clustering-based grouping."""
    linkage_method: LinkageMethod = LinkageMethod.WARD
    distance_metric: DistanceMetric = DistanceMetric.CORRELATION
    k_selection_method: KSelectionMethod = KSelectionMethod.SILHOUETTE
    num_groups: Optional[int] = None  # Manual K (or upper cap when variance_threshold also set)
    variance_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "If set, K is the minimum number of PCs such that their cumulative explained "
            "variance >= variance_threshold. Features whose unconstrained dominant PC is in "
            "the noise tail (index >= K) are re-allocated to whichever of the top-K PCs has "
            "the highest absolute loading for that feature. "
            "When both variance_threshold and num_groups are specified, "
            "K = min(K_auto, num_groups) — num_groups acts as an upper cap. "
            "When only num_groups is set, K = num_groups (legacy fixed-K behaviour)."
        ),
    )
    min_groups: Optional[int] = Field(None, ge=2, description="Minimum K for silhouette search; None → 2")
    max_groups: Optional[int] = Field(None, ge=2, description="Maximum K for silhouette search; None → D-1")
    subsample_size: Optional[int] = Field(None, description="Max samples for correlation")
    use_gpu: bool = Field(True, description="Use CuPy GPU acceleration if available")


class GradientParams(BaseModel):
    """
    Parameters for gradient-based refinement.

    Core objectives (always computed):
    1. Orthogonality: Minimize inter-group correlation
    2. Informativeness: Maximize per-group feature variance

    Optional custom loss (injected downstream):
    3. Custom loss callback (e.g., VQ-VAE reconstruction loss)
    """
    num_epochs: int = Field(500, ge=1)
    learning_rate: float = Field(0.01, gt=0.0)
    temperature_start: float = Field(1.0, gt=0.0)
    temperature_end: float = Field(0.5, gt=0.0)

    # Loss weights (multi-objective optimization)
    orthogonality_weight: float = Field(1.0, ge=0.0, description="Inter-group correlation penalty")
    informativeness_weight: float = Field(1.0, ge=0.0, description="Per-group variance reward")
    custom_loss_weight: float = Field(0.0, ge=0.0, description="Weight for injected custom loss")

    # Early stopping
    orthogonality_target: float = Field(0.15, ge=0.0, le=1.0)
    device: Literal["cuda", "cpu", "auto"] = "auto"


class PreprocessingParams(BaseModel):
    """Parameters for feature preprocessing."""
    method: Literal["mad", "zscore", "minmax", "none"] = "mad"
    mad_constant: float = Field(1.4826, description="MAD scaling constant")
    clip_outliers: bool = Field(False, description="Clip extreme values")
    clip_std_threshold: float = Field(5.0, gt=0.0, description="Std threshold for clipping")


class SplittingParams(BaseModel):
    """Parameters for recursive mega-group splitting."""
    enabled: bool = Field(False, description="Enable recursive splitting")
    max_group_size: int = Field(40, ge=1)
    max_recursion_depth: int = Field(3, ge=1)
    min_features_per_group: int = Field(3, ge=2)


class GroupingConfig(BaseModel):
    """Complete configuration for feature grouping."""
    method: Literal["correlation", "pca_striped", "pca_raw", "opq"] = Field(
        "correlation",
        description=(
            "Grouping method. 'correlation' uses Ward hierarchical clustering (legacy); "
            "'pca_striped' rotates to PCA basis then assigns PC i → group i%M, giving each "
            "group an equal share of high/medium/low variance (recommended); "
            "'pca_raw' uses PCA loadings as a grouping oracle to assign raw features to groups "
            "by dominant PC — no rotation at inference, per-group pyramid encoders act on raw "
            "feature slices [B, T, G_k] directly; "
            "'opq' uses FAISS OPQ for optimal product quantization (requires faiss-cpu)."
        ),
    )
    preprocessing: PreprocessingParams = Field(default_factory=PreprocessingParams)
    clustering: ClusteringParams = Field(default_factory=ClusteringParams)
    gradient: GradientParams = Field(default_factory=GradientParams)
    splitting: SplittingParams = Field(default_factory=SplittingParams)
    random_seed: Optional[int] = None

    # Pipeline control
    skip_gradient_refinement: bool = Field(False, description="Skip gradient refinement")

    # Validation thresholds
    min_samples_required: int = Field(50, ge=1, description="Minimum samples for grouping")
    min_features_required: int = Field(2, ge=2, description="Minimum features for grouping")

    # Per-family config overrides.  When set, these replace the top-level config
    # for the corresponding feature family.  When None (default), the factory
    # falls back to the family-specific default (InitialGroupingConfig /
    # ThetaGroupingConfig) so non-temporal families are never forced to use the
    # temporal num_groups / variance_threshold settings.
    initial: Optional['GroupingConfig'] = Field(
        None,
        description="Config override for initial-feature grouping.",
    )
    theta: Optional['GroupingConfig'] = Field(
        None,
        description="Config override for theta-feature grouping.",
    )


class TemporalGroupingConfig(GroupingConfig):
    """Temporal features — Ward clustering; K from data unless num_groups is set."""
    clustering: ClusteringParams = Field(
        default_factory=lambda: ClusteringParams(linkage_method=LinkageMethod.WARD)
    )
    min_samples_required: int = 100


class InitialGroupingConfig(GroupingConfig):
    """Initial features — Ward clustering; K from data via silhouette scoring."""
    clustering: ClusteringParams = Field(
        default_factory=lambda: ClusteringParams(linkage_method=LinkageMethod.WARD)
    )
    min_samples_required: int = 50


class ThetaGroupingConfig(GroupingConfig):
    """Theta (parameter) features — Ward clustering; K from data via silhouette scoring."""
    clustering: ClusteringParams = Field(
        default_factory=lambda: ClusteringParams(linkage_method=LinkageMethod.WARD)
    )
    min_samples_required: int = 50
    skip_gradient_refinement: bool = True


class FeatureGroup(BaseModel):
    """A single feature group."""
    name: str
    feature_indices: List[int]
    feature_names: List[str]
    size: int

    @property
    def indices(self) -> List[int]:
        """Get feature indices."""
        return self.feature_indices


class GroupingResult(BaseModel):
    """Result of feature grouping operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    groups: Dict[str, FeatureGroup]
    num_groups: int
    total_features: int
    config: GroupingConfig
    linear_transform: Optional[Any] = Field(
        None,
        description=(
            "LinearTransform applied before grouping (PCA or OPQ rotation). "
            "None for correlation-based grouping. Must be applied to raw temporal "
            "features at inference time before passing to per-group encoders."
        ),
    )

    def to_dict(self) -> Dict[str, List[int]]:
        """Convert to v1-compatible dict format."""
        return {name: group.feature_indices for name, group in self.groups.items()}
