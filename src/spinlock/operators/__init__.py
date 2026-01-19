"""Neural operator builders and blocks for Spinlock."""

from .blocks import (
    BaseBlock,
    ConvBlock,
    ResidualBlock,
    FiLMResidualBlock,
    StochasticBlock,
    DownsampleBlock,
    UpsampleBlock,
    OutputLayer
)
from .afno import SpectralMixingBlock, AFNOBlock, FiLMAFNOBlock
from .u_afno import (
    UAFNOOperator,
    UNetEncoder,
    UNetDecoder,
    FiLMUNetEncoder,
    FiLMUNetDecoder,
    FiLMUAFNOOperator,
)
from .film import (
    FiLMConfig,
    FiLMLayerSpec,
    FiLMLayer,
    FiLMGenerator,
    FiLMProjectionHead,
    build_film_layer_specs,
    estimate_film_parameter_overhead,
)
from .simple_cnn import SimpleCNNOperator
from .builder import OperatorBuilder, NeuralOperator
from .parameters import (
    OperatorParameters,
    SamplingMetrics,
    DatasetMetadata,
    BatchMetadata
)
from .partitioning import (
    ArchitecturePartition,
    bucket_channels,
    get_architecture_signature,
    partition_operators,
    get_partition_stats,
    print_partition_summary,
)

__all__ = [
    # Base blocks
    "BaseBlock",
    "ConvBlock",
    "ResidualBlock",
    "FiLMResidualBlock",
    "StochasticBlock",
    "DownsampleBlock",
    "UpsampleBlock",
    "OutputLayer",
    # AFNO blocks
    "SpectralMixingBlock",
    "AFNOBlock",
    "FiLMAFNOBlock",
    # U-AFNO operator
    "UAFNOOperator",
    "UNetEncoder",
    "UNetDecoder",
    # FiLM U-AFNO operator
    "FiLMUNetEncoder",
    "FiLMUNetDecoder",
    "FiLMUAFNOOperator",
    # FiLM modulation
    "FiLMConfig",
    "FiLMLayerSpec",
    "FiLMLayer",
    "FiLMGenerator",
    "FiLMProjectionHead",
    "build_film_layer_specs",
    "estimate_film_parameter_overhead",
    # CNN operator
    "SimpleCNNOperator",
    # Builder and wrapper
    "OperatorBuilder",
    "NeuralOperator",
    # Parameters and metadata
    "OperatorParameters",
    "SamplingMetrics",
    "DatasetMetadata",
    "BatchMetadata",
    # Partitioning (CUDA optimization)
    "ArchitecturePartition",
    "bucket_channels",
    "get_architecture_signature",
    "partition_operators",
    "get_partition_stats",
    "print_partition_summary",
]
