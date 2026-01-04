"""Neural operator builders and blocks for Spinlock."""

from .blocks import (
    BaseBlock,
    ConvBlock,
    ResidualBlock,
    StochasticBlock,
    DownsampleBlock,
    UpsampleBlock,
    OutputLayer
)
from .afno import SpectralMixingBlock, AFNOBlock
from .u_afno import UAFNOOperator, UNetEncoder, UNetDecoder
from .simple_cnn import SimpleCNNOperator
from .builder import OperatorBuilder, NeuralOperator
from .parameters import (
    OperatorParameters,
    SamplingMetrics,
    DatasetMetadata,
    BatchMetadata
)

__all__ = [
    # Base blocks
    "BaseBlock",
    "ConvBlock",
    "ResidualBlock",
    "StochasticBlock",
    "DownsampleBlock",
    "UpsampleBlock",
    "OutputLayer",
    # AFNO blocks
    "SpectralMixingBlock",
    "AFNOBlock",
    # U-AFNO operator
    "UAFNOOperator",
    "UNetEncoder",
    "UNetDecoder",
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
]
