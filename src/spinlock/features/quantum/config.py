"""Configuration for quantum feature extraction."""

from dataclasses import dataclass


@dataclass
class QuantumConfig:
    """Configuration for quantum feature extraction.

    Attributes:
        enabled: Whether to extract quantum features
        include_purity: Compute purity Tr(ρ²) and linear entropy
        include_entropy: Compute von Neumann entropy (approximate)
        include_coherence: Compute coherence measure (off-diagonal sum)
        include_variance: Compute position/momentum uncertainties
        include_decoherence: Estimate decoherence rate from coherence decay
        hbar: Reduced Planck constant (default: 1.0 in natural units)
        domain_size: Spatial domain size for coordinate grids
    """

    enabled: bool = True
    include_purity: bool = True
    include_entropy: bool = True
    include_coherence: bool = True
    include_variance: bool = True
    include_decoherence: bool = False  # Requires full time series
    hbar: float = 1.0
    domain_size: float = 10.0
