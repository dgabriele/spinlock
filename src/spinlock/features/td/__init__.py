"""TD (Temporal Dynamics) feature family.

Temporal encoding of operator behavior during rollout. Unlike SDF which aggregates
trajectories to scalar summaries, TD preserves full time series information for
VQ-VAE tokenization via 1D CNN encoder.

Features:
- Per-timestep features: Loaded from SDF's existing per_timestep extraction
- Derived temporal curves: Energy, variance, smoothness trajectories
- Output shape: [N, M, T, D_td] full time series

Example:
    >>> from spinlock.features.td import TDExtractor, TDConfig
    >>> config = TDConfig(include_per_timestep=True, include_derived_curves=True)
    >>> extractor = TDExtractor(config=config, device=torch.device('cuda'))
    >>> result = extractor.extract_all(trajectories=traj, per_timestep_features=sdf_per_t)
    >>> sequences = result['sequences']  # [N, M, T, D_td]
"""

from .config import TDConfig
from .extractor import TDExtractor

__all__ = ["TDConfig", "TDExtractor"]
