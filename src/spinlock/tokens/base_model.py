"""Base class for tokenizer models (VQ and NL).

Provides the shared interface and utility methods that both
JointHierarchicalVQVAE (discrete VQ) and NLTokenizerModel (continuous VAE)
inherit from. The key shared concept is *family-based encoding*: different
encoder branches for temporal, initial, and theta features, with family
presence auto-detected from group_indices keys.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseTokenizerModel(nn.Module, ABC):
    """Abstract base for tokenizer neural network models.

    Subclasses must implement:
        - forward(): Full forward pass (encode + bottleneck + decode)
        - encode(): Encode inputs to latent representation

    Provides:
        - parse_families(): Discover family names from group_indices
        - Family/group bookkeeping
    """

    group_indices: Dict[str, List[int]]
    families: List[str]

    @staticmethod
    def parse_families(group_indices: Dict[str, List[int]]) -> List[str]:
        """Parse unique families from group_indices keys.

        Family presence is derived from group_indices: if there are keys
        starting with "theta_", the theta family exists. No boolean flag needed.

        Args:
            group_indices: Dict with keys like "temporal_group_1", "initial_group_2"

        Returns:
            Sorted list of unique families, e.g., ["initial", "temporal", "theta"]
        """
        families = set()
        for key in group_indices.keys():
            family = key.split("_", 1)[0]
            families.add(family)
        return sorted(families)

    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, Any]:
        """Full forward pass through the model."""
        ...

    @abstractmethod
    def encode(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode inputs to latent representation."""
        ...
