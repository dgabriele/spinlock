"""Decoder modules for VQ tokenizer inverse heads."""

from .initial_spatial import SpatialICDecoder
from .initial_spatial_latent import SpatialICLatentDecoder

__all__ = ["SpatialICDecoder", "SpatialICLatentDecoder"]
