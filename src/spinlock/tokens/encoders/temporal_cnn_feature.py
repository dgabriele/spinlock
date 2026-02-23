"""Per-frame 2D CNN for learned temporal feature extraction.

Replaces the hand-crafted TemporalFeatureOrchestrator (spatial means, stds,
FFT, etc.) with a learnable ResNet-3 applied independently to each trajectory
frame. This provides dense, spatially-specific gradients instead of the
diffuse ~1/HW gradients from summary statistics.

Architecture:
    [B, T, C, H, W] → reshape [B*T, C, H, W]
                     → InitialCNNEncoder (ResNet-3)  (gradient-checkpointed)
                     → [B*T, D_per_frame]
                     → reshape [B, T, D_per_frame]

Reuses InitialCNNEncoder via composition — no code duplication.
Uses gradient checkpointing to bound backward-pass memory when B*T is large.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spinlock.features.initial.cnn_encoder import FrameCNNEncoder


class TemporalCNNFeatureEncoder(nn.Module):
    """Per-frame 2D CNN for learned temporal feature extraction.

    Processes each trajectory frame independently with shared weights.
    The same FrameCNNEncoder (ResNet-3) is applied to every timestep,
    producing per-frame feature vectors that preserve temporal ordering.

    Uses gradient checkpointing per chunk: intermediate CNN activations are
    discarded during forward and recomputed during backward, trading ~2x
    compute for O(chunk_size) instead of O(B*T) activation memory.

    Args:
        in_channels: Number of input channels per frame (auto-detected from data)
        embedding_dim: Output dimension per frame (should be num_groups * group_dim)
    """

    # Frames per CNN forward pass.  Only one chunk's intermediate activations
    # are live at a time during backward (gradient checkpointing).
    CHUNK_SIZE = 128

    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.frame_encoder = FrameCNNEncoder(
            embedding_dim=embedding_dim,
            in_channels=in_channels,
        )
        self.embedding_dim = embedding_dim

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Encode a single chunk — target for gradient checkpointing."""
        return self.frame_encoder(chunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each trajectory frame independently.

        Input may reside on CPU (typical for large trajectories).  Frames are
        streamed to the model's device in chunks of ``CHUNK_SIZE``.  Each chunk
        uses gradient checkpointing so only one chunk's CNN activations are
        live during the backward pass.

        Args:
            x: Trajectory tensor [B, T, C, H, W] (any device)

        Returns:
            Per-frame features [B, T, embedding_dim] on model device
        """
        B, T, C, H, W = x.shape
        flat = x.reshape(B * T, C, H, W)
        device = next(self.parameters()).device

        if B * T <= self.CHUNK_SIZE:
            features = self.frame_encoder(flat.to(device))
        else:
            chunks = []
            for i in range(0, B * T, self.CHUNK_SIZE):
                chunk = flat[i : i + self.CHUNK_SIZE].to(device)
                if self.training:
                    out = checkpoint(
                        self._encode_chunk, chunk, use_reentrant=False
                    )
                else:
                    out = self._encode_chunk(chunk)
                chunks.append(out)
            features = torch.cat(chunks, dim=0)

        return features.reshape(B, T, self.embedding_dim)  # [B, T, D]
