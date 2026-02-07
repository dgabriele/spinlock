"""Utilities for Phase 2 validation experiments.

Provides model loading, IC sampling, and other utilities for running
validation experiments on trained MNO and VQ-VAE models.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
from pathlib import Path
import sys

# Import models
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spinlock.mno.backbone import NOABackbone
from spinlock.tokens import VQTokenizer


def load_mno_checkpoint(checkpoint_path: str, device: str = "cuda") -> NOABackbone:
    """Load trained MNO from checkpoint.

    Args:
        checkpoint_path: Path to MNO checkpoint (.pt file)
        device: Device to load model on

    Returns:
        Loaded MNO model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MNO checkpoint not found: {checkpoint_path}")

    print(f"Loading MNO checkpoint from {checkpoint_path}")

    # Load checkpoint (weights_only=False for compatibility with numpy objects)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
    elif "config" in checkpoint:
        # Handle nested config structure (config.model contains model params)
        if isinstance(checkpoint["config"], dict) and "model" in checkpoint["config"]:
            config = checkpoint["config"]["model"]
        else:
            config = checkpoint["config"]
    else:
        # Default config for reaction-diffusion
        print("Warning: No config found in checkpoint, using defaults")
        config = {
            "in_channels": 2,
            "out_channels": 2,
            "base_channels": 32,
            "encoder_levels": 3,
            "modes": 16,
            "afno_blocks": 4,
            "dropout": 0.1,
        }

    # Rename 'film' to 'film_config' if present (NOABackbone expects 'film_config')
    if "film" in config:
        config["film_config"] = config.pop("film")

    # Create model
    mno = NOABackbone(**config)

    # Load state dict
    if "model_state_dict" in checkpoint:
        mno.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        mno.load_state_dict(checkpoint["state_dict"])
    else:
        mno.load_state_dict(checkpoint)

    # Move to device and set eval mode
    mno = mno.to(device)
    mno.eval()

    print(f"  ✓ MNO loaded successfully")
    print(f"    Channels: {config.get('in_channels', 2)} → {config.get('out_channels', 2)}")
    print(f"    Parameters: {sum(p.numel() for p in mno.parameters()):,}")

    return mno


def load_vqvae_checkpoint(
    checkpoint_path: str, device: str = "cuda"
) -> VQTokenizer:
    """Load trained VQ-VAE from checkpoint.

    MIGRATED: Now uses VQTokenizer instead of V1 CategoricalHierarchicalVQVAE.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint (.pt file)
        device: Device to load model on

    Returns:
        Loaded VQTokenizer in eval mode

    Raises:
        ValueError: If checkpoint is V1 format (no longer supported)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")

    print(f"Loading VQ-VAE checkpoint from {checkpoint_path}")

    # Quick check for V1 vs V2 checkpoint format
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # VQTokenizer checkpoints have 'tokenizer_version' or use TokenizerConfig
    is_v2 = (
        "tokenizer_version" in checkpoint or
        (isinstance(checkpoint.get("config"), dict) and
         "encoder" in checkpoint.get("config", {}))
    )

    if not is_v2:
        raise ValueError(
            f"V1 VQ-VAE checkpoint format is no longer supported.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"\n"
            f"Please retrain using the VQTokenizer system:\n"
            f"  poetry run spinlock train-vq-tokenizer --config configs/vqvae_50k.yaml\n"
            f"\n"
            f"V1 checkpoints (CategoricalHierarchicalVQVAE) are no longer supported."
        )

    # Load using VQTokenizer
    tokenizer = VQTokenizer.from_checkpoint(checkpoint_path)

    # Move to device
    tokenizer.model = tokenizer.model.to(device)
    tokenizer.model.eval()

    # Print info
    config = checkpoint["config"]
    num_groups = len(checkpoint.get("group_indices", {}))
    num_params = sum(p.numel() for p in tokenizer.model.parameters())

    print(f"  ✓ VQTokenizer loaded (V2 format)")
    print(f"    Groups: {num_groups}")
    print(f"    Parameters: {num_params:,}")

    return tokenizer


def sample_initial_condition(
    num_channels: int = 2,
    spatial_size: Tuple[int, int] = (64, 64),
    ic_type: str = "smooth_random",
    device: str = "cuda",
    seed: Optional[int] = None,
) -> Tensor:
    """Sample initial condition for episode.

    Args:
        num_channels: Number of state channels
        spatial_size: (H, W) spatial dimensions
        ic_type: Type of initial condition:
            - "smooth_random": Smooth random fields (low-pass filtered)
            - "random": Pure random (Gaussian noise)
            - "blob": Random Gaussian blobs
            - "zero": Zero initial condition
        device: Device to create tensor on
        seed: Optional random seed for reproducibility

    Returns:
        Initial condition [C, H, W]
    """
    if seed is not None:
        torch.manual_seed(seed)

    H, W = spatial_size

    if ic_type == "smooth_random":
        # Generate smooth random field via low-pass filtering
        u0 = torch.randn(num_channels, H, W, device=device)

        # FFT-based low-pass filter
        for c in range(num_channels):
            fft = torch.fft.rfft2(u0[c])

            # Create low-pass mask (keep low 25% of frequencies)
            H_freq, W_freq = fft.shape
            mask = torch.zeros_like(fft)
            mask[:H_freq//4, :W_freq//4] = 1.0

            # Apply filter
            fft_filtered = fft * mask
            u0[c] = torch.fft.irfft2(fft_filtered, s=(H, W))

        # Normalize to reasonable range
        u0 = u0 * 0.5

    elif ic_type == "random":
        # Pure Gaussian noise
        u0 = torch.randn(num_channels, H, W, device=device) * 0.3

    elif ic_type == "blob":
        # Random Gaussian blobs
        u0 = torch.zeros(num_channels, H, W, device=device)

        num_blobs = torch.randint(3, 8, (1,)).item()
        for _ in range(num_blobs):
            # Random center
            cy = torch.rand(1).item()
            cx = torch.rand(1).item()

            # Random amplitude and width
            amplitude = torch.randn(1).item()
            sigma = torch.rand(1).item() * 0.1 + 0.05

            # Create Gaussian blob
            y = torch.linspace(0, 1, H, device=device)
            x = torch.linspace(0, 1, W, device=device)
            Y, X = torch.meshgrid(y, x, indexing="ij")

            dist_sq = (Y - cy) ** 2 + (X - cx) ** 2
            blob = torch.exp(-dist_sq / (2 * sigma**2)) * amplitude

            # Add to random channel
            c = torch.randint(0, num_channels, (1,)).item()
            u0[c] += blob

    elif ic_type == "zero":
        # Zero initial condition (will rely entirely on perturbations)
        u0 = torch.zeros(num_channels, H, W, device=device)

    else:
        raise ValueError(f"Unknown ic_type: {ic_type}")

    return u0


def get_vqvae_num_categories_and_levels(vqvae: VQTokenizer) -> Tuple[int, int]:
    """Get number of categories and levels from VQTokenizer model.

    MIGRATED: Now works with VQTokenizer instead of V1 CategoricalHierarchicalVQVAE.

    Args:
        vqvae: VQTokenizer instance

    Returns:
        (num_categories, num_levels) tuple
    """
    # VQTokenizer has a hierarchical model with quantizers per category
    model = vqvae.model

    # Count unique category names (temporal_group_X, initial_group_Y, etc.)
    categories = set()
    num_levels = None

    # group_indices maps "family_category_Ll" to feature indices
    # e.g., {"temporal_group_1_L0": [...], "temporal_group_1_L1": [...], ...}
    for key in vqvae.group_indices.keys():
        # Parse key like "temporal_group_1_L0" -> category="temporal_group_1", level=0
        parts = key.rsplit("_L", 1)
        if len(parts) == 2:
            category = parts[0]
            level = int(parts[1])
            categories.add(category)

            # Track max level + 1 to get num_levels
            if num_levels is None:
                num_levels = level + 1
            else:
                num_levels = max(num_levels, level + 1)

    num_categories = len(categories)
    if num_levels is None:
        num_levels = 1  # Default if no hierarchy detected

    return num_categories, num_levels
