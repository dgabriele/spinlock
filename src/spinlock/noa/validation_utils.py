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
from spinlock.noa.backbone import NOABackbone
from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE


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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    elif "config" in checkpoint and isinstance(checkpoint["config"], dict):
        # Full config with nested model config
        if "model" in checkpoint["config"]:
            model_config = checkpoint["config"]["model"]
        else:
            model_config = checkpoint["config"]
    else:
        # Default config for reaction-diffusion
        print("Warning: No config found in checkpoint, using defaults")
        model_config = {
            "in_channels": 2,
            "out_channels": 2,
            "base_channels": 32,
            "encoder_levels": 3,
            "modes": 16,
            "afno_blocks": 4,
            "dropout": 0.1,
        }

    # Create model
    mno = NOABackbone(**model_config)

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
    print(f"    Channels: {model_config.get('in_channels', 2)} → {model_config.get('out_channels', 2)}")
    print(f"    Base channels: {model_config.get('base_channels', 32)}")
    print(f"    Parameters: {sum(p.numel() for p in mno.parameters()):,}")

    return mno


def load_vqvae_checkpoint(
    checkpoint_path: str, device: str = "cuda"
) -> CategoricalHierarchicalVQVAE:
    """Load trained VQ-VAE from checkpoint.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint (.pt file)
        device: Device to load model on

    Returns:
        Loaded VQ-VAE model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")

    print(f"Loading VQ-VAE checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config
    if "model_config" not in checkpoint:
        raise ValueError("No model_config found in VQ-VAE checkpoint")

    model_config = checkpoint["model_config"]
    full_config = checkpoint.get("config", {})

    # Import VQ-VAE config class
    from spinlock.encoding.categorical_vqvae import CategoricalVQVAEConfig

    # Construct config from checkpoint
    vqvae_config = CategoricalVQVAEConfig(
        input_dim=model_config["input_dim"],
        group_indices=model_config["group_indices"],
        group_embedding_dim=model_config.get("group_embedding_dim", 256),
        group_hidden_dim=model_config.get("group_hidden_dim", 512),
        levels=model_config.get("levels"),
        commitment_cost=full_config.get("commitment_cost", 0.25),
        orthogonality_weight=full_config.get("orthogonality_weight", 0.1),
        informativeness_weight=full_config.get("informativeness_weight", 0.1),
        use_ema=full_config.get("use_ema", True),
        decay=full_config.get("decay", 0.99),
        dropout=full_config.get("dropout", 0.1),
    )

    # Create model
    vqvae = CategoricalHierarchicalVQVAE(vqvae_config)

    # Load state dict (strip "vqvae." prefix if present)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip "vqvae." prefix from keys if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("vqvae."):
            cleaned_key = key[len("vqvae."):]
            cleaned_state_dict[cleaned_key] = value
        elif not key.startswith("initial_encoder."):  # Skip initial encoder weights
            cleaned_state_dict[key] = value

    vqvae.load_state_dict(cleaned_state_dict)

    # Move to device and set eval mode
    vqvae = vqvae.to(device)
    vqvae.eval()

    num_categories = len(model_config["group_indices"])
    print(f"  ✓ VQ-VAE loaded successfully")
    print(f"    Input dim: {model_config['input_dim']}")
    print(f"    Categories: {num_categories}")
    print(f"    Parameters: {sum(p.numel() for p in vqvae.parameters()):,}")

    return vqvae


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


def get_vqvae_num_categories_and_levels(vqvae: CategoricalHierarchicalVQVAE) -> Tuple[int, int]:
    """Get number of categories and levels from VQ-VAE model.

    Args:
        vqvae: VQ-VAE model

    Returns:
        (num_categories, num_levels) tuple
    """
    # VQ-VAE has one quantizer per category
    num_categories = len(vqvae.vq_layers)

    # Assume all categories have same number of levels (hierarchical)
    # Check first quantizer
    if hasattr(vqvae.vq_layers[0], "num_levels"):
        num_levels = vqvae.vq_layers[0].num_levels
    else:
        # Single level per category
        num_levels = 1

    return num_categories, num_levels
