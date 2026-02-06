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
) -> CategoricalHierarchicalVQVAE:
    """Load trained VQ-VAE from checkpoint.

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint (.pt file)
        device: Device to load model on

    Returns:
        Loaded VQ-VAE model in eval mode
    """
    from spinlock.encoding import CategoricalVQVAEConfig
    from spinlock.encoding.learnable_assignment import PerFamilyAssignmentMatrix

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")

    print(f"Loading VQ-VAE checkpoint from {checkpoint_path}")

    # Load checkpoint (weights_only=False for compatibility with numpy objects)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    model_config = checkpoint.get("model_config", config)
    state_dict = checkpoint["model_state_dict"]

    # Create VQ-VAE config
    vqvae_config = CategoricalVQVAEConfig(
        input_dim=model_config["input_dim"],
        group_indices=model_config["group_indices"],
        group_embedding_dim=model_config.get("group_embedding_dim", 512),
        group_hidden_dim=model_config.get("group_hidden_dim", 2048),
        levels=model_config.get("levels"),
    )

    # Check if this checkpoint uses learnable assignments
    has_learnable_assignment = config.get("learnable_assignment") is not None
    assignment_matrix = None

    if has_learnable_assignment:
        # Extract assignment matrix structure from checkpoint
        vqvae_prefix = "vqvae." if any(k.startswith("vqvae.") for k in state_dict.keys()) else ""
        assignment_prefix = f"{vqvae_prefix}assignment_matrix.family_matrices."

        # Find all family names and their shapes
        feature_families = {}
        categories_per_family = {}

        for key in state_dict.keys():
            if key.startswith(assignment_prefix) and key.endswith(".logits"):
                # Extract family name: "vqvae.assignment_matrix.family_matrices.initial_cluster.logits" -> "initial_cluster"
                family_name = key[len(assignment_prefix):-len(".logits")]
                shape = state_dict[key].shape  # [num_features, num_categories]
                num_features, num_categories = shape

                # Create dummy feature indices (will be filled by load_state_dict)
                feature_families[family_name] = list(range(num_features))
                categories_per_family[family_name] = num_categories

        # Renumber features globally to avoid overlap
        global_offset = 0
        for family_name in sorted(feature_families.keys()):
            num_feats = len(feature_families[family_name])
            feature_families[family_name] = list(range(global_offset, global_offset + num_feats))
            global_offset += num_feats

        assignment_matrix = PerFamilyAssignmentMatrix(
            feature_families=feature_families,
            categories_per_family=categories_per_family
        )

    # Handle torch.compile prefix if present
    if any(k.startswith("vqvae.") for k in state_dict.keys()):
        vqvae_state_dict = {
            k.replace("vqvae.", ""): v
            for k, v in state_dict.items()
            if k.startswith("vqvae.")
        }
        vqvae = CategoricalHierarchicalVQVAE(vqvae_config, assignment_matrix=assignment_matrix)
        vqvae.load_state_dict(vqvae_state_dict)
    else:
        vqvae = CategoricalHierarchicalVQVAE(vqvae_config, assignment_matrix=assignment_matrix)
        vqvae.load_state_dict(state_dict)

    vqvae = vqvae.to(device)
    vqvae.eval()

    mode = "learnable" if has_learnable_assignment else "static"
    print(f"  ✓ VQ-VAE loaded (input_dim={model_config['input_dim']}, mode={mode})")
    print(f"    Categories: {len(model_config['group_indices'])}")
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
