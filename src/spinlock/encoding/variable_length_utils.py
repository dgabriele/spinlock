"""
Utilities for variable-length trajectory support in VQ-VAE training.

Provides helper functions for:
- Config parsing and validation
- Length sampler creation
- Dataset augmentation with trajectory length sampling
- Mask propagation through training pipeline
"""

from typing import Dict, Any, Optional
import torch
import numpy as np


def parse_variable_length_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse and validate variable-length configuration from training config.

    Args:
        config: Full training configuration dict

    Returns:
        Parsed variable-length config dict, or None if disabled

    Raises:
        ValueError: If configuration is invalid

    Examples:
        >>> config = {"families": {"temporal": {"encoder_params": {
        ...     "variable_length": {"enabled": True, "min_timesteps": 8, "max_timesteps": 500}
        ... }}}}
        >>> vl_config = parse_variable_length_config(config)
        >>> vl_config["min_timesteps"]  # 8
    """
    # Check if families config exists
    if "families" not in config:
        return None

    families_config = config["families"]

    # Look for variable_length config in any family
    # (Currently only temporal families support it)
    for family_name, family_config in families_config.items():
        encoder_params = family_config.get("encoder_params", {})
        vl_config = encoder_params.get("variable_length", {})

        if not vl_config.get("enabled", False):
            continue

        # Found enabled variable_length config - validate it
        min_T = vl_config.get("min_timesteps")
        max_T = vl_config.get("max_timesteps")

        if min_T is None or max_T is None:
            raise ValueError(
                f"variable_length enabled for {family_name} but "
                f"min_timesteps or max_timesteps not specified"
            )

        if min_T < 1:
            raise ValueError(f"min_timesteps must be >= 1, got {min_T}")

        if min_T >= max_T:
            raise ValueError(
                f"min_timesteps ({min_T}) must be < max_timesteps ({max_T})"
            )

        # Check against downsample_factors if specified
        downsample_factors = encoder_params.get("downsample_factors", [1, 2, 4, 8])
        max_factor = max(downsample_factors)

        if min_T < max_factor:
            # This is allowed with adaptive pyramid, but warn user
            print(
                f"Warning: min_timesteps ({min_T}) < max downsample_factor ({max_factor}). "
                f"Some pyramid levels will be skipped for short trajectories."
            )

        # Return the parsed config
        return vl_config

    # No variable_length config found or all disabled
    return None


def create_length_sampler(
    vl_config: Dict[str, Any]
) -> Optional["TrajectoryLengthSampler"]:
    """Create TrajectoryLengthSampler from config.

    Args:
        vl_config: Variable-length configuration dict

    Returns:
        TrajectoryLengthSampler instance, or None if config is None

    Examples:
        >>> vl_config = {"min_timesteps": 8, "max_timesteps": 500, "sampling_strategy": "uniform"}
        >>> sampler = create_length_sampler(vl_config)
        >>> lengths = sampler.sample_lengths(batch_size=32)
    """
    if vl_config is None:
        return None

    from .trajectory_length_sampler import TrajectoryLengthSampler

    sampler = TrajectoryLengthSampler(
        min_timesteps=vl_config["min_timesteps"],
        max_timesteps=vl_config["max_timesteps"],
        strategy=vl_config.get("sampling_strategy", "uniform"),
        geometric_decay=vl_config.get("geometric_decay", 0.01),
        length_bins=vl_config.get("length_bins", []),
    )

    return sampler


def create_length_curriculum_scheduler(
    vl_config: Dict[str, Any]
) -> Optional["LengthCurriculumScheduler"]:
    """Create LengthCurriculumScheduler from config if curriculum learning enabled.

    Args:
        vl_config: Variable-length configuration dict

    Returns:
        LengthCurriculumScheduler instance, or None if not enabled

    Examples:
        >>> vl_config = {
        ...     "use_length_curriculum": True,
        ...     "curriculum_epochs": 200,
        ...     "curriculum_start_min": 400,
        ...     "min_timesteps": 8,
        ... }
        >>> scheduler = create_length_curriculum_scheduler(vl_config)
        >>> scheduler.get_min_timesteps(epoch=100)  # ~200
    """
    if vl_config is None:
        return None

    if not vl_config.get("use_length_curriculum", False):
        return None

    from .trajectory_length_sampler import LengthCurriculumScheduler

    scheduler = LengthCurriculumScheduler(
        start_min=vl_config.get("curriculum_start_min", vl_config["max_timesteps"]),
        end_min=vl_config["min_timesteps"],
        total_epochs=vl_config.get("curriculum_epochs", 100),
        schedule=vl_config.get("curriculum_schedule", "linear"),
    )

    return scheduler


def augment_dataset_with_lengths(
    dataset_class,
    features: np.ndarray,
    length_sampler: Optional["TrajectoryLengthSampler"],
    **dataset_kwargs
):
    """Create dataset with variable-length support.

    Augments the standard FeatureDataset to include trajectory length sampling
    and mask creation on __getitem__.

    Args:
        dataset_class: Base dataset class (FeatureDataset)
        features: Feature array [N, T, D] or [N, D]
        length_sampler: TrajectoryLengthSampler instance, or None for standard mode
        **dataset_kwargs: Additional kwargs for dataset_class

    Returns:
        Dataset instance with variable-length support

    Notes:
        If length_sampler is None, returns standard dataset (backward compatible).
        If provided, each __getitem__ will include:
            - mask: Boolean mask [T] indicating valid timesteps
            - length: Scalar tensor with actual length
    """
    # Standard mode: no length sampling
    if length_sampler is None:
        return dataset_class(features, **dataset_kwargs)

    # Variable-length mode: wrap dataset to add length sampling
    class VariableLengthDataset(dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.length_sampler = length_sampler

            # Infer max_T from features shape
            if len(self.features.shape) == 3:
                # [N, T, D] format
                self.max_T = self.features.shape[1]
            else:
                # [N, D] format - no temporal dimension
                raise ValueError(
                    "Variable-length mode requires temporal features [N, T, D], "
                    f"got shape {self.features.shape}"
                )

        def __getitem__(self, idx):
            item = super().__getitem__(idx)

            # Sample trajectory length
            length = self.length_sampler.sample_lengths(1)[0]

            # Create mask
            mask = self.length_sampler.create_mask(
                length.unsqueeze(0),
                self.max_T
            )[0]  # [T]

            # Add to item dict
            item["mask"] = mask
            item["length"] = length

            return item

    return VariableLengthDataset(features, **dataset_kwargs)


def extract_mask_info_from_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """Extract mask information from batch for loss computation.

    Args:
        batch: Batch dict from DataLoader
        device: Target device

    Returns:
        mask_info dict with num_valid_timesteps, or None if no mask

    Examples:
        >>> batch = {"mask": torch.ones(32, 256, dtype=torch.bool), "length": torch.full((32,), 256)}
        >>> mask_info = extract_mask_info_from_batch(batch, torch.device("cuda"))
        >>> mask_info["num_valid_timesteps"]  # [32] with values = 256
    """
    if "mask" not in batch or "length" not in batch:
        return None

    mask = batch["mask"].to(device)  # [B, T]
    lengths = batch["length"].to(device)  # [B]

    # Compute number of valid timesteps (should match lengths)
    num_valid = mask.sum(dim=1)  # [B]

    return {
        "num_valid_timesteps": num_valid,
        "lengths": lengths,
    }


def update_sampler_for_curriculum(
    length_sampler: Optional["TrajectoryLengthSampler"],
    curriculum_scheduler: Optional["LengthCurriculumScheduler"],
    epoch: int
) -> None:
    """Update length sampler's min_timesteps based on curriculum schedule.

    Args:
        length_sampler: TrajectoryLengthSampler to update
        curriculum_scheduler: LengthCurriculumScheduler for curriculum learning
        epoch: Current training epoch

    Examples:
        >>> sampler = TrajectoryLengthSampler(min_timesteps=400, max_timesteps=500)
        >>> scheduler = LengthCurriculumScheduler(start_min=400, end_min=8, total_epochs=200)
        >>> update_sampler_for_curriculum(sampler, scheduler, epoch=100)
        >>> sampler.min_T  # ~200 (halfway through curriculum)
    """
    if length_sampler is None or curriculum_scheduler is None:
        return

    # Get min_timesteps for current epoch
    new_min_T = curriculum_scheduler.get_min_timesteps(epoch)

    # Update sampler
    length_sampler.min_T = new_min_T
