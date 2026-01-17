"""Impulse perturbations for probing MNO dynamics.

Impulse perturbations are single-timestep delta-function perturbations that:
- Excite eigenmodes of the learned dynamical system
- Probe transient response characteristics
- Test sensitivity to localized/distributed forcing
"""

from typing import Dict, Any, Literal, Optional, Tuple
import torch
from torch import Tensor
import numpy as np

from .base import BasePerturbation


class ImpulsePerturbation(BasePerturbation):
    """Single-timestep impulse perturbation.

    Applies a spatially-structured perturbation at a single timestep.
    The perturbation is added to the state: u_perturbed = u + perturbation_field

    Spatial patterns:
    - gaussian_blob: Localized Gaussian perturbation
    - uniform: Spatially uniform forcing
    - random: Random spatial pattern
    - fourier_mode: Single Fourier mode (specific wavelength)
    - checkerboard: Alternating +/- pattern (high frequency)
    """

    def __init__(
        self,
        t_impulse: int,
        amplitude: float,
        pattern: Literal["gaussian_blob", "uniform", "random", "fourier_mode", "checkerboard"],
        spatial_size: Tuple[int, int],
        num_channels: int = 2,
        channel_idx: Optional[int] = None,
        device: str = "cpu",
        # Gaussian blob parameters
        center: Optional[Tuple[float, float]] = None,
        sigma: float = 0.1,
        # Fourier mode parameters
        wavenumber: Optional[Tuple[int, int]] = None,
        # Random seed
        seed: Optional[int] = None,
    ):
        """Initialize impulse perturbation.

        Args:
            t_impulse: Timestep to apply impulse (0-indexed)
            amplitude: Amplitude of perturbation
            pattern: Spatial pattern type
            spatial_size: (H, W) spatial dimensions
            num_channels: Number of channels in state
            channel_idx: If specified, only perturb this channel (0-indexed).
                        If None, perturb all channels.
            device: Device placement
            center: (y, x) center for gaussian_blob, normalized to [0, 1]
                   If None, defaults to (0.5, 0.5) - center of domain
            sigma: Standard deviation for gaussian_blob (fraction of domain size)
            wavenumber: (ky, kx) for fourier_mode pattern
                       If None, defaults to (1, 1) - fundamental mode
            seed: Random seed for reproducible random patterns
        """
        super().__init__(device)
        self.t_impulse = t_impulse
        self.amplitude = amplitude
        self.pattern = pattern
        self.spatial_size = spatial_size
        self.num_channels = num_channels
        self.channel_idx = channel_idx
        self.center = center if center is not None else (0.5, 0.5)
        self.sigma = sigma
        self.wavenumber = wavenumber if wavenumber is not None else (1, 1)
        self.seed = seed

        # Pre-generate perturbation field for efficiency
        self._perturbation_field = self._generate_perturbation_field()

    def _generate_perturbation_field(self) -> Tensor:
        """Generate spatial perturbation pattern.

        Returns:
            Perturbation field [C, H, W]
        """
        H, W = self.spatial_size

        if self.pattern == "gaussian_blob":
            field = self._gaussian_blob(H, W)
        elif self.pattern == "uniform":
            field = torch.ones((H, W), device=self.device)
        elif self.pattern == "random":
            if self.seed is not None:
                rng = torch.Generator(device=self.device).manual_seed(self.seed)
                field = torch.randn((H, W), generator=rng, device=self.device)
            else:
                field = torch.randn((H, W), device=self.device)
        elif self.pattern == "fourier_mode":
            field = self._fourier_mode(H, W)
        elif self.pattern == "checkerboard":
            field = self._checkerboard(H, W)
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

        # Normalize to unit amplitude, then scale
        field = field / (field.abs().max() + 1e-8)
        field = field * self.amplitude

        # Expand to channels [C, H, W]
        if self.channel_idx is not None:
            # Only perturb specified channel
            full_field = torch.zeros((self.num_channels, H, W), device=self.device)
            full_field[self.channel_idx] = field
        else:
            # Perturb all channels with same pattern
            full_field = field.unsqueeze(0).repeat(self.num_channels, 1, 1)

        return full_field

    def _gaussian_blob(self, H: int, W: int) -> Tensor:
        """Generate Gaussian blob perturbation.

        Args:
            H, W: Spatial dimensions

        Returns:
            Gaussian field [H, W]
        """
        y = torch.linspace(0, 1, H, device=self.device)
        x = torch.linspace(0, 1, W, device=self.device)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        cy, cx = self.center
        dist_sq = (Y - cy) ** 2 + (X - cx) ** 2
        gaussian = torch.exp(-dist_sq / (2 * self.sigma**2))

        return gaussian

    def _fourier_mode(self, H: int, W: int) -> Tensor:
        """Generate single Fourier mode perturbation.

        Args:
            H, W: Spatial dimensions

        Returns:
            Sinusoidal field [H, W]
        """
        y = torch.linspace(0, 2 * np.pi, H, device=self.device)
        x = torch.linspace(0, 2 * np.pi, W, device=self.device)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        ky, kx = self.wavenumber
        fourier = torch.sin(ky * Y) * torch.sin(kx * X)

        return fourier

    def _checkerboard(self, H: int, W: int) -> Tensor:
        """Generate checkerboard pattern (high-frequency).

        Args:
            H, W: Spatial dimensions

        Returns:
            Checkerboard field [H, W] with values in {-1, +1}
        """
        y_idx = torch.arange(H, device=self.device).unsqueeze(1)
        x_idx = torch.arange(W, device=self.device).unsqueeze(0)

        checkerboard = ((y_idx + x_idx) % 2) * 2.0 - 1.0  # Values in {-1, +1}

        return checkerboard.float()

    def apply(self, state: Tensor, t: int) -> Tensor:
        """Apply impulse perturbation at specified timestep.

        Args:
            state: Current state [B, C, H, W] or [C, H, W]
            t: Current timestep

        Returns:
            Perturbed state if t == t_impulse, else unchanged
        """
        if not self.is_active(t):
            return state

        # Handle both batched and unbatched inputs
        is_batched = state.ndim == 4

        if is_batched:
            # Broadcast perturbation to batch: [C, H, W] -> [1, C, H, W] -> [B, C, H, W]
            perturbation = self._perturbation_field.unsqueeze(0)
        else:
            perturbation = self._perturbation_field

        return state + perturbation

    def is_active(self, t: int) -> bool:
        """Active only at t_impulse.

        Args:
            t: Current timestep

        Returns:
            True if t == t_impulse
        """
        return t == self.t_impulse

    def get_metadata(self) -> Dict[str, Any]:
        """Return impulse perturbation metadata.

        Returns:
            Dictionary with all perturbation parameters
        """
        metadata = super().get_metadata()
        metadata.update(
            {
                "t_impulse": self.t_impulse,
                "amplitude": self.amplitude,
                "pattern": self.pattern,
                "spatial_size": self.spatial_size,
                "num_channels": self.num_channels,
                "channel_idx": self.channel_idx,
            }
        )

        # Add pattern-specific parameters
        if self.pattern == "gaussian_blob":
            metadata["center"] = self.center
            metadata["sigma"] = self.sigma
        elif self.pattern == "fourier_mode":
            metadata["wavenumber"] = self.wavenumber
        elif self.pattern == "random":
            metadata["seed"] = self.seed

        return metadata


class ImpulsePerturbationFactory:
    """Factory for creating common impulse perturbation patterns.

    Provides convenient constructors for standard experimental designs.
    """

    @staticmethod
    def center_blob(
        t_impulse: int,
        amplitude: float,
        spatial_size: Tuple[int, int],
        sigma: float = 0.1,
        num_channels: int = 2,
        device: str = "cpu",
    ) -> ImpulsePerturbation:
        """Create Gaussian blob at center of domain.

        Args:
            t_impulse: Timestep to apply impulse
            amplitude: Perturbation amplitude
            spatial_size: (H, W) spatial dimensions
            sigma: Gaussian width (fraction of domain)
            num_channels: Number of state channels
            device: Device placement

        Returns:
            ImpulsePerturbation with Gaussian blob at center
        """
        return ImpulsePerturbation(
            t_impulse=t_impulse,
            amplitude=amplitude,
            pattern="gaussian_blob",
            spatial_size=spatial_size,
            num_channels=num_channels,
            center=(0.5, 0.5),
            sigma=sigma,
            device=device,
        )

    @staticmethod
    def corner_blob(
        t_impulse: int,
        amplitude: float,
        spatial_size: Tuple[int, int],
        corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"],
        sigma: float = 0.1,
        num_channels: int = 2,
        device: str = "cpu",
    ) -> ImpulsePerturbation:
        """Create Gaussian blob at corner of domain.

        Args:
            t_impulse: Timestep to apply impulse
            amplitude: Perturbation amplitude
            spatial_size: (H, W) spatial dimensions
            corner: Which corner to place blob
            sigma: Gaussian width (fraction of domain)
            num_channels: Number of state channels
            device: Device placement

        Returns:
            ImpulsePerturbation with Gaussian blob at specified corner
        """
        corner_positions = {
            "top_left": (0.25, 0.25),
            "top_right": (0.25, 0.75),
            "bottom_left": (0.75, 0.25),
            "bottom_right": (0.75, 0.75),
        }
        center = corner_positions[corner]

        return ImpulsePerturbation(
            t_impulse=t_impulse,
            amplitude=amplitude,
            pattern="gaussian_blob",
            spatial_size=spatial_size,
            num_channels=num_channels,
            center=center,
            sigma=sigma,
            device=device,
        )

    @staticmethod
    def random_blob(
        t_impulse: int,
        amplitude: float,
        spatial_size: Tuple[int, int],
        sigma: float = 0.1,
        num_channels: int = 2,
        seed: Optional[int] = None,
        device: str = "cpu",
    ) -> ImpulsePerturbation:
        """Create Gaussian blob at random location.

        Args:
            t_impulse: Timestep to apply impulse
            amplitude: Perturbation amplitude
            spatial_size: (H, W) spatial dimensions
            sigma: Gaussian width (fraction of domain)
            num_channels: Number of state channels
            seed: Random seed for reproducibility
            device: Device placement

        Returns:
            ImpulsePerturbation with Gaussian blob at random location
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        # Random center in [0.2, 0.8] to avoid edges
        center = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))

        return ImpulsePerturbation(
            t_impulse=t_impulse,
            amplitude=amplitude,
            pattern="gaussian_blob",
            spatial_size=spatial_size,
            num_channels=num_channels,
            center=center,
            sigma=sigma,
            device=device,
        )

    @staticmethod
    def uniform_forcing(
        t_impulse: int,
        amplitude: float,
        spatial_size: Tuple[int, int],
        num_channels: int = 2,
        channel_idx: Optional[int] = None,
        device: str = "cpu",
    ) -> ImpulsePerturbation:
        """Create spatially uniform impulse.

        Args:
            t_impulse: Timestep to apply impulse
            amplitude: Perturbation amplitude
            spatial_size: (H, W) spatial dimensions
            num_channels: Number of state channels
            channel_idx: If specified, only perturb this channel
            device: Device placement

        Returns:
            ImpulsePerturbation with uniform spatial pattern
        """
        return ImpulsePerturbation(
            t_impulse=t_impulse,
            amplitude=amplitude,
            pattern="uniform",
            spatial_size=spatial_size,
            num_channels=num_channels,
            channel_idx=channel_idx,
            device=device,
        )
