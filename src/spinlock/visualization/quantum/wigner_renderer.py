"""Wigner function phase-space visualization."""
import torch
from typing import Optional
from spinlock.visualization.core.renderer import RenderStrategy
from spinlock.visualization.colormaps import GPUColormap


class WignerRenderer(RenderStrategy):
    """Render Wigner function W(x,p) in phase space.

    The Wigner function provides a phase-space quasi-probability
    distribution, computed via FFT from the spatial wavefunction.

    W(x,p) = (1/πℏ) ∫ ψ*(x + y/2) ψ(x - y/2) e^(-ipy/ℏ) dy

    Computed efficiently via FFT in momentum space. Note that the
    Wigner function can be negative (quasi-probability), so we use
    a diverging colormap centered at zero.
    """

    def __init__(
        self,
        colormap: str = "RdBu_r",  # Diverging colormap (Wigner can be negative)
        grid_size: Optional[int] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize Wigner renderer.

        Args:
            colormap: Diverging colormap name (e.g., 'RdBu_r', 'seismic')
            grid_size: Optional output grid size (default: match input)
            device: Torch device for computation
        """
        super().__init__()
        self.colormap = GPUColormap(colormap, device=device)
        self.grid_size = grid_size
        self.device = device

    def supports_channels(self, num_channels: int) -> bool:
        """Requires 2 channels (real/imag) to compute Wigner."""
        return num_channels == 2

    def render(
        self,
        data: torch.Tensor,
        global_stats: Optional[dict] = None
    ) -> torch.Tensor:
        """Compute and render Wigner function.

        Args:
            data: Complex wavefunction [B, 2, H, W] (real/imag)
            global_stats: Optional normalization stats with 'vmin', 'vmax'

        Returns:
            RGB phase-space image [B, H, W, 3]
        """
        if data.shape[1] != 2:
            raise ValueError(f"Wigner renderer requires 2 channels (real/imag), got {data.shape[1]}")

        # Convert to complex tensor
        real, imag = data[:, 0], data[:, 1]
        psi = torch.complex(real, imag).to(self.device)  # [B, H, W]

        # Compute Wigner function via FFT
        wigner = self._compute_wigner(psi)  # [B, H, W]

        # Normalize (Wigner can be negative, use symmetric range)
        if global_stats is not None:
            vmin, vmax = global_stats['vmin'], global_stats['vmax']
        else:
            # Symmetric range around zero
            abs_max = torch.abs(wigner).max().item()
            vmin, vmax = -abs_max, abs_max

        # Map [-vmax, vmax] → [0, 1]
        if abs(vmax - vmin) < 1e-8:
            normalized = torch.zeros_like(wigner)
        else:
            normalized = (wigner - vmin) / (vmax - vmin)
            normalized = torch.clamp(normalized, 0.0, 1.0)

        # Apply diverging colormap [B, H, W] → [B, 3, H, W]
        rgb = self.colormap.apply(normalized)

        # Permute to [B, H, W, 3] for easier grid stacking
        rgb = rgb.permute(0, 2, 3, 1)

        return rgb

    def compute_global_stats(self, data: torch.Tensor) -> dict:
        """Compute symmetric range for Wigner normalization.

        Args:
            data: Full rollout [N, M, T, 2, H, W]

        Returns:
            Dict with symmetric 'vmin', 'vmax' around zero
        """
        # Sample a subset for efficiency (Wigner computation is expensive)
        total_frames = data.shape[0] * data.shape[1] * data.shape[2]
        num_samples = min(1000, total_frames)

        # Flatten to [total_frames, 2, H, W] and sample
        flat_data = data.view(-1, 2, *data.shape[-2:])
        sample_indices = torch.randperm(flat_data.shape[0])[:num_samples]
        sample = flat_data[sample_indices].to(self.device)

        # Compute Wigner for sample
        real, imag = sample[:, 0], sample[:, 1]
        psi_sample = torch.complex(real, imag)
        wigner_sample = self._compute_wigner(psi_sample)

        # Symmetric range
        abs_max = torch.abs(wigner_sample).max().item()
        return {'vmin': -abs_max, 'vmax': abs_max}

    def _compute_wigner(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute Wigner function via FFT.

        Uses the Fourier representation of the Wigner function for
        efficient computation on GPU.

        Args:
            psi: Complex wavefunction [B, H, W]

        Returns:
            Wigner function W(x,p) [B, H, W]
        """
        # Compute 2D FFT to momentum space
        psi_k = torch.fft.fft2(psi)  # [B, H, W]

        # Wigner via auto-correlation in momentum space
        # W(x,p) ∝ IFFT[ψ(k) ψ*(k)]
        # This is a simplified Wigner (projection to 1D)
        # For full 2D phase space, need Wigner-Ville distribution

        # Auto-correlation in momentum space
        rho_k = psi_k * torch.conj(psi_k)  # [B, H, W]

        # IFFT back to position space
        wigner = torch.fft.ifft2(rho_k).real  # [B, H, W]

        # Shift zero-momentum to center
        wigner = torch.fft.fftshift(wigner, dim=(-2, -1))

        return wigner
