"""Shared spectral utilities for Fourier-based encoding/decoding.

Provides the Hermitian iFFT2 reconstruction used by both:
- SpectralGridDecoder (CVAE/rollout_vae): MLP-predicted coeffs → grids
- InitialSpectralInverse (VQ inverse head): quantized features → grids
- SpectralICEncoder (VQ encoder): grids → Fourier features (forward FFT)
"""

import torch


def hermitian_ifft2(
    coeffs: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Reconstruct real-valued 2D grids from low-frequency Fourier coefficients.

    Places K×K complex coefficients into a full H×W spectrum with Hermitian
    symmetry (F[k1,k2] = conj(F[-k1,-k2])), ensuring real-valued iFFT output.

    Args:
        coeffs: Complex Fourier coefficients [B, C, K, K].
        H: Target grid height.
        W: Target grid width.

    Returns:
        Real-valued grids [B, C, H, W].
    """
    B, C, K, _ = coeffs.shape

    # Derive complex dtype from input to support mixed precision
    complex_dtype = (
        torch.complex128 if coeffs.dtype == torch.complex128 else torch.complex64
    )
    full = torch.zeros(B, C, H, W, dtype=complex_dtype, device=coeffs.device)

    # Positive-frequency quadrant: rows [0, K), cols [0, K)
    full[:, :, :K, :K] = coeffs

    # Force DC component to be real (imaginary = 0)
    full[:, :, 0, 0] = full[:, :, 0, 0].real.to(full.dtype)

    # Row-0 negative columns: F[0, W-j] = conj(F[0, j]) for j=1..K-1
    full[:, :, 0, W - K + 1 :] = torch.conj(coeffs[:, :, 0, 1:].flip(-1))

    # Negative rows, col-0: F[H-i, 0] = conj(F[i, 0]) for i=1..K-1
    full[:, :, H - K + 1 :, 0] = torch.conj(coeffs[:, :, 1:, 0].flip(-1))

    # Negative rows, negative cols: F[H-i, W-j] = conj(F[i, j]) for i,j=1..K-1
    full[:, :, H - K + 1 :, W - K + 1 :] = torch.conj(
        coeffs[:, :, 1:, 1:].flip(-2, -1)
    )

    # Inverse FFT → real grid
    grids = torch.fft.ifft2(full).real  # [B, C, H, W]

    return grids
