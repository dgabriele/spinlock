"""Diffusion sampling utilities."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


class DiffusionSampler:
    """Generate token sequences using trained diffusion model.

    Args:
        diffusion: DiscreteD3PM model
        denoiser: Trained denoising network
        device: Computation device
    """

    def __init__(
        self,
        diffusion: nn.Module,
        denoiser: nn.Module,
        device: str = "cuda"
    ):
        self.diffusion = diffusion
        self.denoiser = denoiser
        self.device = device

        self.diffusion.eval()
        self.denoiser.eval()

    def sample_unconditional(
        self,
        num_samples: int,
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate unconditional samples.

        Args:
            num_samples: Total number of samples
            batch_size: Batch size for generation

        Returns:
            List of token dicts
        """
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Sampling"):
                current_batch_size = min(batch_size, num_samples - i * batch_size)

                tokens = self.diffusion.sample(
                    batch_size=current_batch_size,
                    denoising_network=self.denoiser,
                    device=self.device
                )

                # Convert to CPU and add to list
                tokens_cpu = {k: v.cpu() for k, v in tokens.items()}
                all_samples.append(tokens_cpu)

        return self._concatenate_batches(all_samples)

    def sample_theta_conditioned(
        self,
        theta_tokens: Dict[str, torch.Tensor],
        num_samples: int,
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate samples with fixed theta tokens.

        Args:
            theta_tokens: Fixed theta token dict [1]
            num_samples: Number of samples to generate
            batch_size: Batch size for generation

        Returns:
            List of token dicts with fixed theta
        """
        # Extract theta keys
        theta_keys = [k for k in theta_tokens.keys() if k.startswith('theta_')]

        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Sampling (theta-conditioned)"):
                current_batch_size = min(batch_size, num_samples - i * batch_size)

                # Create observed dict and x_0_dict for theta tokens
                observed_dict = self._create_observed_dict(
                    theta_tokens, theta_keys, current_batch_size
                )
                x_0_dict = self._repeat_tokens(
                    theta_tokens, theta_keys, current_batch_size
                )

                tokens = self.diffusion.sample(
                    batch_size=current_batch_size,
                    observed_dict=observed_dict,
                    x_0_dict=x_0_dict,
                    denoising_network=self.denoiser,
                    device=self.device
                )

                tokens_cpu = {k: v.cpu() for k, v in tokens.items()}
                all_samples.append(tokens_cpu)

        return self._concatenate_batches(all_samples)

    def sample_inpainting(
        self,
        observed_tokens: Dict[str, torch.Tensor],
        observed_mask: Dict[str, torch.Tensor],
        num_samples: int,
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """Complete partial token sequences using inpainting.

        Args:
            observed_tokens: Known token values [1]
            observed_mask: Boolean mask indicating observed positions [1]
            num_samples: Number of completions to generate
            batch_size: Batch size for generation

        Returns:
            List of completed token dicts
        """
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Sampling (inpainting)"):
                current_batch_size = min(batch_size, num_samples - i * batch_size)

                # Repeat masks and tokens for batch
                observed_dict_batch = {
                    k: v.repeat(current_batch_size).to(self.device)
                    for k, v in observed_mask.items()
                }
                x_0_dict_batch = {
                    k: v.repeat(current_batch_size).to(self.device)
                    for k, v in observed_tokens.items()
                }

                tokens = self.diffusion.sample(
                    batch_size=current_batch_size,
                    observed_dict=observed_dict_batch,
                    x_0_dict=x_0_dict_batch,
                    denoising_network=self.denoiser,
                    device=self.device
                )

                tokens_cpu = {k: v.cpu() for k, v in tokens.items()}
                all_samples.append(tokens_cpu)

        return self._concatenate_batches(all_samples)

    def _create_observed_dict(
        self,
        tokens: Dict[str, torch.Tensor],
        keys: List[str],
        batch_size: int
    ) -> Dict[str, torch.BoolTensor]:
        """Create observed mask dict."""
        return {
            k: torch.ones(batch_size, dtype=torch.bool, device=self.device)
            for k in keys
        }

    def _repeat_tokens(
        self,
        tokens: Dict[str, torch.Tensor],
        keys: List[str],
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Repeat tokens for batch."""
        return {
            k: tokens[k].repeat(batch_size).to(self.device)
            for k in keys
        }

    def _concatenate_batches(
        self,
        batches: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert list of batch dicts to list of sample dicts."""
        if not batches:
            return []

        # Get all keys
        keys = batches[0].keys()

        # Concatenate along batch dimension
        concatenated = {
            k: torch.cat([b[k] for b in batches], dim=0)
            for k in keys
        }

        # Convert to list of dicts (one per sample)
        num_samples = concatenated[next(iter(keys))].shape[0]
        samples = [
            {k: concatenated[k][i] for k in keys}
            for i in range(num_samples)
        ]

        return samples
