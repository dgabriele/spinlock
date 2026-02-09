"""Trajectory in-painting orchestrator using diffusion models."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from experiments.diffusion.models.discrete_d3pm import DiscreteD3PM
from experiments.diffusion.models.denoising_network import DenoisingNetwork
from spinlock.tokens.tokenizer import VQTokenizer


class TrajectoryInpainter:
    """Orchestrates trajectory → tokens → diffusion → reconstruction pipeline."""

    def __init__(
        self,
        diffusion_model: DiscreteD3PM,
        denoising_network: DenoisingNetwork,
        tokenizer: VQTokenizer,
        device: str = "cuda"
    ):
        """
        Initialize inpainter.

        Args:
            diffusion_model: Trained discrete diffusion model
            denoising_network: Denoising network for sampling
            tokenizer: VQTokenizer for encoding/decoding
            device: Device for inference
        """
        self.diffusion = diffusion_model
        self.denoiser = denoising_network
        self.tokenizer = tokenizer
        self.device = device

        # Set to eval mode
        self.diffusion.eval()
        self.denoiser.eval()
        self.tokenizer.model.eval()

    @torch.no_grad()
    def reconstruct_trajectory(
        self,
        temporal_features: torch.Tensor,
        initial_manual: torch.Tensor,
        initial_raw: torch.Tensor,
        mask_pattern: Dict[str, torch.BoolTensor],
        num_steps: int = 50,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full reconstruction pipeline.

        Args:
            temporal_features: [T, D_t] temporal features
            initial_manual: [D_i] initial condition manual features
            initial_raw: [C, H, W] initial condition raw fields
            mask_pattern: Dict[category-level] → boolean mask
            num_steps: Number of diffusion timesteps
            verbose: Enable verbose logging

        Returns:
            {
                'original_tokens': Dict[str, Tensor],
                'masked_tokens': Dict[str, Tensor],
                'reconstructed_tokens': Dict[str, Tensor],
                'original_features': Dict[str, Tensor],
                'reconstructed_features': Dict[str, Tensor],
                'mask_pattern': Dict[str, BoolTensor]
            }
        """
        # 1. Tokenize original trajectory
        if verbose:
            print("  [1/4] Tokenizing trajectory...")
        original_tokens = self._tokenize_trajectory(
            temporal_features, initial_manual, initial_raw
        )

        # 2. Apply masking pattern
        if verbose:
            print("  [2/4] Applying masking pattern...")
        masked_tokens, observed_mask = self._apply_masking(
            original_tokens, mask_pattern
        )

        # 3. Run diffusion reconstruction
        if verbose:
            print("  [3/4] Running diffusion sampling...")
        reconstructed_tokens = self._reconstruct_via_diffusion(
            masked_tokens, observed_mask, num_steps
        )

        # 4. Detokenize both original and reconstructed
        if verbose:
            print("  [4/4] Detokenizing results...")
        original_decoded = self._detokenize(original_tokens)
        reconstructed_decoded = self._detokenize(reconstructed_tokens)

        return {
            'original_tokens': original_tokens,
            'masked_tokens': masked_tokens,
            'reconstructed_tokens': reconstructed_tokens,
            'original_features': original_decoded,
            'reconstructed_features': reconstructed_decoded,
            'mask_pattern': mask_pattern
        }

    def _tokenize_trajectory(
        self,
        temporal: torch.Tensor,
        initial_manual: torch.Tensor,
        initial_raw: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Trajectory features → discrete tokens.

        Args:
            temporal: [T, D_t]
            initial_manual: [D_i]
            initial_raw: [C, H, W]

        Returns:
            Dict of tokens [1, T] or [1, 1] per category-level
        """
        # Add batch dimension
        temporal_batch = temporal.unsqueeze(0).to(self.device)
        initial_manual_batch = initial_manual.unsqueeze(0).to(self.device)
        initial_raw_batch = initial_raw.unsqueeze(0).to(self.device)

        # Tokenize
        tokens = self.tokenizer.tokenize(
            temporal_features=temporal_batch,
            initial_manual=initial_manual_batch,
            initial_raw=initial_raw_batch
        )

        return tokens

    def _apply_masking(
        self,
        tokens: Dict[str, torch.Tensor],
        mask_pattern: Dict[str, torch.BoolTensor]
    ) -> tuple:
        """
        Apply masking pattern to tokens.

        Args:
            tokens: Dict of token tensors
            mask_pattern: Dict of boolean masks

        Returns:
            (masked_tokens, observed_mask)
        """
        masked_tokens = {}
        observed_mask = {}

        for key, token_values in tokens.items():
            # Determine number of tokens (handle both [B] and [B, T] shapes)
            if token_values.ndim == 1:
                num_tokens = 1
                default_mask = torch.ones(num_tokens, dtype=torch.bool)
            else:
                num_tokens = token_values.shape[1]
                default_mask = torch.ones(num_tokens, dtype=torch.bool)

            # Get mask for this category-level
            mask = mask_pattern.get(key, default_mask).to(self.device)

            # Expand mask to batch dimension if needed
            if token_values.ndim == 1:
                # [B] tensor - mask is just [1] for scalar
                mask_batch = mask  # Keep as [1]
                # For scalar tokens, masking doesn't apply (keep all)
                masked_tokens[key] = token_values.clone()
                observed_mask[key] = torch.ones(1, dtype=torch.bool, device=self.device)
            else:
                # [B, T] tensor
                mask_batch = mask.unsqueeze(0)  # [1, T]

                # Create masked version (set masked positions to 0)
                masked_tokens[key] = token_values.clone()
                masked_tokens[key][:, ~mask] = 0

                # Store observed mask
                observed_mask[key] = mask_batch

        return masked_tokens, observed_mask

    @torch.no_grad()
    def _reconstruct_via_diffusion(
        self,
        masked_tokens: Dict[str, torch.Tensor],
        observed_mask: Dict[str, torch.BoolTensor],
        num_steps: int
    ) -> Dict[str, torch.Tensor]:
        """
        Run diffusion sampling with RePaint inpainting.

        Args:
            masked_tokens: Tokens with masked regions zeroed
            observed_mask: Boolean mask of observed tokens
            num_steps: Number of diffusion timesteps

        Returns:
            Reconstructed token dictionary
        """
        # Sample using diffusion model with conditioning on observed tokens
        reconstructed = self.diffusion.sample(
            batch_size=1,
            denoising_network=self.denoiser,
            device=self.device,
            observed_dict=observed_mask,
            x_0_dict=masked_tokens  # RePaint: keep observed tokens fixed
        )

        return reconstructed

    def _detokenize(
        self,
        tokens: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokens → reconstructed features via embedding lookup.

        Args:
            tokens: Dict of discrete tokens

        Returns:
            Dict of continuous features (embeddings)
        """
        # Get embeddings from quantizers instead of full decode
        embeddings = {}
        for key, token_indices in tokens.items():
            if key in self.tokenizer.model.quantizers:
                quantizer = self.tokenizer.model.quantizers[key]
                # Get embeddings for these tokens
                if token_indices.ndim == 1:
                    # [B] - single token per sample
                    emb = quantizer.embedding(token_indices)  # [B, D]
                else:
                    # [B, T] - sequence of tokens
                    emb = quantizer.embedding(token_indices)  # [B, T, D]
                embeddings[key] = emb

        return embeddings
