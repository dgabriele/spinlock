"""LFM adapter, NL listener, and token bridge for NLTokenizer.

LFMAdapter wraps LFM's MultilingualVAEGenerator — no architecture replication.
NLListener decodes NL token distributions back to latent z (roundtrip signal).
NLTokenBridge implements LFM's TokenBridge protocol for interoperability.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .nl_config import LFMAdapterConfig, NLListenerConfig

logger = logging.getLogger(__name__)


class LFMAdapter(nn.Module):
    """Thin wrapper around LFM's MultilingualVAEGenerator for NLTokenizer.

    Takes latent z vectors and generates NL token distributions via the
    frozen autoregressive decoder. The decoder is pretrained on multilingual
    text and provides the linguistic prior.

    Args:
        config: LFMAdapterConfig with decoder architecture and checkpoint paths
    """

    def __init__(self, config: LFMAdapterConfig):
        super().__init__()
        self._config = config

        from lfm.generator.multilingual_vae import MultilingualVAEGenerator
        from lfm.generator.config import GeneratorConfig

        gen_config = GeneratorConfig(
            latent_dim=config.latent_dim,
            vocab_size=config.vocab_size,
            max_output_len=config.max_output_len,
            decoder_hidden_dim=config.decoder_hidden_dim,
            decoder_num_layers=config.decoder_num_layers,
            decoder_num_heads=config.decoder_num_heads,
            pretrained_decoder_path=(
                str(config.pretrained_decoder_path)
                if config.pretrained_decoder_path
                else None
            ),
            spm_model_path=(
                str(config.spm_model_path)
                if config.spm_model_path
                else None
            ),
            freeze_decoder=config.freeze_decoder,
            temperature=config.temperature,
            hard_sample=config.hard_sample,
        )
        self.generator = MultilingualVAEGenerator(gen_config)
        logger.info(
            "LFMAdapter: vocab=%d, max_len=%d, decoder_layers=%d, freeze=%s",
            config.vocab_size, config.max_output_len,
            config.decoder_num_layers, config.freeze_decoder,
        )

    def generate(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate NL tokens from latent z.

        Reshapes z [B, latent_dim] → [B, 1, latent_dim] as a single-token
        embedding sequence for the generator's pooling → VAE → decode pipeline.

        Args:
            z: [B, latent_dim] latent vectors
            mask: [B, 1] boolean mask (default: all True)

        Returns:
            Dict with keys from MultilingualVAEGenerator.forward():
                - tokens: [B, seq_len] int64
                - token_probs: [B, seq_len, vocab_size+2] float
                - embeddings: [B, seq_len, decoder_hidden_dim] float
                - lengths: [B] int
                - mask: [B, seq_len] bool
                - mu: [B, latent_dim]
                - logvar: [B, latent_dim]
        """
        embeddings = z.unsqueeze(1)  # [B, 1, latent_dim]
        if mask is None:
            mask = torch.ones(z.size(0), 1, dtype=torch.bool, device=z.device)
        return self.generator(embeddings, mask)

    def decode_to_text(self, token_ids: torch.Tensor) -> list[str]:
        """Decode token IDs to human-readable strings.

        Requires spm_model_path to be configured.
        """
        return self.generator.decode_to_text(token_ids)


class NLListener(nn.Module):
    """Decodes NL token distributions back to latent z.

    Jointly trained (NOT frozen). The listener and generator's input
    projection co-train under roundtrip L2 loss, creating an information
    bottleneck: the NL expression must encode enough about z that the
    listener can recover it.

    Architecture:
        token_probs [B, seq_len, vocab] → soft embedding lookup
        → Transformer encoder → mean pooling → Linear → z_hat [B, latent_dim]

    Args:
        config: NLListenerConfig with architecture settings
    """

    def __init__(self, config: NLListenerConfig):
        super().__init__()
        # +2 for BOS/EOS tokens (consistent with LFM's full_vocab)
        self.embedding = nn.Embedding(config.vocab_size + 2, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,
        )
        self.head = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        token_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode NL token probs back to latent z.

        Uses soft embedding lookup (differentiable through Gumbel-Softmax):
        soft_embeds = token_probs @ embedding.weight

        Args:
            token_probs: [B, seq_len, vocab_size+2] soft token distributions
            mask: [B, seq_len] boolean mask (True = valid position)

        Returns:
            z_hat: [B, latent_dim] predicted latent vector
        """
        # Differentiable soft embedding lookup
        soft_embeds = token_probs @ self.embedding.weight  # [B, seq_len, hidden_dim]

        # Transformer encoder with padding mask
        encoded = self.encoder(
            soft_embeds,
            src_key_padding_mask=~mask,  # True = ignore position
        )

        # Masked mean pooling
        mask_float = mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

        return self.head(pooled)  # [B, latent_dim]


class NLTokenBridge:
    """Implements LFM's TokenBridge protocol for interoperability.

    Wraps LFMAdapter.generate() to return a TokenBridgeOutput TypedDict
    compatible with LanguageFaculty.forward(tokens=..., embeddings=...).

    Args:
        adapter: LFMAdapter instance
    """

    def __init__(self, adapter: LFMAdapter):
        self._adapter = adapter

    def __call__(self, observation: Any) -> Dict[str, Any]:
        """Generate NL tokens from observation (latent z tensor).

        Args:
            observation: [B, latent_dim] tensor or any input accepted by adapter

        Returns:
            TokenBridgeOutput dict with 'tokens', 'embeddings', 'mask' keys.
        """
        from lfm._types import TokenBridgeOutput

        result = self._adapter.generate(observation)
        output: TokenBridgeOutput = {
            "tokens": result["tokens"],
            "embeddings": result["embeddings"],
            "mask": result["mask"],
        }
        return output
