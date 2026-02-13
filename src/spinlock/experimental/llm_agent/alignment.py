"""Contrastive Rosetta alignment between token space and English space.

CLIP-style contrastive learning that aligns the denoiser's token
embeddings with LLM-encoded physics descriptions. After training,
any English query can be projected into the shared space and used
to retrieve or generate matching token sequences.

Training pairs are self-supervised:
    (token_sequence, auto_description) generated from QBM rollouts.

Architecture:
    Token side:  tokens → TokenEmbedder → [B, D_shared]
    Text side:   description → LLM+LoRA → mean-pool → Linear → [B, D_shared]
    Loss:        Symmetric InfoNCE with in-batch negatives (CLIP loss)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spinlock.experimental.llm_agent.config import AlignmentConfig
from spinlock.experimental.llm_agent.embedder import TokenEmbedder

logger = logging.getLogger(__name__)


class RosettaAlignment(nn.Module):
    """Contrastive alignment between token space and English space.

    After training, retrieve nearest tokens for any English query:
        "turbulent decaying system" → LLM embed → nearest in token space

    Trainable parameters:
        - TokenEmbedder: attention pooling query + projection layer
        - LoRA adapters on the LLM (via peft)
        - text_projector: Linear(D_llm → D_shared)
        - temperature: learned log-temperature scalar

    Frozen parameters:
        - DenoisingNetwork (denoiser backbone)
        - LLM backbone (only LoRA adapters train)

    Args:
        token_embedder: TokenEmbedder instance wrapping the frozen denoiser.
        llm_model: HuggingFace causal LM with LoRA adapters applied.
        llm_tokenizer: Corresponding tokenizer.
        config: AlignmentConfig with shared_dim, temperature, etc.
    """

    def __init__(
        self,
        token_embedder: TokenEmbedder,
        llm_model: Any,
        llm_tokenizer: Any,
        config: AlignmentConfig,
    ):
        super().__init__()
        self.token_embedder = token_embedder
        self.llm = llm_model
        self.tokenizer = llm_tokenizer
        self.config = config

        # Project LLM hidden states → shared dim
        llm_hidden_size = llm_model.config.hidden_size
        self.text_projector = nn.Linear(llm_hidden_size, config.shared_dim)

        # Learned log-temperature (initialized from config, trained via gradient)
        self.log_temperature = nn.Parameter(
            torch.tensor(config.temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def encode_text(self, descriptions: List[str]) -> torch.Tensor:
        """Encode text descriptions into the shared embedding space.

        Uses the LLM's last hidden states with mean-pooling over
        non-padding tokens, then projects to D_shared.

        Args:
            descriptions: List of natural language descriptions.

        Returns:
            [B, D_shared] — L2-normalizable text embeddings.
        """
        tokens = self.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        tokens = {k: v.to(self.llm.device) for k, v in tokens.items()}

        outputs = self.llm(**tokens, output_hidden_states=True)
        # Last hidden state: [B, seq_len, D_llm]
        hidden = outputs.hidden_states[-1]

        # Mean-pool over non-padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).float()  # [B, seq_len, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return self.text_projector(pooled)  # [B, D_shared]

    def encode_tokens(self, tokens_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode token sequences into the shared embedding space.

        Args:
            tokens_dict: Dict mapping key → token indices [B].

        Returns:
            [B, D_shared] — L2-normalizable token embeddings.
        """
        return self.token_embedder(tokens_dict)

    def compute_loss(
        self,
        token_embs: torch.Tensor,
        text_embs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Symmetric InfoNCE loss (CLIP loss).

        The diagonal of the similarity matrix should be high (matching
        pairs) while off-diagonals should be low (non-matching pairs).

        Args:
            token_embs: [B, D_shared] — token embeddings (from encode_tokens)
            text_embs: [B, D_shared] — text embeddings (from encode_text)

        Returns:
            Tuple of (scalar loss, metrics dict with loss components and accuracy)
        """
        # L2 normalize for cosine similarity
        token_embs = F.normalize(token_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

        # Similarity matrix: [B, B]
        logits = token_embs @ text_embs.T / self.temperature

        # Labels: diagonal (each token matches its own description)
        labels = torch.arange(len(logits), device=logits.device)

        # Symmetric loss: token→text and text→token
        loss_t2e = F.cross_entropy(logits, labels)
        loss_e2t = F.cross_entropy(logits.T, labels)
        loss = (loss_t2e + loss_e2t) / 2

        # Accuracy: fraction of correct nearest-neighbor retrievals
        with torch.no_grad():
            t2e_acc = (logits.argmax(dim=-1) == labels).float().mean()
            e2t_acc = (logits.T.argmax(dim=-1) == labels).float().mean()

        metrics = {
            "loss": loss.item(),
            "loss_token_to_text": loss_t2e.item(),
            "loss_text_to_token": loss_e2t.item(),
            "accuracy_t2e": t2e_acc.item(),
            "accuracy_e2t": e2t_acc.item(),
            "temperature": self.temperature.item(),
        }
        return loss, metrics

    def forward(
        self,
        tokens_dict: Dict[str, torch.Tensor],
        descriptions: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Full forward pass: encode both sides, compute loss.

        Args:
            tokens_dict: Dict mapping key → token indices [B].
            descriptions: List of B natural language descriptions.

        Returns:
            Tuple of (scalar loss, metrics dict)
        """
        token_embs = self.encode_tokens(tokens_dict)
        text_embs = self.encode_text(descriptions)
        return self.compute_loss(token_embs, text_embs)


class AlignmentTrainer:
    """Manages the alignment training loop.

    Accumulates (token_dict, description) pairs from synthesis runs,
    then trains the RosettaAlignment model when enough pairs are
    available.

    Args:
        alignment: RosettaAlignment model.
        config: AlignmentConfig with learning rate, batch size, etc.
    """

    def __init__(self, alignment: RosettaAlignment, config: AlignmentConfig):
        self.alignment = alignment
        self.config = config

        # Only optimize trainable parameters (LoRA + projectors + pooling)
        trainable = [p for p in alignment.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)

        # Pair buffer: list of (tokens_dict, description)
        self._buffer: List[Tuple[Dict[str, torch.Tensor], str]] = []

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def accumulate(
        self, tokens_dict: Dict[str, torch.Tensor], description: str
    ) -> None:
        """Add a (token_sequence, description) training pair.

        Args:
            tokens_dict: Dict mapping key → single token index (scalar tensors).
            description: Auto-generated physics description for this sequence.
        """
        # Detach and clone to CPU to avoid holding GPU memory
        pair_tokens = {
            k: v.detach().cpu().clone() for k, v in tokens_dict.items()
        }
        self._buffer.append((pair_tokens, description))

    def should_train(self) -> bool:
        """Check if enough pairs have accumulated for training."""
        return len(self._buffer) >= self.config.accumulation_threshold

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch over the accumulated buffer.

        Returns:
            Dict of epoch-level metrics (mean loss, accuracy, etc.)
        """
        import random

        self.alignment.train()
        device = next(self.alignment.text_projector.parameters()).device
        bs = self.config.batch_size

        # Shuffle buffer
        indices = list(range(len(self._buffer)))
        random.shuffle(indices)

        epoch_metrics: Dict[str, List[float]] = {
            "loss": [], "accuracy_t2e": [], "accuracy_e2t": [],
        }

        for start in range(0, len(indices), bs):
            batch_indices = indices[start:start + bs]
            if len(batch_indices) < 2:
                continue  # Need at least 2 for contrastive loss

            # Collate batch
            batch_tokens: Dict[str, List[torch.Tensor]] = {}
            batch_descriptions: List[str] = []
            for idx in batch_indices:
                tokens_dict, desc = self._buffer[idx]
                batch_descriptions.append(desc)
                for k, v in tokens_dict.items():
                    batch_tokens.setdefault(k, []).append(v)

            # Stack into batched tensors
            batched = {
                k: torch.stack(vs).to(device) for k, vs in batch_tokens.items()
            }

            # Forward + backward
            self.optimizer.zero_grad()
            loss, metrics = self.alignment(batched, batch_descriptions)
            loss.backward()
            self.optimizer.step()

            epoch_metrics["loss"].append(metrics["loss"])
            epoch_metrics["accuracy_t2e"].append(metrics["accuracy_t2e"])
            epoch_metrics["accuracy_e2t"].append(metrics["accuracy_e2t"])

        return {k: sum(vs) / max(len(vs), 1) for k, vs in epoch_metrics.items()}

    def train(self) -> List[Dict[str, float]]:
        """Run full alignment training for config.train_epochs.

        Returns:
            List of per-epoch metrics dicts.
        """
        logger.info(
            f"Starting alignment training: {self.config.train_epochs} epochs, "
            f"{len(self._buffer)} pairs, batch_size={self.config.batch_size}"
        )
        all_metrics = []
        for epoch in range(self.config.train_epochs):
            metrics = self.train_epoch()
            all_metrics.append(metrics)
            logger.info(
                f"  Epoch {epoch+1}/{self.config.train_epochs}: "
                f"loss={metrics['loss']:.4f} "
                f"acc_t2e={metrics['accuracy_t2e']:.3f} "
                f"acc_e2t={metrics['accuracy_e2t']:.3f}"
            )

        self.alignment.eval()
        return all_metrics
