"""NLTokenizer — high-level interface for continuous VAE + LFM NL generation.

Supports two feature modes (matching VQTokenizer):
- **Learned** (default, production): Raw trajectories [B, T, C, H, W] via
  PyramidFirstEncoder. Lazy dataset, on-the-fly trajectory generation.
- **Manual** (legacy): Pre-extracted temporal features [B, T, D].

Usage:
    >>> config = NLTokenizerConfig(...)
    >>> tokenizer = NLTokenizer(config)
    >>> tokenizer.train(dataset, output_dir="checkpoints/nl/")
    >>>
    >>> tokenizer = NLTokenizer.from_checkpoint("checkpoints/nl/best.pt")
    >>> z = tokenizer.encode(temporal_raw=trajectories, theta_features=theta)
    >>> text = tokenizer.generate_text(z)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from spinlock.data import SpinlockDataset

from .base_tokenizer import BaseTokenizer
from .nl_config import NLTokenizerConfig
from .nl_model import NLTokenizerModel
from .nl_lfm_adapter import LFMAdapter, NLListener
from .nl_trainer import NLTokenizerTrainer
from .nl_checkpoint import load_nl_checkpoint

logger = logging.getLogger(__name__)


class NLTokenizer(BaseTokenizer):
    """High-level interface for continuous VAE + NL tokenization.

    Args:
        config: NLTokenizerConfig
        model: Optional pre-initialized NLTokenizerModel
        adapter: Optional pre-initialized LFMAdapter
        listener: Optional pre-initialized NLListener
        group_indices: Optional pre-computed group indices
    """

    def __init__(
        self,
        config: NLTokenizerConfig,
        model: Optional[NLTokenizerModel] = None,
        adapter: Optional[LFMAdapter] = None,
        listener: Optional[NLListener] = None,
        group_indices: Optional[Dict[str, list]] = None,
    ):
        super().__init__(config, model=model, group_indices=group_indices)
        self.adapter = adapter
        self.listener = listener

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────

    def train(
        self,
        dataset: Union[SpinlockDataset, str, Path],
        output_dir: Union[str, Path] = "checkpoints",
        checkpoint_prefix: str = "nl_tokenizer",
        **kwargs,
    ) -> Dict[str, Any]:
        """Train NLTokenizer on dataset.

        Routes to learned or manual mode based on config.feature_source.
        """
        output_dir = Path(output_dir)

        if isinstance(dataset, (str, Path)):
            logger.info(f"Loading dataset from {dataset}")
            dataset = SpinlockDataset.from_file(str(dataset))

        return self._train_learned_mode(dataset, output_dir, checkpoint_prefix)

    # ──────────────────────────────────────────────────────────────
    # Learned mode (production path)
    # ──────────────────────────────────────────────────────────────

    def _train_learned_mode(
        self,
        dataset: SpinlockDataset,
        output_dir: Path,
        checkpoint_prefix: str,
    ) -> Dict[str, Any]:
        """Train with PyramidFirstEncoder on raw trajectories.

        Mirrors VQTokenizer._train_learned_mode():
        1. Create lazy dataset (ICs read per-sample from HDF5)
        2. Create sequential group indices (learned projection does grouping)
        3. Create replayer for on-the-fly trajectory generation
        4. Auto-detect dimensions from dataset
        5. Create model, adapter, listener, trainer
        6. Train with per-batch trajectory generation
        """
        logger.info("LEARNED MODE: PyramidFirst temporal features + on-the-fly trajectories")

        # ── Prepare data ──
        learned_dataset, replayer = self._prepare_learned_mode_data(dataset)

        # ── Auto-detect dimensions ──
        param_dim = dataset.theta_param_dim
        num_channels = dataset.num_channels

        # Auto-detect in_channels for PyramidFirstEncoder
        learned_cfg = self.config.encoder.temporal.learned
        if learned_cfg is not None and num_channels is not None and learned_cfg.in_channels is None:
            learned_cfg.in_channels = num_channels
            logger.info(f"Auto-detected in_channels={num_channels}")

        # Auto-detect theta param_dim
        if param_dim is not None and self.config.encoder.theta is not None:
            if self.config.encoder.theta.param_dim is None:
                self.config.encoder.theta.param_dim = param_dim
                logger.info(f"Auto-detected theta param_dim={param_dim}")

        # Auto-detect generation_timesteps
        if self.config.generation_timesteps is None:
            ts = dataset._dimension_cache.get("temporal_timesteps")
            self.config.generation_timesteps = ts or 64
            logger.info(f"Generation timesteps: {self.config.generation_timesteps}")

        # ── Create model ──
        logger.info("Creating NLTokenizerModel (learned mode)")
        self.model = NLTokenizerModel(
            self.config,
            self.group_indices,
            theta_param_dim=param_dim,
        )

        # ── Auto-set listener latent_dim to match z_full ──
        self.config.listener.latent_dim = self.model.z_full_dim
        logger.info(f"z_full_dim={self.model.z_full_dim}, lfm_projection_dim={self.config.vae.lfm_projection_dim}")

        # ── Create adapter + listener ──
        logger.info("Creating LFMAdapter and NLListener")
        self.adapter = LFMAdapter(self.config.lfm_adapter)
        self.listener = NLListener(self.config.listener)

        # ── Create trainer with replayer ──
        trainer = NLTokenizerTrainer(
            self.model, self.adapter, self.listener,
            self.config, self.group_indices,
            normalization_stats=self.normalization_stats,
            replayer=replayer,
        )

        # ── Train (pass lazy dataset — no tensors in RAM) ──
        logger.info(f"Starting learned-mode training for {self.config.training.num_epochs} epochs")
        history = trainer.train(
            dataset=learned_dataset,
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
        )

        learned_dataset.close_lazy()
        logger.info("Learned-mode training complete")
        return history

    def _prepare_learned_mode_data(
        self,
        dataset: SpinlockDataset,
    ) -> tuple:
        """Create lazy dataset, group indices, and replayer.

        Returns:
            (learned_dataset, replayer)
        """
        # ── Lazy dataset (ICs read per-sample, params eager) ──
        total = dataset.raw_input_shape[0]
        max_override = getattr(dataset, "_max_samples_override", None)
        if max_override is not None:
            total = min(total, max_override)
            logger.info(f"Dataset capped to {total} samples")

        learned_dataset = SpinlockDataset(
            str(dataset.file_path),
            max_samples=total,
            realization_mode=self.config.realization_mode,
            lazy_ics=True,
        )
        logger.info(
            f"Lazy dataset: {learned_dataset.n_samples} samples, "
            f"ICs read per-sample from HDF5"
        )

        # ── Group indices (sequential slicing for learned projection) ──
        learned_cfg = self.config.encoder.temporal.learned
        if learned_cfg is None:
            raise ValueError("feature_source='learned' requires encoder.temporal.learned")

        num_groups = learned_cfg.num_groups
        d_group = self.config.encoder.embedding_dim
        self.group_indices = {}

        for i in range(num_groups):
            self.group_indices[f"temporal_group_{i}"] = list(
                range(i * d_group, (i + 1) * d_group)
            )

        # Add theta group if configured
        param_dim = dataset.theta_param_dim
        if param_dim is not None and self.config.encoder.theta is not None:
            theta_cfg = self.config.encoder.theta
            if theta_cfg.variant == "direct":
                for i in range(param_dim):
                    self.group_indices[f"theta_param_{i}"] = [i]
            else:
                self.group_indices["theta_group_0"] = list(range(param_dim))

        logger.info(f"Group indices: {len(self.group_indices)} groups")

        # ── Replayer for on-the-fly trajectory generation ──
        replayer = self._create_replayer()

        return learned_dataset, replayer

    def _create_replayer(self):
        """Create trajectory replayer from generation_config_path."""
        config_path = self.config.generation_config_path
        if config_path is None:
            logger.warning("No generation_config_path — trajectories must be in dataset")
            return None

        import yaml as _yaml

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(config_path) as f:
            gen_config = _yaml.safe_load(f)
        operator_type = gen_config.get("simulation", {}).get("operator_type")

        match operator_type:
            case "lenia":
                from spinlock.lenia.replay_adapter import LeniaReplayAdapter
                replayer = LeniaReplayAdapter.from_config(config_path, device=device)
            case "cnn" | "u_afno":
                from spinlock.mno.cno_replay import CNOReplayer
                replayer = CNOReplayer.from_config(
                    config_path, device=device,
                    cache_size=self.config.replayer_cache_size,
                )
            case _:
                raise NotImplementedError(
                    f"No replay adapter for operator_type='{operator_type}'"
                )

        logger.info(f"Created {type(replayer).__name__} for '{operator_type}'")
        return replayer

    # ──────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────

    def encode(
        self,
        temporal_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode features to latent z.

        Returns:
            z: [B, latent_dim]
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        self.model.eval()
        with torch.no_grad():
            result = self.model.encode(
                temporal_raw=temporal_raw,
                theta_features=theta_features,
                temporal_mask=temporal_mask,
                temporal_lengths=temporal_lengths,
            )
        return result["z_full"]

    def generate_text(self, z_full: torch.Tensor) -> List[str]:
        """Generate NL expressions from z_full via LFM projection.

        Args:
            z_full: [B, z_full_dim] full latent vector

        Returns:
            List of B text strings
        """
        if self.adapter is None or self.model is None:
            raise ValueError("Model/adapter not initialized.")

        self.model.eval()
        self.adapter.eval()
        with torch.no_grad():
            z_lfm = self.model.z_to_lfm(z_full)
            gen_out = self.adapter.generate(z_lfm)
            return self.adapter.decode_to_text(gen_out["tokens"])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        **kwargs,
    ) -> "NLTokenizer":
        """Load NLTokenizer from saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading NL checkpoint from {checkpoint_path}")

        ckpt = load_nl_checkpoint(checkpoint_path)

        model = NLTokenizerModel(
            ckpt.config, ckpt.group_indices,
            temporal_input_dim=ckpt.temporal_input_dim,
            theta_param_dim=ckpt.theta_param_dim,
            initial_input_dim=ckpt.initial_input_dim,
        )
        model.load_state_dict(ckpt.model_state_dict)
        model.eval()

        adapter = LFMAdapter(ckpt.config.lfm_adapter)
        adapter.load_state_dict(ckpt.adapter_state_dict)
        adapter.eval()

        listener = NLListener(ckpt.config.listener)
        listener.load_state_dict(ckpt.listener_state_dict)
        listener.eval()

        tokenizer = cls(
            ckpt.config, model=model, adapter=adapter,
            listener=listener, group_indices=ckpt.group_indices,
        )
        tokenizer.normalization_stats = ckpt.normalization_stats
        logger.info("NL checkpoint loaded")
        return tokenizer

