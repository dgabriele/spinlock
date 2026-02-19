"""Roundtrip fidelity diagnostic for D3PM unconditional generation.

Measures: per-key token match rate, per-family match rate, codebook utilization,
and mean pairwise Jaccard diversity across generated samples.

Roundtrip path:
    D3PM.sample()
        ↓ (generated tokens)
    VQTokenizer.decode() → (theta, u0)
        ↓ (operator params + initial conditions)
    [optional] Replayer.rollout() → trajectory
        ↓ (simulated trajectory)
    TemporalFeatureOrchestrator.extract_per_timestep()
        ↓ (raw temporal features)
    feature_mask [kept_feature_indices]
        ↓ (cleaned temporal features matching tokenizer training distribution)
    VQTokenizer.tokenize() → reencoded tokens
        ↓
    compare generated ↔ reencoded per token key

Theta + initial families always participate.
Temporal family participates only when a replayer config is provided.
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.diffusion.diagnostics.config import RoundtripDiagnosticConfig
from experiments.diffusion.diagnostics.replayers import (
    load_replayer,
    get_replayer_type_name,
)

logger = logging.getLogger(__name__)


@dataclass
class RoundtripResult:
    """Results of a roundtrip fidelity diagnostic run."""

    # --- Roundtrip match rates (subset of all keys) ---
    per_key_match_rate: Dict[str, float]
    """token key → fraction of samples where generated == reencoded."""

    family_match_rate: Dict[str, float]
    """family → mean match rate across all keys in that family."""

    # --- Coverage (all generated keys, always available) ---
    per_key_utilization: Dict[str, float]
    """token key → fraction of codebook codes observed in num_samples."""

    # --- Diversity (always available) ---
    mean_pairwise_jaccard: float
    """Mean Jaccard similarity between random pairs of generated token sets.
    High → samples are similar (low diversity). Target: < 0.5."""

    # --- Provenance ---
    num_samples: int
    full_roundtrip_families: List[str]
    """Families that have a match rate (theta, initial, + temporal if replayer)."""

    coverage_only_families: List[str]
    """Families with only utilization/diversity metrics (temporal without replayer)."""

    replayer_type: Optional[str]
    """Operator type used for simulation, e.g. 'cno' or 'qbm'. None if no replayer."""

    config: Any  # RoundtripDiagnosticConfig (Any to avoid dataclass JSON issues)

    def save_json(self, path: Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "per_key_match_rate": self.per_key_match_rate,
            "family_match_rate": self.family_match_rate,
            "per_key_utilization": self.per_key_utilization,
            "mean_pairwise_jaccard": self.mean_pairwise_jaccard,
            "num_samples": self.num_samples,
            "full_roundtrip_families": self.full_roundtrip_families,
            "coverage_only_families": self.coverage_only_families,
            "replayer_type": self.replayer_type,
            "config": {
                "diffusion_checkpoint": str(self.config.diffusion_checkpoint),
                "tokenizer_checkpoint": str(self.config.tokenizer_checkpoint),
                "num_samples": self.config.num_samples,
                "batch_size": self.config.batch_size,
                "device": self.config.device,
                "seed": self.config.seed,
                "output_dir": str(self.config.output_dir),
                "replayer": (
                    {
                        "config_path": str(self.config.replayer.config_path),
                        "num_realizations": self.config.replayer.num_realizations,
                        "num_timesteps": self.config.replayer.num_timesteps,
                        "operator_type_override": self.config.replayer.operator_type_override,
                    }
                    if self.config.replayer is not None
                    else None
                ),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {path}")

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        print()
        print("=" * 60)
        print("ROUNDTRIP FIDELITY DIAGNOSTIC — SUMMARY")
        print("=" * 60)
        print(f"  Samples:         {self.num_samples}")
        print(f"  Replayer:        {self.replayer_type or 'none (partial roundtrip)'}")
        print(f"  Full-roundtrip:  {self.full_roundtrip_families}")
        print(f"  Coverage-only:   {self.coverage_only_families}")
        print()

        # Per-family match rates
        if self.family_match_rate:
            print("  Family match rates (token survives decode→sim→reencode):")
            for family, rate in sorted(self.family_match_rate.items()):
                bar = _rate_bar(rate)
                print(f"    {family:25s}  {rate:.3f}  {bar}")
            print()

        # Per-key match rates (top/bottom 5)
        if self.per_key_match_rate:
            sorted_keys = sorted(
                self.per_key_match_rate.items(), key=lambda x: -x[1]
            )
            print("  Per-key match rate (best 5):")
            for key, rate in sorted_keys[:5]:
                print(f"    {key:40s}  {rate:.3f}")
            if len(sorted_keys) > 10:
                print("  ...")
                print("  Per-key match rate (worst 5):")
                for key, rate in sorted_keys[-5:]:
                    print(f"    {key:40s}  {rate:.3f}")
            print()

        # Utilization
        all_utils = list(self.per_key_utilization.values())
        if all_utils:
            print("  Codebook utilization (fraction of codes seen):")
            print(f"    min={min(all_utils):.3f}  mean={sum(all_utils)/len(all_utils):.3f}  max={max(all_utils):.3f}")
            print()

        # Diversity
        print(f"  Mean pairwise Jaccard:  {self.mean_pairwise_jaccard:.4f}")
        print("  (0 = all unique, 1 = all identical)")
        print("=" * 60)
        print()


def _rate_bar(rate: float, width: int = 20) -> str:
    filled = int(rate * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


class RoundtripFidelityDiagnostic:
    """Orchestrates the full roundtrip fidelity evaluation.

    Lifecycle:
        diag = RoundtripFidelityDiagnostic(config)
        result = diag.run()
        result.print_summary()
        result.save_json(...)
    """

    def __init__(self, config: RoundtripDiagnosticConfig):
        self.config = config
        self.device = torch.device(config.device)

        self._tokenizer = None
        self._diffusion = None
        self._denoiser = None
        self._replayer = None
        self._temporal_extractor = None
        self._temporal_feature_mask: Optional[np.ndarray] = None
        self._initial_extractor = None  # for initial_hybrid: computes initial_manual from u0
        self._vocab_sizes: Dict[str, int] = {}

        # Discovered at init
        self._full_roundtrip_families: List[str] = []
        self._coverage_only_families: List[str] = []
        self._replayer_type: Optional[str] = None

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        logger.info("Initializing RoundtripFidelityDiagnostic...")
        self._load_tokenizer()
        self._load_diffusion_model()
        self._load_initial_extractor()  # always — detects hybrid mode from tokenizer config
        if self.config.replayer is not None:
            self._load_replayer()
            self._load_temporal_extractor()
        self._resolve_families()
        logger.info(
            f"Diagnostic ready: full_roundtrip={self._full_roundtrip_families}, "
            f"coverage_only={self._coverage_only_families}, "
            f"replayer={self._replayer_type}"
        )

    def _load_tokenizer(self) -> None:
        from spinlock.tokens.tokenizer import VQTokenizer

        logger.info(f"Loading VQTokenizer from {self.config.tokenizer_checkpoint}")
        self._tokenizer = VQTokenizer.from_checkpoint(
            self.config.tokenizer_checkpoint
        )
        # Move tokenizer model to compute device.
        # decode() auto-moves inputs to the model's device, but tokenize() does not.
        self._tokenizer.model.to(self.device)
        logger.info("VQTokenizer loaded")

    def _load_diffusion_model(self) -> None:
        from spinlock.experimental.diffusion.models import (
            DenoisingNetwork,
            TemporalResolutionDenoisingNetwork,
            DiscreteD3PM,
            DiffusionSchedule,
        )
        from spinlock.tokens.schema import TokenSchema

        logger.info(f"Loading diffusion model from {self.config.diffusion_checkpoint}")
        checkpoint = torch.load(
            self.config.diffusion_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        config = checkpoint["config"]

        # Extract vocab sizes from transition matrices (authoritative source)
        diffusion_state = checkpoint["diffusion_state_dict"]
        diffusion_cfg = config.diffusion if hasattr(config, "diffusion") else config
        transition_type = getattr(diffusion_cfg, "transition_type", "uniform")

        vocab_sizes = {}
        for param_key, param in diffusion_state.items():
            if param_key.startswith("transition_matrices."):
                token_key = param_key[len("transition_matrices."):]
                V_eff = param.shape[1]
                if transition_type == "absorbing":
                    V_eff -= 1
                vocab_sizes[token_key] = V_eff

        self._vocab_sizes = vocab_sizes

        # Build category_level_info
        category_level_info = {}
        for key in vocab_sizes:
            parsed = TokenSchema.parse_key(key)
            category_level_info[key] = {
                "family": parsed.family,
                "category": parsed.category,
                "level": parsed.level,
            }

        # Reconstruct schedule
        schedule = DiffusionSchedule(
            num_timesteps=getattr(diffusion_cfg, "num_timesteps", 50),
            beta_start=getattr(diffusion_cfg, "beta_start", 0.0001),
            beta_end=getattr(diffusion_cfg, "beta_end", 0.02),
            schedule_type=getattr(diffusion_cfg, "schedule_type", "cosine"),
        )

        beta_scaling = getattr(diffusion_cfg, "beta_scaling", "uniform")
        self._diffusion = DiscreteD3PM(
            vocab_sizes,
            schedule,
            category_level_info,
            transition_type=transition_type,
            beta_scaling=beta_scaling,
        )
        self._diffusion.load_state_dict(checkpoint["diffusion_state_dict"])
        self._diffusion.to(self.device)
        self._diffusion.eval()

        # Build denoising network
        model_cfg = config.model if hasattr(config, "model") else config
        temporal_res = getattr(model_cfg, "temporal_resolution", None)

        if temporal_res and getattr(temporal_res, "enabled", False):
            from spinlock.tokens.schema import _TRUNC_RE
            truncations = set()
            for key in vocab_sizes:
                m = _TRUNC_RE.match(key)
                if m:
                    truncations.add(int(m.group(2)))
            truncation_lengths = sorted(truncations)

            self._denoiser = TemporalResolutionDenoisingNetwork(
                vocab_sizes=vocab_sizes,
                category_level_info=category_level_info,
                truncation_lengths=truncation_lengths,
                hidden_dim=getattr(model_cfg, "hidden_dim", 256),
                num_layers=getattr(model_cfg, "num_layers", 6),
                num_heads=getattr(model_cfg, "num_heads", 8),
                dropout=getattr(model_cfg, "dropout", 0.1),
                use_hierarchical_guidance=getattr(model_cfg, "use_hierarchical_guidance", True),
                hierarchical_guidance_weight=getattr(model_cfg, "hierarchical_guidance_weight", 0.1),
                guidance_mode=getattr(model_cfg, "hierarchical_guidance_mode", "global"),
                transition_type=transition_type,
                use_temporal_bias=getattr(temporal_res, "use_temporal_bias", True),
                temporal_bias_init=getattr(temporal_res, "temporal_bias_init", "causal"),
                temporal_bias_strength=getattr(temporal_res, "temporal_bias_strength", 0.1),
                enforce_causality=getattr(temporal_res, "enforce_causality", True),
            )
        else:
            self._denoiser = DenoisingNetwork(
                vocab_sizes=vocab_sizes,
                category_level_info=category_level_info,
                hidden_dim=getattr(model_cfg, "hidden_dim", 256),
                num_layers=getattr(model_cfg, "num_layers", 6),
                num_heads=getattr(model_cfg, "num_heads", 8),
                dropout=getattr(model_cfg, "dropout", 0.1),
                use_hierarchical_guidance=getattr(model_cfg, "use_hierarchical_guidance", True),
                hierarchical_guidance_weight=getattr(model_cfg, "hierarchical_guidance_weight", 0.1),
                guidance_mode=getattr(model_cfg, "hierarchical_guidance_mode", "global"),
                transition_type=transition_type,
            )

        self._denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
        self._denoiser.to(self.device)
        self._denoiser.eval()

        logger.info(
            f"Diffusion model loaded: {len(vocab_sizes)} token keys, "
            f"transition={transition_type}"
        )

    def _load_replayer(self) -> None:
        self._replayer = load_replayer(
            self.config.replayer,
            self._tokenizer,
            self.config.device,
        )
        self._replayer_type = get_replayer_type_name(self._replayer)

    def _load_initial_extractor(self) -> None:
        """Load IC feature extractor for initial_hybrid tokenizers.

        InitialHybridEncoder requires both initial_raw [B, C, H, W] AND
        initial_manual [B, D_manual] computed from the same IC. This extractor
        computes the manual statistical features from decoded u0, mirroring
        SynthesisVerificationPipeline._load_initial_extractor().
        """
        # Only needed when the tokenizer uses initial_hybrid mode
        enc_cfg = self._tokenizer.config.encoder.initial
        if getattr(enc_cfg, 'variant', None) == 'hybrid':
            from spinlock.features.initial.ic_feature_extractors import (
                InitialConditionsFeatureExtractor,
            )
            self._initial_extractor = InitialConditionsFeatureExtractor(
                device=str(self.device)
            )
            logger.info("Initial condition feature extractor loaded (for initial_manual)")

    def _load_temporal_extractor(self) -> None:
        """Load temporal feature extractor and feature cleaning mask.

        Mirrors `SynthesisVerificationPipeline._load_temporal_extractor()` and
        `_load_temporal_feature_mask()` in pipeline.py.
        """
        from spinlock.features.temporal.config import TemporalFeatureConfig
        from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator

        config = TemporalFeatureConfig()
        config.quantum.enabled = False  # match 247-feature dataset generation
        # Use CPU for feature extraction: the per-timestep statistics (histograms,
        # spectra, etc.) are memory-hungry and don't benefit from GPU acceleration
        # in diagnostic mode. This avoids OOM when processing long trajectories.
        self._temporal_extractor = TemporalFeatureOrchestrator(
            device=torch.device("cpu"), config=config
        )

        # Load temporal feature cleaning mask
        if (
            self._tokenizer.feature_metadata is not None
            and "temporal" in self._tokenizer.feature_metadata.families
        ):
            temporal_meta = self._tokenizer.feature_metadata.families["temporal"]
            self._temporal_feature_mask = np.array(temporal_meta.kept_feature_indices)
            logger.info(
                f"Temporal feature mask: {temporal_meta.original_feature_count} "
                f"→ {temporal_meta.cleaned_feature_count} features"
            )
        else:
            logger.warning(
                "No feature_metadata in tokenizer checkpoint — "
                "temporal features will be passed without cleaning"
            )

    def _resolve_families(self) -> None:
        """Determine which families participate in full vs. coverage-only roundtrip."""
        all_families = set()
        for key in self._vocab_sizes:
            from spinlock.tokens.schema import TokenSchema
            parsed = TokenSchema.parse_key(key)
            all_families.add(parsed.family)

        full_rt = []
        cov_only = []

        for family in sorted(all_families):
            if family == "temporal" and self._replayer is None:
                cov_only.append(family)
            else:
                full_rt.append(family)

        self._full_roundtrip_families = full_rt
        self._coverage_only_families = cov_only

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> RoundtripResult:
        """Execute the diagnostic and return results."""
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        all_generated: List[Dict[str, torch.Tensor]] = []
        match_accum: Dict[str, List[float]] = defaultdict(list)

        num_batches = 0
        for batch_start in range(0, self.config.num_samples, self.config.batch_size):
            B = min(self.config.batch_size, self.config.num_samples - batch_start)
            logger.info(
                f"Batch {num_batches + 1}: samples {batch_start}–{batch_start + B}"
            )

            generated = self._generate_batch(B)

            # Roundtrip and compare
            try:
                reencoded = self._roundtrip_batch(generated, seed_offset=batch_start)
                # Build a lookup: base_key → generated_token
                # For trunc-T format, use the T256 (highest T) variant as canonical.
                gen_base = self._normalize_tokens_for_decode(generated)
                for key in reencoded:
                    if key not in gen_base:
                        continue
                    gen_tok = gen_base[key].to(self.device)
                    re_tok = reencoded[key].to(self.device)
                    rate = (gen_tok == re_tok).float().mean().item()
                    match_accum[key].append(rate)
            except Exception as e:
                logger.warning(f"Roundtrip failed for batch {num_batches}: {e}", exc_info=True)

            all_generated.append(generated)
            num_batches += 1

        return self._aggregate(match_accum, all_generated)

    # ------------------------------------------------------------------
    # Internal: generation
    # ------------------------------------------------------------------

    def _generate_batch(self, B: int) -> Dict[str, torch.Tensor]:
        """Sample B token sets unconditionally from DiscreteD3PM."""
        with torch.no_grad():
            tokens = self._diffusion.sample(
                batch_size=B,
                denoising_network=self._denoiser,
                device=str(self.device),
            )
        return tokens

    # ------------------------------------------------------------------
    # Internal: roundtrip
    # ------------------------------------------------------------------

    def _normalize_tokens_for_decode(
        self, tokens: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remap temporal resolution tokens to base key format for VQTokenizer.decode().

        Temporal resolution D3PM generates keys like:
            temporal_group_0_trunc_T256_L0
        VQTokenizer.decode() expects base keys like:
            temporal_group_0_L0

        Strategy: for each base temporal key expected by the tokenizer,
        prefer the T256 (full-resolution) truncation variant if present.
        Non-temporal and non-truncated keys pass through unchanged.

        This remapping is necessary whenever the diffusion model uses a
        TemporalResolutionDenoisingNetwork (multi-truncation token schema).
        """
        import re
        trunc_re = re.compile(r'^(.+)_trunc_T(\d+)_(L\d+)$')

        # Collect all truncation variants: base_key → {T: token_tensor}
        trunc_by_base: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        non_trunc: Dict[str, torch.Tensor] = {}

        for key, val in tokens.items():
            m = trunc_re.match(key)
            if m:
                base_key = f"{m.group(1)}_{m.group(3)}"  # e.g. temporal_group_0_L0
                T_val = int(m.group(2))
                trunc_by_base[base_key][T_val] = val
            else:
                non_trunc[key] = val

        if not trunc_by_base:
            # No truncation variants — already base format
            return tokens

        remapped: Dict[str, torch.Tensor] = dict(non_trunc)
        for base_key, t_dict in trunc_by_base.items():
            # Prefer highest T (full resolution); fall back to largest available
            best_T = max(t_dict.keys())
            remapped[base_key] = t_dict[best_T]

        logger.debug(
            f"Remapped {len(trunc_by_base)} temporal-resolution keys → base format "
            f"(selected T={max(max(d.keys()) for d in trunc_by_base.values())} truncation)"
        )
        return remapped

    def _roundtrip_batch(
        self,
        tokens: Dict[str, torch.Tensor],
        seed_offset: int,
    ) -> Dict[str, torch.Tensor]:
        """Decode tokens → (optionally simulate) → retokenize.

        Returns only the families that can be compared:
        - theta + initial: always present (decode gives theta + u0 directly)
        - temporal: only if replayer was configured
        """
        # Move tokens to device for decode
        tokens_dev = {
            k: v.to(self.device) for k, v in tokens.items()
        }

        # For temporal-resolution D3PM: remap trunc_T256 → base key format
        tokens_for_decode = self._normalize_tokens_for_decode(tokens_dev)

        theta, u0 = self._tokenizer.decode(tokens_for_decode)
        # theta: [B, param_dim], u0: [B, C, H, W]

        if self._replayer is not None:
            return self._full_roundtrip(theta, u0, seed_offset)
        else:
            return self._partial_roundtrip(theta, u0)

    def _compute_initial_manual(self, u0: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute initial_manual features from decoded u0 if extractor is available."""
        if self._initial_extractor is None:
            return None
        # InitialConditionsFeatureExtractor.extract_all expects [B, C, H, W]
        return self._initial_extractor.extract_all(u0)

    def _partial_roundtrip(
        self,
        theta: torch.Tensor,
        u0: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Roundtrip without simulation: theta + initial only."""
        initial_manual = self._compute_initial_manual(u0)
        return self._tokenizer.tokenize(
            initial_manual=initial_manual,
            initial_raw=u0,
            theta_features=theta,
        )

    def _full_roundtrip(
        self,
        theta: torch.Tensor,
        u0: torch.Tensor,
        seed_offset: int,
    ) -> Dict[str, torch.Tensor]:
        """Full roundtrip: simulate trajectory, extract features, retokenize."""
        trajectory = self._call_replayer(theta, u0, seed_offset)
        # trajectory: [B, M, T, C, H, W]

        # Move to CPU for feature extraction — temporal stats (histograms, spectra)
        # are memory-intensive and don't need GPU. The extractor runs on CPU.
        temporal_raw = self._temporal_extractor.extract_per_timestep(
            trajectory.cpu()
        )
        # temporal_raw: [B, T, D_raw]

        # Apply feature cleaning mask (same indices as tokenizer training)
        if self._temporal_feature_mask is not None:
            mask_tensor = torch.tensor(
                self._temporal_feature_mask, dtype=torch.long, device=temporal_raw.device
            )
            temporal_features = temporal_raw[:, :, mask_tensor]
        else:
            temporal_features = temporal_raw

        # Move temporal_features to model device for tokenization
        temporal_features = temporal_features.to(self.device)
        initial_manual = self._compute_initial_manual(u0)
        return self._tokenizer.tokenize(
            temporal_features=temporal_features,
            initial_manual=initial_manual,
            initial_raw=u0,
            theta_features=theta,
        )

    def _call_replayer(
        self,
        theta: torch.Tensor,
        u0: torch.Tensor,
        seed_offset: int,
    ) -> torch.Tensor:
        """Dispatch to the correct replayer interface.

        Returns trajectory [B, M, T, C, H, W] in all cases.

        CNOReplayer: loops over batch calling rollout(params, ic=u0_i)
        QBMReplayer: calls rollout_batch_with_explicit_ic(params, psi_0)
        """
        theta_np = theta.cpu().numpy()
        replayer_type = type(self._replayer).__name__
        cfg = self.config.replayer
        seed = self.config.seed + seed_offset

        if replayer_type == "QBMReplayer":
            # QBM: explicit IC roundtrip using decoded u0 as psi_0
            # Returns [B, T, C, H, W] — add M=1 dimension
            traj = self._replayer.rollout_batch_with_explicit_ic(
                theta_np,
                psi_0=u0.to(self._replayer.device),
                num_timesteps=cfg.num_timesteps,
            )
            # [B, T, C, H, W] → [B, 1, T, C, H, W]
            traj = traj.unsqueeze(1)

        elif replayer_type == "CNOReplayer":
            # CNO: loop over batch (no rollout_batch on CNOReplayer)
            B = theta_np.shape[0]
            trajs = []
            for i in range(B):
                ic_i = u0[i].unsqueeze(0)  # [1, C, H, W]
                # rollout returns [1, M, T+1, C, H, W] for M>1 with return_all_steps
                # or [1, T+1, C, H, W] for M=1
                t_i = self._replayer.rollout(
                    params_vector=theta_np[i],
                    ic=ic_i.to(self._replayer.device),
                    timesteps=cfg.num_timesteps,
                    num_realizations=cfg.num_realizations,
                    seed=seed + i,
                    return_all_steps=True,
                )
                # Normalize to [1, M, T, C, H, W]
                if t_i.dim() == 4:
                    # [1, T+1, C, H] — shouldn't happen with return_all_steps
                    t_i = t_i.unsqueeze(1)
                if t_i.dim() == 5:
                    # [1, T+1, C, H, W] — M=1 case
                    t_i = t_i.unsqueeze(1)
                trajs.append(t_i)

            traj = torch.cat(trajs, dim=0)  # [B, M, T+1, C, H, W]

        else:
            raise ValueError(
                f"Unsupported replayer type '{replayer_type}'. "
                f"Expected 'QBMReplayer' or 'CNOReplayer'."
            )

        return traj

    # ------------------------------------------------------------------
    # Internal: metrics aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        match_accum: Dict[str, List[float]],
        all_generated: List[Dict[str, torch.Tensor]],
    ) -> RoundtripResult:
        """Compute final metrics from accumulated batch results."""
        # --- Per-key match rates ---
        per_key_match_rate = {
            key: float(np.mean(rates))
            for key, rates in match_accum.items()
        }

        # --- Family match rates ---
        family_totals: Dict[str, List[float]] = defaultdict(list)
        for key, rate in per_key_match_rate.items():
            from spinlock.tokens.schema import TokenSchema
            parsed = TokenSchema.parse_key(key)
            family_totals[parsed.family].append(rate)

        family_match_rate = {
            fam: float(np.mean(rates))
            for fam, rates in family_totals.items()
        }

        # --- Per-key utilization ---
        per_key_utilization = self._compute_utilization(all_generated)

        # --- Mean pairwise Jaccard ---
        mean_jaccard = self._compute_pairwise_jaccard(all_generated)

        # --- Actual sample count ---
        num_samples = sum(
            next(iter(batch.values())).shape[0]
            for batch in all_generated
            if batch
        )

        return RoundtripResult(
            per_key_match_rate=per_key_match_rate,
            family_match_rate=family_match_rate,
            per_key_utilization=per_key_utilization,
            mean_pairwise_jaccard=mean_jaccard,
            num_samples=num_samples,
            full_roundtrip_families=self._full_roundtrip_families,
            coverage_only_families=self._coverage_only_families,
            replayer_type=self._replayer_type,
            config=self.config,
        )

    def _compute_utilization(
        self, all_generated: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute codebook utilization: unique codes seen / total codes."""
        seen: Dict[str, set] = defaultdict(set)

        for batch in all_generated:
            for key, tokens in batch.items():
                seen[key].update(tokens.cpu().tolist())

        return {
            key: len(seen_codes) / self._vocab_sizes[key]
            for key, seen_codes in seen.items()
            if key in self._vocab_sizes
        }

    def _compute_pairwise_jaccard(
        self,
        all_generated: List[Dict[str, torch.Tensor]],
        max_pairs: int = 10_000,
    ) -> float:
        """Compute mean pairwise Jaccard similarity between token sets.

        Follows the token-set diversity pattern from
        docs/tokenizer-diversity-CORRECTED-FINAL.md.

        Samples up to max_pairs random pairs to keep cost O(1) in N.
        """
        # Collect all samples as flat tuples (one tuple per sample)
        all_samples: List[Tuple] = []
        for batch in all_generated:
            if not batch:
                continue
            B = next(iter(batch.values())).shape[0]
            sorted_keys = sorted(batch.keys())
            for i in range(B):
                tok_set = tuple(
                    int(batch[k][i].item()) for k in sorted_keys
                )
                all_samples.append(tok_set)

        N = len(all_samples)
        if N < 2:
            return 0.0

        num_pairs = min(max_pairs, N * (N - 1) // 2)
        rng = random.Random(self.config.seed)

        total_jaccard = 0.0
        for _ in range(num_pairs):
            i, j = rng.sample(range(N), 2)
            a = set(all_samples[i])
            b = set(all_samples[j])
            union = a | b
            if not union:
                continue
            intersection = a & b
            total_jaccard += len(intersection) / len(union)

        return total_jaccard / num_pairs if num_pairs > 0 else 0.0
