"""Quick sanity check: sample from trained D3PM and compare to real tokens.

Usage:
    python experiments/diffusion/scripts/sample_check.py \
        --checkpoint experiments/diffusion/results/v8_joint/v8_joint_d3pm_best.pt \
        --tokenized-path datasets/ds_lenia_fourier_10k_pretokenized.h5 \
        --tokenizer checkpoints/lenia/vq/v3_fourier_10k/vq_tokenizer_best.pt \
        --n-samples 64
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.schema import TokenSchema
from spinlock.experimental.diffusion.models import DiscreteD3PM, DiffusionSchedule, DenoisingNetwork
from spinlock.experimental.diffusion.data import PretokenizedDiffusionDataset, collate_dict_batch
from spinlock.experimental.diffusion.data.hierarchical_masking import (
    HierarchicalMaskGenerator, MaskingStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_from_checkpoint(checkpoint_path, tokenizer_path, device="cuda"):
    """Load D3PM + denoiser from checkpoint, vocab from tokenizer."""
    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    schema = TokenSchema.from_tokenizer(tokenizer)
    vocab_sizes = schema.vocab_sizes_dict()
    cat_info = schema.category_level_info_dict()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    # Rebuild D3PM
    import json
    graded_cfg = config.diffusion.graded_schedule
    scale_factors = graded_cfg.scale_factors or {}
    if graded_cfg.position_scale_factors_path:
        with open(graded_cfg.position_scale_factors_path) as f:
            scale_factors = json.load(f)

    diffusion = DiscreteD3PM(
        vocab_sizes,
        DiffusionSchedule(
            num_timesteps=config.diffusion.num_timesteps,
            schedule_type=config.diffusion.schedule_type,
        ),
        cat_info,
        transition_type=config.diffusion.transition_type,
        beta_scaling=config.diffusion.beta_scaling,
        graded_schedule_enabled=graded_cfg.enabled,
        graded_scale_factors=scale_factors,
        non_temporal_scale=graded_cfg.non_temporal_scale,
        family_scale_overrides=graded_cfg.family_scale_overrides,
    )
    diffusion.load_state_dict(ckpt["diffusion_state_dict"])

    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=cat_info,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        use_hierarchical_guidance=config.model.use_hierarchical_guidance,
        hierarchical_guidance_weight=config.model.hierarchical_guidance_weight,
        guidance_mode=config.model.hierarchical_guidance_mode,
        transition_type=config.diffusion.transition_type,
    )
    denoiser.load_state_dict(ckpt["denoiser_state_dict"])

    diffusion.to(device).eval()
    denoiser.to(device).eval()

    return diffusion, denoiser, vocab_sizes, cat_info


def load_real_tokens(tokenized_path, tokenizer_path, n_samples, truncation_length=512):
    """Load a batch of real tokens from pretokenized dataset."""
    # Use tokenizer-derived vocab (matches training), not pretokenized schema
    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    schema = TokenSchema.from_tokenizer(tokenizer)
    vocab_sizes = schema.vocab_sizes_dict()
    cat_info = schema.category_level_info_dict()

    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=cat_info,
        mask_probability=0.5,
    )
    ds = PretokenizedDiffusionDataset(
        tokenized_dataset_path=Path(tokenized_path),
        mask_generator=mask_gen,
        truncation_length=truncation_length,
    )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=n_samples, shuffle=True, collate_fn=collate_dict_batch,
    )
    batch = next(iter(loader))
    return {k: v for k, v in batch["tokens"].items()}


@torch.no_grad()
def main(args):
    device = "cuda"

    logger.info("Loading model from checkpoint...")
    diffusion, denoiser, vocab_sizes, cat_info = load_from_checkpoint(
        args.checkpoint, args.tokenizer, device,
    )

    logger.info(f"Sampling {args.n_samples} token sets (unconditional, T→0)...")
    generated = diffusion.sample(
        batch_size=args.n_samples,
        denoising_network=denoiser,
        device=device,
    )

    logger.info(f"Loading {args.n_samples} real token sets for comparison...")
    real = load_real_tokens(
        args.tokenized_path, args.tokenizer, args.n_samples,
    )

    # === Analysis ===
    families = defaultdict(list)
    for key, info in cat_info.items():
        families[info["family"]].append(key)

    print("\n" + "=" * 70)
    print("D3PM SAMPLING SANITY CHECK")
    print("=" * 70)

    # 1. Per-family token distribution comparison
    print(f"\n{'Family':<12} {'Metric':<20} {'Real':>10} {'Generated':>10} {'Match':>8}")
    print("-" * 62)

    for family in sorted(families.keys()):
        keys = families[family]

        # Collect all tokens for this family
        real_all = torch.cat([real[k] for k in keys], dim=0).numpy()
        gen_all = torch.cat([generated[k].cpu() for k in keys], dim=0).numpy()

        # Unique token count
        real_unique = len(np.unique(real_all))
        gen_unique = len(np.unique(gen_all))

        # Mean / std
        real_mean = real_all.mean()
        gen_mean = gen_all.mean()

        # Mode overlap: top-10 most common tokens
        real_counter = Counter(real_all.tolist())
        gen_counter = Counter(gen_all.tolist())
        real_top10 = set(t for t, _ in real_counter.most_common(10))
        gen_top10 = set(t for t, _ in gen_counter.most_common(10))
        top10_overlap = len(real_top10 & gen_top10)

        print(f"{family:<12} {'unique tokens':<20} {real_unique:>10} {gen_unique:>10} {'':>8}")
        print(f"{'':12} {'mean value':<20} {real_mean:>10.2f} {gen_mean:>10.2f} {'':>8}")
        print(f"{'':12} {'top-10 overlap':<20} {'':>10} {'':>10} {top10_overlap:>5}/10")

    # 2. Per-key exact match rate (are generated tokens in-vocab?)
    oov_count = 0
    total_count = 0
    for key in vocab_sizes:
        v = vocab_sizes[key]
        gen_tokens = generated[key].cpu()
        oov = (gen_tokens >= v).sum().item()
        oov_count += oov
        total_count += gen_tokens.numel()

    print(f"\nOut-of-vocab tokens: {oov_count}/{total_count} ({100*oov_count/total_count:.2f}%)")

    # 3. Per-key marginal distribution divergence (JS divergence)
    js_divs = defaultdict(list)
    for key in vocab_sizes:
        v = vocab_sizes[key]
        family = cat_info[key]["family"]

        real_hist = np.bincount(real[key].numpy(), minlength=v).astype(float)
        gen_hist = np.bincount(generated[key].cpu().numpy(), minlength=v).astype(float)

        # Normalize
        real_hist /= real_hist.sum() + 1e-10
        gen_hist /= gen_hist.sum() + 1e-10

        # JS divergence
        m = 0.5 * (real_hist + gen_hist)
        js = 0.5 * (
            np.sum(real_hist * np.log(real_hist / (m + 1e-10) + 1e-10))
            + np.sum(gen_hist * np.log(gen_hist / (m + 1e-10) + 1e-10))
        )
        js_divs[family].append(js)

    print(f"\n{'Family':<12} {'Mean JS div':>12} {'Max JS div':>12} {'Keys':>6}")
    print("-" * 44)
    for family in sorted(js_divs.keys()):
        divs = js_divs[family]
        print(f"{family:<12} {np.mean(divs):>12.4f} {np.max(divs):>12.4f} {len(divs):>6}")

    # 4. Cross-sample diversity: how many unique token-sets?
    # Flatten each sample's tokens to a tuple for hashing
    sample_hashes = set()
    for i in range(args.n_samples):
        token_tuple = tuple(
            generated[key][i].item() for key in sorted(vocab_sizes.keys())
        )
        sample_hashes.add(token_tuple)

    print(f"\nSample diversity: {len(sample_hashes)}/{args.n_samples} unique token-sets")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3PM sampling sanity check")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenized-path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    args = parser.parse_args()
    main(args)
