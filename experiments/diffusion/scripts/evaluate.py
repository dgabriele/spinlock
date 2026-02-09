"""Evaluation script for discrete diffusion token completion."""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from spinlock.tokens.tokenizer import VQTokenizer
from experiments.diffusion.models import DiscreteD3PM, DenoisingNetwork
from experiments.diffusion.data import (
    HierarchicalMaskGenerator,
    MaskingStrategy,
    DiffusionCompletionDataset,
    collate_dict_batch,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    diffusion: DiscreteD3PM,
    denoiser: DenoisingNetwork,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict:
    """Evaluate diffusion model on dataset.

    Args:
        diffusion: DiscreteD3PM model
        denoiser: DenoisingNetwork model
        dataloader: Evaluation dataloader
        device: Device for computation

    Returns:
        Dict with evaluation metrics
    """
    denoiser.eval()
    diffusion.eval()

    total_accuracy = 0.0
    per_level_accuracy = {}
    num_batches = 0

    for batch in dataloader:
        # Move to device
        tokens = {k: v.to(device) for k, v in batch['tokens'].items()}
        observed = {k: v.to(device) for k, v in batch['observed'].items()}
        target = {k: v.to(device) for k, v in batch['target'].items()}

        # Full sampling loop (T → 0)
        batch_size = next(iter(tokens.values())).shape[0]
        generated = diffusion.sample(
            batch_size=batch_size,
            observed_dict=observed,
            x_0_dict=tokens,  # For RePaint
            denoising_network=denoiser,
            device=device,
        )

        # Compute accuracy on target positions
        total_correct = 0
        total_count = 0

        for key in tokens.keys():
            predictions = generated[key]
            targets = tokens[key]
            mask = target[key]

            # Per-category accuracy
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_count += mask.sum().item()

            # Per-level accuracy
            level = key.split('_L')[-1]
            if level not in per_level_accuracy:
                per_level_accuracy[level] = {'correct': 0, 'total': 0}

            per_level_accuracy[level]['correct'] += correct.sum().item()
            per_level_accuracy[level]['total'] += mask.sum().item()

        accuracy = total_correct / total_count if total_count > 0 else 0.0
        total_accuracy += accuracy
        num_batches += 1

    # Compute average accuracy
    avg_accuracy = total_accuracy / num_batches

    # Compute per-level accuracy
    level_acc = {}
    for level, counts in per_level_accuracy.items():
        level_acc[f"L{level}"] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0

    return {
        'overall_accuracy': avg_accuracy,
        'per_level_accuracy': level_acc,
    }


def main(args):
    """Main evaluation entry point."""
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']

    # Extract vocab sizes
    vocab_sizes = {}
    category_level_info = {}
    tokenizer = VQTokenizer.from_checkpoint(Path(config['dataset']['tokenizer_checkpoint']))

    for quantizer_key, quantizer in tokenizer.model.quantizers.items():
        vocab_size = quantizer.embedding.weight.shape[0]
        vocab_sizes[quantizer_key] = vocab_size

        parts = quantizer_key.split('_')
        family = parts[0]
        level = int(parts[-1][1:])
        category = '_'.join(parts[1:-1])

        category_level_info[quantizer_key] = {
            'family': family,
            'category': category,
            'level': level,
        }

    # Create mask generator
    mask_generator = HierarchicalMaskGenerator(
        strategy=MaskingStrategy(config['masking']['strategy']),
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=config['masking'].get('mask_probability', 0.5),
        seed=config['masking'].get('seed', 42),
    )

    # Create dataset
    dataset = DiffusionCompletionDataset(
        dataset_path=Path(config['dataset']['path']),
        tokenizer_checkpoint=Path(config['dataset']['tokenizer_checkpoint']),
        mask_generator=mask_generator,
        cache_tokens=True,
        device='cpu',
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dict_batch,
        num_workers=4,
    )

    # Create models
    diffusion_schedule = checkpoint['diffusion_state_dict']
    diffusion = DiscreteD3PM(
        vocab_sizes,
        config['diffusion'],
        category_level_info,
    )
    diffusion.load_state_dict(checkpoint['diffusion_state_dict'])

    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        **config['model'],
    )
    denoiser.load_state_dict(checkpoint['denoiser_state_dict'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion.to(device)
    denoiser.to(device)

    # Evaluate
    logger.info(f"Evaluating on {len(dataset)} samples")
    metrics = evaluate_model(diffusion, denoiser, dataloader, device=device)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info("\nPer-Level Accuracy:")
    for level, acc in sorted(metrics['per_level_accuracy'].items()):
        logger.info(f"  {level}: {acc:.4f}")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate discrete diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size",
    )

    args = parser.parse_args()
    main(args)
