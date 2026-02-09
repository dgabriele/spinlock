"""CLI command for visualizing diffusion-based trajectory in-painting."""

import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import torch
import numpy as np

from spinlock.cli.base import CLICommand
from spinlock.tokens.tokenizer import VQTokenizer
from experiments.diffusion.models.discrete_d3pm import DiscreteD3PM
from experiments.diffusion.models.denoising_network import DenoisingNetwork
from experiments.diffusion.visualization.trajectory_inpainter import TrajectoryInpainter
from experiments.diffusion.visualization.comparison_visualizer import ComparisonVisualizer
from experiments.diffusion.visualization.masking_strategies import MaskingStrategy
import experiments.diffusion.config as diffusion_config

# Fix for unpickling: checkpoint was saved with 'config' module name
sys.modules['config'] = diffusion_config


class VisualizeDiffusionInpaintingCommand(CLICommand):
    """Visualize trajectory completion using trained diffusion model."""

    @property
    def name(self) -> str:
        return "visualize-diffusion-inpainting"

    @property
    def help(self) -> str:
        return "Generate trajectory in-painting visualizations using discrete diffusion"

    @property
    def description(self) -> str:
        return """
        Demonstrates trajectory completion from sparse observations using the trained
        discrete diffusion model. Creates comparison visualizations showing:
        - Original complete trajectory
        - Masked sparse observations
        - Diffusion reconstruction
        - Reconstruction error analysis
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        # Input group
        input_group = parser.add_argument_group("input configuration")
        input_group.add_argument(
            "--diffusion-checkpoint",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to trained diffusion model checkpoint (.pt)"
        )
        input_group.add_argument(
            "--tokenizer-checkpoint",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to VQTokenizer v2 checkpoint (.pt)"
        )
        input_group.add_argument(
            "--dataset",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to HDF5 dataset with trajectories"
        )

        # Sampling group
        sample_group = parser.add_argument_group("trajectory sampling")
        sample_group.add_argument(
            "--num-samples",
            type=int,
            default=5,
            metavar="N",
            help="Number of trajectories to visualize (default: 5)"
        )
        sample_group.add_argument(
            "--sample-indices",
            type=str,
            default=None,
            metavar="I1,I2,...",
            help="Specific trajectory indices to use (comma-separated)"
        )

        # Masking group
        mask_group = parser.add_argument_group("masking configuration")
        mask_group.add_argument(
            "--mask-strategy",
            choices=["temporal", "random", "keyframe"],
            default="temporal",
            help="Masking strategy: temporal (middle 50%%), random, keyframe"
        )
        mask_group.add_argument(
            "--mask-ratio",
            type=float,
            default=0.5,
            metavar="R",
            help="Fraction of trajectory to mask (default: 0.5)"
        )

        # Diffusion group
        diffusion_group = parser.add_argument_group("diffusion sampling")
        diffusion_group.add_argument(
            "--num-diffusion-steps",
            type=int,
            default=50,
            metavar="T",
            help="Number of diffusion timesteps (default: 50)"
        )
        diffusion_group.add_argument(
            "--device",
            choices=["cuda", "cpu"],
            default="cuda",
            help="Device for inference (default: cuda)"
        )

        # Output group
        output_group = parser.add_argument_group("output configuration")
        output_group.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            metavar="PATH",
            help="Output directory for visualizations"
        )
        output_group.add_argument(
            "--format",
            choices=["frames", "video", "both"],
            default="both",
            help="Output format (default: both)"
        )
        output_group.add_argument(
            "--fps",
            type=int,
            default=24,
            metavar="N",
            help="Frames per second for videos (default: 24)"
        )

        # Visualization group
        vis_group = parser.add_argument_group("visualization")
        vis_group.add_argument(
            "--colormap",
            default="seismic",
            help="Colormap for field rendering (default: seismic)"
        )
        vis_group.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )

    def execute(self, args: Namespace) -> int:
        """Main execution pipeline."""
        # Phase 1: Validation
        device = self._validate_and_setup(args)
        if device is None:
            return 1

        # Phase 2: Load models
        print("=" * 80)
        print("LOADING MODELS")
        print("=" * 80)
        models = self._load_models(args, device)
        if models is None:
            return 1
        diffusion, denoiser, tokenizer = models

        # Phase 3: Load trajectory samples
        print("\n" + "=" * 80)
        print("LOADING TRAJECTORY SAMPLES")
        print("=" * 80)
        samples = self._load_trajectory_samples(args)
        if samples is None:
            return 1

        # Phase 4: Run reconstruction
        print("\n" + "=" * 80)
        print("RUNNING DIFFUSION RECONSTRUCTION")
        print("=" * 80)
        results = self._run_reconstruction_pipeline(
            samples, diffusion, denoiser, tokenizer, args, device
        )
        if results is None:
            return 1

        # Phase 5: Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        success = self._generate_visualizations(results, args)
        if not success:
            return 1

        # Phase 6: Report summary
        self._print_summary(results, args)
        return 0

    def _validate_and_setup(self, args: Namespace) -> Optional[torch.device]:
        """Validate inputs and setup device."""
        # Check files exist
        if not args.diffusion_checkpoint.exists():
            print(f"Error: Diffusion checkpoint not found: {args.diffusion_checkpoint}")
            return None

        if not args.tokenizer_checkpoint.exists():
            print(f"Error: Tokenizer checkpoint not found: {args.tokenizer_checkpoint}")
            return None

        if not args.dataset.exists():
            print(f"Error: Dataset not found: {args.dataset}")
            return None

        # Parse device
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = torch.device("cpu")

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Device: {device}")
        print(f"✓ Output directory: {args.output_dir}")

        return device

    def _load_models(
        self,
        args: Namespace,
        device: torch.device
    ) -> Optional[Tuple[DiscreteD3PM, DenoisingNetwork, VQTokenizer]]:
        """Load diffusion model and tokenizer."""
        try:
            # Load tokenizer
            print(f"Loading VQTokenizer from: {args.tokenizer_checkpoint}")
            tokenizer = VQTokenizer.from_checkpoint(args.tokenizer_checkpoint)
            tokenizer.model.to(device)
            tokenizer.model.eval()
            print("✓ VQTokenizer loaded")

            # Get vocab sizes from the pre-tokenized dataset (actual max token values)
            # This matches how the diffusion model was trained
            print(f"Extracting vocab sizes from pre-tokenized tokens...")

            # We need to load a tokenized dataset to get the vocab sizes
            # Use the one from the diffusion config if available
            tokenized_path = Path("datasets/50k_baseline_tokenized.h5")
            if not tokenized_path.exists():
                print(f"Error: Pre-tokenized dataset not found: {tokenized_path}")
                print("This is needed to determine vocab sizes for the diffusion model.")
                return None

            vocab_sizes = {}
            with h5py.File(tokenized_path, 'r') as f:
                for key in f['tokens'].keys():
                    tokens = f[f'tokens/{key}'][:]
                    max_token = int(tokens.max())
                    vocab_sizes[key] = max_token + 1  # Vocab size is max_token + 1

            print(f"  Extracted {len(vocab_sizes)} category-levels")

            # Create category_level_info by parsing keys
            category_level_info = {}
            for key in vocab_sizes.keys():
                # Parse key format: "{family}_{category}_L{level}"
                parts = key.rsplit('_L', 1)
                if len(parts) == 2:
                    family_category = parts[0]
                    level = int(parts[1])

                    # Further split family and category
                    if family_category.startswith('temporal_'):
                        family = 'temporal'
                        category = family_category[len('temporal_'):]
                    elif family_category.startswith('initial_'):
                        family = 'initial'
                        category = family_category[len('initial_'):]
                    else:
                        family = 'unknown'
                        category = family_category

                    category_level_info[key] = {
                        'family': family,
                        'category': category,
                        'level': level
                    }

            # Use maximum vocab size for reference
            max_vocab_size = max(vocab_sizes.values())
            category_levels = sorted(list(vocab_sizes.keys()))

            print(f"  Vocab sizes range: {min(vocab_sizes.values())}-{max_vocab_size} across {len(category_levels)} category-levels")

            # Load diffusion checkpoint
            print(f"Loading diffusion checkpoint from: {args.diffusion_checkpoint}")
            checkpoint = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)

            # Extract configuration
            exp_config = checkpoint['config']

            # Initialize DiscreteD3PM with vocab_sizes dict
            from experiments.diffusion.models.discrete_d3pm import DiffusionSchedule

            diffusion = DiscreteD3PM(
                vocab_sizes=vocab_sizes,
                schedule=DiffusionSchedule(
                    num_timesteps=exp_config.diffusion.num_timesteps,
                    beta_start=exp_config.diffusion.beta_start,
                    beta_end=exp_config.diffusion.beta_end,
                    schedule_type=exp_config.diffusion.schedule_type
                )
            )
            diffusion.to(device)
            diffusion.eval()
            print("✓ DiscreteD3PM initialized")

            # Initialize DenoisingNetwork
            denoiser = DenoisingNetwork(
                vocab_sizes=vocab_sizes,
                category_level_info=category_level_info,
                hidden_dim=exp_config.model.hidden_dim,
                num_layers=exp_config.model.num_layers,
                num_heads=exp_config.model.num_heads,
                dropout=exp_config.model.dropout,
                use_hierarchical_guidance=exp_config.model.use_hierarchical_guidance,
                hierarchical_guidance_weight=exp_config.model.hierarchical_guidance_weight
            )
            denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
            denoiser.to(device)
            denoiser.eval()
            print("✓ DenoisingNetwork loaded")

            return diffusion, denoiser, tokenizer

        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_trajectory_samples(
        self,
        args: Namespace
    ) -> Optional[list]:
        """Load sample trajectories from dataset."""
        try:
            with h5py.File(args.dataset, 'r') as f:
                # Get dataset length from inputs
                total_samples = f['inputs/fields'].shape[0]

                # Determine which samples to load
                if args.sample_indices is not None:
                    indices = [int(i) for i in args.sample_indices.split(',')]
                else:
                    # Sample random indices
                    rng = np.random.RandomState(42)
                    indices = rng.choice(
                        total_samples,
                        size=min(args.num_samples, total_samples),
                        replace=False
                    ).tolist()

                print(f"Loading {len(indices)} samples: {indices}")

                # Load samples
                samples = []
                for idx in indices:
                    # Load vorticity field (first channel, mean across realizations)
                    # inputs/fields shape: [N, M, C, H, W] where M=3 realizations
                    fields = f['inputs/fields'][idx]  # [M, C, H, W]
                    vorticity = torch.from_numpy(fields[:, 0, :, :]).float()  # [M, H, W]
                    vorticity_mean = vorticity.mean(dim=0)  # [H, W] - average across realizations

                    # Load raw initial condition (for tokenizer hybrid encoder)
                    initial_raw = torch.from_numpy(fields.mean(axis=0)).float()  # [C, H, W] - mean across realizations

                    # Load features
                    temporal_features = torch.from_numpy(
                        f['features/temporal/features'][idx]
                    ).float()  # [T, D_t]

                    initial_manual = torch.from_numpy(
                        f['features/initial/aggregated/features'][idx]
                    ).float()  # [D_i]

                    samples.append({
                        'index': idx,
                        'vorticity_initial': vorticity_mean,  # [H, W] - just initial condition
                        'initial_raw': initial_raw,  # [C, H, W]
                        'temporal_features': temporal_features,
                        'initial_manual': initial_manual
                    })

                print(f"✓ Loaded {len(samples)} trajectories")
                print(f"  Vorticity (initial) shape: {samples[0]['vorticity_initial'].shape}")
                print(f"  Initial raw shape: {samples[0]['initial_raw'].shape}")
                print(f"  Temporal features shape: {samples[0]['temporal_features'].shape}")
                print(f"  Initial manual features shape: {samples[0]['initial_manual'].shape}")

                return samples

        except Exception as e:
            print(f"Error loading trajectory samples: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_reconstruction_pipeline(
        self,
        samples: list,
        diffusion: DiscreteD3PM,
        denoiser: DenoisingNetwork,
        tokenizer: VQTokenizer,
        args: Namespace,
        device: torch.device
    ) -> Optional[list]:
        """Execute tokenization → diffusion → detokenization."""
        try:
            # Initialize inpainter
            inpainter = TrajectoryInpainter(
                diffusion_model=diffusion,
                denoising_network=denoiser,
                tokenizer=tokenizer,
                device=device
            )

            # Process each sample
            results = []
            for i, sample in enumerate(samples):
                print(f"\n[{i+1}/{len(samples)}] Processing sample {sample['index']}...")

                # Get a sample token dict to determine structure
                with torch.no_grad():
                    sample_tokens = tokenizer.tokenize(
                        temporal_features=sample['temporal_features'].unsqueeze(0).to(device),
                        initial_manual=sample['initial_manual'].unsqueeze(0).to(device),
                        initial_raw=sample['initial_raw'].unsqueeze(0).to(device)
                    )

                # Create masking pattern
                mask_pattern = MaskingStrategy.create_token_mask_dict(
                    token_dict=sample_tokens,
                    strategy=args.mask_strategy,
                    mask_ratio=args.mask_ratio
                )

                # Run reconstruction
                reconstruction = inpainter.reconstruct_trajectory(
                    temporal_features=sample['temporal_features'],
                    initial_manual=sample['initial_manual'],
                    initial_raw=sample['initial_raw'],
                    mask_pattern=mask_pattern,
                    num_steps=args.num_diffusion_steps,
                    verbose=args.verbose
                )

                # Store token-level data for visualization
                results.append({
                    'sample_index': sample['index'],
                    'original_tokens': reconstruction['original_tokens'],
                    'masked_tokens': reconstruction['masked_tokens'],
                    'reconstructed_tokens': reconstruction['reconstructed_tokens'],
                    'mask_pattern': mask_pattern
                })

                print(f"  ✓ Reconstruction complete")

            return results

        except Exception as e:
            print(f"Error in reconstruction pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _generate_visualizations(
        self,
        results: list,
        args: Namespace
    ) -> bool:
        """Create token-level accuracy visualizations."""
        try:
            # Process each result
            for i, result in enumerate(results):
                sample_idx = result['sample_index']
                print(f"\n[{i+1}/{len(results)}] Generating visualizations for sample {sample_idx}...")

                # Create output directory for this sample
                sample_dir = args.output_dir / f"sample_{sample_idx}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Generate token accuracy analysis
                self._generate_token_accuracy_plot(
                    original_tokens=result['original_tokens'],
                    reconstructed_tokens=result['reconstructed_tokens'],
                    mask_pattern=result['mask_pattern'],
                    output_dir=sample_dir
                )

            return True

        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_token_accuracy_plot(
        self,
        original_tokens: Dict[str, torch.Tensor],
        reconstructed_tokens: Dict[str, torch.Tensor],
        mask_pattern: Dict[str, torch.BoolTensor],
        output_dir: Path
    ):
        """Generate comprehensive token accuracy visualization."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Compute metrics
        category_metrics = {}
        temporal_groups = []
        initial_groups = []

        for key in sorted(original_tokens.keys()):
            orig = original_tokens[key].cpu()
            recon = reconstructed_tokens[key].cpu()
            mask = mask_pattern.get(key, torch.ones_like(orig, dtype=torch.bool))

            # Compute accuracy on masked tokens only
            if orig.ndim == 1:
                # Scalar token
                correct = (orig == recon).float().item()
                total = 1
                masked_count = (~mask).sum().item() if mask.ndim > 0 else 0
            else:
                # Sequence
                masked_positions = ~mask.squeeze(0) if mask.ndim > 1 else ~mask
                if masked_positions.sum() > 0:
                    correct = (orig[:, masked_positions] == recon[:, masked_positions]).float().mean().item()
                    total = masked_positions.sum().item()
                    masked_count = total
                else:
                    correct = 1.0  # No masked tokens = perfect
                    total = 0
                    masked_count = 0

            category_metrics[key] = {
                'accuracy': correct,
                'masked_count': masked_count,
                'total': orig.numel()
            }

            if 'temporal' in key:
                temporal_groups.append((key, correct, masked_count))
            else:
                initial_groups.append((key, correct, masked_count))

        # Create visualization
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Overall accuracy by category-level
        ax1 = fig.add_subplot(gs[0, :])
        keys = sorted(category_metrics.keys())
        accuracies = [category_metrics[k]['accuracy'] * 100 for k in keys]
        colors = ['#2ecc71' if acc >= 90 else '#e74c3c' if acc < 50 else '#f39c12' for acc in accuracies]

        bars = ax1.bar(range(len(keys)), accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
        ax1.axhline(y=90, color='orange', linestyle='--', alpha=0.3, label='90%')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Token Reconstruction Accuracy by Category-Level', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels([k.replace('_', '\n') for k in keys], rotation=45, ha='right', fontsize=6)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if category_metrics[keys[i]]['masked_count'] > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.0f}%', ha='center', va='bottom', fontsize=7)

        # 2. Temporal groups accuracy
        ax2 = fig.add_subplot(gs[1, 0])
        if temporal_groups:
            temp_keys = [k.split('_')[2] + '_' + k.split('_')[-1] for k, _, _ in temporal_groups]
            temp_accs = [acc * 100 for _, acc, _ in temporal_groups]
            ax2.bar(range(len(temp_keys)), temp_accs, color='#3498db', alpha=0.7, edgecolor='black')
            ax2.axhline(y=100, color='green', linestyle='--', alpha=0.3)
            ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax2.set_title('Temporal Categories', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 105)
            ax2.set_xticks(range(len(temp_keys)))
            ax2.set_xticklabels(temp_keys, rotation=45, ha='right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. Initial groups accuracy
        ax3 = fig.add_subplot(gs[1, 1])
        if initial_groups:
            init_keys = [k.split('_')[2] + '_' + k.split('_')[-1] for k, _, _ in initial_groups]
            init_accs = [acc * 100 for _, acc, _ in initial_groups]
            ax3.bar(range(len(init_keys)), init_accs, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax3.axhline(y=100, color='green', linestyle='--', alpha=0.3)
            ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax3.set_title('Initial Categories', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 105)
            ax3.set_xticks(range(len(init_keys)))
            ax3.set_xticklabels(init_keys, rotation=45, ha='right', fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        total_masked = sum(m['masked_count'] for m in category_metrics.values())
        total_correct = sum(m['accuracy'] * m['masked_count'] for m in category_metrics.values())
        overall_acc = (total_correct / total_masked * 100) if total_masked > 0 else 100

        summary_text = f"""
        RECONSTRUCTION SUMMARY
        {'='*60}
        Total Category-Levels: {len(category_metrics)}
        Total Masked Tokens: {total_masked}
        Overall Accuracy: {overall_acc:.2f}%

        Temporal Categories: {len(temporal_groups)} groups
        Initial Categories: {len(initial_groups)} groups

        Perfect Reconstruction (100%): {sum(1 for m in category_metrics.values() if m['accuracy'] == 1.0)} groups
        High Accuracy (≥90%): {sum(1 for m in category_metrics.values() if m['accuracy'] >= 0.9)} groups
        Low Accuracy (<50%): {sum(1 for m in category_metrics.values() if m['accuracy'] < 0.5)} groups
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Diffusion Token Reconstruction Analysis', fontsize=16, fontweight='bold', y=0.98)

        # Save
        output_path = output_dir / "token_accuracy_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Token accuracy analysis saved: {output_path}")
        print(f"  ✓ Overall accuracy: {overall_acc:.2f}% ({int(total_correct)}/{total_masked} masked tokens correct)")

    def _print_summary(self, results: list, args: Namespace):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Compute token-level metrics
        all_accuracies = []
        all_masked_counts = []

        for result in results:
            sample_accuracy = []
            sample_masked = 0

            for key in result['original_tokens'].keys():
                orig = result['original_tokens'][key].cpu()
                recon = result['reconstructed_tokens'][key].cpu()
                mask = result['mask_pattern'].get(key, torch.ones_like(orig, dtype=torch.bool))

                if orig.ndim == 1:
                    if (orig == recon).all():
                        sample_accuracy.append(1.0)
                    else:
                        sample_accuracy.append(0.0)
                    masked_count = (~mask).sum().item() if mask.ndim > 0 else 0
                else:
                    masked_positions = ~mask.squeeze(0) if mask.ndim > 1 else ~mask
                    if masked_positions.sum() > 0:
                        acc = (orig[:, masked_positions] == recon[:, masked_positions]).float().mean().item()
                        sample_accuracy.append(acc)
                        masked_count = masked_positions.sum().item()
                    else:
                        sample_accuracy.append(1.0)
                        masked_count = 0

                sample_masked += masked_count

            all_accuracies.append(np.mean(sample_accuracy) if sample_accuracy else 1.0)
            all_masked_counts.append(sample_masked)

        # Create summary JSON
        summary = {
            "num_samples": len(results),
            "mask_strategy": args.mask_strategy,
            "mask_ratio": args.mask_ratio,
            "num_diffusion_steps": args.num_diffusion_steps,
            "metrics": {
                "mean_token_accuracy": float(np.mean(all_accuracies)),
                "std_token_accuracy": float(np.std(all_accuracies)),
                "total_masked_tokens": int(np.sum(all_masked_counts)),
                "mean_masked_per_sample": float(np.mean(all_masked_counts))
            },
            "sample_indices": [r['sample_index'] for r in results],
            "diffusion_checkpoint": str(args.diffusion_checkpoint),
            "tokenizer_checkpoint": str(args.tokenizer_checkpoint)
        }

        # Save summary
        summary_path = args.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print metrics
        print(f"Number of samples: {len(results)}")
        print(f"Mask strategy: {args.mask_strategy} (ratio={args.mask_ratio})")
        print(f"\nToken Reconstruction Quality:")
        print(f"  Mean Accuracy: {summary['metrics']['mean_token_accuracy']*100:.2f}% ± {summary['metrics']['std_token_accuracy']*100:.2f}%")
        print(f"  Total Masked Tokens: {summary['metrics']['total_masked_tokens']}")
        print(f"  Avg Masked per Sample: {summary['metrics']['mean_masked_per_sample']:.1f}")
        print(f"\n✓ Summary saved: {summary_path}")
        print(f"✓ All outputs saved to: {args.output_dir}")
