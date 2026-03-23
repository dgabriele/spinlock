"""Train NL Tokenizer command for Spinlock CLI.

Trains the NLTokenizer (continuous VAE + LFM integration) on a Lenia dataset.
The NLTokenizer encodes dynamics into continuous latent vectors that project
into LFM's frozen autoregressive decoder to generate natural language expressions.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class TrainNLTokenizerCommand(CLICommand):
    """Command to train NL tokenizer (continuous VAE + LFM)."""

    @property
    def name(self) -> str:
        return "train-nl-tokenizer"

    @property
    def help(self) -> str:
        return "Train NL tokenizer (continuous VAE + LFM text generation)"

    @property
    def description(self) -> str:
        return """
Train NL tokenizer for continuous latent → natural language generation.

Architecture:
- Family encoders (temporal, IC, theta) → concatenated embeddings
- VAE bottleneck: embeddings → (μ, logvar) → z = (z_coarse ‖ z_fine)
- LFM generator: z → Gumbel-Softmax NL tokens (frozen decoder)
- NL listener: token_probs → z_hat (roundtrip fidelity signal)
- Inverse decoders: z → theta_hat, z → IC_hat

Training stages:
1. VAE warmup: KL weight ramps 0 → full. Recon + inverse only.
2. Full + listener: Listener roundtrip loss enabled.

Examples:
  spinlock train-nl-tokenizer --config configs/lenia/nl/nl_v1.yaml
  spinlock train-nl-tokenizer --config configs/lenia/nl/nl_v1_smoke.yaml --device cuda
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--config", type=Path, required=True, metavar="PATH",
            help="Path to NLTokenizer config YAML",
        )
        parser.add_argument(
            "--dataset", type=Path, metavar="PATH",
            help="Override dataset path from config",
        )
        parser.add_argument(
            "--output", type=Path, metavar="PATH",
            help="Override output directory",
        )
        parser.add_argument(
            "--epochs", type=int, metavar="N",
            help="Override number of training epochs",
        )
        parser.add_argument(
            "--batch-size", type=int, metavar="N",
            help="Override batch size",
        )
        parser.add_argument(
            "--device", type=str, choices=["cuda", "cpu", "auto"],
            help="Override device",
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="Enable verbose logging",
        )
        parser.add_argument(
            "--max-samples", type=int, metavar="N",
            help="Limit dataset to first N samples (smoke tests)",
        )

    def execute(self, args: Namespace) -> int:
        """Execute the train-nl-tokenizer command."""
        import logging
        import yaml
        import torch
        from spinlock.data import SpinlockDataset
        from spinlock.tokens.nl_config import NLTokenizerConfig
        from spinlock.tokens.nl_tokenizer import NLTokenizer

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)

        # Validate config
        if not self.validate_file_exists(args.config, "Configuration file"):
            return 1

        # Load config
        logger.info(f"Loading config from {args.config}")
        try:
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            return self.error(f"Failed to load config: {e}")

        # Extract dataset path
        dataset_path = args.dataset or config_dict.pop("dataset_path", None)
        if not dataset_path:
            return self.error("No dataset path specified (use --dataset or set in config)")

        dataset_path = Path(dataset_path)
        if not self.validate_file_exists(dataset_path, "Dataset"):
            return 1

        # Apply CLI overrides
        output_dir = args.output or config_dict.pop("output_dir", "checkpoints/nl")
        output_dir = Path(output_dir)

        if args.epochs:
            config_dict.setdefault("training", {})["num_epochs"] = args.epochs
        if args.batch_size:
            config_dict.setdefault("training", {})["batch_size"] = args.batch_size
        if args.device:
            config_dict.setdefault("training", {})["device"] = args.device

        # Parse config
        try:
            config = NLTokenizerConfig(**config_dict)
        except Exception as e:
            return self.error(f"Invalid config: {e}")

        # Load dataset
        logger.info(f"Loading dataset: {dataset_path}")
        max_samples = getattr(args, "max_samples", None)
        dataset = SpinlockDataset.from_file(str(dataset_path), max_samples=max_samples)

        # Report GPU
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU: {gpu} ({mem:.1f} GB)")
        else:
            logger.warning("No GPU available — training will be slow")

        # Train
        tokenizer = NLTokenizer(config)
        try:
            history = tokenizer.train(
                dataset=dataset,
                output_dir=output_dir,
                checkpoint_prefix="nl_tokenizer",
            )
        except Exception as e:
            logger.exception("Training failed")
            return 1

        logger.info("NLTokenizer training complete. Checkpoints in %s", output_dir)
        return 0
