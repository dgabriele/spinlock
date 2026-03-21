"""Train discrete diffusion model CLI command."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class TrainDiffusionCommand(CLICommand):
    """Train a discrete D3PM diffusion model on pretokenized token data."""

    @property
    def name(self) -> str:
        return "train-diffusion"

    @property
    def help(self) -> str:
        return "Train discrete D3PM diffusion model on pretokenized tokens"

    @property
    def description(self) -> str:
        return """Train a discrete D3PM diffusion model for token completion.

Requires a pretokenized dataset (produced by spinlock tokenize-dataset) and
a config YAML specifying the model, diffusion schedule, and training parameters.

Examples:
  spinlock train-diffusion --config configs/cno/d3pm_v1.yaml
  spinlock train-diffusion --config configs/cno/d3pm_v1.yaml --resume experiments/cno/diffusion/checkpoint_best.pt
"""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            metavar="PATH",
            help="Path to diffusion experiment config YAML",
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            metavar="PATH",
            help="Path to checkpoint to resume training from",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=None,
            metavar="N",
            help="Limit dataset to first N samples (for smoke tests)",
        )

    def execute(self, args: Namespace) -> int:
        import importlib.util
        train_script = (
            Path(__file__).resolve().parents[3]
            / "experiments" / "diffusion" / "scripts" / "train.py"
        )
        spec = importlib.util.spec_from_file_location("diffusion_train", train_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(args)
        return 0
