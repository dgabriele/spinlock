"""Refine D3PM via offline hard-target loop (inverse generation test)."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class RefineD3PMCommand(CLICommand):
    """Run offline hard-target refinement loop for D3PM inverse generation."""

    @property
    def name(self) -> str:
        return "refine-d3pm"

    @property
    def help(self) -> str:
        return "Run D3PM refinement loop (inverse generation: temporal → theta+IC)"

    @property
    def description(self) -> str:
        return """Offline hard-target refinement for D3PM inverse generation.

Tests and improves the D3PM's ability to infer (theta, IC) from observed
temporal tokens by:
  1. Sampling novel parameters → rollout → tokenize
  2. D3PM inpaints theta+IC from observed temporal tokens
  3. Decode predicted params → rollout → retokenize → quality filter
  4. Fine-tune on accepted hard targets

Examples:
  spinlock refine-d3pm --config experiments/diffusion/configs/v13_refinement.yaml
  spinlock refine-d3pm --config experiments/diffusion/configs/v13_refinement.yaml --resume
  spinlock refine-d3pm --config experiments/diffusion/configs/v13_refinement.yaml --max-samples 50
"""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            metavar="PATH",
            help="Path to refinement config YAML",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=None,
            metavar="N",
            help="Override max_samples (for smoke tests)",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Override device (e.g., 'cpu' for smoke tests)",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume from last completed cycle checkpoint in output_dir",
        )

    def execute(self, args: Namespace) -> int:
        import importlib.util
        script = (
            Path(__file__).resolve().parents[3]
            / "experiments" / "diffusion" / "scripts" / "refine_d3pm.py"
        )
        spec = importlib.util.spec_from_file_location("refine_d3pm", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(args)
        return 0
