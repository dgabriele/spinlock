"""CLI command for computing per-position noise scale factors."""

import logging
from argparse import ArgumentParser, Namespace

from .base import CLICommand

logger = logging.getLogger(__name__)


class ComputePositionScalesCommand(CLICommand):
    """Compute per-position noise scales from cross-truncation token divergence."""

    @property
    def name(self) -> str:
        return "compute-position-scales"

    @property
    def help(self) -> str:
        return "Compute per-position noise scales from cross-truncation token divergence"

    @property
    def description(self) -> str:
        return (
            "Reads a multi-truncation pretokenized HDF5, compares token agreement "
            "between shortest and longest truncation for each position, and outputs "
            "per-position scale factors as JSON. Positions stable across truncations "
            "get low scale (resolve early); divergent positions get high scale (resolve late)."
        )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input",
            type=str,
            required=True,
            help="Path to multi-truncation pretokenized HDF5",
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Path to write JSON output",
        )
        parser.add_argument(
            "--min-temporal-scale",
            type=float,
            default=0.5,
            help="Floor for temporal scale factors (default: 0.5)",
        )
        parser.add_argument(
            "--max-temporal-scale",
            type=float,
            default=1.0,
            help="Ceiling for temporal scale factors (default: 1.0)",
        )
        parser.add_argument(
            "--non-temporal-scale",
            type=float,
            default=0.3,
            help="Scale factor for non-temporal keys (default: 0.3)",
        )

    def execute(self, args: Namespace) -> int:
        from experiments.diffusion.scripts.compute_position_scales import (
            compute_position_scales,
        )

        logger.info(f"Computing position scales from {args.input}")

        result = compute_position_scales(
            tokenized_path=args.input,
            output_path=args.output,
            non_temporal_scale=args.non_temporal_scale,
            min_temporal_scale=args.min_temporal_scale,
            max_temporal_scale=args.max_temporal_scale,
        )

        logger.info(f"Wrote {len(result)} scale factors to {args.output}")
        return 0
