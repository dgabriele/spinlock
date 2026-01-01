"""
Spinlock CLI package.

Provides modular command implementations for the Spinlock CLI.
Commands use the Command pattern for clean separation and testability.
"""

import sys
import argparse

from .base import CLICommand, ConfigurableCommand
from .generate import GenerateCommand
from .info import InfoCommand
from .validate import ValidateCommand
from .visualize import VisualizeCommand
from .visualize_ic_types import VisualizeICTypesCommand
from .extract_features import ExtractFeaturesCommand
from .train_vqvae import TrainVQVAECommand

__all__ = [
    "CLICommand",
    "ConfigurableCommand",
    "GenerateCommand",
    "InfoCommand",
    "ValidateCommand",
    "VisualizeCommand",
    "VisualizeICTypesCommand",
    "ExtractFeaturesCommand",
    "TrainVQVAECommand",
    "main",
]


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with subcommands.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="spinlock",
        description="Spinlock - Stochastic Neural Operator Dataset Generator",
        epilog="""
Examples:
  # Generate dataset
  spinlock generate --config configs/experiments/default_10k.yaml

  # Show dataset info
  spinlock info --dataset datasets/default_10k.h5

  # Validate dataset
  spinlock validate --dataset datasets/default_10k.h5

For more help on a specific command:
  spinlock <command> --help
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version="Spinlock v1.0.0")

    # Create subcommands
    subparsers = parser.add_subparsers(
        title="commands", description="Available commands", dest="command", required=True
    )

    # Register commands
    commands = [
        GenerateCommand(),
        InfoCommand(),
        ValidateCommand(),
        VisualizeCommand(),
        VisualizeICTypesCommand(),
        ExtractFeaturesCommand(),
        TrainVQVAECommand(),
    ]

    for command in commands:
        # Create subparser for this command
        cmd_parser = subparsers.add_parser(
            command.name,
            help=command.help,
            description=command.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Let command add its arguments
        command.add_arguments(cmd_parser)

        # Set command as default handler
        cmd_parser.set_defaults(command_handler=command)

    return parser


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    # Execute command
    try:
        return args.command_handler.execute(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
