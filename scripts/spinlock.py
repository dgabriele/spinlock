#!/usr/bin/env python3
"""
Spinlock CLI - Main entry point.

Routes commands to modular command implementations in spinlock.cli package.
This script is a thin router; all logic lives in command classes.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from spinlock.cli import (
    GenerateCommand,
    InfoCommand,
    ValidateCommand,
    VisualizeCommand,
    VisualizeICTypesCommand,
)


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


if __name__ == "__main__":
    sys.exit(main())
