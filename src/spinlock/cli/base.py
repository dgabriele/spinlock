"""
Base command class for Spinlock CLI.

Provides abstract interface and shared functionality for CLI commands.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict, Any
import sys


class CLICommand(ABC):
    """
    Abstract base class for CLI commands.

    Subclasses implement specific commands (generate, info, validate).
    Uses Command pattern for clean separation and testability.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Command name (e.g., 'generate')."""
        pass

    @property
    @abstractmethod
    def help(self) -> str:
        """Short help text for command."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed command description."""
        pass

    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """
        Add command-specific arguments to parser.

        Args:
            parser: ArgumentParser for this command
        """
        pass

    @abstractmethod
    def execute(self, args: Namespace) -> int:
        """
        Execute the command.

        Args:
            args: Parsed command-line arguments

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        pass

    def error(self, message: str, exit_code: int = 1) -> int:
        """
        Print error message and return exit code.

        Args:
            message: Error message to print
            exit_code: Exit code to return

        Returns:
            The exit code
        """
        print(f"Error: {message}", file=sys.stderr)
        return exit_code

    def validate_file_exists(self, path: Path, description: str = "File") -> bool:
        """
        Validate that a file exists.

        Args:
            path: Path to validate
            description: Human-readable description for error message

        Returns:
            True if file exists, False otherwise (with error printed)
        """
        if not path.exists():
            print(f"Error: {description} not found: {path}", file=sys.stderr)
            return False
        return True

    def validate_file_readable(self, path: Path, description: str = "File") -> bool:
        """
        Validate that a file is readable.

        Args:
            path: Path to validate
            description: Human-readable description for error message

        Returns:
            True if file is readable, False otherwise (with error printed)
        """
        if not self.validate_file_exists(path, description):
            return False

        try:
            with open(path, 'r'):
                pass
            return True
        except PermissionError:
            print(f"Error: {description} not readable: {path}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error: Cannot read {description.lower()}: {path}: {e}", file=sys.stderr)
            return False


class ConfigurableCommand(CLICommand):
    """
    Base class for commands that load and apply configuration.

    Provides utilities for loading configs and applying CLI overrides.
    """

    def load_config(
        self,
        config_path: Path,
        verbose: bool = False
    ) -> Optional[Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file
            verbose: Print loading information

        Returns:
            Loaded SpinlockConfig or None on error
        """
        from spinlock.config import load_config

        if not self.validate_file_exists(config_path, "Configuration file"):
            return None

        if verbose:
            print(f"Loading configuration from: {config_path}")

        try:
            return load_config(config_path)
        except Exception as e:
            print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
            return None

    def apply_overrides(
        self,
        config: Any,
        overrides: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """
        Apply CLI overrides to configuration.

        Args:
            config: SpinlockConfig object to modify
            overrides: Dictionary of override_path -> value
            verbose: Print applied overrides

        Example:
            >>> overrides = {
            ...     "dataset.output_path": Path("/tmp/output.h5"),
            ...     "sampling.total_samples": 5000
            ... }
            >>> command.apply_overrides(config, overrides)
        """
        for path, value in overrides.items():
            if value is None:
                continue

            # Split dotted path into components
            parts = path.split('.')
            obj = config

            # Navigate to parent object
            for part in parts[:-1]:
                if not hasattr(obj, part):
                    print(f"Warning: Unknown config path: {path}", file=sys.stderr)
                    continue
                obj = getattr(obj, part)

            # Set final attribute
            final_attr = parts[-1]
            if not hasattr(obj, final_attr):
                print(f"Warning: Unknown config attribute: {path}", file=sys.stderr)
                continue

            setattr(obj, final_attr, value)

            if verbose:
                print(f"  Override: {path} = {value}")
