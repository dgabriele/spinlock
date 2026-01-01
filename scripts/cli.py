#!/usr/bin/env python3
"""
Spinlock CLI - Direct execution entry point.

Passthrough to spinlock.cli.main() for direct script execution.
Prefer using 'poetry run spinlock' over direct script execution.
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from spinlock.cli import main

if __name__ == "__main__":
    sys.exit(main())
