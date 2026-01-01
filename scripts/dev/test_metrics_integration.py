#!/usr/bin/env python
"""Quick test of integrated topology and per-category metrics."""

import sys
import logging
from pathlib import Path
from argparse import Namespace

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spinlock.cli.train_vqvae import TrainVQVAECommand

# Create args (mimicking CLI arguments)
args = Namespace(
    config=Path("test_vqvae_config.yaml"),
    input=None,  # Use value from config
    output=None,  # Use value from config
    epochs=None,  # Use value from config
    batch_size=None,  # Use value from config
    learning_rate=None,
    resume_from=None,
    device=None,
    no_torch_compile=True,  # Disable for faster testing
    dry_run=False,
    verbose=True,
    val_every_n_epochs=None,  # Use value from config
)

# Run command
cmd = TrainVQVAECommand()
sys.exit(cmd.execute(args))
