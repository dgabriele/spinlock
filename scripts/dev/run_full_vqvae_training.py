#!/usr/bin/env python
"""Full VQ-VAE training on 1K dataset with integrated metrics."""

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

# Create args
args = Namespace(
    config=Path("configs/vqvae_1k_full_training.yaml"),
    input=None,
    output=None,
    epochs=None,
    batch_size=None,
    learning_rate=None,
    resume_from=None,
    device=None,
    no_torch_compile=False,
    dry_run=False,
    verbose=True,
    val_every_n_epochs=None,
)

# Run command
cmd = TrainVQVAECommand()
sys.exit(cmd.execute(args))
