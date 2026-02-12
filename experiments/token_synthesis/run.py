#!/usr/bin/env python3
"""Entry point for token synthesis experiments.

Usage:
    python experiments/token_synthesis/run.py --config experiments/token_synthesis/configs/qbm_synthesis.yaml

Or via the runner module directly:
    python -m spinlock.experimental.token_synthesis.runner --config ...
"""

from spinlock.experimental.token_synthesis.runner import main

if __name__ == "__main__":
    main()
