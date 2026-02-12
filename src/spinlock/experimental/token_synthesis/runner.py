"""CLI runner for token synthesis self-play pipeline."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from spinlock.experimental.token_synthesis.config import TokenSynthesisConfig
from spinlock.experimental.token_synthesis.pipeline import SynthesisVerificationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Token Synthesis Self-Play")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to cycle checkpoint to resume from",
    )
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        raw_config = yaml.safe_load(f)

    config = TokenSynthesisConfig(**raw_config)
    pipeline = SynthesisVerificationPipeline(config)

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint resume for pipeline state

    metrics = pipeline.run()
    logger.info(f"Pipeline complete. Metrics keys: {list(metrics.keys())}")


if __name__ == "__main__":
    main()
