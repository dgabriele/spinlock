#!/usr/bin/env python3
"""CLI entry point for D3PM roundtrip fidelity diagnostic.

Usage:
    # Smoke test (fast, 32 samples)
    python experiments/diffusion/diagnostics/scripts/run_roundtrip.py \\
        --config configs/cno/diffusion/diagnostic/roundtrip.yaml \\
        --num-samples 32 --output-dir /tmp/diag_smoke/

    # Full run
    python experiments/diffusion/diagnostics/scripts/run_roundtrip.py \\
        --config configs/cno/diffusion/diagnostic/roundtrip.yaml

Expected output JSON sanity checks:
    - family_match_rate values in [0, 1] (not NaN, not trivially 1.0 for all)
    - per_key_utilization: temporal keys show spread across codebook
    - mean_pairwise_jaccard > 0 (samples are not all identical)
    - replayer_type == "cno" or "qbm" (confirms registry dispatch worked)
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spinlock.experimental.common.config.loader import load_experiment_config
from experiments.diffusion.diagnostics.config import RoundtripDiagnosticConfig
from experiments.diffusion.diagnostics.roundtrip import RoundtripFidelityDiagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roundtrip fidelity diagnostic for D3PM unconditional generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to RoundtripDiagnosticConfig YAML",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override config.num_samples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config.batch_size",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config.output_dir",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config.device (e.g. 'cpu', 'cuda', 'cuda:1')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.seed",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_experiment_config(config_path, RoundtripDiagnosticConfig)

    # Apply CLI overrides
    if args.num_samples is not None:
        config = config.model_copy(update={"num_samples": args.num_samples})
    if args.batch_size is not None:
        config = config.model_copy(update={"batch_size": args.batch_size})
    if args.output_dir is not None:
        config = config.model_copy(update={"output_dir": Path(args.output_dir)})
    if args.device is not None:
        config = config.model_copy(update={"device": args.device})
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run diagnostic
    print(f"Running roundtrip fidelity diagnostic: {args.config}")
    print(f"  num_samples={config.num_samples}, batch_size={config.batch_size}")
    print(f"  device={config.device}, replayer={config.replayer is not None}")
    print()

    diag = RoundtripFidelityDiagnostic(config)
    result = diag.run()

    # Print summary
    result.print_summary()

    # Save JSON
    output_path = output_dir / "roundtrip_results.json"
    result.save_json(output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
