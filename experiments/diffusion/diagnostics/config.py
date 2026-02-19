"""Configuration classes for D3PM roundtrip fidelity diagnostics."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ReplayerConfig(BaseModel):
    """Optional simulator section. Omit the 'replayer' key to skip temporal-family roundtrip.

    The operator type is NOT specified here — it is auto-detected from the
    VQTokenizer checkpoint's stored feature metadata.
    Set operator_type_override only when auto-detection fails.
    """

    config_path: Path
    """Substrate YAML for the replayer (same config used to generate the dataset)."""

    num_realizations: int = 1
    """Number of stochastic realizations per sample. Keep low for diagnostic speed."""

    num_timesteps: int = 256
    """Number of simulation timesteps (full trajectory)."""

    operator_type_override: Optional[str] = None
    """Force a specific operator type: 'cno' or 'qbm'.
    Only needed if auto-detection from the tokenizer checkpoint fails."""


class RoundtripDiagnosticConfig(BaseModel):
    """Configuration for the roundtrip fidelity diagnostic.

    The diagnostic generates tokens via D3PM, decodes to physics parameters,
    optionally runs the operator simulator, re-extracts features, retokenizes,
    and measures per-key match rate.
    """

    diffusion_checkpoint: Path
    """DiffusionTrainer checkpoint (.pt). Contains denoiser_state_dict, diffusion_state_dict,
    and the full DiffusionExperimentConfig."""

    tokenizer_checkpoint: Path
    """VQTokenizer checkpoint (.pt). Used for decode() and re-tokenize()."""

    num_samples: int = 1000
    """Total number of token samples to generate and evaluate."""

    batch_size: int = 64
    """Generation and evaluation batch size."""

    device: str = "cuda"
    """PyTorch device string."""

    seed: int = 42
    """Base random seed for reproducible generation."""

    output_dir: Path = Path("experiments/diagnostics/roundtrip/")
    """Directory to write results JSON and summary."""

    replayer: Optional[ReplayerConfig] = None
    """Operator simulator config. None → partial roundtrip (theta + initial only)."""
