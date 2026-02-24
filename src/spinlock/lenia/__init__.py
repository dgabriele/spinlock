"""Lenia continuous cellular automaton simulator for Spinlock datasets."""

from .params import (
    DEFAULT_RANGES,
    LeniaBatchTensors,
    LeniaParamRanges,
    LeniaParams,
    sobol_batch_to_tensors,
    sobol_expected_dims,
    sobol_to_lenia_params,
)
from .simulator import (
    LeniaSimulator,
    LeniaKernelBuilder,
    GaussianKernelBuilder,
    KERNEL_BUILDERS,
    build_kernel_ffts_batched,
    build_multiring_kernel_ffts_batched,
)
from .initial_conditions import LeniaICGenerator
from .replayer import LeniaReplayer
from .replay_adapter import LeniaReplayAdapter

__all__ = [
    "DEFAULT_RANGES",
    "LeniaBatchTensors",
    "LeniaParamRanges",
    "LeniaParams",
    "sobol_batch_to_tensors",
    "sobol_expected_dims",
    "sobol_to_lenia_params",
    "LeniaSimulator",
    "LeniaKernelBuilder",
    "GaussianKernelBuilder",
    "KERNEL_BUILDERS",
    "build_kernel_ffts_batched",
    "build_multiring_kernel_ffts_batched",
    "LeniaICGenerator",
    "LeniaReplayer",
    "LeniaReplayAdapter",
]
