"""GPU memory management for LLM ↔ CPU swapping.

In swap mode, the LLM (~2.2GB fp16) is moved to CPU during exploration
batches (which only need denoiser + QBM on GPU), then loaded back for
scoring and response generation. On larger GPUs, resident mode keeps
everything on GPU as a no-op.

Follows the same .to(device) / .to('cpu') pattern used by the token
synthesis pipeline (pipeline.py:499-528).
"""

import gc
import logging
from contextlib import contextmanager

import torch

from spinlock.execution.memory import MemoryManager

logger = logging.getLogger(__name__)


class LLMGpuManager:
    """Manages LLM placement between GPU and CPU.

    In 'resident' mode: all methods are no-ops — the LLM stays on GPU.
    In 'swap' mode: load_llm() moves LLM to GPU + clears cache,
                    offload_llm() moves LLM to CPU + clears cache.

    The llm_on_gpu() context manager provides clean swap boundaries:
    LLM is loaded before the block, offloaded after.

    Args:
        llm: The LLM model (PeftModel wrapping a CausalLM).
        device: Target GPU device string or torch.device.
        mode: "resident" or "swap".
    """

    def __init__(self, llm, device, mode: str):
        self.llm = llm
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode
        self._on_gpu = True  # LLM starts on GPU after initialize()

    def load_llm(self) -> None:
        """Move LLM to GPU (no-op if resident or already on GPU)."""
        if self.mode == "resident" or self._on_gpu:
            return

        logger.debug("Loading LLM to GPU")
        with MemoryManager.managed_memory(self.device):
            self.llm.to(self.device)
        self._on_gpu = True

    def offload_llm(self) -> None:
        """Move LLM to CPU (no-op if resident or already on CPU)."""
        if self.mode == "resident" or not self._on_gpu:
            return

        logger.debug("Offloading LLM to CPU")
        self.llm.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        self._on_gpu = False

    @contextmanager
    def llm_on_gpu(self):
        """Context manager: ensure LLM is on GPU, offload after.

        Usage::

            with gpu_manager.llm_on_gpu():
                scores = evaluator.score_batch(query, descriptions)
                response = conversation.compose_response(...)
        """
        self.load_llm()
        try:
            yield
        finally:
            self.offload_llm()

    @property
    def is_on_gpu(self) -> bool:
        """Whether the LLM is currently on GPU."""
        return self._on_gpu
