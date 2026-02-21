"""Conditioning adapters for V2 MNO.

The ConditioningAdapter protocol decouples how conditioning inputs (parameters,
initial conditions, latent codes, etc.) are mapped to backbone-compatible tensors.
This enables future adapter variants (latent-conditioned, token-conditioned) without
modifying the V2MNO model or training loop.
"""

from typing import Dict, FrozenSet, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class ConditioningAdapter(Protocol):
    """Maps arbitrary conditioning inputs to backbone-compatible format."""

    def prepare(self, raw: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Transform raw conditioning dict into backbone-ready format.

        Must return at least ``{"ic": ...}``. May also return ``{"params": ...}``
        for parameter-conditioned backbones (FiLM / concat).
        """
        ...

    @property
    def required_keys(self) -> FrozenSet[str]:
        """Keys that must be present in the raw conditioning dict."""
        ...


class ThetaICAdapter:
    """Default adapter: theta + IC -> backbone params + initial condition.

    Expects raw dict with keys ``"theta"`` ([B, P]) and ``"ic"`` ([B, C, H, W]).
    Passes through as ``{"params": theta, "ic": ic}`` for the MNOBackbone.
    """

    @property
    def required_keys(self) -> FrozenSet[str]:
        return frozenset({"theta", "ic"})

    def prepare(self, raw: Dict[str, Tensor]) -> Dict[str, Tensor]:
        missing = self.required_keys - raw.keys()
        if missing:
            raise ValueError(f"Missing conditioning keys: {missing}")
        return {"ic": raw["ic"], "params": raw["theta"]}
