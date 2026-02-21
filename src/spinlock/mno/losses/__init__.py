"""MNO loss components.

The modular loss classes (MSELed, VQLed, ParameterSensitive, etc.) have been
removed. The v2 trajectory-first approach uses TrajectoryLoss directly.

Retained:
  - ContrastiveLoss: Core InfoNCE with MoCo queue, reused by v2 TrajectoryLoss.
"""

from spinlock.mno.losses.components.contrastive import ContrastiveLoss

__all__ = [
    "ContrastiveLoss",
]
