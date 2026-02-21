"""V2 MNO: Trajectory-first training.

Train the MNO as an accurate trajectory surrogate first (MSE + contrastive),
then separately train a VQTokenizer on MNO outputs and align token spaces.

Usage:
    spinlock train-mno-v2 --config configs/cno/mno/v2_trajectory.yaml
"""

from spinlock.mno.v2.conditioning import ConditioningAdapter, ThetaICAdapter
from spinlock.mno.v2.config import (
    V2ContrastiveConfig,
    V2LossConfig,
    V2MNOConfig,
    V2TrainingConfig,
)
from spinlock.mno.v2.evaluation import TrajectoryEvaluator
from spinlock.mno.v2.loss import TrajectoryLoss
from spinlock.mno.v2.model import V2MNO
from spinlock.mno.v2.trainer import V2Trainer

__all__ = [
    "ConditioningAdapter",
    "ThetaICAdapter",
    "TrajectoryEvaluator",
    "TrajectoryLoss",
    "V2ContrastiveConfig",
    "V2LossConfig",
    "V2MNOConfig",
    "V2TrainingConfig",
    "V2MNO",
    "V2Trainer",
]
