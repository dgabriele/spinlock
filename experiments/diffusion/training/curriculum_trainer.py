"""Curriculum learning trainer for staged masking strategy progression."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch

from .diffusion_trainer import DiffusionTrainer
from experiments.diffusion.data import HierarchicalMaskGenerator, MaskingStrategy

logger = logging.getLogger(__name__)


class CurriculumStage:
    """Single stage in curriculum learning.

    Args:
        name: Stage name (e.g., "coarse_only")
        strategy: Masking strategy to use
        num_epochs: Number of epochs for this stage
        learning_rate: Optional LR override for this stage
        mask_probability: Optional mask probability override
    """

    def __init__(
        self,
        name: str,
        strategy: MaskingStrategy,
        num_epochs: int,
        learning_rate: float = None,
        mask_probability: float = None,
    ):
        self.name = name
        self.strategy = strategy
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.mask_probability = mask_probability


class CurriculumDiffusionTrainer(DiffusionTrainer):
    """Diffusion trainer with curriculum learning support.

    Extends DiffusionTrainer to support staged training with different
    masking strategies at each stage.

    Args:
        curriculum_stages: List of CurriculumStage instances
        vocab_sizes: Dict mapping category-level → vocab_size
        category_level_info: Dict mapping key → {family, category, level}
        ... (other args same as DiffusionTrainer)

    Example:
        >>> stages = [
        ...     CurriculumStage("coarse", MaskingStrategy.COARSE_ONLY, num_epochs=10),
        ...     CurriculumStage("hierarchical", MaskingStrategy.HIERARCHICAL, num_epochs=10),
        ...     CurriculumStage("random", MaskingStrategy.RANDOM, num_epochs=10),
        ... ]
        >>> trainer = CurriculumDiffusionTrainer(stages, ...)
        >>> trainer.train_curriculum()
    """

    def __init__(
        self,
        curriculum_stages: List[CurriculumStage],
        vocab_sizes: Dict[str, int],
        category_level_info: Dict[str, Dict[str, any]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.curriculum_stages = curriculum_stages
        self.vocab_sizes = vocab_sizes
        self.category_level_info = category_level_info
        self.current_stage_idx = 0

        # Track stage history
        self.stage_history = []

        logger.info(
            f"CurriculumDiffusionTrainer initialized: {len(curriculum_stages)} stages, "
            f"total_epochs={sum(s.num_epochs for s in curriculum_stages)}"
        )

    def train_curriculum(self) -> Dict[str, Any]:
        """Train through all curriculum stages.

        Returns:
            Dict with complete training history across all stages
        """
        logger.info("=" * 70)
        logger.info("Starting Curriculum Learning")
        logger.info("=" * 70)

        total_epochs = sum(stage.num_epochs for stage in self.curriculum_stages)
        logger.info(
            f"Total stages: {len(self.curriculum_stages)}, "
            f"total epochs: {total_epochs}"
        )

        for stage_idx, stage in enumerate(self.curriculum_stages):
            self.current_stage_idx = stage_idx
            logger.info("\n" + "=" * 70)
            logger.info(
                f"Stage {stage_idx + 1}/{len(self.curriculum_stages)}: {stage.name}"
            )
            logger.info(f"Strategy: {stage.strategy}, Epochs: {stage.num_epochs}")
            logger.info("=" * 70)

            # Create new mask generator for this stage
            stage_mask_gen = self._create_stage_mask_generator(stage)

            # Update dataset's mask generator
            self._update_dataset_mask_generator(stage_mask_gen)

            # Optionally adjust learning rate
            if stage.learning_rate is not None:
                self._set_learning_rate(stage.learning_rate)
                logger.info(f"Learning rate set to {stage.learning_rate}")

            # Train for this stage
            stage_start_epoch = self.current_epoch
            stage_end_epoch = stage_start_epoch + stage.num_epochs

            stage_metrics = self.train(num_epochs=stage.num_epochs)

            # Record stage results
            self.stage_history.append({
                'stage_idx': stage_idx,
                'stage_name': stage.name,
                'strategy': str(stage.strategy),
                'start_epoch': stage_start_epoch,
                'end_epoch': stage_end_epoch,
                'metrics': stage_metrics,
            })

            # Save stage checkpoint
            self.save_stage_checkpoint(stage)

            logger.info(f"Stage {stage.name} complete")

        logger.info("\n" + "=" * 70)
        logger.info("Curriculum Learning Complete")
        logger.info("=" * 70)

        # Combine history from all stages
        combined_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'stages': self.stage_history,
        }

        for stage_record in self.stage_history:
            metrics = stage_record['metrics']
            combined_history['train_loss'].extend(metrics.get('train_loss', []))
            combined_history['val_loss'].extend(metrics.get('val_loss', []))
            combined_history['learning_rate'].extend(metrics.get('learning_rate', []))

        return combined_history

    def _create_stage_mask_generator(
        self, stage: CurriculumStage
    ) -> HierarchicalMaskGenerator:
        """Create mask generator for current stage."""
        mask_prob = stage.mask_probability or 0.5
        return HierarchicalMaskGenerator(
            strategy=stage.strategy,
            vocab_sizes=self.vocab_sizes,
            category_level_info=self.category_level_info,
            mask_probability=mask_prob,
            seed=self.config.get('seed', 42) + self.current_stage_idx,  # Different seed per stage
        )

    def _update_dataset_mask_generator(self, new_generator: HierarchicalMaskGenerator):
        """Update mask generator in train/val datasets.

        Note: This assumes datasets expose a .mask_generator attribute.
        """
        if hasattr(self.train_loader.dataset, 'dataset'):
            # Handle random_split wrapper
            base_dataset = self.train_loader.dataset.dataset
        else:
            base_dataset = self.train_loader.dataset

        if hasattr(base_dataset, 'mask_generator'):
            base_dataset.mask_generator = new_generator
            logger.info("Train dataset mask generator updated")
        else:
            logger.warning("Cannot update mask generator: dataset has no mask_generator attribute")

        # Update val dataset
        if hasattr(self.val_loader.dataset, 'dataset'):
            base_val_dataset = self.val_loader.dataset.dataset
        else:
            base_val_dataset = self.val_loader.dataset

        if hasattr(base_val_dataset, 'mask_generator'):
            base_val_dataset.mask_generator = new_generator
            logger.info("Val dataset mask generator updated")

    def _set_learning_rate(self, lr: float):
        """Set learning rate for optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_stage_checkpoint(self, stage: CurriculumStage):
        """Save checkpoint at end of stage."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'denoiser_state_dict': self.denoiser.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
            'stage_idx': self.current_stage_idx,
            'stage_name': stage.name,
            'stage_history': self.stage_history,
        }

        prefix = self.config['output']['checkpoint_prefix']
        path = self.output_dir / f"{prefix}_stage_{stage.name}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Stage checkpoint saved: {path}")


# Predefined curriculum templates

def get_coarse_to_fine_curriculum() -> List[CurriculumStage]:
    """Standard coarse-to-fine curriculum.

    Stage 1: COARSE_ONLY (10 epochs) - Learn hierarchical structure from L0
    Stage 2: HIERARCHICAL (10 epochs) - Refine with L0+L1 → L2
    Stage 3: RANDOM (10 epochs) - Generalize to arbitrary masking
    """
    return [
        CurriculumStage(
            name="coarse_only",
            strategy=MaskingStrategy.COARSE_ONLY,
            num_epochs=10,
        ),
        CurriculumStage(
            name="hierarchical",
            strategy=MaskingStrategy.HIERARCHICAL,
            num_epochs=10,
        ),
        CurriculumStage(
            name="random",
            strategy=MaskingStrategy.RANDOM,
            num_epochs=10,
        ),
    ]


def get_fast_curriculum() -> List[CurriculumStage]:
    """Fast curriculum for quick experimentation.

    Stage 1: COARSE_ONLY (5 epochs)
    Stage 2: RANDOM (10 epochs)
    """
    return [
        CurriculumStage(
            name="coarse_only",
            strategy=MaskingStrategy.COARSE_ONLY,
            num_epochs=5,
        ),
        CurriculumStage(
            name="random",
            strategy=MaskingStrategy.RANDOM,
            num_epochs=10,
        ),
    ]


def get_fine_tuning_curriculum(pretrained_epochs: int = 20) -> List[CurriculumStage]:
    """Curriculum for fine-tuning with decreasing learning rates.

    Stage 1: RANDOM with high LR (pretrained_epochs)
    Stage 2: HIERARCHICAL with medium LR (5 epochs) - Refine fine details
    Stage 3: RANDOM with low LR (5 epochs) - Final polish
    """
    return [
        CurriculumStage(
            name="random_pretrain",
            strategy=MaskingStrategy.RANDOM,
            num_epochs=pretrained_epochs,
            learning_rate=1e-4,
        ),
        CurriculumStage(
            name="hierarchical_refine",
            strategy=MaskingStrategy.HIERARCHICAL,
            num_epochs=5,
            learning_rate=5e-5,
        ),
        CurriculumStage(
            name="random_polish",
            strategy=MaskingStrategy.RANDOM,
            num_epochs=5,
            learning_rate=1e-5,
        ),
    ]
