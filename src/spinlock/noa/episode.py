"""Episode management for autonomous MNO operation.

An episode represents a complete perturbation-response experiment:
1. Start from initial state u₀ (fresh or from previous episode)
2. Apply perturbation at specified timesteps
3. Evolve MNO until early stopping criteria met
4. Extract token sequence via VQ-VAE
5. Store complete episode record

Supports dual modes:
- Continuous: Chain episodes (u₀ from previous episode's final state)
- Exploratory: Independent episodes (fresh u₀ each time)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
from torch import Tensor
import time

from spinlock.perturbations.base import BasePerturbation
from spinlock.noa.backbone import NOABackbone
from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE
from .early_stopping import EarlyStoppingCriterion, EpisodeState


@dataclass
class Episode:
    """Complete record of a perturbation-response experiment.

    Attributes:
        episode_id: Unique identifier for this episode
        perturbation: Applied perturbation
        initial_state: Starting state u₀ [C, H, W]
        trajectory: Full state evolution [T, C, H, W]
        token_sequence: VQ-VAE tokens [T, num_categories * num_levels]
        stop_reason: Why episode terminated
        metadata: Additional information (timing, convergence, etc.)
    """

    episode_id: str
    perturbation: BasePerturbation
    initial_state: Tensor
    trajectory: Tensor
    token_sequence: Tensor
    stop_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def final_state(self) -> Tensor:
        """Get final state of episode.

        Returns:
            Final state [C, H, W]
        """
        return self.trajectory[-1]

    def num_steps(self) -> int:
        """Get number of timesteps in episode.

        Returns:
            Number of timesteps
        """
        return self.trajectory.shape[0]


class EpisodeRunner:
    """Executes MNO under perturbations until early stopping.

    The runner is mode-agnostic: it takes u₀ as input and doesn't care
    whether it's a fresh sample or from a previous episode. The calling
    code (ExplorationLoop, manual scripts, etc.) decides the mode:

    Continuous mode (Phase 4-5):
        u₀ = previous_episode.final_state()
        episode = runner.run_episode(u₀, new_perturbation)

    Exploratory mode (Phase 2-3):
        u₀ = sample_initial_condition()
        episode = runner.run_episode(u₀, perturbation)
    """

    def __init__(
        self,
        mno: NOABackbone,
        vqvae: CategoricalHierarchicalVQVAE,
        early_stopping: EarlyStoppingCriterion,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize episode runner.

        Args:
            mno: Trained Meta-Neural Operator backbone
            vqvae: Trained VQ-VAE for behavioral tokenization
            early_stopping: Early stopping criterion
            device: Device for computation
        """
        self.mno = mno.to(device)
        self.vqvae = vqvae.to(device)
        self.early_stopping = early_stopping
        self.device = device

        # Set models to eval mode
        self.mno.eval()
        self.vqvae.eval()

        # Initialize feature extractor matching VQ-VAE training distribution
        # Note: Extracts per-timestep features (TEMPORAL family only)
        from .vqvae_feature_extraction import VQVAEFeatureExtractor
        self.feature_extractor = VQVAEFeatureExtractor(device=device)

        self._episode_counter = 0

    def run_episode(
        self,
        u0: Tensor,
        perturbation: BasePerturbation,
        max_steps: Optional[int] = None,
        store_trajectory: bool = True,
    ) -> Episode:
        """Execute MNO rollout with perturbation until early stopping.

        Args:
            u0: Initial state [C, H, W] (can be fresh sample or from previous episode)
            perturbation: Perturbation to apply during rollout
            max_steps: Override maximum steps (if None, use early_stopping's max)
            store_trajectory: If False, only store first and last states (memory savings)

        Returns:
            Complete episode record

        Note:
            This method is mode-agnostic. Whether u0 comes from a fresh sample
            or a previous episode is determined by the caller.

            Feature extraction uses VQVAEFeatureExtractor which matches the exact
            features used during VQ-VAE training (63D per-timestep: spatial +
            spectral + cross-channel).
        """
        # Reset early stopping for new episode
        self.early_stopping.reset()

        # Move to device and add batch dimension if needed
        if u0.ndim == 3:
            u0 = u0.unsqueeze(0)  # [1, C, H, W]
        u0 = u0.to(self.device)

        # Storage for trajectory and tokens
        trajectory_list = [] if store_trajectory else [u0[0].cpu()]
        token_list = []

        # Current state
        u_current = u0.clone()
        u_prev = None

        # Timing
        start_time = time.time()

        # Rollout loop
        with torch.no_grad():
            for t in range(max_steps if max_steps is not None else 10000):
                # Apply perturbation if active
                if perturbation.is_active(t):
                    u_current = perturbation.apply(u_current, t)

                # Extract features matching VQ-VAE training distribution
                features = self.feature_extractor.extract(u_current)  # [B, 63]

                # VQ-VAE encoding
                vq_output = self.vqvae(features)
                tokens = self._extract_tokens(vq_output)  # [num_categories * num_levels]

                # Store
                if store_trajectory:
                    trajectory_list.append(u_current[0].cpu())
                token_list.append(tokens.cpu())

                # Check early stopping
                episode_state = EpisodeState(
                    t=t,
                    state=u_current[0],
                    prev_state=u_prev[0] if u_prev is not None else None,
                    tokens=token_list,
                )

                should_stop, stop_reason = self.early_stopping.should_stop(episode_state)
                if should_stop:
                    break

                # MNO forward step: u_{t+1} = MNO(u_t)
                # Note: In autonomous mode, MNO evolves state without θ conditioning
                u_prev = u_current.clone()
                u_current = self._mno_step(u_current)

        # Final time
        elapsed_time = time.time() - start_time

        # Stack trajectory and tokens
        if store_trajectory:
            trajectory = torch.stack(trajectory_list)  # [T, C, H, W]
        else:
            # Store only first and last
            final_state = u_current[0].cpu()
            trajectory = torch.stack([trajectory_list[0], final_state])  # [2, C, H, W]

        token_sequence = torch.stack(token_list)  # [T, num_categories * num_levels]

        # Create episode record
        episode_id = self._generate_episode_id()
        episode = Episode(
            episode_id=episode_id,
            perturbation=perturbation,
            initial_state=u0[0].cpu(),
            trajectory=trajectory,
            token_sequence=token_sequence,
            stop_reason=stop_reason,
            metadata={
                "num_steps": t + 1,
                "elapsed_time": elapsed_time,
                "store_trajectory": store_trajectory,
                "perturbation_metadata": perturbation.get_metadata(),
            },
        )

        return episode

    def _mno_step(self, u: Tensor) -> Tensor:
        """Execute single MNO forward step.

        Args:
            u: Current state [B, C, H, W]

        Returns:
            Next state [B, C, H, W]

        Note:
            In autonomous mode, MNO evolves state without explicit θ conditioning.
            The MNO has learned P(u_{t+1} | u_t) from training, which implicitly
            captures dynamics across the training ensemble.
        """
        with torch.no_grad():
            u_next = self.mno.single_step(u)
        return u_next

    def _extract_tokens(self, vq_output: Dict[str, Any]) -> Tensor:
        """Extract token sequence from VQ-VAE output.

        Args:
            vq_output: VQ-VAE output dictionary

        Returns:
            Token tensor [num_categories * num_levels]
        """
        # VQ-VAE returns tokens in the output dict
        # Format: [B, num_categories * num_levels]
        tokens = vq_output["tokens"]

        # Remove batch dimension (we're processing single states)
        if tokens.ndim == 2:
            tokens = tokens[0]  # [num_categories * num_levels]

        return tokens

    def _generate_episode_id(self) -> str:
        """Generate unique episode ID.

        Returns:
            Episode ID string
        """
        self._episode_counter += 1
        timestamp = int(time.time() * 1000)  # Milliseconds
        return f"ep_{timestamp}_{self._episode_counter:06d}"


class ContinuousExplorationLoop:
    """Continuous mode: Chain episodes together (Phase 4-5).

    Each episode starts from the final state of the previous episode,
    creating a continuous online trajectory perturbed adaptively.

    Example:
        >>> loop = ContinuousExplorationLoop(runner, perturbation_generator)
        >>> episodes = loop.run(u0_initial, n_perturbations=100)
        >>> # Results in 100 chained episodes exploring behavioral manifold
    """

    def __init__(
        self,
        runner: EpisodeRunner,
        perturbation_generator: callable,
    ):
        """Initialize continuous exploration loop.

        Args:
            runner: Episode runner
            perturbation_generator: Function that generates perturbations
                                   Signature: (current_state) -> BasePerturbation
        """
        self.runner = runner
        self.perturbation_generator = perturbation_generator

    def run(
        self,
        u0_initial: Tensor,
        n_perturbations: int,
        on_episode_complete: Optional[callable] = None,
    ) -> List[Episode]:
        """Run continuous exploration from initial state.

        Args:
            u0_initial: Initial state [C, H, W]
            n_perturbations: Number of perturbations to apply
            on_episode_complete: Optional callback after each episode
                                Signature: (episode) -> None

        Returns:
            List of chained episodes
        """
        episodes = []
        current_state = u0_initial

        for i in range(n_perturbations):
            # Generate perturbation based on current state
            perturbation = self.perturbation_generator(current_state)

            # Run episode starting from current state
            episode = self.runner.run_episode(
                u0=current_state, perturbation=perturbation
            )

            # Store episode
            episodes.append(episode)

            # Callback
            if on_episode_complete is not None:
                on_episode_complete(episode)

            # Update current state to final state of this episode
            current_state = episode.final_state()

            print(
                f"Episode {i+1}/{n_perturbations}: "
                f"{episode.num_steps()} steps, "
                f"stopped by {episode.stop_reason}"
            )

        return episodes


class ExploratoryLoop:
    """Exploratory mode: Independent episodes (Phase 2-3).

    Each episode starts from a fresh sampled initial condition,
    providing broad coverage of state space for validation and
    memory building.

    Example:
        >>> loop = ExploratoryLoop(runner, ic_sampler, perturbation_sampler)
        >>> episodes = loop.run(n_episodes=1000)
        >>> # Results in 1000 independent episodes across diverse states
    """

    def __init__(
        self,
        runner: EpisodeRunner,
        ic_sampler: callable,
        perturbation_sampler: callable,
    ):
        """Initialize exploratory loop.

        Args:
            runner: Episode runner
            ic_sampler: Function that samples initial conditions
                       Signature: () -> Tensor [C, H, W]
            perturbation_sampler: Function that samples perturbations
                                 Signature: () -> BasePerturbation
        """
        self.runner = runner
        self.ic_sampler = ic_sampler
        self.perturbation_sampler = perturbation_sampler

    def run(
        self,
        n_episodes: int,
        on_episode_complete: Optional[callable] = None,
    ) -> List[Episode]:
        """Run exploratory episodes.

        Args:
            n_episodes: Number of independent episodes
            on_episode_complete: Optional callback after each episode

        Returns:
            List of independent episodes
        """
        episodes = []

        for i in range(n_episodes):
            # Sample fresh initial condition
            u0 = self.ic_sampler()

            # Sample perturbation
            perturbation = self.perturbation_sampler()

            # Run episode
            episode = self.runner.run_episode(u0=u0, perturbation=perturbation)

            # Store episode
            episodes.append(episode)

            # Callback
            if on_episode_complete is not None:
                on_episode_complete(episode)

            print(
                f"Episode {i+1}/{n_episodes}: "
                f"{episode.num_steps()} steps, "
                f"stopped by {episode.stop_reason}"
            )

        return episodes
