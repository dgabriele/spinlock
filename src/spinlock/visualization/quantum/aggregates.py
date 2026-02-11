"""Quantum aggregate renderers for observable overlays."""
import torch
import numpy as np
from typing import Optional, Literal


class QuantumObservableOverlay:
    """Overlay quantum observables as text on frames.

    Displays time-series values of purity, entropy, coherence as
    evolving text overlays on rendered frames.

    Uses PIL for text rendering to maintain compatibility with
    existing video export pipeline.
    """

    def __init__(
        self,
        observable: Literal["purity", "entropy", "coherence_mean", "coherence_max", "uncertainty_product"] = "purity",
        display_mode: Literal["text", "none"] = "text",
        position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "top_left",
    ):
        """Initialize overlay renderer.

        Args:
            observable: Which quantum observable to display
                - "purity": Tr(ρ²) - measure of state mixedness
                - "entropy": von Neumann entropy S = -Tr(ρ ln ρ)
                - "coherence_mean": Mean off-diagonal coherence
                - "coherence_max": Maximum coherence
                - "uncertainty_product": Δx Δp uncertainty
            display_mode: How to render
                - "text": Text overlay on frame
                - "none": No overlay (pass through)
            position: Where to place text overlay
        """
        self.observable = observable
        self.display_mode = display_mode
        self.position = position

        # Map observable names to feature indices in quantum features tensor
        # Based on quantum features starting at index 178 in temporal features
        # Order: purity, entropy, coherence_mean, coherence_max, ...
        self._feature_index_map = {
            'purity': 0,
            'entropy': 1,
            'coherence_mean': 2,
            'coherence_max': 3,
            'uncertainty_product': 7,  # Δx Δp at index 7
        }

    def extract_observable(
        self,
        quantum_features: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Extract observable values at a given timestep.

        Args:
            quantum_features: Quantum features [N, T, D_quantum]
            timestep: Timestep index to extract

        Returns:
            Observable values [N] at the given timestep
        """
        feat_idx = self._feature_index_map[self.observable]
        return quantum_features[:, timestep, feat_idx]  # [N]

    def render_overlay(
        self,
        base_frame: torch.Tensor,
        observable_values: torch.Tensor,
        rollout_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render observable as overlay on base frame.

        Args:
            base_frame: RGB frame [B, H, W, 3] in range [0, 1]
            observable_values: Observable at current time [B]
            rollout_indices: Optional rollout IDs for labeling

        Returns:
            Frame with overlay [B, H, W, 3]
        """
        if self.display_mode == "none":
            return base_frame

        # Convert to uint8 for PIL
        frame_np = (base_frame.cpu().numpy() * 255).astype(np.uint8)

        # Add text overlay using PIL
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            # Fallback: return without overlay if PIL not available
            return base_frame

        overlaid_frames = []
        for i in range(frame_np.shape[0]):
            # Create PIL image
            img = Image.fromarray(frame_np[i])
            draw = ImageDraw.Draw(img)

            # Format text
            value = observable_values[i].item()
            if rollout_indices is not None:
                text = f"#{rollout_indices[i].item()} {self.observable}: {value:.4f}"
            else:
                text = f"{self.observable}: {value:.4f}"

            # Position mapping
            H, W = img.size[1], img.size[0]
            pos_map = {
                'top_left': (10, 10),
                'top_right': (W - 200, 10),
                'bottom_left': (10, H - 30),
                'bottom_right': (W - 200, H - 30),
            }
            position = pos_map[self.position]

            # Try to use a larger font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Add text with outline for visibility
            outline_color = (0, 0, 0)  # Black outline
            text_color = (255, 255, 255)  # White text

            # Draw outline
            for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text(
                    (position[0] + offset[0], position[1] + offset[1]),
                    text,
                    fill=outline_color,
                    font=font
                )

            # Draw main text
            draw.text(position, text, fill=text_color, font=font)

            # Convert back to numpy
            overlaid_frames.append(np.array(img))

        # Stack and convert back to tensor
        overlaid = np.stack(overlaid_frames, axis=0)
        return torch.from_numpy(overlaid).float() / 255.0


class QuantumAggregateRenderer:
    """Compute and render aggregate statistics across multiple rollouts.

    Extends the concept of AggregateRenderer to quantum observables,
    computing mean, std, min, max across rollouts at each timestep.
    """

    def __init__(
        self,
        observable: str = "purity",
        aggregate_fn: Literal["mean", "std", "min", "max"] = "mean",
    ):
        """Initialize aggregate renderer.

        Args:
            observable: Which quantum observable to aggregate
            aggregate_fn: Aggregation function to apply
        """
        self.observable = observable
        self.aggregate_fn = aggregate_fn

        self._feature_index_map = {
            'purity': 0,
            'entropy': 1,
            'coherence_mean': 2,
            'coherence_max': 3,
            'uncertainty_product': 7,
        }

    def compute(
        self,
        quantum_features: torch.Tensor,  # [N, T, D_quantum]
    ) -> torch.Tensor:
        """Compute aggregate observable over rollouts.

        Args:
            quantum_features: Quantum features [N, T, D_quantum]

        Returns:
            Aggregated observable [T] - one value per timestep
        """
        feat_idx = self._feature_index_map[self.observable]
        obs_values = quantum_features[:, :, feat_idx]  # [N, T]

        # Apply aggregation along rollout dimension
        if self.aggregate_fn == "mean":
            return obs_values.mean(dim=0)  # [T]
        elif self.aggregate_fn == "std":
            return obs_values.std(dim=0)
        elif self.aggregate_fn == "min":
            return obs_values.min(dim=0).values
        elif self.aggregate_fn == "max":
            return obs_values.max(dim=0).values
        else:
            raise ValueError(f"Unknown aggregate function: {self.aggregate_fn}")
