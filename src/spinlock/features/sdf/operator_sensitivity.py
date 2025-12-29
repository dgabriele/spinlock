"""
Operator Sensitivity Feature Extractor.

Extracts features characterizing how neural operators respond to input perturbations:
- Lipschitz estimates (local sensitivity to noise)
- Gain curves (response to amplitude scaling)
- Linearity metrics (deviation from linear behavior)

CRITICAL: This extractor requires access to the operator during extraction.
It must be called during dataset generation when operators are in memory.

Shape conventions:
- Input field: [C, H, W] or [N, C, H, W]
- Output features: Dict[str, torch.Tensor] with shape [] (scalars)

Author: Claude (Anthropic)
Date: December 2025
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class OperatorSensitivityExtractor:
    """
    Extract operator sensitivity features by re-executing operators with perturbed inputs.

    Characterizes operator response to:
    1. Small random perturbations (Lipschitz estimates)
    2. Input amplitude scaling (gain curves)
    3. Nonlinearity and saturation behavior

    This extractor is expensive (requires multiple forward passes) but provides
    unique insight into operator properties.

    Attributes:
        device: Torch device (CPU or CUDA)
        lipschitz_epsilon_scales: Perturbation magnitudes for Lipschitz estimation
        gain_scale_factors: Amplitude scaling factors for gain curves
    """

    def __init__(
        self,
        device: torch.device,
        lipschitz_epsilon_scales: Optional[List[float]] = None,
        gain_scale_factors: Optional[List[float]] = None,
    ):
        """
        Initialize operator sensitivity extractor.

        Args:
            device: Torch device
            lipschitz_epsilon_scales: Perturbation scales (default: [1e-4, 1e-3, 1e-2])
            gain_scale_factors: Amplitude scales (default: [0.5, 0.75, 1.25, 1.5])
        """
        self.device = device

        # Default Lipschitz perturbation scales
        if lipschitz_epsilon_scales is None:
            lipschitz_epsilon_scales = [1e-4, 1e-3, 1e-2]
        self.lipschitz_epsilon_scales = lipschitz_epsilon_scales

        # Default gain curve scaling factors
        if gain_scale_factors is None:
            gain_scale_factors = [0.5, 0.75, 1.25, 1.5]
        self.gain_scale_factors = gain_scale_factors

    def extract(
        self,
        operator: nn.Module,
        input_field: torch.Tensor,
        config: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract operator sensitivity features.

        Args:
            operator: Neural operator (nn.Module) in eval mode
            input_field: Input field [C, H, W] or [N, C, H, W]
            config: SDFOperatorSensitivityConfig (optional)

        Returns:
            Dictionary of feature name → scalar tensor

        Edge cases:
            - If operator raises error during forward pass, returns NaN features
            - If input_field is batched [N, C, H, W], uses first sample [0]
        """
        # Handle batched input (use first sample)
        if input_field.ndim == 4:
            input_field = input_field[0]  # [C, H, W]

        # Ensure input is on correct device
        input_field = input_field.to(self.device)

        # Set operator to eval mode
        operator.eval()

        features = {}

        # 1. Lipschitz estimates (if enabled in config)
        if config is None or config.include_lipschitz:
            try:
                lipschitz_features = self._compute_lipschitz_estimates_batched(
                    operator, input_field, config
                )
                features.update(lipschitz_features)
            except Exception:
                # On failure, return NaN for all Lipschitz features
                for eps in self.lipschitz_epsilon_scales:
                    features[f"lipschitz_eps_{eps:.0e}"] = torch.tensor(
                        float("nan"), device=self.device
                    )

        # 2. Gain curves (if enabled in config)
        if config is None or config.include_gain_curve:
            try:
                gain_features = self._compute_gain_curve(operator, input_field, config)
                features.update(gain_features)
            except Exception:
                # On failure, return NaN for all gain features
                for scale in self.gain_scale_factors:
                    features[f"gain_scale_{scale:.2f}"] = torch.tensor(
                        float("nan"), device=self.device
                    )

        # 3. Linearity metrics (if enabled in config)
        if config is None or config.include_linearity_metrics:
            try:
                linearity_features = self._compute_linearity_metrics(
                    operator, input_field, config
                )
                features.update(linearity_features)
            except Exception:
                # On failure, return NaN for linearity metrics
                features["linearity_r2"] = torch.tensor(float("nan"), device=self.device)
                features["saturation_degree"] = torch.tensor(
                    float("nan"), device=self.device
                )
                features["compression_ratio"] = torch.tensor(
                    float("nan"), device=self.device
                )

        return features

    def _compute_lipschitz_estimates_batched(
        self,
        operator: nn.Module,
        input_field: torch.Tensor,
        config: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Lipschitz constant estimates for multiple perturbation scales.

        Lipschitz constant L approximated as:
            L ≈ ||O(x + δ) - O(x)|| / ||δ||

        where δ is Gaussian noise with std ε.

        Args:
            operator: Neural operator
            input_field: [C, H, W]
            config: Optional configuration

        Returns:
            Dictionary: {f"lipschitz_eps_{eps:.0e}": L_estimate}
        """
        C, H, W = input_field.shape

        # Get epsilon scales from config or use defaults
        if config is not None and hasattr(config, "lipschitz_epsilon_scales"):
            epsilon_scales = config.lipschitz_epsilon_scales
        else:
            epsilon_scales = self.lipschitz_epsilon_scales

        # Compute baseline output O(x)
        with torch.no_grad():
            baseline_output = operator(input_field.unsqueeze(0))  # [1, C, H, W]
            if baseline_output.ndim == 4 and baseline_output.shape[0] == 1:
                baseline_output = baseline_output[0]  # [C, H, W]

        features = {}

        # Batch perturbations for efficiency
        num_perturbations = len(epsilon_scales)

        # Create batch of perturbed inputs [num_eps, C, H, W]
        perturbations = []
        deltas = []
        for eps in epsilon_scales:
            delta = torch.randn_like(input_field) * eps
            perturbed_input = input_field + delta
            perturbations.append(perturbed_input)
            deltas.append(delta)

        perturbations_batch = torch.stack(perturbations, dim=0)  # [num_eps, C, H, W]
        deltas_batch = torch.stack(deltas, dim=0)  # [num_eps, C, H, W]

        # Single batched forward pass through operator
        with torch.no_grad():
            perturbed_outputs = operator(perturbations_batch)  # [num_eps, C, H, W]

        # Compute Lipschitz estimates
        for i, eps in enumerate(epsilon_scales):
            output_diff = perturbed_outputs[i] - baseline_output  # [C, H, W]
            delta_norm = torch.norm(deltas_batch[i])
            output_diff_norm = torch.norm(output_diff)

            # Lipschitz constant estimate
            lipschitz_est = output_diff_norm / (delta_norm + 1e-10)

            features[f"lipschitz_eps_{eps:.0e}"] = lipschitz_est

        return features

    def _compute_gain_curve(
        self,
        operator: nn.Module,
        input_field: torch.Tensor,
        config: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gain curve: response to input amplitude scaling.

        Gain(α) = ||O(α·x)|| / ||O(x)||

        Reveals operator behavior under amplitude variation:
        - Linear operator: Gain(α) = α
        - Compressive nonlinearity: Gain(α) < α for α > 1
        - Expansive nonlinearity: Gain(α) > α for α > 1

        Args:
            operator: Neural operator
            input_field: [C, H, W]
            config: Optional configuration

        Returns:
            Dictionary: {f"gain_scale_{scale:.2f}": gain}
        """
        # Get scale factors from config or use defaults
        if config is not None and hasattr(config, "gain_scale_factors"):
            scale_factors = config.gain_scale_factors
        else:
            scale_factors = self.gain_scale_factors

        # Compute baseline output energy ||O(x)||
        with torch.no_grad():
            baseline_output = operator(input_field.unsqueeze(0))  # [1, C, H, W]
            if baseline_output.ndim == 4 and baseline_output.shape[0] == 1:
                baseline_output = baseline_output[0]  # [C, H, W]
            baseline_energy = torch.norm(baseline_output)

        features = {}

        # Batch scaled inputs for efficiency
        scaled_inputs = torch.stack(
            [input_field * scale for scale in scale_factors], dim=0
        )  # [num_scales, C, H, W]

        # Single batched forward pass
        with torch.no_grad():
            scaled_outputs = operator(scaled_inputs)  # [num_scales, C, H, W]

        # Compute gain for each scale
        for i, scale in enumerate(scale_factors):
            output_energy = torch.norm(scaled_outputs[i])
            gain = output_energy / (baseline_energy + 1e-10)

            features[f"gain_scale_{scale:.2f}"] = gain

        return features

    def _compute_linearity_metrics(
        self,
        operator: nn.Module,
        input_field: torch.Tensor,
        config: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute linearity metrics based on gain curve analysis.

        Metrics:
        - linearity_r2: R² of linear fit to gain curve (1.0 = perfectly linear)
        - saturation_degree: Gain decrease at high amplitudes (compression)
        - compression_ratio: gain(0.5) / gain(1.5) (>1 = compressive, <1 = expansive)

        Args:
            operator: Neural operator
            input_field: [C, H, W]
            config: Optional configuration

        Returns:
            Dictionary with 3 linearity metrics
        """
        # Recompute gain curve (could cache from _compute_gain_curve, but keep clean)
        if config is not None and hasattr(config, "gain_scale_factors"):
            scale_factors = config.gain_scale_factors
        else:
            scale_factors = self.gain_scale_factors

        # Compute baseline output
        with torch.no_grad():
            baseline_output = operator(input_field.unsqueeze(0))
            if baseline_output.ndim == 4 and baseline_output.shape[0] == 1:
                baseline_output = baseline_output[0]
            baseline_energy = torch.norm(baseline_output)

        # Compute gains
        scaled_inputs = torch.stack(
            [input_field * scale for scale in scale_factors], dim=0
        )
        with torch.no_grad():
            scaled_outputs = operator(scaled_inputs)

        gains = []
        for i in range(len(scale_factors)):
            output_energy = torch.norm(scaled_outputs[i])
            gain = output_energy / (baseline_energy + 1e-10)
            gains.append(gain.item())

        # Convert to numpy for linear regression
        import numpy as np

        scales = np.array(scale_factors)
        gains = np.array(gains)

        # 1. Linearity R²: fit linear model gain = a * scale + b
        # For ideal linear operator: a=1, b=0
        A = np.vstack([scales, np.ones(len(scales))]).T
        a, b = np.linalg.lstsq(A, gains, rcond=None)[0]

        # Compute R²
        ss_res = np.sum((gains - (a * scales + b)) ** 2)
        ss_tot = np.sum((gains - np.mean(gains)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        # 2. Saturation degree: gain decrease at high amplitudes
        # Compare gain at max scale vs expected linear gain
        max_scale_idx = np.argmax(scales)
        expected_gain = scales[max_scale_idx]
        actual_gain = gains[max_scale_idx]
        saturation_degree = expected_gain - actual_gain  # Positive = compression

        # 3. Compression ratio: gain(low) / gain(high)
        # Find gains closest to 0.5 and 1.5
        idx_low = np.argmin(np.abs(scales - 0.5))
        idx_high = np.argmin(np.abs(scales - 1.5))
        compression_ratio = gains[idx_low] / (gains[idx_high] + 1e-10)

        features = {
            "linearity_r2": torch.tensor(r2, device=self.device),
            "saturation_degree": torch.tensor(saturation_degree, device=self.device),
            "compression_ratio": torch.tensor(compression_ratio, device=self.device),
        }

        return features
