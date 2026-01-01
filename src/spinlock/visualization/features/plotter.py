"""
Matplotlib-based feature plotting with SVG export.

Creates compact line charts showing mean ± std envelopes across
realizations for time series features.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Tuple, Optional


class FeatureLinePlotter:
    """
    Matplotlib plotter for time series features.

    Creates tall SVG visualizations with:
    - One line chart per feature (stacked vertically)
    - Mean line + shaded ±1 std envelope across realizations
    - Category headers grouping related features
    - Clean, publication-ready styling

    Example:
        >>> plotter = FeatureLinePlotter(figsize_per_feature=(2.56, 0.6), dpi=100)
        >>> features_by_category = {
        ...     'spatial': {'mean': values1, 'std': values2},
        ...     'spectral': {'fft_power': values3}
        ... }
        >>> plotter.create_tall_svg(features_by_category, timesteps, output_path)
    """

    def __init__(
        self,
        figsize_per_feature: Tuple[float, float] = (2.56, 0.6),
        dpi: int = 100,
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """
        Initialize feature line plotter.

        Args:
            figsize_per_feature: (width, height) in inches for each feature plot
                Default (2.56, 0.6) = 256px × 60px @ 100 DPI
            dpi: Dots per inch for rasterization
            style: Matplotlib style (default: clean seaborn grid)
        """
        self.figsize_per_feature = figsize_per_feature
        self.dpi = dpi
        self.style = style

        # Color scheme (matplotlib default blue)
        self.line_color = '#1f77b4'
        self.fill_color = '#1f77b4'
        self.fill_alpha = 0.3

        # Font sizes
        self.feature_fontsize = 9
        self.category_fontsize = 10
        self.tick_fontsize = 8

    def plot_feature(
        self,
        timesteps: np.ndarray,
        values: np.ndarray,
        feature_name: str,
        ax: plt.Axes
    ) -> None:
        """
        Plot single feature with mean ± std envelope.

        Args:
            timesteps: Time points [T]
            values: Feature values [M, T] across M realizations
            feature_name: Name for y-axis label
            ax: Matplotlib axes to plot on

        Note:
            If values is 1D [T], plots it directly without envelope.
            If values is 2D [M, T], computes mean ± std across M.
        """
        # Handle both 1D [T] and 2D [M, T] inputs
        if values.ndim == 1:
            # Single realization: plot directly
            mean = values
            std = None
        elif values.ndim == 2:
            # Multiple realizations: compute mean ± std
            mean = np.mean(values, axis=0)  # [T]
            std = np.std(values, axis=0)    # [T]
        else:
            raise ValueError(f"Expected values shape [T] or [M, T], got {values.shape}")

        # Plot mean line
        ax.plot(timesteps, mean, color=self.line_color, linewidth=1.5, zorder=2)

        # Plot ±1 std envelope if available
        if std is not None:
            ax.fill_between(
                timesteps,
                mean - std,
                mean + std,
                color=self.fill_color,
                alpha=self.fill_alpha,
                zorder=1
            )

        # Styling
        ax.set_xlim(timesteps[0], timesteps[-1])

        # Y-axis label (feature name) on left, horizontal
        ax.set_ylabel(
            feature_name,
            fontsize=self.feature_fontsize,
            rotation=0,
            ha='right',
            va='center',
            labelpad=10
        )

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Tick styling
        ax.tick_params(labelsize=self.tick_fontsize)

        # Only show x-axis label on bottom plot (handled by caller)

    def create_tall_svg(
        self,
        features_by_category: Dict[str, Dict[str, np.ndarray]],
        timesteps: np.ndarray,
        output_path: Path,
        operator_idx: Optional[int] = None,
        operator_params: Optional[Dict] = None
    ) -> None:
        """
        Create tall SVG with all features grouped by category.

        Args:
            features_by_category: Nested dict {category: {feature_name: values}}
                - values shape: [M, T] for multiple realizations or [T] for single
            timesteps: Time points [T]
            output_path: Path to save SVG file
            operator_idx: Optional operator index for title
            operator_params: Optional parameter dict for title metadata

        Layout:
            - Categories appear in order (spatial, spectral, ...)
            - Within category: features in alphabetical order
            - Category headers: Bold text separators
            - Feature plots: Stacked line charts
            - Bottom plot gets x-axis label

        Example:
            >>> features_by_category = {
            ...     'spatial': {
            ...         'mean': np.random.rand(3, 250),  # 3 realizations, 250 timesteps
            ...         'std': np.random.rand(3, 250)
            ...     },
            ...     'spectral': {
            ...         'fft_power': np.random.rand(3, 250)
            ...     }
            ... }
            >>> timesteps = np.arange(250)
            >>> plotter.create_tall_svg(features_by_category, timesteps, Path("output.svg"))
        """
        # Apply style
        plt.style.use(self.style)

        # Count total features
        total_features = sum(len(feats) for feats in features_by_category.values())
        num_categories = len(features_by_category)

        if total_features == 0:
            raise ValueError("No features to plot")

        # Calculate figure height
        feature_height = self.figsize_per_feature[1]
        category_header_height = 0.3  # inches
        margin_top_bottom = 0.5  # inches
        total_height = (total_features * feature_height +
                       num_categories * category_header_height +
                       margin_top_bottom)

        # Create figure
        fig = plt.figure(
            figsize=(self.figsize_per_feature[0], total_height),
            dpi=self.dpi
        )

        # Add title if operator info provided
        if operator_idx is not None:
            title = f"Operator {operator_idx} - Time Series Features"
            if operator_params:
                # Add key parameters to title
                param_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in list(operator_params.items())[:3])
                title += f"\n{param_str}"
            fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)

        # Create grid layout
        # One row for each category header + feature
        total_rows = num_categories + total_features
        gs = gridspec.GridSpec(
            nrows=total_rows,
            ncols=1,
            hspace=0.4,
            left=0.25,    # Space for feature names on left
            right=0.95,
            top=0.98,
            bottom=0.02
        )

        row_idx = 0
        all_axes = []

        for category, features in features_by_category.items():
            if len(features) == 0:
                continue

            # Category header
            ax_header = fig.add_subplot(gs[row_idx, 0])
            ax_header.text(
                0.0, 0.5,
                category.upper().replace('_', ' '),
                fontsize=self.category_fontsize,
                fontweight='bold',
                va='center'
            )
            ax_header.axis('off')
            row_idx += 1

            # Feature plots (alphabetical order)
            for feature_name in sorted(features.keys()):
                values = features[feature_name]

                ax = fig.add_subplot(gs[row_idx, 0])
                self.plot_feature(timesteps, values, feature_name, ax)
                all_axes.append(ax)
                row_idx += 1

        # Add x-axis label only to bottom plot
        if all_axes:
            all_axes[-1].set_xlabel('Timestep', fontsize=self.feature_fontsize)

        # Save as SVG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close(fig)

    def create_comparison_svg(
        self,
        features_list: list[Dict[str, Dict[str, np.ndarray]]],
        timesteps: np.ndarray,
        output_path: Path,
        labels: Optional[list[str]] = None
    ) -> None:
        """
        Create comparison SVG showing multiple operators side-by-side.

        Args:
            features_list: List of feature dicts (one per operator)
            timesteps: Time points [T]
            output_path: Path to save SVG file
            labels: Optional labels for each operator

        Note:
            This creates a wider SVG with operators in columns.
            Useful for comparing diverse operators.
        """
        if not features_list:
            raise ValueError("Empty features list")

        n_operators = len(features_list)
        if labels is None:
            labels = [f"Op {i}" for i in range(n_operators)]

        # For comparison, create multiple columns
        # Each column is one operator's features
        # Implementation simplified: create separate SVGs and combine externally
        # (Full implementation would use GridSpec with multiple columns)

        raise NotImplementedError(
            "Comparison plotting not yet implemented. "
            "Use create_tall_svg() for individual operators."
        )
