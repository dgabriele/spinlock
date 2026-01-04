"""
SimpleCNNOperator: CNN operator with intermediate feature extraction support.

This class wraps CNN layers in a way that allows extracting intermediate
activations for learned feature extraction, mirroring the U-AFNO interface.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Literal


class SimpleCNNOperator(nn.Module):
    """
    CNN operator with intermediate feature extraction support.

    Replaces nn.Sequential for CNN operators, providing:
    - Standard forward() for normal inference
    - get_intermediate_features() for learned feature extraction

    Architecture (typical):
        Input → ConvBlock(early) → ResidualBlocks(mid_N) → StochasticBlock → OutputLayer → Output

    Extractable features:
        - "early": After first ConvBlock (local edges/gradients)
        - "mid_N": After each subsequent block (progressively abstract)
        - "pre_output": Before final OutputLayer (integrated representation)

    Example:
        ```python
        # Built by OperatorBuilder.build_simple_cnn()
        operator = SimpleCNNOperator(
            conv_blocks=[conv1, res1, res2, res3],
            stochastic_block=stochastic,
            output_layer=output
        )

        # Normal forward pass
        out = operator(x)

        # Extract intermediate features
        features = operator.get_intermediate_features(x, extract_from="all")
        # Returns: {"early": tensor, "mid_1": tensor, "mid_2": tensor, "pre_output": tensor}
        ```
    """

    def __init__(
        self,
        conv_blocks: nn.ModuleList,
        stochastic_block: Optional[nn.Module] = None,
        output_layer: Optional[nn.Module] = None,
    ):
        """
        Initialize SimpleCNNOperator.

        Args:
            conv_blocks: List of convolutional blocks (ConvBlock, ResidualBlock)
            stochastic_block: Optional stochastic noise injection block
            output_layer: Final output projection layer
        """
        super().__init__()
        self.conv_blocks = conv_blocks
        self.stochastic_block = stochastic_block
        self.output_layer = output_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C_out, H, W]
        """
        for block in self.conv_blocks:
            x = block(x)

        if self.stochastic_block is not None:
            x = self.stochastic_block(x)

        if self.output_layer is not None:
            x = self.output_layer(x)

        return x

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        extract_from: Literal["early", "mid", "pre_output", "all"] = "all",
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate activations from conv blocks.

        This method runs forward pass while capturing intermediate activations
        at specified extraction points. Used for learned feature extraction.

        Args:
            x: Input tensor [B, C, H, W]
            extract_from: Which features to extract:
                - "early": First conv block only (local features)
                - "mid": Middle blocks only (mid-level patterns)
                - "pre_output": Final hidden state before output layer
                - "all": All of the above (default)
            layer_indices: Optional specific block indices to extract.
                           If provided, overrides extract_from for mid blocks.

        Returns:
            Dict mapping feature names to tensors:
                - "early": [B, C_early, H, W] - after first conv
                - "mid_1", "mid_2", ...: [B, C_mid, H, W] - after each subsequent block
                - "pre_output": [B, C_pre, H, W] - before output layer

        Example:
            ```python
            features = operator.get_intermediate_features(x, extract_from="all")

            # Access specific features
            early = features["early"]      # Local edge/gradient features
            mid_2 = features["mid_2"]      # Mid-level abstract patterns
            pre_out = features["pre_output"]  # Integrated representation
            ```
        """
        features: Dict[str, torch.Tensor] = {}
        extract_early = extract_from in ("early", "all")
        extract_mid = extract_from in ("mid", "all")
        extract_pre_output = extract_from in ("pre_output", "all")

        # Process conv blocks and capture activations
        for i, block in enumerate(self.conv_blocks):
            x = block(x)

            if i == 0 and extract_early:
                features["early"] = x
            elif i > 0 and extract_mid:
                # Check layer_indices filter if provided
                if layer_indices is None or i in layer_indices:
                    features[f"mid_{i}"] = x

        # Capture pre-output (before stochastic and output layers)
        if extract_pre_output:
            features["pre_output"] = x

        # Continue forward for proper output (but don't store)
        if self.stochastic_block is not None:
            x = self.stochastic_block(x)

        if self.output_layer is not None:
            x = self.output_layer(x)

        return features

    def get_feature_channels(self) -> Dict[str, int]:
        """
        Get the number of channels at each extraction point.

        Useful for configuring downstream feature aggregation layers.

        Returns:
            Dict mapping feature names to channel counts
        """
        channels: Dict[str, int] = {}

        for i, block in enumerate(self.conv_blocks):
            # Get output channels from block
            if hasattr(block, "out_channels"):
                out_ch = block.out_channels
            elif hasattr(block, "conv") and hasattr(block.conv, "out_channels"):
                out_ch = block.conv.out_channels
            else:
                # Fallback: inspect last conv layer
                for module in block.modules():
                    if isinstance(module, nn.Conv2d):
                        out_ch = module.out_channels

            if i == 0:
                channels["early"] = out_ch
            else:
                channels[f"mid_{i}"] = out_ch

        # pre_output has same channels as last conv block
        if self.conv_blocks:
            channels["pre_output"] = channels.get(f"mid_{len(self.conv_blocks)-1}",
                                                   channels.get("early", 0))

        return channels

    def __repr__(self) -> str:
        num_blocks = len(self.conv_blocks)
        has_stochastic = self.stochastic_block is not None
        has_output = self.output_layer is not None
        return (
            f"SimpleCNNOperator("
            f"conv_blocks={num_blocks}, "
            f"stochastic={has_stochastic}, "
            f"output={has_output})"
        )
