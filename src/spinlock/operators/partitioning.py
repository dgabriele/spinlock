"""
Architecture Partitioning for CUDA Optimization.

Groups operators by architecture signature (graph-defining parameters),
enabling torch.compile kernel reuse across operators in the same partition.

Key insight: Operators with same (num_layers, base_channels_bucket, kernel_size)
have identical computational graphs - only weights differ. torch.compile can
generate one optimized kernel per partition and reuse it for all operators.

Expected partitions: ~32 (4 num_layers × 4 channel buckets × 2 kernel sizes)
Expected speedup: 1.3-1.5× from compile kernel reuse

Author: Claude (Anthropic)
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import torch.nn as nn

from spinlock.operators.parameters import OperatorParameters


@dataclass
class ArchitecturePartition:
    """
    A partition of operators sharing the same architecture signature.

    All operators in a partition have identical computational graphs,
    differing only in their weight values. This allows a single compiled
    kernel to be reused for all operators in the partition.

    Attributes:
        signature: Tuple of graph-defining parameters (num_layers, channels_bucket, kernel_size)
        template: Uncompiled template operator (for weight loading)
        compiled_template: torch.compile'd version for fast inference
        operator_indices: List of operator indices in this partition
        state_dicts: Cached state_dicts for batched execution (Phase 2)
    """

    signature: Tuple
    template: nn.Module
    compiled_template: Optional[nn.Module] = None
    operator_indices: List[int] = field(default_factory=list)
    state_dicts: List[Dict[str, torch.Tensor]] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of operators in this partition."""
        return len(self.operator_indices)

    def __repr__(self) -> str:
        return (
            f"ArchitecturePartition(signature={self.signature}, "
            f"size={self.size}, compiled={self.compiled_template is not None})"
        )


def bucket_channels(channels: int, bucket_size: int = 16) -> int:
    """
    Bucket channel count to reduce partition count.

    Maps continuous channel values to discrete buckets:
    - [16, 31] → 16
    - [32, 47] → 32
    - [48, 63] → 48
    - [64, ...] → 64

    Args:
        channels: Actual channel count
        bucket_size: Size of each bucket (default: 16)

    Returns:
        Bucketed channel value
    """
    # Round down to nearest bucket
    bucket = (channels // bucket_size) * bucket_size
    # Ensure minimum of bucket_size
    return max(bucket, bucket_size)


def get_architecture_signature(
    params: Union[Dict[str, Any], OperatorParameters],
    bucket_size: int = 0,
) -> Tuple:
    """
    Extract graph-defining parameters as hashable signature.

    Only includes parameters that affect the computational graph structure:
    - num_layers: Changes number of operations
    - base_channels: Changes tensor shapes (MUST be exact for weight loading)
    - kernel_size: Changes convolution kernel dimensions

    Parameters that DON'T affect the graph (excluded):
    - noise_scale, noise_type: Same ops, different values
    - dropout_rate: Same ops, different probability
    - activation, normalization: Usually fixed to single choice

    Args:
        params: Operator parameters (dict or OperatorParameters)
        bucket_size: Deprecated - channels MUST be exact for weight compatibility

    Returns:
        Tuple of (num_layers, base_channels, kernel_size)
    """
    # Handle both dict and OperatorParameters
    if isinstance(params, dict):
        num_layers = params.get("num_layers", 3)
        base_channels = params.get("base_channels", 32)
        kernel_size = params.get("kernel_size", 3)
    else:
        num_layers = getattr(params, "num_layers", 3)
        base_channels = getattr(params, "base_channels", 32)
        kernel_size = getattr(params, "kernel_size", 3)

    # NOTE: base_channels MUST be exact (not bucketed) for weight loading compatibility
    # Operators in same partition must have identical tensor shapes
    return (
        num_layers,
        base_channels,
        kernel_size,
    )


def partition_operators(
    all_params: List[Union[Dict[str, Any], OperatorParameters]],
    builder,  # OperatorBuilder - avoid circular import
    device: torch.device,
    bucket_size: int = 16,
    compile_mode: str = "reduce-overhead",
    enable_compile: bool = True,
) -> Dict[Tuple, ArchitecturePartition]:
    """
    Partition operators by architecture signature and optionally compile templates.

    Groups operators with identical graph structure, builds one template per group,
    and optionally compiles each template for optimized inference.

    Args:
        all_params: List of operator parameters
        builder: OperatorBuilder instance
        device: Target device (cuda/cpu)
        bucket_size: Channel bucketing granularity
        compile_mode: torch.compile mode ("reduce-overhead", "max-autotune", "default")
        enable_compile: Whether to compile templates (disable for debugging)

    Returns:
        Dict mapping signature tuples to ArchitecturePartition objects

    Example:
        >>> from spinlock.operators.builder import OperatorBuilder
        >>> builder = OperatorBuilder()
        >>> partitions = partition_operators(all_params, builder, torch.device('cuda'))
        >>> print(f"Created {len(partitions)} partitions")
        Created 12 partitions
    """
    partitions: Dict[Tuple, ArchitecturePartition] = {}

    for idx, params in enumerate(all_params):
        sig = get_architecture_signature(params, bucket_size)

        if sig not in partitions:
            # Build template operator for this architecture
            template = builder.build_simple_cnn(params)
            template = template.to(device)
            template.eval()

            # Disable gradient tracking for inference
            for param in template.parameters():
                param.requires_grad = False

            # Optionally compile the template
            compiled = None
            if enable_compile and device.type == "cuda":
                try:
                    compiled = torch.compile(
                        template,
                        mode=compile_mode,
                        fullgraph=False,  # Allow partial graphs
                        dynamic=False,  # Static shapes for better optimization
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to compile partition {sig}: {e}")
                    compiled = template  # Fallback to uncompiled

            partitions[sig] = ArchitecturePartition(
                signature=sig,
                template=template,
                compiled_template=compiled if compiled is not None else template,
                operator_indices=[idx],
            )
        else:
            partitions[sig].operator_indices.append(idx)

    return partitions


def get_partition_stats(partitions: Dict[Tuple, ArchitecturePartition]) -> Dict[str, Any]:
    """
    Get statistics about partition distribution.

    Args:
        partitions: Dict of partitions from partition_operators()

    Returns:
        Dict with partition statistics
    """
    sizes = [p.size for p in partitions.values()]
    total_ops = sum(sizes)

    return {
        "num_partitions": len(partitions),
        "total_operators": total_ops,
        "min_partition_size": min(sizes) if sizes else 0,
        "max_partition_size": max(sizes) if sizes else 0,
        "avg_partition_size": total_ops / len(partitions) if partitions else 0,
        "signatures": list(partitions.keys()),
    }


def print_partition_summary(partitions: Dict[Tuple, ArchitecturePartition]) -> None:
    """Print a summary of partition distribution."""
    stats = get_partition_stats(partitions)

    print("\n" + "=" * 60)
    print("ARCHITECTURE PARTITIONS")
    print("=" * 60)
    print(f"Total partitions: {stats['num_partitions']}")
    print(f"Total operators: {stats['total_operators']}")
    print(f"Partition sizes: min={stats['min_partition_size']}, "
          f"max={stats['max_partition_size']}, "
          f"avg={stats['avg_partition_size']:.1f}")
    print()
    print("Signature distribution (num_layers, channels_bucket, kernel_size):")
    for sig, partition in sorted(partitions.items(), key=lambda x: -x[1].size):
        compiled_str = "compiled" if partition.compiled_template is not None else "eager"
        print(f"  {sig}: {partition.size} operators ({compiled_str})")
    print("=" * 60 + "\n")
