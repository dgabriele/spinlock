"""
Dataset generation pipeline orchestrator.

Coordinates all components:
- Parameter sampling (Sobol)
- Operator building (from parameters)
- Input generation (GRF, structured)
- GPU execution (parallel inference)
- Storage (HDF5)

Design principles:
- Pipeline pattern: Composable stages
- Dependency injection: Testable components
- Progress tracking: User visibility
- Resource management: Clean cleanup
"""

import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import time

from .generators import InputFieldGenerator
from .storage import HDF5DatasetWriter
from ..sampling import StratifiedSobolSampler
from ..operators import OperatorBuilder, NeuralOperator
from ..execution import ParallelExecutor, AdaptiveBatchSizer, MemoryManager
from ..config import SpinlockConfig


class DatasetGenerationPipeline:
    """
    High-performance dataset generation pipeline.

    Orchestrates complete workflow:
    1. Sample parameter space
    2. Build operators from parameters
    3. Generate input fields
    4. Execute operators (with stochastic realizations)
    5. Store results in HDF5

    Example:
        ```python
        from spinlock.config import load_config

        config = load_config("config.yaml")
        pipeline = DatasetGenerationPipeline(config)
        pipeline.generate()
        ```
    """

    def __init__(self, config: SpinlockConfig):
        """
        Initialize dataset generation pipeline.

        Args:
            config: Complete Spinlock configuration
        """
        self.config = config

        # Setup device
        self.device = self._setup_device()

        # Initialize components
        self.sampler = self._create_sampler()
        self.operator_builder = OperatorBuilder()
        self.input_generator = self._create_input_generator()
        self.parallel_executor = self._create_parallel_executor()

        # Statistics
        self.stats = {
            "total_time": 0.0,
            "sampling_time": 0.0,
            "generation_time": 0.0,
            "inference_time": 0.0,
            "storage_time": 0.0,
            "samples_generated": 0,
        }

    def _setup_device(self) -> torch.device:
        """Setup torch device from config."""
        device_str = self.config.simulation.device

        if device_str == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        else:
            return torch.device(device_str)

    def _create_sampler(self) -> StratifiedSobolSampler:
        """Create parameter space sampler from config."""
        return StratifiedSobolSampler.from_config(self.config.parameter_space, self.config.sampling)

    def _create_input_generator(self) -> InputFieldGenerator:
        """Create input field generator from config."""
        # Fixed dimensions for MVP (homogeneous operators)
        # Future: Extract from parameter space for heterogeneous support
        grid_size = 64
        num_channels = 3

        return InputFieldGenerator(
            grid_size=grid_size, num_channels=num_channels, device=self.device
        )

    def _create_parallel_executor(self) -> ParallelExecutor:
        """Create parallel executor from config."""
        strategy = self.config.simulation.parallelism.strategy

        device_ids = None
        if self.config.simulation.parallelism.devices != "auto":
            device_ids = self.config.simulation.parallelism.devices

        return ParallelExecutor(strategy=strategy, device_ids=device_ids)

    def generate(self) -> None:
        """
        Execute complete dataset generation pipeline.

        Main entry point that coordinates all stages.
        """
        start_time = time.time()

        print("=" * 60)
        print("SPINLOCK DATASET GENERATION")
        print("=" * 60)
        print(f"Output: {self.config.dataset.output_path}")
        print(f"Samples: {self.config.sampling.total_samples}")
        print(f"Realizations: {self.config.simulation.num_realizations}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")

        # Stage 1: Sample parameter space
        print("Stage 1/4: Sampling parameter space...")
        sample_start = time.time()
        parameters = self.sampler.sample(self.config.sampling.total_samples)
        validation_metrics = self.sampler.validate(parameters)
        self.stats["sampling_time"] = time.time() - sample_start

        print(f"✓ Generated {len(parameters)} parameter sets")
        print(f"  Discrepancy: {validation_metrics['discrepancy']:.6f}")
        print(f"  Max correlation: {validation_metrics['max_correlation']:.6f}\n")

        # Stage 2-4: Generate dataset in batches
        self._generate_dataset_batches(parameters, validation_metrics)

        # Final statistics
        self.stats["total_time"] = time.time() - start_time
        self._print_final_statistics()

    def _generate_dataset_batches(
        self, parameters: NDArray[np.float64], validation_metrics: Dict[str, Any]
    ) -> None:
        """
        Generate dataset in batches.

        Args:
            parameters: Sampled parameter sets [N, P]
            validation_metrics: Sampling validation metrics
        """
        num_samples = len(parameters)
        batch_size = self.config.sampling.batch_size

        # Create HDF5 writer
        with HDF5DatasetWriter(
            output_path=self.config.dataset.output_path,
            grid_size=64,  # TODO: Extract from config
            input_channels=3,  # TODO: Extract from config
            output_channels=3,  # TODO: Extract from config
            num_realizations=self.config.simulation.num_realizations,
            num_parameter_sets=num_samples,
            compression=self.config.dataset.storage.compression,
            compression_opts=self.config.dataset.storage.compression_level,
            chunk_size=self.config.dataset.storage.chunk_size,
        ) as writer:

            # Write metadata
            writer.write_metadata(
                {
                    "config": self.config.model_dump(mode="json"),
                    "sampling_metrics": validation_metrics,
                }
            )

            # Process in batches
            num_batches = (num_samples + batch_size - 1) // batch_size

            print(f"Stage 2-4: Generating dataset ({num_batches} batches)...")

            with tqdm(total=num_samples, desc="Generating") as pbar:
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, num_samples)
                    current_batch_size = batch_end - batch_start

                    # Extract parameter batch
                    param_batch = parameters[batch_start:batch_end]

                    # Process batch
                    batch_inputs, batch_outputs = self._process_batch(
                        param_batch, current_batch_size
                    )

                    # Write to HDF5
                    store_start = time.time()
                    writer.write_batch(
                        parameters=param_batch, inputs=batch_inputs, outputs=batch_outputs
                    )
                    self.stats["storage_time"] += time.time() - store_start

                    self.stats["samples_generated"] += current_batch_size
                    pbar.update(current_batch_size)

                    # Memory cleanup
                    if batch_idx % 10 == 0 and self.device.type == "cuda":
                        torch.cuda.empty_cache()

        print(f"\n✓ Dataset saved: {self.config.dataset.output_path}")

    def _process_batch(
        self, param_batch: NDArray[np.float64], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single batch: build operators, generate inputs, simulate.

        Args:
            param_batch: Parameter values [B, P]
            batch_size: Batch size

        Returns:
            Tuple of (inputs, outputs)
            - inputs: [B, C_in, H, W]
            - outputs: [B, M, C_out, H, W]
        """
        # Build operators from parameters with fixed dimensions for MVP
        operators = []

        # Fixed dimensions (MVP constraint: homogeneous operators)
        fixed_input_channels = 3
        fixed_output_channels = 3
        fixed_grid_size = 64

        for params in param_batch:
            # Map parameters
            param_dict = self._map_single_parameter_set(params)

            # Add fixed dimensions if not present
            param_dict.setdefault("input_channels", fixed_input_channels)
            param_dict.setdefault("output_channels", fixed_output_channels)
            param_dict.setdefault("grid_size", fixed_grid_size)

            # Build operator
            model = self.operator_builder.build_simple_cnn(param_dict)
            operator = NeuralOperator(model)

            # Prepare for inference
            operator = MemoryManager.optimize_for_inference(operator)
            operator = operator.to(self.device)

            operators.append(operator)

        # Generate input fields (all operators have same dimensions in MVP)
        gen_start = time.time()
        inputs = self.input_generator.generate_batch(
            batch_size=batch_size,
            field_type=self.config.simulation.input_generation.method,
            length_scale=self.config.simulation.input_generation.length_scale,
            variance=self.config.simulation.input_generation.variance,
        )
        self.stats["generation_time"] += time.time() - gen_start

        # Run inference with stochastic realizations
        inf_start = time.time()
        outputs = self._run_inference_batch(
            operators, inputs, self.config.simulation.num_realizations
        )
        self.stats["inference_time"] += time.time() - inf_start

        return inputs, outputs

    def _map_single_parameter_set(self, params: NDArray[np.float64]) -> Dict[str, Any]:
        """
        Map unit parameters [0,1]^d to actual parameter values.

        Args:
            params: Unit parameters

        Returns:
            Dictionary of parameter values
        """
        # Flatten all parameter specs into a single dict
        all_specs = {}
        all_specs.update(self.config.parameter_space.architecture)
        all_specs.update(self.config.parameter_space.stochastic)
        all_specs.update(self.config.parameter_space.operator)

        # Convert Pydantic models to dicts
        all_specs_dict = {}
        for name, spec in all_specs.items():
            all_specs_dict[name] = spec.model_dump()

        # Map using builder
        return self.operator_builder.map_parameters(params, all_specs_dict)

    def _run_inference_batch(
        self, operators: list[NeuralOperator], inputs: torch.Tensor, num_realizations: int
    ) -> torch.Tensor:
        """
        Run inference for a batch of operators.

        Args:
            operators: List of operators
            inputs: Input fields [B, C_in, H, W]
            num_realizations: Number of stochastic realizations

        Returns:
            Outputs [B, M, C_out, H, W]
        """
        all_outputs = []

        use_amp = self.config.simulation.precision in ["float16", "bfloat16"]

        with torch.no_grad():
            for i, operator in enumerate(operators):
                # Single input for this operator
                input_i = inputs[i : i + 1]

                # Generate realizations
                with torch.cuda.amp.autocast(enabled=use_amp):
                    realizations = operator.generate_realizations(
                        input_i,
                        num_realizations=num_realizations,
                        base_seed=i,  # Use index as seed for reproducibility
                    )

                # realizations shape: [1, M, C_out, H, W]
                # We want [M, C_out, H, W]
                all_outputs.append(realizations[0])

        # Stack: [B, M, C_out, H, W]
        return torch.stack(all_outputs, dim=0)

    def _print_final_statistics(self) -> None:
        """Print final generation statistics."""
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)

        total_time = self.stats["total_time"]
        samples = self.stats["samples_generated"]

        print(f"Total samples: {samples:,}")
        print(f"Realizations per sample: {self.config.simulation.num_realizations}")
        print(f"Total outputs: {samples * self.config.simulation.num_realizations:,}")
        print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Throughput: {samples/total_time:.2f} samples/sec")

        print(f"\nTime breakdown:")
        print(
            f"  Sampling: {self.stats['sampling_time']:.2f}s "
            f"({self.stats['sampling_time']/total_time*100:.1f}%)"
        )
        print(
            f"  Input generation: {self.stats['generation_time']:.2f}s "
            f"({self.stats['generation_time']/total_time*100:.1f}%)"
        )
        print(
            f"  Inference: {self.stats['inference_time']:.2f}s "
            f"({self.stats['inference_time']/total_time*100:.1f}%)"
        )
        print(
            f"  Storage (HDF5): {self.stats['storage_time']:.2f}s "
            f"({self.stats['storage_time']/total_time*100:.1f}%)"
        )

        if self.device.type == "cuda":
            mem_stats = MemoryManager.get_memory_stats(self.device)
            print(f"\nGPU Memory:")
            print(f"  Peak allocated: {mem_stats['max_allocated']:.2f} GB")

        print("=" * 60)
