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
        Generate dataset in batches with group-by-grid-size strategy.

        Args:
            parameters: Sampled parameter sets [N, P]
            validation_metrics: Sampling validation metrics
        """
        num_samples = len(parameters)
        batch_size = self.config.sampling.batch_size

        # Group operators by grid size first to determine max size needed
        print("Grouping operators by grid size...")
        grid_size_groups = self._group_by_grid_size(parameters)

        # Determine actual max grid size (no padding needed if all same size)
        self.max_grid_size = max(grid_size_groups.keys())  # Store as instance variable
        if len(grid_size_groups) == 1:
            print(f"Single grid size detected: {self.max_grid_size}×{self.max_grid_size} (no padding needed)")
        else:
            print(f"Multiple grid sizes detected, will pad to {self.max_grid_size}×{self.max_grid_size}")

        # Create HDF5 writer with actual max grid size (not hardcoded 256)
        with HDF5DatasetWriter(
            output_path=self.config.dataset.output_path,
            grid_size=self.max_grid_size,  # Use actual max, not hardcoded 256
            input_channels=3,  # TODO: Extract from config
            output_channels=3,  # TODO: Extract from config
            num_realizations=self.config.simulation.num_realizations,
            num_parameter_sets=num_samples,
            compression=self.config.dataset.storage.compression,
            compression_opts=self.config.dataset.storage.compression_level,
            chunk_size=self.config.dataset.storage.chunk_size,
            track_ic_metadata=True,  # Enable discovery metadata
        ) as writer:

            # Write metadata
            writer.write_metadata(
                {
                    "config": self.config.model_dump(mode="json"),
                    "sampling_metrics": validation_metrics,
                }
            )

            print(f"Grid size distribution:")
            for grid_size, indices in sorted(grid_size_groups.items()):
                print(f"  {grid_size}×{grid_size}: {len(indices)} operators ({len(indices)/num_samples*100:.1f}%)")
            print()

            # Process each grid size group separately
            print(f"Stage 2-4: Generating dataset (group-by-grid-size strategy)...\n")

            with tqdm(total=num_samples, desc="Generating") as pbar:
                for grid_size in sorted(grid_size_groups.keys()):
                    indices = grid_size_groups[grid_size]
                    group_params = parameters[indices]

                    print(f"\nProcessing {grid_size}×{grid_size} grid ({len(indices)} operators)...")

                    # Adaptive batch processing for this grid size
                    current_batch_size = batch_size
                    min_batch_size = 1
                    group_processed = 0

                    # Intelligent batch size search state
                    max_safe_batch_size = None  # Will be determined through search
                    in_search_mode = False  # True after first OOM, incrementally searching for max
                    search_increment_pct = 0.05  # Increase by 5% each successful batch during search

                    while group_processed < len(indices):
                        batch_start_idx = group_processed
                        batch_end_idx = min(batch_start_idx + current_batch_size, len(indices))
                        actual_batch_size = batch_end_idx - batch_start_idx

                        # Extract parameter batch
                        param_batch = group_params[batch_start_idx:batch_end_idx]
                        batch_indices = indices[batch_start_idx:batch_end_idx]

                        try:
                            # Process batch with this grid size
                            batch_inputs, batch_outputs, metadata = self._process_batch_with_metadata(
                                param_batch, actual_batch_size, grid_size
                            )

                            # Write to HDF5 (with metadata and proper indexing)
                            store_start = time.time()
                            self._write_batch_to_hdf5(
                                writer, batch_indices, param_batch, batch_inputs, batch_outputs, metadata
                            )
                            self.stats["storage_time"] += time.time() - store_start

                            self.stats["samples_generated"] += actual_batch_size
                            group_processed += actual_batch_size
                            pbar.update(actual_batch_size)

                            # Memory cleanup
                            if group_processed % (10 * current_batch_size) == 0:
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()

                            # Intelligent batch size adaptation
                            if self.device.type == "cuda":
                                if in_search_mode and max_safe_batch_size is None:
                                    # In search mode: incrementally increase by small percentage
                                    # to find the true maximum batch size
                                    new_batch_size = max(
                                        current_batch_size + 1,  # At least +1
                                        int(current_batch_size * (1 + search_increment_pct))
                                    )
                                    print(f"  ↑ Search mode: {current_batch_size} → {new_batch_size} (+{search_increment_pct*100:.0f}%)")
                                    current_batch_size = new_batch_size

                                elif max_safe_batch_size is None:
                                    # No OOM yet, not in search mode - use initial batch size
                                    # (This happens on first iteration before any OOM)
                                    pass

                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                # OOM recovery: intelligent search for optimal batch size
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()

                                if current_batch_size <= min_batch_size:
                                    raise RuntimeError(
                                        f"OOM with minimum batch size ({min_batch_size}). "
                                        f"GPU memory insufficient for this task."
                                    ) from e

                                if not in_search_mode:
                                    # First OOM: halve batch size and enter search mode
                                    new_batch_size = max(min_batch_size, current_batch_size // 2)
                                    print(f"\n  ⚠ OOM at batch_size={current_batch_size}")
                                    print(f"  → Reducing to {new_batch_size}, entering search mode (will increment by {search_increment_pct*100:.0f}% to find optimal size)")
                                    current_batch_size = new_batch_size
                                    in_search_mode = True
                                else:
                                    # Second OOM while searching: we've found the limit
                                    # Back off by 10% as safety margin and lock it in
                                    max_safe_batch_size = int(current_batch_size * 0.9)
                                    current_batch_size = max_safe_batch_size
                                    print(f"\n  ⚠ OOM at batch_size={int(current_batch_size / 0.9)} during search")
                                    print(f"  → Optimal batch size found: {max_safe_batch_size} (90% of max for safety)")
                                    print(f"  → Search complete, will use batch_size={max_safe_batch_size} for remainder")

                                # Don't update group_processed, retry this batch
                            else:
                                raise

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
        all_specs.update(self.config.parameter_space.evolution)

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
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
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

    def _group_by_grid_size(self, parameters: NDArray[np.float64]) -> Dict[int, NDArray[np.int64]]:
        """
        Group parameter indices by grid size.

        Args:
            parameters: All parameter sets [N, P]

        Returns:
            Dict mapping grid_size -> array of indices
        """
        from collections import defaultdict

        groups = defaultdict(list)

        for idx, params in enumerate(parameters):
            param_dict = self._map_single_parameter_set(params)
            grid_size = int(param_dict.get("grid_size", 64))  # Default to 64 if not found
            groups[grid_size].append(idx)

        # Convert lists to numpy arrays
        return {grid_size: np.array(indices, dtype=np.int64) for grid_size, indices in groups.items()}

    def _process_batch_with_metadata(
        self, param_batch: NDArray[np.float64], batch_size: int, grid_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process a batch with grid-size-specific logic and metadata tracking.

        Args:
            param_batch: Parameter values [B, P]
            batch_size: Batch size
            grid_size: Grid size for this batch

        Returns:
            Tuple of (inputs, outputs, metadata)
            - inputs: [B, C_in, H, W] (padded to max_grid_size if needed)
            - outputs: [B, M, C_out, H, W] (padded to max_grid_size if needed)
            - metadata: Dict with ic_types, evolution_policies, grid_sizes, noise_regimes
        """
        # Process batch with variable grid size and track IC types used
        inputs, outputs, ic_types_used = self._process_batch_variable_size_with_tracking(
            param_batch, batch_size, grid_size
        )

        # Extract metadata from parameters
        evolution_policies = []
        noise_regimes = []
        grid_sizes = []

        for params in param_batch:
            param_dict = self._map_single_parameter_set(params)

            # Evolution policy
            evolution_policies.append(param_dict.get("update_policy", "autoregressive"))

            # Grid size
            grid_sizes.append(grid_size)

            # Noise regime (classify based on noise_scale)
            noise_scale = param_dict.get("noise_scale", 0.01)
            if noise_scale < 0.01:
                noise_regimes.append("low")
            elif noise_scale < 0.1:
                noise_regimes.append("medium")
            else:
                noise_regimes.append("high")

        metadata = {
            "ic_types": ic_types_used,
            "evolution_policies": evolution_policies,
            "grid_sizes": grid_sizes,
            "noise_regimes": noise_regimes,
        }

        return inputs, outputs, metadata

    def _process_batch_variable_size_with_tracking(
        self, param_batch: NDArray[np.float64], batch_size: int, grid_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Process batch with variable grid size and track IC types used.

        Args:
            param_batch: Parameter values [B, P]
            batch_size: Batch size
            grid_size: Grid size for this batch

        Returns:
            Tuple of (inputs, outputs, ic_types_used)
            - inputs: [B, C_in, H, W] (padded to max_grid_size if needed)
            - outputs: [B, M, C_out, H, W] (padded to max_grid_size if needed)
            - ic_types_used: List of IC types used for each sample
        """
        # Build operators with this grid size
        operators = []

        fixed_input_channels = 3
        fixed_output_channels = 3

        for params in param_batch:
            param_dict = self._map_single_parameter_set(params)

            # Override grid_size (already extracted)
            param_dict["input_channels"] = fixed_input_channels
            param_dict["output_channels"] = fixed_output_channels
            param_dict["grid_size"] = grid_size

            # Build operator
            model = self.operator_builder.build_simple_cnn(param_dict)
            operator = NeuralOperator(model)

            # Prepare for inference
            operator = MemoryManager.optimize_for_inference(operator)
            operator = operator.to(self.device)

            operators.append(operator)

        # Generate input fields with this grid size
        input_generator = InputFieldGenerator(
            grid_size=grid_size, num_channels=fixed_input_channels, device=self.device
        )

        gen_start = time.time()

        # Sample IC types for each sample in batch
        ic_types_used = []
        all_inputs = []

        for i in range(batch_size):
            # Determine IC type for this sample
            ic_method = self.config.simulation.input_generation.method
            if ic_method == "sampled":
                # Sample IC type based on weights
                ic_type = self._sample_ic_type()
            else:
                ic_type = ic_method

            ic_types_used.append(ic_type)

            # Get parameters for this IC type
            ic_params = self._get_ic_params(ic_type)

            # Generate single input
            input_i = input_generator.generate_batch(
                batch_size=1,
                field_type=ic_type,
                **ic_params
            )
            all_inputs.append(input_i)

        # Stack all inputs
        inputs = torch.cat(all_inputs, dim=0)
        self.stats["generation_time"] += time.time() - gen_start

        # Run inference
        inf_start = time.time()
        outputs = self._run_inference_batch(
            operators, inputs, self.config.simulation.num_realizations
        )
        self.stats["inference_time"] += time.time() - inf_start

        # Pad to max_grid_size if needed (only if variable grid sizes)
        if grid_size < self.max_grid_size:
            inputs = self._pad_to_max_size(inputs, target_size=self.max_grid_size)
            outputs = self._pad_to_max_size(outputs, target_size=self.max_grid_size)

        return inputs, outputs, ic_types_used

    def _sample_ic_type(self) -> str:
        """
        Sample IC type based on configured weights.

        Returns:
            IC type name
        """
        weights = self.config.simulation.input_generation.ic_type_weights
        if not weights:
            # Default to gaussian_random_field if no weights specified
            return "gaussian_random_field"

        # Normalize weights
        total_weight = sum(weights.values())
        ic_types = list(weights.keys())
        probs = [weights[ic] / total_weight for ic in ic_types]

        # Sample
        import random
        return random.choices(ic_types, weights=probs, k=1)[0]

    def _get_ic_params(self, ic_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific IC type from config.

        Args:
            ic_type: IC type name

        Returns:
            Dict of parameters for this IC type
        """
        config_gen = self.config.simulation.input_generation

        # Check if this IC type has specific config
        if ic_type == "multiscale_grf" and config_gen.multiscale_grf:
            return config_gen.multiscale_grf.copy()
        elif ic_type == "localized" and config_gen.localized:
            return config_gen.localized.copy()
        elif ic_type == "composite" and config_gen.composite:
            return config_gen.composite.copy()
        elif ic_type == "heavy_tailed" and config_gen.heavy_tailed:
            return config_gen.heavy_tailed.copy()
        # Tier 1 domain-specific ICs
        elif ic_type == "quantum_wave_packet" and config_gen.quantum_wave_packet:
            return config_gen.quantum_wave_packet.copy()
        elif ic_type == "turing_pattern" and config_gen.turing_pattern:
            return config_gen.turing_pattern.copy()
        elif ic_type == "thermal_gradient" and config_gen.thermal_gradient:
            return config_gen.thermal_gradient.copy()
        elif ic_type == "morphogen_gradient" and config_gen.morphogen_gradient:
            return config_gen.morphogen_gradient.copy()
        elif ic_type == "reaction_front" and config_gen.reaction_front:
            return config_gen.reaction_front.copy()
        # Tier 2 domain-specific ICs
        elif ic_type == "light_cone" and config_gen.light_cone:
            return config_gen.light_cone.copy()
        elif ic_type == "critical_fluctuation" and config_gen.critical_fluctuation:
            return config_gen.critical_fluctuation.copy()
        elif ic_type == "phase_boundary" and config_gen.phase_boundary:
            return config_gen.phase_boundary.copy()
        elif ic_type == "bz_reaction" and config_gen.bz_reaction:
            return config_gen.bz_reaction.copy()
        elif ic_type == "shannon_entropy" and config_gen.shannon_entropy:
            return config_gen.shannon_entropy.copy()
        # Tier 3 domain-specific ICs
        elif ic_type == "interference_pattern" and config_gen.interference_pattern:
            return config_gen.interference_pattern.copy()
        elif ic_type == "cell_population" and config_gen.cell_population:
            return config_gen.cell_population.copy()
        elif ic_type == "chromatin_domain" and config_gen.chromatin_domain:
            return config_gen.chromatin_domain.copy()
        elif ic_type == "shock_front" and config_gen.shock_front:
            return config_gen.shock_front.copy()
        elif ic_type == "gene_expression" and config_gen.gene_expression:
            return config_gen.gene_expression.copy()
        # Tier 4 research frontiers ICs
        elif ic_type == "coherent_state" and config_gen.coherent_state:
            return config_gen.coherent_state.copy()
        elif ic_type == "relativistic_wave_packet" and config_gen.relativistic_wave_packet:
            return config_gen.relativistic_wave_packet.copy()
        elif ic_type == "mutual_information" and config_gen.mutual_information:
            return config_gen.mutual_information.copy()
        elif ic_type == "regulatory_network" and config_gen.regulatory_network:
            return config_gen.regulatory_network.copy()
        elif ic_type == "dla_cluster" and config_gen.dla_cluster:
            return config_gen.dla_cluster.copy()
        elif ic_type == "error_correcting_code" and config_gen.error_correcting_code:
            return config_gen.error_correcting_code.copy()
        elif ic_type == "gaussian_random_field":
            return {
                "length_scale": config_gen.length_scale,
                "variance": config_gen.variance
            }
        else:
            # Default params
            return {
                "length_scale": config_gen.length_scale,
                "variance": config_gen.variance
            }

    def _pad_to_max_size(self, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Pad tensor to target size with zeros (dynamic based on dataset grid sizes).

        Args:
            tensor: Input tensor [..., H, W]
            target_size: Target spatial size

        Returns:
            Padded tensor [..., target_size, target_size]
        """
        current_size = tensor.shape[-1]
        if current_size >= target_size:
            return tensor

        pad_total = target_size - current_size
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        # Pad last two dimensions (H, W)
        return torch.nn.functional.pad(
            tensor, (pad_left, pad_right, pad_left, pad_right), mode="constant", value=0
        )

    def _write_batch_to_hdf5(
        self,
        writer: HDF5DatasetWriter,
        batch_indices: NDArray[np.int64],
        param_batch: NDArray[np.float64],
        batch_inputs: torch.Tensor,
        batch_outputs: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Write batch to HDF5 with proper indexing and metadata.

        Args:
            writer: HDF5 writer
            batch_indices: Global indices for this batch
            param_batch: Parameters
            batch_inputs: Input fields
            batch_outputs: Output fields
            metadata: Per-sample metadata
        """
        # We need to write to specific indices, not sequential
        # But HDF5DatasetWriter expects sequential writes
        # For now, store temporarily and write sequentially
        # This is a limitation we'll need to address

        # Workaround: Write batch sequentially (assumes group-by-grid processes in order)
        writer.write_batch(
            parameters=param_batch,
            inputs=batch_inputs,
            outputs=batch_outputs,
            ic_types=metadata["ic_types"],
            evolution_policies=metadata["evolution_policies"],
            grid_sizes=metadata["grid_sizes"],
            noise_regimes=metadata["noise_regimes"],
        )

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
