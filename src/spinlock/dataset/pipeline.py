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
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
import time
import gc
import queue
import threading
from dataclasses import dataclass

from .generators import InputFieldGenerator
from .storage import HDF5DatasetWriter
from ..sampling import StratifiedSobolSampler
from ..operators import OperatorBuilder, NeuralOperator
from ..execution import ParallelExecutor, AdaptiveBatchSizer, MemoryManager
from ..config import SpinlockConfig
from ..features.sdf import SummaryExtractor, SummaryConfig
from ..features.storage import HDF5FeatureWriter


@dataclass
class FeatureWriteTask:
    """Task for async feature writing."""
    batch_idx_start: int
    per_timestep: np.ndarray
    per_trajectory: np.ndarray
    aggregated: np.ndarray


class AsyncFeatureWriter:
    """
    Asynchronous feature writer using background thread.
    
    Enables pipelining: GPU feature extraction can proceed while
    previous batch is being written to HDF5 (I/O bound).
    """
    
    def __init__(self, feature_writer: HDF5FeatureWriter, max_queue_size: int = 2):
        """
        Initialize async feature writer.
        
        Args:
            feature_writer: HDF5FeatureWriter instance
            max_queue_size: Maximum number of batches to buffer
        """
        self.feature_writer = feature_writer
        self.write_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.write_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error: Optional[Exception] = None
        
    def start(self) -> None:
        """Start background write thread."""
        if self.write_thread is not None:
            return
            
        self.write_thread = threading.Thread(
            target=self._write_worker,
            daemon=True,
            name="FeatureWriter"
        )
        self.write_thread.start()
    
    def stop(self) -> None:
        """Stop background write thread and wait for completion."""
        if self.write_thread is None:
            return
            
        self._stop_event.set()
        self.write_queue.put(None)  # Sentinel to wake worker
        self.write_thread.join(timeout=30.0)
        
        if self._error is not None:
            raise RuntimeError("Error in feature write thread") from self._error
    
    def enqueue(self, task: FeatureWriteTask) -> None:
        """
        Enqueue feature write task (non-blocking if queue not full).
        
        Args:
            task: Feature write task
        """
        if self._error is not None:
            raise RuntimeError("Feature writer thread encountered error") from self._error
            
        self.write_queue.put(task, block=True)
    
    def wait_for_completion(self) -> None:
        """Wait for all queued writes to complete."""
        self.write_queue.join()
    
    def _write_worker(self) -> None:
        """Background thread worker for HDF5 writes."""
        try:
            while not self._stop_event.is_set():
                try:
                    task = self.write_queue.get(timeout=0.1)
                    if task is None:  # Sentinel
                        break
                        
                    self.feature_writer.write_sdf_batch(
                        batch_idx=task.batch_idx_start,
                        per_timestep=task.per_timestep,
                        per_trajectory=task.per_trajectory,
                        aggregated=task.aggregated
                    )
                    
                    self.write_queue.task_done()
                except queue.Empty:
                    continue
        except Exception as e:
            self._error = e
            self.write_queue.task_done()


class FeatureExtractionPipeline:
    """
    Optimized feature extraction pipeline.
    
    Handles GPU feature extraction with efficient memory management
    and eliminates wasteful recomputation.
    """
    
    def __init__(self, sdf_extractor: SummaryExtractor, device: torch.device):
        """
        Initialize feature extraction pipeline.
        
        Args:
            sdf_extractor: SDF feature extractor
            device: Computation device
        """
        self.sdf_extractor = sdf_extractor
        self.device = device
    
    def extract_all(
        self,
        batch_outputs: torch.Tensor  # [B, M, T, C, H, W]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all features efficiently (no recomputation).
        
        Args:
            batch_outputs: Trajectory tensors [B, M, T, C, H, W]
            
        Returns:
            Tuple of (per_timestep_np, per_trajectory_np, aggregated_np)
        """
        with torch.no_grad():
            # Stage 1: Extract per-timestep features
            per_timestep_gpu = self.sdf_extractor.extract_per_timestep(batch_outputs)
            per_timestep_np = per_timestep_gpu.cpu().numpy()
            del per_timestep_gpu
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Stage 2: Extract per-trajectory features
            per_trajectory_gpu = self.sdf_extractor.extract_per_trajectory(batch_outputs)
            per_trajectory_np = per_trajectory_gpu.cpu().numpy()
            
            # Stage 3: Aggregate using same GPU tensor (no recomputation)
            aggregated_list = []
            for method in ['mean', 'std', 'cv']:
                agg_gpu = self.sdf_extractor.aggregate_realizations(
                    per_trajectory_gpu, method=method
                )
                aggregated_list.append(agg_gpu.cpu())
                del agg_gpu
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            aggregated_np = torch.cat(aggregated_list, dim=1).numpy()
            
            # Free GPU memory
            del per_trajectory_gpu, aggregated_list
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return per_timestep_np, per_trajectory_np, aggregated_np


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
            "feature_extraction_time": 0.0,
            "samples_generated": 0,
        }

        # Policy cache for temporal mode (reduces instantiation overhead)
        self._policy_cache: Dict[Tuple[str, float, float], Any] = {}
        
        # Feature extraction pipeline (initialized when needed)
        self._feature_pipeline: Optional[FeatureExtractionPipeline] = None
        self._async_feature_writer: Optional[AsyncFeatureWriter] = None

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

    def _get_or_create_policy(
        self,
        policy_type: str,
        dt: float = 0.01,
        alpha: float = 0.5
    ) -> Any:
        """
        Get cached policy or create new one.

        Caches policies by (policy_type, dt, alpha) tuple to avoid repeated
        instantiation overhead during dataset generation.

        Args:
            policy_type: "autoregressive", "residual", or "convex"
            dt: Step size for residual policy
            alpha: Mixing parameter for convex policy

        Returns:
            Cached or newly created UpdatePolicy instance

        Note:
            Unused parameters are normalized to 0.0 for consistent cache keys:
            - autoregressive: dt=0.0, alpha=0.0
            - residual: alpha=0.0
            - convex: dt=0.0
        """
        from ..rollout import create_update_policy

        # Normalize unused params for consistent hashing
        if policy_type == "autoregressive":
            dt, alpha = 0.0, 0.0
        elif policy_type == "residual":
            alpha = 0.0
        elif policy_type == "convex":
            dt = 0.0

        cache_key = (policy_type, dt, alpha)

        if cache_key not in self._policy_cache:
            self._policy_cache[cache_key] = create_update_policy(
                policy_type=policy_type,
                dt=dt,
                alpha=alpha
            )

        return self._policy_cache[cache_key]

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

        # Cleanup after generation
        # Wait for async feature writes to complete
        if self._async_feature_writer is not None:
            self._async_feature_writer.wait_for_completion()
            self._async_feature_writer.stop()
        
        # Close feature writer if opened
        if hasattr(self, "_feature_writer") and self._feature_writer is not None:
            self._feature_writer.__exit__(None, None, None)
        
        self.cleanup()

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

        # Get store_trajectories setting (defaults to False for storage efficiency)
        # Feature-only mode saves ~99% storage (10GB vs 1.2TB for 10K temporal operators)
        store_trajectories = getattr(self.config.dataset.storage, 'store_trajectories', False)

        if not store_trajectories:
            print("\n⚠️  FEATURE-ONLY MODE ENABLED")
            print("   Trajectories will NOT be stored to save space")
            print("   Features must be extracted during generation or from memory")
            print("   Dataset size: <10 GB (vs. ~1.2 TB with trajectories)\n")

        # Initialize inline feature extraction
        sdf_extractor = None
        feature_writer = None
        
        if not store_trajectories:
            print("Initializing inline feature extractor...")
            sdf_config = SummaryConfig()  # Default: all features
            sdf_extractor = SummaryExtractor(device=self.device, config=sdf_config)
            print(f"  Feature extractor ready on {self.device}\n")
            self._sdf_extractor = sdf_extractor

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
            store_trajectories=store_trajectories,  # Feature-only mode support
            num_timesteps=self.config.simulation.num_timesteps,  # Temporal support
        ) as writer:

            # Initialize feature writer for inline extraction
            if sdf_extractor is not None:
                from pathlib import Path
                feature_writer = HDF5FeatureWriter(
                    dataset_path=Path(self.config.dataset.output_path),
                    overwrite=False  # Append mode - don't overwrite existing data
                )
                feature_writer.__enter__()
                self._feature_writer = feature_writer

                # Create SDF storage groups
                registry = sdf_extractor.get_feature_registry()
                sdf_config = sdf_extractor.config

                # Calculate dimensions for logging
                # Per-timestep categories: spatial, spectral, cross_channel
                per_timestep_dim = (
                    len(registry.get_feature_names(category='spatial')) +
                    len(registry.get_feature_names(category='spectral')) +
                    len(registry.get_feature_names(category='cross_channel'))
                )

                # Per-trajectory categories: temporal, causality, invariant_drift, operator_sensitivity
                per_trajectory_dim = (
                    len(registry.get_feature_names(category='temporal')) +
                    len(registry.get_feature_names(category='causality')) +
                    len(registry.get_feature_names(category='invariant_drift')) +
                    len(registry.get_feature_names(category='operator_sensitivity'))
                )

                feature_writer.create_sdf_group(
                    num_samples=num_samples,
                    num_realizations=self.config.simulation.num_realizations,
                    num_timesteps=self.config.simulation.num_timesteps,
                    registry=registry,
                    config=sdf_config,
                    compression=self.config.dataset.storage.compression,
                    compression_opts=self.config.dataset.storage.compression_level,
                    chunk_size=self.config.dataset.storage.chunk_size
                )

                # Initialize optimized feature extraction pipeline
                self._feature_pipeline = FeatureExtractionPipeline(
                    sdf_extractor=sdf_extractor,
                    device=self.device
                )
                
                # Initialize async feature writer for pipelining
                self._async_feature_writer = AsyncFeatureWriter(
                    feature_writer=feature_writer,
                    max_queue_size=2  # Buffer 2 batches
                )
                self._async_feature_writer.start()

                print(f"Feature storage initialized:")
                print(f"  Per-timestep: {per_timestep_dim} features")
                print(f"  Per-trajectory: {per_trajectory_dim} features")
                print(f"  Aggregated: {per_trajectory_dim * 3} features")
                print(f"  Total registry size: {registry.num_features} features")
                print(f"  Async I/O: Enabled (pipelined writes)\n")

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

                            # Aggressive memory cleanup after HDF5 write
                            del batch_inputs, batch_outputs, metadata
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()

                            self.stats["samples_generated"] += actual_batch_size
                            group_processed += actual_batch_size
                            pbar.update(actual_batch_size)

                            # Periodic garbage collection (every 5 batches instead of 10)
                            if group_processed % (5 * current_batch_size) == 0:
                                gc.collect()
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

            # Set seed for deterministic operator initialization
            # Note: Global seed derived from batch position
            # For deterministic replay: torch.manual_seed(op_global_idx)
            torch.manual_seed(hash(str(params)) % (2**31))  # Deterministic per params
            
            # Build operator
            model = self.operator_builder.build_simple_cnn(param_dict)
            operator = NeuralOperator(model)

            # Prepare for inference
            operator = MemoryManager.optimize_for_inference(operator)
            operator = operator.to(self.device)

            # torch.compile() disabled to reduce GPU memory pressure during inline feature extraction
            # Enables batch_size=2 instead of batch_size=1 (saves ~30% wall time)
            # operator.enable_compile(mode="reduce-overhead")

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
            operators, inputs, self.config.simulation.num_realizations, param_batch
        )
        self.stats["inference_time"] += time.time() - inf_start

        # Explicitly delete operators to break reference cycles and free GPU memory
        for op in operators:
            del op
        del operators
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection

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
        self, operators: list[NeuralOperator], inputs: torch.Tensor, num_realizations: int,
        param_batch: Optional[NDArray[np.float64]] = None
    ) -> torch.Tensor:
        """
        Run inference for a batch of operators.

        Args:
            operators: List of operators
            inputs: Input fields [B, C_in, H, W]
            num_realizations: Number of stochastic realizations
            param_batch: Parameter values [B, P] (needed for temporal mode to extract evolution params)

        Returns:
            Outputs [B, M, C_out, H, W] (snapshot mode, T=1)
                 or [B, M, T, C_out, H, W] (temporal mode, T>1)
        """
        num_timesteps = self.config.simulation.num_timesteps

        # Temporal mode: Use OperatorRollout for T>1
        if num_timesteps > 1:
            from ..rollout import OperatorRollout

            if param_batch is None:
                raise ValueError("param_batch required for temporal mode to extract evolution parameters")

            use_amp = self.config.simulation.precision in ["float16", "bfloat16"]

            # Performance instrumentation (first batch only)
            if not hasattr(self, '_temporal_batch_count'):
                self._temporal_batch_count = 0

            if self._temporal_batch_count == 0:
                t_start = time.time()

            # ============================================================================
            # Phase 1: Group operators by policy tuple
            # ============================================================================
            # Rationale: Operators are heterogeneous (cannot batch forward passes),
            # but can share OperatorRollout instances and cached policies within
            # policy groups. This eliminates repeated instantiation overhead.
            #
            # For 10K operators with 2 policy types:
            # - Before: 10K policy instantiations (~600s) + 10K rollout instantiations (~400s)
            # - After: 2 policy instantiations (~120ms) + 2 rollout instantiations (~80ms)
            # - Savings: ~1000s (~17 minutes, ~1.5% of 19h total)
            # ============================================================================

            policy_groups = defaultdict(list)  # {(policy_type, dt, alpha): [(idx, op, input), ...]}

            for i, operator in enumerate(operators):
                # Extract evolution parameters from sampled params
                param_dict = self._map_single_parameter_set(param_batch[i])

                # Group by policy tuple
                policy_tuple = (
                    param_dict.get("update_policy", "residual"),
                    param_dict.get("dt", 0.01),
                    param_dict.get("alpha", 0.5),
                )

                # Store with original index for order preservation
                policy_groups[policy_tuple].append((i, operator, inputs[i]))

            if self._temporal_batch_count == 0:
                t_group = time.time()

            # Pre-allocate tensor directly instead of list to avoid torch.stack memory duplication
            # Shape: [B, M, T, C_out, H, W]
            B = len(operators)
            M = num_realizations
            T = num_timesteps
            C = inputs.shape[1]  # Assuming C_out == C_in
            H, W = inputs.shape[-2], inputs.shape[-1]
            all_outputs = torch.zeros(B, M, T, C, H, W, dtype=torch.float32, device=self.device)

            # ============================================================================
            # Phase 2: Process each policy group with shared rollout instance
            # ============================================================================
            with torch.no_grad():
                for policy_tuple, group_items in policy_groups.items():
                    policy_type, dt, alpha = policy_tuple

                    # Get cached policy (or create if first time)
                    policy = self._get_or_create_policy(policy_type, dt, alpha)

                    # Create ONE rollout instance for this entire policy group
                    # This amortizes instantiation cost across all operators in group
                    rollout = OperatorRollout(
                        policy=policy,
                        num_timesteps=num_timesteps,
                        device=self.device,
                        compute_metrics=False,  # Skip expensive metrics during bulk generation
                    )

                    # Parallel processing with CUDA streams (if available and batch size allows)
                    num_operators = len(group_items)
                    use_parallel = (
                        self.device.type == "cuda" and
                        num_operators > 1 and
                        torch.cuda.is_available()
                    )
                    
                    if use_parallel:
                        # Adaptive stream count based on batch size and available memory
                        max_streams = min(4, num_operators)  # Cap at 4 streams
                        streams = [torch.cuda.Stream() for _ in range(max_streams)]
                        
                        # Process operators across streams
                        for stream_idx, (op_idx, operator, input_i) in enumerate(group_items):
                            stream = streams[stream_idx % max_streams]
                            
                            with torch.cuda.stream(stream):
                                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                                    trajectories, _, _ = rollout.evolve_operator(
                                        operator=operator.model,
                                        initial_condition=input_i,
                                        num_realizations=num_realizations,
                                        base_seed=op_idx,
                                    )
                                
                                # Store at original index (synchronized access is safe here)
                                all_outputs[op_idx] = trajectories
                                del trajectories
                        
                        # Synchronize all streams before moving to next policy group
                        for stream in streams:
                            stream.synchronize()
                        
                        # Periodic GPU cache cleanup
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        # Sequential processing (fallback for CPU or small batches)
                        for op_idx, operator, input_i in group_items:
                            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                                trajectories, _, _ = rollout.evolve_operator(
                                    operator=operator.model,
                                    initial_condition=input_i,
                                    num_realizations=num_realizations,
                                    base_seed=op_idx,
                                )
                            
                            all_outputs[op_idx] = trajectories
                            del trajectories
                            
                            # Periodic GPU cache cleanup every 10 operators
                            if (op_idx + 1) % 10 == 0 and self.device.type == "cuda":
                                torch.cuda.empty_cache()

                    # Free rollout after processing this policy group
                    del rollout

            # Performance logging (first batch only) - BEFORE deleting policy_groups
            if self._temporal_batch_count == 0:
                t_end = time.time()
                print(f"\n[Temporal Rollout - First Batch]")
                print(f"  Operators: {len(operators)}")
                print(f"  Policy groups: {len(policy_groups)}")
                print(f"  Group sizes: {[len(items) for items in policy_groups.values()]}")
                print(f"  Grouping time: {(t_group - t_start)*1000:.1f}ms")
                print(f"  Processing time: {(t_end - t_group):.2f}s")
                print(f"  Time per operator: {(t_end - t_group)/len(operators):.3f}s")
                print(f"  Cached policies: {len(self._policy_cache)}\n")

            self._temporal_batch_count += 1

            # Free policy_groups dict after all processing (AFTER logging)
            del policy_groups
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            # all_outputs already has shape [B, M, T, C_out, H, W] (no stack needed)
            return all_outputs

        # Snapshot mode: Original behavior for T=1
        else:
            # Pre-allocate tensor directly to avoid torch.stack memory duplication
            # Shape: [B, M, C_out, H, W]
            B = len(operators)
            M = num_realizations
            C = inputs.shape[1]  # Assuming C_out == C_in
            H, W = inputs.shape[-2], inputs.shape[-1]
            all_outputs = torch.zeros(B, M, C, H, W, dtype=torch.float32, device=self.device)

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
                    # We want [M, C_out, H, W] - direct assignment instead of append
                    all_outputs[i] = realizations[0]

                    # Free realizations immediately after storing
                    del realizations, input_i

                    # Periodic GPU cache cleanup every 10 operators
                    if (i + 1) % 10 == 0 and self.device.type == "cuda":
                        torch.cuda.empty_cache()

            # all_outputs already has shape [B, M, C_out, H, W] (no stack needed)
            return all_outputs

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

            # Set seed for deterministic operator initialization
            # Note: Global seed derived from batch position
            # For deterministic replay: torch.manual_seed(op_global_idx)
            torch.manual_seed(hash(str(params)) % (2**31))  # Deterministic per params
            
            # Build operator
            model = self.operator_builder.build_simple_cnn(param_dict)
            operator = NeuralOperator(model)

            # Prepare for inference
            operator = MemoryManager.optimize_for_inference(operator)
            operator = operator.to(self.device)

            # torch.compile() disabled to reduce GPU memory pressure during inline feature extraction
            # Enables batch_size=2 instead of batch_size=1 (saves ~30% wall time)
            # operator.enable_compile(mode="reduce-overhead")

            operators.append(operator)

        # Generate input fields with this grid size
        input_generator = InputFieldGenerator(
            grid_size=grid_size, num_channels=fixed_input_channels, device=self.device
        )

        gen_start = time.time()

        # OPTIMIZATION: Vectorized input generation - group by IC type
        ic_method = self.config.simulation.input_generation.method

        # Sample IC types for entire batch first
        if ic_method == "sampled":
            ic_types_used = [self._sample_ic_type() for _ in range(batch_size)]
        else:
            ic_types_used = [ic_method] * batch_size

        # Group indices by IC type for batch generation
        from collections import defaultdict
        ic_type_to_indices = defaultdict(list)
        for i, ic_type in enumerate(ic_types_used):
            ic_type_to_indices[ic_type].append(i)

        # Generate batches per IC type (amortize GPU kernel launches)
        all_inputs = []
        for ic_type, indices in ic_type_to_indices.items():
            ic_params = self._get_ic_params(ic_type)

            # Get base IC type for generator (strip aliases like _v0, _low, etc.)
            base_ic_type = self._get_base_ic_type(ic_type)

            # OPTIMIZATION: Generate entire batch for this IC type at once
            batch_inputs = input_generator.generate_batch(
                batch_size=len(indices),
                field_type=base_ic_type,
                **ic_params,
            )
            all_inputs.append((indices, batch_inputs))

        # Pre-allocate and fill output tensor
        if all_inputs:
            # Get shape from first batch
            first_batch = all_inputs[0][1]
            inputs = torch.zeros(
                (batch_size, *first_batch.shape[1:]),
                dtype=first_batch.dtype,
                device=first_batch.device,
            )

            # Place batches in correct positions
            for indices, batch_inputs in all_inputs:
                for i, idx in enumerate(indices):
                    inputs[idx] = batch_inputs[i]

            # Free all_inputs list after copying to prevent accumulation
            for indices, batch_inputs in all_inputs:
                del batch_inputs
            del all_inputs
        else:
            # Fallback (should never happen)
            raise RuntimeError("No inputs generated")
        self.stats["generation_time"] += time.time() - gen_start

        # Run inference
        inf_start = time.time()
        outputs = self._run_inference_batch(
            operators, inputs, self.config.simulation.num_realizations, param_batch
        )
        self.stats["inference_time"] += time.time() - inf_start

        # Explicitly delete operators to break reference cycles and free GPU memory
        for op in operators:
            del op
        del operators
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection

        # OPTIMIZATION: Skip padding for single grid size datasets
        # (only pad if variable grid sizes exist and grid_size < max_grid_size)
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

    def _get_base_ic_type(self, ic_type: str) -> str:
        """
        Get base IC type from alias (e.g., gaussian_random_field_v0 → gaussian_random_field).

        Args:
            ic_type: IC type name (possibly with alias suffix)

        Returns:
            Base IC type name
        """
        import re
        # Strip common alias patterns: _v[0-9], _low, _mid, _high
        base_type = re.sub(r'_(v\d+|low|mid|high)$', '', ic_type)
        return base_type

    def _get_ic_params(self, ic_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific IC type from config.

        Supports IC type aliases (e.g., gaussian_random_field_v0).
        If alias exists in config, uses those parameters.
        Otherwise, falls back to base IC type parameters.

        Args:
            ic_type: IC type name (possibly with alias suffix)

        Returns:
            Dict of parameters for this IC type
        """
        config_gen = self.config.simulation.input_generation

        # First, try to get parameters directly from the alias name
        # This allows configs to specify gaussian_random_field_v0, multiscale_grf_low, etc.
        ic_type_attr = ic_type.replace('-', '_')  # Handle kebab-case
        if hasattr(config_gen, ic_type_attr):
            params = getattr(config_gen, ic_type_attr)
            if params is not None and isinstance(params, dict):
                return params.copy()

        # If no alias-specific config found, check base IC type
        # (This handles the case where the config uses aliases but we need base parameters)
        base_ic_type = self._get_base_ic_type(ic_type)

        # Check if base IC type has specific config
        if base_ic_type == "multiscale_grf" and hasattr(config_gen, 'multiscale_grf') and config_gen.multiscale_grf:
            return config_gen.multiscale_grf.copy()
        elif base_ic_type == "localized" and hasattr(config_gen, 'localized') and config_gen.localized:
            return config_gen.localized.copy()
        elif base_ic_type == "structured" and hasattr(config_gen, 'structured') and config_gen.structured:
            return config_gen.structured.copy()
        elif base_ic_type == "composite" and hasattr(config_gen, 'composite') and config_gen.composite:
            return config_gen.composite.copy()
        elif base_ic_type == "heavy_tailed" and hasattr(config_gen, 'heavy_tailed') and config_gen.heavy_tailed:
            return config_gen.heavy_tailed.copy()
        # Tier 1 domain-specific ICs
        elif base_ic_type == "quantum_wave_packet" and hasattr(config_gen, 'quantum_wave_packet') and config_gen.quantum_wave_packet:
            return config_gen.quantum_wave_packet.copy()
        elif base_ic_type == "turing_pattern" and hasattr(config_gen, 'turing_pattern') and config_gen.turing_pattern:
            return config_gen.turing_pattern.copy()
        elif base_ic_type == "thermal_gradient" and hasattr(config_gen, 'thermal_gradient') and config_gen.thermal_gradient:
            return config_gen.thermal_gradient.copy()
        elif base_ic_type == "morphogen_gradient" and hasattr(config_gen, 'morphogen_gradient') and config_gen.morphogen_gradient:
            return config_gen.morphogen_gradient.copy()
        elif base_ic_type == "reaction_front" and hasattr(config_gen, 'reaction_front') and config_gen.reaction_front:
            return config_gen.reaction_front.copy()
        # Tier 2 domain-specific ICs
        elif base_ic_type == "light_cone" and hasattr(config_gen, 'light_cone') and config_gen.light_cone:
            return config_gen.light_cone.copy()
        elif base_ic_type == "critical_fluctuation" and hasattr(config_gen, 'critical_fluctuation') and config_gen.critical_fluctuation:
            return config_gen.critical_fluctuation.copy()
        elif base_ic_type == "phase_boundary" and hasattr(config_gen, 'phase_boundary') and config_gen.phase_boundary:
            return config_gen.phase_boundary.copy()
        elif base_ic_type == "bz_reaction" and hasattr(config_gen, 'bz_reaction') and config_gen.bz_reaction:
            return config_gen.bz_reaction.copy()
        elif base_ic_type == "shannon_entropy" and hasattr(config_gen, 'shannon_entropy') and config_gen.shannon_entropy:
            return config_gen.shannon_entropy.copy()
        # Tier 3 domain-specific ICs
        elif base_ic_type == "interference_pattern" and hasattr(config_gen, 'interference_pattern') and config_gen.interference_pattern:
            return config_gen.interference_pattern.copy()
        elif base_ic_type == "cell_population" and hasattr(config_gen, 'cell_population') and config_gen.cell_population:
            return config_gen.cell_population.copy()
        elif base_ic_type == "chromatin_domain" and hasattr(config_gen, 'chromatin_domain') and config_gen.chromatin_domain:
            return config_gen.chromatin_domain.copy()
        elif base_ic_type == "shock_front" and hasattr(config_gen, 'shock_front') and config_gen.shock_front:
            return config_gen.shock_front.copy()
        elif base_ic_type == "gene_expression" and hasattr(config_gen, 'gene_expression') and config_gen.gene_expression:
            return config_gen.gene_expression.copy()
        # Tier 4 research frontiers ICs
        elif base_ic_type == "coherent_state" and hasattr(config_gen, 'coherent_state') and config_gen.coherent_state:
            return config_gen.coherent_state.copy()
        elif base_ic_type == "relativistic_wave_packet" and hasattr(config_gen, 'relativistic_wave_packet') and config_gen.relativistic_wave_packet:
            return config_gen.relativistic_wave_packet.copy()
        elif base_ic_type == "mutual_information" and hasattr(config_gen, 'mutual_information') and config_gen.mutual_information:
            return config_gen.mutual_information.copy()
        elif base_ic_type == "regulatory_network" and hasattr(config_gen, 'regulatory_network') and config_gen.regulatory_network:
            return config_gen.regulatory_network.copy()
        elif base_ic_type == "dla_cluster" and hasattr(config_gen, 'dla_cluster') and config_gen.dla_cluster:
            return config_gen.dla_cluster.copy()
        elif base_ic_type == "error_correcting_code" and hasattr(config_gen, 'error_correcting_code') and config_gen.error_correcting_code:
            return config_gen.error_correcting_code.copy()
        elif base_ic_type == "gaussian_random_field":
            # Gaussian random field uses default length_scale and variance
            return {
                "length_scale": config_gen.length_scale,
                "variance": config_gen.variance
            }
        else:
            # For unknown types, return empty dict (let generator use defaults)
            # This prevents passing incompatible parameters like length_scale to structured, etc.
            return {}

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


    def _extract_and_write_features_inline(
        self,
        batch_outputs: torch.Tensor,  # [B, M, T, C, H, W]
        batch_idx_start: int,
        feature_pipeline: FeatureExtractionPipeline,
        async_writer: Optional[AsyncFeatureWriter]
    ) -> None:
        """
        Extract SDF features inline during generation (GPU-optimized).
        
        Uses optimized extraction pipeline and async writing for pipelining.
        
        Args:
            batch_outputs: Trajectory tensors [B, M, T, C, H, W]
            batch_idx_start: Starting index for this batch in dataset
            feature_pipeline: FeatureExtractionPipeline instance
            async_writer: AsyncFeatureWriter for non-blocking writes (None = blocking)
        """
        # Extract features efficiently (no recomputation)
        feat_start = time.time()
        per_timestep_np, per_trajectory_np, aggregated_np = feature_pipeline.extract_all(
            batch_outputs
        )
        self.stats["feature_extraction_time"] += time.time() - feat_start
        
        # Write asynchronously (non-blocking) or synchronously
        if async_writer is not None:
            task = FeatureWriteTask(
                batch_idx_start=batch_idx_start,
                per_timestep=per_timestep_np,
                per_trajectory=per_trajectory_np,
                aggregated=aggregated_np
            )
            async_writer.enqueue(task)
        else:
            # Fallback: synchronous write (for backward compatibility)
            self._feature_writer.write_sdf_batch(
                batch_idx=batch_idx_start,
                per_timestep=per_timestep_np,
                per_trajectory=per_trajectory_np,
                aggregated=aggregated_np
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
        # Extract features inline if enabled (GPU-optimized)
        if self._feature_pipeline is not None:
            self._extract_and_write_features_inline(
                batch_outputs=batch_outputs,
                batch_idx_start=batch_indices[0],  # First index in batch
                feature_pipeline=self._feature_pipeline,
                async_writer=self._async_feature_writer
            )

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
        if self.stats['feature_extraction_time'] > 0:
            print(
                f"  Feature extraction: {self.stats['feature_extraction_time']:.2f}s "
                f"({self.stats['feature_extraction_time']/total_time*100:.1f}%)"
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

    def cleanup(self) -> None:
        """
        Explicit cleanup for long-running pipelines.

        Clears cached policies and forces garbage collection to free memory.
        Called automatically after dataset generation completes.
        """
        # Clear policy cache
        for key in list(self._policy_cache.keys()):
            del self._policy_cache[key]
        self._policy_cache.clear()

        # Force GPU memory cleanup if CUDA available
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()
