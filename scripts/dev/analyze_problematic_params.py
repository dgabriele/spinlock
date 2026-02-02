"""Extract and analyze parameter vectors from batches that cause CUDA hangs."""

import h5py
import numpy as np
import torch
from pathlib import Path
import yaml

def load_dataset_params(dataset_path, n_samples=1000):
    """Load parameter vectors from dataset."""
    with h5py.File(dataset_path, "r") as f:
        params = f["parameters"]["params"][:n_samples]
        ic = f["inputs"]["fields"][:n_samples]
    return params, ic

def analyze_batch_params(params, batch_indices, batch_size=4):
    """Analyze parameter vectors from specific batches."""
    results = {}

    for batch_idx in batch_indices:
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_params = params[start_idx:end_idx]

        results[batch_idx] = {
            "indices": list(range(start_idx, end_idx)),
            "params": batch_params,
            "stats": {
                "mean": np.mean(batch_params, axis=0).tolist(),
                "std": np.std(batch_params, axis=0).tolist(),
                "min": np.min(batch_params, axis=0).tolist(),
                "max": np.max(batch_params, axis=0).tolist(),
            }
        }

    return results

def reconstruct_operator_params(unit_params, config_path):
    """Reconstruct operator parameters from Sobol unit params."""
    from spinlock.operators.builder import OperatorBuilder

    # Load parameter space from config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    parameter_space = config["parameter_space"]

    # Flatten parameter space (same as CNOReplayer)
    flat_param_spec = {}
    for category in ["architecture", "stochastic", "operator", "evolution"]:
        if category in parameter_space:
            for name, spec in parameter_space[category].items():
                flat_param_spec[f"{category}.{name}"] = spec

    # Map parameters
    builder = OperatorBuilder()
    mapped = builder.map_parameters(unit_params, flat_param_spec)

    # Reconstruct nested structure
    params = {
        "input_channels": 1,
        "output_channels": 1,
    }

    for flat_name, value in mapped.items():
        name = flat_name.split(".")[-1]
        params[name] = value

    return params

def check_for_anomalies(batch_results):
    """Check for potential issues in parameter vectors."""
    issues = []

    for batch_idx, data in batch_results.items():
        params = data["params"]
        stats = data["stats"]

        # Check for extreme values
        if np.any(np.array(stats["min"]) < 0.01):
            issues.append(f"Batch {batch_idx}: Very low parameter values detected")

        if np.any(np.array(stats["max"]) > 0.99):
            issues.append(f"Batch {batch_idx}: Very high parameter values detected")

        # Check for near-zero variance
        if np.any(np.array(stats["std"]) < 0.001):
            issues.append(f"Batch {batch_idx}: Near-zero variance in parameters")

        # Check parameter ranges
        param_ranges = np.array(stats["max"]) - np.array(stats["min"])
        if np.any(param_ranges > 0.8):
            issues.append(f"Batch {batch_idx}: Large parameter range variation")

    return issues

def main():
    dataset_path = "datasets/100k_full_features.h5"
    config_path = "configs/experiments/local_100k_optimized.yaml"
    problematic_batches = [83, 84, 85]
    batch_size = 4

    print("=" * 60)
    print("ANALYZING PROBLEMATIC PARAMETER VECTORS")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading parameters from {dataset_path}...")
    params, ic = load_dataset_params(dataset_path, n_samples=1000)
    print(f"  Loaded {len(params)} samples")
    print(f"  Parameter shape: {params.shape}")

    # Analyze problematic batches
    print(f"\nAnalyzing batches {problematic_batches}...")
    batch_results = analyze_batch_params(params, problematic_batches, batch_size)

    # Check for anomalies
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION")
    print("=" * 60)
    issues = check_for_anomalies(batch_results)

    if issues:
        print("\nPotential issues found:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ No obvious anomalies detected in parameter ranges")

    # Reconstruct operator parameters for each sample
    print("\n" + "=" * 60)
    print("RECONSTRUCTED OPERATOR PARAMETERS")
    print("=" * 60)

    for batch_idx in problematic_batches:
        print(f"\nBatch {batch_idx}:")
        batch_params = batch_results[batch_idx]["params"]

        for i, sample_params in enumerate(batch_params):
            sample_idx = batch_idx * batch_size + i
            print(f"\n  Sample {sample_idx} (index in batch: {i}):")
            print(f"    Raw Sobol params: {sample_params}")

            try:
                operator_params = reconstruct_operator_params(sample_params, config_path)
                print(f"    Reconstructed operator config:")
                for key, value in operator_params.items():
                    print(f"      {key}: {value}")

                # Check for potentially problematic configurations
                if "num_layers" in operator_params:
                    if operator_params["num_layers"] > 8:
                        print(f"      ⚠ WARNING: Very deep network ({operator_params['num_layers']} layers)")

                if "channels" in operator_params:
                    if operator_params["channels"] > 128:
                        print(f"      ⚠ WARNING: Very wide network ({operator_params['channels']} channels)")

                if "kernel_size" in operator_params:
                    if operator_params["kernel_size"] > 7:
                        print(f"      ⚠ WARNING: Large kernel size ({operator_params['kernel_size']})")

            except Exception as e:
                print(f"      ✗ Failed to reconstruct: {e}")

    # Save results
    output_file = Path("analysis_problematic_params.npz")
    np.savez(
        output_file,
        **{f"batch_{idx}_params": batch_results[idx]["params"]
           for idx in problematic_batches}
    )
    print(f"\n\nParameter vectors saved to: {output_file}")

    # Suggest next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Test each parameter vector individually with CNO replayer
2. Identify which specific operator configuration causes CUDA hang
3. Add parameter validation to reject problematic configurations
4. Implement timeout mechanism for operator construction/rollout
    """)

if __name__ == "__main__":
    main()
