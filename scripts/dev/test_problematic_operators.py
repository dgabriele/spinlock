"""Test each problematic parameter vector individually to isolate CUDA hang cause.

Uses subprocess with timeout to safely test operators that may hang.
"""

import h5py
import numpy as np
import subprocess
import sys
import time
from pathlib import Path

def test_single_operator(sample_idx, timeout=30):
    """Test a single operator in a subprocess with timeout.

    Args:
        sample_idx: Index of sample to test
        timeout: Timeout in seconds

    Returns:
        ("success", None) or ("timeout", None) or ("error", error_message)
    """
    test_script = f"""
import sys
import h5py
import torch
import numpy as np
from spinlock.noa.cno_replay import CNOReplayer

sample_idx = {sample_idx}

# Load single sample
with h5py.File("datasets/100k_full_features.h5", "r") as f:
    params = f["parameters"]["params"][sample_idx:sample_idx + 1]
    ic = f["inputs"]["fields"][sample_idx:sample_idx + 1]

# Create replayer
replayer = CNOReplayer.from_config(
    "configs/experiments/local_100k_optimized.yaml",
    device="cuda",
    cache_size=1
)

# Convert to tensors
ic_tensor = torch.from_numpy(ic).to("cuda")
params_np = params[0]

print(f"Testing sample {{sample_idx}}...", flush=True)
print(f"  params: {{params_np}}", flush=True)

try:
    # Attempt rollout
    trajectory = replayer.rollout(
        params_vector=params_np,
        ic=ic_tensor,
        timesteps=32,
        num_realizations=1,
        return_all_steps=True,
    )
    print(f"  ✓ SUCCESS: Generated trajectory with shape {{trajectory.shape}}", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"  ✗ ERROR: {{e}}", flush=True)
    sys.exit(1)
"""

    # Write test script to temp file
    temp_script = Path(f"/tmp/test_operator_{sample_idx}.py")
    temp_script.write_text(test_script)

    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            ["poetry", "run", "python", str(temp_script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/home/daniel/projects/spinlock"
        )

        if result.returncode == 0:
            return ("success", result.stdout)
        else:
            return ("error", result.stderr)

    except subprocess.TimeoutExpired:
        return ("timeout", None)
    # Don't delete temp script for debugging
    # finally:
    #     temp_script.unlink(missing_ok=True)

def main():
    print("=" * 60)
    print("TESTING PROBLEMATIC OPERATORS")
    print("=" * 60)
    print("\nTesting each operator individually with 30s timeout...")
    print("This will identify which specific parameter vector causes CUDA hang.\n")

    # Test batches 83-85 (samples 332-343)
    batch_size = 4
    problematic_batches = [83, 84, 85]

    all_results = {}

    for batch_idx in problematic_batches:
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        print(f"\nBatch {batch_idx} (samples {start_idx}-{end_idx - 1}):")
        print("-" * 60)

        for sample_idx in range(start_idx, end_idx):
            print(f"\n  Sample {sample_idx}:", flush=True)

            start_time = time.time()
            status, output = test_single_operator(sample_idx, timeout=30)
            elapsed = time.time() - start_time

            all_results[sample_idx] = {
                "batch": batch_idx,
                "status": status,
                "elapsed": elapsed,
                "output": output
            }

            if status == "success":
                print(f"    ✓ SUCCESS ({elapsed:.1f}s)")
            elif status == "timeout":
                print(f"    ⚠ TIMEOUT after {elapsed:.1f}s - THIS IS THE PROBLEMATIC ONE!")
            else:
                print(f"    ✗ ERROR ({elapsed:.1f}s)")
                if output:
                    print(f"    Error: {output[:500]}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successes = [idx for idx, r in all_results.items() if r["status"] == "success"]
    timeouts = [idx for idx, r in all_results.items() if r["status"] == "timeout"]
    errors = [idx for idx, r in all_results.items() if r["status"] == "error"]

    print(f"\nSuccessful: {len(successes)} samples")
    if successes:
        print(f"  Indices: {successes}")

    print(f"\nTimeouts (CUDA hang): {len(timeouts)} samples")
    if timeouts:
        print(f"  Indices: {timeouts}")
        print("\n  ⚠ These parameter vectors cause CUDA kernel deadlock!")

    print(f"\nErrors: {len(errors)} samples")
    if errors:
        print(f"  Indices: {errors}")

    # If we found timeout cases, extract their parameters
    if timeouts:
        print("\n" + "=" * 60)
        print("PROBLEMATIC PARAMETER VECTORS")
        print("=" * 60)

        with h5py.File("datasets/100k_full_features.h5", "r") as f:
            params_all = f["parameters"]["params"]

            for idx in timeouts:
                params = params_all[idx]
                print(f"\nSample {idx}:")
                print(f"  Raw parameters: {params}")

                # Reconstruct operator config
                from spinlock.operators.builder import OperatorBuilder
                import yaml

                with open("configs/experiments/local_100k_optimized.yaml") as cfg_file:
                    config = yaml.safe_load(cfg_file)

                parameter_space = config["parameter_space"]
                flat_param_spec = {}
                for category in ["architecture", "stochastic", "operator", "evolution"]:
                    if category in parameter_space:
                        for name, spec in parameter_space[category].items():
                            flat_param_spec[f"{category}.{name}"] = spec

                builder = OperatorBuilder()
                mapped = builder.map_parameters(params, flat_param_spec)

                reconstructed = {"input_channels": 1, "output_channels": 1}
                for flat_name, value in mapped.items():
                    name = flat_name.split(".")[-1]
                    reconstructed[name] = value

                print(f"  Reconstructed config:")
                for key, value in reconstructed.items():
                    print(f"    {key}: {value}")

        # Next steps
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("""
Now that we've identified the problematic parameter vector(s), we need to:

1. Analyze the operator configuration that causes the hang
2. Identify the root cause (e.g., specific layer config, memory issue, etc.)
3. Add validation to reject these configurations during dataset generation
4. Or fix the operator construction to handle these edge cases
5. Or add timeout/watchdog mechanism in CNO replayer

Common causes of CUDA hangs:
- Very large memory allocations (large kernels × many channels × deep networks)
- Specific kernel sizes that trigger CUDA bugs
- Interaction between specific hyperparameters
- Numerical instability in operator construction
        """)

if __name__ == "__main__":
    main()
