"""
Compute Ground-Truth Tokens command for Spinlock CLI.

Generates ground-truth VQ tokens for token-conditioned NOA training by:
1. Loading dataset (initial conditions + parameters)
2. Generating ground-truth trajectories using CNO replayer
3. Extracting features from trajectories
4. Computing VQ tokens via frozen VQ-VAE encoder
5. Saving tokens to HDF5 for training

Prerequisites for token-conditioned NOA training:
    1. Generated dataset (spinlock generate)
    2. Trained VQ-VAE (spinlock train-vqvae)
    3. Ground-Truth tokens (this command)

Documentation:
    - Token conditioning: docs/two-stage-curriculum-architecture.md
    - VQ-VAE checkpoint format: docs/vqvae/checkpoint-format.md
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import h5py
import torch
from tqdm import tqdm

from .base import CLICommand


class ComputeGroundTruthTokensCommand(CLICommand):
    """
    Command to precompute ground-truth VQ tokens for token-conditioned NOA training.

    Given a dataset of CNO parameters and initial conditions, generates ground-truth
    trajectories and extracts VQ tokens for each sample.
    """

    @property
    def name(self) -> str:
        return "compute-ground-truth-tokens"

    @property
    def help(self) -> str:
        return "Precompute ground-truth VQ tokens for token-conditioned NOA training"

    @property
    def description(self) -> str:
        return """
Precompute ground-truth VQ tokens for token-conditioned NOA training.

This command generates ground-truth VQ tokens for each sample in the dataset by:
1. Loading the CNO replayer and VQ-VAE encoder
2. Generating ground-truth trajectories for each (ic, params) pair
3. Extracting multi-modal features from trajectories
4. Computing discrete VQ tokens via VQ-VAE encoder
5. Saving tokens to HDF5 file for efficient loading during training

Prerequisites:
  1. Generated dataset containing initial conditions and parameters
  2. Trained VQ-VAE checkpoint on trajectory features

Output:
  HDF5 file containing:
    - tokens: [N, num_tokens] discrete token indices
    - Metadata: num_samples, num_tokens, vqvae_checkpoint, dataset_source

Examples:
  # Generate ground-truth tokens for 1000 samples
  spinlock compute-ground-truth-tokens \\
      --dataset datasets/100k_full_features.h5 \\
      --cno-config configs/experiments/local_100k_optimized.yaml \\
      --vqvae-checkpoint checkpoints/production/100k_3family_v1/best_model.pt \\
      --output datasets/100k_ground-truth_tokens_1k.h5 \\
      --n-samples 1000

  # Process all samples in dataset
  spinlock compute-ground-truth-tokens \\
      --dataset datasets/100k_full_features.h5 \\
      --cno-config configs/experiments/local_100k_optimized.yaml \\
      --vqvae-checkpoint checkpoints/production/100k_3family_v1/best_model.pt \\
      --output datasets/100k_ground-truth_tokens_full.h5
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add compute-ground-truth-tokens command arguments."""
        # Required arguments
        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to input dataset HDF5 file",
        )

        parser.add_argument(
            "--cno-config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to CNO replayer configuration YAML",
        )

        parser.add_argument(
            "--vqvae-checkpoint",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to trained VQ-VAE checkpoint",
        )

        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="PATH",
            help="Output path for ground-truth tokens HDF5 file",
        )

        # Optional arguments
        parser.add_argument(
            "--n-samples",
            type=int,
            default=None,
            metavar="N",
            help="Number of samples to process (default: all)",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="Batch size for processing (default: 16)",
        )

        parser.add_argument(
            "--timesteps",
            type=int,
            default=32,
            metavar="T",
            help="Number of timesteps for trajectory generation (default: 32)",
        )

        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "cpu"],
            help="Device to use for computation (default: cuda)",
        )

    def execute(self, args: Namespace) -> int:
        """Execute ground-truth token computation."""
        from spinlock.noa import CNOReplayer, AlignedFeatureExtractor
        from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE
        from spinlock.encoding import CategoricalVQVAEConfig

        # Determine device
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load dataset
        print(f"Loading dataset: {args.dataset}")
        if not args.dataset.exists():
            return self.error(f"Dataset not found: {args.dataset}")

        dataset = h5py.File(args.dataset, 'r')

        # Determine dataset structure
        if "inputs" in dataset and "fields" in dataset["inputs"]:
            ic_key = "inputs/fields"
            params_key = "parameters/params"
        elif "initial_conditions" in dataset:
            ic_key = "initial_conditions"
            params_key = "parameters"
        else:
            available_keys = list(dataset.keys())
            dataset.close()
            return self.error(f"Unknown dataset format. Available keys: {available_keys}")

        n_samples_total = len(dataset[ic_key])
        n_samples = args.n_samples if args.n_samples is not None else n_samples_total
        n_samples = min(n_samples, n_samples_total)
        print(f"  {n_samples} samples to process (out of {n_samples_total} total)")

        # Load CNO replayer
        print(f"Loading CNO replayer: {args.cno_config}")
        if not args.cno_config.exists():
            dataset.close()
            return self.error(f"CNO config not found: {args.cno_config}")

        replayer = CNOReplayer.from_config(
            config_path=str(args.cno_config),
            device=str(device),
            cache_size=8,
        )

        # Load VQ-VAE
        print(f"Loading VQ-VAE: {args.vqvae_checkpoint}")
        if not args.vqvae_checkpoint.exists():
            dataset.close()
            return self.error(f"VQ-VAE checkpoint not found: {args.vqvae_checkpoint}")

        try:
            checkpoint = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
            config = checkpoint["config"]
            model_config = checkpoint.get("model_config", config)
            state_dict = checkpoint["model_state_dict"]

            # Handle torch.compile prefix
            has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
            if has_orig_mod:
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            # Check if wrapped
            is_wrapped = any(k.startswith("initial_encoder.") for k in state_dict.keys())

            # Create VQ-VAE config
            vqvae_config = CategoricalVQVAEConfig(
                input_dim=model_config["input_dim"],
                group_indices=model_config["group_indices"],
                group_embedding_dim=model_config.get("group_embedding_dim", config.get("group_embedding_dim")),
                group_hidden_dim=model_config.get("group_hidden_dim", config.get("group_hidden_dim")),
                levels=model_config.get("levels", config.get("levels")),
            )

            if is_wrapped:
                vqvae_state_dict = {
                    k.replace("vqvae.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("vqvae.")
                }
                vqvae = CategoricalHierarchicalVQVAE(vqvae_config)
                vqvae.load_state_dict(vqvae_state_dict)
            else:
                vqvae = CategoricalHierarchicalVQVAE(vqvae_config)
                vqvae.load_state_dict(state_dict)

            vqvae = vqvae.to(device)
            vqvae.eval()

            print(f"  ✓ VQ-VAE loaded (input_dim={model_config['input_dim']})")

        except Exception as e:
            dataset.close()
            return self.error(f"Failed to load VQ-VAE: {e}")

        # Load feature extractor
        print(f"Loading feature extractor from VQ-VAE checkpoint")
        try:
            feature_extractor = AlignedFeatureExtractor.from_checkpoint(
                str(args.vqvae_checkpoint),
                device=str(device)
            )
            print(f"  ✓ Feature extractor loaded")
        except Exception as e:
            dataset.close()
            return self.error(f"Failed to load feature extractor: {e}")

        # Determine token shape
        print("Determining token shape...")
        with torch.no_grad():
            ic_data = dataset[ic_key][0]
            if ic_data.ndim == 3:
                dummy_ic = torch.tensor(ic_data[0, :, :], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            else:
                dummy_ic = torch.tensor(ic_data, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            dummy_params_numpy = dataset[params_key][0]

            dummy_traj = replayer.rollout(
                params_vector=dummy_params_numpy,
                ic=dummy_ic,
                timesteps=args.timesteps,
                num_realizations=1,
            )

            dummy_features = feature_extractor(dummy_traj, ic=dummy_ic)
            dummy_tokens = vqvae.get_tokens(dummy_features)
            num_tokens = dummy_tokens.shape[1]

        print(f"Token shape: [{num_tokens}] per sample")

        # Prepare output file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = h5py.File(args.output, 'w')
        tokens_dataset = output_file.create_dataset(
            "tokens",
            shape=(n_samples, num_tokens),
            dtype='int32',
            compression='gzip',
        )

        # Process in batches
        print(f"Computing tokens for {n_samples} samples...")
        num_batches = (n_samples + args.batch_size - 1) // args.batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, n_samples)
                batch_size = end_idx - start_idx

                # Load batch
                ics = []
                params_list = []

                for i in range(start_idx, end_idx):
                    ic_data = dataset[ic_key][i]
                    if ic_data.ndim == 3:
                        ic = torch.tensor(ic_data[0, :, :], device=device, dtype=torch.float32).unsqueeze(0)
                    else:
                        ic = torch.tensor(ic_data, device=device, dtype=torch.float32).unsqueeze(0)
                    ics.append(ic)

                    params_numpy = dataset[params_key][i]
                    params_list.append(params_numpy)

                ics_batch = torch.stack(ics)  # [B, 1, H, W]

                # Generate trajectories for batch
                trajectories = []
                for i in range(batch_size):
                    traj = replayer.rollout(
                        params_vector=params_list[i],
                        ic=ics_batch[i:i+1],
                        timesteps=args.timesteps,
                        num_realizations=1,
                    )
                    trajectories.append(traj)

                trajectories_batch = torch.cat(trajectories, dim=0)  # [B, T, 1, H, W]

                # Extract features
                features = feature_extractor(trajectories_batch, ic=ics_batch)  # [B, feature_dim]

                # Get tokens
                tokens = vqvae.get_tokens(features)  # [B, num_tokens]

                # Save to HDF5
                tokens_dataset[start_idx:end_idx] = tokens.cpu().numpy()

        # Save metadata
        output_file.attrs['num_samples'] = n_samples
        output_file.attrs['num_tokens'] = num_tokens
        output_file.attrs['vqvae_checkpoint'] = str(args.vqvae_checkpoint)
        output_file.attrs['dataset_source'] = str(args.dataset)
        output_file.attrs['timesteps'] = args.timesteps

        output_file.close()
        dataset.close()

        print(f"\n✓ Tokens saved to: {args.output}")
        print(f"  Shape: [{n_samples}, {num_tokens}]")
        print(f"  Compression: gzip")

        return 0
