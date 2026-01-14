# Distributed Training Guide

This guide explains how to set up and run distributed training for MNO across multiple GPUs on different machines.

## Overview

Spinlock supports multi-node, multi-GPU distributed training using PyTorch's DistributedDataParallel (DDP). This enables:

- **Data parallelism**: Split batches across multiple GPUs
- **Gradient synchronization**: Automatic all-reduce of gradients
- **Linear speedup**: Near-linear scaling with number of GPUs
- **Remote GPUs**: SSH-based launching to remote machines

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Local Machine     │         │   Remote Machine    │
│   (Master Node)     │         │   (Worker Node)     │
│                     │         │                     │
│  GPU 0 (Rank 0)     │◄────────┤  GPU 0 (Rank 1)     │
│  - Process group    │  NCCL   │  - Process group    │
│  - Gradient sync    │         │  - Gradient sync    │
│  - Checkpointing    │         │                     │
└─────────────────────┘         └─────────────────────┘
         │                                 ▲
         │                                 │
         └────── SSH Launch ───────────────┘
```

## Prerequisites

### 1. SSH Access to Remote Machine

Ensure password-less SSH access to the remote machine:

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519

# Copy key to remote machine
ssh-copy-id daniel@100.111.3.32

# Test connection
ssh daniel@100.111.3.32 "echo 'SSH works!'"
```

### 2. Identical Environment on Remote Machine

The remote machine must have:
- Same Spinlock installation (same version)
- Same Python environment (via Poetry)
- Same dataset files (or accessible via shared filesystem)
- CUDA-enabled GPU(s)

```bash
# On remote machine
cd /path/to/spinlock
git pull  # Ensure code is up to date
poetry install  # Ensure environment matches
```

### 3. Network Configuration

- Firewall must allow TCP connections on `master_port` (default: 29500)
- Both machines must be able to reach each other via IP/hostname

## Configuration

### Config File Format

Add a `distributed` section to your training config:

```yaml
distributed:
  enabled: true
  backend: nccl  # Use 'nccl' for GPU, 'gloo' for CPU
  master_port: 29500  # Port for distributed coordination

  nodes:
    # Master node (must be localhost)
    - host: localhost
      gpus: [0]  # List of GPU indices to use

    # Remote nodes
    - host: 100.111.3.32  # IP or hostname
      gpus: [0]  # List of GPU indices to use
      ssh_user: daniel  # Optional: defaults to current user
      ssh_port: 22  # Optional: defaults to 22
      python_path: "poetry run python"  # Optional
      working_dir: "/home/daniel/projects/spinlock"  # Optional

# Rest of training config...
model:
  # ...

training:
  batch_size: 2  # Per-GPU batch size
  # Effective batch size = 2 * 2 GPUs = 4
  # ...
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable distributed training | `false` |
| `backend` | Communication backend (`nccl`, `gloo`, `mpi`) | `nccl` |
| `master_port` | Port for process coordination | `29500` |
| `nodes` | List of compute nodes | `[]` |
| `nodes[].host` | Hostname or IP (first must be `localhost`) | - |
| `nodes[].gpus` | List of GPU indices on this node | `[0]` |
| `nodes[].ssh_user` | SSH username | Current user |
| `nodes[].ssh_port` | SSH port | `22` |
| `nodes[].python_path` | Python executable path | `poetry run python` |
| `nodes[].working_dir` | Working directory on remote | Current directory |

### Batch Size Considerations

**Important**: The `batch_size` in your config is the **per-GPU batch size**.

- **Single GPU**: `batch_size: 4` → effective batch size = 4
- **2 GPUs (distributed)**: `batch_size: 2` → effective batch size = 2 × 2 = 4

To maintain the same effective batch size when going distributed, **divide your original batch size by the number of GPUs**.

## Usage

### Basic Example

```bash
# Run distributed training
poetry run spinlock train-meta-operator \
  --config configs/noa/experiments/phase2/exp_pure_mse_v4_10k_distributed.yaml
```

That's it! The launcher will:
1. Parse the distributed config
2. SSH to remote machines
3. Launch training processes on each GPU
4. Coordinate gradient synchronization
5. Handle cleanup on completion/interrupt

### Resuming from Checkpoint

```bash
# Resume works the same way
poetry run spinlock train-meta-operator \
  --config configs/noa/experiments/phase2/exp_pure_mse_v4_10k_distributed.yaml \
  --resume-from checkpoints/noa/pure_mse_v4_10k_distributed/meta_operator_best.pt
```

### Monitoring Progress

Only rank 0 (master) prints training logs. Remote workers run silently.

```
[Rank 0/2] Process group initialized on nccl backend
[Rank 0/2] Using GPU: cuda:0
Creating NOA backbone...
  ✓ NOA created (226,072,521 parameters)
  ✓ Model wrapped in DistributedDataParallel
...
  Batch 10/2250: total=0.588583, ic=0.3118, traj=0.4950, time=8.88s
```

### Stopping Training

Press `Ctrl+C` to stop. The launcher will:
1. Send termination signal to all processes
2. Clean up SSH connections
3. Close distributed process group

## Troubleshooting

### SSH Connection Issues

```
Error: ssh: connect to host 100.111.3.32 port 22: Connection refused
```

**Solution**: Ensure SSH is running on remote machine and firewall allows connections:

```bash
# On remote machine
sudo systemctl start sshd
sudo systemctl enable sshd

# Check firewall
sudo ufw allow 22/tcp
```

### NCCL Initialization Timeout

```
Error: NCCL initialization timeout after 1800 seconds
```

**Solution**: Check that master_port is not blocked by firewall:

```bash
# On master machine
sudo ufw allow 29500/tcp

# On remote machine
sudo ufw allow from <master-ip> to any port 29500
```

### Out of Memory on One GPU

```
Error: CUDA error: out of memory
```

**Solution**: Reduce per-GPU batch size in config:

```yaml
training:
  batch_size: 1  # Reduce from 2 to 1
```

### Different CUDA Versions

```
Error: NCCL version mismatch
```

**Solution**: Ensure both machines have compatible CUDA/NCCL versions:

```bash
# Check CUDA version
nvcc --version

# Check PyTorch/NCCL version
python -c "import torch; print(torch.version.cuda, torch.cuda.nccl.version())"
```

### Process Hangs at Initialization

```
[Rank 0/2] Process group initialized on nccl backend
[Rank 0/2] Using GPU: cuda:0
<hangs>
```

**Solution**: Check that remote process started successfully. SSH manually to debug:

```bash
# Manual test
ssh daniel@100.111.3.32 "cd /path/to/spinlock && \
  CUDA_VISIBLE_DEVICES=0 RANK=1 WORLD_SIZE=2 MASTER_ADDR=<your-ip> MASTER_PORT=29500 \
  poetry run python -m spinlock.cli.train_meta_operator --config ... --distributed-rank 1 --distributed-world-size 2"
```

## Performance Tips

### Network Bandwidth

- Use 10GbE or faster network for best performance
- NCCL benefits from high-bandwidth, low-latency networks
- Consider InfiniBand for large-scale deployments

### Gradient Accumulation

Combine distributed training with gradient accumulation for effective large batch sizes:

```yaml
training:
  batch_size: 2  # Per-GPU batch size
  gradient_accumulation_steps: 4  # Accumulate 4 batches
  # Effective batch size = 2 × 2 GPUs × 4 steps = 16
```

### Data Loading

Use enough workers to keep GPUs fed:

```yaml
data:
  num_workers: 4  # Per-GPU workers
```

### Sequential Sampling

For Sobol sequences (sequential sampling), the DistributedSampler ensures each GPU gets non-overlapping samples:

- Rank 0: samples 0, 2, 4, 6, ...
- Rank 1: samples 1, 3, 5, 7, ...

This preserves low-discrepancy properties while avoiding duplication.

## Implementation Details

### DDP Wrapper

The model is wrapped in `torch.nn.parallel.DistributedDataParallel`:

```python
if is_distributed:
    noa = DDP(noa, device_ids=[local_rank], output_device=local_rank)
```

This handles:
- Forward pass: Independent on each GPU
- Backward pass: Gradients all-reduced across GPUs
- Parameter update: Synchronized optimizer step

### Checkpoint Saving

Only rank 0 saves checkpoints to avoid conflicts:

```python
if rank == 0:
    torch.save(checkpoint, path)
```

Checkpoints contain the **unwrapped model** (not DDP wrapper) for compatibility.

### Sampler Synchronization

`DistributedSampler` ensures:
- Each GPU sees different samples
- No sample duplication across GPUs
- Epoch shuffling synchronized across workers

## Multi-Node Scaling

### 3+ Machines

```yaml
distributed:
  nodes:
    - host: localhost
      gpus: [0]
    - host: 192.168.1.10
      gpus: [0, 1]  # Use 2 GPUs on this machine
    - host: 192.168.1.11
      gpus: [0]
# World size = 1 + 2 + 1 = 4 GPUs
```

### Multiple GPUs per Machine

```yaml
distributed:
  nodes:
    - host: localhost
      gpus: [0, 1, 2, 3]  # Use all 4 local GPUs
    - host: 100.111.3.32
      gpus: [0]
# World size = 4 + 1 = 5 GPUs
```

## FAQ

**Q: Can I mix CPU and GPU nodes?**
A: No, all nodes must use the same device type. Use `backend: nccl` for GPU or `backend: gloo` for CPU.

**Q: Does this work with VQ-VAE training?**
A: Not yet. This implementation is specific to MNO training. VQ-VAE distributed training would require additional work.

**Q: What if my remote machine has a different directory structure?**
A: Specify `working_dir` in the node config to set the working directory on the remote machine.

**Q: Can I resume from a checkpoint trained with a different number of GPUs?**
A: Yes! Checkpoints are device-agnostic. You can train with 2 GPUs, then resume with 4 GPUs or vice versa.

**Q: What about fault tolerance?**
A: Currently, if any worker fails, the entire training stops. For production deployments, consider elastic training frameworks like `torchx` or `torch.distributed.elastic`.

## See Also

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [Independent Optimization Guide](noa-vqvae-independent.md) - MNO training overview
