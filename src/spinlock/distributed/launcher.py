"""
Multi-node distributed training launcher.

Launches training processes on local and remote nodes via SSH.
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from typing import List, Optional
import atexit

from .config import DistributedConfig, NodeConfig


class DistributedLauncher:
    """
    Launcher for distributed training across multiple nodes.

    Handles SSH connections, process launching, and cleanup.
    """

    def __init__(self, config: DistributedConfig, script_path: str, script_args: List[str]):
        """
        Initialize launcher.

        Args:
            config: Distributed configuration
            script_path: Path to the training script
            script_args: Arguments to pass to the training script
        """
        self.config = config
        self.script_path = script_path
        self.script_args = script_args
        self.processes: List[subprocess.Popen] = []

        # Register cleanup handler
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n\nReceived interrupt signal, cleaning up...")
        self.cleanup()
        sys.exit(1)

    def launch(self) -> None:
        """Launch training processes on all nodes."""
        rank_mapping = self.config.get_rank_mapping()

        print(f"\n{'='*70}")
        print(f"Launching Distributed Training")
        print(f"{'='*70}")
        print(f"World size: {self.config.world_size}")
        print(f"Master: {self.config.master_addr}:{self.config.master_port}")
        print(f"Backend: {self.config.backend}")
        print(f"Nodes: {len(self.config.nodes)}")
        print(f"{'='*70}\n")

        # Launch processes on each node
        for node_idx, local_gpu_idx, global_rank in rank_mapping:
            node = self.config.nodes[node_idx]

            print(f"[Node {node_idx}] Launching rank {global_rank} on {node.host}:GPU{local_gpu_idx}")

            if node.is_local:
                self._launch_local(node, local_gpu_idx, global_rank)
            else:
                self._launch_remote(node, local_gpu_idx, global_rank)

            # Small delay to avoid race conditions
            time.sleep(0.5)

        print(f"\n{'='*70}")
        print(f"All processes launched successfully!")
        print(f"{'='*70}\n")

        # Wait for all processes to complete
        self._wait_for_completion()

    def _launch_local(self, node: NodeConfig, gpu_idx: int, rank: int) -> None:
        """Launch process on local node."""
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu_idx),
            "RANK": str(rank),
            "LOCAL_RANK": "0",  # Single GPU per process
            "WORLD_SIZE": str(self.config.world_size),
            "MASTER_ADDR": self.config.master_addr,
            "MASTER_PORT": str(self.config.master_port),
        })

        # Build command
        python_cmd = node.python_path or sys.executable
        cmd = [python_cmd, "-m", self.script_path] + self.script_args + [
            "--distributed-rank", str(rank),
            "--distributed-world-size", str(self.config.world_size),
        ]

        # Launch process
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self.processes.append(proc)

    def _launch_remote(self, node: NodeConfig, gpu_idx: int, rank: int) -> None:
        """Launch process on remote node via SSH."""
        # Determine working directory
        working_dir = node.working_dir or os.getcwd()

        # Determine Python command
        python_cmd = node.python_path or "poetry run python"

        # Build remote command
        remote_env = " ".join([
            f"CUDA_VISIBLE_DEVICES={gpu_idx}",
            f"RANK={rank}",
            f"LOCAL_RANK=0",
            f"WORLD_SIZE={self.config.world_size}",
            f"MASTER_ADDR={self.config.master_addr}",
            f"MASTER_PORT={self.config.master_port}",
        ])

        script_args_str = " ".join(self.script_args)
        remote_cmd = (
            f"cd {working_dir} && "
            f"{remote_env} {python_cmd} -m {self.script_path} {script_args_str} "
            f"--distributed-rank {rank} --distributed-world-size {self.config.world_size}"
        )

        # Build SSH command
        ssh_user = node.ssh_user or os.getenv("USER")
        ssh_cmd = [
            "ssh",
            "-p", str(node.ssh_port),
            f"{ssh_user}@{node.host}",
            remote_cmd,
        ]

        # Launch SSH process
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self.processes.append(proc)

    def _wait_for_completion(self) -> None:
        """Wait for all processes to complete."""
        try:
            # Wait for all processes
            exit_codes = [proc.wait() for proc in self.processes]

            # Check for failures
            if any(code != 0 for code in exit_codes):
                failed_ranks = [i for i, code in enumerate(exit_codes) if code != 0]
                print(f"\n⚠ Warning: Some processes failed: ranks {failed_ranks}")
                sys.exit(1)
            else:
                print(f"\n✓ All processes completed successfully!")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user, cleaning up...")
            self.cleanup()
            sys.exit(1)

    def cleanup(self) -> None:
        """Terminate all running processes."""
        for proc in self.processes:
            if proc.poll() is None:  # Process still running
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception:
                    pass

        self.processes.clear()


def launch_distributed_training(
    config: DistributedConfig,
    script_path: str,
    script_args: List[str],
) -> None:
    """
    Launch distributed training.

    Args:
        config: Distributed configuration
        script_path: Path to training script
        script_args: Arguments to pass to script
    """
    config.validate()

    launcher = DistributedLauncher(config, script_path, script_args)
    launcher.launch()
