"""Decoherence rate estimation via exponential fitting"""

import torch


def estimate_decoherence_rate(
    coherence_trace: torch.Tensor,  # [N, T] coherence over time
    dt: float = 0.01,  # Time step
) -> torch.Tensor:
    """Estimate decoherence rate γ from exponential fit.

    Model: C(t) = C₀ exp(-γt)

    Fit via log-linear regression:
    log C(t) = log C₀ - γt

    Uses least squares: γ = -slope of log(C) vs t

    Args:
        coherence_trace: [N, T] coherence measure over time
        dt: Time step between samples

    Returns:
        [N] estimated decoherence rate per rollout
    """
    N, T = coherence_trace.shape

    # Add small epsilon to avoid log(0)
    coherence_safe = torch.clamp(coherence_trace, min=1e-10)
    log_coherence = torch.log(coherence_safe)  # [N, T]

    # Time vector: t = [0, dt, 2dt, ..., (T-1)dt]
    time = torch.arange(T, device=coherence_trace.device, dtype=torch.float32) * dt
    time = time.unsqueeze(0)  # [1, T]

    # Linear regression: log C = a + b·t
    # b = -γ (decoherence rate)

    # Use least squares: b = cov(t, log C) / var(t)
    t_mean = time.mean(dim=-1, keepdim=True)
    log_c_mean = log_coherence.mean(dim=-1, keepdim=True)

    cov = ((time - t_mean) * (log_coherence - log_c_mean)).mean(dim=-1)
    var_t = ((time - t_mean) ** 2).mean(dim=-1)

    slope = cov / (var_t + 1e-10)  # [N]

    # Decoherence rate γ = -slope
    gamma = -slope

    return gamma
