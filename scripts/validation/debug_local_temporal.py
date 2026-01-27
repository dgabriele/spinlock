#!/usr/bin/env python3
"""Debug local temporal feature extraction."""

import torch
from spinlock.features.temporal.temporal import TemporalFeatureExtractor

device = torch.device('cpu')
extractor = TemporalFeatureExtractor(device=device, window_size=5, medium_window=20)

B, C, H, W = 4, 3, 64, 64

# Feed enough timesteps to build history (must call extract to populate buffer)
for t in range(10):
    u_temp = torch.randn(B, C, H, W, device=device)
    _ = extractor.extract(u_temp)  # This populates the history buffer

# Now manually extract and debug with a new input
u = torch.randn(B, C, H, W, device=device)
# Need to manually add to history buffer like extract() does
extractor.history_buffer.append(u.detach().cpu())
window = min(extractor.short_window, len(extractor.history_buffer))

print(f"Window size: {window}")
print(f"Short window config: {extractor.short_window}")

if window >= 2:
    history = torch.stack([
        extractor.history_buffer[-i].to(u.device)
        for i in range(1, window + 1)
    ], dim=0)

    print(f"\nHistory shape: {history.shape}")

    features = []

    # Count as we go
    temp_mean = history.mean(dim=0)
    temp_mean_norm = torch.norm(temp_mean, p=2, dim=(2, 3))
    features.append(temp_mean_norm)
    print(f"1. temp_mean_norm: {temp_mean_norm.shape}")

    temp_std = history.std(dim=0)
    temp_std_norm = torch.norm(temp_std, p=2, dim=(2, 3))
    features.append(temp_std_norm)
    print(f"2. temp_std_norm: {temp_std_norm.shape}")

    temp_var = history.var(dim=0)
    temp_var_norm = torch.norm(temp_var, p=2, dim=(2, 3))
    features.append(temp_var_norm)
    print(f"3. temp_var_norm: {temp_var_norm.shape}")

    # Skewness
    eps = 1e-8
    centered = history - temp_mean.unsqueeze(0)
    temp_std_full = history.std(dim=0)
    temp_skew = ((centered ** 3).mean(dim=0)) / (temp_std_full ** 3 + eps)
    temp_skew_norm = torch.norm(temp_skew, p=2, dim=(2, 3))
    features.append(temp_skew_norm)
    print(f"4. temp_skew_norm: {temp_skew_norm.shape}")

    # Kurtosis
    temp_kurt = ((centered ** 4).mean(dim=0)) / (temp_std_full ** 4 + eps)
    temp_kurt_norm = torch.norm(temp_kurt, p=2, dim=(2, 3))
    features.append(temp_kurt_norm)
    print(f"5. temp_kurt_norm: {temp_kurt_norm.shape}")

    # Min/max
    temp_min, _ = torch.min(torch.norm(history, p=2, dim=(3, 4)), dim=0)
    temp_max, _ = torch.max(torch.norm(history, p=2, dim=(3, 4)), dim=0)
    features.append(temp_min)
    features.append(temp_max)
    print(f"6-7. min/max: {temp_min.shape}, {temp_max.shape}")

    # Range
    temp_range = temp_max - temp_min
    features.append(temp_range)
    print(f"8. range: {temp_range.shape}")

    # MAD
    mad = torch.mean(torch.abs(history - temp_mean.unsqueeze(0)), dim=0)
    mad_norm = torch.norm(mad, p=2, dim=(2, 3))
    features.append(mad_norm)
    print(f"9. mad: {mad_norm.shape}")

    # Autocorr lag-1
    if window >= 3:
        autocorr = extractor._compute_autocorr(history)
        features.append(autocorr)
        print(f"10. autocorr lag-1: {autocorr.shape}")
    else:
        features.append(torch.zeros(B, C, device=device))
        print(f"10. autocorr lag-1 (zeros): [B, C]")

    # Multi-lag autocorrelation
    print("\nMulti-lag autocorrelation:")
    lags = [2, 5, 10, 20]
    for i, lag in enumerate(lags):
        if window > lag:
            autocorr_lag = extractor._compute_autocorr(history, lag=lag)
            autocorr_mean = autocorr_lag.mean(dim=1, keepdim=True)
            features.append(autocorr_mean)
            print(f"  {11+i}. lag-{lag}: {autocorr_mean.shape}")
        else:
            features.append(torch.zeros(B, 1, device=device))
            print(f"  {11+i}. lag-{lag} (zeros): [B, 1]")

    # Mean of all lags
    if window > 2:
        all_lags = [extractor._compute_autocorr(history, lag=lag).mean(dim=1, keepdim=True)
                   for lag in [1, 2, 5, 10, 20] if window > lag]
        if all_lags:
            mean_autocorr = torch.stack(all_lags, dim=0).mean(dim=0)
            features.append(mean_autocorr)
            print(f"15. mean of all lags: {mean_autocorr.shape} (computed from {len(all_lags)} lags)")
        else:
            features.append(torch.zeros(B, 1, device=device))
            print(f"15. mean of all lags (zeros): [B, 1]")
    else:
        features.append(torch.zeros(B, 1, device=device))
        print(f"15. mean of all lags (zeros): [B, 1]")

    # Trend
    trend = extractor._compute_trend(history)
    features.append(trend)
    print(f"16. trend: {trend.shape}")

    # Flatten and concatenate
    print(f"\nTotal features before reshape: {len(features)}")
    features = [f.reshape(B, -1) for f in features]
    print(f"Features after reshape: {[f.shape for f in features]}")

    result = torch.cat(features, dim=-1)
    print(f"\nFinal shape: {result.shape}")
    print(f"Expected: [4, 39]")
