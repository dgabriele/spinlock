Aligned Feature Extraction

  Core Insight: The VQ-VAE's features (SUMMARY, TEMPORAL) ARE physics descriptors. We just need to extract them the same way the dataset was generated.

  Architecture

  NOA Trajectory ──┐                      ┌── L_traj (raw MSE)
                   │                      │
                   ├─► SummaryExtractor ──┼── L_summary (physics stats)
                   │   (same as dataset)  │
                   │                      │
                   ├─► TemporalExtractor ─┼── L_temporal (dynamics)
                   │   (same as dataset)  │
                   │                      │
                   └─► VQ-VAE Encoder ────┴── L_latent (behavioral vocab)
                       (frozen 3family_v1)

  CNO Target ──────┴─► (same extraction) ─► Compare

  Loss Structure

  L = L_traj + λ₁*L_summary + λ₂*L_temporal + λ₃*L_latent

  L_traj:     MSE(pred_trajectory, target_trajectory)     # Direct physics
  L_summary:  MSE(pred_summary_feats, target_summary_feats)  # 360D stats
  L_temporal: MSE(pred_temporal_feats, target_temporal_feats) # 256×63D dynamics  
  L_latent:   MSE(pred_z_pre, target_z_pre)               # VQ manifold

  Key Changes

  1. Use dataset's extractors - Import the exact SummaryExtractor config from pipeline
  2. Extract from both trajectories - Pred AND target get same feature extraction
  3. INITIAL handled via IC - Use VQ-VAE's CNN encoder on the IC directly
  4. Feature-space comparison - Real physics comparison before VQ encoding

  Why This Works

  - Same semantics: Features mean the same thing as VQ-VAE training
  - Real physics: SUMMARY/TEMPORAL capture actual dynamical behavior
  - Gradient flow: Pred features ← trajectory ← NOA weights
  - 187D compatibility: Concatenated features match 100k_3family_v1 input
