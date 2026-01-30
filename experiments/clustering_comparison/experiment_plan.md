# Clustering Method Comparison Experiments

## Goal
Compare different hierarchical clustering configurations to optimize:
1. **Orthogonality**: Max inter-cluster correlation < 0.15 target
2. **Reconstruction quality**: Maximize feature reconstruction
3. **Codebook utilization**: Maximize usage of codebook entries
4. **Number of clusters**: Find optimal granularity

## Experiments

### Experiment 1: Silhouette Score (Baseline)
- **K selection**: silhouette
- **Linkage**: ward
- **Distance**: correlation
- **Rationale**: Original method for comparison

### Experiment 2: Average Linkage + Gap Statistic
- **K selection**: gap_statistic
- **Linkage**: average (better for elongated clusters)
- **Distance**: correlation
- **Rationale**: Average linkage may find better cluster boundaries

### Experiment 3: Elbow Method
- **K selection**: elbow
- **Linkage**: ward
- **Distance**: correlation
- **Rationale**: Faster alternative to gap statistic

### Experiment 4: Manual K=8
- **K selection**: manual
- **Manual K**: 8 per family
- **Linkage**: ward
- **Distance**: correlation
- **Rationale**: Force more clusters for better orthogonality

### Experiment 5: Distance Threshold
- **Distance threshold**: 0.6 (based on dendrogram inspection)
- **Linkage**: ward
- **Distance**: correlation
- **Rationale**: Manual cut at visible cluster boundary

## Training Parameters
- **Epochs**: 20 (sufficient for comparison)
- **Dataset**: cno_50k_3channel_dev.h5
- **Device**: CUDA
- **Dendrogram export**: Enabled for all experiments

## Metrics to Compare
1. Number of categories discovered
2. Max inter-cluster correlation (orthogonality)
3. Reconstruction quality (1 - L_recon)
4. Validation loss
5. Codebook utilization
6. Training time
