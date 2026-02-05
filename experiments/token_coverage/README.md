# Token Coverage Analysis

## Status: Placeholder (Week 3+)

This experiment package is reserved for analyzing token space coverage and usage patterns in the VQ-VAE + MNO system.

## Planned Analyses

### Token Usage Statistics
- Which tokens are frequently used vs rare/dead?
- Token usage distribution across dataset
- Category-level and level-level usage patterns

### Token Co-occurrence
- Which token combinations appear together?
- Hierarchical relationships (L0 → L1 → L2 patterns)
- Cross-category correlations

### Coverage Metrics
- Dataset coverage: fraction of codebook utilized
- Completion model predictions: do they explore new token combinations?
- Comparison: ground truth vs predicted token distributions

### Visualizations
- Token distribution histograms
- Co-occurrence matrices
- Hierarchical clustering of token usage
- t-SNE/UMAP embeddings of token representations

## Future Implementation

This analysis will be implemented after the trajectory completion experiment demonstrates working token-level predictions (Week 3+).

The token coverage analysis will provide insights into:
1. Whether the learned codebook is efficiently utilized
2. How the completion model explores the token space
3. Potential dead codes that could be pruned or reinitialized

## Dependencies

- Trajectory completion results (token predictions)
- VQ-VAE codebook access
- Dataset tokenization results
