# Reproducibility notes

The miniature examples in this repository are smoke tests. Full-scale training requires replacing the example files with the actual profiled training subset and using the same decoding and context-length settings across baselines.

Recommended logging during full runs:

- format pass rate;
- fold rate by posterior solvability bucket;
- average declared budget versus realized token usage;
- reward components and their moving averages;
- regime-wise token consumption and correctness.
