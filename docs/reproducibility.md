# Reproducibility notes

The miniature examples in this repository are smoke tests. Full-scale training requires replacing the example files with the actual profiled training subset and using the same decoding and context-length settings across baselines.

Recommended logging during full runs:

- format pass rate;
- fold rate by posterior solvability bucket;
- average declared budget versus realized token usage;
- reward components and their moving averages;
- regime-wise token consumption and correctness.

The GRPO integration targets TRL 0.24.0. The main config uses 1,024 trajectories
(64 queries × 16 rollouts), 300 optimizer steps, clip 0.0625, zero KL, and
sequence-normalized GRPO loss. Its accumulation setting assumes eight training ranks.
Use the TRL `vllm-serve` server from `scripts/launch_vllm_server.sh`; the OpenAI-compatible
vLLM API does not provide the policy-weight synchronization needed by GRPO.
Stage-1 LoRA adapters are merged into the base policy before Stage-2 full-parameter training.
