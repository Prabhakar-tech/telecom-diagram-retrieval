# M9A E0 Zero-shot CLIP Test Baseline

This is a held-out duplicate-safe visual baseline evaluating zero-shot CLIP (`openai/clip-vit-base-patch32`) exclusively on the M9A Test Split candidate pool.

## Key Constraints
- Candidate pool size is 758 test images.
- Evaluated query count is Q1=758, Q2=758, Q3=723.
- Query IDs preserve original/global row identity for traceability.
- Top 100 predictions are stored for inspection.
- Metrics are computed at K=10.
- Ranks beyond 10 are stored but contribute 0.0 to MRR@10.
- No training has occurred.
- This baseline is required before visual domain adaptation (M9A_E1).
- Results should be interpreted separately from full-corpus M2–M8 ablation.

## Baseline Results
The zero-shot performance on the test split establishes the initial visual domain gap. 

If E0 is weak compared to text baselines (which typically achieve >90% R@1 on Q1), it supports the visual domain-gap motivation for E1. This is a held-out estimate of visual-only retrieval capacity.
