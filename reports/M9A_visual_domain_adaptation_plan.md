# M9A: Telecom-Aligned Visual Encoder Pilot

## Motivation
Previous experiments showed that zero-shot CLIP and ColPali performed poorly on telecom technical diagrams, while text metadata retrieval remained much stronger. This suggests a visual domain gap.
M9A tests whether a CLIP/ViT-style visual encoder can be aligned to telecom diagrams using weak supervision from image-caption pairs.

## Hypothesis
Zero-shot visual encoders fail because they are not aligned to telecom engineering diagrams. If we adapt CLIP/ViT using telecom image-caption pairs, visual retrieval may improve over zero-shot CLIP on a held-out duplicate-safe test split.

## Relation to M2–M9
This experiment is optional and must remain separate from the full-corpus M2–M8 ablation. It acts as an auxiliary pilot to see if visual domain gap can be reduced.

## Why this is not replacing text-first architecture
The full thesis architecture is text-first. Visual domain adaptation provides a proof-of-concept auxiliary branch, not a replacement for the highly effective BM25/BGE hybrid.

## Planned Methods
- **Method A:** Projection-only adaptation (Freeze backbone, train heads)
- **Method B:** LoRA/adapters (If PEFT is cleanly supported)
- **Method C:** Full fine-tuning (High risk, generally avoided)

## Leakage Prevention Strategy
Split by MD5 duplicate group, not individual row. All duplicate images must remain in the same split (70% train, 10% val, 20% test).

## Metrics
Evaluate on held-out test rows only: Recall@1, 2, 3, 5, 10; MRR@10; duplicate-aware Recall@K; duplicate-aware MRR@10.

## Expected Outputs
- Adapted model weights
- Test split predictions
- M9A performance summary

## Risks
- Small dataset size may lead to rapid overfitting.
- Adapted model might still underperform text baselines.

## What Result is Useful
If Adapted CLIP > Zero-shot CLIP on the held-out test split, it supports the hypothesis that the visual domain gap can be reduced through telecom-specific alignment.

## What Result is Negative but Informative
If Adapted CLIP still fails to beat text baselines, text and metadata remain more reliable than visual-only retrieval for this dataset.
