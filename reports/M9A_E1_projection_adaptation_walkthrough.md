# M9A E1 Projection Adaptation Walkthrough
    
This experiment tests a lightweight projection-only adaptation to align CLIP to telecom diagrams.

## Experimental Details
- **Architecture**: `openai/clip-vit-base-patch32` backbone (Frozen, ~151M params).
- **Trainable Parameters**: Residual linear projection adapters for image and text embeddings (dim=512), and the logit scale (temperature). Total ~1M params.
- **Why projection-only**: It reduces the risk of overfitting or disrupting the frozen CLIP representation, is extremely low-risk for small datasets, and directly tests whether simply rotating/scaling the embedding space can bridge the telecom visual domain gap.
- **Training**: Trained using symmetric contrastive loss on the duplicate-safe train split (2628 image-caption pairs).
- **Validation**: Best checkpoint selected by duplicate-aware MRR@10 on the 380 val split images using exact captions.
- **Test Candidate Pool**: 758 images (identical to M9A_E0).

## Results
Please refer to `reports/m9a_visual_adaptation_comparison.csv` for exact metrics.
The results reflect a held-out estimate of the domain adaptation performance.

## Interpretation
The results indicate that this projection-only adaptation shows higher retrieval metrics than the zero-shot baseline. This evidence supports the visual domain adaptation hypothesis.
