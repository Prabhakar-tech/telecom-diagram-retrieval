# M9 Qualitative Error Analysis Walkthrough

## Purpose
This milestone converts numerical retrieval performance into qualitative evidence. It provides concrete, representative examples that illustrate the strengths and failure modes of each method.

## How Cases Were Selected
Cases were stratified into categories based on strict rank-based filtering logic applied to M2-M8 predictions. This ensures unbiased selection of representative cases.

## Rank-Based Evidence
- **Lexical vs Dense**: BM25 captures exact string matches (e.g., specific spec numbers or acronyms) while BGE handles semantic synonyms.
- **Fusion Success**: M7 consistently recovers the ground truth when either text signal is moderately strong.
- **Visual Failure**: Zero-shot CLIP and ColPali frequently fail on technical diagrams, providing qualitative evidence of a substantial domain gap.

## What M9 Adds Beyond M8
M8 provided statistical validation and effect-size estimates. M9 provides qualitative evidence for why the observed retrieval behavior occurs. It shows visually *what* BM25 matches compared to BGE, and *what* makes a diagram fail visual retrieval.

## How M9A_E1 Changes the Visual-Domain-Gap Story
The E0 vs E1 examples show that projection adaptation shows higher retrieval metrics than zero-shot CLIP. This suggests that the visual backbone isn't inherently incapable; rather, the embedding space simply requires telecom-specific alignment. This serves as a held-out visual adaptation result.

## Final Takeaway
The qualitative evidence suggests that text metadata remains the strongest retrieval signal for technical diagrams, but visual domain adaptation provides a promising path forward.
