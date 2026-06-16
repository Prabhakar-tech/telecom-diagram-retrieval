# M11 One-Page Defense Summary

## Problem
Telecom engineering documents (e.g., 3GPP specifications) contain thousands of complex, abstract diagrams (flow charts, protocol stacks, ladders). Accurately retrieving these diagrams based on textual queries is challenging due to highly technical vocabulary, abstract imagery, and visual domain gaps in generic models.

## Dataset
We constructed a duplicate-aware corpus of 3,766 telecom technical diagrams. Queries were stratified into three types:
- **Q1**: Direct Captions
- **Q2**: Paraphrased Questions
- **Q3**: Context-extracted Queries

## Methods Evaluated
We rigorously evaluated multiple paradigms:
- Lexical (BM25)
- Dense (BGE-base/large)
- Vision-Language (Zero-shot CLIP, ColPali)
- Extracted Text (OCR)
- Hybrid Fusion & Reranking (M7, M5.5)

## Selected Configurations by Query Type
- **Q1**: M7 score_fusion_bm25_075_bge_025
- **Q2**: M5.5 H1a
- **Q3**: M5.5 union_top50_rerank

## Model Trained
We trained a lightweight projection-only CLIP adaptation component (M9A). Using a frozen CLIP backbone and duplicate-safe splits, we observed that telecom-specific weak supervision yields an auxiliary visual branch that shows higher held-out visual retrieval metrics than zero-shot CLIP.

## Key Numbers
- BM25 establishes an extremely strong lexical baseline (MRR@10 > 0.88 on Q1).
- M7 hybrid fusion yields statistically detectable but practically modest gains (e.g., +0.001969 on Q1, +0.016659 on Q3).
- Zero-shot visual models (CLIP, ColPali) struggle heavily compared to text baselines.

## Claim Boundary
- **We do not claim** top leaderboard position performance, a single shared retriever, or that visual retrieval replaces text metadata matching. 
- **We do claim** the creation of a rigorous benchmark, empirical quantification of the visual domain gap, and a text-first query-type-aware architecture supported by an evaluated auxiliary visual adaptation branch.

## Final Contribution
The core contribution is a systematic, duplicate-aware evaluation of multimodal retrieval systems on abstract telecom diagrams, resulting in an evidence-backed text-first architecture and establishing an empirical baseline for future domain-specific visual adaptation.
