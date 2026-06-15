# M7 Hybrid Lexical + Dense Retrieval Plan

## Objective
To formally evaluate the hybrid combination of traditional lexical retrieval (BM25) and dense semantic retrieval (BGE-base) over the multimodal telecom dataset. The goal is to prove that hybrid fusion recovers complementary strengths (exact token matching from BM25 and semantic paraphrasing from BGE) without losing domain-specific accuracy.

## Methodology
The M7 experiment evaluates multiple configurations across all three query types (Q1 Captions, Q2 Paraphrased, Q3 Context):

### A. Single-Channel References
We include the best performing baselines to calculate fusion lift:
- `bm25_caption`
- `bm25_caption_context`
- `bge_caption`
- `bge_caption_context`

### B. Reciprocal Rank Fusion (RRF)
We fuse the best BM25 and BGE channels using RRF (`score = weight / (k + rank)`).
- **Ablation on `k`**: Testing $k \in \{10, 30, 60\}$.
- **Ablation on Weights**:
  - BM25-heavy (0.75 / 0.25)
  - Balanced (0.50 / 0.50)
  - Dense-heavy (0.25 / 0.75)

### C. Normalized Score Fusion
To rigorously validate fusion beyond rank approximations, we perform Min-Max normalization on the raw prediction scores from BM25 and BGE, followed by linear combinations:
- `score_fusion_bm25_075_bge_025`
- `score_fusion_bm25_050_bge_050`
- `score_fusion_bm25_025_bge_075`

### D. Auxiliary Signals (Acronym Expansion & CLIP)
Based on M6.5, domain acronym expansion is applied strictly as a low-weight auxiliary signal to prevent token dilution.
- `hybrid_plus_acronym_005`
- `hybrid_plus_acronym_010`

We also apply CLIP globally with extremely low weight (`0.05`) strictly as a diagnostic test to verify its established poor complementarity.

## Metrics & Outputs
For every system we record:
- Standard Recall@1, 2, 3, 5, 10
- Duplicate-aware Recall and MRR@10
- Per-query Win/Loss/Tie matrices against single channels
- Bootstrap 95% Confidence Intervals

## Verification
- Outputs are saved to `reports/m7_hybrid_lexical_dense_results.json` and a win-loss dataframe in `reports/m7_per_query_win_loss.csv`.
- The final architecture's defense strategy is consolidated in `THESIS_STAGE_DEFENSE_CARDS.md`.
