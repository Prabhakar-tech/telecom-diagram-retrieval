# Milestone 5.5: Text Fusion and Rank-1 Reranking Walkthrough

This walkthrough details the findings from Milestone 5.5, evaluating Reciprocal Rank Fusion (RRF) across our four text baselines (BM25 and BGE) and a cross-encoder reranker.

## 1. Overall Objectives
- Implement Reciprocal Rank Fusion (RRF) across `B1`, `B2`, `D1`, `D2`.
- Perform an ablation study on the RRF constant `k`.
- Test weighted and heuristic routing variants.
- Test cross-encoder reranking using `BAAI/bge-reranker-base` on the top candidates (union of base top-100).

---

## 2. RRF Constant (`k`) Ablation Study

We evaluated RRF across all 4 channels (H1c variant) varying the traditional constant `k`:

| Metric (R@1) | `k=10` | `k=30` | `k=60` | Best `k` |
|--------------|--------|--------|--------|----------|
| **Q1**       | 0.7761 | 0.7809 | 0.7793 | **30**   |
| **Q2**       | 0.7448 | 0.7397 | 0.7339 | **10**   |
| **Q3**       | 0.5093 | 0.4788 | 0.4689 | **10**   |

> [!TIP]
> **Smaller `k` is better for strict precision.** For our use-case where R@1 is the primary bottleneck, smaller constants (e.g., `k=10` or `k=30`) outperformed the standard `k=60`. By heavily punishing lower-ranked documents, small `k` effectively functions as a sharp intersection filter at the very top ranks.

---

## 3. Reranking Results (Cross-Encoder)

We reranked the top 20 and top 50 candidates sourced from the union of all 4 baseline systems using `BAAI/bge-reranker-base`.

| System | Q1 R@1 | Q2 R@1 | Q3 R@1 |
|--------|--------|--------|--------|
| **Best Single Baseline** | **0.7966** *(B1)* | **0.7616** *(D1)* | 0.6304 *(B2)* |
| **union_top20_rerank** | 0.7387 | 0.6810 | 0.6660 |
| **union_top50_rerank** | 0.7371 | 0.6696 | **0.6939** |

> [!WARNING]
> **Reranking HURTS explicit caption queries.** For Q1 and Q2, the reranker actively degraded Rank-1 precision relative to the strongest caption-based text baselines. The cross-encoder likely struggles with the extreme lexical brevity of Q1/Q2 and attempts to map them to context paragraphs that are merely semantically adjacent rather than strictly matching the caption.

> [!TIP]
> **Reranking significantly HELPS long-context queries.** For Q3, passing the top 50 candidates through the cross-encoder improved R@1 from **0.6304 (B2)** to **0.6939**, a substantial +6.3% absolute gain. The cross-encoder correctly interprets long-form specification context better than simple bi-encoders or BM25.

---

## 4. Rank-2 Near Miss Fixes vs Hurts

By evaluating the difference between `B1` (the best Q1 baseline) and our `H1a` fusion (`B1 + D1`), we can see how fusion acts on Rank-1 precision.

### Fusion Fixing Near-Misses (Good)
In cases like `q1_50`, `q1_605`, and `q1_934`, BM25 failed to rank the correct image 1st (ranking it 2nd instead). `H1a` fusion successfully promoted the correct document to Rank 1. This happens when the dense retriever (`D1`) strongly agrees on the correct candidate, breaking the tie or overcoming a slight BM25 lexical mismatch.

### Fusion Hurting Rank-1 (Bad)
Conversely, in cases like `q1_49`, `q1_346`, and `q1_604`, BM25 was perfectly correct (Rank 1), but the dense score was so misguided that the RRF fusion averaged the correct candidate down to Rank 2.

---

## 5. Official Recommendations for Next Steps

1. **Do not use Cross-Encoders globally.** The strict precision degradation on Q1/Q2 makes them unviable as a universal late-stage step.
2. **Text Backbone Setup:**
   - We should retain `B1` or `H1a` for Q1 direct caption queries.
   - We should use `H1a` for Q2 paraphrased caption queries.
   - We should selectively route Q3 context queries to `union_top50_rerank` or use `B2` when reranking cost is too high.
3. **Move to M6 (Visual Document Retrieval):** We have effectively saturated the metadata-only text retrieval baselines. Our next milestone should be evaluating pure visual models like ColPali to handle diagrams without metadata.

---

## 6. Manager Review: Corrected Best-by-Query-Type Result

The M5.5 results show improvement, but not from a single universal method.

| Query Set | Best Previous Baseline | Best M5.5 Method | Change |
|---|---:|---:|---:|
| Q1 Direct Caption | B1 R@1 = 0.7966, MRR@10 = 0.8903 | H1a R@1 = 0.7971, MRR@10 = 0.8902 | Essentially tied |
| Q2 Paraphrased Caption | D1 R@1 = 0.7616, MRR@10 = 0.8613 | H1a R@1 = 0.7677, MRR@10 = 0.8701 | Small useful gain |
| Q3 Context Query | B2 R@1 = 0.6304, MRR@10 = 0.7631 | union_top50_rerank R@1 = 0.6940, MRR@10 = 0.8081 | Strong gain |

### Corrected Interpretation

Fusion and reranking do improve the retrieval system, but selectively:

- For Q1, exact BM25 caption matching is already near-optimal.
- For Q2, caption-level BM25 + BGE fusion gives a small but meaningful rank-1 improvement.
- For Q3, cross-encoder reranking over union candidates gives the strongest improvement, raising R@1 by about +6.35 percentage points over BM25 caption+context.

Therefore, the official text-only backbone before OCR/ColPali should be:

| Query Type | Recommended Text Method |
|---|---|
| Q1 Direct Caption | B1 or H1a |
| Q2 Paraphrased Caption | H1a |
| Q3 Context Query | union_top50_rerank |

### Thesis Claim

M5.5 shows that the remaining bottleneck is not deeper retrieval but rank-1 precision. A universal fusion/reranking strategy is not optimal; instead, query-type-aware retrieval is needed. Cross-encoder reranking is valuable for long context queries, but it hurts short exact-caption queries.
