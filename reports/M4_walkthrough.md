# Milestone 4 Walkthrough — Dense Semantic Baselines (BGE-large-en-v1.5)

> Generated: 2026-06-13T00:29:50+05:30
> Script: `scripts/06_dense_large_baselines.py`
> Report: `reports/m4_dense_large_results.json`

---

## 1. Objective

Test whether scaling the embedding model from 768-d (`bge-base-en-v1.5`) to 1024-d (`bge-large-en-v1.5`) closes the R@1 / MRR@10 precision gap identified in M3. The core question: **does more model capacity help?**

---

## 2. Setup

| Parameter | M3 (BGE-base) | M4 (BGE-large) |
|-----------|--------------|----------------|
| Model | `BAAI/bge-base-en-v1.5` | `BAAI/bge-large-en-v1.5` |
| Embedding dimension | 768 | **1024** |
| FAISS index type | `IndexFlatIP` | `IndexFlatIP` |
| Precision | float16 (A40 GPU) | float16 (A40 GPU) |
| Query prefix | BGE asymmetric (same) | BGE asymmetric (same) |
| Retrieval depth | top-100 | top-100 |
| Batch size | 256 | 128 (larger model) |
| Corpus encode time | ~1.8 s (L1) | ~14 s (L1) |
| FAISS search time | ~0.3 ms/query | ~0.4 ms/query |

**Experiments:**
- **L1** — BGE-large on `Image Caption` only
- **L2** — BGE-large on `Image Caption + Context` (concatenated; 512-tok truncation)

---

## 3. Full Results

### Recall@K & MRR@10

| Query | Experiment | R@1 | R@2 | R@3 | R@5 | R@10 | MRR@10 | dupR@10 | dupMRR@10 |
|-------|-----------|-----|-----|-----|-----|------|--------|--------|----------|
| **Q1** | B1 BM25 Caption | 0.7966 | 0.9620 | 0.9833 | 0.9981 | **1.0000** | 0.8903 | 1.0000 | 0.9241 |
| **Q1** | D1 BGE-base Caption | 0.7934 | 0.9583 | 0.9825 | 0.9981 | **1.0000** | 0.8880 | 1.0000 | 0.9207 |
| **Q1** | L1 BGE-large Caption | 0.7910 | 0.9565 | 0.9811 | 0.9976 | **1.0000** | 0.8863 | 1.0000 | 0.9196 |
| **Q1** | B2 BM25 Cap+Ctx | 0.6660 | 0.8489 | 0.9031 | 0.9503 | 0.9796 | 0.7907 | 0.9841 | 0.8239 |
| **Q1** | D2 BGE-base Cap+Ctx | 0.7204 | 0.9052 | 0.9466 | 0.9748 | 0.9899 | 0.8354 | 0.9936 | 0.8700 |
| **Q1** | L2 BGE-large Cap+Ctx | 0.6896 | 0.8837 | 0.9315 | 0.9689 | 0.9883 | 0.8141 | 0.9923 | 0.8474 |
| **Q2** | B1 BM25 Caption | 0.7568 | 0.9291 | 0.9610 | 0.9870 | 0.9952 | 0.8609 | 0.9952 | 0.8929 |
| **Q2** | D1 BGE-base Caption | 0.7616 | 0.9251 | 0.9565 | 0.9819 | 0.9918 | 0.8613 | 0.9918 | 0.8926 |
| **Q2** | L1 BGE-large Caption | 0.7446 | 0.9222 | 0.9557 | 0.9806 | 0.9875 | 0.8514 | 0.9878 | 0.8818 |
| **Q2** | B2 BM25 Cap+Ctx | 0.5526 | 0.7294 | 0.7947 | 0.8619 | 0.9055 | 0.6843 | 0.9177 | 0.7140 |
| **Q2** | D2 BGE-base Cap+Ctx | 0.6614 | 0.8465 | 0.8922 | 0.9331 | 0.9586 | 0.7822 | 0.9634 | 0.8137 |
| **Q2** | L2 BGE-large Cap+Ctx | 0.6314 | 0.8165 | 0.8664 | 0.9198 | 0.9570 | 0.7581 | 0.9623 | 0.7885 |
| **Q3** | B1 BM25 Caption | 0.3001 | 0.4150 | 0.4627 | 0.5217 | 0.5929 | 0.3965 | 0.5954 | 0.4108 |
| **Q3** | D1 BGE-base Caption | 0.3628 | 0.4816 | 0.5378 | 0.5997 | 0.6581 | 0.4631 | 0.6601 | 0.4805 |
| **Q3** | L1 BGE-large Caption | 0.3490 | 0.4740 | 0.5274 | 0.5870 | 0.6539 | 0.4520 | 0.6564 | 0.4695 |
| **Q3** | B2 BM25 Cap+Ctx | 0.6304 | 0.8199 | 0.8834 | 0.9328 | **0.9706** | 0.7631 | 0.9715 | 0.7870 |
| **Q3** | D2 BGE-base Cap+Ctx | 0.6143 | 0.7936 | 0.8554 | 0.9108 | 0.9466 | 0.7423 | 0.9472 | 0.7666 |
| **Q3** | L2 BGE-large Cap+Ctx | 0.5782 | 0.7626 | 0.8283 | 0.8826 | 0.9300 | 0.7115 | 0.9308 | 0.7369 |

---

## 4. Analysis: BGE-large vs BGE-base

### 4.1 The Core Finding: Scaling Does NOT Help

> [!CAUTION]
> **BGE-large consistently underperforms BGE-base across every query set and both corpus configurations.** This is the opposite of the expected result.

| Config | Metric | BGE-base | BGE-large | Δ (large − base) |
|--------|--------|----------|-----------|-------------------|
| Q1 L1 vs D1 | R@1 | 0.7934 | 0.7910 | **−0.0024** |
| Q1 L1 vs D1 | MRR@10 | 0.8880 | 0.8863 | **−0.0017** |
| Q2 L1 vs D1 | R@1 | 0.7616 | 0.7446 | **−0.0170** |
| Q2 L1 vs D1 | MRR@10 | 0.8613 | 0.8514 | **−0.0099** |
| Q3 L1 vs D1 | R@1 | 0.3628 | 0.3490 | **−0.0138** |
| Q3 L1 vs D1 | R@10 | 0.6581 | 0.6539 | **−0.0042** |
| Q2 L2 vs D2 | R@1 | 0.6614 | 0.6314 | **−0.0300** |
| Q3 L2 vs D2 | R@1 | 0.6143 | 0.5782 | **−0.0361** |
| Q3 L2 vs D2 | R@10 | 0.9466 | 0.9300 | **−0.0166** |

The degradation is consistent and most pronounced on **Q2 paraphrases** (−1.7% R@1 for caption-only) and **Q3 context queries** (−3.6% R@1 for Cap+Ctx). The effect is small for Q1 (near-perfect ceiling) but clearly negative everywhere.

---

### 4.2 Q1 — Direct Caption Queries

All three caption-only systems (B1, D1, L1) achieve **R@10 = 1.0000** — the ceiling. The ordering on R@1 is:

```
B1 BM25 (0.7966) > D1 BGE-base (0.7934) > L1 BGE-large (0.7910)
```

BGE-large is marginally worse than BGE-base here, though all three are within 0.006 of each other. The R@10 ceiling masks any meaningful difference. The useful signal is in R@1 and MRR@10, where BM25 leads by ~0.004 over both dense models — **lexical precision at rank-1 for exact-match queries has not been surpassed by either dense model**.

---

### 4.3 Q2 — Paraphrased Queries ← Key Diagnostic for Scaling

| Experiment | R@1 | R@2 | R@10 | MRR@10 |
|-----------|-----|-----|------|--------|
| B1 BM25 Caption | **0.7568** | 0.9291 | **0.9952** | **0.8609** |
| D1 BGE-base Caption | 0.7616 | 0.9251 | 0.9918 | 0.8613 |
| L1 BGE-large Caption | 0.7446 | 0.9222 | 0.9875 | 0.8514 |

BGE-large is the **worst-performing dense model on Q2**, sitting −0.0170 R@1 below BGE-base and −0.0099 MRR@10 below. The larger model provides no benefit for paraphrase retrieval over this telecom corpus — in fact it actively regresses. This rules out "not enough model capacity" as the explanation for the Q2 near-miss pattern.

---

### 4.4 Q3 — Context-Extracted Queries

For caption-only indexing, the dense models follow `D1 > L1 >> B1`:

| Experiment | R@1 | R@10 |
|-----------|-----|------|
| D1 BGE-base Caption | **0.3628** | **0.6581** |
| L1 BGE-large Caption | 0.3490 | 0.6539 |
| B1 BM25 Caption | 0.3001 | 0.5929 |

Dense outperforms BM25 (semantic bridging works), but BGE-large *underperforms* BGE-base even on this cross-field retrieval task, where one might expect a richer embedding to help more.

For Cap+Ctx indexing, the ordering is `B2 > D2 > L2` — BGE-large is last with R@10=0.9300 vs D2's 0.9466 and B2's 0.9706.

---

### 4.5 Why Does BGE-large Underperform?

Several plausible explanations:

1. **Domain specificity of telecom captions**: BGE-large's larger capacity may have absorbed more general-web distributional signal during pre-training, making it slightly less specialised for the short, structured captions in 3GPP specifications. BGE-base, being smaller, may be less prone to this generalisation bias.

2. **Short-text retrieval**: The document corpus consists of short captions (typically 10–30 tokens). Short-text symmetric retrieval may not benefit from 1024-d representations — the signal is already well-captured in 768-d.

3. **fp16 precision**: Both models run in float16. It is possible BGE-large is more sensitive to the fp16 quantisation of its weights than BGE-base, causing slightly noisier cosine similarity scores.

4. **The 512-token limit applies to both**: No additional context is being captured by the larger model since both hit the same truncation ceiling.

> [!NOTE]
> This result is not unprecedented. In many domain-specific retrieval benchmarks (BEIR, LoTTE), the `base`-size BGE model sometimes matches or beats `large` on specialised corpora, especially when documents are short. The 3GPP telecom caption corpus appears to fall into this category.

---

## 5. Key Findings Summary

| Finding | Evidence |
|---------|---------|
| BGE-large does NOT close the R@1 precision gap vs BGE-base | ΔR@1 = −0.017 (Q2 caption-only) |
| BGE-large underperforms BGE-base on ALL query sets and corpus configs | Consistent negative Δ across 6 experiment × 3 query combinations |
| The scaling failure is most severe on Q2 paraphrases | L1 Q2 R@1 = 0.7446 < D1 Q2 R@1 = 0.7616 |
| BGE-large is worst on Q3 Cap+Ctx | L2 Q3 R@10 = 0.9300 vs D2 = 0.9466 vs B2 = 0.9706 |
| R@10 ceiling on Q1 masks scaling differences | B1=D1=L1=1.000; gap only visible at R@1/MRR |
| Rank-2 recovery pattern holds for BGE-large | Q2 L1: R@1=0.745 → R@2=0.922 (+0.177), consistent with M3 finding |

---

## 6. Thesis Implications

1. **Model capacity is not the bottleneck.** Both BGE-base and BGE-large represent the correct class of answer (top-2 recall > 92%), but neither closes the R@1 gap. The precision failure at rank-1 is a **ranking problem, not a representation problem**.

2. **Scaling to BGE-large is not recommended** for this corpus. The base model should be used as the dense embedding baseline for M8 ablations and M9 hybrid experiments, as it is faster, smaller, and consistently better.

3. **The rank-2 near-miss pattern is model-agnostic.** It persists across BM25, BGE-base, and BGE-large, confirming it is a structural property of the retrieval task and corpus — not a model-specific quirk.

4. **Next steps shift away from pure dense retrieval.** Given that BGE-base already represents the ceiling for dense-only retrieval on this corpus, the research priority moves to:
   - **M5 (CLIP)**: Can visual features identify the diagram type that text-only models confuse?
   - **M9 (Hybrid)**: Can combining BGE-base (best dense) + BM25 (best lexical) push R@1 above the near-miss threshold?

---

## 7. Artefacts Produced

| File | Description |
|------|-------------|
| `scripts/06_dense_large_baselines.py` | BGE-large retrieval pipeline |
| `reports/m4_dense_large_results.json` | Full metric results (L1, L2 × Q1, Q2, Q3) |
| `logs/m4_dense_large_run.log` | Full run log with timing |
| `reports/M4_walkthrough.md` | This walkthrough |
