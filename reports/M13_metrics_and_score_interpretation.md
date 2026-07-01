# Metrics and Score Interpretation

This document explains the metrics used to evaluate the system and, crucially, how to interpret the scoring output (especially BM25) during live demos.

## Primary Evaluation Metrics
We use standard information retrieval metrics, adapted for our duplicate-aware ground truth.

### Recall@K
* **Definition:** Out of all queries, what percentage had a relevant diagram in the top K retrieved results?
* **Why we use it:** In a real-world scenario, a user looks at the top few results. Recall@5 tells us if the correct image was on the first "page" of results.
* **Duplicate-Aware:** If *any* image from the correct duplicate group is in the top K, the query scores 1. Otherwise, 0.

### MRR@10 (Mean Reciprocal Rank)
* **Definition:** Where exactly in the top 10 did the correct image appear? It averages the reciprocal of the rank (1/rank).
* **Why we use it:** Recall@10 treats rank 1 and rank 10 equally. MRR@10 gives much higher credit for putting the correct image at rank 1.
* **Interpretation:** Higher MRR means the correct result is closer to the top of the list.

## Manual Validation Metrics (Product Demo)
In the M12/M12A product demos, we don't always have a strict ground truth (free-form queries). We use manual grading:
* **0 = Wrong/Irrelevant:** The diagram has nothing to do with the query.
* **1 = Partially Relevant:** The diagram is related to the domain/concept, but doesn't completely answer the specific query.
* **2 = Clearly Relevant:** The diagram is exactly what the user is looking for.
* **Top-1 Relevance:** Is the absolute top result grade 2?
* **Top-5 Contains Relevant:** Is there at least one grade 2 result in the top 5?

## Explicit Question: Why are BM25 scores not comparable across queries?
**Question:** "Why does Query A have a top BM25 score of 20, and Query B has a top BM25 score of 40? Does that mean Query B's result is twice as good?"

**Answer:** No. BM25 scores are **query-local**. They cannot be directly compared across different queries.
The score depends heavily on:
1. **Query Length:** Longer queries accumulate scores across more terms, generally producing higher totals.
2. **Term Rarity (IDF - Inverse Document Frequency):** If Query B contains extremely rare telecom acronyms, matches for those terms are weighted much higher than common words in Query A.
3. **Document Length Normalization:** BM25 penalizes matching terms in very long documents, assuming the term is diluted.
4. **Term Overlap:** How many times the query terms appear in the document (Term Frequency).

Therefore, a score of 40 vs 20 just means different math happened based on the words used. You must only compare scores *within the same ranking list for a single query*. The ranking order matters, the absolute score does not.
