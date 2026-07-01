# Final System Architecture Explained

When presenting the final state of the thesis, it is critical to distinguish between the **Offline Research Benchmark Architecture** (how we generated the academic metrics) and the **Live Product Demo Architecture** (how a user actually interacts with the system today).

## 1. Research Evaluation Architecture (Offline Benchmark)

This architecture is used to run large-scale evaluations against our synthetic query sets (Q1, Q2, Q3) to determine the strong evaluated models under our benchmark.

* **Target Dataset:** 3,766 telecom diagrams with ground truth duplicate mapping.
* **Input:** Bulk JSON/CSV files containing thousands of queries (Q1: Captions, Q2: Paraphrased, Q3: Context).
* **Processing:**
  * For **Q1 (Direct Captions):** We use the M7 hybrid fusion model: `score_fusion_bm25_075_bge_025`. This weights lexical exact-matching highly.
  * For **Q2 (Paraphrased):** We use the M5.5 text fusion configuration: `H1a`. This handles the semantic variations better.
  * For **Q3 (Context):** We use the M5.5 text fusion configuration: `union_top50_rerank`. This handles long-form paragraphs effectively.
* **Auxiliary Branch:** The M9A Projection-adapted CLIP model serves as an auxiliary visual branch. It does not replace the text-first backend, but supports the conclusion that visual domain adaptation is viable for future multimodal fusions.
* **Output:** JSON metric files (Recall@K, MRR@10) and ablation tables used for the thesis defense.

## 2. Product Demo Architecture (Live System)

This architecture is the practical culmination of the thesis. It is what a user interacts with in the M12 and M12A scripts.

* **Target Dataset:** The same 3,766 diagrams, indexed for live retrieval.
* **Input:** A single, raw text string typed by a user (M12A) or selected from a curated list (M12).
* **Processing Pipeline (Text-First Backend):**
  1. **Query Input:** User types a free-form query.
  2. **Tokenization & Indexing:** The query is tokenized against the BM25 inverted index of image captions and context paragraphs.
  3. **Scoring:** The BM25 algorithm computes query-local scores. (Note: These scores determine the ranking for this specific query, but cannot be compared absolutely against scores from a different query).
  4. **Ranking:** The system selects the Top-K images with the highest scores.
* **Output Generation:**
  * **HTML/Images:** The system generates `latest_results.html`, detailed PNG contact sheets, and copies the original high-resolution retrieved images into a `retrieved_images/` folder for immediate inspection.
* **Validation Mode:** In this mode, we rely on manual review (Top-1 Relevance, Top-5 Contains Relevant) rather than automated ground-truth mapping, as the free-form queries do not have pre-computed ground truths.

## Key Difference for Defense
If asked, "What is your final model?", the answer is:
"For rigorous academic benchmarking, we utilize a modality-specific ensemble (M7/M5.5 fusions). However, for the live product demo (M12A), we deployed the BM25 text-first backend over captions and context, as our M8 ablation supported the conclusion that it provides a robust, low-latency baseline for highly technical jargon queries."
