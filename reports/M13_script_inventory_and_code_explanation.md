# Script Inventory and Code Explanation

This document provides a detailed breakdown of the critical scripts written for the telecom diagram retrieval project, explaining their purpose, architecture, and how to discuss them during a defense.

## Data Loading & Auditing
### `scripts/01_data_loader.py` & `scripts/audit_data.py`
* **Purpose:** Loads the initial CSV, cleans it, and identifies duplicate images.
* **Input Files:** `/DATA1/prabhakar/telecom/All Images Path.csv`
* **Output Files:** Duplicate mappings, clean data subsets.
* **Core Logic:** Uses hashing to identify identical visual images that appear in different subclauses, generating a mapping of duplicate sets.
* **Important Parameters:** Image path resolution rules.
* **Pipeline Fit:** Step 1. The absolute foundation.
* **What could go wrong:** Broken image paths if the external hard drive is unmounted.
* **Oral Explanation:** "We started by writing a data loader to ingest the 3,766 diagrams. Crucially, we found that 3GPP reuses diagrams. We had to build a duplicate-aware mapping so we wouldn't penalize a model for retrieving a valid duplicate diagram."

## Query Generation
### `scripts/02_query_generator.py` & `scripts/03_q2_paraphraser.py`
* **Purpose:** Synthesizes Q1 (captions), Q2 (paraphrased), and Q3 (context) queries.
* **Input Files:** Cleaned dataset metadata.
* **Output Files:** Query sets in `queries/`.
* **Core Logic:** Q1 is a direct copy. Q2 uses LLMs/paraphrasers to alter syntax. Q3 extracts context snippets.
* **Pipeline Fit:** Step 2. Required before any retrieval evaluation.
* **Oral Explanation:** "We didn't have real user queries, so we simulated three difficulty tiers: direct captions, paraphrased queries to test semantic matching, and long-context queries simulating a user pasting a paragraph from a standard."

## Lexical & Dense Baselines
### `scripts/04_bm25_baselines.py`, `scripts/05_dense_baselines.py`
* **Purpose:** Evaluates BM25 (sparse/lexical) and BGE/e5 (dense) retrievers.
* **Core Logic:** Tokenizes text, builds an inverted index for BM25. Embeds text into dense vectors for BGE and computes cosine similarity.
* **Outputs:** `m2_bm25_results.json`, `m3_dense_results.json`.
* **Oral Explanation:** "We built standard baselines. BM25 indexes keyword overlap, which turned out to be extremely strong for telecom acronyms. Dense models project text into vector space, but sometimes struggle with highly specific jargon out-of-the-box."

## Hybrid Fusion
### `scripts/12_hybrid_lexical_dense.py`
* **Purpose:** M7 hybrid fusion of BM25 and Dense scores.
* **Core Logic:** Min-max normalizes the BM25 and Dense scores, then computes a weighted sum (e.g., `score = w * BM25 + (1-w) * Dense`). Evaluates multiple weight grids.
* **Important Parameters:** The weighting factor (e.g., 0.75 BM25, 0.25 BGE).
* **Oral Explanation:** "We found that lexical and dense models have complementary strengths. This script searches for the selected linear combination of their normalized scores, giving us our strong text-first baseline."

## Master Ablation
### `scripts/13_m8_audit_and_ablation.py`
* **Purpose:** Consolidates all metrics into a master table for M8.
* **Outputs:** `m8_master_ablation_table.csv`.
* **Core Logic:** Reads all JSON result files from M1 to M7 and formats them into a single comprehensive matrix.
* **Oral Explanation:** "This is the script that generated our main results table, bringing together every experiment to definitively show which configuration won."

## Visual Adaptation (M9A)
### `scripts/17_m9a_e0_zeroshot_clip_test.py`, `scripts/18_m9a_e1_projection_adaptation.py`
* **Purpose:** Evaluates zero-shot CLIP and then trains a projection layer to adapt CLIP to telecom diagrams.
* **Core Logic:** E0 extracts visual and text embeddings and computes similarity. E1 trains a lightweight linear layer on top of frozen CLIP features using a contrastive loss.
* **Pipeline Fit:** Auxiliary visual branch exploration.
* **Oral Explanation:** "Standard CLIP fails on line drawings. We wrote an adaptation script to train a projection layer on top of CLIP's features. While it didn't beat text retrieval, it showed significant improvement over zero-shot visual retrieval."

## Product Demo
### `scripts/22_m12a_free_form_retrieval.py`
* **Purpose:** The true free-form interactive product demo.
* **Input Files:** User query string, full dataset index.
* **Output Files:** HTML contact sheets, copied images to `retrieved_images/`.
* **Core Logic:** Takes a raw user query, applies the BM25 text-first backend over captions and context, and outputs the top-k results in a highly visual HTML format.
* **What could go wrong:** Out-of-vocabulary terms dropping BM25 scores.
* **Oral Explanation:** "This is the culmination of the work. It takes a raw string, runs our selected text-backend pipeline on the fly against all 3,766 diagrams, and generates a visual report of the top matches."
