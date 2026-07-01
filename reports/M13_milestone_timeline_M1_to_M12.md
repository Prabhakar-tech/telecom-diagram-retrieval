# Milestone Timeline: M1 to M12 (and M12A)

This document tracks the progression of the telecom diagram retrieval thesis from Milestone 1 to Milestone 12A, providing a complete narrative of what was built, why, and how it evolved.

## M1: Data Loading and Audit
* **Research Question / Purpose:** How to load and structure the 3,766 telecom diagrams extracted from 3GPP standards?
* **Key Idea:** Create a robust data pipeline that maps images to their metadata (captions, context, source subclause) and establishes a duplicate-aware ground truth framework.
* **Scripts Written/Used:** `scripts/01_data_loader.py`, `scripts/audit_data.py`
* **Inputs:** `/DATA1/prabhakar/telecom/All Images Path.csv`, `/DATA5/prabhakar/telecom/extracted_images/images/`
* **Outputs:** Data audits, duplicate mapping dictionaries.
* **Metrics:** Data completeness, duplicate count.
* **Main Result:** Successfully loaded 3,766 images, discovering duplicate visual content that motivated the duplicate-aware evaluation.
* **Significance:** Established the foundational dataset.
* **Limitation:** Raw data only, no retrieval yet.
* **Influence on Next:** Set the stage for generating queries to search against this dataset.

## M2: Query Generation
* **Research Question / Purpose:** How do we evaluate retrieval if we don't have user queries?
* **Key Idea:** Synthesize multiple query sets (Q1: direct captions, Q2: paraphrased, Q3: context-style) to simulate different user search intents.
* **Scripts Written/Used:** `scripts/02_query_generator.py`, `scripts/03_q2_paraphraser.py`
* **Inputs:** Dataset metadata (captions, context).
* **Outputs:** Q1, Q2, Q3 query sets in `queries/` folder.
* **Metrics:** Linguistic diversity, semantic preservation.
* **Main Result:** Created a robust evaluation benchmark with three levels of query difficulty.
* **Significance:** Allowed quantitative evaluation of retrieval methods.
* **Limitation:** Synthetic queries might not exactly match true user queries.
* **Influence on Next:** We can now test standard text retrieval baselines.

## M3 & M4: Lexical and Dense Baselines
* **Research Question / Purpose:** What is the baseline performance of standard text-based retrieval?
* **Key Idea:** Evaluate BM25 (lexical) and BGE (dense) retrieval against the query sets.
* **Scripts Written/Used:** `scripts/04_bm25_baselines.py`, `scripts/05_dense_baselines.py`, `scripts/06_dense_large_baselines.py`
* **Inputs:** Queries, document text (captions/context).
* **Outputs:** Initial retrieval metrics, JSON results.
* **Metrics:** Recall@1/5/10, MRR@10.
* **Main Result:** BM25 established a very strong baseline, often showing higher retrieval metrics than basic dense models on telecom terminology.
* **Significance:** Showed that lexical overlap is critical for standard-specific jargon.
* **Limitation:** Fails on vocabulary mismatch or semantic paraphrasing (Q2).
* **Influence on Next:** Motivated exploring hybrid systems and visual/multimodal models.

## M5 & M5.5: Text Fusion and Reranking
* **Research Question / Purpose:** Can we combine the strengths of different text representations?
* **Key Idea:** Use reciprocal rank fusion and score fusion on text features, plus try out-of-the-box CLIP.
* **Scripts Written/Used:** `scripts/08_text_fusion_rerank.py`, `scripts/07_clip_baseline.py`
* **Inputs:** Outputs of BM25 and dense retrievers.
* **Outputs:** M5.5 text fusion results.
* **Metrics:** Recall@k, MRR@10.
* **Main Result:** Text fusion raised overall performance, especially for Q3 where M5.5 `union_top50_rerank` was selected. CLIP performed poorly in the zero-shot setting.
* **Significance:** Text-based fusion is the most reliable method for complex queries.
* **Limitation:** Visual models (CLIP) struggle with telecom diagrams out-of-the-box.
* **Influence on Next:** Pushed for OCR and domain adaptation to fix visual model weaknesses.

## M6 & M6.5: OCR and Acronym Expansion
* **Research Question / Purpose:** Can we extract text from images to help retrieval? Can we expand acronyms in queries?
* **Key Idea:** Run EasyOCR on diagrams and expand 3GPP acronyms to handle shorthand queries.
* **Scripts Written/Used:** `scripts/09_ocr_baselines.py`, `scripts/11_acronym_expansion.py`
* **Inputs:** Images (for OCR), queries (for acronyms).
* **Outputs:** OCR text metadata, expanded queries.
* **Metrics:** Retrieval lift after adding OCR/acronyms.
* **Main Result:** Mixed. OCR was noisy. Acronym expansion helped some queries but hurt others if expanded incorrectly.
* **Significance:** Highlighted the difficulty of extracting evidence directly from diagrams.
* **Limitation:** Current OCR struggles with dense diagram arrows and text overlap.
* **Influence on Next:** Looked towards end-to-end multimodal retrieval (ColPali) and hybrid text models (M7).

## M6B: ColPali Baseline
* **Research Question / Purpose:** Can a modern multimodal document retrieval model (ColPali) solve the task natively?
* **Key Idea:** Use vision-language models for retrieval without explicit OCR.
* **Scripts Written/Used:** `scripts/10_colpali_baseline.py`
* **Inputs:** Images and queries.
* **Outputs:** ColPali retrieval metrics.
* **Metrics:** Recall@k.
* **Main Result:** Underperformed strong BM25 baselines on this specific domain.
* **Significance:** Supported the conclusion that generic multimodal models still need domain adaptation for highly technical schemas.
* **Limitation:** High compute cost, low accuracy on telecom jargon.
* **Influence on Next:** Pivot back to strong text-based hybrid systems (M7).

## M7: Hybrid Lexical-Dense Fusion
* **Research Question / Purpose:** What combination of BM25 and Dense embeddings works best in our evaluated setting?
* **Key Idea:** Exhaustive search over fusion weights (e.g., BM25 0.75 + BGE 0.25).
* **Scripts Written/Used:** `scripts/12_hybrid_lexical_dense.py`
* **Inputs:** BM25 and BGE scores.
* **Outputs:** Optimized fusion configurations.
* **Metrics:** Recall@k, MRR@10.
* **Main Result:** Discovered that `score_fusion_bm25_075_bge_025` is the selected configuration for Q1.
* **Significance:** Yielded the highest overall text-retrieval scores.
* **Limitation:** Still text-first, doesn't "look" at the image.
* **Influence on Next:** Set the final configurations evaluated in the master ablation (M8).

## M8: Master Ablation and Statistical Validation
* **Research Question / Purpose:** Are the improvements statistically significant?
* **Key Idea:** Consolidate all results from M1 to M7 and perform paired statistical tests.
* **Scripts Written/Used:** `scripts/13_m8_audit_and_ablation.py`, `scripts/14_m8_statistical_validation.py`
* **Inputs:** All previous metric JSON files.
* **Outputs:** Master ablation tables, statistical significance markers.
* **Metrics:** p-values, paired Recall/MRR differences.
* **Main Result:** Confirmed that BM25 and hybrid fusions are significantly better than pure dense or zero-shot visual models.
* **Significance:** Solidified the core findings of the thesis.
* **Limitation:** purely quantitative.
* **Influence on Next:** We need to know *why* systems fail, leading to qualitative analysis (M9).

## M9: Qualitative Error Analysis
* **Research Question / Purpose:** What do the failure cases look like?
* **Key Idea:** Generate contact sheets of successful and failed retrievals for manual inspection.
* **Scripts Written/Used:** `scripts/19_m9_generate_qualitative_gallery.py`
* **Inputs:** Top-k predictions, images.
* **Outputs:** Visual galleries (contact sheets).
* **Metrics:** Manual inspection taxonomy.
* **Main Result:** Identified failure modes like generic queries, highly overlapping visual schemas, and reliance on fine-grained message labels.
* **Significance:** Provided deep insight into model behavior beyond pure numbers.
* **Limitation:** Manual and subjective.
* **Influence on Next:** Showed that visual models need help, leading to M9A.

## M9A: Visual Domain Adaptation (Sub-milestone)
* **Research Question / Purpose:** Can we train CLIP on the 3GPP data to make it better at technical diagrams?
* **Key Idea:** Train a projection layer on top of CLIP using train/val/test splits, leaving the text-first architecture intact but strengthening the visual auxiliary branch.
* **Scripts Written/Used:** `scripts/17_m9a_e0_zeroshot_clip_test.py`, `scripts/18_m9a_e1_projection_adaptation.py`
* **Inputs:** Image-caption pairs, split dynamically to avoid data leakage (duplicate-aware).
* **Outputs:** Adapted CLIP models and increased visual retrieval metrics.
* **Metrics:** Zero-shot vs. Projection Recall@k.
* **Main Result:** Projection adaptation increased visual retrieval metrics over zero-shot CLIP, though it still lags behind strong text hybrids.
* **Significance:** Supported the conclusion that domain adaptation is viable for technical diagrams.
* **Limitation:** Data scale is small (3.7k images).
* **Influence on Next:** Finalized the system architecture components.

## M10 & M11: Final Architecture and Claim Boundary
* **Research Question / Purpose:** How do we package the final system and defend the claims?
* **Key Idea:** Define the selected architectures for Q1/Q2/Q3 and bound claims to avoid saying "global best" or "solved".
* **Scripts Written/Used:** N/A (Documentation phase)
* **Outputs:** M10 and M11 markdown reports.
* **Main Result:** Established a defensible research position focusing on empirical evidence and system trade-offs.
* **Significance:** Prepared the narrative for defense.
* **Influence on Next:** Moved to building the actual product demo.

## M12 & M12A: Product Demos and Free-Form Retrieval
* **Research Question / Purpose:** Does this work in a real-world scenario?
* **Key Idea:** Build an interactive/batch retrieval demo. M12 uses prepared queries; M12A allows true free-form user queries using the BM25 text-first backend.
* **Scripts Written/Used:** `scripts/20_m12_interactive_retrieval_demo.py`, `scripts/22_m12a_free_form_retrieval.py`
* **Inputs:** Any text query.
* **Outputs:** Contact sheets, HTML results, copied retrieved images.
* **Metrics:** Top-1 relevance, Top-5 contains relevant (manual grading).
* **Main Result:** The system is highly effective for caption/context-aligned queries (e.g., "handover failure") but struggles with message-level queries (e.g., "MSG1").
* **Significance:** This supports the conclusion that the system is a useful research demo prototype.
* **Limitation:** Requires further OCR/VLM integration for deep message extraction.
* **Influence on Next:** Sets up future work for Diagram-grounded QA.
