# Thesis Defense Strategy & Stage-by-Stage Cards

This document outlines the logical progression of the multimodal telecom document retrieval thesis. It is designed to act as a defense guide. The core narrative is that **the final architecture is evidence-derived. Components are included only when experiments show measurable benefit.**

---

## 1. Dataset Creation and Weak Supervision
* **Why this stage was needed**: We needed a representative telecom multimodal corpus. Since manual annotation of 3,766 complex 3GPP/Ericsson diagrams is prohibitively expensive, we relied on automated extraction of captions and text contexts.
* **Research idea / theory**: Weak supervision (using surrounding text as proxies for image meaning) is a standard bootstrap technique in multimodal IR (e.g., ALIGN, CLIP dataset).
* **Script used**: `01_data_loader.py`, `02_query_generator.py`
* **Inputs**: Raw PDFs, HTML dumps, CSV metadata mapping images to captions and subclause text.
* **Metrics**: Dataset size, number of duplicate image hashes (589 exact duplicates found).
* **Key result**: Generated 3,766 Q1 (Caption) queries and 3,542 Q3 (Context) queries. Established the `duplicate_mapping.json` to avoid penalizing correct visual matches.
* **What it proved**: A large-scale telecom diagram corpus can be automatically harvested with reliable ground-truth proxies.
* **What failed**: Direct context extraction was noisy; 224 queries lacked sufficient context text and were dropped from Q3.
* **Why the next stage followed logically**: Before applying complex neural models, we needed a strong traditional lexical baseline.
* **Possible external question**: *Isn't evaluating on the caption just testing text-to-text matching rather than multimodal retrieval?*
* **Defense answer**: Yes, Q1 acts as an upper-bound diagnostic. However, users often search using descriptive text conceptually similar to a caption. This is why we also generated Q2 (Paraphrased) and Q3 (Context) to simulate realistic, noisy user queries where exact lexical match fails.

---

## 2. Random Baseline / Evaluation Sanity
* **Why this stage was needed**: To prove that the evaluation metrics (especially duplicate-aware MRR@10) were mathematically sound and not artificially inflated.
* **Research idea / theory**: Establish a floor performance metric.
* **Script used**: Implicitly validated in early script iterations and metrics testing.
* **Inputs**: Q1/Q2/Q3 sets, `duplicate_mapping.json`.
* **Metrics**: MRR@10, Recall@10.
* **Key result**: Random retrieval MRR is ~0.001.
* **What it proved**: The search space is large enough (3,766 candidates) that random guessing fails. High scores represent genuine signal capture.
* **What failed**: N/A.
* **Why the next stage followed logically**: With a firm lower bound and validated metrics, we could begin the actual baselines.
* **Possible external question**: *Why use duplicate-aware metrics?*
* **Defense answer**: Telecom standards reuse identical diagrams across different specification versions. Penalizing a model for retrieving an identical pixel array simply because its row index differs from the "ground truth" would underestimate true retrieval quality.

---

## 3. BM25 Caption-Only (B1)
* **Why this stage was needed**: To establish the absolute minimum baseline using traditional TF-IDF sparse lexical retrieval.
* **Research idea / theory**: Robertson's BM25 algorithm. Lexical models excel at exact token matching (critical for telecom acronyms).
* **Script used**: `04_bm25_baselines.py`
* **Inputs**: Q1/Q2/Q3, Image Caption column.
* **Metrics**: duplicate-aware MRR@10.
* **Key result**: Q1 MRR=0.8597, Q2=0.8141, Q3=0.3232.
* **What it proved**: BM25 is extremely powerful for exact-match short queries (Q1/Q2).
* **What failed**: Failed catastrophically on Q3 (Context queries) because long, noisy context dilutes the TF-IDF signal and causes vocabulary mismatch.
* **Why the next stage followed logically**: If short text is good, does adding more document text (the subclause context) improve retrieval?
* **Possible external question**: *Why is BM25 so high on Q1/Q2?*
* **Defense answer**: Telecom terminology contains highly specific, rare tokens (e.g., "eNodeB", "PRACH"). BM25 strongly weights rare exact matches, making it perfectly suited when the query shares vocabulary with the caption.

---

## 4. BM25 Caption + Context (B2)
* **Why this stage was needed**: To test if expanding the index document to include surrounding text helps locate diagrams.
* **Research idea / theory**: Document expansion increases recall by bridging vocabulary gaps.
* **Script used**: `04_bm25_baselines.py`
* **Inputs**: Q1/Q2/Q3, Image Caption + Context columns concatenated.
* **Metrics**: duplicate-aware MRR@10.
* **Key result**: Q1 MRR=0.7203, Q2=0.5947, Q3=0.6747.
* **What it proved**: Adding context drastically improves Q3 (MRR jumped from 0.32 to 0.67) because Q3 queries are derived from the context itself.
* **What failed**: Adding context significantly degraded Q1/Q2 performance (Q1 MRR dropped from 0.85 to 0.72) because long text introduces noise and token dilution for short queries.
* **Why the next stage followed logically**: Lexical retrieval cannot gracefully handle paraphrased or mismatched vocabulary without explicit synonym dictionaries. We needed a semantic retriever.
* **Possible external question**: *Why does adding information (context) hurt short query performance?*
* **Defense answer**: BM25 relies on length normalization. When a document becomes 2,000 words long, a 5-word query match gets heavily penalized by the length normalizer compared to a 10-word caption document.

---

## 5. Dense BGE Caption-Only (D1)
* **Why this stage was needed**: To test if modern Transformer-based semantic embeddings can capture intent beyond exact keyword matching.
* **Research idea / theory**: Dense retrieval (using BGE-base-en-v1.5, a state-of-the-art embedding model trained on large datasets).
* **Script used**: `05_dense_baselines.py`
* **Inputs**: Q1/Q2/Q3, Image Caption column.
* **Metrics**: duplicate-aware MRR@10.
* **Key result**: Q1 MRR=0.8600, Q2=0.8268, Q3=0.3893.
* **What it proved**: Dense retrieval slightly outperforms BM25 on exact/paraphrased queries (Q2 increased from 0.8141 to 0.8268). It successfully bridges vocabulary gaps.
* **What failed**: Still struggled heavily on long Q3 context queries (MRR 0.38).
* **Why the next stage followed logically**: If Dense models also struggle on Q3 without document context, we must test Dense retrieval on Caption+Context.
* **Possible external question**: *Did semantic models "solve" the problem?*
* **Defense answer**: They provided marginal but consistent gains on paraphrased text (Q2). However, they did not exhibit the massive leap often seen in general domain IR, largely because the telecom domain relies so heavily on exact acronyms that BM25 already handles exceptionally well.

---

## 6. Dense BGE Caption + Context (D2)
* **Why this stage was needed**: To test Dense models on long document text.
* **Research idea / theory**: Dense representations of long contexts (up to 512 tokens).
* **Script used**: `05_dense_baselines.py`
* **Inputs**: Q1/Q2/Q3, Image Caption + Context columns.
* **Metrics**: duplicate-aware MRR@10.
* **Key result**: Q1 MRR=0.7431, Q2=0.6475, Q3=0.7188.
* **What it proved**: Similar to B2, adding context heavily boosts Q3 (0.38 -> 0.71) but degrades Q1/Q2.
* **What failed**: The embedding bottleneck. BGE squashes 512 tokens into a single 768-D vector, leading to loss of specific details (the "lost in the middle" problem).
* **Why the next stage followed logically**: Before abandoning base dense models, we needed to verify if a larger model size (parameters/dimensions) would solve the bottleneck.
* **Possible external question**: *Why didn't D2 beat B2 on Q3?*
* **Defense answer**: BGE is optimized for short query-to-passage matching. When encoding entire long subclauses, specific telecom facts are washed out in the dense vector average, whereas BM25 perfectly indexes every individual token.

---

## 7. BGE-Large Scaling Test (M4)
* **Why this stage was needed**: To test the scaling laws. Does increasing parameter count from 109M to 326M and dimension from 768 to 1024 fix the D2 context degradation?
* **Research idea / theory**: Model scaling (BGE-large-en-v1.5).
* **Script used**: `06_dense_large_baselines.py`
* **Inputs**: Q1/Q2/Q3, Caption & Context columns.
* **Metrics**: duplicate-aware MRR@10.
* **Key result**: L1 and L2 performance was functionally identical to D1 and D2 (differences < 0.01 MRR).
* **What it proved**: The performance limit is not model size, but the fundamental domain gap and the limitations of single-vector dense pooling over long technical text.
* **What failed**: Scaling up provided zero meaningful ROI while increasing inference latency heavily.
* **Why the next stage followed logically**: Since text metadata alone had seemingly hit a ceiling, we needed to extract signal from the images themselves.
* **Possible external question**: *Why didn't a larger model help?*
* **Defense answer**: The models lack pre-training on 3GPP architectural specifications. A larger generic model simply has a better general English representation, which offers no advantage for highly specialized 5G network topology graphs.

---

## 8. CLIP Global Visual Retrieval (M5)
* **Why this stage was needed**: To test standard global image-text retrieval.
* **Research idea / theory**: CLIP (Contrastive Language-Image Pretraining) maps text and images into a shared semantic space.
* **Script used**: `07_clip_baseline.py`
* **Inputs**: Q1/Q2/Q3, raw image arrays.
* **Metrics**: MRR@10, Recall@10.
* **Key result**: Extremely poor performance across all queries (MRR ~0.007).
* **What it proved**: Global visual models trained on natural images (dogs, cars) are completely blind to technical telecom diagram semantics (boxes, lines, arrows, acronyms).
* **What failed**: CLIP zero-shot retrieval is unusable as a primary channel for engineering diagrams.
* **Why the next stage followed logically**: If visual global semantics fail, we must return to text but optimize how we combine the strong, complementary text baselines.
* **Possible external question**: *Why test CLIP if it is known to struggle with diagrams?*
* **Defense answer**: It was required to scientifically establish the baseline zero-shot visual capability on our specific dataset. Proving its failure empirically justified the shift toward OCR and advanced text fusion.

---

## 9. Text Fusion and Reranking (M5.5)
* **Why this stage was needed**: To combine the strengths of BM25 (exact match) and Dense (semantic match), and to test if a cross-encoder could re-sort candidate lists effectively.
* **Research idea / theory**: Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
* **Script used**: `08_text_fusion_rerank.py`
* **Inputs**: B1, B2, D1, D2 predictions.
* **Metrics**: duplicate-aware MRR@10.
* **Key result**: Reranking the union top-50 from all systems achieved strong balanced performance (Q1=0.79, Q2=0.72, Q3=0.74).
* **What it proved**: Fusion and reranking create a robust system that doesn't collapse on any single query type.
* **What failed**: Reranking introduces massive computational latency (running 50 cross-encoder passes per query), which is not ideal for real-time systems.
* **Why the next stage followed logically**: Since text metadata was heavily optimized, we needed to extract the actual text trapped inside the diagram image pixels.
* **Possible external question**: *Is reranking worth the compute cost?*
* **Defense answer**: Only for long, complex queries (like Q3). For short queries (Q1/Q2), a simple RRF fusion of B1 and D1 is much faster and yields higher absolute MRR.

---

## 10. EasyOCR Retrieval (M6a)
* **Why this stage was needed**: To extract explicit text (acronyms, component names) embedded inside the diagram images.
* **Research idea / theory**: Optical Character Recognition (EasyOCR).
* **Script used**: `09_ocr_baselines.py`
* **Inputs**: Raw images.
* **Metrics**: MRR@10.
* **Key result**: OCR-only MRR was ~0.19 (Q1). When OCR text was appended to captions, it degraded performance.
* **What it proved**: OCR can retrieve *some* documents based solely on image pixels.
* **What failed**: The OCR quality was too noisy. Reading tiny pixelated text in telecom graphs yielded garbage characters, and appending this noise to clean captions actively degraded BM25 scores.
* **Why the next stage followed logically**: We needed a visual model that understands text *without* relying on brittle external OCR pipelines.
* **Possible external question**: *Why didn't OCR improve the caption baseline?*
* **Defense answer**: Our captions are already high quality. Appending noisy OCR text diluted the clean TF-IDF token distribution. OCR is only useful if the document lacks metadata entirely.

---

## 11. ColPali OCR-free Visual Retrieval (M6b)
* **Why this stage was needed**: To test the latest state-of-the-art vision-language retrieval architecture that processes document images natively via patch embeddings.
* **Research idea / theory**: ColPali (Vision Encoder + Late Interaction pooling).
* **Script used**: `10_colpali_baseline.py`
* **Inputs**: Raw images, Q1/Q2/Q3 text.
* **Metrics**: MRR@10.
* **Key result**: Performance was near zero (MRR ~0.0100).
* **What it proved**: ColPali completely failed to bridge the domain gap.
* **What failed**: While ColPali is excellent at reading standard document layouts (PDFs, receipts), 3GPP network diagrams lack standard linear text flow. The vision encoder failed to interpret the spatial topology of the boxes and acronyms.
* **Why the next stage followed logically**: Visual and OCR models consistently failed to beat the text metadata baselines. The remaining frontier was domain-specific query understanding.
* **Possible external question**: *ColPali is state-of-the-art on Document VQA. Why did it fail here?*
* **Defense answer**: ColPali relies on a SigLIP vision backbone. SigLIP is trained to align patches with natural language. Telecom diagrams are highly abstract structural topologies. Without domain-specific fine-tuning, the patch embeddings cannot map "box with MME" to the semantic query tokens.

---

## 12. Domain Acronym Expansion (M6.5)
* **Why this stage was needed**: To explicitly tackle the vocabulary mismatch problem where a query uses "User Equipment" but the diagram caption says "UE".
* **Research idea / theory**: Domain-specific query rewriting and lexicon canonicalization.
* **Script used**: `11_acronym_expansion.py`
* **Inputs**: Q1/Q2/Q3.
* **Metrics**: MRR@10.
* **Key result**: Pure expansion hurt BM25 (MRR dropped ~0.07). Low-weight fusion (w=0.10) provided small gains.
* **What it proved**: Pure expansion dilutes high-signal acronym tokens with highly frequent generic English words (e.g., "Equipment", "Management").
* **What failed**: Direct query string replacement.
* **Why the next stage followed logically**: With all components tested (Lexical, Dense, Visual, OCR, Rewriting), we had the evidence required to build the final optimized M7 Hybrid system.
* **Possible external question**: *Why does expanding an acronym hurt retrieval?*
* **Defense answer**: BM25 scores rely on term frequency-inverse document frequency (TF-IDF). "MME" is rare and specific (high weight). "Mobility", "Management", and "Entity" are very common words in a telecom corpus (low weight). Expanding replaces a high-signal token with low-signal noise, causing unrelated documents containing "Management" to rank artificially high.

---

## 13. M7 Hybrid Lexical + Dense Retrieval
* **Why this stage was needed**: To formally combine the strongest components identified throughout the thesis into the final architecture.
* **Research idea / theory**: Late fusion of sparse (BM25) and dense (BGE) retrieval channels. BM25 catches exact acronyms; BGE catches semantic paraphrasing.
* **Script used**: `12_hybrid_lexical_dense.py`
* **Inputs**: Best outputs from M2, M3, M5, M6.5.
* **Metrics**: Standard MRR@10, duplicate-aware MRR@10, Recall@K, Bootstrap Confidence Intervals, Win/Loss ratios.
* **Key result**:
  * Q1 Standard MRR@10 improved from best single lexical baseline around `0.890` to `0.892`.
  * Q2 Standard MRR@10 improved from around `0.861` to `0.867`.
  * Q3 Standard MRR@10 improved from around `0.763` to `0.780`.
  * Best fusion:
    * Q1/Q2: Min-Max Score Fusion with BM25-heavy weighting `0.75 BM25 / 0.25 BGE`
    * Q3: Min-Max Score Fusion with balanced weighting `0.50 BM25 / 0.50 BGE`
* **What it proved**:
  * BM25 and BGE are complementary.
  * BM25 preserves exact telecom acronym matching.
  * BGE adds semantic/paraphrase robustness.
  * Score fusion is more useful than pure rank fusion for this dataset because it preserves confidence margins.
* **What failed**:
  * RRF was not the strongest fusion method.
  * CLIP as a low-weight visual auxiliary degraded or failed to improve retrieval.
  * Acronym expansion gave only marginal additional gains after BM25+BGE fusion.
  * Therefore visual/OCR/acronym channels should remain auxiliary or diagnostic, not primary.
* **Why the next stage followed logically**:
  * M7 completes the primary text-hybrid retrieval search.
  * Next stage should consolidate all completed systems into a master ablation table, confidence intervals, qualitative failure taxonomy, and final architecture selection.
* **Possible external question**: *Why did you discard CLIP and OCR from the final architecture?*
* **Defense answer**: Our experiments showed they introduced more noise than signal. The final architecture is strictly evidence-derived; we only retain components that offer measurable MRR lift.

---
*(End of Stage-by-Stage Cards)*
