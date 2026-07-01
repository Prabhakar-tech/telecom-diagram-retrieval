# Results and Significance Explained

This document summarizes the core experimental findings of the thesis, grading their success and explaining what the results mean for the overall system.

## 1. Baselines and Text Models
* **BM25 Performance:** Very strong. Because telecom standards rely heavily on precise jargon and acronyms, lexical matching is highly effective for Q1 and Q3.
  * **Classification:** **Good**
* **BGE (Dense) Performance:** Weaker than BM25 on exact keyword matches, but helpful for semantic paraphrasing (Q2).
  * **Classification:** **Mixed**
* **M7 Hybrid Lexical+Dense Fusion:** Combining BM25 and BGE yielded the strong evaluated text-retrieval scores. This formed the backbone of our text-first backend.
  * **Classification:** **Good**
* **Q1/Q2/Q3 Selected Configurations (from M8 Master Ablation):**
  * Q1 (Direct): M7 `score_fusion_bm25_075_bge_025`
  * Q2 (Paraphrased): M5.5 `H1a`
  * Q3 (Context): M5.5 `union_top50_rerank`

## 2. Visual and Multimodal Models
* **CLIP (Zero-Shot) Limitations:** Standard vision-language models struggle with abstract line drawings, highly dense text boxes, and specialized technical concepts.
  * **Classification:** **Weak (Limitation)**
* **M9A Visual Adaptation:** Training a projection layer on top of CLIP increased visual retrieval metrics over zero-shot CLIP, supporting the conclusion that domain adaptation is useful. However, it still did not surpass the text-first baselines.
  * **Classification:** **Mixed (Future Improvement)**
* **ColPali Performance:** This modern document-retrieval VLM failed to beat BM25 on this specific dataset, likely due to a lack of telecom-specific training data.
  * **Classification:** **Weak (Limitation)**

## 3. Image Processing and Data Enrichment
* **OCR Limitations (EasyOCR):** Attempting to extract evidence directly from the diagrams yielded noisy text. Dense arrows, overlapping labels, and poor resolution hurt performance.
  * **Classification:** **Weak (Limitation)**
* **Acronym Expansion:** Expanding 3GPP acronyms helped some queries but introduced noise for others if the context was ambiguous.
  * **Classification:** **Mixed**

## 4. System Evaluation and Qualitative Analysis
* **M8 Master Ablation:** The statistical paired tests confirmed that hybrid text models significantly outperform purely visual or pure dense models on this dataset.
* **M9 Qualitative Gallery:** Manual inspection of failure cases revealed that the system fails when the defining evidence is buried inside the diagram labels (e.g., specific message names like "MSG1") rather than in the caption or context paragraph.
* **M11 Claim Boundary:** We must be careful not to claim this solves telecom retrieval flawlessly or beats modern LLMs. We claim this is an empirically evaluated text-first architecture with a domain-adapted visual auxiliary branch.

## 5. Product Demo (M12/M12A) and Manual Validation
* **M12 Prepared Queries & M12A True Free-Form Retrieval:** The final interactive demos allow users to type arbitrary queries and see visual results immediately.
* **Manual Validation Results:**
  * **Good Cases:** Caption/context-aligned queries (e.g., "handover failure", "LTE protocol stack", "paging"). The system is strong at retrieving high-level architectural or procedural diagrams.
  * **Mixed/Weak Cases:** Message-level queries (e.g., "random access MSG1-MSG4", "UE context release request"). The text-first backend fails here because "MSG1" is only written *inside* the diagram, not in the caption.
  * **Classification:** **Mixed (Motivation for Future Work)**

## 6. The Big Takeaway for Defense
The results indicate a strong text-first performance level with the current BM25 and dense-embedding setup. The remaining failure cases (message-level queries) strongly motivate the need for advanced OCR, VLM evidence extraction, and Diagram-grounded QA as the next major step.
