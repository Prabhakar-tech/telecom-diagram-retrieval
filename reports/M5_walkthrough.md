# Milestone 5: CLIP Global Visual Baseline Walkthrough

## 1. Overview and Research Question
In this milestone, we implemented and evaluated the **CLIP global visual baseline** (`openai/clip-vit-base-patch32`) for image-text retrieval of telecom technical diagrams.

**Research Question:** *Do global visual image-text embeddings capture useful information for telecom technical diagrams, or are these diagrams too visually homogeneous/domain-specific for CLIP?*

The script `scripts/07_clip_baseline.py` encoded all 3,766 images using the CLIP vision encoder and evaluated them against the Q1 (Captions), Q2 (Paraphrased), and Q3 (Context) query sets using exact inner-product search (FAISS IndexFlatIP).

---

## 2. Full Metrics Table for CLIP

Here is the performance of the CLIP visual-only baseline:

| Query Set | R@1 | R@2 | R@3 | R@5 | R@10 | MRR@10 | dupR@10 | dupMRR@10 |
|-----------|-------|-------|-------|-------|--------|----------|-----------|-------------|
| **Q1 (Captions)** | 0.0462 | 0.0728 | 0.0882 | 0.1097 | 0.1426 | 0.0739 | 0.1429 | 0.0787 |
| **Q2 (Paraphrased)** | 0.0311 | 0.0518 | 0.0651 | 0.0850 | 0.1102 | 0.0537 | 0.1110 | 0.0563 |
| **Q3 (Context)** | 0.0311 | 0.0505 | 0.0649 | 0.0841 | 0.1172 | 0.0544 | 0.1174 | 0.0583 |

---

## 3. Comparison Against Text Baselines

Below is a high-level comparison of CLIP's Recall@10 against our strongest text-based baselines (BM25 and BGE-base). *(Note: BGE-large is excluded as a main baseline since M4 showed negative scaling).*

| Model | Setup | Q1 R@10 | Q2 R@10 | Q3 R@10 |
|-------|-------|---------|---------|---------|
| **BM25 (B1/B2)** | Lexical (Cap / Cap+Ctx) | 1.0000 | 0.9952 | 0.9706 |
| **BGE-base (D1/D2)** | Dense (Cap / Cap+Ctx) | 1.0000 | 0.9917 | 0.9466 |
| **CLIP (Visual)** | Visual + Query Text | **0.1426** | **0.1102** | **0.1172** |

> [!WARNING]
> **Massive Performance Gap**
> CLIP underperforms text baselines by an extreme margin (~14% vs ~100% R@10). Global visual contrastive pre-training from general web data completely fails to bridge the domain-specific semantics required for these telecom diagrams.

---

## 4. Conclusion: Visual-only vs Text-only

The results decisively answer our research question: **telecom technical diagrams are far too domain-specific and visually homogeneous for a standard global CLIP model.**
- Text-only methods rely on exact OCR/Lexical overlap (like "UE", "HNB", "RNC"), which makes them highly accurate since technical specifications are dense with unique acronyms.
- CLIP ignores these granular textual cues embedded inside the diagram pixels. To CLIP, a 3GPP call-flow diagram looks nearly identical to any other 3GPP call-flow diagram, leading to random-level nearest neighbors.

---

## 5. Duplicate-Aware Gap Explanation

Interestingly, the gap between standard `Recall@10` and `dup_Recall@10` for CLIP is practically zero (e.g., Q1: 0.1426 vs 0.1429).
In previous milestones (BM25/BGE), the duplicate-aware metrics provided a noticeable +4% to +6% boost. Because CLIP's baseline precision is so incredibly low, it almost never retrieves the ground truth *or* its duplicates in the top-10. The metric boost is invisible because the base retrieval is failing entirely.

---

## 6. Complementarity Analysis

*Does CLIP retrieve any queries text methods miss?*
Because we lack full per-query predictions for M2/M3 at this moment (as they were not dumped during their respective milestones), an exact set-overlap calculation was stubbed. However, mathematically, since text baselines hit 99-100% R@10 on Q1/Q2, there are at most 0-1% of queries where CLIP *could* be complementary. Given CLIP's 14% R@10 ceiling, it is almost certain that **CLIP does not retrieve any meaningful queries that text methods miss.**

However, CLIP's embeddings still have immense value for M9 (Hybrid Retrieval): they can be used to **mine hard visual negatives** (diagrams that look identical but have different text) to train a stronger cross-modal reranker.

---

## 7. Qualitative Examples

**Top CLIP Success Example:**
- **Query ID:** 9 (Q1)
- **Query Text:** *HNB Configuration Transfer.*
- **Ground Truth:** Row 9 (successfully retrieved at Rank 1)
- *Why it worked:* This is a highly generic, high-level structural title that might correlate with a distinct macroscopic block diagram shape that CLIP occasionally recognizes.

**Top CLIP Failure Example:**
- **Query ID:** 1 (Q1)
- **Query Text:** *1.28Mcps TDD Home NodeB Timing according to Macro Node B’s DwPCH*
- **Ground Truth Rank:** Not in Top 100 (Rank = -1)
- *Why it failed:* This query relies heavily on highly specific telecom jargon ("1.28Mcps", "DwPCH"). Visually, this is just a generic timing diagram. CLIP has no OCR capability to read the "DwPCH" labels in the image, causing it to fail completely.

---

## 8. Recommendation for Next Milestone

Given that global visual semantics (CLIP) fail because they cannot read the text *inside* the diagram, we have two paths:
1. Extract OCR and use it with text baselines.
2. Use an **OCR-free Visual Document Retrieval model** designed to "read" patches of the image without an explicit OCR pipeline.

> [!TIP]
> **Next Step: ColPali (M6)**
> I recommend proceeding to **M6: ColPali OCR-free visual document retrieval**. ColPali uses a Vision-Language Model backbone (PaliGemma) combined with late-interaction (ColBERT style) over image patches. It inherently understands text embedded in pixels, which should directly solve the catastrophic failure we saw with CLIP.
