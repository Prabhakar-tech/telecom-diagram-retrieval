# M9 Qualitative Analysis Plan

## 1. Goal
Provide qualitative evidence that explains the numerical results from M2–M8, answering:
1. Why do text systems work so well?
2. Why do visual systems fail zero-shot?
3. Why does context help Q3 but hurt short caption queries?
4. Where does M7 hybrid help over BM25/BGE?
5. Where does reranking help over score fusion?
6. What are the main failure modes?
7. Which examples should be shown in thesis/viva?

## 2. Selection Strategy
We categorize candidates computationally by examining the rank patterns across BM25, BGE, M7, CLIP, ColPali, and OCR predictions. This ensures a non-biased, data-driven selection of representative cases.

## 3. Available Prediction Files
From the audit, we have access to base BM25/BGE scores, best M7 logic, CLIP, ColPali, and OCR predictions.

## 4. Missing Prediction Files (Limitation)
**Limitation:** M5.5 cross-encoder reranker prediction files were not systematically saved. Because running the cross-encoder locally on the dataset takes significant time, Option B was selected: **selective reranker reconstruction is skipped for the planning phase**. The "Reranker helps Q3" category is not currently included in the final gallery. If needed, this must be regenerated later.

## 5. Final Selected Gallery Quality Rules
A case can enter the final thesis gallery only if:
* ground truth image path exists
* query text is complete
* ground truth caption is available
* at least 2 method ranks are available
* reason_for_selection is supported by ranks
* the example teaches one clear thesis lesson

## 6. What Can Be Done
- We have computationally selected excellent candidates for "M7 helps", "BM25 vs Dense", "Visual Failures", and "Final Failures" with full rank evidence and metadata.
- We have generated the failure taxonomy template and gallery plan.

## 7. What Needs Regeneration
- Visual gallery images will need to be gathered in a later step.

## 8. Proposed Qualitative Exhibits
- Top-5 image contact sheets for each system to illustrate rank differences.
- Highlighting acronyms in queries vs OCR noise.
- Side-by-side comparisons of `bm25_caption` vs `bge_caption`.
