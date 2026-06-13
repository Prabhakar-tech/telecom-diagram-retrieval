# M6b: ColPali OCR-Free Visual Document Retrieval

## Goal
Evaluate whether an OCR-free visual-document retriever (ColPali) can utilize diagram layout and embedded text directly from image patches to improve retrieval beyond metadata/caption baselines.

## Preflight & Setup
- **Environment**: Isolated `colpali_env` with Python 3.10, PyTorch 2.5.1, and `colpali-engine` to prevent dependency conflicts with previous PEFT models.
- **Model**: `vidore/colpali-v1.2` in `bfloat16` precision.
- **Index Footprint**: Encoded all 3,766 images into 1,030 patches each. The resulting PyTorch tensor was `~948 MB`, proving that ColPali indexing is exceptionally lightweight for this dataset.
- **Encoding Speed**: Indexing completed in ~6 minutes (batch size 16). Full query scoring across all 11,074 queries took less than 2 minutes.

## Results Summary
The ColPali baseline performed very poorly as a zero-shot baseline on telecom diagram retrieval out-of-the-box.

| Query Set | Dup-Recall@1 | Dup-Recall@10 | Dup-MRR@10 |
|---|---|---|---|
| Q1 (Captions) | 0.0027 | 0.0186 | 0.0061 |
| Q2 (Paraphrased) | 0.0019 | 0.0114 | 0.0040 |
| Q3 (Context) | 0.0034 | 0.0294 | 0.0091 |

## Conclusion
> [!WARNING]
> ColPali underperformed on semantic diagram retrieval without fine-tuning.

While ColPali exhibits strong zero-shot retrieval for general document QA (rich textual pages with charts), telecom diagrams present dense, highly-specific domain acronyms that lack surrounding textual flow. The vision-transformer patches fail to map semantic text queries (e.g. "Community owned NPN access points") to the scattered structural layout elements of the diagrams.

Therefore, for technical diagrams, **metadata-aware text retrieval (fusion of BM25 + BGE) remains drastically superior** to the tested zero-shot ColPali baseline unless fine-tuned specifically on telecom schematics.

## Manager Review: M6b Final Interpretation

ColPali indexing and scoring were technically successful:
* 3,766 images encoded successfully.
* Index size was approximately 948 MB.
* Full query scoring completed safely.
* No GPU or memory failures occurred.

However, the zero-shot ColPali retrieval result is extremely weak:
* Q1 Dup-R@10 ≈ 0.0186
* Q2 Dup-R@10 ≈ 0.0114
* Q3 Dup-R@10 ≈ 0.0294

This shows that zero-shot ColPali does not understand the telecom-specific diagram semantics out of the box. It is worse than metadata-aware BM25/BGE baselines and also weaker than expected for a visual-document retriever.

The finding should not be framed as “ColPali is bad generally.” It should be framed as:

“Zero-shot ColPali is not effective for this telecom technical diagram benchmark without domain adaptation or fine-tuning.”

Final recommendation:
* Do not use zero-shot ColPali as a primary retrieval branch.
* Keep it as a negative OCR-free visual-document baseline.
* The final architecture should remain metadata-text dominant.
* Future visual improvements should require domain adaptation, acronym-aware query expansion, or fine-tuning with hard negatives.
