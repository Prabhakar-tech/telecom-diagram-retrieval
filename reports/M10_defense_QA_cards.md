# M10 Defense Q&A Cards

### Q: Why not use all 3,766 images in the qualitative gallery?
**A:** The M9 gallery provides representative examples to illustrate the statistical phenomena validated/estimated in M8. Exhaustively visualizing all 3,766 images is impractical for a thesis document and unnecessary, as the comprehensive numerical metrics (M2-M8) already provide the statistical validation across the full corpus. The gallery serves as qualitative evidence for *why* the observed retrieval behavior occurs.

### Q: Why is BM25 strong?
**A:** BM25 is strong because telecom technical diagrams are highly symbolic and their accompanying metadata (captions, contextual references) is dense with specific acronyms, protocol names, and exact reference numbers. Sparse lexical matching preserves the high signal-to-noise ratio of these specific identifiers without semantic dilution.

### Q: Why does dense retrieval help sometimes?
**A:** Dense retrieval (BGE) helps when the query vocabulary shifts away from the exact metadata phrasing. For instance, when users pose paraphrased questions (Q2) or use synonyms, dense embeddings can bridge the vocabulary mismatch and retrieve the correct diagram where exact string matching would fail.

### Q: Why is M7 useful if the gain is small?
**A:** M7 hybrid fusion is useful because lexical and dense signals complement each other. It catches the semantic misses of BM25 and the exact-keyword misses of BGE. Even though the performance ceiling is already high due to strong BM25 baselines, the M7 fusion provides statistically detectable increases (+0.001969 on Q1, +0.006804 on Q2, +0.016659 on Q3). It is useful as a robustness-oriented fusion, yielding a more robust candidate strategy in cases where lexical and dense signals differ, not as a single winner across all query types.

### Q: Is the final system one universal model?
**A:** No, the final recommendation is query-type aware rather than a single shared retriever. Different query types (e.g. direct captions vs paraphrased vs context-extracted) benefit optimally from different strategies like hybrid fusion or broader candidate reranking.

### Q: Why do CLIP/ColPali underperform?
**A:** Zero-shot vision-language models like CLIP and ColPali show a substantial domain gap on this corpus. They are predominantly trained on natural images and struggle to encode the abstract, high-density symbolic syntax, graphs, and textual annotations typical of technical telecom engineering diagrams.

### Q: What does M9A suggest or not suggest?
**A:** M9A suggests that the frozen visual representation in CLIP can be partially aligned under the M9A setup to the telecom domain using projection-only adaptation. It provides held-out visual adaptation evidence that weak supervision shows higher retrieval metrics in visual retrieval. However, it does not suggest that this simple adaptation outperforms the text-first architecture on the full corpus, as it was only tested as an auxiliary visual branch on a duplicate-safe subset. It does not compare against text systems under the same full-corpus candidate setup.

### Q: Is there train/test leakage?
**A:** No. The M9A adaptation strategy employs strict duplicate-safe splitting. The qualitative gallery examples (e.g., `q1_17`, `q1_27`) demonstrating visual adaptation success are drawn exclusively from the held-out TEST split. There is no overlap between the pairs used to train the projection adapters and the pairs used to evaluate them.

### Q: What is the final contribution?
**A:** The final contribution is a comprehensive, multi-modal evaluation of retrieval methods on a novel corpus of telecom engineering diagrams. It establishes a strong text-first architecture baseline, quantifies the specific failure modes of zero-shot visual models on technical figures, and provides evidence that lightweight domain adaptation can partially bridge this visual domain gap.

### Q: What are the limitations?
**A:**
- The text-first architecture heavily relies on the availability of clean metadata.
- Generic dense models (like BGE-large) can experience domain drift on highly technical text.
- OCR extraction, while diagnostically useful, often introduces noise that dilutes primary retrieval accuracy compared to clean captions.

### Q: What would you do next?
**A:** Future work should explore end-to-end visual document retrieval architectures (like Qwen2-VL or fine-tuned ColPali) specifically adapted to telecom standards, aiming to bridge the visual gap natively without relying on text extraction. Additionally, parsing topological structures into graph representations could improve retrieval of complex flow charts and ladders.
