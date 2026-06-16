# M10 Final Evidence-Based Architecture and Defense Notes

## Final Retrieval Architecture Recommendation
Based on the extensive multi-stage evaluation (M2-M9A) across 3,766 telecom technical diagrams, the recommended architecture is a **Text-First Hybrid Retrieval System**. The primary retrieval signal should rely on exact and semantic matching of text metadata (captions and context), with visual retrieval serving only as an auxiliary branch after domain adaptation.

### Recommended System per Query Type
- **Q1 Direct Captions**:
  Recommended evaluated configuration: M7 score fusion BM25 0.75 + BGE 0.25.
  Practical note: BM25 remains extremely strong; M7 gain over strongest BM25 is tiny, so BM25 is a strong simple baseline and M7 is the selected evaluated fusion when maximum score is needed.
- **Q2 Paraphrased Questions**:
  Recommended evaluated configuration: M5.5 H1a.
  Practical note: hybrid lexical+dense ideas remain useful, but M7 is not the selected evaluated Q2 system in the final master ablation.
- **Q3 Context-extracted Queries**:
  Recommended evaluated configuration: M5.5 union_top50_rerank.
  Practical note: context-style queries benefit from broader candidate generation and reranking/union strategies more than relying on a single fusion setting.

The final architecture is therefore text-first and query-type aware, not a single shared retriever.

## Architectural Evidence and Justifications

### Why Text-First Retrieval Remains Strongest
Telecom diagrams represent highly structured, symbolic information (e.g., protocol stacks, flow charts) rather than natural photographic scenes. The textual metadata accompanying these figures provides the most direct mapping to user queries.

### Why BM25 is Strong
BM25 excels because telecom queries are dense with standard-specific acronyms, procedure names, and exact reference numbers (e.g., "TS 38.331", "RRCConnectionReconfiguration"). Sparse lexical matching preserves the high signal-to-noise ratio of these specific identifiers.

### Why Dense Retrieval Helps Sometimes
Dense retrieval (BGE-base) helps bridge vocabulary mismatches. It effectively retrieves relevant figures when queries use synonyms, paraphrased questions, or indirect conceptual descriptions instead of verbatim caption text.

### Why M7 Hybrid Gain is Statistically Detectable but Practically Small/Modest
M7 fusion provides statistically detectable increases because it catches the semantic misses of BM25 and the exact-keyword misses of BGE. However, because BM25 is already highly accurate on this dataset, the absolute performance ceiling is very high, making the practical lift of hybrid fusion modest (often single-digit percentage gains).

### Why Zero-Shot Visual Retrieval Underperforms
Zero-shot vision-language models (like CLIP and ColPali) show a substantial visual domain gap. They are trained predominantly on natural images and struggle to interpret the abstract, high-density symbolic syntax of technical telecom engineering diagrams.

### What M9A Visual Adaptation Adds
M9A projection-only adaptation demonstrates that the frozen visual representation in CLIP can be partially aligned under the M9A setup to the telecom domain. E1 shows higher held-out visual retrieval metrics than zero-shot CLIP. This suggests the visual backbone is not inherently incapable, but requires domain-specific weak supervision.

### Representative Nature of the M9 Gallery
The M9 qualitative gallery provides representative examples to illustrate the statistical findings from M8. It converts numerical results into qualitative evidence. It is not an exhaustive per-image evaluation, but rather a curated set of cases that ground the numerical phenomena in observable behavior.

### Leakage-Safe Explanation of M9A
The visual adaptation findings are supported by evidence; the audit indicates no train/validation overlap for the selected M9A gallery cases. The M9A cases shown in the gallery (e.g., `q1_17`, `q1_27`) are drawn exclusively from the duplicate-safe held-out TEST split. There is no train/validation overlap in the adaptation evaluation, ensuring the observed performance lift reflects genuine generalization.

## Limitations
1. **Reliance on Metadata**: The primary text-first architecture assumes the availability of clean captions or contextual text.
2. **Dense Saturation**: BGE-large underperformed BGE-base, indicating that generic dense models may saturate or experience domain drift on highly technical text.
3. **OCR Noise**: Raw text extracted via OCR often introduces noise that dilutes retrieval precision compared to clean metadata.

## Future Work
1. **End-to-End Visual Document Retrieval**: Further training of OCR-free architectures (like Qwen2-VL or ColPali) specifically on telecom standards could bridge the visual gap natively.
2. **Structure-Aware Retrieval**: Parsing the topological structure of flow charts and ladders into graph-based representations.
3. **Full-Corpus Visual Alignment**: Scaling the M9A adaptation strategy across the entire corpus for a direct, candidate-matched comparison against the text baselines.
