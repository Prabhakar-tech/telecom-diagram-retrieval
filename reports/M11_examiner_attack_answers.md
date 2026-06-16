# M11 Examiner Attack Answers

### 1. "Did you actually build a model or only evaluate baselines?"
We built an evaluated telecom technical diagram retrieval system. While the core architecture leverages existing lexical and dense text retrievers as baselines, we systematically evaluated them on a novel, duplicate-aware telecom benchmark. Additionally, we trained a lightweight projection-only CLIP adaptation component (M9A) that shows higher held-out visual retrieval metrics than the zero-shot baseline.

### 2. "Is this just copy-paste of open-source models?"
No. Applying open-source models to highly specialized domains like 3GPP telecom specifications requires rigorous handling of data. We constructed a novel benchmark with query stratification (Q1/Q2/Q3), implemented strict duplicate-aware splitting logic, performed systematic ablations (M8), and engineered a query-type-aware text-first architecture. 

### 3. "What exactly is your contribution?"
Our contribution is a comprehensive, duplicate-aware evaluation of multimodal retrieval systems on abstract telecom diagrams. We established a strong text-first architecture, quantified the specific failure modes of zero-shot visual models on technical figures, and provided empirical evidence that lightweight domain adaptation can partially bridge this visual domain gap.

### 4. "Does your model beat top leaderboard performance?"
We do not claim global top leaderboard position performance, as this is a highly specialized domain without a widely established public benchmark. Instead, we established a strong empirical baseline and evaluated the comparative strengths of lexical, dense, hybrid, and visual retrieval methods within this domain.

### 5. "Why is visual retrieval weak?"
Zero-shot vision-language models (like CLIP and ColPali) struggle because they are predominantly trained on natural, photographic images. Telecom diagrams consist of abstract, high-density symbolic syntax, flow charts, ladders, and textual annotations, representing a significant visual domain gap.

### 6. "Why use text if the task is image retrieval?"
Because in technical domains, images (diagrams) are deeply coupled with their semantic context. The text metadata (captions and surrounding text) provides a much higher signal-to-noise ratio for exact technical acronyms and protocol identifiers than the raw pixel data, making a text-first architecture the most robust approach.

### 7. "Why not evaluate all images qualitatively?"
The M9 qualitative gallery provides representative examples to illustrate the statistical phenomena validated/estimated in M8. Exhaustively visualizing all 3,766 images is impractical for a thesis document and unnecessary, as the comprehensive numerical metrics (M2-M8) already provide the statistical validation across the full corpus.

### 8. "Is there train/test leakage?"
The audit indicates no train/validation overlap for the selected M9A gallery examples. For the trained M9A component, the split audit confirms that there is no train/validation overlap for the selected test examples, supporting the interpretation that the observed visual gain is not caused by those selected examples appearing in train/validation.

### 9. "Why is M7 useful if the increase is small?"
M7 hybrid fusion is useful because lexical and dense signals complement each other. It catches the semantic misses of BM25 and the exact-keyword misses of BGE. Even though the performance ceiling is already high due to strong BM25 baselines, the M7 fusion yields statistically detectable gains (+0.001969 on Q1) and serves as a more robust candidate strategy in cases where lexical and dense signals differ.

### 10. "What is the final architecture?"
The final architecture is text-first and query-type aware, consisting of an initial query categorization step, followed by the selected evaluated configuration for that type (M7 for Q1, M5.5 H1a for Q2, M5.5 union reranking for Q3). It incorporates an auxiliary domain-adapted visual branch but heavily relies on robust text metadata matching.
