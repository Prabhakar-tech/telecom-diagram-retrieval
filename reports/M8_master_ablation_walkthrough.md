# M8 Master Ablation & Statistical Validation Walkthrough

## 1. Purpose of Master Ablation
The purpose of this milestone is to consolidate all experimental results from M2 through M7 into a comprehensive, statistically evaluated ablation package. It serves to estimate statistical uncertainty, assess whether observed differences are robust, and provide evidence for architecture selection based on all methods tested throughout the thesis.

## 2. Why Aggregate Metrics Alone Are Not Enough
While aggregate metrics like MRR@10 offer a high-level summary of model performance, they mask the distribution of successes and failures across individual queries. Comparing two models that both achieve 0.850 MRR might reveal that they solve completely different query subsets. Therefore, we quantify effect size and confidence intervals using per-query results (where available) to determine if small aggregate differences are statistically meaningful or just dataset noise.

## 3. Full System Family Comparison
The master ablation tables clearly demonstrate the retrieval hierarchy across modalities:
1. **Text Metadata Dominates**: Text-based retrieval over clean metadata (captions, contexts) remains vastly superior to any visual-only or raw-OCR approach.
2. **Dense vs Lexical**: Dense retrieval (BGE) helps paraphrased semantics but does not replace the necessity of BM25 for exact technical acronyms. Furthermore, BGE-large does not provide substantial help over BGE-base for our specific telecom diagrams.
3. **Visual/OCR/Domain Extensions**: 
   - CLIP and ColPali performed very poorly as zero-shot baselines, proving a substantial domain gap for telecom engineering diagrams.
   - OCR extracts some signal but raw OCR concatenation hurts clean metadata retrieval.
   - Acronym expansion helps only as a low-weight auxiliary signal.

## 4. Best-by-Query-Type Results
M7 is not universally best across all query types. The best overall evidence suggests a text-first architecture where score fusion is strongest for caption-like queries, while reranking remains strongest for long/context queries.
- **Q1 (Captions)**: Caption-only lexical metadata (`bm25_caption`) provides a very strong baseline. Hybrid fusion (`score_fusion_bm25_075_bge_025`) provides marginal but consistent top performance.
- **Q2 (Paraphrased)**: Dense embeddings combined with lexical retrieval provide a balanced approach, with BM25-heavy hybrid fusion taking the lead, though earlier hybrid versions like M5.5's `H1a` also performed remarkably well.
- **Q3 (Context)**: Caption+context is critical here. BM25 suffers dilution, making the balanced 50/50 lexical+dense hybrid the best M7 performer, although `union_top50_rerank` from M5.5 holds the top overall score across all experiments.

## 5. Fusion Lift Analysis
Hybrid fusion provides a small but consistent improvement over strong saturated text baselines. The practical value of M7 is not a large score jump, but evidence that BM25 and BGE capture complementary retrieval signals: exact acronym matching and semantic paraphrase matching. The main thesis contribution is not a large M7 gain, but the full ablation pattern across text, dense, OCR, visual, acronym, and hybrid systems.
- The effect size of M7 gains over the best BM25 single baseline is **tiny** for Q1 (approx +0.002 MRR).
- The gain is **small** for Q2 (+0.006 MRR).
- For Q3, the gain is **modest** (+0.017 MRR) over the best individual channel.

## 6. Statistical Confidence Summary
Bootstrap 95% confidence intervals were generated (using 1,000 resamples) utilizing local raw prediction caches. Where intervals were computed, they provide uncertainty estimates for the observed paired differences across the available query sets.

Even when a paired confidence interval excludes zero, we do not overstate the result:
- For Q1: M7 vs BM25 is statistically detectable but tiny.
- For Q2: M7 vs BM25 is statistically detectable and small.
- For Q3: M7 vs BM25 is statistically detectable and modest.

Although the confidence interval excludes zero in some comparisons, the absolute effect size is tiny to modest, so the result should be interpreted as statistically detectable but practically limited. If the mean delta is positive but confidence intervals overlap zero, the gain should be interpreted as a small trend rather than a statistically decisive improvement.

## 7. What Results Are Positive
- The consistent dominance of `bm25_caption` across short queries.
- The reliable capability of BGE-base to understand paragraph-level context when combined with exact lexical matching.
- The incremental lift achieved by fusing the raw normalized scores.

## 8. What Results Are Negative But Useful
- The poor zero-shot performance of global visual models (CLIP) and OCR-free document models (ColPali) to handle complex engineering vector diagrams.
- The realization that raw expansion via dictionaries causes token dilution if not strictly down-weighted.
These negative results are highly useful because they directly validate the decision to pivot away from complex vision-encoders back to robust metadata management.

## 9. Final Evidence-Based Conclusions
The strongest evidence is not just the M7 improvement. The strongest evidence is the full ablation pattern: text metadata dominates visual-only retrieval, caption+context is necessary for long queries, and dense retrieval is a required partner for lexical exact-matching. The final architecture is evidence-derived from this complete spectrum of tests.

## 10. Final Architecture Recommendations
The current evidence supports a text-first final architecture, with visual/OCR/acronym channels treated as auxiliary, diagnostic, or fallback components unless later qualitative analysis shows otherwise. The primary retrieval backbone is currently supported by evidence as a Min-Max Score Fusion of BM25 and BGE-base. M7 provides statistically detectable gains in some paired comparisons, but the effect sizes remain tiny to small. Therefore, M7 should be interpreted as evidence for complementary BM25/BGE signals, not as a large performance breakthrough.
