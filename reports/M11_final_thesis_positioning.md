# M11 Final Thesis Positioning

## What we built
We built an evaluated telecom technical diagram retrieval system. The benchmark is structured around three distinct query types (Q1 Direct Captions, Q2 Paraphrased Questions, Q3 Context-extracted Queries) and evaluated using standard retrieval metrics (MRR@10, Recall@k) over a dataset of 3,766 images. We systematically evaluated multiple retrieval paradigms:
- Lexical retrieval (BM25)
- Dense retrieval (BGE-base/large)
- Visual/OCR retrieval (Zero-shot CLIP, ColPali, OCR text matching)
- Hybrid fusion and reranking (M7 score fusion, M5.5 union reranking)

The final architecture recommended by this evaluation is a **text-first query-type-aware architecture**, supported by an auxiliary visual adaptation component (M9A).

## What is the actual architecture?
The final architecture is not a single model, but rather a query-type-aware retrieval framework:
- **Input query**: Processed to determine the query type (Q1, Q2, Q3).
- **Text retrieval branch**: The primary backbone, heavily leveraging exact metadata matches via BM25, and semantic matches via dense embeddings.
- **Hybrid/Rerank branch**: Combines candidate lists to maximize robustness.
- **Optional visual branch**: Domain-adapted visual embeddings serve as an auxiliary signal, not a replacement for text.
- **Final ranked output**: The best-matching technical diagrams retrieved from the corpus.

Based on the master ablation, the selected configuration per query type is:
- **Q1 selected configuration**: M7 score_fusion_bm25_075_bge_025
- **Q2 selected configuration**: M5.5 H1a
- **Q3 selected configuration**: M5.5 union_top50_rerank

## What is the actual trained model?
The only newly trained component in this thesis is the **M9A projection-only CLIP adaptation**.
Rather than training a massive foundation model from scratch, we applied lightweight projection adapters on top of a frozen CLIP backbone. These adapters were trained using a duplicate-safe train/val/test split to reduce train/test contamination risk. Held-out test evaluation indicates that the adapted model (E1) shows higher held-out visual retrieval metrics than zero-shot CLIP (E0), providing an empirical demonstration for telecom visual domain adaptation. However, it does not replace the text-first architecture.

## What is not claimed
To maintain strict academic honesty, this thesis explicitly establishes the following claim boundary:
- **Not the global leaderboard leader**: We do not claim top leaderboard position performance on global benchmarks.
- **Not close-to-peak**: We avoid "close to the absolute peak" claims unless future same-setting comparisons explicitly show it.
- **Not deployment-ready**: The system is an evaluated research prototype, not a deployed commercial system.
- **Not a single shared visual retrieval solution**: Zero-shot vision-language models struggle on telecom diagrams.
- **Not evidence that visual retrieval replaces text retrieval**: The strongest results remain firmly grounded in text and hybrid systems.

## What is novel/valuable?
The value of this thesis lies in its rigorous evaluation framework and empirical findings:
- Creation of a **telecom-specific diagram retrieval benchmark** with multi-query evaluation (Q1/Q2/Q3).
- Strict **duplicate-aware evaluation** to prevent data leakage in dense and visual model testing.
- A systematic comparison of lexical, dense, hybrid, OCR, and visual systems on highly abstract technical imagery.
- Clear evidence that text metadata dominates retrieval performance in this domain.
- Clear evidence that zero-shot visual models struggle heavily with the semantic gap of technical diagrams.
- Empirical evidence that lightweight domain adaptation helps visual retrieval on held-out test splits.
- A final defense-ready architecture and claim boundary that provides a concrete baseline for future telecom multi-modal research.
