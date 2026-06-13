# Thesis Knowledge Base
## Multimodal Image Retrieval for Telecom Technical Diagrams

> This file is the canonical reference for all experimental constraints,
> baselines, metrics, and infrastructure details. Every script in this
> repository must remain consistent with these definitions.

---

## 1. Domain

- **Task**: Cross-Modal Retrieval of Telecom Technical Diagrams
- **Standards Body**: 3GPP (3rd Generation Partnership Project)
- **Dataset**: 3,766 images from 3GPP technical specification documents
  - Image types: Call Ladders, Flow Charts, Architecture Diagrams, Block Diagrams, etc.
  - Extracted from specifications across TS 25.xxx, TS 36.xxx, TS 38.xxx series
- **Duplicate images**: 281 groups (589 images) share identical content (MD5-verified)
- **Unique visual content**: 3,458 distinct images

---

## 2. Query Sets

| Query Set | ID Prefix | Description | Count | Source |
|-----------|-----------|-------------|-------|--------|
| Q1 — Direct Captions | `q1_` | Verbatim `Image Caption` from the paths CSV | 3,766 | `q1_captions.json` |
| Q2 — Paraphrased | `q2_` | LLM-generated natural-language questions from Q1 captions | 3,766 (target) | `q2_paraphrased.json` |
| Q3 — Context-extracted | `q3_` | First substantive sentence from the `Context` column | 3,542 | `q3_context.json` |

---

## 3. Retrieval Baselines

| Baseline | Type | Model / Method | Notes |
|----------|------|----------------|-------|
| BM25 (B1/B2) | Lexical | `rank_bm25` over tokenised captions/context | Sparse baseline; B1=Caption only, B2=Caption+Context |
| BGE-base (D1/D2) | Dense Semantic | `BAAI/bge-base-en-v1.5` | 768-d, FAISS `IndexFlatIP`; **designated dense baseline for M9 hybrid** — outperforms BGE-large on this corpus |
| BGE-large (L1/L2) | Dense Semantic | `BAAI/bge-large-en-v1.5` | 1024-d; **underperforms BGE-base across all configs** (M4 result) — short-text saturation / domain drift; not recommended for further experiments |
| CLIP | Global Visual | `openai/clip-vit-base-patch32` or `openai/clip-vit-large-patch14` | Encode (query_text, image) pairs |
| Text Fusion (H1c) | Late Fusion | Reciprocal Rank Fusion (RRF) | Fuses top candidates from B1, B2, D1, D2. An RRF `k=10` constant proved superior for Rank-1 precision. |
| Reranker (CE) | Cross-Encoder | `BAAI/bge-reranker-base` | Applied to the union of top-50 from base text retrievers. **Significantly improved Q3** (+6% R@1), but hurt Q1/Q2 caption matching. |
| EasyOCR (M6a) | Visual Text | `easyocr` + `rank_bm25` | Extracted text from diagrams. **Adds noise and dilutes exact caption matching**, lowering R@1 across all query types. |
| ColPali (M6b) | OCR-free Visual Document | `vidore/colpali-v1.2` | Failed to retrieve diagrams zero-shot. Extremely poor semantic matching for dense acronyms in technical figures. |
| Qwen2-VL | OCR-free Visual Document | `Qwen/Qwen2-VL-7B-Instruct` | Vision-language encoder |

---

## 4. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Recall@1** | `1 if gt ∈ top-1 else 0` | Strict precision — is the correct image ranked first? Primary gap metric |
| **Recall@2** | `1 if gt ∈ top-2 else 0` | **Introduced M2–M4** to quantify the non-linear rank-2 near-miss recovery pattern; the R@1→R@2 jump (+16–19%) is 3–5× larger than R@2→R@5, revealing failures at rank-1 are near-misses |
| **Recall@3** | `1 if gt ∈ top-3 else 0` | Intermediate cut-off; bridges R@2 and R@5 to confirm recovery curve shape |
| **Recall@5** | `1 if gt ∈ top-5 else 0` | Standard shallow cut-off |
| **Recall@10** | `1 if gt ∈ top-10 else 0` | Primary retrieval cut-off; most baselines near-saturate here |
| **MRR@10** | `1/rank` of first relevant hit, 0 if rank > 10 | Ranking quality — how high the correct result appears; closely tracks R@1 |
| **Duplicate-Aware Recall@K** | Same as Recall@K, but any image in the same MD5 hash group counts as a hit | Accounts for 281 duplicate groups (589 images) with identical visual content |
| **Duplicate-Aware MRR@10** | `1/rank` of first hit from the MD5-equivalent set | Prevents penalising retrieval of pixel-identical images |

- All metrics computed per-query, then macro-averaged across the query set.
- Ground truth: each query maps to exactly one `ground_truth_row` index.
- Duplicate awareness: uses `eval/duplicate_mapping.json` (MD5 content hashes).
- Implemented in `eval/metrics.py`; `k_values=(1, 2, 3, 5, 10)` is the canonical call signature.

---

## 5. Hardware & Constraints

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA A40 (48 GB VRAM) |
| Environment | Conda: `/DATA1/prabhakar/llava_env` (Python 3.10) |
| Work Directory | `/DATA5/prabhakar/telecom_retrieval/` |
| Images | `/DATA5/prabhakar/telecom/extracted_images/images/` |
| HF Cache | `/DATA5/prabhakar/hf_cache/` |
| Framework | PyTorch 2.11.0 + Transformers 5.5.4 |
| Constraint | **Purely open-source models only. No paid APIs.** |
| Precision | `bfloat16` for inference (A40 supports it natively) |

---

## 6. File Layout

```
/DATA5/prabhakar/telecom_retrieval/
├── eval/
│   ├── duplicate_mapping.json      # MD5 hash → row indices
│   └── metrics.py                  # Evaluation module (Recall@K, MRR, dup-aware)
├── queries/
│   ├── q1_captions.json            # Q1 direct caption queries (3,766)
│   ├── q2_paraphrased.json         # Q2 LLM-paraphrased queries (3,766)
│   └── q3_context.json             # Q3 context-extracted queries (3,542)
├── scripts/
│   ├── 01_data_loader.py           # Data loading + MD5 duplicate detection
│   ├── 02_query_generator.py       # Q1 & Q3 generation
│   ├── 03_q2_paraphraser.py        # Q2 LLM paraphrasing
│   ├── 04_bm25_baselines.py        # M2: BM25 B1/B2 experiments
│   ├── 05_dense_baselines.py       # M3: BGE-base D1/D2 experiments
│   ├── 06_dense_large_baselines.py # M4: BGE-large L1/L2 experiments
│   ├── 07_clip_baseline.py         # M5: CLIP visual baseline
│   ├── 08_text_fusion_rerank.py    # M5.5: Text fusion and reranking
│   └── 09_ocr_baselines.py         # M6a: EasyOCR extraction and retrieval
├── indexes/                        # FAISS indexes and extracted data
│   └── m6_ocr_extracted_text.json  # M6a: Extracted OCR text
├── notebooks/                      # Exploration notebooks
├── cache/                          # Intermediate caches
├── logs/                           # Run logs (m2–m4 runs)
└── reports/
    ├── m2_bm25_results.json        # M2 BM25 full metrics
    ├── m3_dense_results.json       # M3 BGE-base full metrics
    ├── m4_dense_large_results.json # M4 BGE-large full metrics
    ├── m5_clip_results.json        # M5 CLIP full metrics
    ├── m55_text_fusion_rerank_results.json # M5.5 text fusion and rerank metrics
    ├── m6_ocr_quality_report.json  # M6a OCR quality stats
    ├── m6_ocr_results.json         # M6a OCR retrieval metrics
    ├── M2_walkthrough.md           # M2 analysis report
    ├── M3_walkthrough.md           # M3 analysis report
    ├── M4_walkthrough.md           # M4 analysis report
    ├── M5_walkthrough.md           # M5 analysis report
    ├── M55_text_fusion_rerank_walkthrough.md # M5.5 analysis report
    └── M6_ocr_visual_walkthrough.md # M6a analysis report
```

---

## 7. Source CSVs

| CSV | Path | Rows | Key Columns |
|-----|------|------|-------------|
| Paths CSV | `/DATA1/prabhakar/telecom/All Images Path.csv` | 3,766 | Context, Source, Subclause, Image Path, Image Caption |
| Metadata CSV | `/DATA1/prabhakar/telecom/thesis_diagram_analysis_final.xlsx - Diagram Analysis.csv` | 3,766 | Image Name, Matched Category, Agent 2–4 analysis, QA scores |
