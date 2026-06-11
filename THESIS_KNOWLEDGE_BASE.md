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
| BM25 | Lexical | `rank_bm25` over tokenised captions/context | Sparse retrieval baseline |
| BGE-base | Dense Semantic | `BAAI/bge-base-en-v1.5` | 768-d embeddings, encode text → FAISS |
| BGE-large | Dense Semantic | `BAAI/bge-large-en-v1.5` | 1024-d embeddings, stronger baseline |
| CLIP | Global Visual | `openai/clip-vit-base-patch32` or `openai/clip-vit-large-patch14` | Encode (query_text, image) pairs |
| ColPali | OCR-free Visual Document | `vidore/colpali-v1.3` | Late-interaction over page patches |
| Qwen2-VL | OCR-free Visual Document | `Qwen/Qwen2-VL-7B-Instruct` | Vision-language encoder |

---

## 4. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Recall@10** | `|relevant ∩ top-10| / |relevant|` | Measures if the correct image appears in top-10 results |
| **MRR@10** | `1/rank` of first relevant hit (0 if not in top-10) | Measures ranking quality — how high the correct result appears |
| **Duplicate-Aware Recall@10** | Same as Recall@10, but any image in the same MD5 hash group counts as a hit | Accounts for identical visual duplicates in the dataset |

- All metrics are computed per-query, then averaged across the query set.
- Ground truth: each query maps to exactly one `ground_truth_row` index.
- Duplicate awareness: uses `eval/duplicate_mapping.json` (MD5 content hashes).

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
│   └── duplicate_mapping.json      # MD5 hash → row indices
├── queries/
│   ├── q1_captions.json            # Q1 direct caption queries
│   ├── q2_paraphrased.json         # Q2 LLM-paraphrased queries (pending)
│   └── q3_context.json             # Q3 context-extracted queries
├── scripts/
│   ├── 01_data_loader.py           # Data loading + duplicate detection
│   ├── 02_query_generator.py       # Q1 & Q3 generation
│   └── 03_q2_paraphraser.py        # Q2 LLM paraphrasing (pending)
├── indexes/                        # FAISS / BM25 indexes (future)
├── models/                         # Model checkpoints (future)
├── notebooks/                      # Exploration notebooks
├── cache/                          # Intermediate caches
├── logs/                           # Run logs
└── reports/                        # Evaluation reports
```

---

## 7. Source CSVs

| CSV | Path | Rows | Key Columns |
|-----|------|------|-------------|
| Paths CSV | `/DATA1/prabhakar/telecom/All Images Path.csv` | 3,766 | Context, Source, Subclause, Image Path, Image Caption |
| Metadata CSV | `/DATA1/prabhakar/telecom/thesis_diagram_analysis_final.xlsx - Diagram Analysis.csv` | 3,766 | Image Name, Matched Category, Agent 2–4 analysis, QA scores |
