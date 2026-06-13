# Thesis Master State
## Multimodal Image Retrieval for Telecom Technical Diagrams

> Last updated: 2026-06-13T00:29+05:30

---

## Milestone Tracker

| Milestone | Description | Status | Date |
|-----------|-------------|--------|------|
| M1.0 | Environment audit, directory setup, path verification | ✅ Complete | 2026-06-12 |
| M1.1 | Data loader + MD5 duplicate mapping (281 groups, 589 dupes) | ✅ Complete | 2026-06-12 |
| M1.2 | Q1 (3,766 captions) + Q3 (3,542 context queries) generation | ✅ Complete | 2026-06-12 |
| M1.3 | Knowledge base creation + Q2 paraphrase generation (local LLM) | ✅ Complete | 2026-06-12 |
| M2 | BM25 lexical baseline (B1 Caption, B2 Cap+Ctx) | ✅ Complete | 2026-06-12 |
| M3 | BGE-base dense embedding index (D1 Caption, D2 Cap+Ctx) | ✅ Complete | 2026-06-12 |
| M4 | BGE-large dense embedding index (L1 Caption, L2 Cap+Ctx) | ✅ Complete | 2026-06-13 |
| M5 | CLIP global visual baseline | ✅ Complete | 2026-06-13 |
| M5.5 | Text fusion and rank-1 reranking | ✅ Complete | 2026-06-13 |
| M6a | EasyOCR extraction and BM25 text baselines | ✅ Complete | 2026-06-13 |
| M6b | ColPali OCR-free visual document retrieval | ⬜ **Next** | — |
| M7 | Qwen2-VL visual document retrieval | ⬜ Pending | — |
| M8 | Cross-baseline evaluation & ablation tables | ⬜ Pending | — |
| M9 | Hybrid retrieval experiments | ⬜ Pending | — |
| M10 | Error analysis & qualitative examples | ⬜ Pending | — |
| M11 | Final thesis figures & report generation | ⬜ Pending | — |

---

## Active Artifacts

| File | Path | Status |
|------|------|--------|
| Duplicate mapping | `eval/duplicate_mapping.json` | ✅ Ready |
| Q1 queries | `queries/q1_captions.json` (3,766) | ✅ Ready |
| Q3 queries | `queries/q3_context.json` (3,542) | ✅ Ready |
| Q2 queries | `queries/q2_paraphrased.json` (3,766) | ✅ Ready |
| Knowledge base | `THESIS_KNOWLEDGE_BASE.md` | ✅ Ready |
| Eval module | `eval/metrics.py` | ✅ Ready |
| BM25 results | `reports/m2_bm25_results.json` | ✅ Ready |
| Dense script | `scripts/05_dense_baselines.py` | ✅ Ready |
| Dense results | `reports/m3_dense_results.json` | ✅ Ready |
| M3 walkthrough | `reports/M3_walkthrough.md` | ✅ Ready |
| Dense large script | `scripts/06_dense_large_baselines.py` | ✅ Ready |
| Dense large results | `reports/m4_dense_large_results.json` | ✅ Ready |
| M4 walkthrough | `reports/M4_walkthrough.md` | ✅ Ready |
| Text fusion script | `scripts/08_text_fusion_rerank.py` | ✅ Ready |
| M5.5 results | `reports/m55_text_fusion_rerank_results.json` | ✅ Ready |
| M5.5 walkthrough | `reports/M55_text_fusion_rerank_walkthrough.md` | ✅ Ready |
| M6a OCR script | `scripts/09_ocr_baselines.py` | ✅ Ready |
| M6a OCR index | `indexes/m6_ocr_extracted_text.json` | ✅ Ready |
| M6a OCR results | `reports/m6_ocr_results.json` | ✅ Ready |
| M6a walkthrough | `reports/M6_ocr_visual_walkthrough.md` | ✅ Ready |
---

## Environment

- **Conda env**: `/DATA1/prabhakar/llava_env` (Python 3.10)
- **GPUs**: 4× NVIDIA A40 (48 GB each)
- **Key packages**: torch 2.11.0, transformers 5.5.4, faiss-gpu 1.7.2, pandas 2.3.3, rank_bm25 0.2.2, imagehash 4.3.2
- **HF cache**: `/DATA5/prabhakar/hf_cache/`
