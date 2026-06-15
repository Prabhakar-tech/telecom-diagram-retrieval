# Thesis Master State
## Multimodal Image Retrieval for Telecom Technical Diagrams

> Last updated: 2026-06-15T22:51+05:30

---

## Milestone Tracker

| Milestone | Description | Status | Date |
|-----------|-------------|--------|------|
| M1.0 | Environment audit, directory setup, path verification | ✅ Complete | 2026-06-13 |
| M1.1 | Data loader + MD5 duplicate mapping (281 groups, 589 dupes) | ✅ Complete | 2026-06-13 |
| M1.2 | Q1 (3,766 captions) + Q3 (3,542 context queries) generation | ✅ Complete | 2026-06-13 |
| M1.3 | Knowledge base creation + Q2 paraphrase generation (local LLM) | ✅ Complete | 2026-06-13 |
| M2 | BM25 lexical baseline (B1 Caption, B2 Cap+Ctx) | ✅ Complete | 2026-06-13 |
| M3 | BGE-base dense embedding index (D1 Caption, D2 Cap+Ctx) | ✅ Complete | 2026-06-13 |
| M4 | BGE-large dense embedding index (L1 Caption, L2 Cap+Ctx) | ✅ Complete | 2026-06-13 |
| M5 | CLIP global visual baseline | ✅ Complete | 2026-06-13 |
| M5.5 | Text fusion and rank-1 reranking | ✅ Complete | 2026-06-13 |
| M6a | EasyOCR extraction and BM25 text baselines | ✅ Complete | 2026-06-13 |
| M6b | ColPali OCR-free visual document retrieval | ✅ Complete | 2026-06-13 |
| M6.5 | Domain Acronym Expansion & Query Rewriting | ✅ Complete | 2026-06-14 |
| M7 | Hybrid Lexical + Dense Text Retrieval | ✅ Complete | 2026-06-15 |
| M8 | Master Ablation + Statistical Validation | ✅ Complete | 2026-06-15 |
| M9 | Qualitative Error Analysis + Retrieval Gallery | ⬜ Pending | — |
| M10 | Final Evidence-Based Architecture + Defense Notes | ⬜ Pending | — |
| M9A | Visual Domain Adaptation Pilot Planning | ✅ Complete | 2026-06-16 |
| M9A_E0 | Zero-shot CLIP on duplicate-safe held-out test split | ✅ Complete | 2026-06-16 |
| Optional | Qwen2-VL / ColQwen2 visual document retrieval | ⬜ Optional | — |
| Optional | Source-aware / structure-aware retrieval | ⬜ Optional | — |

---

## Active Artifacts

| File | Path | Status |
|------|------|--------|
| M8 ablation table | `reports/m8_master_ablation_table.csv` | ✅ Ready |
| M8 best-by-type | `reports/m8_best_by_query_type.csv` | ✅ Ready |
| M8 mod comparison | `reports/m8_modality_comparison_table.csv` | ✅ Ready |
| M8 paired comp | `reports/m8_paired_comparison_summary.csv` | ✅ Ready |
| M8 stats JSON | `reports/m8_statistical_validation.json` | ✅ Ready |
| M8 walkthrough | `reports/M8_master_ablation_walkthrough.md` | ✅ Ready |
| M8 audit/stats scripts | `scripts/13_m8_audit_and_ablation.py`, `scripts/14_m8_statistical_validation.py` | ✅ Ready |
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
| M6b ColPali index | `indexes/colpali_index/` | ✅ Ready |
| M6b walkthrough | `reports/M6b_colpali_walkthrough.md` | ✅ Ready |
| M6.5 Lexicon | `reports/m65_acronym_lexicon.json` | ✅ Ready |
| M6.5 expansion script| `scripts/11_acronym_expansion.py` | ✅ Ready |
| M6.5 results | `reports/m65_acronym_expansion_results.json` | ✅ Ready |
| M6.5 walkthrough | `reports/M65_acronym_expansion_walkthrough.md` | ✅ Ready |
| M7 hybrid script | `scripts/12_hybrid_lexical_dense.py` | ✅ Ready |
| M7 plan | `reports/M7_hybrid_lexical_dense_plan.md` | ✅ Ready |
| M7 results | `reports/m7_hybrid_lexical_dense_results.json` | ✅ Ready |
| M7 win/loss matrix | `reports/m7_per_query_win_loss.csv` | ✅ Ready |
| M7 walkthrough | `reports/M7_hybrid_lexical_dense_walkthrough.md` | ✅ Ready |
| M9A planning document | `reports/M9A_visual_domain_adaptation_plan.md` | ✅ Ready |
| M9A feasibility audit | `reports/m9a_visual_adaptation_feasibility_audit.json` | ✅ Ready |
| M9A split strategy | `reports/m9a_duplicate_safe_split_strategy.md` | ✅ Ready |
| M9A expected experiments | `reports/m9a_expected_experiments.csv` | ✅ Ready |
| M9A image caption pairs | `reports/m9a_image_caption_pair_audit.csv` | ✅ Ready |
| M9A split summary | `reports/m9a_dry_run_split_summary.json` | ✅ Ready |
| M9A duplicate audit | `reports/m9a_duplicate_group_audit.csv` | ✅ Ready |
| M9A planning script | `scripts/16_m9a_visual_adaptation_planning.py` | ✅ Ready |
| M9A E0 split rows | `data/m9a_splits/train_rows.json`, `val_rows.json`, `test_rows.json` | ✅ Ready |
| M9A E0 split audit | `reports/m9a_final_split_audit.json` | ✅ Ready |
| M9A E0 results | `reports/m9a_e0_zeroshot_clip_test_results.json` | ✅ Ready |
| M9A E0 predictions | `reports/m9a_e0_zeroshot_clip_test_predictions.json` | ✅ Ready |
| M9A E0 comparison | `reports/m9a_visual_adaptation_comparison.csv` | ✅ Ready |
| M9A E0 walkthrough | `reports/M9A_E0_zeroshot_clip_test_walkthrough.md` | ✅ Ready |
| M9A E0 test script | `scripts/17_m9a_e0_zeroshot_clip_test.py` | ✅ Ready |
---

## Environment

- **Conda env**: `/DATA1/prabhakar/llava_env` (Python 3.10)
- **GPUs**: 4× NVIDIA A40 (48 GB each)
- **Key packages**: torch 2.11.0, transformers 5.5.4, faiss-gpu 1.7.2, pandas 2.3.3, rank_bm25 0.2.2, imagehash 4.3.2
- **HF cache**: `/DATA5/prabhakar/hf_cache/`
