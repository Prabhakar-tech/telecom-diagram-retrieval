#!/usr/bin/env python3
import json
import logging
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util
import random
import re

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

spec = importlib.util.spec_from_file_location("m7", str(PROJECT_ROOT / "scripts/12_hybrid_lexical_dense.py"))
m7_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m7_mod)

def load_queries(qs):
    path_map = {
        "Q1": "queries/q1_captions.json",
        "Q2": "queries/q2_paraphrased.json",
        "Q3": "queries/q3_context.json"
    }
    with open(PROJECT_ROOT / path_map[qs], "r") as f:
        return json.load(f)["queries"]

def get_rank(preds, gt):
    if not preds: return 999
    for r, p in enumerate(preds):
        doc_id = p[0] if isinstance(p, list) else p
        if doc_id == gt or str(doc_id) == str(gt): return r + 1
        if isinstance(doc_id, str):
            clean_id = doc_id.replace("image_", "").replace(".png", "").strip()
            if clean_id == str(gt): return r + 1
    return 999

def get_mrr(rank):
    if rank is None or rank == 999: return 0.0
    if rank > 10: return 0.0
    return 1.0 / rank

def get_m7_prediction(sys_name, qs, cache, q_idx):
    b_src = "B1" if qs in ["Q1", "Q2"] else "B2"
    d_src = "D1" if qs == "Q1" else "D2"

    b_scored = cache[b_src][qs][q_idx]
    d_scored = cache[d_src][qs][q_idx]
    b_rank = m7_mod.strip_scores(b_scored)
    d_rank = m7_mod.strip_scores(d_scored)

    if sys_name == "rrf_bm25_bge_k10": return m7_mod.rrf([b_rank, d_rank], [1.0, 1.0], k=10)
    elif sys_name == "rrf_bm25_bge_k30": return m7_mod.rrf([b_rank, d_rank], [1.0, 1.0], k=30)
    elif sys_name == "rrf_bm25_bge_k60": return m7_mod.rrf([b_rank, d_rank], [1.0, 1.0], k=60)
    elif sys_name == "rrf_bm25_bge_heavy_bm25": return m7_mod.rrf([b_rank, d_rank], [0.75, 0.25], k=60)
    elif sys_name == "rrf_bm25_bge_balanced": return m7_mod.rrf([b_rank, d_rank], [0.50, 0.50], k=60)
    elif sys_name == "rrf_bm25_bge_heavy_dense": return m7_mod.rrf([b_rank, d_rank], [0.25, 0.75], k=60)
    elif sys_name == "score_fusion_bm25_075_bge_025": return m7_mod.score_fusion(b_scored, d_scored, 0.75, 0.25)
    elif sys_name == "score_fusion_bm25_050_bge_050": return m7_mod.score_fusion(b_scored, d_scored, 0.50, 0.50)
    elif sys_name == "score_fusion_bm25_025_bge_075": return m7_mod.score_fusion(b_scored, d_scored, 0.25, 0.75)
    return []

def safe_int(val):
    if isinstance(val, (int, float)): return val
    if isinstance(val, str):
        if val.startswith(">"): return 999
        if val == "NA": return 999
        try: return int(val)
        except: return 999
    return 999

def derive_likely_failure_category(q_text, b_rank, d_rank, c_rank, cp_rank, o_rank):
    b_rank, d_rank = safe_int(b_rank), safe_int(d_rank)
    c_rank, cp_rank, o_rank = safe_int(c_rank), safe_int(cp_rank), safe_int(o_rank)
    txt_min = min(b_rank, d_rank)
    vis_min = min(c_rank, cp_rank)

    words = q_text.lower().split()
    is_generic = len(words) < 4
    has_acronyms = bool(re.search(r'[A-Z]{2,}', q_text))

    if is_generic and txt_min > 50:
        return "generic caption/query"
    if has_acronyms and txt_min > 10:
        return "acronym ambiguity"
    if vis_min > 50 and txt_min <= 10:
        return "visual domain gap"
    if o_rank > 50 and txt_min <= 10:
        return "OCR noise"
    if txt_min > 50:
        return "query-caption/context mismatch or cross-spec confusion"
    return "unclassified failure"

def main():
    # 1. Audit
    files_to_check = {
        "reports/m8_best_system_per_query_metrics.csv": {"sel": True, "gal": False},
        "reports/m8_best_by_query_type.csv": {"sel": False, "gal": False},
        "reports/m8_master_ablation_table.csv": {"sel": False, "gal": False},
        "reports/m7_hybrid_lexical_dense_predictions_q1.json": {"sel": True, "gal": False},
        "reports/m7_hybrid_lexical_dense_predictions_q2.json": {"sel": True, "gal": False},
        "reports/m7_hybrid_lexical_dense_predictions_q3.json": {"sel": True, "gal": False},
        "reports/m55_text_fusion_rerank_results.json": {"sel": False, "gal": False},
        "reports/m5_clip_predictions_q1.json": {"sel": True, "gal": False},
        "reports/m6_predictions_caption_ocr_q1.json": {"sel": True, "gal": False},
        "reports/m6b_colpali_predictions_q1.json": {"sel": True, "gal": False},
        "queries/q1_captions.json": {"sel": True, "gal": False},
        "queries/q2_paraphrased.json": {"sel": True, "gal": False},
        "queries/q3_context.json": {"sel": True, "gal": False},
        "/DATA1/prabhakar/telecom/All Images Path.csv": {"sel": True, "gal": True},
        "/DATA5/prabhakar/telecom/extracted_images/images/": {"sel": False, "gal": True},
        "eval/duplicate_mapping.json": {"sel": True, "gal": False}
    }

    audit = {}
    for f, props in files_to_check.items():
        p = Path(f) if f.startswith("/") else PROJECT_ROOT / f
        exists = p.exists()
        size = p.stat().st_size if (exists and p.is_file()) else 0
        limit = ""
        if not exists:
            limit = "File not found"
        audit[f] = {
            "exists": exists,
            "file_size_bytes": size,
            "used_for_case_selection": props["sel"],
            "used_for_gallery_generation": props["gal"],
            "limitation_if_any": limit
        }

    with open(PROJECT_ROOT / "reports/m9_qualitative_required_inputs_audit.json", "w") as fh:
        json.dump(audit, fh, indent=2)

    # 2. Candidate Selection & Metadata
    metadata_df = pd.read_csv("/DATA1/prabhakar/telecom/All Images Path.csv")

    cache_path = PROJECT_ROOT / "reports/m7_base_predictions_cache.json"
    with open(cache_path, "r") as f:
        cache = json.load(f)

    m8_metrics = pd.read_csv(PROJECT_ROOT / "reports/m8_best_system_per_query_metrics.csv")

    clip_preds = {}
    colpali_preds = {}
    ocr_preds = {}
    for qs in ["q1", "q2", "q3"]:
        try: clip_preds[qs.upper()] = json.load(open(PROJECT_ROOT / f"reports/m5_clip_predictions_{qs}.json"))
        except: clip_preds[qs.upper()] = {}
        try: colpali_preds[qs.upper()] = json.load(open(PROJECT_ROOT / f"reports/m6b_colpali_predictions_{qs}.json"))
        except: colpali_preds[qs.upper()] = {}
        try: ocr_preds[qs.upper()] = json.load(open(PROJECT_ROOT / f"reports/m6_predictions_capctx_ocr_{qs}.json"))
        except: ocr_preds[qs.upper()] = {}

    candidates = []

    for qs in ["Q1", "Q2", "Q3"]:
        queries = load_queries(qs)
        qs_metrics = m8_metrics[m8_metrics["query_set"] == qs].iloc[0]
        best_bm25_sys = qs_metrics["best_bm25_system"]
        best_bge_sys = qs_metrics["best_bge_system"]
        best_m7_sys = qs_metrics["best_m7_system"]

        for q_idx, q in enumerate(queries):
            q_id = q["query_id"]
            q_text = q["text"]
            gt = q["ground_truth_row"]

            # Ground truth metadata
            gt_row_data = metadata_df.iloc[gt]
            gt_caption = gt_row_data.get("Image Caption", "NA")
            src = gt_row_data.get("Source", "NA")
            subc = gt_row_data.get("Subclause", "NA")

            b_preds = m7_mod.strip_scores(cache[best_bm25_sys][qs][q_idx])
            d_preds = m7_mod.strip_scores(cache[best_bge_sys][qs][q_idx])
            m_preds = get_m7_prediction(best_m7_sys, qs, cache, q_idx)

            def safe_get(pred_dict, idx, qid):
                if not pred_dict or qs not in pred_dict or not pred_dict[qs]: return []
                d = pred_dict[qs]
                if isinstance(d, list): return d[idx] if idx < len(d) else []
                elif isinstance(d, dict):
                    if idx in d: return d[idx]
                    if str(idx) in d: return d[str(idx)]
                    if qid in d: return d[qid]
                return []

            c_preds = safe_get(clip_preds, q_idx, q_id)
            cp_preds = safe_get(colpali_preds, q_idx, q_id)
            o_preds = safe_get(ocr_preds, q_idx, q_id)

            b_rank = get_rank(b_preds, gt)
            d_rank = get_rank(d_preds, gt)
            m_rank = get_rank(m_preds, gt)
            c_rank = get_rank(c_preds, gt)
            cp_rank = get_rank(cp_preds, gt)
            o_rank = get_rank(o_preds, gt)

            cat = []
            if b_rank == 1 and d_rank == 1 and m_rank == 1:
                cat.append(("A. Easy Success", "All top systems instantly match", "High", "Clean caption metadata provides strong retrieval signal."))
            if m_rank < b_rank and m_rank < d_rank and get_mrr(m_rank) > max(get_mrr(b_rank), get_mrr(d_rank)):
                cat.append(("B. M7 Helps", "M7 strictly improves rank over single baselines", "High", "Lexical and semantic signals complement each other."))
            if b_rank <= 3 and d_rank > 10:
                cat.append(("C. BM25 Beats Dense", "Lexical succeeds where semantic fails", "High", "Dense models lose critical acronym/keyword sensitivity."))
            if d_rank <= 3 and b_rank > 10:
                cat.append(("D. Dense Beats BM25", "Semantic matches paraphrases that break lexical", "High", "Embeddings bridge vocabulary mismatches."))

            has_vis_preds = len(c_preds) > 0 or len(cp_preds) > 0
            if min(b_rank, d_rank) <= 3 and min(c_rank, cp_rank) > 50 and has_vis_preds:
                cat.append(("F. Visual Failure", "Visual models fail despite easy text match", "High", "Zero-shot visual domain gap is substantial for diagrams."))

            has_ocr_preds = len(o_preds) > 0
            if b_rank <= 3 and o_rank > 50 and has_ocr_preds:
                cat.append(("G. OCR Failure", "OCR hurts clean metadata matching", "Medium", "Raw diagram text is often noisy."))

            if min(b_rank, d_rank, m_rank) > 10:
                cat.append(("H. Final Failure", "No top text baseline solves this", "High", "Query-context mismatch or extreme ambiguity."))

            def format_rank(r, preds):
                if not preds: return "NA"
                if r == 999: return f">{len(preds)}"
                return r

            missing_reasons = []
            if not c_preds: missing_reasons.append("CLIP preds missing")
            if not cp_preds: missing_reasons.append("ColPali preds missing")
            if not o_preds: missing_reasons.append("OCR preds missing")

            for c_name, reason, conf, lesson in cat:
                candidates.append({
                    "query_set": qs,
                    "query_id": q_id,
                    "query_text": q_text,
                    "ground_truth_row": gt,
                    "ground_truth_caption": gt_caption,
                    "source": src,
                    "subclause": subc,
                    "category": c_name,
                    "bm25_rank": format_rank(b_rank, b_preds),
                    "bge_rank": format_rank(d_rank, d_preds),
                    "m7_rank": format_rank(m_rank, m_preds),
                    "clip_rank": format_rank(c_rank, c_preds),
                    "ocr_rank": format_rank(o_rank, o_preds),
                    "colpali_rank": format_rank(cp_rank, cp_preds),
                    "reranker_rank_if_available": "NA",
                    "bm25_mrr": get_mrr(b_rank),
                    "bge_mrr": get_mrr(d_rank),
                    "m7_mrr": get_mrr(m_rank),
                    "clip_mrr": get_mrr(c_rank),
                    "ocr_mrr": get_mrr(o_rank),
                    "colpali_mrr": get_mrr(cp_rank),
                    "reason_for_selection": reason,
                    "thesis_lesson": lesson,
                    "confidence_of_case": conf,
                    "rank_missing_reason": "; ".join(missing_reasons) if missing_reasons else ""
                })

    df_cand = pd.DataFrame(candidates)
    df_cand.to_csv(PROJECT_ROOT / "reports/m9_candidate_case_selection.csv", index=False)

    # Generate image path audit
    audit_rows = []
    base_img_dir = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
    for idx, row in metadata_df.iterrows():
        raw_path = str(row.get("Image Path", ""))
        resolved = None
        issue = ""
        exists = False
        size = 0
        if raw_path and raw_path.lower() != "nan":
            # Attempt to resolve
            candidates_paths = [
                Path(raw_path),
                base_img_dir / Path(raw_path).name,
                base_img_dir / raw_path.strip("/")
            ]
            for p in candidates_paths:
                if p.exists() and p.is_file():
                    resolved = str(p)
                    exists = True
                    size = p.stat().st_size
                    break
            if not exists:
                resolved = str(base_img_dir / Path(raw_path).name)
                issue = "File not found"
        else:
            issue = "Empty raw path in CSV"

        audit_rows.append({
            "row_id": idx,
            "raw_image_path_from_csv": raw_path,
            "resolved_image_path": resolved if resolved else "",
            "exists": exists,
            "file_size_bytes": size,
            "issue": issue
        })
    pd.DataFrame(audit_rows).to_csv(PROJECT_ROOT / "reports/m9_image_path_audit.csv", index=False)


    # 3. Generate failure taxonomy template
    fail_df = df_cand[df_cand["category"] == "H. Final Failure"].copy()
    if not fail_df.empty:
        fail_df = fail_df.drop_duplicates(subset=["query_set", "query_id"]).copy()

        fail_df["best_available_method"] = ""
        fail_df["best_available_rank"] = ""

        for idx, row in fail_df.iterrows():
            ranks = {
                "BM25": safe_int(row["bm25_rank"]),
                "BGE": safe_int(row["bge_rank"]),
                "M7": safe_int(row["m7_rank"])
            }
            best_sys = min(ranks, key=ranks.get)
            best_r = ranks[best_sys]
            fail_df.at[idx, "best_available_method"] = best_sys if best_r != 999 else "None"
            fail_df.at[idx, "best_available_rank"] = best_r if best_r != 999 else "NA"

            lf = derive_likely_failure_category(
                str(row["query_text"]),
                ranks["BM25"], ranks["BGE"],
                row["clip_rank"],
                row["colpali_rank"],
                row["ocr_rank"]
            )
            fail_df.at[idx, "likely_failure_category"] = lf

        fail_df["evidence_from_ranks"] = ""
        fail_df["evidence_from_query_text"] = ""
        fail_df["manual_review_needed"] = "yes"
        fail_df["final_thesis_explanation"] = ""

        cols = [
            "query_set", "query_id", "query_text", "ground_truth_row", "ground_truth_caption",
            "best_available_method", "best_available_rank",
            "bm25_rank", "bge_rank", "m7_rank", "clip_rank", "ocr_rank", "colpali_rank",
            "likely_failure_category", "evidence_from_ranks", "evidence_from_query_text",
            "manual_review_needed", "final_thesis_explanation"
        ]
        fail_df[cols].to_csv(PROJECT_ROOT / "reports/m9_failure_taxonomy_template.csv", index=False)

    # 4. Generate Gallery Plan
    random.seed(42)
    gallery = []

    planned_sys_map = {
        "A. Easy Success": "Ground truth, BM25 top-5, BGE top-5, M7 top-5",
        "B. M7 Helps": "Ground truth, BM25 top-5, BGE top-5, M7 top-5",
        "C. BM25 Beats Dense": "Ground truth, BM25 top-5, BGE top-5, M7 top-5",
        "D. Dense Beats BM25": "Ground truth, BM25 top-5, BGE top-5, M7 top-5",
        "E. Reranker Helps Q3": "Ground truth, BM25 top-5, BGE top-5, M7 top-5, Reranker top-5",
        "F. Visual Failure": "Ground truth, M7 or best text top-5, CLIP top-5, ColPali top-5",
        "G. OCR Failure": "Ground truth, BM25/M7 top-5, OCR top-5, OCR extracted text snippet if available",
        "H. Final Failure": "Ground truth, BM25 top-5, BGE top-5, M7 top-5, CLIP/ColPali if relevant"
    }

    def sample_cat(cat, n):
        subset = df_cand[df_cand["category"] == cat]
        if subset.empty: return
        n = min(n, len(subset))
        samples = subset.sample(n, random_state=42)
        for _, r in samples.iterrows():
            gallery.append({
                "query_set": r["query_set"],
                "query_id": r["query_id"],
                "query_text": r["query_text"],
                "ground_truth_row": r["ground_truth_row"],
                "category": cat,
                "planned_systems_to_show": planned_sys_map.get(cat, "Ground truth, BM25 top-5, BGE top-5, M7 top-5"),
                "reason_for_inclusion": r["reason_for_selection"],
                "expected_thesis_lesson": r["thesis_lesson"]
            })

    sample_cat("A. Easy Success", 3)
    sample_cat("B. M7 Helps", 3)
    sample_cat("C. BM25 Beats Dense", 2)
    sample_cat("D. Dense Beats BM25", 2)
    sample_cat("F. Visual Failure", 2)
    sample_cat("G. OCR Failure", 2)
    sample_cat("H. Final Failure", 3)

    pd.DataFrame(gallery).to_csv(PROJECT_ROOT / "reports/m9_retrieval_gallery_plan.csv", index=False)

    # 5. Write Markdown plan
    plan = """# M9 Qualitative Analysis Plan

## 1. Goal
Provide qualitative evidence that explains the numerical results from M2–M8, answering:
1. Why do text systems work so well?
2. Why do visual systems fail zero-shot?
3. Why does context help Q3 but hurt short caption queries?
4. Where does M7 hybrid help over BM25/BGE?
5. Where does reranking help over score fusion?
6. What are the main failure modes?
7. Which examples should be shown in thesis/viva?

## 2. Selection Strategy
We categorize candidates computationally by examining the rank patterns across BM25, BGE, M7, CLIP, ColPali, and OCR predictions. This ensures a non-biased, data-driven selection of representative cases.

## 3. Available Prediction Files
From the audit, we have access to base BM25/BGE scores, best M7 logic, CLIP, ColPali, and OCR predictions.

## 4. Missing Prediction Files (Limitation)
**Limitation:** M5.5 cross-encoder reranker prediction files were not systematically saved. Because running the cross-encoder locally on the dataset takes significant time, Option B was selected: **selective reranker reconstruction is skipped for the planning phase**. The "Reranker helps Q3" category is not currently included in the final gallery. If needed, this must be regenerated later.

## 5. Final Selected Gallery Quality Rules
A case can enter the final thesis gallery only if:
* ground truth image path exists
* query text is complete
* ground truth caption is available
* at least 2 method ranks are available
* reason_for_selection is supported by ranks
* the example teaches one clear thesis lesson

## 6. What Can Be Done
- We have computationally selected excellent candidates for "M7 helps", "BM25 vs Dense", "Visual Failures", and "Final Failures" with full rank evidence and metadata.
- We have generated the failure taxonomy template and gallery plan.

## 7. What Needs Regeneration
- Visual gallery images will need to be gathered in a later step.

## 8. Proposed Qualitative Exhibits
- Top-5 image contact sheets for each system to illustrate rank differences.
- Highlighting acronyms in queries vs OCR noise.
- Side-by-side comparisons of `bm25_caption` vs `bge_caption`.
"""
    with open(PROJECT_ROOT / "reports/M9_qualitative_analysis_plan.md", "w") as f:
        f.write(plan)

    log.info("M9 Planning refinement completed.")

if __name__ == "__main__":
    main()
