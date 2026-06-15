import os
import json
import csv
import traceback
from pathlib import Path

REPORTS_DIR = Path("reports")

REQUIRED_FILES = [
    "m2_bm25_results.json",
    "m3_dense_results.json",
    "m4_dense_large_results.json",
    "m5_clip_results.json",
    "m55_text_fusion_rerank_results.json",
    "m6_ocr_results.json",
    "m6b_colpali_results.json",
    "m65_acronym_expansion_results.json",
    "m7_hybrid_lexical_dense_results.json",
    "master_results.csv",
    "m7_hybrid_lexical_dense_predictions_q1.json",
    "m7_hybrid_lexical_dense_predictions_q2.json",
    "m7_hybrid_lexical_dense_predictions_q3.json"
]

def get_method_family(sys_id, milestone):
    sys_id_lower = sys_id.lower()
    if milestone in ["M2"]: return "lexical text"
    if milestone in ["M3", "M4"]: return "dense text"
    if milestone == "M5": return "global visual"
    if milestone == "M6a": return "OCR text"
    if milestone == "M6b": return "OCR-free visual-document"
    if milestone == "M6.5":
        if "pure" in sys_id_lower: return "acronym query expansion"
        return "hybrid text" # If it's fusion
    if milestone in ["M5.5", "M7"]: return "hybrid text"
    
    # Fallbacks based on name
    if "bm25" in sys_id_lower and "bge" not in sys_id_lower: return "lexical text"
    if "bge" in sys_id_lower and "bm25" not in sys_id_lower: return "dense text"
    if "ocr" in sys_id_lower: return "OCR text"
    if "clip" in sys_id_lower: return "global visual"
    if "colpali" in sys_id_lower: return "OCR-free visual-document"
    return "hybrid text"

def add_normalized_row(ms, sys_id, qs, metrics, fname, normalized_data):
    existing = [d for d in normalized_data if d["system_id"] == sys_id and d["query_set"] == qs and d["milestone"] == ms]
    if not existing:
        normalized_data.append({
            "milestone": ms,
            "system_id": sys_id,
            "method_family": get_method_family(sys_id, ms),
            "query_set": qs,
            "metrics": {
                "recall@1": metrics.get("recall@1", 0),
                "recall@2": metrics.get("recall@2", 0),
                "recall@5": metrics.get("recall@5", 0),
                "recall@10": metrics.get("recall@10", 0),
                "mrr@10": metrics.get("mrr@10", 0),
                "dup_recall@1": metrics.get("dup_recall@1", 0),
                "dup_mrr@10": metrics.get("dup_mrr@10", 0),
                "num_queries": metrics.get("num_queries", 0)
            },
            "source_result_file": fname,
            "notes": ""
        })
    else:
        existing[0]["metrics"].update({
            "num_queries": metrics.get("num_queries", 0)
        })
        if metrics.get("dup_mrr@10") and not existing[0]["metrics"].get("dup_mrr@10"):
            existing[0]["metrics"]["dup_mrr@10"] = metrics["dup_mrr@10"]
            existing[0]["metrics"]["dup_recall@1"] = metrics.get("dup_recall@1", 0)

def main():
    audit_log = {}
    normalized_data = []

    # 1. Read master_results.csv as baseline data
    master_csv_path = REPORTS_DIR / "master_results.csv"
    if master_csv_path.exists():
        audit_log["master_results.csv"] = {
            "exists": True, "file_size": os.path.getsize(master_csv_path),
            "used_for_aggregate_metrics": True, "used_for_per_query_statistics": False,
            "reason_if_not_used": ""
        }
        with open(master_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sys_id = row.get("Experiment", "")
                if not sys_id: continue
                milestone = row.get("Milestone", "")
                qs = row.get("QuerySet", "") or row.get("Query_Set", "")
                if ":" in qs: qs = qs.split(":")[0].strip() # e.g. "Q1: Captions" -> "Q1"
                
                metrics = {
                    "recall@1": float(row.get("R@1", row.get("Recall@1", 0)) or 0),
                    "recall@2": float(row.get("R@2", 0) or 0),
                    "recall@5": float(row.get("R@5", row.get("Recall@5", 0)) or 0),
                    "recall@10": float(row.get("R@10", row.get("Recall@10", 0)) or 0),
                    "mrr@10": float(row.get("MRR@10", 0) or 0),
                    "dup_recall@1": float(row.get("dupR@1", 0) or 0),
                    "dup_mrr@10": float(row.get("dupMRR@10", 0) or 0),
                    "num_queries": 0
                }
                normalized_data.append({
                    "milestone": milestone,
                    "system_id": sys_id,
                    "method_family": get_method_family(sys_id, milestone),
                    "query_set": qs,
                    "metrics": metrics,
                    "source_result_file": "master_results.csv",
                    "notes": ""
                })
    else:
        audit_log["master_results.csv"] = {
            "exists": False, "file_size": 0, "used_for_aggregate_metrics": False, 
            "used_for_per_query_statistics": False, "reason_if_not_used": "File not found"
        }

    # 2. Parse all JSON result files
    for fname in REQUIRED_FILES:
        if fname == "master_results.csv": continue
        
        fpath = REPORTS_DIR / fname
        is_prediction = "predictions" in fname
        
        if not fpath.exists():
            audit_log[fname] = {
                "exists": False, "file_size": 0, "used_for_aggregate_metrics": False,
                "used_for_per_query_statistics": False, "reason_if_not_used": "File not found"
            }
            continue
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            audit_log[fname] = {
                "exists": True, "file_size": os.path.getsize(fpath),
                "used_for_aggregate_metrics": not is_prediction,
                "used_for_per_query_statistics": is_prediction,
                "reason_if_not_used": ""
            }
            
            # If it's a prediction file, we don't extract aggregate metrics here
            if is_prediction: continue
            
            # Schema normalization
            if isinstance(data, dict):
                # Format 1: {"experiments": {"sys_id": {"results": {"Q1": {...}}}}}
                if "experiments" in data:
                    systems = data["experiments"]
                    for sys_id, sys_data in systems.items():
                        results = sys_data.get("results", {})
                        for qs, metrics in results.items():
                            if qs in ["Q1", "Q2", "Q3"] and isinstance(metrics, dict):
                                ms = data.get("milestone", fname.split("_")[0].upper())
                                add_normalized_row(ms, sys_id, qs, metrics, fname, normalized_data)
                # Format 2: {"sys_id": {"Q1": {"mrr@10": ...}}}
                else:
                    for sys_id, qs_data in data.items():
                        if isinstance(qs_data, dict):
                            for qs, metrics in qs_data.items():
                                if qs in ["Q1", "Q2", "Q3"] and isinstance(metrics, dict):
                                    ms = fname.split("_")[0].upper()
                                    add_normalized_row(ms, sys_id, qs, metrics, fname, normalized_data)
                                    
        except Exception as e:
            audit_log[fname] = {
                "exists": True, "file_size": os.path.getsize(fpath),
                "used_for_aggregate_metrics": False, "used_for_per_query_statistics": False,
                "reason_if_not_used": f"parse_failed: {str(e)}"
            }

    with open(REPORTS_DIR / "m8_required_results_audit.json", "w") as f:
        json.dump(audit_log, f, indent=2)

    # 3. Create m8_master_ablation_table.csv
    csv_columns = [
        "milestone", "system_id", "method_family", "text_input", "image_input", "fusion_type",
        "query_set", "recall@1", "recall@2", "recall@3", "recall@5", "recall@10", "mrr@10",
        "dup_recall@1", "dup_recall@2", "dup_recall@3", "dup_recall@5", "dup_recall@10", "dup_mrr@10",
        "num_queries", "script_used", "result_file_used", "thesis_interpretation"
    ]
    
    def derive_inputs(sys_id, family):
        text_input = "None"
        image_input = "None"
        fusion_type = "None"
        sys_id_lower = sys_id.lower()
        
        if "caption" in sys_id_lower and "context" in sys_id_lower: text_input = "Caption+Context"
        elif "caption" in sys_id_lower: text_input = "Caption"
        elif "context" in sys_id_lower: text_input = "Context"
        else: text_input = "Caption+Context" # Assume full if not specified
        
        if "clip" in sys_id_lower or "colpali" in sys_id_lower:
            image_input = "Raw Image"
            text_input = "Query Only"
        if "ocr" in sys_id_lower:
            image_input = "OCR Extracted"
            
        if "fusion" in sys_id_lower or "hybrid" in sys_id_lower:
            if "rrf" in sys_id_lower or "rank" in sys_id_lower: fusion_type = "Rank Fusion"
            else: fusion_type = "Score Fusion"
            
        return text_input, image_input, fusion_type
    
    with open(REPORTS_DIR / "m8_master_ablation_table.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in normalized_data:
            ti, ii, ft = derive_inputs(row["system_id"], row["method_family"])
            writer.writerow({
                "milestone": row["milestone"],
                "system_id": row["system_id"],
                "method_family": row["method_family"],
                "text_input": ti,
                "image_input": ii,
                "fusion_type": ft,
                "query_set": row["query_set"],
                "recall@1": row["metrics"].get("recall@1", 0),
                "recall@2": row["metrics"].get("recall@2", 0),
                "recall@3": row["metrics"].get("recall@3", 0),
                "recall@5": row["metrics"].get("recall@5", 0),
                "recall@10": row["metrics"].get("recall@10", 0),
                "mrr@10": row["metrics"].get("mrr@10", 0),
                "dup_recall@1": row["metrics"].get("dup_recall@1", 0),
                "dup_recall@2": row["metrics"].get("dup_recall@2", 0),
                "dup_recall@3": row["metrics"].get("dup_recall@3", 0),
                "dup_recall@5": row["metrics"].get("dup_recall@5", 0),
                "dup_recall@10": row["metrics"].get("dup_recall@10", 0),
                "dup_mrr@10": row["metrics"].get("dup_mrr@10", 0),
                "num_queries": row["metrics"].get("num_queries", 0),
                "script_used": "Derived",
                "result_file_used": row["source_result_file"],
                "thesis_interpretation": ""
            })

    # 4. Create m8_best_by_query_type.csv
    best_by_qs = {}
    for qs in ["Q1", "Q2", "Q3"]:
        qs_data = [r for r in normalized_data if r["query_set"] == qs]
        if not qs_data: continue
        best_mrr = max(qs_data, key=lambda x: x["metrics"].get("mrr@10", 0))
        best_dup_mrr = max(qs_data, key=lambda x: x["metrics"].get("dup_mrr@10", 0))
        best_r1 = max(qs_data, key=lambda x: x["metrics"].get("recall@1", 0))
        best_r10 = max(qs_data, key=lambda x: x["metrics"].get("recall@10", 0))
        best_by_qs[qs] = {
            "Q": qs,
            "best_mrr@10_sys": best_mrr["system_id"], "best_mrr@10_val": best_mrr["metrics"].get("mrr@10", 0),
            "best_dup_mrr@10_sys": best_dup_mrr["system_id"], "best_dup_mrr@10_val": best_dup_mrr["metrics"].get("dup_mrr@10", 0),
            "best_r1_sys": best_r1["system_id"], "best_r1_val": best_r1["metrics"].get("recall@1", 0),
            "best_r10_sys": best_r10["system_id"], "best_r10_val": best_r10["metrics"].get("recall@10", 0),
        }
    with open(REPORTS_DIR / "m8_best_by_query_type.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["Q", "best_mrr@10_sys", "best_mrr@10_val", "best_dup_mrr@10_sys", "best_dup_mrr@10_val", "best_r1_sys", "best_r1_val", "best_r10_sys", "best_r10_val"])
        writer.writeheader()
        for qs, data in best_by_qs.items(): writer.writerow(data)

    # 5. Modality comparison table
    modality_best = {}
    for row in normalized_data:
        fam = row["method_family"]
        qs = row["query_set"]
        if qs not in ["Q1", "Q2", "Q3"]: continue
        val = row["metrics"].get("mrr@10", 0)
        if fam not in modality_best: modality_best[fam] = {"Q1": 0, "Q2": 0, "Q3": 0}
        if val > modality_best[fam][qs]: modality_best[fam][qs] = val

    with open(REPORTS_DIR / "m8_modality_comparison_table.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["method_family", "best_Q1_mrr@10", "best_Q2_mrr@10", "best_Q3_mrr@10"])
        writer.writeheader()
        for fam, qs_vals in modality_best.items():
            writer.writerow({"method_family": fam, "best_Q1_mrr@10": qs_vals["Q1"], "best_Q2_mrr@10": qs_vals["Q2"], "best_Q3_mrr@10": qs_vals["Q3"]})

    # 6. Effect Size and Fusion Lift Table
    def get_best_for_family(fam, qs):
        cands = [r for r in normalized_data if r["method_family"] == fam and r["query_set"] == qs]
        if not cands: return None
        return max(cands, key=lambda x: x["metrics"].get("mrr@10", 0))

    def effect_label(delta):
        if delta < 0.005: return "tiny"
        elif delta <= 0.010: return "small"
        elif delta <= 0.030: return "modest"
        else: return "large"

    def get_best_for_milestone(ms, qs):
        cands = [r for r in normalized_data if r["milestone"] == ms and r["query_set"] == qs]
        if not cands: return None
        return max(cands, key=lambda x: x["metrics"].get("mrr@10", 0))

    effect_rows = []
    for qs in ["Q1", "Q2", "Q3"]:
        best_m7 = get_best_for_milestone("M7", qs)
        best_bm25 = get_best_for_family("lexical text", qs)
        best_bge = get_best_for_family("dense text", qs)
        
        # M7 vs BM25
        if best_m7 and best_bm25:
            delta = best_m7["metrics"]["mrr@10"] - best_bm25["metrics"]["mrr@10"]
            effect_rows.append({
                "comparison": "M7 vs best BM25", "query_set": qs, "metric": "mrr@10",
                "baseline_score": best_bm25["metrics"]["mrr@10"], "candidate_score": best_m7["metrics"]["mrr@10"],
                "absolute_delta": delta, "percentage_point_delta": delta * 100,
                "relative_delta_percent": (delta / best_bm25["metrics"]["mrr@10"] * 100) if best_bm25["metrics"]["mrr@10"] else 0,
                "effect_size_label": effect_label(delta), "interpretation": "Hybrid provides " + effect_label(delta) + " gain"
            })
            
        # M7 vs BGE
        if best_m7 and best_bge:
            delta = best_m7["metrics"]["mrr@10"] - best_bge["metrics"]["mrr@10"]
            effect_rows.append({
                "comparison": "M7 vs best BGE", "query_set": qs, "metric": "mrr@10",
                "baseline_score": best_bge["metrics"]["mrr@10"], "candidate_score": best_m7["metrics"]["mrr@10"],
                "absolute_delta": delta, "percentage_point_delta": delta * 100,
                "relative_delta_percent": (delta / best_bge["metrics"]["mrr@10"] * 100) if best_bge["metrics"]["mrr@10"] else 0,
                "effect_size_label": effect_label(delta), "interpretation": "Hybrid provides " + effect_label(delta) + " gain"
            })

    with open(REPORTS_DIR / "m8_effect_size_interpretation.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "query_set", "metric", "baseline_score", "candidate_score", "absolute_delta", "percentage_point_delta", "relative_delta_percent", "effect_size_label", "interpretation"])
        writer.writeheader()
        for r in effect_rows: writer.writerow(r)

    # Fusion lift table (simplified version of the effect size table)
    with open(REPORTS_DIR / "m8_fusion_lift_table.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "query_set", "absolute_delta", "percentage_point_delta"])
        writer.writeheader()
        for r in effect_rows:
            writer.writerow({
                "comparison": r["comparison"], "query_set": r["query_set"],
                "absolute_delta": r["absolute_delta"], "percentage_point_delta": r["percentage_point_delta"]
            })

if __name__ == "__main__":
    main()
