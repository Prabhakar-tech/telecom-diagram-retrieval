#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import reciprocal_rank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# Import M7 functions
spec = importlib.util.spec_from_file_location("m7", str(PROJECT_ROOT / "scripts/12_hybrid_lexical_dense.py"))
m7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m7)

def load_queries(qs):
    path_map = {
        "Q1": "queries/q1_captions.json",
        "Q2": "queries/q2_paraphrased.json",
        "Q3": "queries/q3_context.json"
    }
    with open(PROJECT_ROOT / path_map[qs], "r") as f:
        return json.load(f)["queries"]

def get_m7_prediction(sys_name, qs, cache, q_idx):
    b_src = "B1" if qs in ["Q1", "Q2"] else "B2"
    d_src = "D1" if qs == "Q1" else "D2"
    
    b_scored = cache[b_src][qs][q_idx]
    d_scored = cache[d_src][qs][q_idx]
    b_rank = m7.strip_scores(b_scored)
    d_rank = m7.strip_scores(d_scored)
    
    if sys_name == "rrf_bm25_bge_k10": return m7.rrf([b_rank, d_rank], [1.0, 1.0], k=10)
    elif sys_name == "rrf_bm25_bge_k30": return m7.rrf([b_rank, d_rank], [1.0, 1.0], k=30)
    elif sys_name == "rrf_bm25_bge_k60": return m7.rrf([b_rank, d_rank], [1.0, 1.0], k=60)
    elif sys_name == "rrf_bm25_bge_heavy_bm25": return m7.rrf([b_rank, d_rank], [0.75, 0.25], k=60)
    elif sys_name == "rrf_bm25_bge_balanced": return m7.rrf([b_rank, d_rank], [0.50, 0.50], k=60)
    elif sys_name == "rrf_bm25_bge_heavy_dense": return m7.rrf([b_rank, d_rank], [0.25, 0.75], k=60)
    elif sys_name == "score_fusion_bm25_075_bge_025": return m7.score_fusion(b_scored, d_scored, 0.75, 0.25)
    elif sys_name == "score_fusion_bm25_050_bge_050": return m7.score_fusion(b_scored, d_scored, 0.50, 0.50)
    elif sys_name == "score_fusion_bm25_025_bge_075": return m7.score_fusion(b_scored, d_scored, 0.25, 0.75)
    return []

def main():
    log.info("Loading cache...")
    cache_path = PROJECT_ROOT / "reports/m7_base_predictions_cache.json"
    with open(cache_path, "r") as f:
        cache = json.load(f)

    # All candidate M7 systems
    m7_systems = [
        "rrf_bm25_bge_k10", "rrf_bm25_bge_k30", "rrf_bm25_bge_k60",
        "rrf_bm25_bge_heavy_bm25", "rrf_bm25_bge_balanced", "rrf_bm25_bge_heavy_dense",
        "score_fusion_bm25_075_bge_025", "score_fusion_bm25_050_bge_050", "score_fusion_bm25_025_bge_075"
    ]

    per_query_rows = []
    paired_summary_rows = []
    audit_rows = []

    out_json = {"paired_comparisons": {}}

    for qs in ["Q1", "Q2", "Q3"]:
        queries = load_queries(qs)
        num_queries = len(queries)
        
        # 1. Find best BM25 and best BGE by MRR@10
        bm25_scores = {"B1": [], "B2": []}
        bge_scores = {"D1": [], "D2": []}
        
        for q_idx, q in enumerate(queries):
            gt = q["ground_truth_row"]
            for sys_id in ["B1", "B2"]:
                preds = m7.strip_scores(cache[sys_id][qs][q_idx])
                bm25_scores[sys_id].append(reciprocal_rank(preds, gt, 10))
            for sys_id in ["D1", "D2"]:
                preds = m7.strip_scores(cache[sys_id][qs][q_idx])
                bge_scores[sys_id].append(reciprocal_rank(preds, gt, 10))

        bm25_means = {k: np.mean(v) for k, v in bm25_scores.items()}
        bge_means = {k: np.mean(v) for k, v in bge_scores.items()}
        
        best_bm25_sys = max(bm25_means, key=bm25_means.get)
        best_bge_sys = max(bge_means, key=bge_means.get)

        # 2. Find best M7 by MRR@10
        m7_scores = {s: [] for s in m7_systems}
        for q_idx, q in enumerate(queries):
            gt = q["ground_truth_row"]
            for s in m7_systems:
                preds = get_m7_prediction(s, qs, cache, q_idx)
                m7_scores[s].append(reciprocal_rank(preds, gt, 10))

        m7_means = {k: np.mean(v) for k, v in m7_scores.items()}
        best_m7_sys = max(m7_means, key=m7_means.get)

        # Populate Audit Log
        audit_rows.append({
            "query_set": qs,
            "comparison": "BM25",
            "selected_baseline_system": best_bm25_sys,
            "selected_baseline_metric": "MRR@10",
            "selected_baseline_score_from_aggregate": bm25_means[best_bm25_sys],
            "selected_candidate_system": best_m7_sys,
            "selected_candidate_metric": "MRR@10",
            "selected_candidate_score_from_aggregate": m7_means[best_m7_sys],
            "mean_baseline_mrr_from_per_query": np.mean(bm25_scores[best_bm25_sys]),
            "mean_candidate_mrr_from_per_query": np.mean(m7_scores[best_m7_sys]),
            "consistency_check_passed": abs(bm25_means[best_bm25_sys] - np.mean(bm25_scores[best_bm25_sys])) < 1e-5 and abs(m7_means[best_m7_sys] - np.mean(m7_scores[best_m7_sys])) < 1e-5,
            "notes": "BM25 vs M7"
        })

        # Sanity checking per query set
        mean_bm25 = np.mean(bm25_scores[best_bm25_sys])
        mean_bge = np.mean(bge_scores[best_bge_sys])
        mean_m7 = np.mean(m7_scores[best_m7_sys])
        
        m7_minus_bm25_deltas = []
        m7_minus_bge_deltas = []
        
        for q_idx, q in enumerate(queries):
            q_id = q["query_id"]
            gt = q["ground_truth_row"]
            
            b_mrr = bm25_scores[best_bm25_sys][q_idx]
            d_mrr = bge_scores[best_bge_sys][q_idx]
            m_mrr = m7_scores[best_m7_sys][q_idx]
            
            d_bm25 = m_mrr - b_mrr
            d_bge = m_mrr - d_mrr
            
            m7_minus_bm25_deltas.append(d_bm25)
            m7_minus_bge_deltas.append(d_bge)
            
            per_query_rows.append({
                "query_set": qs,
                "query_id": q_id,
                "ground_truth": gt,
                "best_bm25_system": best_bm25_sys,
                "best_bm25_rank": -1, # omit rank to keep logic simple, mrr is enough
                "best_bm25_mrr_contribution": b_mrr,
                "best_bge_system": best_bge_sys,
                "best_bge_rank": -1,
                "best_bge_mrr_contribution": d_mrr,
                "best_m7_system": best_m7_sys,
                "best_m7_rank": -1,
                "best_m7_mrr_contribution": m_mrr,
                "m7_minus_bm25_delta": d_bm25,
                "m7_minus_bge_delta": d_bge,
                "m7_vs_bm25_outcome": "win" if d_bm25 > 0 else ("loss" if d_bm25 < 0 else "tie"),
                "m7_vs_bge_outcome": "win" if d_bge > 0 else ("loss" if d_bge < 0 else "tie")
            })

        # Sanity Verify
        mean_d_bm25 = np.mean(m7_minus_bm25_deltas)
        expected_d_bm25 = mean_m7 - mean_bm25
        if abs(mean_d_bm25 - expected_d_bm25) > 1e-5:
            msg = f"Sanity Failure: {qs} expected delta {expected_d_bm25} != {mean_d_bm25}"
            log.error(msg)
            out_json["sanity_failure"] = msg

        def get_interp(ci):
            if not ci: return "not_available"
            if ci[0] > 0: return "positive_ci_excludes_zero"
            elif ci[1] < 0: return "negative_ci_excludes_zero"
            else: return "ci_overlaps_zero"

        for comp_name, deltas in [("M7 vs BM25", m7_minus_bm25_deltas), ("M7 vs BGE", m7_minus_bge_deltas)]:
            wins = sum(1 for d in deltas if d > 0)
            losses = sum(1 for d in deltas if d < 0)
            ties = sum(1 for d in deltas if d == 0)
            
            np.random.seed(42)
            boot_means = []
            d_arr = np.array(deltas)
            for _ in range(1000):
                boot_means.append(np.mean(np.random.choice(d_arr, size=num_queries, replace=True)))
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
            ci = [float(ci_low), float(ci_high)]

            interp = get_interp(ci)

            paired_summary_rows.append({
                "query_set": qs, "comparison": comp_name, "num_queries": num_queries,
                "win_count": wins, "loss_count": losses, "tie_count": ties,
                "mean_delta_mrr": float(np.mean(deltas)), "median_delta_mrr": float(np.median(deltas)),
                "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1],
                "significance_interpretation": interp
            })

            k = f"{comp_name.replace(' ', '_')}_{qs}"
            out_json["paired_comparisons"][k] = {
                "win_count": wins, "loss_count": losses, "tie_count": ties,
                "mean_delta": float(np.mean(deltas)),
                "bootstrap_ci": ci,
                "interpretation": interp
            }

    pd.DataFrame(per_query_rows).to_csv(PROJECT_ROOT / "reports/m8_best_system_per_query_metrics.csv", index=False)
    pd.DataFrame(paired_summary_rows).to_csv(PROJECT_ROOT / "reports/m8_paired_comparison_summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(PROJECT_ROOT / "reports/m8_paired_system_selection_audit.csv", index=False)

    with open(PROJECT_ROOT / "reports/m8_statistical_validation.json", "w") as f:
        json.dump(out_json, f, indent=2)

    log.info("Validation Complete.")

if __name__ == "__main__":
    main()
