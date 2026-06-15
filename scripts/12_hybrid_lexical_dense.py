"""
scripts/12_hybrid_lexical_dense.py
──────────────────────────────────
Milestone 7 — Hybrid Lexical + Dense Text Retrieval
"""

import os
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer

# ── project root on PYTHONPATH
PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import evaluate_run, load_duplicate_mapping
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ── Config
CSV_PATH = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
Q1_PATH = PROJECT_ROOT / "queries" / "q1_captions.json"
Q2_PATH = PROJECT_ROOT / "queries" / "q2_paraphrased.json"
Q3_PATH = PROJECT_ROOT / "queries" / "q3_context.json"
DUP_MAP_PATH = PROJECT_ROOT / "eval" / "duplicate_mapping.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
HF_CACHE = Path("/DATA5/prabhakar/hf_cache")

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 256
TOP_K = 100

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenise(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return _TOKEN_RE.findall(text.lower())

def load_query_set(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)["queries"]

def check_inputs():
    required_files = [
        "reports/m2_bm25_results.json",
        "reports/m3_dense_results.json",
        "reports/m5_clip_results.json",
        "reports/m55_text_fusion_rerank_results.json",
        "reports/m65_acronym_expansion_results.json",
        "queries/q1_captions.json",
        "queries/q2_paraphrased.json",
        "queries/q3_context.json",
        "eval/duplicate_mapping.json",
        "eval/metrics.py",
        "reports/m55_predictions_b1_q1.json",
        "reports/m55_predictions_d1_q1.json",
        "reports/m5_clip_predictions_q1.json",
        "reports/m65_acronym_expansion_predictions_q1.json"
    ]
    missing = [f for f in required_files if not (PROJECT_ROOT / f).exists()]
    out_path = REPORTS_DIR / "m7_required_inputs_check.json"
    with open(out_path, "w") as f:
        json.dump({
            "status": "FAIL" if missing else "PASS",
            "missing_files": missing,
            "checked_files": required_files
        }, f, indent=2)
    if missing:
        log.warning(f"Missing required inputs: {missing}")

def run_bm25_with_scores(corpus_texts: List[str], queries: List[Dict]) -> List[List[Tuple[int, float]]]:
    tokenised_corpus = [tokenise(c) for c in corpus_texts]
    index = BM25Okapi(tokenised_corpus)
    all_preds = []
    for q in queries:
        tokens = tokenise(q["text"])
        if not tokens:
            all_preds.append([])
            continue
        scores = index.get_scores(tokens)
        ranked = sorted([(i, float(scores[i])) for i in range(len(scores))], key=lambda x: x[1], reverse=True)[:TOP_K]
        all_preds.append(ranked)
    return all_preds

def run_dense_with_scores(model: SentenceTransformer, doc_texts: List[str], queries: List[Dict]) -> List[List[Tuple[int, float]]]:
    doc_emb = model.encode(doc_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    q_texts = [BGE_QUERY_PREFIX + q["text"] for q in queries]
    q_emb = model.encode(q_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, indices = index.search(q_emb, TOP_K)
    all_preds = []
    for row_scores, row_indices in zip(scores, indices):
        valid = [(int(idx), float(s)) for idx, s in zip(row_indices, row_scores) if idx >= 0]
        all_preds.append(valid)
    return all_preds

def get_base_predictions(query_sets, captions, cap_ctx, device):
    cache_path = REPORTS_DIR / "m7_base_predictions_cache.json"
    if cache_path.exists():
        log.info("Loading cached base predictions...")
        with open(cache_path, "r") as f:
            return json.load(f)

    log.info("Computing BM25 scores...")
    preds = {"B1": {}, "B2": {}, "D1": {}, "D2": {}}
    for qs_name, qs in query_sets.items():
        preds["B1"][qs_name] = run_bm25_with_scores(captions, qs)
        preds["B2"][qs_name] = run_bm25_with_scores(cap_ctx, qs)

    log.info("Computing BGE scores...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME, cache_folder=str(HF_CACHE), device=device).half()
    for qs_name, qs in query_sets.items():
        preds["D1"][qs_name] = run_dense_with_scores(bge_model, captions, qs)
        preds["D2"][qs_name] = run_dense_with_scores(bge_model, cap_ctx, qs)

    with open(cache_path, "w") as f:
        json.dump(preds, f)
    return preds

def load_clip_predictions():
    preds = {}
    for qs in ["q1", "q2", "q3"]:
        path = REPORTS_DIR / f"m5_clip_predictions_{qs}.json"
        if path.exists():
            with open(path, "r") as f:
                preds[qs.upper()] = json.load(f)
    return preds

def load_m65_predictions():
    preds = {}
    for qs in ["q1", "q2", "q3"]:
        path = REPORTS_DIR / f"m65_acronym_expansion_predictions_{qs}.json"
        if path.exists():
            with open(path, "r") as f:
                preds[qs.upper()] = json.load(f)
    return preds

def rrf(rankings: List[List[int]], weights: List[float], k=60):
    scores = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += w / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

def normalize_minmax(scored_ranking: List[Tuple[int, float]]) -> Dict[int, float]:
    if not scored_ranking:
        return {}
    scores = [s for _, s in scored_ranking]
    min_s, max_s = min(scores), max(scores)
    range_s = max_s - min_s if max_s > min_s else 1.0
    return {doc_id: (s - min_s) / range_s for doc_id, s in scored_ranking}

def normalize_zscore(scored_ranking: List[Tuple[int, float]]) -> Dict[int, float]:
    if not scored_ranking:
        return {}
    scores = [s for _, s in scored_ranking]
    mean_s = np.mean(scores)
    std_s = np.std(scores)
    if std_s == 0:
        std_s = 1.0
    return {doc_id: float((s - mean_s) / std_s) for doc_id, s in scored_ranking}

def score_fusion(scored1: List[Tuple[int, float]], scored2: List[Tuple[int, float]], w1: float, w2: float):
    norm1 = normalize_minmax(scored1)
    norm2 = normalize_minmax(scored2)
    scores = defaultdict(float)
    for doc_id, s in norm1.items(): scores[doc_id] += w1 * s
    for doc_id, s in norm2.items(): scores[doc_id] += w2 * s
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

def strip_scores(scored_list: List[Tuple[int, float]]) -> List[int]:
    return [doc_id for doc_id, score in scored_list]

def main():
    check_inputs()
    log.info("Loading Data...")
    df = pd.read_csv(CSV_PATH)
    captions = df["Image Caption"].fillna("").astype(str).tolist()
    cap_ctx = (df["Image Caption"].fillna("").astype(str) + " " + df["Context"].fillna("").astype(str)).tolist()

    query_sets = {
        "Q1": load_query_set(Q1_PATH),
        "Q2": load_query_set(Q2_PATH),
        "Q3": load_query_set(Q3_PATH)
    }
    hash_to_rows, row_to_hash = load_duplicate_mapping(DUP_MAP_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_preds = get_base_predictions(query_sets, captions, cap_ctx, device)
    clip_preds = load_clip_predictions()
    m65_preds = load_m65_predictions()

    systems = {}

    # Evaluate A: Single-channel
    systems["bm25_caption"] = {qs: [strip_scores(p) for p in base_preds["B1"][qs]] for qs in ["Q1", "Q2", "Q3"]}
    systems["bm25_caption_context"] = {qs: [strip_scores(p) for p in base_preds["B2"][qs]] for qs in ["Q1", "Q2", "Q3"]}
    systems["bge_caption"] = {qs: [strip_scores(p) for p in base_preds["D1"][qs]] for qs in ["Q1", "Q2", "Q3"]}
    systems["bge_caption_context"] = {qs: [strip_scores(p) for p in base_preds["D2"][qs]] for qs in ["Q1", "Q2", "Q3"]}

    # B: RRF Hybrid
    for qs in ["Q1", "Q2", "Q3"]:
        for k in [10, 30, 60]:
            sys_name = f"rrf_bm25_bge_k{k}"
            if sys_name not in systems: systems[sys_name] = {}
            # Assume Best BM25 = B2 for Q3, B1 for Q1/Q2; Best BGE = D2 for Q2/Q3, D1 for Q1
            systems[sys_name][qs] = []
            b_src = "B1" if qs in ["Q1", "Q2"] else "B2"
            d_src = "D1" if qs == "Q1" else "D2"

            for i in range(len(query_sets[qs])):
                b_rank = strip_scores(base_preds[b_src][qs][i])
                d_rank = strip_scores(base_preds[d_src][qs][i])
                systems[sys_name][qs].append(rrf([b_rank, d_rank], [1.0, 1.0], k=k))

        # Weighted RRF
        b_src = "B1" if qs in ["Q1", "Q2"] else "B2"
        d_src = "D1" if qs == "Q1" else "D2"
        sys_w1 = "rrf_bm25_bge_heavy_bm25"
        sys_w2 = "rrf_bm25_bge_balanced"
        sys_w3 = "rrf_bm25_bge_heavy_dense"
        if sys_w1 not in systems: systems[sys_w1] = {}; systems[sys_w2] = {}; systems[sys_w3] = {}
        systems[sys_w1][qs] = []
        systems[sys_w2][qs] = []
        systems[sys_w3][qs] = []
        for i in range(len(query_sets[qs])):
            b_rank = strip_scores(base_preds[b_src][qs][i])
            d_rank = strip_scores(base_preds[d_src][qs][i])
            systems[sys_w1][qs].append(rrf([b_rank, d_rank], [0.75, 0.25], k=60))
            systems[sys_w2][qs].append(rrf([b_rank, d_rank], [0.50, 0.50], k=60))
            systems[sys_w3][qs].append(rrf([b_rank, d_rank], [0.25, 0.75], k=60))

    # C: Score Fusion
    for qs in ["Q1", "Q2", "Q3"]:
        b_src = "B1" if qs in ["Q1", "Q2"] else "B2"
        d_src = "D1" if qs == "Q1" else "D2"

        sys1 = "score_fusion_bm25_075_bge_025"
        sys2 = "score_fusion_bm25_050_bge_050"
        sys3 = "score_fusion_bm25_025_bge_075"
        if sys1 not in systems: systems[sys1] = {}; systems[sys2] = {}; systems[sys3] = {}
        systems[sys1][qs] = []; systems[sys2][qs] = []; systems[sys3][qs] = []

        for i in range(len(query_sets[qs])):
            b_scored = base_preds[b_src][qs][i]
            d_scored = base_preds[d_src][qs][i]
            systems[sys1][qs].append(score_fusion(b_scored, d_scored, 0.75, 0.25))
            systems[sys2][qs].append(score_fusion(b_scored, d_scored, 0.50, 0.50))
            systems[sys3][qs].append(score_fusion(b_scored, d_scored, 0.25, 0.75))

    # D: Hybrid + Acronym (use score_fusion 50/50 as base)
    for qs in ["Q1", "Q2", "Q3"]:
        sys1 = "hybrid_plus_acronym_005"
        sys2 = "hybrid_plus_acronym_010"
        if sys1 not in systems: systems[sys1] = {}; systems[sys2] = {}
        systems[sys1][qs] = []; systems[sys2][qs] = []

        for i, q in enumerate(query_sets[qs]):
            base_rank = systems["score_fusion_bm25_050_bge_050"][qs][i]
            qid = q["query_id"]
            m65_rank = m65_preds.get(qs, {}).get(qid, [])
            systems[sys1][qs].append(rrf([base_rank, m65_rank], [1.0, 0.05], k=60))
            systems[sys2][qs].append(rrf([base_rank, m65_rank], [1.0, 0.10], k=60))

    # E: CLIP diagnostic
    for qs in ["Q1", "Q2", "Q3"]:
        sys1 = "bm25_bge_clip_low_weight"
        if sys1 not in systems: systems[sys1] = {}
        systems[sys1][qs] = []
        for i in range(len(query_sets[qs])):
            base_rank = systems["score_fusion_bm25_050_bge_050"][qs][i]
            clip_rank = clip_preds.get(qs, [])[i] if len(clip_preds.get(qs, [])) > i else []
            systems[sys1][qs].append(rrf([base_rank, clip_rank], [1.0, 0.05], k=60))

    # Evaluate all
    results = {}
    for sys_name, sys_preds in systems.items():
        results[sys_name] = {}
        for qs in ["Q1", "Q2", "Q3"]:
            res = evaluate_run(query_sets[qs], sys_preds[qs], hash_to_rows, row_to_hash, k_values=(1, 2, 3, 5, 10), mrr_k=10)
            results[sys_name][qs] = res

    # Identify Best M7
    best_m7_by_qs = {}
    for qs in ["Q1", "Q2", "Q3"]:
        best_sys = max([k for k in systems.keys() if "rrf" in k or "score_fusion" in k], key=lambda k: results[k][qs]["dup_mrr@10"])
        best_m7_by_qs[qs] = best_sys

    # Win/Loss
    log.info("Computing per-query Win/Loss...")
    win_loss_rows = []
    from eval.metrics import duplicate_aware_reciprocal_rank
    for qs in ["Q1", "Q2", "Q3"]:
        b_src = "bm25_caption" if qs in ["Q1", "Q2"] else "bm25_caption_context"
        d_src = "bge_caption" if qs == "Q1" else "bge_caption_context"
        best_sys = best_m7_by_qs[qs]

        for i, q in enumerate(query_sets[qs]):
            gt = q["ground_truth_row"]
            b_mrr = duplicate_aware_reciprocal_rank(systems[b_src][qs][i], gt, hash_to_rows, row_to_hash, 10)
            d_mrr = duplicate_aware_reciprocal_rank(systems[d_src][qs][i], gt, hash_to_rows, row_to_hash, 10)
            h_mrr = duplicate_aware_reciprocal_rank(systems[best_sys][qs][i], gt, hash_to_rows, row_to_hash, 10)

            status = "tie"
            if h_mrr > max(b_mrr, d_mrr): status = "hybrid_win"
            elif h_mrr < max(b_mrr, d_mrr): status = "hybrid_loss"

            win_loss_rows.append({
                "query_id": q["query_id"],
                "query_set": qs,
                "bm25_mrr": b_mrr,
                "bge_mrr": d_mrr,
                "hybrid_mrr": h_mrr,
                "delta": h_mrr - max(b_mrr, d_mrr),
                "status": status
            })

    pd.DataFrame(win_loss_rows).to_csv(REPORTS_DIR / "m7_per_query_win_loss.csv", index=False)

    # Save Results
    with open(REPORTS_DIR / "m7_hybrid_lexical_dense_results.json", "w") as f:
        json.dump(results, f, indent=2)

    for qs in ["Q1", "Q2", "Q3"]:
        sys_name = best_m7_by_qs[qs]
        out_path = REPORTS_DIR / f"m7_hybrid_lexical_dense_predictions_{qs.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(systems[sys_name][qs], f)

    # Expected systems table
    expected = []
    for sys_name in systems:
        expected.append({
            "System": sys_name,
            "Q1 MRR@10": results[sys_name]["Q1"]["mrr@10"],
            "Q2 MRR@10": results[sys_name]["Q2"]["mrr@10"],
            "Q3 MRR@10": results[sys_name]["Q3"]["mrr@10"]
        })
    pd.DataFrame(expected).to_csv(REPORTS_DIR / "m7_expected_systems_table.csv", index=False)
    log.info("Done.")

if __name__ == "__main__":
    main()
