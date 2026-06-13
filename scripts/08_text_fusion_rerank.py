"""
scripts/08_text_fusion_rerank.py
────────────────────────────────
Milestone 5.5 — Text Fusion and Rank-1 Reranking

Implements:
  - Recomputing B1, B2, D1, D2 predictions
  - Fusion: RRF with k ablation (10, 30, 60)
  - Fusion variants: H1a, H1b, H1c, H1d (weighted), H1_router, H1_oracle
  - Reranking: Cross-encoder on union of top-100 candidates
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── project root on PYTHONPATH ────────────────────────────────────────────────
PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import evaluate_run, load_duplicate_mapping
from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "reports" / "m55_text_fusion_rerank.log")
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH     = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
Q1_PATH      = PROJECT_ROOT / "queries" / "q1_captions.json"
Q2_PATH      = PROJECT_ROOT / "queries" / "q2_paraphrased.json"
Q3_PATH      = PROJECT_ROOT / "queries" / "q3_context.json"
DUP_MAP_PATH = PROJECT_ROOT / "eval"    / "duplicate_mapping.json"
REPORTS_DIR  = PROJECT_ROOT / "reports"
HF_CACHE     = Path("/DATA5/prabhakar/hf_cache")

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE   = 256
TOP_K        = 100

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenise(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return _TOKEN_RE.findall(text.lower())

def load_query_set(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["queries"]

# ─────────────────────────────────────────────────────────────────────────────
# BM25 & BGE Base
# ─────────────────────────────────────────────────────────────────────────────
def run_bm25(corpus_texts: List[str], queries: List[Dict]) -> List[List[int]]:
    tokenised_corpus = [tokenise(c) for c in corpus_texts]
    index = BM25Okapi(tokenised_corpus)
    all_preds = []
    for q in queries:
        tokens = tokenise(q["text"])
        if not tokens:
            all_preds.append([])
            continue
        scores = index.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        all_preds.append(ranked)
    return all_preds

def run_dense(model: SentenceTransformer, doc_texts: List[str], queries: List[Dict]) -> List[List[int]]:
    log.info("Encoding %d corpus docs...", len(doc_texts))
    doc_emb = model.encode(doc_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)

    q_texts = [BGE_QUERY_PREFIX + q["text"] for q in queries]
    log.info("Encoding %d queries...", len(q_texts))
    q_emb = model.encode(q_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    _scores, indices = index.search(q_emb, TOP_K)
    return [[int(i) for i in row if i >= 0] for row in indices]

# ─────────────────────────────────────────────────────────────────────────────
# Fusion functions
# ─────────────────────────────────────────────────────────────────────────────
def rrf(rankings_list: List[List[int]], weights: List[float] = None, k: int = 60) -> List[int]:
    if weights is None:
        weights = [1.0] * len(rankings_list)
    scores = defaultdict(float)
    for ranking, weight in zip(rankings_list, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += weight / (k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs][:TOP_K]

def route_query(query_text: str) -> bool:
    """Returns True if context-heavy should be preferred."""
    keywords = ["figure", "shown", "procedure", "illustrated", "diagram"]
    q_lower = query_text.lower()
    if len(query_text.split()) > 15:
        return True
    for kw in keywords:
        if kw in q_lower:
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 80)
    log.info(" M5.5 Text Fusion and Rank-1 Reranking")
    log.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data
    log.info("Loading corpus...")
    df = pd.read_csv(CSV_PATH)
    captions = df["Image Caption"].fillna("").astype(str).tolist()
    contexts = df["Context"].fillna("").astype(str).tolist()
    sources = df["Source"].fillna("").astype(str).tolist()
    subclauses = df["Subclause"].fillna("").astype(str).tolist()

    cap_ctx = [cap + " " + ctx for cap, ctx in zip(captions, contexts)]

    q1 = load_query_set(Q1_PATH)
    q2 = load_query_set(Q2_PATH)
    q3 = load_query_set(Q3_PATH)
    query_sets = {"Q1": q1, "Q2": q2, "Q3": q3}

    hash_to_rows, row_to_hash = load_duplicate_mapping(DUP_MAP_PATH)

    # 2. Recompute Base Predictions
    log.info("--- Recomputing B1, B2, D1, D2 ---")
    preds = {"B1": {}, "B2": {}, "D1": {}, "D2": {}}

    # BM25
    log.info("Running B1 (BM25 Caption)...")
    for qs_name, qs in query_sets.items():
        preds["B1"][qs_name] = run_bm25(captions, qs)

    log.info("Running B2 (BM25 Cap+Ctx)...")
    for qs_name, qs in query_sets.items():
        preds["B2"][qs_name] = run_bm25(cap_ctx, qs)

    # Dense
    log.info(f"Loading {BGE_MODEL_NAME} for D1/D2...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME, cache_folder=str(HF_CACHE), device=device).half()

    log.info("Running D1 (BGE Caption)...")
    for qs_name, qs in query_sets.items():
        preds["D1"][qs_name] = run_dense(bge_model, captions, qs)

    log.info("Running D2 (BGE Cap+Ctx)...")
    for qs_name, qs in query_sets.items():
        preds["D2"][qs_name] = run_dense(bge_model, cap_ctx, qs)

    del bge_model
    torch.cuda.empty_cache()

    # Save base predictions
    for sys_name in ["B1", "B2", "D1", "D2"]:
        for qs_name in ["Q1", "Q2", "Q3"]:
            fname = f"m55_predictions_{sys_name.lower()}_{qs_name.lower()}.json"
            with open(REPORTS_DIR / fname, "w") as f:
                json.dump(preds[sys_name][qs_name], f)

    # Evaluate baselines
    results = {}
    for sys_name in ["B1", "B2", "D1", "D2"]:
        results[sys_name] = {}
        for qs_name, qs in query_sets.items():
            results[sys_name][qs_name] = evaluate_run(qs, preds[sys_name][qs_name], hash_to_rows, row_to_hash)

    # 3. Fusion Variants
    log.info("--- Running Fusion Variants ---")
    fusion_preds = defaultdict(lambda: defaultdict(list))

    weights_caption_heavy = [1.5, 0.5, 1.5, 0.5] # B1, B2, D1, D2
    weights_context_heavy = [0.5, 1.5, 0.5, 1.5]
    weights_balanced      = [1.0, 1.0, 1.0, 1.0]

    for qs_name, qs in query_sets.items():
        for i, q in enumerate(qs):
            b1 = preds["B1"][qs_name][i]
            b2 = preds["B2"][qs_name][i]
            d1 = preds["D1"][qs_name][i]
            d2 = preds["D2"][qs_name][i]

            # H1c Ablations
            fusion_preds["H1c_k10"][qs_name].append(rrf([b1, b2, d1, d2], k=10))
            fusion_preds["H1c_k30"][qs_name].append(rrf([b1, b2, d1, d2], k=30))
            fusion_preds["H1c_k60"][qs_name].append(rrf([b1, b2, d1, d2], k=60))

            # Sub-combinations
            fusion_preds["H1a"][qs_name].append(rrf([b1, d1], k=60))
            fusion_preds["H1b"][qs_name].append(rrf([b2, d2], k=60))

            # Weighted
            fusion_preds["H1d_caption_heavy"][qs_name].append(rrf([b1, b2, d1, d2], weights_caption_heavy, k=60))
            fusion_preds["H1d_context_heavy"][qs_name].append(rrf([b1, b2, d1, d2], weights_context_heavy, k=60))

            # H1_router
            is_context_heavy = route_query(q["text"])
            if is_context_heavy:
                fusion_preds["H1_router"][qs_name].append(rrf([b1, b2, d1, d2], weights_context_heavy, k=60))
            else:
                fusion_preds["H1_router"][qs_name].append(rrf([b1, b2, d1, d2], weights_caption_heavy, k=60))

            # H1_oracle (Query-set-aware diagnostic)
            if qs_name in ["Q1", "Q2"]:
                fusion_preds["H1_oracle"][qs_name].append(rrf([b1, b2, d1, d2], weights_caption_heavy, k=60))
            else:
                fusion_preds["H1_oracle"][qs_name].append(rrf([b1, b2, d1, d2], weights_context_heavy, k=60))

    for fname_key, fpreds in fusion_preds.items():
        results[fname_key] = {}
        for qs_name, qs in query_sets.items():
            results[fname_key][qs_name] = evaluate_run(qs, fpreds[qs_name], hash_to_rows, row_to_hash)

    # 4. Reranking Phase
    log.info("--- Reranking Phase ---")
    reranker_loaded = False
    reranker = None
    try:
        log.info(f"Loading CrossEncoder: {RERANKER_MODEL}")
        reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
        reranker_loaded = True
    except Exception as e:
        log.warning(f"CrossEncoder failed to load: {e}. Trying fallback...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, cache_dir=str(HF_CACHE))
            model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL, cache_dir=str(HF_CACHE)).to(device).half()

            class FallbackReranker:
                def predict(self, pairs, batch_size=32):
                    all_scores = []
                    for i in range(0, len(pairs), batch_size):
                        batch = pairs[i:i+batch_size]
                        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                        with torch.no_grad():
                            scores = model(**inputs).logits.squeeze(-1).cpu().numpy()
                            if scores.ndim == 0:
                                scores = [float(scores)]
                            else:
                                scores = scores.tolist()
                        all_scores.extend(scores)
                    return all_scores
            reranker = FallbackReranker()
            reranker_loaded = True
        except Exception as e2:
            log.warning(f"Fallback also failed: {e2}. Skipping reranking gracefully.")

    if reranker_loaded:
        for k_cut in [20, 50]:
            exp_name = f"union_top{k_cut}_rerank"
            results[exp_name] = {}
            for qs_name, qs in query_sets.items():
                log.info(f"Reranking {exp_name} for {qs_name}...")
                reranked_preds = []
                for i, q in enumerate(qs):
                    b1 = preds["B1"][qs_name][i]
                    b2 = preds["B2"][qs_name][i]
                    d1 = preds["D1"][qs_name][i]
                    d2 = preds["D2"][qs_name][i]

                    # Candidate pool: Union of top-100
                    union_pool = list(set(b1 + b2 + d1 + d2))

                    # Score union with H1c k=60 to get initial ranking
                    initial_scores = defaultdict(float)
                    for ranking in [b1, b2, d1, d2]:
                        for rank, doc_id in enumerate(ranking):
                            initial_scores[doc_id] += 1.0 / (60 + rank + 1)

                    sorted_union = [doc_id for doc_id, score in sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)]

                    # Take top-K for reranking
                    candidates_to_rerank = sorted_union[:k_cut]
                    remaining = sorted_union[k_cut:]

                    # Prepare text pairs
                    pairs = []
                    for doc_id in candidates_to_rerank:
                        ctx = contexts[doc_id][:1500]
                        doc_text = f"Image Caption: {captions[doc_id]} | Source: {sources[doc_id]} | Subclause: {subclauses[doc_id]} | Context: {ctx}"
                        pairs.append([q["text"], doc_text])

                    # Get scores
                    scores = reranker.predict(pairs, batch_size=32)

                    # Re-sort candidates_to_rerank
                    cand_scores = list(zip(candidates_to_rerank, scores))
                    cand_scores.sort(key=lambda x: x[1], reverse=True)
                    best_candidates = [c[0] for c in cand_scores]

                    # Final list
                    final_ranking = (best_candidates + remaining)[:TOP_K]
                    reranked_preds.append(final_ranking)

                fusion_preds[exp_name][qs_name] = reranked_preds
                results[exp_name][qs_name] = evaluate_run(qs, reranked_preds, hash_to_rows, row_to_hash)

    # 5. Save Results
    out_path = REPORTS_DIR / "m55_text_fusion_rerank_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Append to master results CSV
    master_csv_path = REPORTS_DIR / "master_results.csv"
    update_csv_path = REPORTS_DIR / "m55_master_results_update.csv"

    rows = []
    for exp_name, exp_res in results.items():
        for qs_name, qs_res in exp_res.items():
            rows.append({
                "Milestone": "M5.5",
                "Experiment": exp_name,
                "QuerySet": qs_name,
                "R@1": qs_res["recall@1"],
                "R@2": qs_res["recall@2"],
                "R@5": qs_res["recall@5"],
                "R@10": qs_res["recall@10"],
                "MRR@10": qs_res["mrr@10"],
                "dupR@1": qs_res["dup_recall@1"],
                "dupMRR@10": qs_res["dup_mrr@10"]
            })

    df_new = pd.DataFrame(rows)
    df_new.to_csv(update_csv_path, index=False)

    if master_csv_path.exists():
        df_master = pd.read_csv(master_csv_path)
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        df_combined.to_csv(master_csv_path, index=False)
    else:
        df_new.to_csv(master_csv_path, index=False)

    log.info("Done! Results saved to %s", out_path)

if __name__ == "__main__":
    main()
