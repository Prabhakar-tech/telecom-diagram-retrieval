"""
scripts/05_dense_baselines.py
─────────────────────────────
Milestone 3 — Dense Semantic Baselines (BGE-base-en-v1.5)

Experiments
───────────
  D1  Dense on Image Caption only                 (768-d, FAISS IndexFlatIP)
  D2  Dense on Image Caption + Context (concat)   (768-d, FAISS IndexFlatIP)

Query sets evaluated: Q1 (3,766), Q2 (3,766), Q3 (3,542)

Model  : BAAI/bge-base-en-v1.5
Device : CUDA (A40)
Prec   : bfloat16

Output : /DATA5/prabhakar/telecom_retrieval/reports/m3_dense_results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer

# ── project root on PYTHONPATH ────────────────────────────────────────────────
PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import evaluate_run, load_duplicate_mapping  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
Q1_PATH      = PROJECT_ROOT / "queries" / "q1_captions.json"
Q2_PATH      = PROJECT_ROOT / "queries" / "q2_paraphrased.json"
Q3_PATH      = PROJECT_ROOT / "queries" / "q3_context.json"
DUP_MAP_PATH = PROJECT_ROOT / "eval"    / "duplicate_mapping.json"
REPORT_PATH  = PROJECT_ROOT / "reports" / "m3_dense_results.json"
HF_CACHE     = Path("/DATA5/prabhakar/hf_cache")

MODEL_NAME   = "BAAI/bge-base-en-v1.5"
EMBED_DIM    = 768
BATCH_SIZE   = 256   # large batches for A40 throughput
TOP_K        = 100

# BGE instruction prefix for asymmetric retrieval (document side uses none;
# query side benefits from the instruction prefix per BGE paper).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus() -> pd.DataFrame:
    """Load master CSV, validate required columns."""
    log.info("Loading corpus from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    required = {"Image Caption", "Context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
    log.info("Corpus: %d rows, cols: %s", len(df), list(df.columns))
    return df


def load_query_set(path: Path) -> List[Dict]:
    """Load a JSON query file."""
    log.info("Loading query set: %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    queries = data["queries"]
    meta    = data.get("metadata", {})
    log.info("  %d queries  (set=%s)", len(queries), meta.get("query_set", path.stem))
    return queries


def build_doc_texts_caption(df: pd.DataFrame) -> List[str]:
    """Document texts for D1: Image Caption only."""
    return df["Image Caption"].fillna("").astype(str).tolist()


def build_doc_texts_cap_ctx(df: pd.DataFrame) -> List[str]:
    """
    Document texts for D2: Image Caption + Context (concatenated).
    NaNs handled before concat; BGE tokeniser will truncate at 512 tokens.
    """
    captions = df["Image Caption"].fillna("").astype(str)
    contexts = df["Context"].fillna("").astype(str)
    return (captions + " " + contexts).tolist()


def build_query_texts(queries: List[Dict], use_prefix: bool = True) -> List[str]:
    """
    Extract raw text from query dicts, optionally prepending the BGE
    query-side instruction prefix.
    """
    texts = [q["text"] for q in queries]
    if use_prefix:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Model & Encoding
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: str) -> SentenceTransformer:
    """Load BGE-base model onto *device*."""
    log.info("Loading model: %s  (device=%s)", MODEL_NAME, device)
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(HF_CACHE),
        device=device,
    )
    # Cast underlying transformer to bfloat16 for A40 efficiency
    model = model.half()   # float16; bfloat16 not exposed cleanly in ST
    log.info("  Model loaded in %.2f s", time.perf_counter() - t0)
    return model


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = BATCH_SIZE,
    normalize: bool = True,
    desc: str = "",
) -> np.ndarray:
    """
    Encode *texts* into L2-normalised float32 numpy embeddings.

    Uses sentence_transformers' built-in batched encoding with progress bar.
    The model is in fp16; we upcast to float32 for FAISS compatibility.
    """
    log.info("  Encoding %d texts%s …", len(texts), f"  [{desc}]" if desc else "")
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,   # L2 normalise → cosine sim via dot product
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    log.info("  Encoded in %.2f s  shape=%s", time.perf_counter() - t0, embeddings.shape)
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(doc_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product FAISS index.

    Because embeddings are L2-normalised, inner product == cosine similarity.
    IndexFlatIP is exact (brute-force) — appropriate for 3,766 docs.
    """
    log.info("  Building FAISS IndexFlatIP  (dim=%d, docs=%d) …",
             doc_embeddings.shape[1], doc_embeddings.shape[0])
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    log.info("  FAISS index built in %.3f s  (ntotal=%d)", time.perf_counter() - t0, index.ntotal)
    return index


def retrieve_dense(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[List[int]]:
    """
    Run batched FAISS search and return ranked row-index lists.

    Parameters
    ----------
    index            : pre-built FAISS index
    query_embeddings : (n_queries, dim) float32 array, L2-normalised
    top_k            : number of candidates to return

    Returns
    -------
    List of ranked row-index lists (one per query, best score first).
    """
    log.info("  Searching top-%d for %d queries …", top_k, len(query_embeddings))
    t0 = time.perf_counter()
    _scores, indices = index.search(query_embeddings, top_k)
    elapsed = time.perf_counter() - t0
    log.info("  Search done in %.3f s  (%.2f ms/query)",
             elapsed, elapsed / len(query_embeddings) * 1000)
    # FAISS returns -1 for empty slots; filter them out
    return [
        [int(i) for i in row if i >= 0]
        for row in indices
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    doc_texts: List[str],
    query_sets: Dict[str, List[Dict]],
    model: SentenceTransformer,
    hash_to_rows: Dict,
    row_to_hash: Dict,
    top_k: int = TOP_K,
) -> Dict:
    """
    Encode docs → build index → encode each query set → retrieve → evaluate.

    Returns a result dict keyed by query-set name.
    """
    log.info("\n%s  Experiment: %s  %s", "─" * 28, name, "─" * 28)

    # 1. Encode corpus documents (NO instruction prefix for document side)
    doc_embeddings = encode_texts(model, doc_texts, normalize=True, desc="corpus")

    # 2. Build FAISS index
    index = build_faiss_index(doc_embeddings)

    # 3. Evaluate each query set
    exp_results: Dict[str, Dict] = {}
    for qset_name, queries in query_sets.items():
        log.info("  → Query set: %s  (%d queries)", qset_name, len(queries))

        # Encode queries WITH BGE instruction prefix
        q_texts     = build_query_texts(queries, use_prefix=True)
        q_embeddings = encode_texts(model, q_texts, normalize=True, desc=f"{qset_name} queries")

        # Retrieve
        preds = retrieve_dense(index, q_embeddings, top_k=top_k)

        # Evaluate
        metrics = evaluate_run(
            queries,
            preds,
            hash_to_rows,
            row_to_hash,
            k_values=(1, 2, 3, 5, 10),
            mrr_k=10,
        )
        exp_results[qset_name] = metrics
        log.info(
            "    R@1=%.4f  R@2=%.4f  R@3=%.4f  R@5=%.4f  R@10=%.4f  MRR@10=%.4f"
            "  | dupR@10=%.4f  dupMRR@10=%.4f",
            metrics["recall@1"],
            metrics["recall@2"],
            metrics["recall@3"],
            metrics["recall@5"],
            metrics["recall@10"],
            metrics["mrr@10"],
            metrics["dup_recall@10"],
            metrics["dup_mrr@10"],
        )

    return exp_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def _load_bm25_results() -> Dict:
    """Load M2 BM25 results for comparison table."""
    bm25_path = PROJECT_ROOT / "reports" / "m2_bm25_results.json"
    if not bm25_path.exists():
        log.warning("BM25 results not found at %s — skipping comparison.", bm25_path)
        return {}
    with bm25_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_summary_table(
    d1: Dict[str, Dict],
    d2: Dict[str, Dict],
    bm25: Dict,
) -> None:
    """Print a comparison table of D1/D2 vs B1/B2 at R@1/2/3/5/10."""

    b1 = bm25.get("experiments", {}).get("B1_caption_only",    {}).get("results", {})
    b2 = bm25.get("experiments", {}).get("B2_caption_context", {}).get("results", {})

    header = (
        f"\n{'=' * 106}\n"
        f"  MILESTONE 3 — DENSE (BGE-base) vs BM25 BASELINE COMPARISON\n"
        f"  Model: {MODEL_NAME}\n"
        f"{'=' * 106}\n"
        f"  {'Query':^5}  {'Experiment':^28}"
        f"  {'R@1':>7}  {'R@2':>7}  {'R@3':>7}  {'R@5':>7}  {'R@10':>7}  {'MRR@10':>8}"
        f"  {'dupR@10':>9}  {'dupMRR@10':>11}\n"
        f"  {'-' * 100}"
    )
    print(header)

    q_order = ["Q1", "Q2", "Q3"]
    experiments = [
        ("B1 BM25 (Caption)",  b1),
        ("B2 BM25 (Cap+Ctx)", b2),
        ("D1 BGE  (Caption)", d1),
        ("D2 BGE  (Cap+Ctx)", d2),
    ]

    for qset in q_order:
        for tag, results_dict in experiments:
            m = results_dict.get(qset, {})
            print(
                f"  {qset:^5}  {tag:^28}"
                f"  {m.get('recall@1',    0):>7.4f}"
                f"  {m.get('recall@2',    0):>7.4f}"
                f"  {m.get('recall@3',    0):>7.4f}"
                f"  {m.get('recall@5',    0):>7.4f}"
                f"  {m.get('recall@10',   0):>7.4f}"
                f"  {m.get('mrr@10',      0):>8.4f}"
                f"  {m.get('dup_recall@10', 0):>9.4f}"
                f"  {m.get('dup_mrr@10',   0):>11.4f}"
            )
        if qset != q_order[-1]:
            print(f"  {'-' * 100}")

    print(f"{'=' * 106}\n")

    # Q2 focus: paraphrase mismatch analysis
    b1_q2 = b1.get("Q2", {}).get("recall@10", 0)
    b2_q2 = b2.get("Q2", {}).get("recall@10", 0)
    d1_q2 = d1.get("Q2", {}).get("recall@10", 0)
    d2_q2 = d2.get("Q2", {}).get("recall@10", 0)
    print("  ── Q2 Paraphrase Mismatch Analysis (R@10) ──────────────────────────────")
    print(f"     B1 BM25 Caption  : {b1_q2:.4f}")
    print(f"     B2 BM25 Cap+Ctx  : {b2_q2:.4f}")
    print(f"     D1 BGE  Caption  : {d1_q2:.4f}  (Δ vs B1: {d1_q2 - b1_q2:+.4f})")
    print(f"     D2 BGE  Cap+Ctx  : {d2_q2:.4f}  (Δ vs B2: {d2_q2 - b2_q2:+.4f})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("  Milestone 3 — Dense Semantic Baselines  (BGE-base-en-v1.5)")
    log.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  (GPUs available: %d)", device, torch.cuda.device_count())

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    df = load_corpus()

    doc_caption     = build_doc_texts_caption(df)
    doc_cap_ctx     = build_doc_texts_cap_ctx(df)
    log.info("D1 corpus (Caption only)   : %d documents", len(doc_caption))
    log.info("D2 corpus (Caption+Context): %d documents", len(doc_cap_ctx))

    # ── 2. Load query sets ────────────────────────────────────────────────────
    q1 = load_query_set(Q1_PATH)
    q2 = load_query_set(Q2_PATH)
    q3 = load_query_set(Q3_PATH)
    query_sets = {"Q1": q1, "Q2": q2, "Q3": q3}

    # ── 3. Load duplicate mapping ─────────────────────────────────────────────
    log.info("Loading duplicate mapping …")
    hash_to_rows, row_to_hash = load_duplicate_mapping(DUP_MAP_PATH)

    # ── 4. Load model once ────────────────────────────────────────────────────
    model = load_model(device)

    # ── 5. Run D1 ─────────────────────────────────────────────────────────────
    d1_results = run_experiment(
        name       = "D1: Caption only",
        doc_texts  = doc_caption,
        query_sets = query_sets,
        model      = model,
        hash_to_rows = hash_to_rows,
        row_to_hash  = row_to_hash,
    )

    # ── 6. Run D2 ─────────────────────────────────────────────────────────────
    d2_results = run_experiment(
        name       = "D2: Caption + Context",
        doc_texts  = doc_cap_ctx,
        query_sets = query_sets,
        model      = model,
        hash_to_rows = hash_to_rows,
        row_to_hash  = row_to_hash,
    )

    # ── 7. Assemble report ────────────────────────────────────────────────────
    report = {
        "milestone"   : "M3",
        "description" : "Dense semantic baselines — BAAI/bge-base-en-v1.5",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model"       : MODEL_NAME,
        "embed_dim"   : EMBED_DIM,
        "index_type"  : "IndexFlatIP (exact inner product, L2-normalised = cosine)",
        "retrieval_depth": TOP_K,
        "bge_query_prefix": BGE_QUERY_PREFIX,
        "corpus": {
            "csv_path"  : str(CSV_PATH),
            "total_rows": len(df),
        },
        "experiments": {
            "D1_caption_only": {
                "description": "BGE-base on Image Caption column",
                "results"    : d1_results,
            },
            "D2_caption_context": {
                "description": "BGE-base on Image Caption + Context (concatenated, BGE 512-tok truncation)",
                "results"    : d2_results,
            },
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report saved → %s", REPORT_PATH)

    # ── 8. Console summary ────────────────────────────────────────────────────
    bm25_data = _load_bm25_results()
    _print_summary_table(d1_results, d2_results, bm25_data)

    log.info("✅  Milestone 3 complete.")


if __name__ == "__main__":
    main()
