"""
scripts/06_dense_large_baselines.py
────────────────────────────────────
Milestone 4 — Dense Semantic Baselines (BGE-large-en-v1.5)

Experiments
───────────
  L1  Dense on Image Caption only                 (1024-d, FAISS IndexFlatIP)
  L2  Dense on Image Caption + Context (concat)   (1024-d, FAISS IndexFlatIP)

Query sets evaluated: Q1 (3,766), Q2 (3,766), Q3 (3,542)

Model  : BAAI/bge-large-en-v1.5
Device : CUDA (A40)
Prec   : float16

Output : /DATA5/prabhakar/telecom_retrieval/reports/m4_dense_large_results.json
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
# Configuration  (only MODEL_NAME, EMBED_DIM, REPORT_PATH differ from M3)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
Q1_PATH      = PROJECT_ROOT / "queries" / "q1_captions.json"
Q2_PATH      = PROJECT_ROOT / "queries" / "q2_paraphrased.json"
Q3_PATH      = PROJECT_ROOT / "queries" / "q3_context.json"
DUP_MAP_PATH = PROJECT_ROOT / "eval"    / "duplicate_mapping.json"
REPORT_PATH  = PROJECT_ROOT / "reports" / "m4_dense_large_results.json"
HF_CACHE     = Path("/DATA5/prabhakar/hf_cache")

MODEL_NAME   = "BAAI/bge-large-en-v1.5"   # ← M4: large model
EMBED_DIM    = 1024                         # ← M4: 1024-d embeddings
BATCH_SIZE   = 128   # smaller batches for larger model on A40
TOP_K        = 100

# BGE instruction prefix — identical to M3 (same paper, same asymmetric scheme)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers  (identical to M3)
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
    """Document texts for L1: Image Caption only."""
    return df["Image Caption"].fillna("").astype(str).tolist()


def build_doc_texts_cap_ctx(df: pd.DataFrame) -> List[str]:
    """
    Document texts for L2: Image Caption + Context (concatenated).
    NaNs handled before concat; BGE tokeniser will truncate at 512 tokens.
    """
    captions = df["Image Caption"].fillna("").astype(str)
    contexts = df["Context"].fillna("").astype(str)
    return (captions + " " + contexts).tolist()


def build_query_texts(queries: List[Dict], use_prefix: bool = True) -> List[str]:
    """Extract raw text from query dicts, optionally prepending BGE query prefix."""
    texts = [q["text"] for q in queries]
    if use_prefix:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Model & Encoding
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: str) -> SentenceTransformer:
    """Load BGE-large model onto *device*."""
    log.info("Loading model: %s  (device=%s)", MODEL_NAME, device)
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(HF_CACHE),
        device=device,
    )
    model = model.half()   # float16 on A40
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
    Model runs in fp16; output upcast to float32 for FAISS.
    """
    log.info("  Encoding %d texts%s …", len(texts), f"  [{desc}]" if desc else "")
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
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
    L2-normalised embeddings → inner product == cosine similarity.
    IndexFlatIP is exact — appropriate for 3,766 docs.
    """
    log.info("  Building FAISS IndexFlatIP  (dim=%d, docs=%d) …",
             doc_embeddings.shape[1], doc_embeddings.shape[0])
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    log.info("  FAISS index built in %.3f s  (ntotal=%d)",
             time.perf_counter() - t0, index.ntotal)
    return index


def retrieve_dense(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[List[int]]:
    """Run batched FAISS search; return ranked row-index lists."""
    log.info("  Searching top-%d for %d queries …", top_k, len(query_embeddings))
    t0 = time.perf_counter()
    _scores, indices = index.search(query_embeddings, top_k)
    elapsed = time.perf_counter() - t0
    log.info("  Search done in %.3f s  (%.2f ms/query)",
             elapsed, elapsed / len(query_embeddings) * 1000)
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

    # 1. Encode corpus (NO instruction prefix for document side)
    doc_embeddings = encode_texts(model, doc_texts, normalize=True, desc="corpus")

    # 2. Build FAISS index
    index = build_faiss_index(doc_embeddings)

    # 3. Evaluate each query set
    exp_results: Dict[str, Dict] = {}
    for qset_name, queries in query_sets.items():
        log.info("  → Query set: %s  (%d queries)", qset_name, len(queries))

        # Encode queries WITH BGE instruction prefix
        q_texts      = build_query_texts(queries, use_prefix=True)
        q_embeddings = encode_texts(model, q_texts, normalize=True,
                                    desc=f"{qset_name} queries")

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

def _load_json(path: Path) -> Dict:
    """Load a JSON results file; warn and return {} if missing."""
    if not path.exists():
        log.warning("Results file not found at %s — skipping.", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_summary_table(
    l1: Dict[str, Dict],
    l2: Dict[str, Dict],
    m3_data: Dict,
    bm25_data: Dict,
) -> None:
    """
    Print a 6-row comparison table:
      B1 BM25 Caption, D1 BGE-base Caption, L1 BGE-large Caption
      B2 BM25 Cap+Ctx, D2 BGE-base Cap+Ctx, L2 BGE-large Cap+Ctx
    per query set (Q1, Q2, Q3).
    """
    b1 = bm25_data.get("experiments", {}).get("B1_caption_only",    {}).get("results", {})
    b2 = bm25_data.get("experiments", {}).get("B2_caption_context", {}).get("results", {})
    d1 = m3_data.get("experiments",  {}).get("D1_caption_only",    {}).get("results", {})
    d2 = m3_data.get("experiments",  {}).get("D2_caption_context", {}).get("results", {})

    header = (
        f"\n{'=' * 116}\n"
        f"  MILESTONE 4 — BGE-large vs BGE-base vs BM25 COMPARISON\n"
        f"  Model: {MODEL_NAME}\n"
        f"{'=' * 116}\n"
        f"  {'Query':^5}  {'Experiment':^30}"
        f"  {'R@1':>7}  {'R@2':>7}  {'R@3':>7}  {'R@5':>7}  {'R@10':>7}  {'MRR@10':>8}"
        f"  {'dupR@10':>9}  {'dupMRR@10':>11}\n"
        f"  {'-' * 110}"
    )
    print(header)

    q_order = ["Q1", "Q2", "Q3"]
    experiments = [
        ("B1 BM25  (Caption)",    b1),
        ("D1 BGE-base (Caption)", d1),
        ("L1 BGE-large (Caption)", l1),
        ("B2 BM25  (Cap+Ctx)",    b2),
        ("D2 BGE-base (Cap+Ctx)", d2),
        ("L2 BGE-large (Cap+Ctx)", l2),
    ]

    for qset in q_order:
        for tag, results_dict in experiments:
            m = results_dict.get(qset, {})
            print(
                f"  {qset:^5}  {tag:^30}"
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
            print(f"  {'-' * 110}")

    print(f"{'=' * 116}\n")

    # Q2 focus: scaling gain analysis at R@1 and MRR@10
    for label, base_d, large_d in [
        ("Caption-only",   d1, l1),
        ("Cap+Ctx",        d2, l2),
    ]:
        b_r1  = base_d.get("Q2",  {}).get("recall@1",  0)
        l_r1  = large_d.get("Q2", {}).get("recall@1",  0)
        b_mrr = base_d.get("Q2",  {}).get("mrr@10",    0)
        l_mrr = large_d.get("Q2", {}).get("mrr@10",    0)
        print(f"  ── Q2 Scaling Gain [{label}] ───────────────────────────────────────")
        print(f"     BGE-base  R@1={b_r1:.4f}  MRR@10={b_mrr:.4f}")
        print(f"     BGE-large R@1={l_r1:.4f}  MRR@10={l_mrr:.4f}"
              f"  (ΔR@1={l_r1 - b_r1:+.4f}  ΔMRR={l_mrr - b_mrr:+.4f})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("  Milestone 4 — Dense Semantic Baselines  (BGE-large-en-v1.5)")
    log.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  (GPUs available: %d)", device, torch.cuda.device_count())

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    df = load_corpus()

    doc_caption = build_doc_texts_caption(df)
    doc_cap_ctx = build_doc_texts_cap_ctx(df)
    log.info("L1 corpus (Caption only)   : %d documents", len(doc_caption))
    log.info("L2 corpus (Caption+Context): %d documents", len(doc_cap_ctx))

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

    # ── 5. Run L1 ─────────────────────────────────────────────────────────────
    l1_results = run_experiment(
        name         = "L1: Caption only",
        doc_texts    = doc_caption,
        query_sets   = query_sets,
        model        = model,
        hash_to_rows = hash_to_rows,
        row_to_hash  = row_to_hash,
    )

    # ── 6. Run L2 ─────────────────────────────────────────────────────────────
    l2_results = run_experiment(
        name         = "L2: Caption + Context",
        doc_texts    = doc_cap_ctx,
        query_sets   = query_sets,
        model        = model,
        hash_to_rows = hash_to_rows,
        row_to_hash  = row_to_hash,
    )

    # ── 7. Assemble report ────────────────────────────────────────────────────
    report = {
        "milestone"      : "M4",
        "description"    : "Dense semantic baselines — BAAI/bge-large-en-v1.5",
        "generated_at"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model"          : MODEL_NAME,
        "embed_dim"      : EMBED_DIM,
        "index_type"     : "IndexFlatIP (exact inner product, L2-normalised = cosine)",
        "retrieval_depth": TOP_K,
        "bge_query_prefix": BGE_QUERY_PREFIX,
        "corpus": {
            "csv_path"   : str(CSV_PATH),
            "total_rows" : len(df),
        },
        "experiments": {
            "L1_caption_only": {
                "description": "BGE-large on Image Caption column",
                "results"    : l1_results,
            },
            "L2_caption_context": {
                "description": "BGE-large on Image Caption + Context (concatenated, BGE 512-tok truncation)",
                "results"    : l2_results,
            },
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report saved → %s", REPORT_PATH)

    # ── 8. Console summary ────────────────────────────────────────────────────
    m3_data   = _load_json(PROJECT_ROOT / "reports" / "m3_dense_results.json")
    bm25_data = _load_json(PROJECT_ROOT / "reports" / "m2_bm25_results.json")
    _print_summary_table(l1_results, l2_results, m3_data, bm25_data)

    log.info("✅  Milestone 4 complete.")


if __name__ == "__main__":
    main()
