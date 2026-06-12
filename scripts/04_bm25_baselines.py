"""
scripts/04_bm25_baselines.py
────────────────────────────
Milestone 2 — BM25 Lexical Baselines

Experiments
───────────
  B1  BM25 on Image Caption only
  B2  BM25 on Image Caption + Context (concatenated)

Query sets evaluated: Q1 (3 766), Q2 (3 766), Q3 (3 542)

Corpus: /DATA1/prabhakar/telecom/All Images Path.csv
Output: /DATA5/prabhakar/telecom_retrieval/reports/m2_bm25_results.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi

# ── project root on PYTHONPATH ────────────────────────────────────────────────
PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import evaluate_run, load_duplicate_mapping  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH     = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
Q1_PATH      = PROJECT_ROOT / "queries" / "q1_captions.json"
Q2_PATH      = PROJECT_ROOT / "queries" / "q2_paraphrased.json"
Q3_PATH      = PROJECT_ROOT / "queries" / "q3_context.json"
DUP_MAP_PATH = PROJECT_ROOT / "eval" / "duplicate_mapping.json"
REPORT_PATH  = PROJECT_ROOT / "reports" / "m2_bm25_results.json"

# ──────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> List[str]:
    """Lowercase + alphanumeric tokenisation. Returns [] for NaN / empty."""
    if not isinstance(text, str) or not text.strip():
        return []
    return _TOKEN_RE.findall(text.lower())


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_corpus() -> pd.DataFrame:
    """Load the master CSV. Validate required columns."""
    log.info("Loading corpus from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    required = {"Image Caption", "Context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Found: {list(df.columns)}")
    log.info("Corpus loaded: %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_query_set(path: Path) -> List[Dict]:
    """Load a JSON query file and return the list of query dicts."""
    log.info("Loading query set from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    queries = data["queries"]
    meta    = data.get("metadata", {})
    log.info(
        "  Loaded %d queries  (set=%s)",
        len(queries),
        meta.get("query_set", path.stem),
    )
    return queries


# ──────────────────────────────────────────────────────────────────────────────
# BM25 retrieval
# ──────────────────────────────────────────────────────────────────────────────

def build_bm25_index(tokenised_corpus: List[List[str]]) -> BM25Okapi:
    """Fit and return a BM25Okapi index over the tokenised corpus."""
    t0 = time.perf_counter()
    index = BM25Okapi(tokenised_corpus)
    log.info("  BM25 index built in %.2f s", time.perf_counter() - t0)
    return index


def retrieve_bm25(
    index: BM25Okapi,
    queries: List[Dict],
    top_k: int = 100,
) -> List[List[int]]:
    """
    Retrieve top-K row indices for each query using a pre-built BM25 index.

    Parameters
    ----------
    index    : fitted BM25Okapi
    queries  : list of query dicts, each with key 'text'
    top_k    : number of candidates to return per query

    Returns
    -------
    List of ranked row-index lists (one per query, best first).
    """
    log.info("  Retrieving top-%d results for %d queries …", top_k, len(queries))
    t0 = time.perf_counter()

    all_preds: List[List[int]] = []
    for q in queries:
        tokens = tokenise(q["text"])
        if not tokens:
            # Empty query → return empty ranking (will score 0 everywhere)
            all_preds.append([])
            continue
        scores = index.get_scores(tokens)
        # argsort descending; numpy not always available, use sorted()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        all_preds.append(ranked)

    elapsed = time.perf_counter() - t0
    log.info("  Retrieval done in %.2f s  (%.1f ms/query)", elapsed, elapsed / len(queries) * 1000)
    return all_preds


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    tokenised_corpus: List[List[str]],
    query_sets: Dict[str, List[Dict]],
    hash_to_rows: Dict,
    row_to_hash: Dict,
    top_k: int = 100,
) -> Dict:
    """
    Build one BM25 index and evaluate it against all query sets.

    Returns a result dict keyed by query-set name.
    """
    log.info("\n%s  Experiment: %s  %s", "─" * 30, name, "─" * 30)
    index = build_bm25_index(tokenised_corpus)

    exp_results: Dict[str, Dict] = {}
    for qset_name, queries in query_sets.items():
        log.info("  → Evaluating on %s (%d queries) …", qset_name, len(queries))
        preds = retrieve_bm25(index, queries, top_k=top_k)
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
            "    R@1=%.4f  R@2=%.4f  R@3=%.4f  R@5=%.4f  R@10=%.4f  MRR@10=%.4f  "
            "| dupR@10=%.4f  dupMRR@10=%.4f",
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


def main() -> None:
    log.info("=" * 70)
    log.info("  Milestone 2 — BM25 Baselines")
    log.info("=" * 70)

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    df = load_corpus()

    # Graceful NaN handling: treat as empty string
    captions = df["Image Caption"].fillna("").astype(str).tolist()
    contexts = df["Context"].fillna("").astype(str).tolist()

    # ── 2. Build tokenised corpora ────────────────────────────────────────────
    log.info("\nTokenising corpora …")

    tok_caption = [tokenise(c) for c in captions]
    log.info("  B1 corpus (Caption only)    : %d documents", len(tok_caption))

    tok_caption_context = [
        tokenise(cap + " " + ctx)
        for cap, ctx in zip(captions, contexts)
    ]
    log.info("  B2 corpus (Caption+Context) : %d documents", len(tok_caption_context))

    # ── 3. Load query sets ────────────────────────────────────────────────────
    q1 = load_query_set(Q1_PATH)
    q2 = load_query_set(Q2_PATH)
    q3 = load_query_set(Q3_PATH)

    query_sets = {"Q1": q1, "Q2": q2, "Q3": q3}

    # ── 4. Load duplicate mapping ─────────────────────────────────────────────
    log.info("\nLoading duplicate mapping …")
    hash_to_rows, row_to_hash = load_duplicate_mapping(DUP_MAP_PATH)

    # ── 5. Run experiments ────────────────────────────────────────────────────
    b1_results = run_experiment(
        "B1: Caption only",
        tok_caption,
        query_sets,
        hash_to_rows,
        row_to_hash,
    )

    b2_results = run_experiment(
        "B2: Caption + Context",
        tok_caption_context,
        query_sets,
        hash_to_rows,
        row_to_hash,
    )

    # ── 6. Assemble full report ───────────────────────────────────────────────
    report = {
        "milestone": "M2",
        "description": "BM25 lexical baselines",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": {
            "csv_path": str(CSV_PATH),
            "total_rows": len(df),
        },
        "tokeniser": "lowercase + re[a-z0-9]+",
        "retrieval_depth": 100,
        "experiments": {
            "B1_caption_only": {
                "description": "BM25Okapi on Image Caption column",
                "results": b1_results,
            },
            "B2_caption_context": {
                "description": "BM25Okapi on Image Caption + Context (concatenated)",
                "results": b2_results,
            },
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("\nReport saved → %s", REPORT_PATH)

    # ── 7. Console summary table ──────────────────────────────────────────────
    _print_summary_table(b1_results, b2_results)

    log.info("\n✅  Milestone 2 complete.")


def _print_summary_table(
    b1: Dict[str, Dict],
    b2: Dict[str, Dict],
) -> None:
    """Print a formatted comparison table to stdout."""

    HEADER = (
        f"\n{'=' * 96}\n"
        f"  MILESTONE 2 — BM25 BASELINE SUMMARY\n"
        f"{'=' * 96}\n"
        f"  {'Query':^6}  {'Experiment':^26}"
        f"  {'R@1':>7}  {'R@2':>7}  {'R@3':>7}  {'R@5':>7}  {'R@10':>7}  {'MRR@10':>7}"
        f"  {'dupR@10':>9}  {'dupMRR@10':>10}\n"
        f"  {'-' * 90}"
    )
    print(HEADER)

    q_order = ["Q1", "Q2", "Q3"]
    for qset in q_order:
        for tag, results_dict in [("B1 (Caption)", b1), ("B2 (Cap+Ctx)", b2)]:
            m = results_dict.get(qset, {})
            print(
                f"  {qset:^6}  {tag:^26}"
                f"  {m.get('recall@1',  0):>7.4f}"
                f"  {m.get('recall@2',  0):>7.4f}"
                f"  {m.get('recall@3',  0):>7.4f}"
                f"  {m.get('recall@5',  0):>7.4f}"
                f"  {m.get('recall@10', 0):>7.4f}"
                f"  {m.get('mrr@10',    0):>7.4f}"
                f"  {m.get('dup_recall@10', 0):>9.4f}"
                f"  {m.get('dup_mrr@10',    0):>10.4f}"
            )
        if qset != q_order[-1]:
            print(f"  {'-' * 90}")

    print(f"{'=' * 96}\n")


if __name__ == "__main__":
    main()
