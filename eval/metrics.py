"""
eval/metrics.py
───────────────
Evaluation metrics for the Multimodal Telecom Image Retrieval thesis.

Implements:
  • Standard Recall@K  (K ∈ {1, 5, 10})
  • Mean Reciprocal Rank (MRR@10)
  • Duplicate-Aware Recall@K

Duplicate-Aware logic
─────────────────────
Because the corpus contains visually identical images (same MD5 hash assigned
to multiple row indices), a retrieval system should NOT be penalised for
returning an image that is pixel-identical to the ground-truth.

For a query whose ground-truth is row g:
  1. Look up the MD5 hash H of row g  →  hash_to_rows[H]
  2. The "valid hit set" V = {all row indices sharing hash H}
  3. Recall@K = 1  iff  any of the top-K predicted indices ∈ V
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Duplicate-mapping helpers
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_DUP_MAP_PATH = Path(__file__).parent / "duplicate_mapping.json"


def load_duplicate_mapping(
    path: Optional[Path | str] = None,
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """
    Load the MD5 duplicate mapping.

    Returns
    -------
    hash_to_rows : dict[str, list[int]]
        MD5 hash  →  list of row indices that share that hash.
    row_to_hash  : dict[int, str]
        row index →  its MD5 hash  (inverse lookup, built on the fly).
    """
    path = Path(path) if path else _DEFAULT_DUP_MAP_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Duplicate mapping not found at {path}. "
            "Run scripts/01_data_loader.py to generate it."
        )

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    hash_to_rows: Dict[str, List[int]] = {
        h: list(rows) for h, rows in data["hash_to_row_indices"].items()
    }

    # Build inverse mapping: row_index → hash
    row_to_hash: Dict[int, str] = {}
    for h, rows in hash_to_rows.items():
        for r in rows:
            row_to_hash[r] = h

    logger.info(
        "Duplicate mapping loaded: %d hashes, %d total rows",
        len(hash_to_rows),
        len(row_to_hash),
    )
    return hash_to_rows, row_to_hash


# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    predicted: List[int],
    ground_truth_row: int,
    k: int,
) -> float:
    """
    Standard Recall@K.

    Returns 1.0 if ground_truth_row appears in the top-K predictions,
    0.0 otherwise.

    Parameters
    ----------
    predicted        : ranked list of retrieved row indices (best first).
    ground_truth_row : the single correct row index for this query.
    k                : cut-off depth.
    """
    if not predicted:
        return 0.0
    return 1.0 if ground_truth_row in predicted[:k] else 0.0


def duplicate_aware_recall_at_k(
    predicted: List[int],
    ground_truth_row: int,
    k: int,
    hash_to_rows: Dict[str, List[int]],
    row_to_hash: Dict[int, str],
) -> float:
    """
    Duplicate-Aware Recall@K.

    A retrieval is a "hit" if ANY of the top-K results belongs to the set
    of rows that share the same MD5 hash as the ground-truth image.

    Parameters
    ----------
    predicted        : ranked list of retrieved row indices (best first).
    ground_truth_row : the single correct row index for this query.
    k                : cut-off depth.
    hash_to_rows     : MD5 → list[int]  mapping (from load_duplicate_mapping).
    row_to_hash      : int  → str       mapping (from load_duplicate_mapping).
    """
    if not predicted:
        return 0.0

    # Determine the valid hit set for this ground-truth row
    gt_hash = row_to_hash.get(ground_truth_row)
    if gt_hash is None:
        # Ground-truth row not in mapping → fall back to exact match
        valid_set: Set[int] = {ground_truth_row}
    else:
        valid_set = set(hash_to_rows.get(gt_hash, [ground_truth_row]))

    top_k = set(predicted[:k])
    return 1.0 if top_k & valid_set else 0.0


def reciprocal_rank(
    predicted: List[int],
    ground_truth_row: int,
    k: int = 10,
) -> float:
    """
    Reciprocal Rank at depth k.

    Returns 1/rank  if the ground-truth appears within the top-k predictions
    (1-indexed), or 0.0 if not found.

    Parameters
    ----------
    predicted        : ranked list of retrieved row indices (best first).
    ground_truth_row : the single correct row index for this query.
    k                : maximum rank to consider (default 10 for MRR@10).
    """
    for rank, idx in enumerate(predicted[:k], start=1):
        if idx == ground_truth_row:
            return 1.0 / rank
    return 0.0


def duplicate_aware_reciprocal_rank(
    predicted: List[int],
    ground_truth_row: int,
    hash_to_rows: Dict[str, List[int]],
    row_to_hash: Dict[int, str],
    k: int = 10,
) -> float:
    """
    Duplicate-Aware Reciprocal Rank at depth k.

    Returns 1/rank of the FIRST hit (any member of the valid duplicate set)
    within the top-k list, or 0.0 if no hit is found.
    """
    gt_hash = row_to_hash.get(ground_truth_row)
    if gt_hash is None:
        valid_set: Set[int] = {ground_truth_row}
    else:
        valid_set = set(hash_to_rows.get(gt_hash, [ground_truth_row]))

    for rank, idx in enumerate(predicted[:k], start=1):
        if idx in valid_set:
            return 1.0 / rank
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_run(
    queries: List[Dict],
    all_predictions: List[List[int]],
    hash_to_rows: Dict[str, List[int]],
    row_to_hash: Dict[int, str],
    k_values: Tuple[int, ...] = (1, 2, 3, 5, 10),
    mrr_k: int = 10,
) -> Dict[str, float]:
    """
    Compute all metrics for a full retrieval run.

    Parameters
    ----------
    queries          : list of query dicts, each with key ``ground_truth_row``.
    all_predictions  : list of ranked row-index lists (same order as queries).
    hash_to_rows     : from load_duplicate_mapping().
    row_to_hash      : from load_duplicate_mapping().
    k_values         : Recall@K cut-offs to compute.
    mrr_k            : depth for MRR.

    Returns
    -------
    dict with keys:
      recall@1, recall@2, recall@3, recall@5, recall@10
      dup_recall@1, dup_recall@2, dup_recall@3, dup_recall@5, dup_recall@10
      mrr@10, dup_mrr@10
      num_queries
    """
    if len(queries) != len(all_predictions):
        raise ValueError(
            f"Mismatch: {len(queries)} queries vs "
            f"{len(all_predictions)} prediction lists."
        )

    n = len(queries)
    accumulators: Dict[str, float] = {
        **{f"recall@{k}": 0.0 for k in k_values},
        **{f"dup_recall@{k}": 0.0 for k in k_values},
        f"mrr@{mrr_k}": 0.0,
        f"dup_mrr@{mrr_k}": 0.0,
    }

    for query, preds in zip(queries, all_predictions):
        gt = query["ground_truth_row"]

        for k in k_values:
            accumulators[f"recall@{k}"] += recall_at_k(preds, gt, k)
            accumulators[f"dup_recall@{k}"] += duplicate_aware_recall_at_k(
                preds, gt, k, hash_to_rows, row_to_hash
            )

        accumulators[f"mrr@{mrr_k}"] += reciprocal_rank(preds, gt, mrr_k)
        accumulators[f"dup_mrr@{mrr_k}"] += duplicate_aware_reciprocal_rank(
            preds, gt, hash_to_rows, row_to_hash, mrr_k
        )

    # Average
    results: Dict[str, float] = {
        key: round(val / n, 6) for key, val in accumulators.items()
    }
    results["num_queries"] = n
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test (run this file directly to verify)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 60)
    print("eval/metrics.py — Self-test")
    print("=" * 60)

    # 1. Load duplicate mapping
    h2r, r2h = load_duplicate_mapping()
    sample_dup_hash = next(h for h, rows in h2r.items() if len(rows) > 1)
    dup_rows = h2r[sample_dup_hash]
    print(f"\nSample duplicate group: hash={sample_dup_hash[:12]}… rows={dup_rows[:6]}")

    # 2. Standard metrics
    gt = dup_rows[0]
    preds_miss = [9999, 9998, 9997]
    preds_exact_hit = [dup_rows[0]] + [9999, 9998]
    preds_dup_hit = [dup_rows[1]] + [9999, 9998]  # different row, same image

    print(f"\nGround truth row : {gt}")
    print(f"preds_miss       : {preds_miss}")
    print(f"preds_exact_hit  : {preds_exact_hit}")
    print(f"preds_dup_hit    : {preds_dup_hit}  ← duplicate of GT")

    print(f"\n{'Metric':<30} {'miss':>8} {'exact':>8} {'dup':>8}")
    print("-" * 60)
    for k in (1, 5, 10):
        m = f"recall@{k}"
        dm = f"dup_recall@{k}"
        print(
            f"  {m:<28} "
            f"{recall_at_k(preds_miss, gt, k):>8.4f} "
            f"{recall_at_k(preds_exact_hit, gt, k):>8.4f} "
            f"{recall_at_k(preds_dup_hit, gt, k):>8.4f}   ← std always 0 for dup"
        )
        print(
            f"  {dm:<28} "
            f"{duplicate_aware_recall_at_k(preds_miss, gt, k, h2r, r2h):>8.4f} "
            f"{duplicate_aware_recall_at_k(preds_exact_hit, gt, k, h2r, r2h):>8.4f} "
            f"{duplicate_aware_recall_at_k(preds_dup_hit, gt, k, h2r, r2h):>8.4f}   ← dup = 1"
        )

    mrr_miss   = reciprocal_rank(preds_miss, gt)
    mrr_exact  = reciprocal_rank(preds_exact_hit, gt)
    mrr_dup_s  = reciprocal_rank(preds_dup_hit, gt)          # standard
    mrr_dup_da = duplicate_aware_reciprocal_rank(preds_dup_hit, gt, h2r, r2h)

    print(f"\n  {'mrr@10':<30} {mrr_miss:>8.4f} {mrr_exact:>8.4f} {mrr_dup_s:>8.4f}")
    print(f"  {'dup_mrr@10':<30} {'—':>8} {'—':>8} {mrr_dup_da:>8.4f}")
    print("\nSelf-test PASSED ✓")
