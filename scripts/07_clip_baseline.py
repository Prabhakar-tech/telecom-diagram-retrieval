"""
scripts/07_clip_baseline.py
───────────────────────────
Milestone 5 — Global Visual Baseline (CLIP)

Experiments
───────────
  CLIP-Q1: CLIP image embeddings vs Q1 Captions
  CLIP-Q2: CLIP image embeddings vs Q2 Paraphrased Captions
  CLIP-Q3: CLIP image embeddings vs Q3 Context-Extracted Queries

Model  : openai/clip-vit-base-patch32
Device : CUDA (A40)
Prec   : float16

Output : /DATA5/prabhakar/telecom_retrieval/reports/m5_clip_results.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ── project root on PYTHONPATH ────────────────────────────────────────────────
PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import evaluate_run, load_duplicate_mapping

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "m5_clip_run.log", mode="w")
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
IMAGES_DIR   = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
Q1_PATH      = PROJECT_ROOT / "queries" / "q1_captions.json"
Q2_PATH      = PROJECT_ROOT / "queries" / "q2_paraphrased.json"
Q3_PATH      = PROJECT_ROOT / "queries" / "q3_context.json"
DUP_MAP_PATH = PROJECT_ROOT / "eval"    / "duplicate_mapping.json"
REPORT_PATH  = PROJECT_ROOT / "reports" / "m5_clip_results.json"
HF_CACHE     = Path("/DATA5/prabhakar/hf_cache")

MODEL_NAME   = "openai/clip-vit-base-patch32"
EMBED_DIM    = 512
BATCH_SIZE   = 256
TOP_K        = 100

os.environ["HF_HOME"] = str(HF_CACHE)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_image_path(csv_path: str) -> str:
    basename = os.path.basename(csv_path)
    return str(IMAGES_DIR / basename)


def load_corpus() -> pd.DataFrame:
    log.info("Loading corpus from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df["resolved_path"] = df["Image Path"].apply(resolve_image_path)
    log.info("Corpus: %d rows", len(df))
    return df


def load_query_set(path: Path) -> List[Dict]:
    log.info("Loading query set: %s", path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["queries"]


# ─────────────────────────────────────────────────────────────────────────────
# Model & Encoding
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: str) -> Tuple[CLIPModel, CLIPProcessor]:
    log.info("Loading model: %s  (device=%s)", MODEL_NAME, device)
    t0 = time.perf_counter()
    model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)

    model = model.to(device)
    model.eval()

    log.info("  Model loaded in %.2f s", time.perf_counter() - t0)
    return model, processor


@torch.no_grad()
def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: List[str],
    device: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    log.info("  Encoding %d images...", len(image_paths))
    t0 = time.perf_counter()

    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                log.warning("Failed to load image %s: %s. Using blank image.", p, e)
                images.append(Image.new("RGB", (224, 224), (255, 255, 255)))

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.get_image_features(**inputs)
        if isinstance(outputs, torch.Tensor):
            image_features = outputs
        else:
            image_features = outputs.pooler_output

        # Normalize
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu().numpy())

        if (i + batch_size) % 500 < batch_size:
            log.info("    Processed %d / %d images", min(i + batch_size, len(image_paths)), len(image_paths))

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    log.info("  Encoded images in %.2f s  shape=%s", time.perf_counter() - t0, embeddings.shape)
    return embeddings


@torch.no_grad()
def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: List[str],
    device: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    log.info("  Encoding %d texts...", len(texts))
    t0 = time.perf_counter()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.get_text_features(**inputs)
        if isinstance(outputs, torch.Tensor):
            text_features = outputs
        else:
            text_features = outputs.pooler_output

        # Normalize
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(text_features.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    log.info("  Encoded texts in %.2f s  shape=%s", time.perf_counter() - t0, embeddings.shape)
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(doc_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    log.info("  Building FAISS IndexFlatIP (dim=%d, docs=%d)...",
             doc_embeddings.shape[1], doc_embeddings.shape[0])
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    log.info("  FAISS index built in %.3f s", time.perf_counter() - t0)
    return index


def retrieve_dense(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[List[int]]:
    log.info("  Searching top-%d for %d queries...", top_k, len(query_embeddings))
    t0 = time.perf_counter()
    _scores, indices = index.search(query_embeddings, top_k)
    elapsed = time.perf_counter() - t0
    log.info("  Search done in %.3f s", elapsed)
    return [[int(i) for i in row if i >= 0] for row in indices]


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_clip_experiment(
    query_sets: Dict[str, List[Dict]],
    model: CLIPModel,
    processor: CLIPProcessor,
    index: faiss.IndexFlatIP,
    device: str,
    hash_to_rows: Dict,
    row_to_hash: Dict,
) -> Tuple[Dict, Dict]:

    exp_results = {}
    all_predictions = {}

    for qset_name, queries in query_sets.items():
        log.info("  → Query set: %s (%d queries)", qset_name, len(queries))

        q_texts = [q["text"] for q in queries]
        q_embeddings = encode_texts(model, processor, q_texts, device=device)

        preds = retrieve_dense(index, q_embeddings, top_k=TOP_K)
        all_predictions[qset_name] = preds

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
            "    R@1=%.4f  R@2=%.4f  R@3=%.4f  R@5=%.4f  R@10=%.4f  MRR@10=%.4f | dupR@10=%.4f  dupMRR@10=%.4f",
            metrics["recall@1"], metrics["recall@2"], metrics["recall@3"],
            metrics["recall@5"], metrics["recall@10"], metrics["mrr@10"],
            metrics["dup_recall@10"], metrics["dup_mrr@10"],
        )

    return exp_results, all_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("  Milestone 5 — Global Visual Baseline (CLIP)")
    log.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # 1. Load corpus
    df = load_corpus()
    image_paths = df["resolved_path"].tolist()

    # 2. Load queries
    q1 = load_query_set(Q1_PATH)
    q2 = load_query_set(Q2_PATH)
    q3 = load_query_set(Q3_PATH)
    query_sets = {"Q1": q1, "Q2": q2, "Q3": q3}

    # 3. Load dup mapping
    hash_to_rows, row_to_hash = load_duplicate_mapping(DUP_MAP_PATH)

    # 4. Load model
    model, processor = load_model(device)

    # 5. Encode images
    img_embeddings = encode_images(model, processor, image_paths, device=device)

    # Save image embeddings
    os.makedirs(PROJECT_ROOT / "indexes", exist_ok=True)
    np.save(PROJECT_ROOT / "indexes" / "clip_vit_b32_image_embeddings.npy", img_embeddings)
    with open(PROJECT_ROOT / "indexes" / "clip_vit_b32_image_ids.json", "w") as fh:
        json.dump([i for i in range(len(image_paths))], fh)

    # 6. Build FAISS index
    index = build_faiss_index(img_embeddings)

    # 7. Run evaluation
    t0_eval = time.perf_counter()
    clip_results, all_predictions = run_clip_experiment(
        query_sets, model, processor, index, device, hash_to_rows, row_to_hash
    )
    eval_time = time.perf_counter() - t0_eval

    # 8. Save per-query predictions
    os.makedirs(PROJECT_ROOT / "reports", exist_ok=True)
    for qset_name, preds in all_predictions.items():
        out_path = PROJECT_ROOT / "reports" / f"m5_clip_predictions_{qset_name.lower()}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(preds, fh)

    # Save per-query ranks CSV
    ranks_data = []
    for qset_name, queries in query_sets.items():
        preds = all_predictions[qset_name]
        for idx, (query, pred) in enumerate(zip(queries, preds)):
            gt = query["ground_truth_row"]
            try:
                rank = pred.index(gt) + 1
            except ValueError:
                rank = -1
            ranks_data.append({
                "query_set": qset_name,
                "query_idx": idx,
                "ground_truth_row": gt,
                "rank": rank,
                "top_1_pred": pred[0] if pred else -1
            })
    pd.DataFrame(ranks_data).to_csv(PROJECT_ROOT / "reports" / "m5_clip_per_query_ranks.csv", index=False)

    # 9. Complementarity stub
    comp = {
        "note": "Complementarity calculation skipped or stubbed because per-query prediction JSONs for M2/M3 are not found. Saved CLIP predictions for future hybrid fusion.",
        "M3_dense_available": False,
        "M2_bm25_available": False
    }
    with open(PROJECT_ROOT / "reports" / "m5_clip_complementarity.json", "w") as fh:
        json.dump(comp, fh, indent=2)

    # 10. Assemble and save report
    report = {
        "milestone": "M5",
        "description": "Global Visual Baseline (CLIP)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "eval_time_sec": eval_time,
        "storage_size_mb": os.path.getsize(PROJECT_ROOT / "indexes" / "clip_vit_b32_image_embeddings.npy") / (1024*1024),
        "corpus": {
            "total_images": len(image_paths),
        },
        "experiments": {
            "CLIP_Visual": {
                "results": clip_results
            }
        }
    }

    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report saved → %s", REPORT_PATH)

    log.info("✅ Milestone 5 complete.")


if __name__ == "__main__":
    main()
