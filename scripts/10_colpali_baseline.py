"""
scripts/10_colpali_baseline.py
Executes M6b ColPali image indexing and query evaluation.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from colpali_engine.models import ColPali, ColPaliProcessor
import sys
sys.path.append("/DATA5/prabhakar/telecom_retrieval")
from eval.metrics import load_duplicate_mapping, evaluate_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reports/m6b_colpali_run.log", mode="a"),
        logging.StreamHandler()
    ]
)

def build_index(args):
    logging.info("Starting Phase 1: Image Indexing")

    root = Path("/DATA5/prabhakar/telecom_retrieval")
    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    images_dir = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
    index_dir = root / "indexes" / "colpali_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # The image path is like /DATA1/prabhakar/telecom/extracted_images/images/xyz.png
    # But we map it to our actual folder
    image_ids = []
    image_paths = []

    for idx, row in df.iterrows():
        img_name = row['Image Path'].split('/')[-1]
        img_id = img_name.split('.')[0]
        full_path = images_dir / img_name
        if full_path.exists():
            image_ids.append(img_id)
            image_paths.append(str(full_path))
        else:
            logging.warning(f"Image not found: {full_path}")

    logging.info(f"Found {len(image_paths)} images to index.")

    model_name = "vidore/colpali-v1.2"
    device = "cuda:0"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logging.info(f"Loading model {model_name}...")
    model = ColPali.from_pretrained(model_name, torch_dtype=dtype, device_map=device).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    batch_size = 16
    all_embeddings = []
    failed_images = []

    t0 = time.time()
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Indexing Images"):
        batch_paths = image_paths[i:i+batch_size]
        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.no_grad():
                batch_dict = processor.process_images(images).to(device)
                emb = model(**batch_dict)
                all_embeddings.append(emb.cpu())
        except Exception as e:
            logging.error(f"Failed batch at index {i}: {e}")
            for p in batch_paths:
                failed_images.append(p)

    total_time = time.time() - t0
    logging.info(f"Indexing completed in {total_time:.2f}s.")

    # Concatenate and save
    if all_embeddings:
        full_tensor = torch.cat(all_embeddings, dim=0)
        logging.info(f"Full index shape: {full_tensor.shape}")
        torch.save(full_tensor, index_dir / "image_embeddings.pt")

        with open(index_dir / "image_ids.json", "w") as f:
            json.dump(image_ids, f)

        index_size_mb = (full_tensor.element_size() * full_tensor.nelement()) / (1024**2)
        logging.info(f"Index size on disk: {index_size_mb:.2f} MB")

    if failed_images:
        logging.error(f"Failed images: {len(failed_images)}")
        with open(index_dir / "failed_images.json", "w") as f:
            json.dump(failed_images, f)

    logging.info("Phase 1 Complete.")

def evaluate(args):
    phase = "Sanity Check" if args.sanity else "Full Evaluation"
    logging.info(f"Starting Phase: {phase}")

    root = Path("/DATA5/prabhakar/telecom_retrieval")
    index_dir = root / "indexes" / "colpali_index"

    logging.info("Loading index...")
    try:
        image_embeddings = torch.load(index_dir / "image_embeddings.pt", weights_only=True)
        with open(index_dir / "image_ids.json", "r") as f:
            image_ids = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load index: {e}")
        return

    logging.info(f"Loaded index of shape {image_embeddings.shape}")

    device = "cuda:0"
    dtype = image_embeddings.dtype
    # Move index to GPU for fast scoring
    image_embeddings = image_embeddings.to(device)

    model_name = "vidore/colpali-v1.2"
    logging.info(f"Loading model {model_name}...")
    model = ColPali.from_pretrained(model_name, torch_dtype=dtype, device_map=device).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    h2r, r2h = load_duplicate_mapping()

    queries_data = {
        "q1": json.load(open(root / "queries/q1_captions.json"))["queries"],
        "q2": json.load(open(root / "queries/q2_paraphrased.json"))["queries"],
        "q3": json.load(open(root / "queries/q3_context.json"))["queries"]
    }

    # Create mapping from image_id string to the row index (ground truth row)
    # The image_id is something like "img_001", but the query dict uses "image_id" as well,
    # and the original metrics uses integer indices. Wait, let's load the CSV to map image_id to index.
    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    df = pd.read_csv(csv_path)
    img_id_to_row = {}
    for idx, row in df.iterrows():
        img_id = row['Image Path'].split('/')[-1].split('.')[0]
        img_id_to_row[img_id] = idx

    results = {}
    batch_size = 8

    for q_type, q_list in queries_data.items():
        if args.sanity:
            q_list = q_list[:100]

        logging.info(f"Evaluating {q_type} ({len(q_list)} queries)...")

        predictions_dump = {}
        all_preds_for_metric = []
        eval_queries_for_metric = []

        t0 = time.time()
        for i in tqdm(range(0, len(q_list), batch_size), desc=f"Scoring {q_type}"):
            batch_q = q_list[i:i+batch_size]
            query_texts = [q['text'] for q in batch_q]

            with torch.no_grad():
                batch_dict = processor.process_queries(query_texts).to(device)
                q_emb = model(**batch_dict)
                scores = processor.score(q_emb, image_embeddings)

            topk_scores, topk_indices = torch.topk(scores, k=min(100, scores.size(1)), dim=1)

            topk_scores = topk_scores.cpu().numpy()
            topk_indices = topk_indices.cpu().numpy()

            for j, q in enumerate(batch_q):
                q_id = q['query_id']
                gt_row = q['ground_truth_row']

                retrieved_img_ids = []
                retrieved_rows = []
                for rank in range(topk_indices.shape[1]):
                    img_idx = topk_indices[j, rank] # This is the index in our image_embeddings tensor
                    img_id = image_ids[img_idx]     # This is the string image_id
                    retrieved_img_ids.append(img_id)
                    retrieved_rows.append(img_id_to_row[img_id])

                predictions_dump[q_id] = retrieved_img_ids
                all_preds_for_metric.append(retrieved_rows)
                eval_queries_for_metric.append(q)

        logging.info(f"{q_type} scoring took {time.time()-t0:.2f}s")

        if not args.sanity:
            with open(root / f"reports/m6b_colpali_predictions_{q_type}.json", "w") as f:
                json.dump(predictions_dump, f, indent=2)

        metrics = evaluate_run(eval_queries_for_metric, all_preds_for_metric, h2r, r2h)
        results[q_type] = metrics

        logging.info(f"{q_type} R@10: {metrics['recall@10']:.4f} | Dup-R@10: {metrics['dup_recall@10']:.4f}")

    if args.sanity:
        with open(root / "reports/m6b_colpali_sanity_results.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        with open(root / "reports/m6b_colpali_results.json", "w") as f:
            json.dump(results, f, indent=2)

    logging.info(f"Phase {'Sanity' if args.sanity else 'Full'} Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True, choices=["index", "eval_sanity", "eval_full"])
    args = parser.parse_args()

    if args.phase == "index":
        build_index(args)
    elif args.phase == "eval_sanity":
        args.sanity = True
        evaluate(args)
    elif args.phase == "eval_full":
        args.sanity = False
        evaluate(args)
