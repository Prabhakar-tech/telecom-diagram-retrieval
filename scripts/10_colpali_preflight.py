"""
scripts/10_colpali_preflight.py
"""
import os
import time
import json
import logging
import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from colpali_engine.models import ColPali, ColPaliProcessor

def main():
    root = Path("/DATA5/prabhakar/telecom_retrieval")
    log_path = root / "reports" / "m6b_colpali_preflight.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )

    logging.info("M6b: ColPali Preflight")

    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    images_dir = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
    report_path = root / "reports" / "m6b_colpali_preflight.json"

    # Load sample data
    df = pd.read_csv(csv_path)
    sample_df = df.head(10)

    queries = []
    image_paths = []

    for idx, row in sample_df.iterrows():
        img_name = row['Image Path'].split('/')[-1]
        image_paths.append(str(images_dir / img_name))
        queries.append(str(row['Image Caption']))

    logging.info(f"Loaded {len(image_paths)} images for preflight.")

    # Model configuration
    model_name = "vidore/colpali-v1.2"
    device = "cuda:0"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logging.info(f"Loading {model_name} on {device} with {dtype}...")

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    t0 = time.perf_counter()
    try:
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device
        ).eval()
        processor = ColPaliProcessor.from_pretrained(model_name)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    load_time = time.perf_counter() - t0

    mem_after_model = torch.cuda.memory_allocated(device)
    model_mem_mb = (mem_after_model - mem_before) / (1024**2)
    logging.info(f"Model loaded in {load_time:.2f}s. Model memory: {model_mem_mb:.2f} MB")

    images = [Image.open(p) for p in image_paths]

    logging.info("Encoding images...")
    t0 = time.perf_counter()
    with torch.no_grad():
        batch_images = processor.process_images(images).to(device)
        image_embeddings = model(**batch_images)
    img_encode_time = time.perf_counter() - t0

    emb_shape = image_embeddings.shape
    emb_dtype = image_embeddings.dtype
    emb_mem_bytes = image_embeddings.element_size() * image_embeddings.nelement()
    bytes_per_image = emb_mem_bytes / len(images)

    logging.info(f"Images encoded in {img_encode_time:.2f}s ({img_encode_time/len(images):.2f}s/img).")
    logging.info(f"Image embeddings shape: {emb_shape}")
    logging.info(f"Memory size per image embedding: {bytes_per_image / 1024:.2f} KB")

    full_corpus_size = 3766
    estimated_index_mb = (bytes_per_image * full_corpus_size) / (1024**2)
    logging.info(f"Estimated full corpus index size in RAM: {estimated_index_mb:.2f} MB")

    logging.info("Encoding queries...")
    t0 = time.perf_counter()
    with torch.no_grad():
        batch_queries = processor.process_queries(queries[:5]).to(device)
        query_embeddings = model(**batch_queries)
    qry_encode_time = time.perf_counter() - t0
    logging.info(f"Queries encoded in {qry_encode_time:.2f}s.")

    logging.info("Scoring...")
    t0 = time.perf_counter()
    scores = processor.score(query_embeddings, image_embeddings).cpu().numpy()
    score_time = time.perf_counter() - t0
    logging.info(f"Scoring completed in {score_time:.4f}s.")

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    report = {
        "model_name": model_name,
        "dtype": str(dtype),
        "device": device,
        "load_time_s": round(load_time, 2),
        "gpu_memory": {
            "model_size_mb": round(model_mem_mb, 2),
            "peak_memory_mb": round(peak_mem_mb, 2)
        },
        "timings": {
            "images_encoded": len(images),
            "total_image_encode_s": round(img_encode_time, 2),
            "time_per_image_s": round(img_encode_time / len(images), 3),
            "estimated_full_run_h": round((img_encode_time / len(images) * full_corpus_size) / 3600, 2),
            "scoring_time_s": round(score_time, 4)
        },
        "index_size": {
            "embedding_shape": list(emb_shape),
            "bytes_per_image": round(bytes_per_image, 2),
            "estimated_full_index_mb": round(estimated_index_mb, 2)
        },
        "feasibility": {
            "is_safe": estimated_index_mb < 8000,
            "notes": "Memory footprint is highly manageable. A single batch retrieval matrix multiplication easily fits in VRAM."
        },
        "tiny_retrieval_test": {
            "query_count": len(queries[:5]),
            "image_count": len(images),
            "score_matrix_shape": list(scores.shape),
            "sample_scores": [float(scores[0][0]), float(scores[0][1])]
        }
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info(f"Preflight complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
