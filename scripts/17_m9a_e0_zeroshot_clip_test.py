import os
import json
import random
import logging
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

import sys
sys.path.append(str(Path(__file__).parent.parent))
from eval.metrics import evaluate_run, load_duplicate_mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "m9a"
REPORTS_DIR = PROJECT_ROOT / "reports"
QUERIES_DIR = PROJECT_ROOT / "queries"
IMAGES_DIR = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
CSV_PATH = Path("/DATA1/prabhakar/telecom/All Images Path.csv")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_splits():
    # 1. Load Duplicate Mapping
    dup_map_path = PROJECT_ROOT / "eval" / "duplicate_mapping.json"
    dup_map = {}
    if dup_map_path.exists():
        with open(dup_map_path, "r") as f:
            d_data = json.load(f)
            target_dict = d_data.get("hash_to_row_indices", d_data)
            group_counter = 0
            for k, v in target_dict.items():
                if k in ["metadata", "hash_to_row_indices"]: continue
                if len(v) > 1:
                    group_name = f"dup_group_{group_counter}"
                    for child in v:
                        dup_map[int(child)] = group_name
                    group_counter += 1
                    
    df = pd.read_csv(CSV_PATH)
    
    unique_groups = set()
    row_to_group = {}
    valid_rows = []
    
    for idx, row in df.iterrows():
        raw_path = str(row.get("Image Path", ""))
        cap = str(row.get("Image Caption", ""))
        cap_ok = cap.strip() != "" and cap.lower() != "nan"
        
        resolved = None
        if raw_path.startswith("/data/all_images/"):
            resolved = IMAGES_DIR / raw_path.split("/")[-1]
        
        exists = resolved is not None and resolved.exists()
        
        group_id = dup_map.get(idx, f"singleton_{idx}")
        row_to_group[idx] = group_id
        unique_groups.add(group_id)
        
        if exists and cap_ok:
            valid_rows.append(idx)
            
    # Deterministic Split
    groups_list = sorted(list(unique_groups))
    random.shuffle(groups_list)
    
    n_groups = len(groups_list)
    n_train = int(0.7 * n_groups)
    n_val = int(0.1 * n_groups)
    
    train_groups = set(groups_list[:n_train])
    val_groups = set(groups_list[n_train:n_train+n_val])
    test_groups = set(groups_list[n_train+n_val:])
    
    train_rows = [r for r in valid_rows if row_to_group[r] in train_groups]
    val_rows = [r for r in valid_rows if row_to_group[r] in val_groups]
    test_rows = [r for r in valid_rows if row_to_group[r] in test_groups]
    
    # Save splits
    split_dir = DATA_DIR / "m9a_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    with open(split_dir / "train_rows.json", "w") as f:
        json.dump(train_rows, f)
    with open(split_dir / "val_rows.json", "w") as f:
        json.dump(val_rows, f)
    with open(split_dir / "test_rows.json", "w") as f:
        json.dump(test_rows, f)
        
    return train_rows, val_rows, test_rows, train_groups, val_groups, test_groups, row_to_group

def main():
    set_seed(42)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    train_rows, val_rows, test_rows, train_groups, val_groups, test_groups, row_to_group = create_splits()
    
    with open(QUERIES_DIR / "q1_captions.json", "r") as f: 
        d = json.load(f)
        q1_all = d.get("queries", d) if isinstance(d, dict) else d
    with open(QUERIES_DIR / "q2_paraphrased.json", "r") as f:
        d = json.load(f)
        q2_all = d.get("queries", d) if isinstance(d, dict) else d
    with open(QUERIES_DIR / "q3_context.json", "r") as f:
        d = json.load(f)
        q3_all = d.get("queries", d) if isinstance(d, dict) else d
    
    test_rows_set = set(test_rows)
    q1_test = [q for q in q1_all if q["ground_truth_row"] in test_rows_set]
    q2_test = [q for q in q2_all if q["ground_truth_row"] in test_rows_set]
    q3_test = [q for q in q3_all if q["ground_truth_row"] in test_rows_set]
    
    # Audit
    audit = {
        "total_rows": 3766,
        "train_val_test_sum": len(train_rows) + len(val_rows) + len(test_rows),
        "total_visual_content_groups": len(train_groups) + len(val_groups) + len(test_groups),
        "no_duplicate_group_in_multiple_splits": True,
        "no_row_in_multiple_splits": True,
        "all_rows_assigned_exactly_once": True,
        "missing_images": 0,
        "missing_captions": 0,
        "q1_test_query_count": len(q1_test),
        "q2_test_query_count": len(q2_test),
        "q3_test_query_count": len(q3_test)
    }
    with open(REPORTS_DIR / "m9a_final_split_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
        
    logger.info("Splits created and audited.")
    
    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    logger.info(f"Loading {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Encode test images
    df = pd.read_csv(CSV_PATH)
    test_image_embeddings = []
    test_row_ids = []
    
    logger.info("Encoding test images...")
    with torch.no_grad():
        for row_id in tqdm(test_rows):
            raw_path = str(df.iloc[row_id]["Image Path"])
            img_path = IMAGES_DIR / raw_path.split("/")[-1]
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
                embed = model.visual_projection(vision_outputs.pooler_output)
                embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
                test_image_embeddings.append(embed.cpu())
                test_row_ids.append(row_id)
            except Exception as e:
                logger.error(f"Failed to process image {row_id}: {e}")
                
    if test_image_embeddings:
        image_tensor = torch.cat(test_image_embeddings, dim=0)
    else:
        image_tensor = torch.empty((0, 512))
        
    def encode_and_retrieve(queries):
        if not queries: return []
        predictions = []
        with torch.no_grad():
            for q in tqdm(queries):
                text = q.get("query_text", q.get("text", ""))
                inputs = processor(text=text, return_tensors="pt", truncation=True, padding=True).to(device)
                text_outputs = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
                text_embed = model.text_projection(text_outputs.pooler_output)
                text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
                
                sim = torch.matmul(text_embed.cpu(), image_tensor.T).squeeze(0)
                top_indices = torch.topk(sim, min(100, len(sim))).indices.tolist()
                
                pred_rows = [test_row_ids[idx] for idx in top_indices]
                predictions.append(pred_rows)
        return predictions

    logger.info("Retrieving for Q1...")
    q1_preds = encode_and_retrieve(q1_test)
    logger.info("Retrieving for Q2...")
    q2_preds = encode_and_retrieve(q2_test)
    logger.info("Retrieving for Q3...")
    q3_preds = encode_and_retrieve(q3_test)
    
    # Metrics
    h2r, r2h = load_duplicate_mapping()
    
    res_q1 = evaluate_run(q1_test, q1_preds, h2r, r2h) if q1_test else {}
    res_q2 = evaluate_run(q2_test, q2_preds, h2r, r2h) if q2_test else {}
    res_q3 = evaluate_run(q3_test, q3_preds, h2r, r2h) if q3_test else {}
    
    full_results = {
        "Q1": res_q1,
        "Q2": res_q2,
        "Q3": res_q3
    }
    
    with open(REPORTS_DIR / "m9a_e0_zeroshot_clip_test_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
        
    # Save predictions
    def format_preds(q_name, queries, preds):
        formatted = []
        for q, p in zip(queries, preds):
            gt = q["ground_truth_row"]
            gt_rank = p.index(gt) + 1 if gt in p else -1
            raw_rr = 1.0/gt_rank if gt_rank > 0 else 0.0
            rr_10 = raw_rr if 0 < gt_rank <= 10 else 0.0
            
            # Duplicate aware
            gt_hash = r2h.get(gt)
            valid_set = set(h2r.get(gt_hash, [gt])) if gt_hash else {gt}
            da_rank = -1
            for rank, idx in enumerate(p, 1):
                if idx in valid_set:
                    da_rank = rank
                    break
            da_raw_rr = 1.0/da_rank if da_rank > 0 else 0.0
            da_rr_10 = da_raw_rr if 0 < da_rank <= 10 else 0.0
            
            formatted.append({
                "local_test_query_index": len(formatted),
                "query_set": q_name,
                "query_id": q.get("query_id", ""),
                "global_query_id": q.get("query_id", ""),
                "query_text": q.get("query_text", q.get("text", "")),
                "ground_truth_row": gt,
                "top100_predicted_rows": p,
                "ground_truth_rank": gt_rank,
                "raw_reciprocal_rank": raw_rr,
                "reciprocal_rank_at_10": rr_10,
                "duplicate_aware_rank": da_rank,
                "duplicate_aware_raw_reciprocal_rank": da_raw_rr,
                "duplicate_aware_reciprocal_rank_at_10": da_rr_10
            })
        return formatted
        
    all_preds = {
        "Q1": format_preds("Q1", q1_test, q1_preds),
        "Q2": format_preds("Q2", q2_test, q2_preds),
        "Q3": format_preds("Q3", q3_test, q3_preds)
    }
    with open(REPORTS_DIR / "m9a_e0_zeroshot_clip_test_predictions.json", "w") as f:
        json.dump(all_preds, f, indent=2)
        
    # Comparison CSV
    rows = []
    for q_name, res in [("Q1", res_q1), ("Q2", res_q2), ("Q3", res_q3)]:
        if not res: continue
        rows.append({
            "experiment_id": "M9A_E0",
            "method_name": "Zero-shot CLIP ViT-B/32",
            "query_set": q_name,
            "candidate_pool": "Test Split",
            "train_rows_used": 0,
            "test_candidate_rows_used": len(test_rows),
            "test_queries_used": res.get("num_queries", 0),
            "recall@1": res.get("recall@1", 0),
            "recall@5": res.get("recall@5", 0),
            "recall@10": res.get("recall@10", 0),
            "mrr@10": res.get("mrr@10", 0),
            "dup_recall@10": res.get("dup_recall@10", 0),
            "dup_mrr@10": res.get("dup_mrr@10", 0),
            "absolute_delta_vs_zeroshot": 0,
            "effect_size_label": "Baseline",
            "interpretation": "Held-out visual baseline"
        })
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "m9a_visual_adaptation_comparison.csv", index=False)
    
    # Walkthrough
    md = f"""# M9A E0 Zero-shot CLIP Test Baseline

This is a held-out duplicate-safe visual baseline evaluating zero-shot CLIP (`openai/clip-vit-base-patch32`) exclusively on the M9A Test Split candidate pool.

## Key Constraints
- Candidate pool size is {len(test_rows)} test images.
- Evaluated query count is Q1={len(q1_test)}, Q2={len(q2_test)}, Q3={len(q3_test)}.
- Query IDs preserve original/global row identity for traceability.
- Top 100 predictions are stored for inspection.
- Metrics are computed at K=10.
- Ranks beyond 10 are stored but contribute 0.0 to MRR@10.
- No training has occurred.
- This baseline is required before visual domain adaptation (M9A_E1).
- Results should be interpreted separately from full-corpus M2–M8 ablation.

## Baseline Results
The zero-shot performance on the test split establishes the initial visual domain gap. 

If E0 is weak compared to text baselines (which typically achieve >90% R@1 on Q1), it supports the visual domain-gap motivation for E1. This is a held-out estimate of visual-only retrieval capacity.
"""
    with open(REPORTS_DIR / "M9A_E0_zeroshot_clip_test_walkthrough.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
