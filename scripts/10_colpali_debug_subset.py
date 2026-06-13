import json
import torch
import random
import os
from pathlib import Path
import pandas as pd
from colpali_engine.models import ColPali, ColPaliProcessor
from tqdm import tqdm
import sys
sys.path.append("/DATA5/prabhakar/telecom_retrieval")
from eval.metrics import evaluate_run, load_duplicate_mapping

def generate_debug_examples():
    print("Generating debug examples...")
    root = Path("/DATA5/prabhakar/telecom_retrieval")
    index_dir = root / "indexes" / "colpali_index"

    with open(index_dir / "image_ids.json") as f:
        image_ids = json.load(f)

    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    df = pd.read_csv(csv_path)

    # map image id to gold caption
    id_to_caption = {}
    for idx, row in df.iterrows():
        img_id = row['Image Path'].split('/')[-1].split('.')[0]
        id_to_caption[img_id] = row['Image Caption'] if pd.notna(row['Image Caption']) else ""

    queries = {
        "q1": json.load(open(root / "queries/q1_captions.json"))["queries"],
        "q2": json.load(open(root / "queries/q2_paraphrased.json"))["queries"],
        "q3": json.load(open(root / "queries/q3_context.json"))["queries"]
    }

    out_md = []
    out_md.append("# ColPali Debug Examples\n")

    for q_type, q_list in queries.items():
        with open(root / f"reports/m6b_colpali_predictions_{q_type}.json") as f:
            preds = json.load(f)

        out_md.append(f"## {q_type.upper()} Examples\n")

        # sample 3 or 4 examples
        sample_size = 4 if q_type == "q1" else 3
        sample_q = random.sample(q_list, sample_size)

        for q in sample_q:
            q_id = q['query_id']
            text = q['text']
            gt_row = q['ground_truth_row']

            # Find the true gold image ID from gt_row
            gold_img_path = df.iloc[gt_row]['Image Path']
            gold_img_id = gold_img_path.split('/')[-1].split('.')[0]
            gold_caption = id_to_caption[gold_img_id]

            out_md.append(f"### Query: `{q_id}`")
            out_md.append(f"**Query Text**: {text}")
            out_md.append(f"**Gold Image ID**: {gold_img_id}")
            out_md.append(f"**Gold Caption**: {gold_caption}")

            pred_list = preds.get(q_id, [])[:10]

            out_md.append("\n**Top 10 Predictions**:")
            for i, p_id in enumerate(pred_list):
                p_cap = id_to_caption.get(p_id, "")
                match = "✅" if p_id == gold_img_id else "❌"
                out_md.append(f"{i+1}. {match} `{p_id}` - {p_cap}")

            is_in_top10 = gold_img_id in pred_list
            out_md.append(f"\n**Gold in Top-10?**: {is_in_top10}\n")

    with open(root / "reports/m6b_colpali_debug_examples.md", "w") as f:
        f.write("\n".join(out_md))

    print("Debug examples saved.")

def run_subset_sanity():
    print("Running subset sanity check...")
    root = Path("/DATA5/prabhakar/telecom_retrieval")
    index_dir = root / "indexes" / "colpali_index"

    with open(index_dir / "image_ids.json") as f:
        image_ids = json.load(f)

    # Take first 100 images
    subset_image_ids = image_ids[:100]

    # Load full embeddings
    image_embeddings = torch.load(index_dir / "image_embeddings.pt", weights_only=True)
    subset_embeddings = image_embeddings[:100]

    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    df = pd.read_csv(csv_path)

    # mapping from image_id to index in the subset
    img_id_to_subset_idx = {img_id: i for i, img_id in enumerate(subset_image_ids)}

    # mapping from img_id to true row
    img_id_to_row = {}
    for idx, row in df.iterrows():
        img_id = row['Image Path'].split('/')[-1].split('.')[0]
        img_id_to_row[img_id] = idx

    q1_list = json.load(open(root / "queries/q1_captions.json"))["queries"]

    # Filter q1: gold image must be in subset_image_ids
    subset_q = []
    subset_image_ids_set = set(subset_image_ids)
    for q in q1_list:
        gt_row = q['ground_truth_row']
        gold_img_path = df.iloc[gt_row]['Image Path']
        gold_img_id = gold_img_path.split('/')[-1].split('.')[0]
        if gold_img_id in subset_image_ids_set:
            subset_q.append(q)

    print(f"Subset contains {len(subset_q)} queries matching the 100 images.")

    device = "cuda:0"
    subset_embeddings = subset_embeddings.to(device)

    model_name = "vidore/colpali-v1.2"
    dtype = subset_embeddings.dtype
    model = ColPali.from_pretrained(model_name, torch_dtype=dtype, device_map=device).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    h2r, r2h = load_duplicate_mapping()

    all_preds_for_metric = []
    eval_queries_for_metric = []

    batch_size = 8
    for i in tqdm(range(0, len(subset_q), batch_size), desc="Scoring Subset"):
        batch_q = subset_q[i:i+batch_size]
        query_texts = [q['text'] for q in batch_q]

        with torch.no_grad():
            batch_dict = processor.process_queries(query_texts).to(device)
            q_emb = model(**batch_dict)
            scores = processor.score(q_emb, subset_embeddings)

        topk_scores, topk_indices = torch.topk(scores, k=min(100, scores.size(1)), dim=1)
        topk_indices = topk_indices.cpu().numpy()

        for j, q in enumerate(batch_q):
            retrieved_rows = []
            for rank in range(topk_indices.shape[1]):
                img_idx = topk_indices[j, rank]
                img_id = subset_image_ids[img_idx]
                retrieved_rows.append(img_id_to_row[img_id])

            all_preds_for_metric.append(retrieved_rows)
            eval_queries_for_metric.append(q)

    metrics = evaluate_run(eval_queries_for_metric, all_preds_for_metric, h2r, r2h)

    with open(root / "reports/m6b_colpali_subset_sanity.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Subset sanity completed.")

if __name__ == "__main__":
    generate_debug_examples()
    run_subset_sanity()
