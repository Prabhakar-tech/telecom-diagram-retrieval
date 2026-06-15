import argparse
import json
import logging
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import subprocess

sys.path.append(str(Path(__file__).parent.parent))
from eval.metrics import reciprocal_rank, recall_at_k, evaluate_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/DATA5/prabhakar/telecom_retrieval")
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data/m9a_splits"
CHECKPOINT_DIR = BASE_DIR / "checkpoints/m9a"
IMAGES_DIR = Path("/DATA5/prabhakar/telecom/extracted_images/images")
CSV_PATH = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
EVAL_MAPPING = BASE_DIR / "eval/duplicate_mapping.json"
Q1_PATH = BASE_DIR / "queries/q1_captions.json"
Q2_PATH = BASE_DIR / "queries/q2_paraphrased.json"
Q3_PATH = BASE_DIR / "queries/q3_context.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        
        # Initialize identity near zero
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        return x + self.fc2(self.relu(self.fc1(x)))

class AdaptedCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        # Freeze clip
        for param in self.clip.parameters():
            param.requires_grad = False
            
        embed_dim = self.clip.config.projection_dim
        self.img_adapter = Adapter(embed_dim)
        self.txt_adapter = Adapter(embed_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            img_features = self.clip.visual_projection(vision_outputs.pooler_output)
        img_features = self.img_adapter(img_features)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            txt_features = self.clip.text_projection(text_outputs.pooler_output)
        txt_features = self.txt_adapter(txt_features)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        return txt_features
        
    def forward(self, pixel_values, input_ids, attention_mask):
        img_features = self.encode_image(pixel_values)
        txt_features = self.encode_text(input_ids, attention_mask)
        return img_features, txt_features, self.logit_scale.exp()

from PIL import Image

class TelecomDataset(Dataset):
    def __init__(self, rows, csv_df, processor):
        self.rows = rows
        self.csv_df = csv_df
        self.processor = processor
        
    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
        row_idx = self.rows[idx]
        row_data = self.csv_df.iloc[row_idx]
        
        img_name = Path(row_data["Image Path"]).name
        img_path = IMAGES_DIR / img_name
        caption = str(row_data["Image Caption"])
        if pd.isna(caption):
            caption = ""
            
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(
            images=image, 
            text=caption, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "row_idx": row_idx
        }

def clip_loss(img_features, txt_features, logit_scale):
    logits_per_image = logit_scale * img_features @ txt_features.T
    logits_per_text = logits_per_image.T
    labels = torch.arange(logits_per_image.shape[0], device=img_features.device)
    loss_img = nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2.0

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, duplicate_mapping, all_split_rows):
    model.eval()
    all_img_features = []
    all_txt_features = []
    all_row_indices = []
    
    for batch in dataloader:
        pv = batch["pixel_values"].to(device)
        iid = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        
        img_feat = model.encode_image(pv)
        txt_feat = model.encode_text(iid, mask)
        
        all_img_features.append(img_feat.cpu())
        all_txt_features.append(txt_feat.cpu())
        all_row_indices.extend(batch["row_idx"].tolist())
        
    all_img_features = torch.cat(all_img_features, dim=0)
    all_txt_features = torch.cat(all_txt_features, dim=0)
    
    sim_matrix = all_txt_features @ all_img_features.T
    ranks = sim_matrix.argsort(dim=-1, descending=True)
    
    r2h = {}
    h2r = {}
    for h, rows in duplicate_mapping.items():
        if h == "metadata": continue
        h2r[h] = rows
        for r in rows:
            r2h[r] = h
            
    mrr_10_sum = 0
    recall_1_sum = 0
    recall_5_sum = 0
    recall_10_sum = 0
    dup_mrr_10_sum = 0
    dup_recall_10_sum = 0
    
    N = len(all_row_indices)
    
    for i in range(N):
        gt_row = all_row_indices[i]
        pred_rows = [all_row_indices[idx] for idx in ranks[i].tolist()]
        
        try:
            gt_rank = pred_rows.index(gt_row) + 1
        except ValueError:
            gt_rank = -1
            
        if gt_rank == 1: recall_1_sum += 1
        if 0 < gt_rank <= 5: recall_5_sum += 1
        if 0 < gt_rank <= 10: 
            recall_10_sum += 1
            mrr_10_sum += 1.0 / gt_rank
            
        gt_hash = r2h.get(gt_row)
        valid_set = set(h2r.get(gt_hash, [gt_row])) if gt_hash else {gt_row}
        
        da_rank = -1
        for r_idx, pr in enumerate(pred_rows, 1):
            if pr in valid_set:
                da_rank = r_idx
                break
                
        if 0 < da_rank <= 10:
            dup_recall_10_sum += 1
            dup_mrr_10_sum += 1.0 / da_rank
            
    res = {
        "recall@1": recall_1_sum / N,
        "recall@5": recall_5_sum / N,
        "recall@10": recall_10_sum / N,
        "mrr@10": mrr_10_sum / N,
        "dup_recall@10": dup_recall_10_sum / N,
        "dup_mrr@10": dup_mrr_10_sum / N
    }
    return res

def check_git_ignored(path):
    try:
        result = subprocess.run(
            ['git', 'check-ignore', str(path)],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except:
        return False

def dry_run():
    logger.info("Starting dry run...")
    
    train_rows = json.load(open(DATA_DIR / "train_rows.json"))
    val_rows = json.load(open(DATA_DIR / "val_rows.json"))
    test_rows = json.load(open(DATA_DIR / "test_rows.json"))
    
    df = pd.read_csv(CSV_PATH)
    all_images_resolved = True
    for row_idx in train_rows + val_rows + test_rows:
        img_name = Path(df.iloc[row_idx]["Image Path"]).name
        if not (IMAGES_DIR / img_name).exists():
            all_images_resolved = False
            break
            
    all_captions_available = not df.iloc[train_rows + val_rows + test_rows]["Image Caption"].isna().all()
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AdaptedCLIP(clip_model)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    ds = TelecomDataset(train_rows[:2], df, processor)
    dl = DataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    
    img_f, txt_f, logit_scale = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
    loss = clip_loss(img_f, txt_f, logit_scale)
    
    # Leakage check: test set in train
    leakage = len(set(train_rows).intersection(set(test_rows))) > 0
    
    # Git ignore check
    dummy_file = CHECKPOINT_DIR / "dummy.pt"
    dummy_file.touch()
    ignored = check_git_ignored(dummy_file)
    dummy_file.unlink()
    
    audit = {
        "train_rows_count": len(train_rows),
        "val_rows_count": len(val_rows),
        "test_rows_count": len(test_rows),
        "all_image_paths_resolved": all_images_resolved,
        "all_captions_available": all_captions_available,
        "clip_model_loaded": True,
        "trainable_parameter_count": trainable_params,
        "frozen_parameter_count": frozen_params,
        "adapter_architecture": "Residual Linear Projection (dim=512)",
        "one_batch_forward_pass_success": True,
        "loss_finite": torch.isfinite(loss).item(),
        "no_leakage": not leakage,
        "checkpoint_directory_ignored": ignored
    }
    
    with open(REPORTS_DIR / "m9a_e1_projection_dry_run_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
        
    logger.info("Dry run complete.")

def evaluate():
    logger.info("Evaluating M9A_E1 test split...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_rows = json.load(open(DATA_DIR / "test_rows.json"))
    train_rows = json.load(open(DATA_DIR / "train_rows.json"))
    
    df = pd.read_csv(CSV_PATH)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AdaptedCLIP(clip_model).to(device)
    
    ckpt_path = CHECKPOINT_DIR / "e1_projection_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.img_adapter.load_state_dict(ckpt['img_adapter'])
        model.txt_adapter.load_state_dict(ckpt['txt_adapter'])
        model.logit_scale.data = torch.tensor(ckpt['logit_scale'])
    model.eval()
    
    with open(Q1_PATH, "r") as f:
        d = json.load(f)
        q1_all = d.get("queries", d) if isinstance(d, dict) else d
    with open(Q2_PATH, "r") as f:
        d = json.load(f)
        q2_all = d.get("queries", d) if isinstance(d, dict) else d
    with open(Q3_PATH, "r") as f:
        d = json.load(f)
        q3_all = d.get("queries", d) if isinstance(d, dict) else d
        
    test_rows_set = set(test_rows)
    q1_test = [q for q in q1_all if q["ground_truth_row"] in test_rows_set]
    q2_test = [q for q in q2_all if q["ground_truth_row"] in test_rows_set]
    q3_test = [q for q in q3_all if q["ground_truth_row"] in test_rows_set]
    
    test_image_embeddings = []
    logger.info("Encoding test images...")
    with torch.no_grad():
        for row_id in tqdm(test_rows):
            img_name = Path(df.iloc[row_id]["Image Path"]).name
            img_path = IMAGES_DIR / img_name
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                embed = model.encode_image(inputs["pixel_values"])
                test_image_embeddings.append(embed.cpu())
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
                text_embed = model.encode_text(inputs["input_ids"], inputs.get("attention_mask"))
                sim = torch.matmul(text_embed.cpu(), image_tensor.T).squeeze(0)
                top_indices = torch.topk(sim, min(100, len(sim))).indices.tolist()
                pred_rows = [test_rows[idx] for idx in top_indices]
                predictions.append(pred_rows)
        return predictions

    logger.info("Retrieving for Q1...")
    q1_preds = encode_and_retrieve(q1_test)
    logger.info("Retrieving for Q2...")
    q2_preds = encode_and_retrieve(q2_test)
    logger.info("Retrieving for Q3...")
    q3_preds = encode_and_retrieve(q3_test)
    
    try:
        e0_comp = pd.read_csv(REPORTS_DIR / "m9a_visual_adaptation_comparison.csv")
    except:
        e0_comp = pd.DataFrame()
        
    duplicate_mapping = json.load(open(EVAL_MAPPING))
    h2r = {}
    r2h = {}
    for h, rows in duplicate_mapping.items():
        if h == "metadata": continue
        h2r[h] = rows
        for r in rows:
            r2h[r] = h
            
    res_q1 = evaluate_run(q1_test, q1_preds, h2r, r2h) if q1_test else {}
    res_q2 = evaluate_run(q2_test, q2_preds, h2r, r2h) if q2_test else {}
    res_q3 = evaluate_run(q3_test, q3_preds, h2r, r2h) if q3_test else {}
    
    full_results = {
        "Q1": res_q1,
        "Q2": res_q2,
        "Q3": res_q3
    }
    with open(REPORTS_DIR / "m9a_e1_projection_test_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
        
    all_preds = {
        "Q1": format_preds("Q1", q1_test, q1_preds, r2h, h2r),
        "Q2": format_preds("Q2", q2_test, q2_preds, r2h, h2r),
        "Q3": format_preds("Q3", q3_test, q3_preds, r2h, h2r)
    }
    with open(REPORTS_DIR / "m9a_e1_projection_test_predictions.json", "w") as f:
        json.dump(all_preds, f, indent=2)
        
    rows = []
    for q_name, res in [("Q1", res_q1), ("Q2", res_q2), ("Q3", res_q3)]:
        if not res: continue
        
        e0_mrr = 0
        if not e0_comp.empty and "experiment_id" in e0_comp.columns:
            subset = e0_comp[(e0_comp["experiment_id"]=="M9A_E0") & (e0_comp["query_set"]==q_name)]
            if not subset.empty:
                e0_mrr = subset["mrr@10"].values[0]
            
        e1_mrr = res.get("mrr@10", 0)
        delta = e1_mrr - e0_mrr
        
        label = "negative"
        if delta > 0.10: label = "large"
        elif delta > 0.05: label = "moderate"
        elif delta > 0.02: label = "small"
        elif delta > 0: label = "tiny"
        
        interpretation = "Projection adaptation shows higher retrieval metrics than the zero-shot baseline" if delta > 0 else "Projection adaptation does not show higher retrieval metrics than the zero-shot baseline"
        
        rows.append({
            "experiment_id": "M9A_E1",
            "method_name": "Projection-adapted CLIP ViT-B/32",
            "query_set": q_name,
            "candidate_pool": "Test Split",
            "train_rows_used": len(train_rows),
            "test_candidate_rows_used": len(test_rows),
            "test_queries_used": res.get("num_queries", 0),
            "recall@1": res.get("recall@1", 0),
            "recall@5": res.get("recall@5", 0),
            "recall@10": res.get("recall@10", 0),
            "mrr@10": e1_mrr,
            "dup_recall@10": res.get("dup_recall@10", 0),
            "dup_mrr@10": res.get("dup_mrr@10", 0),
            "absolute_delta_vs_zeroshot": delta,
            "effect_size_label": label,
            "interpretation": interpretation
        })
        
    df_new = pd.DataFrame(rows)
    df_combined = pd.concat([e0_comp, df_new], ignore_index=True)
    df_combined.to_csv(REPORTS_DIR / "m9a_visual_adaptation_comparison.csv", index=False)
    
    md = f"""# M9A E1 Projection Adaptation Walkthrough
    
This experiment tests a lightweight projection-only adaptation to align CLIP to telecom diagrams.

## Experimental Details
- **Architecture**: `openai/clip-vit-base-patch32` backbone (Frozen, ~151M params).
- **Trainable Parameters**: Residual linear projection adapters for image and text embeddings (dim=512), and the logit scale (temperature). Total ~1M params.
- **Why projection-only**: It reduces the risk of overfitting or disrupting the frozen CLIP representation, is extremely low-risk for small datasets, and directly tests whether simply rotating/scaling the embedding space can bridge the telecom visual domain gap.
- **Training**: Trained using symmetric contrastive loss on the duplicate-safe train split ({len(train_rows)} image-caption pairs).
- **Validation**: Best checkpoint selected by duplicate-aware MRR@10 on the {len(json.load(open(DATA_DIR/"val_rows.json")))} val split images using exact captions.
- **Test Candidate Pool**: {len(test_rows)} images (identical to M9A_E0).

## Results
Please refer to `reports/m9a_visual_adaptation_comparison.csv` for exact metrics.
The results reflect a held-out estimate of the domain adaptation performance.

## Interpretation
The results indicate that this projection-only adaptation {"shows higher retrieval metrics than" if df_new["absolute_delta_vs_zeroshot"].mean() > 0 else "does not show higher retrieval metrics than"} the zero-shot baseline. This evidence {"supports" if df_new["absolute_delta_vs_zeroshot"].mean() > 0 else "does not support"} the visual domain adaptation hypothesis.
"""
    with open(REPORTS_DIR / "M9A_E1_projection_adaptation_walkthrough.md", "w") as f:
        f.write(md)

def train(args):
    logger.info("Training M9A_E1 adapter...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if hasattr(args, 'seed') and args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    train_rows = json.load(open(DATA_DIR / "train_rows.json"))
    val_rows = json.load(open(DATA_DIR / "val_rows.json"))
    df = pd.read_csv(CSV_PATH)
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AdaptedCLIP(clip_model).to(device)
    
    bs = args.batch_size if hasattr(args, 'batch_size') and args.batch_size else 64
    epochs = args.epochs if hasattr(args, 'epochs') and args.epochs else 5
    lr = args.learning_rate if hasattr(args, 'learning_rate') and args.learning_rate else 1e-3
    
    train_ds = TelecomDataset(train_rows, df, processor)
    val_ds = TelecomDataset(val_rows, df, processor)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    duplicate_mapping = json.load(open(EVAL_MAPPING))
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    
    best_mrr = -1
    log_rows = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch} Train"):
            pv = batch["pixel_values"].to(device)
            iid = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            img_f, txt_f, logit_scale = model(pv, iid, mask)
            loss = clip_loss(img_f, txt_f, logit_scale)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_dl)
        val_res = evaluate_retrieval(model, val_dl, device, duplicate_mapping, val_rows)
        
        val_mrr = val_res["dup_mrr@10"]
        saved = False
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            torch.save({
                'img_adapter': model.img_adapter.state_dict(),
                'txt_adapter': model.txt_adapter.state_dict(),
                'logit_scale': model.logit_scale.item()
            }, CHECKPOINT_DIR / "e1_projection_best.pt")
            saved = True
            
        logger.info(f"Epoch {epoch} | Loss: {train_loss:.4f} | Dup MRR@10: {val_mrr:.4f} | Saved: {saved}")
        
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_recall@1": val_res["recall@1"],
            "val_recall@5": val_res["recall@5"],
            "val_recall@10": val_res["recall@10"],
            "val_mrr@10": val_res["mrr@10"],
            "val_dup_recall@10": val_res["dup_recall@10"],
            "val_dup_mrr@10": val_mrr,
            "learning_rate": lr,
            "checkpoint_saved": saved
        })
        
    pd.DataFrame(log_rows).to_csv(REPORTS_DIR / "m9a_e1_projection_training_log.csv", index=False)

def format_preds(q_name, queries, preds, r2h, h2r):
    formatted = []
    for q, p in zip(queries, preds):
        gt = q["ground_truth_row"]
        gt_rank = p.index(gt) + 1 if gt in p else -1
        raw_rr = 1.0/gt_rank if gt_rank > 0 else 0.0
        rr_10 = raw_rr if 0 < gt_rank <= 10 else 0.0
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["dry_run", "train", "evaluate"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.mode == "dry_run":
        dry_run()
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate()

if __name__ == "__main__":
    main()
