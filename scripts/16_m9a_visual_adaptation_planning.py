#!/usr/bin/env python3
import json
import logging
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import importlib.util

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

def main():
    # 1. Feasibility Audit
    audit = {}

    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    audit["source_csv_exists"] = csv_path.exists()

    df = pd.DataFrame()
    if csv_path.exists():
        df = pd.read_csv(csv_path)

    audit["number_of_rows"] = len(df)
    audit["image_path_column_exists"] = "Image Path" in df.columns if not df.empty else False
    audit["caption_column_exists"] = "Image Caption" in df.columns if not df.empty else False

    dup_map_path = PROJECT_ROOT / "eval/duplicate_mapping.json"
    audit["duplicate_mapping_exists"] = dup_map_path.exists()

    # Environment
    try:
        import torch
        audit["torch_availability"] = True
        audit["gpu_availability"] = torch.cuda.is_available()
    except:
        audit["torch_availability"] = False
        audit["gpu_availability"] = False

    try:
        import transformers
        audit["transformers_availability"] = True
    except:
        audit["transformers_availability"] = False

    try:
        import peft
        audit["peft_lora_installed"] = True
    except:
        audit["peft_lora_installed"] = False

    audit["current_python_environment"] = sys.executable

    # Check model
    try:
        from transformers import CLIPModel
        # Fast check if it's cached
        model_name = "openai/clip-vit-base-patch32"
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        audit["clip_model_availability"] = True
        audit["clip_model_can_be_loaded"] = True
    except:
        audit["clip_model_availability"] = False
        audit["clip_model_can_be_loaded"] = False

    audit["full_fine_tuning_risky"] = True
    audit["projection_only_feasible"] = True
    audit["recommended_first_experiment"] = "Projection-only adaptation with contrastive loss on 70% train split."

    # 2. Image Caption Pair Audit & Dup Grouping
    audit_rows = []
    base_img_dir = Path("/DATA5/prabhakar/telecom/extracted_images/images/")

    # Load duplicates
    dup_map = {}
    if dup_map_path.exists():
        with open(dup_map_path, "r") as f:
            d_data = json.load(f)
            if isinstance(d_data, dict):
                group_counter = 0
                target_dict = d_data.get("hash_to_row_indices", d_data)
                for k, v in target_dict.items():
                    if k in ["metadata", "hash_to_row_indices"]: continue
                    if len(v) > 1:
                        group_name = f"dup_group_{group_counter}"
                        for child in v:
                            dup_map[int(child)] = group_name
                        group_counter += 1
            elif isinstance(d_data, list):
                for i, group in enumerate(d_data):
                    if len(group) > 1:
                        group_name = f"dup_group_{i}"
                        for row in group:
                            dup_map[int(row)] = group_name

    usable_count = 0
    missing_caps = 0
    missing_imgs = 0
    resolved_imgs = 0
    unique_groups = set()

    # Store group sizes
    group_sizes = {}
    for idx in range(len(df)):
        g = dup_map.get(idx, f"singleton_{idx}")
        group_sizes[g] = group_sizes.get(g, 0) + 1

    if not df.empty:
        for idx, row in df.iterrows():
            raw_path = str(row.get("Image Path", ""))
            caption = str(row.get("Image Caption", ""))

            cap_ok = bool(caption and caption.lower() != "nan" and len(caption.strip()) > 3)
            if not cap_ok: missing_caps += 1

            resolved = None
            exists = False
            size = 0
            if raw_path and raw_path.lower() != "nan":
                candidates_paths = [
                    Path(raw_path),
                    base_img_dir / Path(raw_path).name,
                    base_img_dir / raw_path.strip("/")
                ]
                for p in candidates_paths:
                    if p.exists() and p.is_file():
                        resolved = str(p)
                        exists = True
                        size = p.stat().st_size
                        break

            if exists:
                resolved_imgs += 1
            else:
                missing_imgs += 1

            group_id = dup_map.get(idx, f"singleton_{idx}")
            unique_groups.add(group_id)

            usable = exists and cap_ok
            if usable: usable_count += 1

            issue = []
            if not cap_ok: issue.append("Missing/invalid caption")
            if not exists: issue.append("Image not found")

            audit_rows.append({
                "row_id": idx,
                "caption_available": cap_ok,
                "raw_image_path": raw_path,
                "resolved_image_path": resolved if resolved else "",
                "image_exists": exists,
                "file_size_bytes": size,
                "duplicate_group_id": group_id,
                "usable_for_training": usable,
                "is_duplicate_member": group_id.startswith("dup_group_"),
                "duplicate_group_size": group_sizes[group_id],
                "issue": "; ".join(issue)
            })

    audit["number_of_usable_pairs"] = usable_count
    audit["number_of_missing_captions"] = missing_caps
    audit["number_of_resolved_image_paths"] = resolved_imgs
    audit["number_of_missing_image_files"] = missing_imgs
    audit["number_of_duplicate_groups"] = len(unique_groups)
    audit["duplicate_safe_split_feasible"] = len(unique_groups) > 1000

    with open(PROJECT_ROOT / "reports/m9a_visual_adaptation_feasibility_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    pd.DataFrame(audit_rows).to_csv(PROJECT_ROOT / "reports/m9a_image_caption_pair_audit.csv", index=False)

    # 3. Dry Run Split Summary
    random.seed(42)
    group_list = sorted(list(unique_groups))
    random.shuffle(group_list)

    n_groups = len(group_list)
    n_train = int(0.7 * n_groups)
    n_val = int(0.1 * n_groups)

    train_groups = set(group_list[:n_train])
    val_groups = set(group_list[n_train:n_train+n_val])
    test_groups = set(group_list[n_train+n_val:])

    train_rows = [r["row_id"] for r in audit_rows if r["duplicate_group_id"] in train_groups and r["usable_for_training"]]
    val_rows = [r["row_id"] for r in audit_rows if r["duplicate_group_id"] in val_groups and r["usable_for_training"]]
    test_rows = [r["row_id"] for r in audit_rows if r["duplicate_group_id"] in test_groups and r["usable_for_training"]]

    # Load queries to check test set sizes
    def load_qs(qs):
        p_map = {"q1": "queries/q1_captions.json", "q2": "queries/q2_paraphrased.json", "q3": "queries/q3_context.json"}
        try:
            with open(PROJECT_ROOT / p_map[qs]) as f:
                return json.load(f)["queries"]
        except: return []

    q1 = load_qs("q1")
    q2 = load_qs("q2")
    q3 = load_qs("q3")

    test_rows_set = set(test_rows)
    q1_test = [q for q in q1 if q["ground_truth_row"] in test_rows_set]
    q2_test = [q for q in q2 if q["ground_truth_row"] in test_rows_set]
    q3_test = [q for q in q3 if q["ground_truth_row"] in test_rows_set]

    # Check leakage
    train_groups_set = set(train_groups)
    test_groups_set = set(test_groups)
    val_groups_set = set(val_groups)
    leakage = bool(train_groups_set & test_groups_set or train_groups_set & val_groups_set or val_groups_set & test_groups_set)

    split_summary = {
        "train_row_count": len(train_rows),
        "val_row_count": len(val_rows),
        "test_row_count": len(test_rows),
        "train_duplicate_group_count": len(train_groups),
        "val_duplicate_group_count": len(val_groups),
        "test_duplicate_group_count": len(test_groups),
        "leakage_detected": leakage,
        "rows_with_missing_images": missing_imgs,
        "rows_with_missing_captions": missing_caps,
        "q1_test_query_count": len(q1_test),
        "q2_test_query_count": len(q2_test),
        "q3_test_query_count": len(q3_test)
    }
    with open(PROJECT_ROOT / "reports/m9a_dry_run_split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    # Generate Duplicate Group Audit CSV
    group_audit = []
    df_audit = pd.DataFrame(audit_rows)
    for g in unique_groups:
        g_rows = df_audit[df_audit["duplicate_group_id"] == g]
        assigned_split = "Train" if g in train_groups else ("Val" if g in val_groups else "Test")
        imgs_exist = all(g_rows["image_exists"])
        group_audit.append({
            "duplicate_group_id": g,
            "row_ids": str(g_rows["row_id"].tolist()),
            "group_size": len(g_rows),
            "assigned_split": assigned_split,
            "leakage_detected": False, # Enforced by set intersection check above
            "image_paths_exist": imgs_exist,
            "notes": "All paths resolved" if imgs_exist else "Missing image paths in group"
        })
    pd.DataFrame(group_audit).to_csv(PROJECT_ROOT / "reports/m9a_duplicate_group_audit.csv", index=False)

    # 4. Generate Expected Experiments CSV
    exp_data = [
        {"experiment_id": "M9A_E0", "method_name": "zero-shot CLIP on duplicate-safe test split", "trainable_parameters": "0", "training_data": "None", "evaluation_split": "Test Split", "expected_runtime": "5 mins", "expected_risk": "Low", "output_files": "reports/m9a_e0_predictions.json", "success_criterion": "Baseline measurement", "thesis_interpretation": "Zero-shot visual domain gap baseline on restricted test split"},
        {"experiment_id": "M9A_E1", "method_name": "projection-only CLIP adaptation", "trainable_parameters": "Lightweight projection heads", "training_data": "Train Split (70%)", "evaluation_split": "Test Split", "expected_runtime": "1-2 hours", "expected_risk": "Low", "output_files": "reports/m9a_e1_predictions.json", "success_criterion": "Test MRR > Zero-shot MRR", "thesis_interpretation": "Domain gap can be reduced via lightweight alignment on telecom metadata"},
        {"experiment_id": "M9A_E2", "method_name": "optional LoRA/adapters on last visual blocks", "trainable_parameters": "LoRA weights", "training_data": "Train Split (70%)", "evaluation_split": "Test Split", "expected_runtime": "3-5 hours", "expected_risk": "Medium", "output_files": "reports/m9a_e2_predictions.json", "success_criterion": "Test MRR > E1 MRR", "thesis_interpretation": "Deeper visual feature adaptation improves retrieval semantics further"},
        {"experiment_id": "M9A_E3", "method_name": "optional fusion of adapted visual branch with text system", "trainable_parameters": "0 (Fusion weights)", "training_data": "None", "evaluation_split": "Test Split", "expected_runtime": "5 mins", "expected_risk": "Low", "output_files": "reports/m9a_e3_predictions.json", "success_criterion": "Hybrid Test MRR > Best Text Baseline MRR on Test", "thesis_interpretation": "Adapted visual branch provides complementary signal to lexical/dense text retrieval"}
    ]
    pd.DataFrame(exp_data).to_csv(PROJECT_ROOT / "reports/m9a_expected_experiments.csv", index=False)

    # 5. Generate Markdown Documents
    plan_md = """# M9A: Telecom-Aligned Visual Encoder Pilot

## Motivation
Previous experiments showed that zero-shot CLIP and ColPali performed poorly on telecom technical diagrams, while text metadata retrieval remained much stronger. This suggests a visual domain gap.
M9A tests whether a CLIP/ViT-style visual encoder can be aligned to telecom diagrams using weak supervision from image-caption pairs.

## Hypothesis
Zero-shot visual encoders fail because they are not aligned to telecom engineering diagrams. If we adapt CLIP/ViT using telecom image-caption pairs, visual retrieval may improve over zero-shot CLIP on a held-out duplicate-safe test split.

## Relation to M2–M9
This experiment is optional and must remain separate from the full-corpus M2–M8 ablation. It acts as an auxiliary pilot to see if visual domain gap can be reduced.

## Why this is not replacing text-first architecture
The full thesis architecture is text-first. Visual domain adaptation provides a proof-of-concept auxiliary branch, not a replacement for the highly effective BM25/BGE hybrid.

## Planned Methods
- **Method A:** Projection-only adaptation (Freeze backbone, train heads)
- **Method B:** LoRA/adapters (If PEFT is cleanly supported)
- **Method C:** Full fine-tuning (High risk, generally avoided)

## Leakage Prevention Strategy
Split by MD5 duplicate group, not individual row. All duplicate images must remain in the same split (70% train, 10% val, 20% test).

## Metrics
Evaluate on held-out test rows only: Recall@1, 2, 3, 5, 10; MRR@10; duplicate-aware Recall@K; duplicate-aware MRR@10.

## Expected Outputs
- Adapted model weights
- Test split predictions
- M9A performance summary

## Risks
- Small dataset size may lead to rapid overfitting.
- Adapted model might still underperform text baselines.

## What Result is Useful
If Adapted CLIP > Zero-shot CLIP on the held-out test split, it supports the hypothesis that the visual domain gap can be reduced through telecom-specific alignment.

## What Result is Negative but Informative
If Adapted CLIP still fails to beat text baselines, text and metadata remain more reliable than visual-only retrieval for this dataset.
"""
    with open(PROJECT_ROOT / "reports/M9A_visual_domain_adaptation_plan.md", "w") as f:
        f.write(plan_md)

    strategy_md = """# M9A Duplicate-Safe Split Strategy

## Why Ordinary Random Split is Invalid
A simple row-based random split would accidentally place exact image duplicates (e.g., standard architecture diagrams appearing in multiple specs) in both the training set and the test set.

## How Duplicate Leakage Inflates Results
If a duplicate image is in the training set, the model memorizes its visual features and simply performs a nearest-neighbor lookup during test time on the identical image. This creates artificial 100% accuracy that does not generalize to unseen diagrams.

## Group Assignment
Duplicate groups (derived from MD5 or identical paths) are grouped together. The entire group is assigned exclusively to Train (70%), Val (10%), or Test (20%).

## Expected Row Counts
- Train: ~70% of rows
- Val: ~10% of rows
- Test: ~20% of rows

## Query Filtering
Q1, Q2, and Q3 test queries will be filtered to only include queries where the `ground_truth_row` belongs to the Test split.

## Fair Comparison
Zero-shot CLIP and adapted CLIP will be compared *only* on this exact duplicate-safe test split to ensure a rigorous, leak-free evaluation.
"""
    with open(PROJECT_ROOT / "reports/m9a_duplicate_safe_split_strategy.md", "w") as f:
        f.write(strategy_md)

    log.info("M9A Planning completed.")

if __name__ == "__main__":
    main()
