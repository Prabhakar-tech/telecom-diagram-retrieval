import json
import os
import sys
from pathlib import Path
import pandas as pd
import argparse
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
IMAGES_DIR = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
CSV_PATH = Path("/DATA1/prabhakar/telecom/All Images Path.csv")

# We will just import functions from script 20 if possible, but let's redefine minimal needed to be standalone
import importlib.util
spec = importlib.util.spec_from_file_location("demo", PROJECT_ROOT / "scripts/20_m12_interactive_retrieval_demo.py")
demo = importlib.util.module_from_spec(spec)
sys.modules["demo"] = demo
spec.loader.exec_module(demo)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cs_dir = out_dir / "m12_50_query_contact_sheets"
    cs_dir.mkdir(parents=True, exist_ok=True)

    df_val = pd.read_csv(args.query_set)
    systems = ["final", "BM25", "BGE"]

    all_json = {}
    all_csv = []

    missing_image_count = 0
    sheets_generated = 0

    for _, row in df_val.iterrows():
        q_id = row["query_id"]
        q_text = row["query_text"]
        q_type = row["intended_query_type"]

        all_json[q_id] = {
            "query_text": q_text,
            "query_type": q_type,
            "results": {}
        }

        predictions_for_sheet = {}

        for sys_name in systems:
            preds = demo.get_prediction(sys_name, q_type, q_id)
            preds = preds[:args.top_k]
            parsed_preds = [demo.parse_row_id(p) for p in preds]
            predictions_for_sheet[sys_name] = parsed_preds

            all_json[q_id]["results"][sys_name] = []

            for rank, p in enumerate(parsed_preds):
                if p == -1 or p >= len(demo.df_paths):
                    missing_image_count += 1
                    continue

                img_path = str(Path(demo.df_paths.iloc[p]["Image Path"]).name)
                caption = str(demo.df_paths.iloc[p]["Image Caption"])

                res_obj = {
                    "rank": rank + 1,
                    "image_id": img_path,
                    "caption_snippet": textwrap.shorten(caption, width=100)
                }
                all_json[q_id]["results"][sys_name].append(res_obj)

                all_csv.append({
                    "query_id": q_id,
                    "query_text": q_text,
                    "query_type": q_type,
                    "system": sys_name,
                    "rank": rank + 1,
                    "image_id": img_path,
                    "caption_snippet": textwrap.shorten(caption, width=100)
                })

        # Generate contact sheets for the LLM review packet, plus an initial demo sample.
        llm_review_ids = {
            "q1_1", "q1_8", "q1_12",
            "q2_3", "q2_10", "q2_17",
            "q3_1", "q3_5", "q3_8", "q3_11"
        }
        if q_id in llm_review_ids or sheets_generated < 15:
            cs_path = cs_dir / f"{q_id}_contact_sheet.png"
            demo.create_contact_sheet(q_text, q_type, q_id, predictions_for_sheet, 5, cs_path)
            sheets_generated += 1

    with open(out_dir / "m12_50_query_results.json", "w") as f:
        json.dump(all_json, f, indent=2)

    pd.DataFrame(all_csv).to_csv(out_dir / "m12_50_query_results.csv", index=False)

    # Review template
    review_rows = []
    for q_id, data in all_json.items():
        for sys_name in systems:
            top_1 = data["results"][sys_name][0]["image_id"] if len(data["results"][sys_name]) > 0 else "None"
            review_rows.append({
                "query_id": q_id,
                "query_text": data["query_text"],
                "query_type": data["query_type"],
                "system": sys_name,
                "top1_image_id": top_1,
                "top1_relevance_0_2": "",
                "top5_contains_relevant_yes_no": "",
                "top10_contains_relevant_yes_no": "",
                "best_candidate_rank": "",
                "reviewer_type": "LLM/Manual",
                "reviewer_notes": "",
                "final_manual_decision": ""
            })

    pd.DataFrame(review_rows).to_csv(out_dir / "m12_50_query_manual_review_template.csv", index=False)

    print(f"Processed {len(df_val)} queries.")
    print(f"Systems evaluated: {systems}")
    print(f"Missing images in top K: {missing_image_count}")
    print(f"Contact sheets generated: {sheets_generated}")
    print(f"Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()
