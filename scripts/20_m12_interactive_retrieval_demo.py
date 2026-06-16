import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
from difflib import SequenceMatcher

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
IMAGES_DIR = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
CSV_PATH = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
EVAL_MAPPING = PROJECT_ROOT / "eval" / "duplicate_mapping.json"

df_paths = pd.read_csv(CSV_PATH)
duplicate_mapping = json.load(open(EVAL_MAPPING))

h2r = {}
r2h = {}
for h, rows in duplicate_mapping.items():
    if h == "metadata": continue
    h2r[h] = rows
    for r in rows:
        r2h[r] = h

def is_hit(pred_row, gt_row):
    gt_hash = r2h.get(gt_row)
    valid_set = set(h2r.get(gt_hash, [gt_row])) if gt_hash else {gt_row}
    return pred_row in valid_set

def parse_row_id(p):
    if isinstance(p, int): return p
    if isinstance(p, str):
        if p.startswith('image_'):
            num_part = p.replace('image_', '').split('.')[0]
            try: return int(num_part)
            except: pass
        try: return int(p)
        except: return -1
    return -1

def get_prediction(system, q_set, q_id):
    # Query ids are 1-based, e.g., q1_1, while list predictions are usually 0-based.
    idx = max(0, int(q_id.split('_')[1]) - 1)
    q_set_lower = q_set.lower()
    system_lookup = {"bm25": "BM25", "bge": "BGE", "final": "final"}
    system = system_lookup.get(str(system).strip().lower(), system)

    file_map = {
        "BM25": {
            "q1": "reports/m55_predictions_b1_q1.json",
            "q2": "reports/m55_predictions_b1_q2.json",
            "q3": "reports/m55_predictions_b2_q3.json"
        },
        "BGE": {
            "q1": "reports/m55_predictions_d1_q1.json",
            "q2": "reports/m55_predictions_d1_q2.json",
            "q3": "reports/m55_predictions_d2_q3.json"
        },
        "final": {
            "q1": "reports/m7_hybrid_lexical_dense_predictions_q1.json",
            "q2": "reports/m55_predictions_h1a_q2.json",
            "q3": "reports/m55_predictions_union_top50_rerank_q3.json"
        }
    }

    # Handle the fact that M7 has specific file for final architecture
    if system == "final":
        if q_set_lower == "q1":
            path = PROJECT_ROOT / file_map["final"]["q1"]
        elif q_set_lower == "q2":
            path = PROJECT_ROOT / file_map["final"]["q2"]
        elif q_set_lower == "q3":
            path = PROJECT_ROOT / file_map["final"]["q3"]
        else:
            return []
    elif system in file_map:
        path = PROJECT_ROOT / file_map[system][q_set_lower]
    else:
        return []

    if not path.exists(): return []
    data = json.load(open(path))
    if isinstance(data, dict):
        return data.get(q_id, [])
    elif isinstance(data, list):
        return data[idx] if idx < len(data) else []
    return []

def draw_text(draw, x, y, text, font, fill="black", max_width=400):
    lines = textwrap.wrap(str(text), width=int(max_width/8))
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += 20
    return y

def create_contact_sheet(query_text, query_type, q_id, predictions_dict, top_k, out_path):
    systems = list(predictions_dict.keys())

    img_width, img_height = 350, 350
    margin = 20
    header_height = 150
    row_height = img_height + 80
    num_rows = len(systems)
    num_cols = min(top_k, 5) # limit visual to top 5

    sheet_width = num_cols * (img_width + margin) + margin
    sheet_height = header_height + num_rows * (row_height + margin)

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 16)
        font_bold = ImageFont.truetype("LiberationSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    # Header
    draw.text((margin, margin), f"Query: {textwrap.shorten(query_text, width=200)}", font=font_bold, fill="black")
    draw.text((margin, margin+30), f"Query Type: {query_type.upper()} | ID: {q_id}", font=font, fill="black")
    draw.text((margin, margin+60), "Note: Online mode not supported. Operating in prepared-query mode.", font=font, fill="red")

    def paste_image(x, y, row_idx, title):
        if row_idx < 0 or row_idx >= len(df_paths):
            draw.text((x, y), "No Image", fill="red", font=font)
            return

        img_name = Path(df_paths.iloc[row_idx]["Image Path"]).name
        img_path = IMAGES_DIR / img_name

        try:
            im = Image.open(img_path).convert("RGB")
            im = ImageOps.contain(im, (img_width, img_height))

            sheet.paste(im, (x, y))
            draw.rectangle([x-2, y-2, x+img_width+2, y+img_height+2], outline="gray", width=3)

            caption = df_paths.iloc[row_idx]["Image Caption"]
            short_cap = textwrap.shorten(str(caption), width=45)

            draw.text((x, y + img_height + 5), title, font=font_bold, fill="black")
            draw.text((x, y + img_height + 25), f"Image ID: {img_name}", font=font, fill="black")
            draw.text((x, y + img_height + 45), short_cap, font=font, fill="gray")
        except Exception as e:
            draw.text((x, y), f"Error: {e}", fill="red", font=font)

    y_offset = header_height
    for sys in systems:
        draw.text((margin, y_offset - 20), f"System: {sys}", font=font_bold, fill="black")
        preds = predictions_dict[sys][:num_cols]
        for i, p_row in enumerate(preds):
            p_row = parse_row_id(p_row)
            if p_row == -1: continue
            x_offset = margin + i * (img_width + margin)
            paste_image(x_offset, y_offset, p_row, f"Rank {i+1}")
        y_offset += row_height + margin

    sheet.save(out_path)

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--query-type", type=str, choices=["auto", "q1", "q2", "q3"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--systems", type=str, default="final,bm25,bge")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    systems_to_run = [s.strip() for s in args.systems.split(',')]

    # Load validation queries
    val_set_path = PROJECT_ROOT / "reports/m12_50_query_validation_set.csv"
    if not val_set_path.exists():
        print("Validation set not found. Prepared-query mode requires it.")
        sys.exit(1)

    df_val = pd.read_csv(val_set_path)

    # Match query
    best_match = None
    best_score = 0
    for _, row in df_val.iterrows():
        score = similarity(args.query, row["query_text"])
        if score > best_score:
            best_score = score
            best_match = row

    if best_score < 0.6:
        print("ERROR: True free-form retrieval is not supported because live indexes are not loaded in this environment.")
        print("Running in prepared-query mode. Please provide a query from the 50-query validation set.")
        sys.exit(1)

    q_id = best_match["query_id"]
    q_text = best_match["query_text"]
    q_type = best_match["intended_query_type"] if args.query_type == "auto" else args.query_type

    print(f"Matched Query ID: {q_id} (Score: {best_score:.2f})")
    print(f"Query Type Auto-Resolution: {q_type.upper()}")

    results = {}
    csv_rows = []

    predictions_for_sheet = {}

    for sys_name in systems_to_run:
        preds = get_prediction(sys_name, q_type, q_id)
        preds = preds[:args.top_k]

        parsed_preds = [parse_row_id(p) for p in preds]
        predictions_for_sheet[sys_name] = parsed_preds

        results[sys_name] = []
        for rank, p in enumerate(parsed_preds):
            if p == -1 or p >= len(df_paths):
                continue
            img_path = str(Path(df_paths.iloc[p]["Image Path"]).name)
            caption = str(df_paths.iloc[p]["Image Caption"])

            res_obj = {
                "rank": rank + 1,
                "image_id": img_path,
                "caption_snippet": textwrap.shorten(caption, width=100)
            }
            results[sys_name].append(res_obj)

            csv_rows.append({
                "query": q_text,
                "query_type": q_type,
                "system": sys_name,
                "rank": rank + 1,
                "image_id": img_path,
                "caption_snippet": textwrap.shorten(caption, width=100)
            })

    with open(out_dir / "latest_results.json", "w") as f:
        json.dump({"query": q_text, "query_type": q_type, "results": results}, f, indent=2)

    pd.DataFrame(csv_rows).to_csv(out_dir / "latest_results.csv", index=False)

    cs_path = out_dir / "latest_contact_sheet.png"
    create_contact_sheet(q_text, q_type, q_id, predictions_for_sheet, args.top_k, cs_path)

    print(f"Saved results to {out_dir}")
    print(f"Contact sheet saved to {cs_path}")

if __name__ == "__main__":
    main()
