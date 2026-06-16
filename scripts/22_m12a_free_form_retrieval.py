import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
import re
import math
import shutil

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
IMAGES_DIR = Path("/DATA5/prabhakar/telecom/extracted_images/images/")
CSV_PATH = Path("/DATA1/prabhakar/telecom/All Images Path.csv")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["bm25", "tfidf", "hybrid"], default="bm25")
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--copy-images", type=str, default="true")
    parser.add_argument("--detailed-sheet", type=str, default="true")
    parser.add_argument("--html-report", type=str, default="true")
    parser.add_argument("--sheet-cols", type=int, default=2)
    parser.add_argument("--tile-width", type=int, default=1000)
    parser.add_argument("--tile-height", type=int, default=700)
    return parser.parse_args()

def tokenize(text):
    if pd.isna(text): return []
    text = str(text).lower()
    return re.findall(r'\w+', text)

def build_documents(df):
    docs = []
    for _, row in df.iterrows():
        cap = str(row["Image Caption"]) if not pd.isna(row["Image Caption"]) else ""
        ctx = str(row["Context"]) if not pd.isna(row["Context"]) else ""
        path = str(row["Image Path"]) if not pd.isna(row["Image Path"]) else ""
        doc = f"{cap} {ctx} {path}"
        docs.append(doc)
    return docs

def draw_text(draw, x, y, text, font, fill="black", max_width=400):
    lines = textwrap.wrap(str(text), width=int(max_width/8))
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += 20
    return y

def create_contact_sheet(query_text, method, df, top_indices, top_scores, top_k, out_path):
    img_width, img_height = 350, 350
    margin = 20
    header_height = 150
    row_height = img_height + 80
    num_rows = 1
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
    draw.text((margin, margin+30), f"Method: {method.upper()} (Free-Form Retrieval)", font=font, fill="black")
    draw.text((margin, margin+60), "Note: Searching actual corpus, no precomputed prediction mapping.", font=font, fill="blue")

    def paste_image(x, y, row_idx, score, title):
        if row_idx < 0 or row_idx >= len(df):
            draw.text((x, y), "No Image", fill="red", font=font)
            return

        img_name = Path(df.iloc[row_idx]["Image Path"]).name
        img_path = IMAGES_DIR / img_name

        try:
            im = Image.open(img_path).convert("RGB")
            im = ImageOps.contain(im, (img_width, img_height))

            sheet.paste(im, (x, y))
            draw.rectangle([x-2, y-2, x+img_width+2, y+img_height+2], outline="gray", width=3)

            caption = df.iloc[row_idx]["Image Caption"]
            short_cap = textwrap.shorten(str(caption), width=45)

            draw.text((x, y + img_height + 5), title, font=font_bold, fill="black")
            draw.text((x, y + img_height + 25), f"Image ID: {img_name} | Score: {score:.2f}", font=font, fill="black")
            draw.text((x, y + img_height + 45), short_cap, font=font, fill="gray")
        except Exception as e:
            draw.text((x, y), f"Error: {e}", fill="red", font=font)

    y_offset = header_height
    for i, (idx, score) in enumerate(zip(top_indices[:num_cols], top_scores[:num_cols])):
        x_offset = margin + i * (img_width + margin)
        paste_image(x_offset, y_offset, idx, score, f"Rank {i+1}")

    sheet.save(out_path)

def create_detailed_contact_sheet(query_text, method, df, top_indices, top_scores, args, out_path):
    img_width, img_height = args.tile_width, args.tile_height
    margin = 40
    header_height = 200

    num_items = len(top_indices)
    num_cols = min(num_items, args.sheet_cols)
    if num_cols == 0: return
    num_rows = math.ceil(num_items / num_cols)

    text_height = 200
    row_height = img_height + text_height + margin

    sheet_width = num_cols * (img_width + margin) + margin
    sheet_height = header_height + num_rows * row_height

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 24)
        font_bold = ImageFont.truetype("LiberationSans-Bold.ttf", 28)
        font_title = ImageFont.truetype("LiberationSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_title = ImageFont.load_default()

    draw.text((margin, margin), f"Query: {textwrap.shorten(query_text, width=100)}", font=font_title, fill="black")
    draw.text((margin, margin+60), f"Method: {method.upper()} (Free-Form Retrieval - Detailed)", font=font_bold, fill="black")

    def paste_image(x, y, row_idx, score, rank):
        if row_idx < 0 or row_idx >= len(df):
            draw.text((x, y), "No Image", fill="red", font=font)
            return

        img_name = Path(df.iloc[row_idx]["Image Path"]).name
        img_path = IMAGES_DIR / img_name

        try:
            im = Image.open(img_path).convert("RGB")
            im = ImageOps.contain(im, (img_width, img_height))

            paste_x = x + (img_width - im.width) // 2
            sheet.paste(im, (paste_x, y))
            draw.rectangle([paste_x-2, y-2, paste_x+im.width+2, y+im.height+2], outline="gray", width=4)

            caption = df.iloc[row_idx]["Image Caption"]
            short_cap = textwrap.shorten(str(caption), width=150)

            text_y = y + img_height + 20
            draw.text((x, text_y), f"Rank {rank}", font=font_bold, fill="black")
            draw.text((x, text_y + 35), f"Image ID: {img_name} | Score: {score:.2f}", font=font, fill="black")

            lines = textwrap.wrap(short_cap, width=70)
            cap_y = text_y + 70
            for line in lines:
                draw.text((x, cap_y), line, font=font, fill="dimgray")
                cap_y += 30

        except Exception as e:
            draw.text((x, y), f"Error: {e}", fill="red", font=font)

    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        col = i % num_cols
        row = i // num_cols
        x_offset = margin + col * (img_width + margin)
        y_offset = header_height + row * row_height
        paste_image(x_offset, y_offset, idx, score, i+1)

    sheet.save(out_path)

def create_html_report(query_text, method, dense_available, results, out_path):
    html = f"""<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }}
.container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
.result {{ border-bottom: 1px solid #eee; padding: 20px 0; display: flex; gap: 20px; }}
.result img {{ max-width: 600px; max-height: 400px; border: 1px solid #ccc; }}
.details {{ flex: 1; }}
h1 {{ color: #333; }}
h3 {{ margin-top: 0; }}
</style>
</head>
<body>
<div class="container">
    <h1>M12A Free-Form Retrieval Results</h1>
    <p><strong>Query:</strong> {query_text}</p>
    <p><strong>Method:</strong> {method.upper()}</p>
    <p><strong>Dense Fallback Available:</strong> {dense_available}</p>
    <hr>
"""
    for res in results:
        rank = res["rank"]
        img_id = res["image_id"]
        score = res["score"]
        caption = res["caption_snippet"]
        rank_str = f"{rank:02d}"

        img_src = f"retrieved_images/rank_{rank_str}_{img_id}"

        html += f"""    <div class="result">
        <div>
            <a href="{img_src}" target="_blank">
                <img src="{img_src}" alt="{img_id}">
            </a>
        </div>
        <div class="details">
            <h3>Rank {rank}</h3>
            <p><strong>Image ID:</strong> {img_id}</p>
            <p><strong>Score:</strong> {score:.4f}</p>
            <p><strong>Caption:</strong> {caption}</p>
        </div>
    </div>
"""
    html += "</div></body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"Error: Corpus file {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    docs = build_documents(df)

    # Check if dense is available
    dense_available = False

    if args.method in ["bm25", "hybrid"]:
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [tokenize(doc) for doc in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = tokenize(args.query)
            scores = bm25.get_scores(tokenized_query)
        except ImportError:
            print("Warning: rank_bm25 not installed, falling back to TF-IDF.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(docs)
            q_vec = vectorizer.transform([args.query])
            scores = (X * q_vec.T).toarray().flatten()
    elif args.method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(docs)
        q_vec = vectorizer.transform([args.query])
        scores = (X * q_vec.T).toarray().flatten()

    # Dense retrieval fallback
    if args.method == "hybrid" and not dense_available:
        print("Note: Dense embeddings not found. Hybrid fallback to sparse only.")

    # Get top k
    import numpy as np
    top_k_indices = np.argsort(scores)[::-1][:args.top_k]
    top_k_scores = scores[top_k_indices]

    results = []
    csv_rows = []

    for rank, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        img_name = Path(df.iloc[idx]["Image Path"]).name
        cap = str(df.iloc[idx]["Image Caption"])

        results.append({
            "rank": rank + 1,
            "image_id": img_name,
            "score": float(score),
            "caption_snippet": textwrap.shorten(cap, width=100)
        })

        csv_rows.append({
            "query": args.query,
            "method": args.method,
            "rank": rank + 1,
            "image_id": img_name,
            "score": float(score),
            "caption_snippet": textwrap.shorten(cap, width=100)
        })

    with open(out_dir / "latest_results.json", "w") as f:
        json.dump({
            "query": args.query,
            "method": args.method,
            "dense_available": dense_available,
            "results": results
        }, f, indent=2)

    pd.DataFrame(csv_rows).to_csv(out_dir / "latest_results.csv", index=False)

    cs_path = out_dir / "latest_contact_sheet.png"
    create_contact_sheet(args.query, args.method, df, top_k_indices, top_k_scores, args.top_k, cs_path)

    if args.detailed_sheet.lower() == "true":
        ds_path = out_dir / "latest_contact_sheet_detailed.png"
        create_detailed_contact_sheet(args.query, args.method, df, top_k_indices, top_k_scores, args, ds_path)

    if args.copy_images.lower() == "true":
        img_dir = out_dir / "retrieved_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            rank_str = f"{res['rank']:02d}"
            src = IMAGES_DIR / res['image_id']
            dst = img_dir / f"rank_{rank_str}_{res['image_id']}"
            if src.exists():
                shutil.copy2(src, dst)

    if args.html_report.lower() == "true":
        html_path = out_dir / "latest_results.html"
        create_html_report(args.query, args.method, dense_available, results, html_path)

    print(f"Retrieved {len(results)} results using {args.method}.")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
