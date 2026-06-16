import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
import re

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

    print(f"Retrieved {len(results)} results using {args.method}.")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
