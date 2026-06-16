import json
import sys
from pathlib import Path
import pandas as pd
import argparse
import textwrap

import importlib.util

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
CSV_PATH = Path("/DATA1/prabhakar/telecom/All Images Path.csv")

spec = importlib.util.spec_from_file_location("ff_demo", PROJECT_ROOT / "scripts/22_m12a_free_form_retrieval.py")
ff_demo = importlib.util.module_from_spec(spec)
sys.modules["ff_demo"] = ff_demo
spec.loader.exec_module(ff_demo)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--method", type=str, default="bm25")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cs_dir = out_dir / "m12a_free_form_contact_sheets"
    cs_dir.mkdir(parents=True, exist_ok=True)

    df_val = pd.read_csv(args.query_set)
    df_corpus = pd.read_csv(CSV_PATH)
    docs = ff_demo.build_documents(df_corpus)

    # Initialize BM25 or TF-IDF once
    if args.method in ["bm25", "hybrid"]:
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [ff_demo.tokenize(doc) for doc in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            searcher = lambda q: bm25.get_scores(ff_demo.tokenize(q))
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(docs)
            searcher = lambda q: (X * vectorizer.transform([q]).T).toarray().flatten()
    elif args.method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(docs)
        searcher = lambda q: (X * vectorizer.transform([q]).T).toarray().flatten()

    all_json = {}
    all_csv = []

    missing_images = 0
    empty_results = 0
    sheets_generated = 0

    import numpy as np

    for _, row in df_val.iterrows():
        q_id = row["query_id"]
        q_text = row["query_text"]

        scores = searcher(q_text)
        top_k_indices = np.argsort(scores)[::-1][:args.top_k]
        top_k_scores = scores[top_k_indices]

        results = []
        for rank, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
            img_name = Path(df_corpus.iloc[idx]["Image Path"]).name
            cap = str(df_corpus.iloc[idx]["Image Caption"])

            res_obj = {
                "rank": rank + 1,
                "image_id": img_name,
                "score": float(score),
                "caption_snippet": textwrap.shorten(cap, width=100)
            }
            results.append(res_obj)

            all_csv.append({
                "query_id": q_id,
                "query_text": q_text,
                "method": args.method,
                "rank": rank + 1,
                "image_id": img_name,
                "score": float(score),
                "caption_snippet": textwrap.shorten(cap, width=100)
            })

        all_json[q_id] = {
            "query_text": q_text,
            "method": args.method,
            "results": results
        }

        if len(results) == 0:
            empty_results += 1

        if sheets_generated < 15:
            cs_path = cs_dir / f"{q_id}_free_form_contact_sheet.png"
            ff_demo.create_contact_sheet(q_text, args.method, df_corpus, top_k_indices, top_k_scores, 5, cs_path)
            sheets_generated += 1

    with open(out_dir / "m12a_free_form_results.json", "w") as f:
        json.dump(all_json, f, indent=2)

    pd.DataFrame(all_csv).to_csv(out_dir / "m12a_free_form_results.csv", index=False)

    # Review template
    review_rows = []
    for q_id, data in all_json.items():
        top_1 = data["results"][0] if len(data["results"]) > 0 else None
        review_rows.append({
            "query_id": q_id,
            "query_text": data["query_text"],
            "method": args.method,
            "top1_image_id": top_1["image_id"] if top_1 else "None",
            "top1_caption": top_1["caption_snippet"] if top_1 else "None",
            "top1_relevance_0_2": "",
            "top5_contains_relevant_yes_no": "",
            "top10_contains_relevant_yes_no": "",
            "reviewer_notes": "",
            "final_manual_decision": ""
        })

    pd.DataFrame(review_rows).to_csv(out_dir / "m12a_free_form_manual_review_template.csv", index=False)

    print(f"Processed {len(df_val)} free-form queries using {args.method}.")
    print(f"Missing images in top K: {missing_images}")
    print(f"Empty results: {empty_results}")
    print(f"Contact sheets generated: {sheets_generated}")

if __name__ == "__main__":
    main()
