import json
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
import sys
import re

sys.path.append("/DATA5/prabhakar/telecom_retrieval")
from eval.metrics import evaluate_run, load_duplicate_mapping
import importlib.util

spec = importlib.util.spec_from_file_location("acronym_exp", "scripts/11_acronym_expansion.py")
acronym_exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acronym_exp)
process_lexicon = acronym_exp.process_lexicon
get_aliases = acronym_exp.get_aliases
expand_query = acronym_exp.expand_query

def simple_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [w.lower() for w in re.findall(r'\b\w+\b', text)]

def run_full_eval():
    print("Running Full M6.5 Evaluation...")
    root = Path("/DATA5/prabhakar/telecom_retrieval")
    
    lexicon = process_lexicon()
    aliases = get_aliases()
    
    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    df = pd.read_csv(csv_path)
    
    corpus_docs = []
    for idx, row in df.iterrows():
        cap = row['Image Caption'] if pd.notna(row['Image Caption']) else ""
        ctx = row['Context'] if pd.notna(row['Context']) else ""
        corpus_docs.append(f"{cap} {ctx}")
        
    tokenized_corpus = [simple_tokenize(doc) for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    h2r, r2h = load_duplicate_mapping()
    
    q1 = json.load(open(root / "queries/q1_captions.json"))["queries"]
    q2 = json.load(open(root / "queries/q2_paraphrased.json"))["queries"]
    q3 = json.load(open(root / "queries/q3_context.json"))["queries"]
    
    results = {}
    deltas = []
    
    for q_type, q_list in [("q1", q1), ("q2", q2), ("q3", q3)]:
        print(f"Evaluating {q_type.upper()}...")
        
        orig_preds = []
        exp_preds = []
        fuse_05_preds = []
        fuse_10_preds = []
        fuse_25_preds = []
        fuse_50_preds = []
        
        fusion_10_pred_dict = {}
        
        # Determine gold answers based on df and duplicates
        def get_rr(pred_indices, ground_truth_row):
            gt_hash = r2h.get(ground_truth_row)
            if gt_hash is None:
                valid_set = {ground_truth_row}
            else:
                valid_set = set(h2r.get(gt_hash, [ground_truth_row]))
            
            for rank, idx in enumerate(pred_indices):
                if idx in valid_set:
                    return 1.0 / (rank + 1)
            return 0.0
        
        for q in q_list:
            orig_text = q["text"]
            expanded_text, _, _, _ = expand_query(orig_text, lexicon, aliases)
            
            orig_tokens = simple_tokenize(orig_text)
            exp_tokens = simple_tokenize(expanded_text)
            
            score_orig = np.array(bm25.get_scores(orig_tokens))
            score_exp = np.array(bm25.get_scores(exp_tokens))
            
            s_min, s_max = score_orig.min(), score_orig.max()
            norm_orig = (score_orig - s_min) / (s_max - s_min) if s_max > s_min else score_orig
            
            s_min, s_max = score_exp.min(), score_exp.max()
            norm_exp = (score_exp - s_min) / (s_max - s_min) if s_max > s_min else score_exp
            
            fuse_05 = norm_orig + 0.05 * norm_exp
            fuse_10 = norm_orig + 0.10 * norm_exp
            fuse_25 = norm_orig + 0.25 * norm_exp
            fuse_50 = norm_orig + 0.50 * norm_exp
            
            idx_orig = np.argsort(score_orig)[::-1][:100]
            idx_exp = np.argsort(score_exp)[::-1][:100]
            idx_f05 = np.argsort(fuse_05)[::-1][:100]
            idx_f10 = np.argsort(fuse_10)[::-1][:100]
            idx_f25 = np.argsort(fuse_25)[::-1][:100]
            idx_f50 = np.argsort(fuse_50)[::-1][:100]
            
            orig_preds.append(idx_orig.tolist())
            exp_preds.append(idx_exp.tolist())
            fuse_05_preds.append(idx_f05.tolist())
            fuse_10_preds.append(idx_f10.tolist())
            fuse_25_preds.append(idx_f25.tolist())
            fuse_50_preds.append(idx_f50.tolist())
            
            f10_pred_ids = [int(i) for i in idx_f10]
            fusion_10_pred_dict[q["query_id"]] = f10_pred_ids
            
            ground_truth_row = q["ground_truth_row"]
            rr_orig = get_rr(idx_orig, ground_truth_row)
            rr_f10 = get_rr(idx_f10, ground_truth_row)
            delta = rr_f10 - rr_orig
            
            if abs(delta) > 0.0001:
                deltas.append({
                    "query_id": q["query_id"],
                    "query_type": q_type,
                    "original_query": orig_text,
                    "expanded_query": expanded_text,
                    "rr_orig": rr_orig,
                    "rr_fuse_10": rr_f10,
                    "delta": delta,
                    "helped": delta > 0
                })
        
        m_orig = evaluate_run(q_list, orig_preds, h2r, r2h)
        m_exp = evaluate_run(q_list, exp_preds, h2r, r2h)
        m_fuse05 = evaluate_run(q_list, fuse_05_preds, h2r, r2h)
        m_fuse10 = evaluate_run(q_list, fuse_10_preds, h2r, r2h)
        m_fuse25 = evaluate_run(q_list, fuse_25_preds, h2r, r2h)
        m_fuse50 = evaluate_run(q_list, fuse_50_preds, h2r, r2h)
        
        results[q_type] = {
            "original_bm25": m_orig,
            "pure_expanded_bm25": m_exp,
            "fusion_0.05": m_fuse05,
            "fusion_0.10": m_fuse10,
            "fusion_0.25": m_fuse25,
            "fusion_0.50": m_fuse50
        }
        
        with open(root / f"reports/m65_acronym_expansion_predictions_{q_type}.json", "w") as f:
            json.dump(fusion_10_pred_dict, f, indent=2)
            
    with open(root / "reports/m65_acronym_expansion_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    df_deltas = pd.DataFrame(deltas)
    if len(df_deltas) > 0:
        df_deltas = df_deltas.sort_values("delta", ascending=False)
    df_deltas.to_csv(root / "reports/m65_acronym_expansion_deltas.csv", index=False)
        
    print("Full eval complete.")

if __name__ == "__main__":
    run_full_eval()
