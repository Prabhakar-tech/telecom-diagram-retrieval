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

def run_sanity_eval():
    print("Running M6.5 Sanity Evaluation...")
    root = Path("/DATA5/prabhakar/telecom_retrieval")
    
    # Load lexicon
    lexicon = process_lexicon()
    aliases = get_aliases()
    
    # Load corpus
    csv_path = Path("/DATA1/prabhakar/telecom/All Images Path.csv")
    df = pd.read_csv(csv_path)
    
    # Build B2 corpus (Caption + Context)
    corpus_docs = []
    for idx, row in df.iterrows():
        cap = row['Image Caption'] if pd.notna(row['Image Caption']) else ""
        ctx = row['Context'] if pd.notna(row['Context']) else ""
        corpus_docs.append(f"{cap} {ctx}")
        
    tokenized_corpus = [simple_tokenize(doc) for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    h2r, r2h = load_duplicate_mapping()
    
    q1 = json.load(open(root / "queries/q1_captions.json"))["queries"][:200]
    q2 = json.load(open(root / "queries/q2_paraphrased.json"))["queries"][:200]
    q3 = json.load(open(root / "queries/q3_context.json"))["queries"][:200]
    
    results = {}
    
    for q_type, q_list in [("q1", q1), ("q2", q2), ("q3", q3)]:
        print(f"Evaluating {q_type.upper()}...")
        
        orig_preds = []
        exp_preds = []
        fuse_10_preds = []
        fuse_25_preds = []
        fuse_50_preds = []
        
        for q in q_list:
            orig_text = q["text"]
            expanded_text, _, _, _ = expand_query(orig_text, lexicon, aliases)
            
            orig_tokens = simple_tokenize(orig_text)
            exp_tokens = simple_tokenize(expanded_text)
            
            score_orig = np.array(bm25.get_scores(orig_tokens))
            score_exp = np.array(bm25.get_scores(exp_tokens))
            
            # normalize for fusion (min-max)
            def min_max_norm(s):
                s_min, s_max = s.min(), s.max()
                if s_max > s_min:
                    return (s - s_min) / (s_max - s_min)
                return s
                
            norm_orig = min_max_norm(score_orig)
            norm_exp = min_max_norm(score_exp)
            
            fuse_10 = norm_orig + 0.10 * norm_exp
            fuse_25 = norm_orig + 0.25 * norm_exp
            fuse_50 = norm_orig + 0.50 * norm_exp
            
            # top 100 for each
            orig_preds.append(np.argsort(score_orig)[::-1][:100].tolist())
            exp_preds.append(np.argsort(score_exp)[::-1][:100].tolist())
            fuse_10_preds.append(np.argsort(fuse_10)[::-1][:100].tolist())
            fuse_25_preds.append(np.argsort(fuse_25)[::-1][:100].tolist())
            fuse_50_preds.append(np.argsort(fuse_50)[::-1][:100].tolist())
            
        m_orig = evaluate_run(q_list, orig_preds, h2r, r2h)
        m_exp = evaluate_run(q_list, exp_preds, h2r, r2h)
        m_fuse10 = evaluate_run(q_list, fuse_10_preds, h2r, r2h)
        m_fuse25 = evaluate_run(q_list, fuse_25_preds, h2r, r2h)
        m_fuse50 = evaluate_run(q_list, fuse_50_preds, h2r, r2h)
        
        results[q_type] = {
            "original": m_orig,
            "expanded": m_exp,
            "fuse_0.10": m_fuse10,
            "fuse_0.25": m_fuse25,
            "fuse_0.50": m_fuse50
        }
        
    with open(root / "reports/m65_sanity_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Sanity eval complete.")

if __name__ == "__main__":
    run_sanity_eval()
