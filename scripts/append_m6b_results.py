import pandas as pd
import json

results_file = "reports/master_results.csv"
colpali_json = "reports/m6b_colpali_results.json"

df = pd.read_csv(results_file)
with open(colpali_json, 'r') as f:
    results = json.load(f)

new_rows = []
for q_type, display_name in [("q1", "Q1: Captions"), ("q2", "Q2: Paraphrased"), ("q3", "Q3: Context")]:
    res = results[q_type]
    new_rows.append({
        "Milestone": "M6b",
        "Model": "vidore/colpali-v1.2",
        "Modality": "Vision-Document",
        "Query_Set": display_name,
        "Recall@1": round(res["dup_recall@1"], 4),
        "Recall@5": round(res["dup_recall@5"], 4),
        "Recall@10": round(res["dup_recall@10"], 4),
        "MRR@10": round(res["dup_mrr@10"], 4)
    })

new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv(results_file, index=False)
print("Successfully appended M6b results to master_results.csv")
