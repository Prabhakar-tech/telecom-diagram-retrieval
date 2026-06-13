import json
import pandas as pd

results_path = "/DATA5/prabhakar/telecom_retrieval/reports/m6_ocr_results.json"
csv_path = "/DATA5/prabhakar/telecom_retrieval/reports/master_results.csv"

with open(results_path, "r") as f:
    data = json.load(f)

records = []
for exp_name, exp_data in data["experiments"].items():
    for qset, metrics in exp_data.items():
        records.append({
            "Milestone": "M6a",
            "Experiment": exp_name,
            "QuerySet": qset,
            "R@1": metrics["recall@1"],
            "R@2": metrics["recall@2"],
            "R@5": metrics["recall@5"],
            "R@10": metrics["recall@10"],
            "MRR@10": metrics["mrr@10"],
            "dupR@1": metrics["dup_recall@1"],
            "dupMRR@10": metrics["dup_mrr@10"]
        })

df_new = pd.DataFrame(records)
# Append without header
df_new.to_csv(csv_path, mode='a', header=False, index=False)
print("Updated master_results.csv")
