import json
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path("/DATA5/prabhakar/telecom_retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

def load_queries(qs):
    path_map = {"Q1": "queries/q1_captions.json", "Q2": "queries/q2_paraphrased.json", "Q3": "queries/q3_context.json"}
    with open(PROJECT_ROOT / path_map[qs], "r") as f:
        return json.load(f)["queries"]

files = [
    "reports/m5_clip_predictions_q1.json", "reports/m5_clip_predictions_q2.json", "reports/m5_clip_predictions_q3.json",
    "reports/m6_predictions_capctx_ocr_q1.json", "reports/m6_predictions_caption_ocr_q1.json", "reports/m6_predictions_caption_ocr_q2.json", "reports/m6_predictions_caption_ocr_q3.json",
    "reports/m6b_colpali_predictions_q1.json", "reports/m6b_colpali_predictions_q2.json", "reports/m6b_colpali_predictions_q3.json"
]

audit = {}

for f_name in files:
    p = PROJECT_ROOT / f_name
    qs = "Q" + f_name.split("_q")[-1][0]

    if not p.exists():
        audit[f_name] = {"exists": False}
        continue

    try:
        with open(p, "r") as f:
            data = json.load(f)
    except Exception as e:
        audit[f_name] = {"exists": True, "error": str(e)}
        continue

    q_data = load_queries(qs)
    gt_row = q_data[0]["ground_truth_row"]
    q_id = q_data[0]["query_id"]

    is_dict = isinstance(data, dict)
    num_queries = len(data)

    if is_dict:
        keys = list(data.keys())
        if len(keys) == 0: continue
        key_format = type(keys[0]).__name__ + " (e.g. " + str(keys[0]) + ")"
        first_preds = data[keys[0]]
    else:
        key_format = "list index"
        first_preds = data[0] if len(data) > 0 else []

    top_k = len(first_preds)

    if top_k > 0:
        if isinstance(first_preds[0], list):
            pred_format = "list of lists (e.g. " + str(first_preds[0]) + ")"
            docs = [x[0] for x in first_preds]
        elif isinstance(first_preds[0], dict):
            pred_format = "list of dicts (e.g. " + str(first_preds[0]) + ")"
            # find doc id key
            docs = []
            for item in first_preds:
                for k in ["row", "row_id", "doc_id", "id", "index", "image_id", "docid"]:
                    if k in item:
                        docs.append(item[k])
                        break
        else:
            pred_format = "list of " + type(first_preds[0]).__name__ + " (e.g. " + str(first_preds[0]) + ")"
            docs = first_preds
    else:
        pred_format = "empty"
        docs = []

    gt_found = gt_row in docs or str(gt_row) in docs

    why = "Format is okay"
    if is_dict:
        if str(q_id) not in data and q_id not in data and "0" not in data and 0 not in data:
            why = "Mismatch between query ID and prediction keys: " + str(q_id) + " vs keys like " + str(list(data.keys())[:2])
    elif not is_dict and isinstance(first_preds, dict):
        # wait, if it's a list, and elements are dicts, we need to extract doc_id from the dict
        pass

    audit[f_name] = {
        "exists": True,
        "num_queries": num_queries,
        "query_id_format": key_format,
        "prediction_row_id_format": pred_format,
        "top_k_length": top_k,
        "ground_truth_found_in_first_query": gt_found,
        "why_rank_extraction_na": why
    }

with open(PROJECT_ROOT / "reports/m9_prediction_rank_debug_audit.json", "w") as f:
    json.dump(audit, f, indent=2)
