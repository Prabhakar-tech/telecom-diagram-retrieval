# M12A True Free-Form Product Demo Report

## Why M12A Was Needed
The previous M12 demo operated strictly in **prepared-query mode**. It functioned as a test wrapper that fuzzy-matched natural language inputs against 50 predefined validation queries and loaded precomputed, offline predictions (like `reports/m7_hybrid_lexical_dense_predictions_q1.json`). While effective for controlled validation of the final architecture, it was not a true free-form search engine.

M12A implements a **true free-form retrieval system** over the full 3,766 telecom diagram corpus.

## Difference Between M12 and M12A
- **M12**: Maps input to known query ID $\rightarrow$ Loads precomputed `M7/M5.5` JSON ranks. If you type a completely novel query not in the 50-set, it fails or routes incorrectly.
- **M12A**: Indexes the actual `All Images Path.csv` strings (captions + metadata context) $\rightarrow$ Tokenizes your exact input text $\rightarrow$ Executes live BM25/TF-IDF similarity scoring against all 3,766 diagrams dynamically. It supports new user queries.

## How the Retrieval Works
1. **Document Building**: For each of the 3,766 images, a searchable document is constructed by concatenating its `Image Caption`, `Context`, and `Image Path`.
2. **Indexing**: The script tokenizes these documents and builds a live `rank_bm25` (Okapi BM25) index in memory.
3. **Scoring**: The user query is tokenized and scored against the BM25 index, yielding dynamic Top-K rankings. Dense retrieval (BGE) is currently omitted because corpus-wide embeddings are not cached on disk for on-the-fly loading.

## How to Run One Query
You can run any novel query you want. For example:
```bash
/DATA1/prabhakar/llava_env/bin/python scripts/22_m12a_free_form_retrieval.py \
  --query "how does AMF handle roaming" \
  --top-k 5 \
  --method bm25 \
  --output-dir reports/m12a_free_form_demo_outputs
```

## How to Interpret Results
The outputs (`latest_results.csv`, `latest_results.json`, and `latest_contact_sheet.png`) display the live retrieval ranks and the BM25 relevance score.

## What Works
- True free-form text search successfully retrieves highly relevant technical diagrams using metadata context.
- It robustly handles entirely novel, unseen phrasing.
- BM25 operates fast, requiring no GPU to search the 3.7k documents.

## What is Limited
- **No Dense Fallback**: Because corpus-wide BGE embeddings are not stored, we cannot run true semantic hybrid fusion (like M7) on the fly without running the BGE transformer model dynamically (which is slow). Thus, true free-form is currently limited to lexical BM25/TF-IDF.
- **No Visual Search**: The free-form text search relies entirely on human-authored captions and spec context, not the visual pixels of the diagram.

## How LLMs Can Be Used as Candidate Judges
Just like in M12, an LLM agent can act as a **judge** for these free-form candidate sets.
- You provide Claude/Gemini with the query and the generated M12A contact sheet.
- The LLM determines if the top-5 set actually answers the user's telecom question.
- **Important constraint**: The LLM is judging the *results* of the BM25 index; it is *not* executing the search itself.

## Why This is Stronger for Thesis Defense
This demonstrates that the system isn't just an offline academic experiment (precomputed arrays mapping Q to Document). It is a functioning, research demo prototype that correctly applies the lexical baseline methodology to live, unstructured inputs. It provides evidence for the utility of the text metadata pipeline for real-world telecom retrieval tasks.
