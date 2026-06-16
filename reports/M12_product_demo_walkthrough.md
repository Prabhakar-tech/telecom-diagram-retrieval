# M12 Product Demo Walkthrough

## What the Demo Does
The M12 interactive retrieval demo is a Python tool that simulates a product-like interface for searching telecom technical diagrams. It accepts a natural language query, automatically categorizes its query type, retrieves the top candidate images using the evaluated final text-first architecture (plus BM25 and BGE baselines), and renders a visual contact sheet of the results.

Because a live online index is not deployed in this environment, the demo operates in **prepared-query mode**, matching user input against a predefined 50-query validation set.

## How to Run a Query
You can run the demo by executing `scripts/20_m12_interactive_retrieval_demo.py` and providing a query string. The script uses fuzzy matching to map your query to the nearest known validated query.

### Example Commands

**1. Direct Caption-like Query (Q1)**
```bash
/DATA1/prabhakar/llava_env/bin/python scripts/20_m12_interactive_retrieval_demo.py \
  --query "handover failure call flow between UE and eNodeB" \
  --query-type auto \
  --top-k 5 \
  --output-dir reports/m12_demo_outputs
```

**2. Paraphrased User Query (Q2)**
```bash
/DATA1/prabhakar/llava_env/bin/python scripts/20_m12_interactive_retrieval_demo.py \
  --query "show me the message sequence for setting up an RRC connection" \
  --query-type auto \
  --top-k 5 \
  --output-dir reports/m12_demo_outputs
```

**3. Context-style Query (Q3)**
```bash
/DATA1/prabhakar/llava_env/bin/python scripts/20_m12_interactive_retrieval_demo.py \
  --query "The LTE user plane protocol stack consists of several layers between the UE and the eNodeB. The Packet Data Convergence Protocol (PDCP) handles IP header compression and ciphering..." \
  --query-type auto \
  --top-k 5 \
  --output-dir reports/m12_demo_outputs
```

## What Output Files are Created
Running the demo generates three artifacts in the specified `--output-dir`:
1. `latest_results.json`: A structured JSON file containing the retrieved image IDs, ranks, and caption snippets for each system.
2. `latest_results.csv`: A tabular version of the results for easy auditing.
3. `latest_contact_sheet.png`: A visual contact sheet showing the query text, the final architecture results, and the baseline results side-by-side.

## How to Read Contact Sheets
- **Header**: Shows the query text, auto-detected query type, and ID.
- **Rows**: Each row corresponds to a specific retrieval system (e.g., `final`, `BM25`, `BGE`).
- **Images**: Images are ordered from Rank 1 to Rank K (left to right). Below each image is the image ID and a snippet of its original ground-truth caption.

## How to Run 50-Query Validation
To run the full validation set across all systems and generate contact sheets for bulk review, use the validation script:
```bash
/DATA1/prabhakar/llava_env/bin/python scripts/21_m12_run_50_query_validation.py \
  --query-set reports/m12_50_query_validation_set.csv \
  --top-k 10 \
  --output-dir reports/m12_demo_outputs
```
This generates bulk CSV/JSON results and multiple contact sheets in `reports/m12_demo_outputs/m12_50_query_contact_sheets/`.

## How LLM-Assisted Review Should Be Performed
LLMs (like Claude/Gemini/ChatGPT) can be used to manually review the contact sheets. The reviewer pastes the query and the generated contact sheet into the LLM and asks it to judge the top-5 candidates. **Crucially, the LLM is only judging the retrieved set; it is not searching the corpus itself.** See `reports/M12_llm_assisted_candidate_review_protocol.md` and `reports/m12_llm_review_packet.md` for exact procedures.

## What Claim is Supported
- We have built an evaluated telecom technical diagram retrieval system.
- The system correctly processes queries, routes them to the best text-first configurations (M7, M5.5), and retrieves highly relevant technical diagrams.
- We have established a strong evaluated empirical baseline.

## What Claim is Not Supported
- **Not a global leaderboard peak performer**: We do not claim absolute peak performance against unspecified closed datasets.
- **Not an LLM-beats-text architecture**: The LLM is used as a candidate judge/reranker, not as an offline full-corpus index that replaces BM25/BGE.
- **Not true free-form online retrieval**: Due to offline infrastructure, the demo runs in prepared-query mode against the 50 validated queries.
