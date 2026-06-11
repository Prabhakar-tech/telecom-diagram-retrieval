#!/usr/bin/env python3
"""
02_query_generator.py — Milestone 1.2: Query Generation (Q1 & Q3)
==================================================================
M.Tech Thesis: "Multimodal Image Retrieval for Telecom Technical Diagrams"

Generates two foundational query sets for the ablation study:

  Q1 (Direct Caption Queries):
      The verbatim Image Caption from each row.  Every row produces exactly
      one query, so |Q1| == total rows.

  Q3 (Context-extracted Queries):
      The first substantive sentence from the Context column.
      - Rows with NaN / empty Context are skipped.
      - If the context begins with a "Figure X.Y.Z-N: …" label line,
        that label is consumed as the first sentence and the body text
        follows after the newline.  The first sentence is still used,
        because it is the figure's own descriptive title and serves as
        a valid natural-language query.
      - Sentence boundary: split on '. ' or newline, take the first
        non-empty segment ≥ 10 chars.

Outputs:
  queries/q1_captions.json
  queries/q3_context.json

Usage:
    python scripts/02_query_generator.py
"""

import json
import os
import re
import sys
import time

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
PATHS_CSV = "/DATA1/prabhakar/telecom/All Images Path.csv"
OUTPUT_DIR = "/DATA5/prabhakar/telecom_retrieval/queries/"
Q1_OUTPUT = os.path.join(OUTPUT_DIR, "q1_captions.json")
Q3_OUTPUT = os.path.join(OUTPUT_DIR, "q3_context.json")

# Minimum character length for a sentence to be considered valid
MIN_SENTENCE_LEN = 10


# =============================================================================
# Text Cleaning Utilities
# =============================================================================
def clean_text(text: str) -> str:
    """
    Normalise whitespace and strip control characters from a text string.
    Preserves meaningful content while removing artifacts from CSV parsing.
    """
    if not isinstance(text, str):
        return ""
    # Replace tabs and multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Remove carriage returns
    text = text.replace("\r", "")
    # Strip leading/trailing whitespace
    return text.strip()


def extract_first_sentence(context: str) -> str:
    """
    Extract the first substantive sentence from a Context field.

    Strategy:
      1. Split on newline first (contexts often have structure:
         "Figure label line\\nbody paragraph").
      2. Within the first non-empty segment, split on '. ' to isolate
         the first sentence.
      3. Ensure the result is at least MIN_SENTENCE_LEN characters.
      4. Re-append the trailing period if it was removed by splitting.
    """
    if not isinstance(context, str) or context.strip() == "":
        return ""

    # Split by newline, take segments
    segments = context.strip().split("\n")

    for segment in segments:
        segment = clean_text(segment)
        if len(segment) < MIN_SENTENCE_LEN:
            continue

        # Split on '. ' to get the first sentence within this segment
        sentence_parts = segment.split(". ")
        first = sentence_parts[0].strip()

        # If the split removed a trailing period from a single-sentence
        # segment, the original ended with '.'; restore it.
        if len(first) >= MIN_SENTENCE_LEN:
            # Re-add period if the original segment had one after this part
            if len(sentence_parts) > 1 and not first.endswith("."):
                first = first + "."
            elif segment.endswith(".") and not first.endswith("."):
                first = first + "."
            return first

    # Fallback: return the entire cleaned context if no valid sentence found
    cleaned = clean_text(context)
    if len(cleaned) >= MIN_SENTENCE_LEN:
        # Truncate to first 200 chars as a safety measure for very long texts
        return cleaned[:200].rsplit(" ", 1)[0] + "..."
    return ""


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("=" * 70)
    print("Milestone 1.2: Query Generation (Q1 & Q3)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load CSV
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading Paths CSV...")
    df = pd.read_csv(PATHS_CSV)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Generate Q1 — Direct Caption Queries
    # ------------------------------------------------------------------
    print("[Step 2] Generating Q1 (Direct Caption Queries)...")
    q1_queries = []
    q1_issues = []

    for idx, row in df.iterrows():
        caption = row["Image Caption"]

        # Check for problematic values
        if pd.isna(caption):
            q1_issues.append((idx, "NaN caption"))
            caption_text = ""
        else:
            caption_text = clean_text(str(caption))

        if caption_text == "":
            q1_issues.append((idx, "Empty caption after cleaning"))
            # Still include it with empty text to maintain 1:1 row mapping
            caption_text = f"[EMPTY_CAPTION_ROW_{idx}]"

        q1_queries.append({
            "query_id": f"q1_{idx}",
            "text": caption_text,
            "ground_truth_row": idx,
        })

    print(f"  Generated {len(q1_queries)} Q1 queries")
    if q1_issues:
        print(f"  WARNING: {len(q1_issues)} issues found:")
        for row_idx, issue in q1_issues[:10]:
            print(f"    Row {row_idx}: {issue}")
        if len(q1_issues) > 10:
            print(f"    ... and {len(q1_issues) - 10} more.")
    else:
        print("  No formatting issues found.")

    # Caption length statistics
    caption_lengths = [len(q["text"]) for q in q1_queries]
    print(f"  Caption length — min: {min(caption_lengths)}, "
          f"max: {max(caption_lengths)}, "
          f"mean: {sum(caption_lengths)/len(caption_lengths):.1f}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Generate Q3 — Context-extracted Queries
    # ------------------------------------------------------------------
    print("[Step 3] Generating Q3 (Context-extracted Queries)...")
    q3_queries = []
    q3_skipped = {"nan": 0, "empty": 0, "too_short": 0}

    for idx, row in df.iterrows():
        context = row["Context"]

        # Skip NaN contexts
        if pd.isna(context):
            q3_skipped["nan"] += 1
            continue

        first_sentence = extract_first_sentence(str(context))

        if first_sentence == "":
            q3_skipped["empty"] += 1
            continue

        if len(first_sentence) < MIN_SENTENCE_LEN:
            q3_skipped["too_short"] += 1
            continue

        q3_queries.append({
            "query_id": f"q3_{idx}",
            "text": first_sentence,
            "ground_truth_row": idx,
        })

    print(f"  Generated {len(q3_queries)} Q3 queries")
    print(f"  Skipped — NaN: {q3_skipped['nan']}, "
          f"Empty: {q3_skipped['empty']}, "
          f"Too short (<{MIN_SENTENCE_LEN} chars): {q3_skipped['too_short']}")
    total_skipped = sum(q3_skipped.values())
    print(f"  Total skipped: {total_skipped} "
          f"(coverage: {len(q3_queries)}/{len(df)} = "
          f"{100*len(q3_queries)/len(df):.1f}%)")

    # Sentence length statistics
    if q3_queries:
        sent_lengths = [len(q["text"]) for q in q3_queries]
        print(f"  Sentence length — min: {min(sent_lengths)}, "
              f"max: {max(sent_lengths)}, "
              f"mean: {sum(sent_lengths)/len(sent_lengths):.1f}")

    # Show samples
    print()
    print("  Sample Q3 queries:")
    for q in q3_queries[:5]:
        print(f"    [{q['query_id']}] {q['text'][:100]}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Save outputs
    # ------------------------------------------------------------------
    print("[Step 4] Saving query files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Q1
    q1_output = {
        "metadata": {
            "query_set": "Q1",
            "description": "Direct caption queries — one per image row",
            "total_queries": len(q1_queries),
            "source_csv": PATHS_CSV,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "queries": q1_queries,
    }
    with open(Q1_OUTPUT, "w") as f:
        json.dump(q1_output, f, indent=2, ensure_ascii=False)

    q1_size = os.path.getsize(Q1_OUTPUT) / 1024
    print(f"  Q1 saved: {Q1_OUTPUT} ({q1_size:.1f} KB)")

    # Q3
    q3_output = {
        "metadata": {
            "query_set": "Q3",
            "description": (
                "Context-extracted queries — first substantive sentence "
                "from the Context column"
            ),
            "total_queries": len(q3_queries),
            "skipped": q3_skipped,
            "source_csv": PATHS_CSV,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "queries": q3_queries,
    }
    with open(Q3_OUTPUT, "w") as f:
        json.dump(q3_output, f, indent=2, ensure_ascii=False)

    q3_size = os.path.getsize(Q3_OUTPUT) / 1024
    print(f"  Q3 saved: {Q3_OUTPUT} ({q3_size:.1f} KB)")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MILESTONE 1.2 SUMMARY")
    print("=" * 70)
    print(f"  Source CSV rows:    {len(df)}")
    print(f"  Q1 queries:         {len(q1_queries)}  (expected: {len(df)})")
    print(f"  Q3 queries:         {len(q3_queries)}  (from {len(df)} rows, "
          f"{total_skipped} skipped)")
    q1_ok = "PASS" if len(q1_queries) == len(df) else "FAIL"
    print(f"  Q1 count check:     {q1_ok}")
    print(f"  Q1 file:            {Q1_OUTPUT}")
    print(f"  Q3 file:            {Q3_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
