#!/usr/bin/env python3
"""
03_q2_paraphraser.py — Milestone 1.3: Q2 Paraphrase Generation (Local LLM)
============================================================================
M.Tech Thesis: "Multimodal Image Retrieval for Telecom Technical Diagrams"

Uses a local 7B instruct model (Qwen2.5-7B-Instruct) on the A40 GPU to
convert direct captions (Q1) into natural-language engineer questions (Q2).

Features:
  - Loads model in bfloat16 for A40 memory efficiency (~14 GB VRAM).
  - Batch-saves progress every N queries to prevent data loss on crash.
  - Resumes from the last checkpoint if the script is re-run.
  - Uses tqdm for progress tracking.

Usage:
    python scripts/03_q2_paraphraser.py
    CUDA_VISIBLE_DEVICES=1 python scripts/03_q2_paraphraser.py  # use GPU 1
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE = "/DATA5/prabhakar/hf_cache"

Q1_INPUT = "/DATA5/prabhakar/telecom_retrieval/queries/q1_captions.json"
Q2_OUTPUT = "/DATA5/prabhakar/telecom_retrieval/queries/q2_paraphrased.json"
Q2_CHECKPOINT = "/DATA5/prabhakar/telecom_retrieval/queries/.q2_checkpoint.json"

# How often to save progress (every N queries)
SAVE_EVERY = 100

# Generation parameters
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9

SYSTEM_PROMPT = (
    "You are a technical telecom expert. Convert the provided 3GPP technical "
    "diagram caption into a natural language question that an engineer would "
    "ask to search for this specific diagram. Do NOT add extra information. "
    "Keep it to one single question. Keep all acronyms intact."
)


# =============================================================================
# Utility Functions
# =============================================================================
def load_checkpoint() -> dict:
    """Load previously generated Q2 queries from checkpoint."""
    if os.path.exists(Q2_CHECKPOINT):
        with open(Q2_CHECKPOINT, "r") as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {len(data)} queries already done.")
        return data
    return {}


def save_checkpoint(results: dict):
    """Save current progress to checkpoint file."""
    with open(Q2_CHECKPOINT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def clean_response(text: str) -> str:
    """
    Clean the model's generated text to extract just the question.
    Removes common artifacts like quotes, leading/trailing whitespace,
    and prefixes like "Question:" or "Q:".
    """
    text = text.strip()

    # Remove wrapping quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Remove common prefixes
    for prefix in ["Question:", "Q:", "question:", "q:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Take only the first line if multiple lines were generated
    lines = text.strip().split("\n")
    text = lines[0].strip()

    # Ensure it ends with a question mark
    if text and not text.endswith("?"):
        # If it ends with a period, replace with question mark
        if text.endswith("."):
            text = text[:-1] + "?"
        else:
            text = text + "?"

    return text


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("=" * 70)
    print("Milestone 1.3: Q2 Paraphrase Generation (Local LLM)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load Q1 queries
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading Q1 queries...")
    with open(Q1_INPUT, "r") as f:
        q1_data = json.load(f)
    q1_queries = q1_data["queries"]
    print(f"  Loaded {len(q1_queries)} Q1 queries.")

    # ------------------------------------------------------------------
    # Step 2: Load checkpoint (resume support)
    # ------------------------------------------------------------------
    print("\n[Step 2] Checking for existing checkpoint...")
    checkpoint = load_checkpoint()
    already_done = set(checkpoint.keys())
    remaining = [q for q in q1_queries if str(q["ground_truth_row"]) not in already_done]
    print(f"  Already completed: {len(already_done)}")
    print(f"  Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("\n  All queries already paraphrased! Skipping to save step.")
    else:
        # ------------------------------------------------------------------
        # Step 3: Load model
        # ------------------------------------------------------------------
        print(f"\n[Step 3] Loading model: {MODEL_ID}")
        print(f"  HF cache: {HF_CACHE}")
        print(f"  Device: cuda")
        print(f"  Precision: bfloat16")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")

        # Report GPU memory usage
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024**3)
            if alloc > 0.1:
                print(f"  GPU {i}: {alloc:.1f} GB allocated")

        # ------------------------------------------------------------------
        # Step 4: Generate paraphrases
        # ------------------------------------------------------------------
        print(f"\n[Step 4] Generating Q2 paraphrases for {len(remaining)} queries...")
        print(f"  Saving checkpoint every {SAVE_EVERY} queries")

        t0 = time.time()
        for i, q in enumerate(tqdm(remaining, desc="Q2 Paraphrasing", unit="query")):
            caption = q["text"]
            row_idx = q["ground_truth_row"]

            # Build the chat-format prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Caption: {caption} -> Question:"},
            ]

            # Tokenize using chat template
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            clean_text = clean_response(raw_text)

            # Store result
            checkpoint[str(row_idx)] = {
                "query_id": f"q2_{row_idx}",
                "text": clean_text,
                "ground_truth_row": row_idx,
                "raw_response": raw_text.strip(),
            }

            # Periodic checkpoint save
            if (i + 1) % SAVE_EVERY == 0:
                save_checkpoint(checkpoint)
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                tqdm.write(
                    f"  Checkpoint saved: {len(checkpoint)}/{len(q1_queries)} "
                    f"({rate:.1f} q/s, ETA: {eta/60:.1f} min)"
                )

        # Final checkpoint save
        save_checkpoint(checkpoint)
        elapsed = time.time() - t0
        print(f"\n  Generation complete: {len(remaining)} queries in {elapsed:.1f}s "
              f"({len(remaining)/elapsed:.1f} q/s)")

    # ------------------------------------------------------------------
    # Step 5: Save final Q2 JSON
    # ------------------------------------------------------------------
    print("\n[Step 5] Saving final Q2 JSON...")

    # Sort by row index and build output structure
    q2_queries = []
    for row_idx in sorted(checkpoint.keys(), key=int):
        entry = checkpoint[row_idx]
        q2_queries.append({
            "query_id": entry["query_id"],
            "text": entry["text"],
            "ground_truth_row": entry["ground_truth_row"],
        })

    q2_output = {
        "metadata": {
            "query_set": "Q2",
            "description": (
                "Paraphrased queries — LLM-generated natural-language questions "
                "from Q1 captions"
            ),
            "total_queries": len(q2_queries),
            "model_used": MODEL_ID,
            "generation_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "precision": "bfloat16",
            },
            "system_prompt": SYSTEM_PROMPT,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "queries": q2_queries,
    }

    with open(Q2_OUTPUT, "w") as f:
        json.dump(q2_output, f, indent=2, ensure_ascii=False)

    q2_size = os.path.getsize(Q2_OUTPUT) / 1024
    print(f"  Saved: {Q2_OUTPUT} ({q2_size:.1f} KB)")

    # ------------------------------------------------------------------
    # Step 6: Validation & samples
    # ------------------------------------------------------------------
    print("\n[Step 6] Validation & samples...")
    print(f"  Total Q2 queries: {len(q2_queries)}")
    print(f"  Expected: {len(q1_queries)}")
    status = "PASS" if len(q2_queries) == len(q1_queries) else "MISMATCH"
    print(f"  Count check: {status}")

    # Show random samples
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(q2_queries)), min(5, len(q2_queries)))
    print("\n  Random Caption → Question samples:")
    print("  " + "-" * 60)
    for idx in sample_indices:
        q2 = q2_queries[idx]
        row = q2["ground_truth_row"]
        q1_text = q1_queries[row]["text"]
        print(f"  [{q2['query_id']}]")
        print(f"    Caption: {q1_text}")
        print(f"    Question: {q2['text']}")
        print()

    # Check for quality issues
    empty_count = sum(1 for q in q2_queries if len(q["text"].strip()) < 5)
    no_question_mark = sum(1 for q in q2_queries if not q["text"].strip().endswith("?"))
    print(f"  Quality checks:")
    print(f"    Empty/very short (<5 chars): {empty_count}")
    print(f"    Missing question mark: {no_question_mark}")

    # Length stats
    lengths = [len(q["text"]) for q in q2_queries]
    print(f"    Question length — min: {min(lengths)}, "
          f"max: {max(lengths)}, mean: {sum(lengths)/len(lengths):.1f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MILESTONE 1.3 SUMMARY")
    print("=" * 70)
    print(f"  Model: {MODEL_ID}")
    print(f"  Q2 queries generated: {len(q2_queries)}")
    print(f"  Count check: {status}")
    print(f"  Output: {Q2_OUTPUT}")
    print("=" * 70)

    # Cleanup checkpoint if everything looks good
    if len(q2_queries) == len(q1_queries) and os.path.exists(Q2_CHECKPOINT):
        os.remove(Q2_CHECKPOINT)
        print("  Checkpoint file cleaned up.")


if __name__ == "__main__":
    main()
