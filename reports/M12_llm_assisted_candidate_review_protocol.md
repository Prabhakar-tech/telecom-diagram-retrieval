# M12 LLM-Assisted Candidate Review Protocol

This document outlines the protocol for utilizing large language models (LLMs) such as Claude, Gemini, or ChatGPT as candidate judges and rerankers for the telecom technical diagram retrieval system.

## What LLMs Can Fairly Do

LLM agents can be used as **candidate reviewers** when they are provided with a bounded, retrieved context. When given the query text, top-k retrieved candidates (including image snippets/captions), and contact sheet images, an LLM can fairly:

- **Judge Relevance**: Determine which candidate in the provided set is the most relevant to the query.
- **Top-5 Evaluation**: Mark whether the top-5 candidate set contains a relevant result.
- **Compare Systems**: Assess whether the candidate set produced by the final text-first architecture is better than the set produced by BM25 or BGE for a given query.
- **Explain Reasoning**: Provide a brief, logical explanation of why a specific technical diagram is relevant or irrelevant to the telecom-specific query.

## What LLMs Cannot Fairly Do

LLMs should **not** be treated as direct full-corpus retrieval baselines unless they have access to the same 3,766-image corpus index.

Therefore, do not ask:
> "Find the correct image for 'handover failure' from the whole corpus."

This is an unfair comparison and an invalid offline benchmark because the LLM does not have searchable access to the entire proprietary corpus in the same way BM25 or BGE does. Framing an LLM as a "full retrieval baseline" without corpus access is incorrect.

## Fair Comparison Modes

To maintain strict evaluation integrity, use the following modes:

### 1. Query Rewriting Mode
- **Action**: The LLM rewrites a natural language user query (e.g., "what happens when handover fails") into a structured technical search query (e.g., "handover failure preparation LTE eNodeB").
- **Execution**: Run the text-first retrieval system on the rewritten query.
- **Evaluation**: Compare the retrieval metrics before and after the LLM rewrite.

### 2. Candidate Reranking Mode
- **Action**: Provide the LLM with the top-10 candidates retrieved by the final architecture, BM25, or BGE.
- **Execution**: Ask the LLM to rerank these 10 candidates by relevance to the query.
- **Evaluation**: Compare the MRR@10 of the raw system ranking versus the LLM-reranked list.

### 3. Candidate Judging Mode
- **Action**: Provide the LLM with the user query and a contact sheet image displaying the top-5 results.
- **Execution**: Ask the LLM to judge if a truly relevant image appears in the top-5.
- **Evaluation**: Store the judgment separately to supplement standard Recall@5 metrics, providing a human-like qualitative review of the system's output.
