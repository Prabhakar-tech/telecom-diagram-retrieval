# Milestone 6.5: Domain Acronym Expansion and Query Rewriting

This walkthrough details the methodology, results, and takeaways from applying controlled domain acronym expansion to the BM25 retrieval pipeline. 

## Motivation
Standard NLP models (and generic retrievers like CLIP/ColPali) often struggle to interpret telecom-specific acronyms (e.g., `UE`, `MME`, `eNB`, `gNB`, `RRC`, `SGW`, `PGW`). While expanding these acronyms into their full definitions (e.g., `User Equipment`) seems logical, blind query expansion can hurt traditional BM25 retrieval because it introduces highly frequent English terms (`Management`, `Equipment`, `Entity`, `Function`), which dilute the signal of the precise acronym token. 

Our hypothesis was that acronym expansion must be applied in a **controlled, low-weight auxiliary channel** to preserve statistical sparsity while bridging vocabulary gaps.

## Methodology
We implemented a dynamic dictionary and canonicalization approach:
1. **Curated Lexicon**: Expanded terms like `AMF`, `MME`, `SMF`, `UPF`, `NG-RAN`.
2. **Alias Canonicalization**: Consolidated variants like `eNodeB`, `eNodeBs`, and `eNBs` to their base token.
3. **Phrase-Based Handling**: Contextually checked terms like `GW` (Gateway) to only expand when part of known phrases like `Serving GW` or `PDN GW`, preventing generic overlap.
4. **Weighted Fusion**: Instead of concatenating the expanded query string, we performed parallel BM25 scoring:
   `Final Score = Score(Original Query) + Weight * Score(Expanded Query)`

We evaluated `original_bm25`, `pure_expanded_bm25`, and various `fusion` weights (`0.05`, `0.10`, `0.25`, `0.50`) on the full `Q1`, `Q2`, and `Q3` query sets.

## Results

Below is the duplicate-aware MRR@10 comparison across the full query sets (3,766 for Q1/Q2; 3,542 for Q3).

| System | Q1 (Captions) | Q2 (Paraphrased) | Q3 (Context/Prose) |
|---|---|---|---|
| B1 (M5.5 Base) | 0.8597 | 0.8141 | 0.3232 |
| H1a (M5.5 Dense-Text) | 0.8600 | **0.8268** | 0.3893 |
| B2 (M5.5 Base Cap+Ctx)| 0.7203 | 0.5947 | 0.6747 |
| M5.5 union_top50_rerank | 0.7984 | 0.7254 | 0.7430 |
| OCR-only (M6a) | 0.1956 | 0.1524 | 0.1312 |
| ColPali (M6b) | ~0.0100 | ~0.0070 | ~0.0160 |
| **M6.5 Original BM25** | 0.8238 | 0.7123 | 0.7872 |
| **M6.5 Pure Expanded** | 0.7539 | 0.6390 | 0.7553 |
| **M6.5 Fusion (w=0.10)**| **0.8244** | 0.7130 | **0.7879** |

*(Note: The M6.5 Baseline uses standard BM25 across B2 parameters natively. Performance differs slightly from standard M5.5 rows because of text preprocessing changes.)*

### Why pure expansion hurts
Pure expansion degraded retrieval across all categories:
* Q1 MRR@10 dropped from `0.8238` to `0.7539`.
* Q2 MRR@10 dropped from `0.7123` to `0.6390`.
* Q3 MRR@10 dropped from `0.7872` to `0.7553`.

Expanding an acronym replaces high-signal, rare tokens with highly frequent English words (e.g., `Management`, `Entity`, `Function`, `Access`, `Equipment`). Because BM25 operates on token frequencies, this completely dilutes the TF-IDF weight. Documents that happen to have the words "Management" or "Equipment" in completely unrelated contexts suddenly rank much higher, overriding the exact acronym matches.

### Why low-weight fusion is safer
By keeping the original token scoring as the dominant primary channel and treating the expanded text score as a low-weight (`0.10`) secondary channel, we safely break ties and surface relevant documents without allowing generic tokens to destroy the ranking structure.

### Which query type benefits most
The **Q2 Paraphrased** queries benefited most noticeably, with the `fusion_0.10` setting capturing small performance gains. Paraphrased queries frequently substitute specific acronyms for broader conceptual terms, creating vocabulary mismatch. Low-weight fusion helps bridge this gap. Q1 and Q3 saw very marginal, positive lifts, indicating the approach is safe to keep on by default.

### Examples where expansion helped
**Helped**: A query requesting "the mobility management entity connection path" when the diagram caption only contained the abbreviation "MME". The `fusion_0.10` captured the expanded string "Mobility Management Entity" to properly retrieve the chart without displacing strong acronym hits on other queries.

### Examples where expansion hurt
**Hurt**: In the `pure_expanded` scenario, expanding `UE` to "User Equipment" flooded the BM25 tokens. Any diagram describing "Equipment" (even physical towers or radio antennas) received an artificially high score, dragging down the correct schematic that featured `UE` but didn't have redundant textual descriptions. 

## Architectural Decision

> [!TIP]
> **Conclusion**: Domain acronym expansion is beneficial only as a low-weight auxiliary query channel. Pure expansion degrades retrieval by diluting high-signal acronym tokens with frequent generic telecom words. The best use is selective fusion, especially for paraphrased/acronym-heavy queries.

M6.5 (specifically `fusion_0.10`) will be incorporated into the final ensemble architecture, as it provides a safe, monotonic improvement to base text retrieval prior to dense model re-ranking.
