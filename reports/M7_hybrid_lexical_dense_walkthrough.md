# M7 Hybrid Lexical + Dense Text Retrieval Walkthrough

## 1. Objective and Hypothesis
Throughout the earlier milestones, we established two robust but distinct retrieval paradigms:
- **Lexical (BM25)**: Extremely precise for exact-match technical acronyms and short captions (Q1/Q2). Suffers heavily from token dilution on long context passages (Q3).
- **Dense Semantic (BGE-base)**: Captures intent and paraphrasing better than BM25, maintaining strong performance on context passages when combined with captions, but loses the specific weight of individual telecom jargon tokens.

**Hypothesis**: A late-fusion hybrid system combining the raw scores of BM25 and BGE will out-perform any single channel by recovering complementary strengths, resulting in the most robust overall architecture for the thesis. We also test whether incorporating domain acronym expansion (from M6.5) as a low-weight auxiliary signal provides any further lift.

## 2. Experimental Setup
The experiment (`scripts/12_hybrid_lexical_dense.py`) evaluated various configurations over the full dataset (3,766 Q1, 3,766 Q2, 3,542 Q3):

1. **Single-Channel References**: `bm25_caption`, `bm25_caption_context`, `bge_caption`, `bge_caption_context`.
2. **Reciprocal Rank Fusion (RRF)**: $k \in \{10, 30, 60\}$, varying weights between lexical and dense.
3. **Min-Max Score Fusion**: Linear combination of normalized scores with weights (0.75/0.25, 0.50/0.50, 0.25/0.75).
4. **Auxiliary Acronym Expansion**: Fusing the M6.5 predicted lists with the best base hybrid model at very low weight (0.05, 0.10).
5. **Auxiliary CLIP**: Global visual retrieval at 0.05 weight (as a diagnostic baseline).

## 3. Results Analysis

### Absolute Lift (MRR@10)
| Query Set | Best Single Channel | Hybrid Architecture | Hybrid System | Hybrid MRR@10 |
|-----------|---------------------|---------------------|---------------|---------------|
| **Q1** (Captions) | 0.890 (`bm25_caption`) | `score_fusion_bm25_075_bge_025` | Score Fusion (75% Lexical) | **0.892** (+0.002) (Standard MRR@10) |
| **Q2** (Paraphrased) | 0.861 (`bge_caption`) | `score_fusion_bm25_075_bge_025` | Score Fusion (75% Lexical) | **0.867** (+0.006) (Standard MRR@10) |
| **Q3** (Context) | 0.763 (`bm25_caption_ctx`) | `score_fusion_bm25_050_bge_050` | Score Fusion (50/50 balanced)| **0.779** (+0.016) (Standard MRR@10) |

*Note: Adding Acronym Expansion (`hybrid_plus_acronym_010`) pushed Q3 slightly higher to **0.780**, but the raw Score Fusion provides the cleanest architectural baseline.*

### Key Findings
1. **Score Fusion > Rank Fusion**: Normalizing raw retrieval scores (Min-Max) and linearly combining them yielded strictly better MRR than Reciprocal Rank Fusion (RRF). RRF discards the confidence margins between documents, whereas Score Fusion preserves the high-confidence exact-matches of BM25.
2. **Lexical Dominance on Short Queries**: For Q1 and Q2, the best fusion weight heavily favored BM25 (75% weight). The dense model (25% weight) acted purely as a safety net to catch semantic synonyms that BM25 missed.
3. **Balanced Synergy on Long Queries**: For Q3, the system required a perfect 50/50 balance of lexical and dense context matching to achieve its peak MRR of 0.779, outperforming the best BM25 model by 1.6% absolute MRR.
4. **Auxiliary Signals**: The acronym expansion from M6.5 provided a marginal +0.001 MRR lift on Q3 when applied as a tiny post-processing weight (0.10). CLIP visual retrieval degraded performance across the board, confirming visual global semantics are detrimental here.

## 4. Final Defensive Posture
M7 establishes the strongest primary text-retrieval candidate so far: Min-Max Score Fusion of BM25 and BGE-base. It improves over the best single text channel across Q1, Q2, and Q3, but final architecture selection should still be made after the master ablation and qualitative analysis stages.

By evaluating all components individually and proving that the hybrid model improves upon the single channels, the architecture is strictly evidence-derived. This directly addresses potential external examiner questions regarding why visual models are not supported as primary branches by the current evidence. Cross-encoder reranking remains useful selectively for long/context queries, but not as a universal first-stage retriever.

## 5. Artifacts Created
- `reports/m7_hybrid_lexical_dense_results.json`
- `reports/m7_expected_systems_table.csv`
- `reports/m7_per_query_win_loss.csv`
- Appended results to `reports/master_results.csv`
- Updated `THESIS_MASTER_STATE.md` and `THESIS_STAGE_DEFENSE_CARDS.md`
