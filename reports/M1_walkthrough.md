# Milestone 1 Walkthrough: System Setup, Data Loader & Query Generation

> **Milestones covered**: M1.0 (Environment), M1.1 (Data Loader), M1.2 (Q1 & Q3 Queries), M1.3 (Q2 Paraphrasing & Knowledge Base)
> **Date**: 2026-06-12

---

## 1. Environment Audit (`/DATA1/prabhakar/llava_env`)

| Package | Status | Version |
|---------|--------|---------|
| `pandas` | ✅ Installed | 2.3.3 |
| `torch` | ✅ Installed | 2.11.0 |
| `transformers` | ✅ Installed | 5.5.4 |
| `sentence-transformers` | ✅ Installed | 5.4.1 |
| `faiss-gpu` | ✅ Installed | 1.7.2 |
| `Pillow` | ✅ Installed | 12.1.1 |
| `torchvision` | ✅ Installed | 0.26.0 |
| `rank_bm25` | ✅ Installed (M1.2) | 0.2.2 |
| `imagehash` | ✅ Installed (M1.2) | 4.3.2 |
| `PyWavelets` | ✅ Installed (dep) | 1.8.0 |

> [!NOTE]
> All required packages are now installed. `rank_bm25` and `imagehash` were added during Milestone 1.2.

---

## 2. Directory Structure

All requested directories created under `/DATA5/prabhakar/telecom_retrieval/`:

```
telecom_retrieval/
├── cache/
├── eval/                  ← duplicate_mapping.json lives here
├── indexes/
├── logs/
├── metadata/
├── models/
├── notebooks/
├── queries/
├── reports/
└── scripts/
    └── 01_data_loader.py
```

---

## 3. Path Verification

| Resource | Path | Status |
|----------|------|--------|
| Paths CSV | `/DATA1/prabhakar/telecom/All Images Path.csv` | ✅ 3,766 rows |
| Metadata CSV | `/DATA1/prabhakar/telecom/thesis_diagram_analysis_final.xlsx - Diagram Analysis.csv` | ✅ 3,766 rows |
| Images Directory | `/DATA5/prabhakar/telecom/extracted_images/images/` | ✅ 3,767 files |
| CSV ↔ Disk Match | All 3,766 CSV entries found on disk | ✅ 0 missing |

> [!NOTE]
> The images directory has 3,767 files vs. 3,766 CSV rows — one extra file on disk not referenced in the CSV. This is harmless.

---

## 4. Data Loader Script

**File**: `scripts/01_data_loader.py`

### CSV Schema Summary

**Paths CSV** (5 columns):
- `Context` — Full textual context from the 3GPP spec
- `Source` — Spec document identifier (e.g., "TS 25.469")
- `Subclause` — Section number
- `Image Path` — Original path (needs remapping to disk)
- `Image Caption` — Short caption

**Metadata CSV** (18 columns):
- `Image Name` — Filename (e.g., "image_0.png")
- `Matched Category` — Diagram type (Call Ladder, etc.)
- `Agent 2–4: Components/Relationships/MCQ Description` — Structured analysis
- `QA:` columns — Quality scores
- `Status`, `Error Details`

---

## 5. Duplicate Detection Results

Used **MD5 content hashing** (identical bytes → identical hash) on all 3,766 images.

| Metric | Value |
|--------|-------|
| Total images hashed | 3,766 |
| Unique content hashes | 3,458 |
| Singleton images (unique) | 3,177 |
| Duplicate groups | **281** |
| Images involved in duplicates | **589** |
| Hash errors | 0 |
| Hashing time | 1.3 seconds |

### Duplicate Group Size Distribution

| Group Size | Count |
|------------|-------|
| 2 images | 260 |
| 3 images | 16 |
| 4 images | 4 |
| 5 images | 1 |

> [!TIP]
> **281 duplicate groups** is significant (~15.6% of all images are duplicates). This validates the need for Duplicate-Aware Recall in the evaluation framework — a query matching any duplicate should credit recall for all members of that group.

### Output

**File**: `eval/duplicate_mapping.json` (0.20 MB)

Structure:
```json
{
  "metadata": {
    "total_rows": 3766,
    "total_unique_hashes": 3458,
    "duplicate_groups_count": 281,
    "duplicate_images_count": 589,
    "singleton_count": 3177,
    "hash_algorithm": "md5"
  },
  "hash_to_row_indices": {
    "<md5_hash>": [row_idx_1, row_idx_2, ...],
    "...": "..."
  }
}
```

---

## 6. Query Generation (Milestone 1.2)

**Script**: `scripts/02_query_generator.py`

### Q1 — Direct Caption Queries

| Metric | Value |
|--------|-------|
| Total queries | **3,766** (✅ matches row count) |
| Formatting issues | 0 |
| Caption length (min / mean / max) | 4 / 52.5 / 94 chars |

**Output**: `queries/q1_captions.json` (526 KB)

Sample:
```json
{"query_id": "q1_0", "text": "TNL Update: Unsuccessful operation.", "ground_truth_row": 0}
```

### Q3 — Context-extracted Queries

| Metric | Value |
|--------|-------|
| Total queries | **3,542** (94.1% coverage) |
| Skipped — NaN context | 221 |
| Skipped — Empty after cleaning | 3 |
| Sentence length (min / mean / max) | 15 / 133.4 / 566 chars |

**Output**: `queries/q3_context.json` (776 KB)

Sample:
```json
{"query_id": "q3_0", "text": "Figure 8.9.3-1: TNL Update: Unsuccessful operation.", "ground_truth_row": 0}
```

> [!TIP]
> Q3 extracts the first substantive sentence from the Context column. ~27% of contexts begin with a `Figure X.Y.Z:` label (which doubles as a descriptive title), while the rest begin directly with body text. Both are valid natural-language queries for retrieval.

---

## 7. Q2 Paraphrase Generation (Milestone 1.3)

**Script**: `scripts/03_q2_paraphraser.py`

### Model & Generation Config

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Precision | `bfloat16` |
| GPU | NVIDIA A40 (GPU 1), 14.2 GB VRAM used |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Total time | 1,981s (~33 min) at 1.9 q/s |

### Q2 — Paraphrased Queries

| Metric | Value |
|--------|-------|
| Total queries | **3,766** (✅ matches Q1 count) |
| Empty/very short (<5 chars) | 0 |
| Missing question mark | 0 |
| Question length (min / mean / max) | 35 / 81.0 / 179 chars |

**Output**: `queries/q2_paraphrased.json` (631 KB)

### Sample Caption → Question Conversions

| Query ID | Caption (Q1) | Paraphrased Question (Q2) |
|----------|-------------|---------------------------|
| `q2_2619` | Passive join to MCVideo group communication | What is the process for a passive join in MCVideo group communication? |
| `q2_456` | AMF re-allocation | What is the process or scenario described when the AMF re-allocation occurs? |
| `q2_102` | Handling of Resource Conflict | What is the process for handling resource conflicts in 3GPP standards? |
| `q2_3037` | Providing data for a user entering an ongoing MCData group conversation | What diagram shows the process of providing data for a user joining an ongoing MCData group conversation? |
| `q2_1126` | SCTP server-side illustration for SCTP Multiplexer (port) | What is the SCTP server-side illustration showing for the SCTP Multiplexer (port)? |

---

## 8. Knowledge Base & State Files (Milestone 1.3)

| File | Purpose |
|------|---------|
| `THESIS_KNOWLEDGE_BASE.md` | Canonical reference for domain, baselines, metrics, hardware constraints |
| `THESIS_MASTER_STATE.md` | Milestone progress tracker and active artifacts list |

---

### Directory Tree (End of Milestone 1)

```
telecom_retrieval/
├── THESIS_KNOWLEDGE_BASE.md
├── THESIS_MASTER_STATE.md
├── cache/
├── eval/
│   └── duplicate_mapping.json
├── indexes/
├── logs/
├── metadata/
├── models/
├── notebooks/
├── queries/
│   ├── q1_captions.json
│   ├── q2_paraphrased.json
│   └── q3_context.json
├── reports/
│   └── M1_walkthrough.md        ← this file
└── scripts/
    ├── 01_data_loader.py
    ├── 02_query_generator.py
    └── 03_q2_paraphraser.py
```
