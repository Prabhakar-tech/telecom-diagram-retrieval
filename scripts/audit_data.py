import os
import json
import hashlib
from collections import Counter, defaultdict

CSV = "/DATA1/prabhakar/telecom/All Images Path.csv"
IMG_DIR = "/DATA1/prabhakar/telecom/extracted_images/images"
OUT_DIR = "/DATA5/prabhakar/telecom_retrieval/reports"

os.makedirs(OUT_DIR, exist_ok=True)

print("Checking required paths...")
print("CSV exists:", os.path.exists(CSV), CSV)
print("Image dir exists:", os.path.exists(IMG_DIR), IMG_DIR)

# -----------------------------
# Import checks
# -----------------------------
try:
    import pandas as pd
    print("pandas: OK")
except Exception as e:
    print("pandas import failed:", repr(e))
    raise

try:
    from PIL import Image
    print("PIL/Pillow: OK")
except Exception as e:
    print("PIL/Pillow import failed:", repr(e))
    Image = None

# -----------------------------
# Read CSV robustly
# -----------------------------
read_attempts = [
    {"sep": None, "engine": "python", "encoding": "utf-8"},
    {"sep": ",", "encoding": "utf-8"},
    {"sep": None, "engine": "python", "encoding": "latin1"},
    {"sep": ",", "encoding": "latin1"},
]

df = None
last_error = None

for kwargs in read_attempts:
    try:
        df = pd.read_csv(CSV, **kwargs)
        print("CSV read successful with:", kwargs)
        break
    except Exception as e:
        last_error = e

if df is None:
    raise RuntimeError(f"Could not read CSV. Last error: {last_error}")

print("\n==============================")
print("CSV SUMMARY")
print("==============================")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:")
for i, c in enumerate(df.columns):
    print(f"  {i}: {c}")

print("\nDtypes:")
print(df.dtypes)

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False).head(30))

print("\nDuplicate full CSV rows:", int(df.duplicated().sum()))

# Save CSV preview
df.head(50).to_csv(os.path.join(OUT_DIR, "csv_preview_first_50_rows.csv"), index=False)

# -----------------------------
# Find image files
# -----------------------------
valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

image_files = []
all_files = []

for root, _, files in os.walk(IMG_DIR):
    for f in files:
        p = os.path.join(root, f)
        all_files.append(p)
        if os.path.splitext(f)[1].lower() in valid_exts:
            image_files.append(p)

print("\n==============================")
print("IMAGE FOLDER SUMMARY")
print("==============================")
print("Total files:", len(all_files))
print("Total image files:", len(image_files))

ext_counts = Counter(os.path.splitext(p)[1].lower() for p in all_files)
print("\nExtension counts:")
for ext, cnt in ext_counts.most_common(30):
    print(ext, cnt)

# -----------------------------
# Detect possible path columns
# -----------------------------
print("\n==============================")
print("POSSIBLE CSV IMAGE PATH COLUMNS")
print("==============================")

possible_path_cols = []
path_like_examples = {}

for col in df.columns:
    vals = df[col].dropna().astype(str).head(1000).tolist()
    hits = []
    for v in vals:
        lv = v.lower()
        if any(lv.endswith(ext) for ext in valid_exts) or "/images" in lv or "extracted_images" in lv:
            hits.append(v)
    if hits:
        possible_path_cols.append(col)
        path_like_examples[col] = hits[:10]

print("Possible path columns:", possible_path_cols)
for col, examples in path_like_examples.items():
    print(f"\nColumn: {col}")
    for ex in examples[:5]:
        print("  ", ex)

# -----------------------------
# Check if CSV paths exist
# -----------------------------
print("\n==============================")
print("CSV PATH EXISTENCE CHECK")
print("==============================")

path_existence_report = {}

for col in possible_path_cols:
    vals = df[col].dropna().astype(str).tolist()
    checked = 0
    exists = 0
    missing_examples = []
    existing_examples = []

    for v in vals[:10000]:
        candidates = []

        # direct path
        candidates.append(v)

        # relative to image dir
        candidates.append(os.path.join(IMG_DIR, v))

        # basename inside image dir
        candidates.append(os.path.join(IMG_DIR, os.path.basename(v)))

        found = False
        for c in candidates:
            if os.path.exists(c):
                found = True
                break

        checked += 1
        if found:
            exists += 1
            if len(existing_examples) < 5:
                existing_examples.append(v)
        else:
            if len(missing_examples) < 10:
                missing_examples.append(v)

    path_existence_report[col] = {
        "checked_first_n": checked,
        "exists": exists,
        "missing": checked - exists,
        "existing_examples": existing_examples,
        "missing_examples": missing_examples,
    }

    print(f"\nColumn: {col}")
    print("Checked:", checked)
    print("Exists:", exists)
    print("Missing:", checked - exists)
    print("Existing examples:", existing_examples[:3])
    print("Missing examples:", missing_examples[:3])

# -----------------------------
# Image dimension + corruption check
# -----------------------------
print("\n==============================")
print("IMAGE DIMENSION/CORRUPTION SAMPLE")
print("==============================")

image_stats = {
    "sampled": 0,
    "bad": [],
    "widths": [],
    "heights": [],
    "modes": Counter(),
}

if Image is not None:
    sample_limit = min(len(image_files), 5000)

    for p in image_files[:sample_limit]:
        try:
            with Image.open(p) as im:
                image_stats["sampled"] += 1
                image_stats["widths"].append(im.width)
                image_stats["heights"].append(im.height)
                image_stats["modes"][im.mode] += 1
        except Exception as e:
            if len(image_stats["bad"]) < 50:
                image_stats["bad"].append({"path": p, "error": repr(e)})

    if image_stats["widths"]:
        print("Sampled images:", image_stats["sampled"])
        print("Bad images sample count:", len(image_stats["bad"]))
        print("Width min/max:", min(image_stats["widths"]), max(image_stats["widths"]))
        print("Height min/max:", min(image_stats["heights"]), max(image_stats["heights"]))
        print("Modes:", dict(image_stats["modes"]))
        print("First bad image examples:", image_stats["bad"][:5])
    else:
        print("No image dimensions collected.")
else:
    print("Skipping image dimension check because Pillow is unavailable.")

# -----------------------------
# Duplicate image hash check on sample
# -----------------------------
print("\n==============================")
print("DUPLICATE IMAGE HASH SAMPLE")
print("==============================")

hash_counter = Counter()
hash_examples = defaultdict(list)
hash_sample_limit = min(len(image_files), 3000)

for p in image_files[:hash_sample_limit]:
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        hash_counter[digest] += 1
        if len(hash_examples[digest]) < 3:
            hash_examples[digest].append(p)
    except Exception:
        pass

duplicate_hashes = {h: c for h, c in hash_counter.items() if c > 1}
print("Hash sample checked:", hash_sample_limit)
print("Duplicate hashes found in sample:", len(duplicate_hashes))

dup_examples = []
for h, c in list(duplicate_hashes.items())[:10]:
    dup_examples.append({
        "hash": h,
        "count": c,
        "examples": hash_examples[h],
    })
print("Duplicate examples:", dup_examples[:3])

# -----------------------------
# Text/context richness
# -----------------------------
print("\n==============================")
print("TEXT CONTEXT RICHNESS")
print("==============================")

text_col_report = {}

for col in df.columns:
    vals = df[col].dropna().astype(str)
    if len(vals) == 0:
        continue

    avg_len = vals.head(5000).map(len).mean()
    max_len = vals.head(5000).map(len).max()
    sample_values = vals.head(5).tolist()

    if avg_len > 10 or max_len > 50:
        text_col_report[col] = {
            "non_null": int(vals.shape[0]),
            "avg_length_first_5000": float(avg_len),
            "max_length_first_5000": int(max_len),
            "examples": sample_values,
        }

for col, info in text_col_report.items():
    print(f"\nColumn: {col}")
    print("Non-null:", info["non_null"])
    print("Avg length:", round(info["avg_length_first_5000"], 2))
    print("Max length:", info["max_length_first_5000"])
    print("Examples:")
    for ex in info["examples"][:3]:
        print("  ", ex[:300].replace("\n", " "))

# -----------------------------
# Final JSON report
# -----------------------------
final_report = {
    "csv_path": CSV,
    "image_dir": IMG_DIR,
    "csv_rows": int(len(df)),
    "csv_columns": list(map(str, df.columns)),
    "duplicate_csv_rows": int(df.duplicated().sum()),
    "total_files": int(len(all_files)),
    "total_image_files": int(len(image_files)),
    "extension_counts": dict(ext_counts),
    "possible_path_columns": possible_path_cols,
    "path_like_examples": path_like_examples,
    "path_existence_report": path_existence_report,
    "image_dimension_sample": {
        "sampled": image_stats["sampled"],
        "bad_sample_count": len(image_stats["bad"]),
        "bad_examples": image_stats["bad"][:20],
        "min_width": min(image_stats["widths"]) if image_stats["widths"] else None,
        "max_width": max(image_stats["widths"]) if image_stats["widths"] else None,
        "min_height": min(image_stats["heights"]) if image_stats["heights"] else None,
        "max_height": max(image_stats["heights"]) if image_stats["heights"] else None,
        "modes": dict(image_stats["modes"]),
    },
    "duplicate_hash_sample": {
        "checked": hash_sample_limit,
        "duplicate_hash_count": len(duplicate_hashes),
        "duplicate_examples": dup_examples,
    },
    "text_column_report": text_col_report,
}

out_json = os.path.join(OUT_DIR, "data_audit_report.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

print("\n==============================")
print("SAVED REPORTS")
print("==============================")
print(out_json)
print(os.path.join(OUT_DIR, "csv_preview_first_50_rows.csv"))
