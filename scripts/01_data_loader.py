#!/usr/bin/env python3
"""
01_data_loader.py — Milestone 1: Data Loading & Duplicate-Aware Mapping
========================================================================
M.Tech Thesis: "Multimodal Image Retrieval for Telecom Technical Diagrams"

This script:
  1. Loads both CSVs (paths + metadata).
  2. Resolves image paths from the CSV to the actual on-disk location.
  3. Computes MD5 content hashes of every image file to detect visual duplicates.
  4. Builds a mapping: content_hash → [row_indices] for duplicate-aware recall.
  5. Saves the mapping to eval/duplicate_mapping.json.

Usage:
    python scripts/01_data_loader.py

All paths are hardcoded for the thesis experiment environment.
"""

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
PATHS_CSV = "/DATA1/prabhakar/telecom/All Images Path.csv"
METADATA_CSV = (
    "/DATA1/prabhakar/telecom/"
    "thesis_diagram_analysis_final.xlsx - Diagram Analysis.csv"
)
IMAGES_DIR = "/DATA5/prabhakar/telecom/extracted_images/images/"
OUTPUT_DIR = "/DATA5/prabhakar/telecom_retrieval/eval/"
DUPLICATE_MAPPING_FILE = os.path.join(OUTPUT_DIR, "duplicate_mapping.json")

# MD5 read buffer size (64 KB chunks for memory-efficient hashing)
HASH_CHUNK_SIZE = 65536


# =============================================================================
# Utility Functions
# =============================================================================
def verify_paths() -> bool:
    """Verify that all required input paths exist and are readable."""
    checks = {
        "Paths CSV": PATHS_CSV,
        "Metadata CSV": METADATA_CSV,
        "Images Directory": IMAGES_DIR,
        "Output Directory": OUTPUT_DIR,
    }
    all_ok = True
    for label, path in checks.items():
        exists = os.path.exists(path)
        readable = os.access(path, os.R_OK) if exists else False
        status = "OK" if (exists and readable) else "MISSING/UNREADABLE"
        print(f"  [{status}] {label}: {path}")
        if not (exists and readable):
            all_ok = False
    return all_ok


def compute_file_md5(filepath: str) -> str:
    """Compute the MD5 hex digest of a file's content, reading in chunks."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(csv_path: str) -> str:
    """
    Resolve an image path from the CSV to its actual on-disk location.

    The CSV contains paths like '/data/all_images/image_0.png' but the actual
    files live under IMAGES_DIR. We extract the basename and join.
    """
    basename = os.path.basename(csv_path)
    return os.path.join(IMAGES_DIR, basename)


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("=" * 70)
    print("Milestone 1: Data Loading & Duplicate-Aware Mapping")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Verify paths
    # ------------------------------------------------------------------
    print("\n[Step 1] Verifying input paths...")
    if not verify_paths():
        print("\nERROR: Some paths are missing or unreadable. Aborting.")
        sys.exit(1)
    print("  All paths verified.\n")

    # ------------------------------------------------------------------
    # Step 2: Load CSVs
    # ------------------------------------------------------------------
    print("[Step 2] Loading CSVs...")
    df_paths = pd.read_csv(PATHS_CSV)
    print(f"  Paths CSV: {df_paths.shape[0]} rows, {df_paths.shape[1]} cols")
    print(f"    Columns: {list(df_paths.columns)}")

    df_meta = pd.read_csv(METADATA_CSV)
    print(f"  Metadata CSV: {df_meta.shape[0]} rows, {df_meta.shape[1]} cols")
    print(f"    Columns: {list(df_meta.columns)}")

    # Sanity check: row counts should match
    if df_paths.shape[0] != df_meta.shape[0]:
        print(
            f"\n  WARNING: Row count mismatch — "
            f"Paths({df_paths.shape[0]}) vs Metadata({df_meta.shape[0]})"
        )
    print()

    # ------------------------------------------------------------------
    # Step 3: Resolve & verify image file paths
    # ------------------------------------------------------------------
    print("[Step 3] Resolving image paths and checking file existence...")
    df_paths["resolved_path"] = df_paths["Image Path"].apply(resolve_image_path)

    missing_files = []
    for idx, row in df_paths.iterrows():
        if not os.path.isfile(row["resolved_path"]):
            missing_files.append((idx, row["resolved_path"]))

    if missing_files:
        print(f"  WARNING: {len(missing_files)} image files NOT found on disk.")
        for i, (idx, p) in enumerate(missing_files[:5]):
            print(f"    Row {idx}: {p}")
        if len(missing_files) > 5:
            print(f"    ... and {len(missing_files) - 5} more.")
    else:
        print(f"  All {df_paths.shape[0]} image files found on disk.")
    print()

    # ------------------------------------------------------------------
    # Step 4: Compute MD5 hashes for duplicate detection
    # ------------------------------------------------------------------
    print("[Step 4] Computing MD5 content hashes for all images...")
    t0 = time.time()

    hashes = []
    errors = []
    for idx, row in df_paths.iterrows():
        fpath = row["resolved_path"]
        try:
            h = compute_file_md5(fpath)
            hashes.append(h)
        except (OSError, IOError) as e:
            hashes.append(None)
            errors.append((idx, fpath, str(e)))

        # Progress indicator every 500 images
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    Hashed {idx + 1}/{df_paths.shape[0]} images ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done. Hashed {len(hashes)} images in {elapsed:.1f}s")

    if errors:
        print(f"  WARNING: {len(errors)} files could not be hashed:")
        for idx, fpath, err in errors[:5]:
            print(f"    Row {idx}: {fpath} — {err}")

    df_paths["content_md5"] = hashes
    print()

    # ------------------------------------------------------------------
    # Step 5: Build duplicate mapping (hash → list of row indices)
    # ------------------------------------------------------------------
    print("[Step 5] Building duplicate-aware mapping...")
    hash_to_rows = defaultdict(list)
    for idx, h in enumerate(hashes):
        if h is not None:
            hash_to_rows[h].append(idx)

    # Identify groups with more than one row (actual duplicates)
    total_unique_hashes = len(hash_to_rows)
    duplicate_groups = {h: rows for h, rows in hash_to_rows.items() if len(rows) > 1}
    singleton_count = total_unique_hashes - len(duplicate_groups)
    duplicate_image_count = sum(len(rows) for rows in duplicate_groups.values())

    print(f"  Total unique content hashes: {total_unique_hashes}")
    print(f"  Singleton images (no duplicates): {singleton_count}")
    print(f"  Duplicate groups: {len(duplicate_groups)}")
    print(f"  Images involved in duplicates: {duplicate_image_count}")
    print()

    # Show sample duplicate groups
    if duplicate_groups:
        print("  Sample duplicate groups (first 5):")
        for i, (h, rows) in enumerate(list(duplicate_groups.items())[:5]):
            filenames = [os.path.basename(df_paths.iloc[r]["resolved_path"]) for r in rows]
            print(f"    Hash {h[:12]}... → {len(rows)} images: {filenames}")
        print()

    # ------------------------------------------------------------------
    # Step 6: Save the full mapping (all hashes, not just duplicates)
    # ------------------------------------------------------------------
    print("[Step 6] Saving duplicate mapping...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build the output structure
    output = {
        "metadata": {
            "total_rows": df_paths.shape[0],
            "total_unique_hashes": total_unique_hashes,
            "duplicate_groups_count": len(duplicate_groups),
            "duplicate_images_count": duplicate_image_count,
            "singleton_count": singleton_count,
            "hash_algorithm": "md5",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "hash_to_row_indices": {h: rows for h, rows in hash_to_rows.items()},
    }

    with open(DUPLICATE_MAPPING_FILE, "w") as f:
        json.dump(output, f, indent=2)

    file_size_mb = os.path.getsize(DUPLICATE_MAPPING_FILE) / (1024 * 1024)
    print(f"  Saved to: {DUPLICATE_MAPPING_FILE}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MILESTONE 1 SUMMARY")
    print("=" * 70)
    print(f"  Paths CSV rows:           {df_paths.shape[0]}")
    print(f"  Metadata CSV rows:        {df_meta.shape[0]}")
    print(f"  Images on disk:           {len(os.listdir(IMAGES_DIR))}")
    print(f"  Images successfully hashed: {len(hashes) - len(errors)}")
    print(f"  Unique content hashes:    {total_unique_hashes}")
    print(f"  Duplicate groups:         {len(duplicate_groups)}")
    print(f"  Duplicate mapping saved:  {DUPLICATE_MAPPING_FILE}")
    print(f"  Hash errors:              {len(errors)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
