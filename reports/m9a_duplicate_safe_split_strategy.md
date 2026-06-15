# M9A Duplicate-Safe Split Strategy

## Why Ordinary Random Split is Invalid
A simple row-based random split would accidentally place exact image duplicates (e.g., standard architecture diagrams appearing in multiple specs) in both the training set and the test set.

## How Duplicate Leakage Inflates Results
If a duplicate image is in the training set, the model memorizes its visual features and simply performs a nearest-neighbor lookup during test time on the identical image. This creates artificial 100% accuracy that does not generalize to unseen diagrams.

## Group Assignment
Duplicate groups (derived from MD5 or identical paths) are grouped together. The entire group is assigned exclusively to Train (70%), Val (10%), or Test (20%).

## Expected Row Counts
- Train: ~70% of rows
- Val: ~10% of rows
- Test: ~20% of rows

## Query Filtering
Q1, Q2, and Q3 test queries will be filtered to only include queries where the `ground_truth_row` belongs to the Test split.

## Fair Comparison
Zero-shot CLIP and adapted CLIP will be compared *only* on this exact duplicate-safe test split to ensure a rigorous, leak-free evaluation.
