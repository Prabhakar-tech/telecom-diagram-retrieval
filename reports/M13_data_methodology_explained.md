# Data Methodology Explained

This document explains the foundation of the telecom retrieval project: the data. Understanding the data methodology is critical for defending the evaluation metrics and system design.

## Data Source and Extraction
* **Source:** 3GPP Technical Specifications (standards documents).
* **Extraction:** Diagrams and associated text were extracted to form a tabular dataset.
* **Size:** 3,766 images in total.
* **Storage:**
  * Metadata CSV: `/DATA1/prabhakar/telecom/All Images Path.csv`
  * Image Files: `/DATA5/prabhakar/telecom/extracted_images/images/`

## Key Metadata Columns
For every image, we have the following metadata:
* `Image Path`: The filename of the diagram.
* `Source`: The 3GPP specification number (e.g., TS 38.331).
* `Subclause`: The specific section where the image appears.
* `Image Caption`: The direct caption given to the figure in the text.
* `Context`: The surrounding paragraph text where the figure is referenced.

## The Duplicate Problem and Duplicate-Aware Evaluation
During the data audit, we discovered that 3GPP heavily reuses the exact same diagram in multiple places (different subclauses, sometimes different documents).
* Example: A standard TCP handshake diagram might appear 5 times with 5 different context paragraphs.
* **Why this matters:** If a user searches for "TCP handshake", and the model returns instance #2, but the ground truth label is instance #1, a naive evaluation script will score it as a failure (Recall=0).
* **The Fix:** We implemented a duplicate mapping dictionary based on visual hashing. If the system retrieves *any* instance of the duplicate group, it counts as a success. This is called **duplicate-aware evaluation**.

## Query Sets (Q1, Q2, Q3)
To evaluate the system, we needed queries. We synthesized three sets:
1. **Q1 (Captions):** The exact image caption. Tests direct keyword matching.
2. **Q2 (Paraphrased):** LLM-paraphrased versions of Q1. Tests semantic flexibility and robustness to synonym changes.
3. **Q3 (Context):** The surrounding context paragraph. Tests long-form query processing, simulating a user copying text from a standard to find the corresponding diagram.

## M9A Train/Val/Test Split
For the visual adaptation milestone (M9A), we had to train a model. We split the 3,766 images into training, validation, and test sets.
* **Critical Constraint:** We had to use our duplicate mapping to ensure that no image in the test set had a visual duplicate in the training set (data leakage). This is a vital point to mention during a defense to show methodological rigor.

## Summary Diagram
```
3GPP Standards  --> Extraction --> 3,766 Images + Metadata
                                      |
                                      +--> Find Duplicates --> Duplicate-Aware Eval
                                      |
                                      +--> Generate Q1/Q2/Q3 Queries
                                      |
                                      +--> Train/Test Split (Leakage Free)
```
