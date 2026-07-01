# 30-Minute Oral Walkthrough Script

This script is designed for a 30-minute thesis defense presentation. It uses a clear, conversational tone (with optional Hinglish integration where helpful for flow) to explain the entire project.

## 0–3 min: Problem and Motivation
"Good morning everyone. Today I am presenting my work on Multimodal Retrieval of Telecom Technical Diagrams from 3GPP Standards.
The core problem is this: 3GPP standards are massive text documents, but the most critical information—how protocols work, sequence diagrams, architectures—is locked inside images and diagrams.
If an engineer wants to find the 'LTE Protocol Stack' or a specific 'Handover Failure' sequence, standard Ctrl+F doesn't work well because the evidence is visual.
*Mera goal yeh tha* to build a system that takes a user's text query and accurately retrieves the exact diagram they need from thousands of 3GPP documents."

## 3–7 min: Dataset and Query Sets
"To build this, we first needed data. I extracted 3,766 diagrams from 3GPP specifications, along with their captions and surrounding context paragraphs.
During our data audit (Milestone 1), we found a major issue: 3GPP reuses the exact same diagram in multiple places. If we didn't account for this, our metrics would unfairly penalize the model for finding a valid duplicate. So, I implemented a strict **duplicate-aware evaluation framework** using visual hashing.
Next, we needed queries. Since we didn't have real user logs, we synthesized three tiers of difficulty:
* Q1: Direct image captions (testing keyword matching).
* Q2: LLM-paraphrased queries (testing semantic understanding).
* Q3: Long context paragraphs (testing long-form document retrieval)."

## 7–12 min: Baselines and Methods
"With the data and queries ready, we established our baselines.
First, we tested pure text models: BM25 for lexical matching, and BGE for dense vector embeddings.
What we found was fascinating: BM25 is incredibly strong in this domain. Telecom language is full of hyper-specific jargon and acronyms (like 'eNodeB', 'P-GW'). Lexical models nail exact matches, whereas dense models sometimes blur these highly specific terms.
We also tested visual and multimodal models, including zero-shot CLIP and ColPali. *Surprising baat yeh thi* that these modern models struggled heavily. They are trained on natural images, not dense, abstract technical line drawings with overlapping arrows and text boxes."

## 12–18 min: Results and Ablations
"Because single modalities had weaknesses, I built a hybrid fusion system.
In Milestone 7 and 8, I ran a master ablation study. We discovered that combining BM25 and BGE scores—specifically a weight of 0.75 BM25 and 0.25 BGE—yielded the statistically highest performance for direct queries.
However, numbers only tell half the story. I generated qualitative error galleries (Milestone 9) to see *why* the system failed.
I noticed the system failed when the query asked for something like 'MSG1'. Why? Because 'MSG1' was only written *inside* the diagram's pixels, not in the caption or the context paragraph. The text backend simply couldn't 'see' it."

## 18–22 min: Final Architecture and M9A
"This led to Milestone 9A. If the visual models are weak out-of-the-box, can we adapt them?
I trained a custom projection layer on top of frozen CLIP features, using our 3GPP data with a strict, leakage-free train/test split.
The result? The projection adaptation increased visual retrieval metrics over the zero-shot CLIP baseline.
So, our final research architecture is a strong hybrid text-first backend, with a custom-adapted visual branch supporting the conclusion that domain adaptation is viable for future multimodal systems."

## 22–26 min: Product Demo M12A
"But I didn't just leave it at offline metrics. I built a live product demo (Milestone 12A).
*Ab main aapko live demo dikhata hu.*
This script takes any free-form query, runs it through our optimized BM25 text-backend against all 3,766 diagrams, and instantly generates an HTML contact sheet of the top hits.
We manually validated this demo. It works beautifully for high-level architectural queries like 'Handover sequence'.
*Ek common question aata hai:* 'Why is the BM25 score 20 for this query and 40 for another?'
BM25 scores are query-local. They depend on query length and term rarity. You cannot compare an absolute score of 40 to 20 across different queries; you can only use it to rank results for that specific search."

## 26–30 min: Limitations, Future Work, and Contribution
"To conclude, let's talk about limitations and the future.
The system is currently limited by its reliance on text metadata. Our attempts with OCR were too noisy, and the system struggles with deep message-level extraction.
*Future work* involves Diagram-grounded QA. The retrieval pipeline I've built here will serve as the first step. We will retrieve the diagram, pass it to an advanced Vision-Language Model to extract the specific evidence (like 'MSG1'), and use an LLM to generate a cited answer.
My core contribution is establishing a rigorous, empirically evaluated text-and-visual retrieval foundation on a highly technical, unstudied 3GPP dataset, creating a robust prototype ready for next-generation RAG QA."
