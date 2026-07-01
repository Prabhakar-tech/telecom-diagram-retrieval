# Senior / Professor Defense Q&A

This document prepares you for the most common and difficult questions a senior researcher or professor might ask during a thesis defense.

**1. What exactly did you build?**
I built and evaluated a multimodal retrieval pipeline for 3GPP telecom diagrams. The system takes a text query and returns the most relevant technical diagrams out of a dataset of 3,766 images, using a combination of lexical, dense, and visual models.

**2. Is this only copy-paste of open-source models?**
No. While we evaluated open-source models like BGE, CLIP, and ColPali, the core contributions are the creation of a domain-specific, duplicate-aware evaluation framework, the M7 selected fusion architectures, and the M9A custom-trained visual projection adaptation specifically for telecom line drawings.

**3. What is the dataset?**
3,766 images extracted directly from 3GPP Technical Specifications, paired with their official captions, surrounding context paragraphs, and source subclause metadata.

**4. What is Q1/Q2/Q3?**
Because we lacked real user logs, we synthesized three query sets to simulate difficulty: Q1 (direct exact captions), Q2 (LLM-paraphrased semantics), and Q3 (long-form context paragraphs).

**5. Why duplicate-aware evaluation?**
3GPP heavily reuses diagrams. If a model retrieves a visually identical diagram from a different subclause, standard metrics penalize it as a failure (Recall=0). I built a visual hashing map to ensure models are rewarded for finding *any* valid instance of the diagram.

**6. Why is BM25 so strong here?**
Telecom standard retrieval is a high-jargon, high-acronym domain. BM25 excels at exact keyword matching (e.g., "P-GW", "eNodeB"), which is often more discriminative in this domain than the smooth semantic spaces of generic dense models.

**7. Why are dense/visual models weaker?**
Generic dense text models can blur highly specific acronyms. Zero-shot visual models (like CLIP) are trained on natural photos (dogs, cars) and fail to understand the abstract, high-density text boxes and arrows of a telecom sequence diagram.

**8. What is M7?**
M7 is our selected hybrid text-fusion model. By performing a grid search, we found that a weighted combination of normalized BM25 scores (0.75) and BGE dense scores (0.25) yielded the strong evaluated performance on direct queries.

**9. What is M9A?**
M9A is our visual domain adaptation experiment. We froze a CLIP model and trained a custom projection layer using our 3GPP data (with a strict duplicate-free train/test split) to force the visual model to learn telecom diagrams, showing higher metrics than zero-shot performance.

**10. Did you train any model?**
Yes, in M9A, I trained a visual projection layer using a contrastive loss to adapt CLIP to the 3GPP domain.

**11. Does the model beat existing systems?**
We do not claim "global superiority". We claim that our evaluated hybrid text-first architecture is a strong empirical baseline for this specific 3GPP dataset, showing higher retrieval metrics than zero-shot multimodal approaches such as standard CLIP or ColPali in our evaluated setting.

**12. Why are some BM25 scores 20 and some 40?**
BM25 scores are query-local. A score of 40 on a long query with rare acronyms cannot be directly compared to a score of 20 on a short query. The absolute number doesn't matter; only the relative ranking order for that specific query matters.

**13. Why did random access MSG1-MSG4 fail?**
Our qualitative M9 gallery showed that queries relying on specific message labels (like "MSG1") fail because those terms only exist *inside* the diagram image text, not in the caption or context. Text-first retrieval misses them.

**14. What is the final architecture?**
For academic benchmarking, it's a modality-specific ensemble (M7/M5.5 fusions). For the live product demo, it is a low-latency BM25 text-first backend indexing captions and context.

**15. What is the product demo?**
M12A is a live, interactive script that takes a raw user string, runs the text-first retrieval against all 3,766 diagrams, and generates a visual HTML contact sheet of the top hits instantly.

**16. Can it answer questions?**
Not currently. It retrieves the diagram. Answering questions based on the diagram is the next logical step (Diagram-grounded QA).

**17. What are the limitations?**
The system relies heavily on the text metadata (caption/context). It struggles with deep evidence extraction directly from the pixels, and current OCR is too noisy on these specific diagrams.

**18. What is future work?**
Diagram-grounded QA. Using the retrieved diagrams as context for a Vision-Language Model (VLM) or LLM to answer specific multi-choice questions (e.g., "In this retrieved stack, what layer is above RLC?").

**19. What is the research contribution?**
A rigorous empirical evaluation of multiple retrieval modalities on a highly technical, unstudied domain; a novel duplicate-aware evaluation framework; and a domain-adapted visual projection model for telecom schemas.

**20. How will this become a QA system later?**
The retrieval system built here becomes the "R" in RAG. We will retrieve the top-k diagrams, run OCR/VLM to extract evidence, and pass that to an LLM to generate an answer and cite the supporting image.
