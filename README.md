# Retrieval-Augmented Generation (RAG) Systems: Empirical Evaluation

This repository contains the implementation and evaluation code used in the paper:

**“Retrieval-Augmented Generation Systems: Architectures, Techniques, and Empirical Analysis”**

The codebase supports a comparative study of retrieval-augmented generation pipelines, focusing on retrieval strategies, answer quality, and latency trade-offs under controlled experimental settings.

---

## Purpose and Scope

Large language models often struggle with factual recall and long-context consistency when operating without external knowledge access. Retrieval-Augmented Generation (RAG) addresses this limitation by integrating document retrieval with generation.

The goal of this repository is to:

- Empirically evaluate RAG pipelines against a non-retrieval baseline
- Compare retrieval strategies such as BM25
- Measure both **answer quality** and **latency overhead**

This implementation is designed for **research evaluation**, not production deployment.

---

## Core Components

### Baseline (No Retrieval)

- `baseline_no_rag.py`
- Implements generation without external document retrieval
- Serves as a comparison point to isolate the effect of retrieval

### Retrieval Module

- `bm25_retriever.py`
- Implements sparse lexical retrieval using BM25
- Retrieves top-k documents per query

### Dataset and Corpus Loading

- `load_dataset.py` loads evaluation queries and references
- `load_corpus.py` loads the document collection used for retrieval

---

## Evaluation Pipeline

### Standard Evaluation

- `evaluate_rag.py`
- Compares baseline and RAG outputs
- Measures answer quality using task-specific metrics (e.g., F1)

### RAGAS-Based Evaluation

- `evaluate_ragas.py`
- Uses RAGAS-style metrics to evaluate:
  - Context relevance
  - Faithfulness
  - Answer consistency

These metrics are used to complement standard accuracy-based evaluation.

---

## Results and Visualization

- `plot.py` generates comparison plots
- `f1_latency_comparison.png` visualizes:
  - Trade-off between answer quality (F1 score)
  - Inference latency with and without retrieval

The reported results show that RAG improves factual accuracy at the cost of additional retrieval latency, consistent with findings discussed in the paper.

---

## Experimental Notes

- Retrieval parameters are kept fixed across runs
- All comparisons use the same dataset and corpus
- No fine-tuning of the language model is performed
- The focus is on **system-level behavior**, not model optimization

---

## Limitations

This repository abstracts away several real-world concerns, including:

- Dynamic document updates
- Noisy or adversarial queries
- Multi-hop reasoning across documents

Results should be interpreted as empirical observations under controlled conditions.

---

## Citation

If you use or reference this code, please cite the associated paper.

```bash
@misc{rag_systems_eval_2025,
author = {Shukla, Yashasvi},
title = {Empirical Evaluation of Retrieval-Augmented Generation Systems},
year = {2025},
url = {https://github.com/yashasvi-shukla-me/your-repo-name}
}
```

---

## License

This repository is intended for academic and research use.

YASHASVI SHUKLA (Yashasvi Shukla)

### Screenshots

<img width="1267" height="525"  src="https://github.com/user-attachments/assets/ec0c69e3-f3b9-4fa5-b969-41afe15f5178" />
<img width="874" height="438"  src="https://github.com/user-attachments/assets/826fdf85-4c70-4ddd-83ff-270af552c5d4" />
<img width="720" height="583"  src="https://github.com/user-attachments/assets/b52886a8-a3fb-4a46-8291-2529047dde12" />
<img width="1410" height="559"  src="https://github.com/user-attachments/assets/ead7d640-856a-4544-8610-230215d31335" />
<img width="954" height="648"  src="https://github.com/user-attachments/assets/1e5ea771-0a96-41dc-af81-d71747c86774" />
<img width="962" height="340"  src="https://github.com/user-attachments/assets/10583b8c-ca2d-4337-91e0-674a890987fd" />
<img width="963" height="708"  src="https://github.com/user-attachments/assets/48645fa4-4108-479a-ac09-54c16b233bd8" />
