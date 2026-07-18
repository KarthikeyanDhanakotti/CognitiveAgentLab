# AdvancedRAG_with_RAGAS

Naive vs Advanced RAG on the *"Attention Is All You Need"* paper, graded end-to-end with **RAGAS** — plus a full teaching pack (docs, cell-by-cell walkthrough, standalone scripts, Colab launcher).

## Contents

```
AdvancedRAG_with_RAGAS/
├── RAG_AdvancedRAG.ipynb            # Main workshop notebook (naive + advanced + RAGAS)
├── Session2_Evaluation_HandsOn.ipynb# Eval-focused notebook: BLEU/ROUGE fail demo,
│                                    #   framework landscape, LLM-as-Judge, RAGAS, G-Eval
├── Run_On_Colab.ipynb               # 1-click launcher: writes all scripts to /content
│                                    #   and runs the full eval in Google Colab
├── docs/
│   ├── RAG_Notebook_CellByCell.md   # Every cell of the notebook, explained
│   ├── RAG_Algorithms_DeepDive.md   # Theory: embeddings, ANN, BM25, RRF, RAGAS,
│   │                                #   LLM-as-Judge, end-to-end query flow
│   └── RAG_Workshop_Narration_Guide.md
└── scripts/
    ├── common.py                    # Shared: key loader · Groq client · PDF ·
    │                                #   chunker · FAISS · BM25 preproc · prompt
    ├── naive_rag.py                 # NaiveRAG class + CLI demo
    ├── advanced_rag.py              # AdvancedRAG (Hybrid + RRF + Cross-Encoder rerank)
    ├── evaluate.py                  # RAGAS eval → CSV + PNG bar chart
    ├── requirements.txt
    └── README.md                    # Setup + run instructions
```

## What each pipeline demonstrates

| Pipeline | Retrieval | Reranking | Typical RAGAS lift |
|---|---|---|---|
| **Naive RAG** | FAISS (dense only) | none | baseline |
| **Advanced RAG** | FAISS + BM25 with **RRF** fusion | **Cross-Encoder** (ms-marco MiniLM) | **+50–150 %** on faithfulness & context precision |

The classic failing query — *"Who are the authors of this paper?"* — is unrecoverable by dense embeddings alone (proper nouns have no semantic neighbourhood) but is trivially caught by BM25. The Advanced pipeline shows exactly how the two complementary retrievers + reranking produce the measured lift.

## Quick start — 3 paths

### 1. Colab (easiest)
Open [`Run_On_Colab.ipynb`](Run_On_Colab.ipynb) in Google Colab, add `GROQ_API_KEY` to Colab Secrets, then **Runtime → Run all**.

### 2. Local scripts
```powershell
cd scripts
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY = "gsk_..."
python evaluate.py
```

### 3. Local notebook
Open `RAG_AdvancedRAG.ipynb` in VS Code / Jupyter and run top-to-bottom. Grab a free key from <https://console.groq.com/keys>.

## Evaluation stack

- **Judge LLM:** Groq Llama-3.1-8B (free tier, generous quota); swap to Llama-3.3-70B for stricter grading
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **RAGAS metrics:** Faithfulness · Answer Relevancy · Context Precision · Context Recall
- **Extended coverage in the notebook:** BLEU/ROUGE-fail live demo · G-Eval custom rubric · full framework landscape (RAGAS · G-Eval · DeepEval · TruLens · Phoenix · ARES · LangSmith · OpenAI Evals) · when-to-use / not-to-use LLM-as-Judge

## Full theory & rationale

See [`docs/RAG_Algorithms_DeepDive.md`](docs/RAG_Algorithms_DeepDive.md) for a top-to-bottom explanation covering:

- End-to-end query flow (plain LLM → Naive RAG → Advanced RAG) with latency breakdown
- Chunking strategy trade-offs
- What embeddings are, how they're trained, and 10 sub-sections on flavours, dims, geometry
- Cosine vs Euclidean vs dot product
- FAISS + a full tour of ANN algorithms (Flat, IVF, HNSW, PQ, LSH, ScaNN, DiskANN)
- Why BM25 (and not TF-IDF)
- RRF vs weighted score sum
- Cross-encoder vs bi-encoder rerank
- Every RAGAS metric — formulas, worked examples, interpretation thresholds
- When to (not) use LLM-as-Judge, with bias mitigations

## License

MIT (matches parent repo).
