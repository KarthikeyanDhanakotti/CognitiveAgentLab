"""
evaluate.py — RAGAS evaluation of Naive vs Advanced RAG.

Runs both pipelines on a small golden test set, grades them with 4 RAGAS
metrics (Faithfulness · Answer Relevancy · Context Precision · Context Recall)
using a Groq Llama judge, prints per-pipeline averages + % lift, and saves
optional bar chart + CSV outputs.

Run:
    python evaluate.py
    python evaluate.py --judge llama-3.3-70b-versatile --outdir results/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from advanced_rag import AdvancedRAG
from common import DEFAULT_EMBED_MODEL, load_groq_key
from naive_rag import NaiveRAG


# ── Golden test set (question + ground-truth answer) ─────────────────
TEST_SET: List[Dict[str, str]] = [
    {
        "question": "What is the main idea of this paper?",
        "ground_truth": (
            "The paper proposes the Transformer, a new network architecture based "
            "solely on attention mechanisms, dispensing entirely with recurrence "
            "and convolutions. It achieves superior translation quality while "
            "being more parallelizable and faster to train."
        ),
    },
    {
        "question": "What is the Transformer architecture?",
        "ground_truth": (
            "The Transformer follows an encoder-decoder structure using stacked "
            "self-attention and point-wise fully connected layers for both the "
            "encoder and decoder, as shown in Figure 1."
        ),
    },
    {
        "question": "What datasets were used in the experiments?",
        "ground_truth": (
            "WMT 2014 English-German (about 4.5M sentence pairs), "
            "WMT 2014 English-French (36M sentences), "
            "Wall Street Journal portion of the Penn Treebank (~40K sentences), "
            "and high-confidence + BerkeleyParser corpora (~17M sentences)."
        ),
    },
    {
        "question": "Who are the authors of this paper?",
        "ground_truth": (
            "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, "
            "Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin."
        ),
    },
]


def run_pipeline(rag, testset: List[Dict[str, str]], label: str) -> Dict[str, list]:
    """Run one RAG pipeline over the test set. Returns RAGAS-shaped dict."""
    print(f"\n▶ Running {label} …")
    rows: Dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for i, item in enumerate(testset, 1):
        print(f"   [{i}/{len(testset)}] {item['question'][:60]}…")
        out = rag.query(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(out["answer"])
        rows["contexts"].append([d.page_content for d in out["retrieved_chunks"]])
        rows["ground_truth"].append(item["ground_truth"])
    return rows


def build_ragas_judge(judge_model: str, embeddings):
    """Return (judge_llm, judge_embeddings, metrics) tuple for ragas.evaluate."""
    from langchain_groq import ChatGroq
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:  # ragas >= 0.2 alt layout
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.embeddings.base import LangchainEmbeddingsWrapper

    try:
        chat = ChatGroq(
            model=judge_model, temperature=0, api_key=os.environ["GROQ_API_KEY"]
        )
    except TypeError:
        chat = ChatGroq(
            model=judge_model, temperature=0, groq_api_key=os.environ["GROQ_API_KEY"]
        )

    judge_llm = LangchainLLMWrapper(chat)
    judge_embeddings = LangchainEmbeddingsWrapper(embeddings)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return judge_llm, judge_embeddings, metrics


def evaluate_pipeline(rows, label, judge_llm, judge_embeddings, metrics) -> pd.DataFrame:
    from datasets import Dataset
    from ragas import evaluate

    print(f"\n  Evaluating {label} (4 metrics × {len(rows['question'])} questions)…")
    ds = Dataset.from_dict(rows)
    result = evaluate(ds, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)
    df = result.to_pandas()
    df.insert(0, "pipeline", label)
    return df


def print_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    available = [m for m in metric_cols if m in df_all.columns]
    summary = df_all.groupby("pipeline")[available].mean().round(3)

    print("\n AVERAGE RAGAS SCORES")
    print("=" * 70)
    print(summary)
    print("=" * 70)

    if {"Naive", "Advanced"}.issubset(summary.index):
        lift = (summary.loc["Advanced"] - summary.loc["Naive"]).round(3)
        lift_pct = (
            (summary.loc["Advanced"] - summary.loc["Naive"])
            / summary.loc["Naive"].replace(0, 0.01)
            * 100
        ).round(1)
        print("\n ADVANCED vs NAIVE — Absolute Lift")
        print(lift)
        print("\n ADVANCED vs NAIVE — % Lift")
        print(lift_pct.astype(str) + " %")
    return summary


def save_bar_chart(summary: pd.DataFrame, outfile: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    summary.T.plot(
        kind="bar",
        ax=ax,
        rot=15,
        color=["#C73E1D", "#2E86AB"],
        edgecolor="black",
        width=0.7,
    )
    ax.set_title(
        "Naive vs Advanced RAG — RAGAS Quality Metrics",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Score (0 = bad, 1 = perfect)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Production threshold")
    ax.legend(title="Pipeline", loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"📊 Saved bar chart → {outfile}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS evaluation of Naive vs Advanced RAG")
    p.add_argument(
        "--judge",
        default="llama-3.1-8b-instant",
        help="Groq judge model (default: llama-3.1-8b-instant; try llama-3.3-70b-versatile for stricter grading)",
    )
    p.add_argument(
        "--outdir",
        default="results",
        help="Directory for CSV + PNG outputs (default: results/)",
    )
    p.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip matplotlib bar chart (useful in headless CI)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    load_groq_key()
    print(f"🔧 Judge model: {args.judge}\n")

    # 1. Build both pipelines
    print("═" * 70)
    print(" Building Naive RAG …")
    print("═" * 70)
    naive = NaiveRAG.build()

    print("\n" + "═" * 70)
    print(" Building Advanced RAG …")
    print("═" * 70)
    advanced = AdvancedRAG.build()

    # 2. Generate answers on the golden set
    naive_rows = run_pipeline(naive, TEST_SET, "Naive RAG")
    adv_rows = run_pipeline(advanced, TEST_SET, "Advanced RAG")

    # 3. Get embeddings for RAGAS (same model as retrieval — cheap wrapper reuses cache)
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL)

    # 4. Score both pipelines with RAGAS
    judge_llm, judge_embeddings, metrics = build_ragas_judge(args.judge, embeddings)
    df_naive = evaluate_pipeline(naive_rows, "Naive", judge_llm, judge_embeddings, metrics)
    df_adv = evaluate_pipeline(adv_rows, "Advanced", judge_llm, judge_embeddings, metrics)
    df_all = pd.concat([df_naive, df_adv], ignore_index=True)

    # 5. Print summary + save artefacts
    summary = print_summary(df_all)

    df_all.to_csv(outdir / "ragas_all_scores.csv", index=False)
    summary.to_csv(outdir / "ragas_summary.csv")
    print(f"\n💾 Saved raw scores → {outdir/'ragas_all_scores.csv'}")
    print(f"💾 Saved summary   → {outdir/'ragas_summary.csv'}")

    if not args.no_chart:
        save_bar_chart(summary, outdir / "ragas_comparison.png")

    print("\n✅ Evaluation complete")


if __name__ == "__main__":
    main()
