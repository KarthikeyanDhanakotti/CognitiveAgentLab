"""
AgenticRAG — One-page Reference Architecture Diagram
=====================================================
Generates a LinkedIn/keynote-ready PNG that mirrors the "Agentic RAG
Architecture — 2026 Edition" reference sketch and is grounded 1:1 in
the working notebook AgenticRAG.ipynb.

Layout mirrors the source sketch:
  ┌──────────────────────────────────────────────┬─────────────────┐
  │  1  INPUT & ORCHESTRATION                    │  A. COMMON      │
  │  2  AGENT LOOP  (cyclic)                     │     FAILURE     │
  │  3  KNOWLEDGE & MEMORY LAYER                 │     POINTS      │
  │  4  RETRIEVAL QUALITY PIPELINE               │─────────────────│
  │  5  REASONING & GENERATION                   │  B. WHY         │
  │  6  EVALUATION & FEEDBACK                    │  AGENTIC RAG ?  │
  ├──────────────────────────────────────────────┴─────────────────┤
  │  CORE IDEA — planning → retrieval → verification → response    │
  └────────────────────────────────────────────────────────────────┘

Every label maps to a real function / model / constant in AgenticRAG.ipynb:
  intent_analysis · policy_check · initial_plan · STRATEGY_FOR_INTENT
  agent_loop (MAX_ITERS=3) · rewrite_query · gap_check · escalate_to
  tool_dense_search (FAISS · MiniLM-L6-v2) · tool_keyword_search (BM25)
  tool_hybrid_search (RRF, rrf_k=60) · session_memory · seen_chunk_ids
  dedup_and_filter · freshness_permission_check · rerank (ms-marco)
  build_context (max_chars=4000)
  draft_answer (Groq llama-3.1-8b-instant) · build_citation_line
  verify_grounding · agentic_rag
  RAGAS (faithfulness · answer_relevancy · context_precision · context_recall)

Author: Karthikeyan Dhanakotti (kartdh)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

OUTPUT_PATH = Path(__file__).with_name("AgenticRAG_Architecture.png")

# ---------------------------------------------------------------------------
# Palette — matches the sketch (blue tiers, purple loop, green knowledge,
# orange eval, red failure panel, green "why" panel)
# ---------------------------------------------------------------------------
C = {
    "bg":         "#FFFFFF",
    "title":      "#0D1B2A",
    "subtitle":   "#37474F",
    "text":       "#1A1A1A",
    "muted":      "#546E7A",
    "arrow":      "#37474F",
    "loop_arrow": "#6A1B9A",
    "down_arrow": "#1565C0",

    # (fill, edge, header) per tier
    "input":     ("#E8F1FB", "#1565C0", "#0D47A1"),
    "loop":     ("#F3E8FB", "#6A1B9A", "#4A148C"),
    "know":     ("#E6F4EA", "#2E7D32", "#1B5E20"),
    "quality":  ("#E8F1FB", "#1565C0", "#0D47A1"),
    "reason":   ("#F3E8FB", "#6A1B9A", "#4A148C"),
    "eval":     ("#FFF1E0", "#E65100", "#BF360C"),

    # right-side panels
    "fail":     ("#FDECEA", "#C62828", "#B71C1C"),
    "why":      ("#E6F4EA", "#2E7D32", "#1B5E20"),

    # bottom banner
    "core":     ("#FFF1E0", "#E65100", "#BF360C"),

    # badge (2026 Edition)
    "badge":    ("#FDECEA", "#C62828", "#B71C1C"),
}

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 17, 22                    # inches (portrait, like the sketch)
GRID_W, GRID_H = 100, 130                # abstract layout units

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=C["bg"])
ax.set_xlim(0, GRID_W)
ax.set_ylim(0, GRID_H)
ax.axis("off")

# left column (tiers) vs right column (sidebar panels)
LEFT_X, LEFT_W = 1.5, 68.0
RIGHT_X, RIGHT_W = 71.5, 27.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def box(x, y, w, h, text, kind, fontsize=8.5, weight="bold", subtitle=None,
        rounding=0.7, pad=0.15, edge_lw=1.4, alpha=1.0):
    fill, edge, _ = C[kind]
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        linewidth=edge_lw, edgecolor=edge, facecolor=fill,
        alpha=alpha, zorder=3,
    )
    ax.add_patch(p)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.63, text,
                ha="center", va="center", fontsize=fontsize,
                weight=weight, color=C["text"], zorder=4)
        ax.text(x + w / 2, y + h * 0.28, subtitle,
                ha="center", va="center", fontsize=fontsize - 1.6,
                color=C["muted"], style="italic", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                weight=weight, color=C["text"], zorder=4)


def band(x, y, w, h, number, label, kind, tag=""):
    """Numbered tier band with pale fill + accented border + header."""
    fill, edge, header = C[kind]
    r = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10,rounding_size=0.9",
        facecolor=fill, edgecolor=edge, linewidth=1.6,
        alpha=0.55, zorder=1,
    )
    ax.add_patch(r)
    # number badge
    Circle_r = 1.3
    circ = Circle((x + 2.4, y + h - 1.9), Circle_r,
                  facecolor=edge, edgecolor=edge, zorder=2)
    ax.add_patch(circ)
    ax.text(x + 2.4, y + h - 1.9, str(number),
            ha="center", va="center", fontsize=11,
            weight="bold", color="white", zorder=3)
    # header text
    ax.text(x + 4.6, y + h - 1.9, label,
            fontsize=12, weight="bold", color=header,
            va="center", zorder=2)
    if tag:
        ax.text(x + w - 2.0, y + h - 1.9, tag,
                fontsize=8.4, color=header, ha="right",
                va="center", style="italic", zorder=2,
                clip_on=False)


def arrow(x1, y1, x2, y2, color=None, style="-|>", lw=1.4, curve=0.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color or C["arrow"],
        connectionstyle=f"arc3,rad={curve}", zorder=2,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# TITLE + BADGE
# ---------------------------------------------------------------------------
ax.text(GRID_W / 2 - 4, GRID_H - 2.8,
        "Agentic RAG Architecture",
        ha="center", va="center", fontsize=26, weight="bold",
        color=C["title"])

# 2026 Edition badge (top-right)
badge_fill, badge_edge, _ = C["badge"]
badge = FancyBboxPatch(
    (GRID_W - 15.0, GRID_H - 4.6), 12.5, 3.6,
    boxstyle="round,pad=0.15,rounding_size=0.8",
    facecolor=badge_fill, edgecolor=badge_edge, linewidth=1.6, zorder=3,
)
ax.add_patch(badge)
ax.text(GRID_W - 8.75, GRID_H - 2.8,
        "2026 Edition", ha="center", va="center",
        fontsize=13, weight="bold", color=badge_edge)

ax.text(GRID_W / 2 - 4, GRID_H - 5.6,
        "How retrieval becomes a reasoning loop",
        ha="center", va="center", fontsize=13,
        color=C["subtitle"], style="italic")

ax.text(GRID_W / 2 - 4, GRID_H - 7.6,
        "Every label maps 1:1 to a real function in AgenticRAG.ipynb  "
        "·  MiniLM-L6-v2  ·  BM25  ·  ms-marco CrossEncoder  "
        "·  Groq llama-3.1-8b-instant  ·  RAGAS",
        ha="center", va="center", fontsize=9.2,
        color=C["muted"], style="italic")


# ===========================================================================
#                       LEFT COLUMN — 6 NUMBERED TIERS
# ===========================================================================

# ---------------------------------------------------------------------------
# TIER 1 — INPUT & ORCHESTRATION
# ---------------------------------------------------------------------------
Y1, H1 = 108.5, 11.0
band(LEFT_X, Y1, LEFT_W, H1, 1, "INPUT & ORCHESTRATION", "input",
     tag="initial_plan()")

t1 = [
    ("User Query",         "raw text",                              3.5, 12.0),
    ("Intent + Task\nAnalysis",  "intent_analysis()\n4 labels",     17.5, 13.5),
    ("Planner /\nDecomposer",   "STRATEGY_FOR_INTENT\ndense·keyword·hybrid", 33.0, 15.0),
    ("Policy Check",       "policy_check()\nprompt-injection filter", 50.5, 15.5),
]
for label, sub, dx, w in t1:
    box(LEFT_X + dx, Y1 + 1.5, w, 6.3, label, "input",
        fontsize=8.8, subtitle=sub)

# forward arrows
xs = [3.5 + 12.0, 17.5 + 13.5, 33.0 + 15.0]
xt = [17.5,       33.0,        50.5]
for a, b in zip(xs, xt):
    arrow(LEFT_X + a, Y1 + 4.6, LEFT_X + b, Y1 + 4.6)


# ---------------------------------------------------------------------------
# TIER 2 — AGENT LOOP (dashed border, cyclic layout)
# ---------------------------------------------------------------------------
Y2, H2 = 78.5, 28.5
band(LEFT_X, Y2, LEFT_W, H2, 2, "AGENT LOOP", "loop",
     tag="agent_loop()   MAX_ITERS = 3")

# ── Connector from Tier 1 → Tier 2 (Policy Check ▼ Need More Evidence?) ──
# Solid curved arrow, matches the reference sketch's flow between blocks.
policy_bottom_x = LEFT_X + 50.5 + 15.5 / 2       # Policy Check bottom-center
need_top_x      = LEFT_X + 3.0 + 14.5 / 2        # Need More Evidence? top-center
arrow(policy_bottom_x, Y1 + 1.0,                 # just below Policy Check box
      need_top_x,      Y2 + 16.7,                # just above Need More Evidence?
      color=C["down_arrow"], lw=2.0, curve=-0.30)

# left decision node "Need More Evidence?"
box(LEFT_X + 3.0, Y2 + 10.5, 14.5, 6.0,
    "Need More\nEvidence?", "loop", fontsize=9.5, subtitle="gap_check()")

# central agent circle
CX, CY, CR = LEFT_X + 34.0, Y2 + 13.5, 4.6
c = Circle((CX, CY), CR,
           facecolor=C["loop"][0], edgecolor=C["loop"][1],
           linewidth=2.2, zorder=3)
ax.add_patch(c)
ax.text(CX, CY + 0.8, "Agent Loop", ha="center", va="center",
        fontsize=10, weight="bold", color=C["loop"][2])
ax.text(CX, CY - 1.4, "∞", ha="center", va="center",
        fontsize=15, weight="bold", color=C["loop"][2])

# 4 loop stages around the central circle
stages = [
    # (label, sub, cx, cy, w, h)
    ("Query Rewrite",           "rewrite_query()",         CX + 12.5, CY + 6.5, 15.0, 5.6),
    ("Retrieval Strategy\nSelector", "STRATEGY_FOR_INTENT", CX + 12.5, CY - 0.5, 15.0, 5.6),
    ("Tool / Source\nSelection",     "STRATEGIES map",     CX + 12.5, CY - 7.5, 15.0, 5.6),
    ("Multi-step\nRetrieval",   "STRATEGIES[strategy]\ntop_k = 20",  CX - 4.0, CY - 9.5, 16.0, 6.0),
    ("Gap Detection",           "gap_check() → escalate_to",         CX - 22.0, CY - 9.5, 16.0, 6.0),
]
for label, sub, x, y, w, h in stages:
    box(x, y, w, h, label, "loop", fontsize=8.6, subtitle=sub)

# cyclic arrows (Rewrite → Selector → Tool → Multi-step Retrieval → Gap → Need More?)
arrow(CX + 20.0, CY + 6.5, CX + 20.0, CY + 2.3,
      color=C["loop_arrow"])                              # rewrite → selector
arrow(CX + 20.0, CY - 0.5, CX + 20.0, CY - 4.7,
      color=C["loop_arrow"])                              # selector → tool
arrow(CX + 18.0, CY - 7.5, CX + 12.0, CY - 8.5,
      color=C["loop_arrow"], curve=0.15)                  # tool → multi-step
arrow(CX - 4.0, CY - 8.0, CX - 6.0, CY - 8.0,
      color=C["loop_arrow"])                              # multi-step → gap
arrow(CX - 14.5, CY - 6.5, CX - 24.0, CY + 12.5,
      color=C["loop_arrow"], curve=-0.25)                 # gap → need more?
arrow(LEFT_X + 10.0, Y2 + 16.5, CX + 8.0, Y2 + 20.0,
      color=C["loop_arrow"], curve=-0.25)                 # need more? YES → rewrite

# YES / NO labels
ax.text(LEFT_X + 20.0, Y2 + 20.5, "YES", fontsize=9.5, weight="bold",
        color=C["loop_arrow"])

# NO branch (down to Knowledge tier, dashed)
ax.annotate("", xy=(LEFT_X + 10.0, Y2 - 0.2),
            xytext=(LEFT_X + 10.0, Y2 + 10.5),
            arrowprops=dict(arrowstyle="-|>", linestyle="--",
                            color=C["loop"][2], lw=1.6), zorder=2)
ax.text(LEFT_X + 6.0, Y2 + 5.0, "NO", fontsize=9.5,
        weight="bold", color=C["loop"][2])


# ---------------------------------------------------------------------------
# TIER 3 — KNOWLEDGE & MEMORY LAYER
# ---------------------------------------------------------------------------
Y3, H3 = 63.0, 13.5
band(LEFT_X, Y3, LEFT_W, H3, 3, "KNOWLEDGE & MEMORY LAYER", "know",
     tag="in-memory indexes  ·  session_memory")

t3 = [
    ("Vector DB",         "FAISS + MiniLM-L6-v2",   1.5, 8.5),
    ("Keyword /\nBM25 Search", "rank_bm25",       10.6, 8.5),
    ("SQL /\nStructured Data", "hook (prod)",     19.7, 8.5),
    ("APIs /\nTools",     "STRATEGIES map",        28.8, 8.5),
    ("Documents /\nPDFs / Wikis", "PyMuPDFLoader", 37.9, 8.5),
    ("Session\nMemory",   "seen_chunk_ids",        47.0, 8.5),
    ("Long-term\nMemory", "hook (prod)",           56.1, 8.5),
]
for label, sub, dx, w in t3:
    box(LEFT_X + dx, Y3 + 1.3, w, 7.5, label, "know",
        fontsize=8.4, subtitle=sub)

# down arrows connecting tiers
arrow(LEFT_X + LEFT_W / 2, Y2, LEFT_X + LEFT_W / 2, Y2 - 1.5,
      color=C["down_arrow"], lw=1.8)
arrow(LEFT_X + LEFT_W / 2, Y3, LEFT_X + LEFT_W / 2, Y3 - 1.5,
      color=C["down_arrow"], lw=1.8)


# ---------------------------------------------------------------------------
# TIER 4 — RETRIEVAL QUALITY PIPELINE
# ---------------------------------------------------------------------------
Y4, H4 = 47.5, 12.5
band(LEFT_X, Y4, LEFT_W, H4, 4, "RETRIEVAL QUALITY PIPELINE", "quality",
     tag="retrieval_quality_pipeline()")

t4 = [
    ("Candidate\nChunks",       "top-k = 20",                 1.5, 10.0),
    ("Reranker",                "ms-marco CrossEncoder\ntop 5", 12.5, 10.5),
    ("Dedup +\nFilter",         "dedup_and_filter()",         24.0, 10.5),
    ("Freshness /\nPermission", "freshness_permission_check()", 35.5, 11.0),
    ("Context\nBuilder",        "max_chars = 4000",           47.5, 10.0),
    ("Grounded\nContext",       "→ passed to LLM",            58.0,  8.5),
]
for label, sub, dx, w in t4:
    box(LEFT_X + dx, Y4 + 1.3, w, 6.5, label, "quality",
        fontsize=8.5, subtitle=sub)

xs = [1.5 + 10.0, 12.5 + 10.5, 24.0 + 10.5, 35.5 + 11.0, 47.5 + 10.0]
xt = [12.5,       24.0,        35.5,        47.5,        58.0]
for a, b in zip(xs, xt):
    arrow(LEFT_X + a, Y4 + 4.6, LEFT_X + b, Y4 + 4.6)

arrow(LEFT_X + LEFT_W / 2, Y4, LEFT_X + LEFT_W / 2, Y4 - 1.5,
      color=C["down_arrow"], lw=1.8)


# ---------------------------------------------------------------------------
# TIER 5 — REASONING & GENERATION
# ---------------------------------------------------------------------------
Y5, H5 = 32.0, 12.5
band(LEFT_X, Y5, LEFT_W, H5, 5, "REASONING & GENERATION", "reason",
     tag="agentic_rag()")

t5 = [
    ("LLM Reasoning",          "Groq llama-3.1-8b-instant", 1.5, 12.5),
    ("Draft Answer",           "draft_answer()",           15.5, 11.0),
    ("Citation Builder",       "[Page X · Chunk Y]",       28.0, 12.5),
    ("Verifier /\nGroundedness", "verify_grounding()",     42.0, 13.0),
    ("Final Answer",           "answer + citations\n+ flags", 56.5, 11.5),
]
for label, sub, dx, w in t5:
    box(LEFT_X + dx, Y5 + 1.3, w, 6.5, label, "reason",
        fontsize=8.6, subtitle=sub)

xs = [1.5 + 12.5, 15.5 + 11.0, 28.0 + 12.5, 42.0 + 13.0]
xt = [15.5,       28.0,        42.0,        56.5]
for a, b in zip(xs, xt):
    arrow(LEFT_X + a, Y5 + 4.6, LEFT_X + b, Y5 + 4.6)

arrow(LEFT_X + LEFT_W / 2, Y5, LEFT_X + LEFT_W / 2, Y5 - 1.5,
      color=C["down_arrow"], lw=1.8)


# ---------------------------------------------------------------------------
# TIER 6 — EVALUATION & FEEDBACK
# ---------------------------------------------------------------------------
Y6, H6 = 17.5, 12.5
band(LEFT_X, Y6, LEFT_W, H6, 6, "EVALUATION & FEEDBACK", "eval",
     tag="RAGAS + per-stage instrumentation")

t6 = [
    ("Answer\nCorrectness", "RAGAS\nfaithfulness",       1.5, 10.5),
    ("Retrieval\nPrecision", "RAGAS\ncontext_precision", 12.5, 10.5),
    ("Retrieval\nRecall",    "RAGAS\ncontext_recall",    23.5, 10.5),
    ("Latency",              "chat_traced()",            34.5,  9.5),
    ("Cost",                 "COST_PER_1M",              44.5,  8.5),
    ("User\nFeedback",       "capture loop",             53.5, 11.5),
]
for label, sub, dx, w in t6:
    box(LEFT_X + dx, Y6 + 1.2, w, 7.5, label, "eval",
        fontsize=8.5, subtitle=sub)


# ===========================================================================
#                RIGHT SIDEBAR — FAILURE POINTS + WHY AGENTIC RAG
# ===========================================================================

# A. COMMON FAILURE POINTS
YA, HA = 70.0, 50.0
fill, edge, header = C["fail"]
panelA = FancyBboxPatch(
    (RIGHT_X, YA), RIGHT_W, HA,
    boxstyle="round,pad=0.20,rounding_size=1.0",
    facecolor=fill, edgecolor=edge, linewidth=1.8, alpha=0.7, zorder=1,
)
ax.add_patch(panelA)
ax.text(RIGHT_X + RIGHT_W / 2, YA + HA - 2.5,
        "⚠  A.  COMMON FAILURE POINTS",
        ha="center", va="center", fontsize=12,
        weight="bold", color=header, zorder=2)

failures = [
    ("Wrong source retrieved",   "→ intent-aware strategy + rerank"),
    ("Stale evidence",           "→ freshness_permission_check()"),
    ("Missing permissions",      "→ ACL hook"),
    ("Weak reranking",           "→ ms-marco CrossEncoder"),
    ("Context overload",         "→ build_context(max_chars=4000)"),
    ("Hallucinated synthesis",   "→ verify_grounding()"),
    ("Loop explosion",           "→ MAX_ITERS = 3"),
    ("Memory contamination",     "→ dedup + seen_chunk_ids"),
]
row_h = 4.9
y_start = YA + HA - 6.5
for i, (name, guard) in enumerate(failures):
    y = y_start - i * row_h
    # red dot bullet
    Circle_r = 0.35
    ax.add_patch(Circle((RIGHT_X + 1.8, y + 1.4), Circle_r,
                        facecolor=edge, edgecolor=edge, zorder=3))
    ax.text(RIGHT_X + 3.0, y + 1.9, name,
            fontsize=9.5, weight="bold", color=C["text"],
            va="center", zorder=3)
    ax.text(RIGHT_X + 3.0, y + 0.4, guard,
            fontsize=8.0, style="italic", color=C["muted"],
            va="center", zorder=3)


# B. WHY AGENTIC RAG?
YB, HB = 17.5, 48.0
fill, edge, header = C["why"]
panelB = FancyBboxPatch(
    (RIGHT_X, YB), RIGHT_W, HB,
    boxstyle="round,pad=0.20,rounding_size=1.0",
    facecolor=fill, edgecolor=edge, linewidth=1.8, alpha=0.7, zorder=1,
)
ax.add_patch(panelB)
ax.text(RIGHT_X + RIGHT_W / 2, YB + HB - 2.5,
        "★  B.  WHY AGENTIC RAG?",
        ha="center", va="center", fontsize=12,
        weight="bold", color=header, zorder=2)

reasons = [
    ("Decomposes complex queries",     "planner + intent classifier"),
    ("Adapts retrieval strategy",      "STRATEGY_FOR_INTENT + escalate_to"),
    ("Verifies evidence before answering", "verify_grounding() judge"),
    ("Improves grounding + citations", "build_citation_line()"),
    ("Supports tools + memory + iteration", "STRATEGIES · session_memory · MAX_ITERS"),
    ("Refuses when evidence is missing", '"Not found in document"'),
    ("Bounded, auditable execution",   "trace + per-stage timings"),
]
row_h = 5.6
y_start = YB + HB - 6.5
for i, (name, sub) in enumerate(reasons):
    y = y_start - i * row_h
    # green check mark
    ax.text(RIGHT_X + 2.0, y + 1.9, "✓",
            fontsize=13, weight="bold", color=edge,
            va="center", zorder=3)
    ax.text(RIGHT_X + 4.0, y + 1.9, name,
            fontsize=9.5, weight="bold", color=C["text"],
            va="center", zorder=3)
    ax.text(RIGHT_X + 4.0, y + 0.4, sub,
            fontsize=8.0, style="italic", color=C["muted"],
            va="center", zorder=3)


# ===========================================================================
#                            BOTTOM BANNER — CORE IDEA
# ===========================================================================
Y_CORE, H_CORE = 3.5, 11.0
fill, edge, header = C["core"]
core = FancyBboxPatch(
    (LEFT_X, Y_CORE), GRID_W - 3.0, H_CORE,
    boxstyle="round,pad=0.25,rounding_size=1.2",
    facecolor=fill, edgecolor=edge, linewidth=2.0, alpha=0.75, zorder=1,
)
ax.add_patch(core)
ax.text(LEFT_X + 4.0, Y_CORE + H_CORE - 2.6, "★  CORE IDEA",
        fontsize=13, weight="bold", color=header, va="center", zorder=2)
ax.text(GRID_W / 2, Y_CORE + H_CORE / 2 - 0.6,
        "Agentic RAG is not just retrieve → generate.\n"
        "It is a controlled loop of  planning · retrieval · verification · response  "
        "— the LLM chooses the retriever, iterates when evidence is weak, "
        "and refuses to answer beyond what the context supports.",
        ha="center", va="center", fontsize=10.8,
        color=C["text"], zorder=2)


# ---------------------------------------------------------------------------
# ATTRIBUTION FOOTER
# ---------------------------------------------------------------------------
ax.text(GRID_W / 2, 1.0,
        "Built and verified against  AgenticRAG.ipynb   ·   "
        "Karthikeyan Dhanakotti  (kartdh)",
        ha="center", va="center", fontsize=8.5,
        color=C["muted"], style="italic")


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------
plt.tight_layout(pad=0.2)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor=C["bg"])
print(f"✅  wrote {OUTPUT_PATH}   ({OUTPUT_PATH.stat().st_size // 1024} KB)")
