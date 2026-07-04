"""Dump every notebook cell in Colab-copy-friendly order."""
import json
from pathlib import Path

BASE = Path(r"C:\Users\kartdh\repos\CognitiveAgentLab\projects\FineTuning")

NBS = [
    ("MODULE 1 — 01_dataset_quality_audit.ipynb",
     BASE / "module_1_strategy_data/notebooks/01_dataset_quality_audit.ipynb"),
    ("MODULE 2a — 02a_qlora_training_v1_chatdoctor.ipynb",
     BASE / "module_2_finetuning_qlora/notebooks/02a_qlora_training_v1_chatdoctor.ipynb"),
    ("MODULE 2b — 02b_qlora_training_v2_wikidoc.ipynb",
     BASE / "module_2_finetuning_qlora/notebooks/02b_qlora_training_v2_wikidoc.ipynb"),
    ("MODULE 3 — 03_hf_deploy_inference.ipynb",
     BASE / "module_3_deployment_inference/notebooks/03_hf_deploy_inference.ipynb"),
    ("MODULE 4 — 04_langsmith_evaluation.ipynb",
     BASE / "module_4_evaluation_langsmith/notebooks/04_langsmith_evaluation.ipynb"),
]

OUT_DIR = BASE / "docs" / "cells"
OUT_DIR.mkdir(exist_ok=True)


def dump(nb_label: str, nb_path: Path) -> Path:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    out_path = OUT_DIR / (nb_path.stem + ".md")
    lines = []
    lines.append(f"# {nb_label}\n")
    lines.append(f"_Source: `{nb_path.name}`_\n")
    for i, cell in enumerate(data["cells"], 1):
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            lines.append(f"\n## Cell {i} — Markdown\n")
            lines.append(src.rstrip())
            lines.append("")
        else:
            lines.append(f"\n## Cell {i} — Code\n")
            lines.append("```python")
            lines.append(src.rstrip())
            lines.append("```")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


for label, path in NBS:
    out = dump(label, path)
    print(f"OK  {out}")
