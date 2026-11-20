"""Result charts used in the reports (reads run outputs; no recomputation). Output: docs/figures/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from irm import config  # noqa: E402

config.FIGURES.mkdir(parents=True, exist_ok=True)

INK, ACCENT, LIGHT = "#1F3B5C", "#2E75B6", "#9DC3E6"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def rules_before_after():
    rows = {}
    for v in ("v1", "v2"):
        s = json.loads((config.EXTRACTION_DIR / f"w1_textlayer_{v}_all_digital" / "_summary.json").read_text())
        rows[v] = {g: m["f1"] for g, m in s["by_group"].items()}
    df = pd.DataFrame(rows).sort_values("v1")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(df))
    ax.barh([i + 0.2 for i in y], df.v1, height=0.4, color=LIGHT, label="rules v1")
    ax.barh([i - 0.2 for i in y], df.v2, height=0.4, color=ACCENT, label="rules v2 (after fix)")
    ax.set_yticks(list(y)); ax.set_yticklabels(df.index, fontsize=9); ax.set_xlim(0.8, 1.01)
    ax.set_xlabel("field-level F1 (281 digital resumes)")
    style(ax, "Non-LLM reference: F1 by field group, before / after fix"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout(); fig.savefig(config.FIGURES / "w1_rules_before_after.png", dpi=160); plt.close(fig)


def retrieval():
    df = pd.read_csv(config.PROCESSED / "retrieval_check_lsa.csv")
    p = df.pivot_table(index="category", columns="mode", values="hit@5")[["dense", "sparse", "hybrid"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    p.rename(columns={"sparse": "BM25", "dense": "dense (LSA)", "hybrid": "hybrid RRF"}).plot.bar(ax=ax, color=[LIGHT, "#A6A6A6", ACCENT], width=0.75)
    ax.set_ylabel("Hit@5"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=25)
    style(ax, "Preliminary retrieval check (offline LSA dense vs BM25 vs RRF hybrid)")
    ax.set_ylim(0, 1.18); ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(config.FIGURES / "w2_retrieval_check.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    rules_before_after(); retrieval(); print("figures written to", config.FIGURES)
