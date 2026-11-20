"""Employee-level retrieval metrics: Hit@k, MRR@k, Recall@k (chunks are rolled up to employees)."""
from __future__ import annotations

import pandas as pd


def rollup(chunk_ids: list[str]) -> list[int]:
    seen, out = set(), []
    for cid in chunk_ids:
        c = int(cid.split("-")[0])
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def metrics(ranked_codes: list[int], relevant: set[int], k: int) -> dict:
    top = ranked_codes[:k]
    hit = any(c in relevant for c in top)
    rr = next((1.0 / i for i, c in enumerate(top, 1) if c in relevant), 0.0)
    recall = len(relevant & set(top)) / min(len(relevant), k) if relevant else 0.0
    return {f"hit@{k}": float(hit), f"mrr@{k}": rr, f"recall@{k}": recall}


def evaluate(retriever, queries: list[dict], categories: list[str], k_hit: int = 5, k_rec: int = 10) -> pd.DataFrame:
    rows = []
    for q in queries:
        if q["category"] not in categories or not q["relevant_codes"]:
            continue
        rel = set(q["relevant_codes"])
        for mode in ("dense", "sparse", "hybrid"):
            codes = rollup(getattr(retriever, mode)(q["query"], k=100))
            rows.append({"qid": q["qid"], "category": q["category"], "mode": mode, "gt_size": len(rel),
                         f"hit@{k_hit}": metrics(codes, rel, k_hit)[f"hit@{k_hit}"],
                         **{k: v for k, v in metrics(codes, rel, k_rec).items() if not k.startswith("hit")}})
    return pd.DataFrame(rows)
