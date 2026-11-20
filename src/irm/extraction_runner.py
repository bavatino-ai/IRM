"""Batch extraction + evaluation runs (baseline, Gemini tier comparison, before/after prompt fix)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import config
from .evaluation.extraction_eval import compare, scan_only_errors, summarise
from .ground_truth import expected_for, flatten, load_manifest, load_profile
from .ingestion import load, text_layer
from .matching import RecordMatcher
from .llm.gemini_client import QuotaExhausted
from .ocr.base import OCRProvider
from .validation import validate


def select_sample(n: int = 20, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified sample for the human-checked tier comparison: covers every title band,
    resumes with/without certificate, and short/long resumes."""
    man, prof = load_manifest(), load_profile()
    df = man.merge(prof[["Code", "YearOfIssue"]], on="Code")
    df["has_cert"] = df.YearOfIssue.notna()
    df["long"] = df.ProjectsRendered >= df.ProjectsRendered.median()
    picks = []
    for _, g in df.groupby(["Title", "long"]):
        picks.append(g.sample(n=min(len(g), max(1, round(n * len(g) / len(df)))), random_state=seed))
    s = pd.concat(picks).drop_duplicates("Code")
    if len(s) > n:
        s = s.sample(n=n, random_state=seed)
    elif len(s) < n:
        s = pd.concat([s, df[~df.Code.isin(s.Code)].sample(n - len(s), random_state=seed)])
    if s.has_cert.all() and (~df.has_cert).any():      # keep the 'no certificate' edge case in the sample
        s = pd.concat([s.iloc[:-1], df[~df.has_cert].sample(1, random_state=seed)])
    return s.sort_values("Code")[["Code", "File", "Title", "ProjectsRendered", "has_cert"]].reset_index(drop=True)


def run(provider: OCRProvider, files: list[Path], run_name: str, digital_twin_dir: Path | None = None) -> dict:
    """Extract every file, evaluate against ground truth, persist slots + summary."""
    profile, manifest = load_profile(), load_manifest()
    matcher = RecordMatcher(profile)
    out_dir = config.EXTRACTION_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    slots, rows, aborted = [], [], None
    for f in files:
        doc = load(f, render=provider.name != "textlayer")
        try:
            res = provider.extract(doc)
        except QuotaExhausted as e:            # stop; unprocessed files are NOT scored as misses
            why = "provider unavailable" if type(e).__name__ == "ProviderUnavailable" else "daily quota reached"
            aborted = f"{why} after {len(rows)} of {len(files)} documents: {e}"
            print("ABORTED -", aborted[:200])
            break
        code = doc.code_hint
        # grounding source: own text layer, or the digital twin's text for scan variants (evaluation only)
        src = text_layer(f) if doc.has_text_layer else (
            text_layer(digital_twin_dir / f.name.replace("-scan", "")) if digital_twin_dir else "")
        pred = flatten(res.extraction) if res.extraction else None
        exp = expected_for(code, profile, manifest)
        slots += compare(code, exp, pred, src)
        mres = matcher.match(res.extraction) if res.extraction else None
        vrep = validate(res.extraction, text_layer(f) if doc.has_text_layer else "") if res.extraction else None
        rows.append({"file": f.name, "code": code, "kind": doc.kind, "pages": doc.n_pages,
                     "provider": res.provider, "model": res.model, "prompt": res.prompt_version,
                     "ok": res.ok, "errors": "; ".join(res.errors), "latency_s": round(res.latency_s, 3),
                     "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                     "cost_usd": res.cost_usd, "cached": res.cached,
                     "match_code": mres.code if mres else None, "match_method": mres.method if mres else None,
                     "match_correct": (mres.code == code) if mres else False,
                     "review_fields": "|".join(vrep.review_fields) if vrep else "ALL"})
        if res.extraction:
            (out_dir / f"{f.stem}.json").write_text(res.extraction.model_dump_json(indent=1), encoding="utf-8")
    per_doc = pd.DataFrame(rows, columns=["file", "code", "kind", "pages", "provider", "model", "prompt", "ok", "errors",
                                          "latency_s", "input_tokens", "output_tokens", "cost_usd", "cached",
                                          "match_code", "match_method", "match_correct", "review_fields"])
    per_doc.to_csv(out_dir / "_per_document.csv", index=False)
    pd.DataFrame([asdict(s) for s in slots]).to_csv(out_dir / "_slots.csv", index=False)
    summary = summarise(slots)
    summary.update({
        "run": run_name, "provider": provider.name, "model": provider.model,
        "documents_planned": len(files), "documents": len(rows), "aborted": aborted,
        "parsed_ok": int(per_doc.ok.sum()) if len(per_doc) else 0,
        "match_accuracy": round(float(per_doc.match_correct.mean()), 4) if len(per_doc) else 0.0,
        "docs_needing_review": int((per_doc.review_fields != "").sum()) if len(per_doc) else 0,
        "mean_latency_s": round(float(per_doc.latency_s.mean()), 3) if len(per_doc) else 0.0,
        "total_input_tokens": int(per_doc.input_tokens.sum()) if len(per_doc) else 0,
        "total_output_tokens": int(per_doc.output_tokens.sum()) if len(per_doc) else 0,
        "total_cost_usd": (round(float(per_doc.cost_usd.sum()), 4)
                           if per_doc.cost_usd.notna().all() and len(per_doc) else None),
    })
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def scan_only(run_digital: str, run_scan: str) -> dict:
    """Compare slot files of a digital run and a scan run of the same resumes."""
    from .evaluation.extraction_eval import SlotResult
    def _load(name):
        df = pd.read_csv(config.EXTRACTION_DIR / name / "_slots.csv").fillna("")
        return [SlotResult(int(r.code), r.field, r.expected, r.predicted, r.outcome, r.error)
                for r in df.itertuples()]
    d, s = _load(run_digital), _load(run_scan)
    codes = {x.code for x in s}
    so = scan_only_errors([x for x in d if x.code in codes], s)
    return {"scan_only_errors": len(so), "fields": pd.Series([x.field for x in so]).value_counts().to_dict()}
