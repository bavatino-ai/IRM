"""IRM command-line entry point.

Week 1
  python run.py check-apis                         # call Gemini + Claude once, write docs/evaluation/model_pins.json
  python run.py sample                             # 20-resume stratified sample + human-check sheet
  python run.py baseline                           # workload stats + HR timing sheet
  python run.py rules-baseline                     # non-LLM text-layer reference (v1 / v2), all 281 resumes
  python run.py extract --provider gemini --model gemini-2.5-flash --prompt v1 --set sample
  python run.py compare-tiers --prompt v1          # every GEMINI_OCR_CANDIDATES model on the sample
  python run.py before-after --model <selected>    # prompt v1 vs v2 on the sample (+ issue-log input)
Week 2
  python run.py eda | scans | build-db | index --embedder lsa|gemini|bge | eval-set | retrieval-check
  python run.py week1 | week2                      # offline steps of each week, end to end
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from irm import config  # noqa: E402


def _files(which: str) -> list[Path]:
    from irm.extraction_runner import select_sample
    if which == "all":
        return sorted(config.RESUME_DIR.glob("*.pdf"))
    if which == "sample":
        return [config.RESUME_DIR / f for f in select_sample().File]
    if which == "scan":
        return sorted(config.SCAN_DIR.glob("*-scan.pdf"))
    if which == "sample-scan":
        codes = set(select_sample().Code)
        return [f for f in sorted(config.SCAN_DIR.glob("*-scan.pdf")) if int(f.name[:5]) in codes]
    raise SystemExit(f"unknown set {which}")


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_check_apis(a):
    from irm.llm.clients import check_all
    _print(check_all())


def cmd_sample(a):
    import pandas as pd
    from irm.extraction_runner import select_sample
    from irm.ground_truth import expected_for, load_manifest, load_profile
    s = select_sample(a.n)
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(config.EVAL_DIR / "week1_sample_20.csv", index=False)
    prof, man, rows = load_profile(), load_manifest(), []
    for code, f in zip(s.Code, s.File):
        for fld, v in expected_for(int(code), prof, man).items():
            for item in (sorted(v, key=str) if isinstance(v, frozenset) else [v]):
                if item is not None:
                    rows.append({"file": f, "code": code, "field": fld, "expected_value": item,
                                 "verified_in_pdf": "", "comment": "", "checker": "", "date": ""})
    pd.DataFrame(rows).to_csv(config.EVAL_DIR / "week1_human_check_sheet.csv", index=False)
    print(s.to_string()); print(f"{len(rows)} field slots written to the human-check sheet")


def cmd_baseline(a):
    from irm.baseline import workload, write_timing_sheet
    w = workload(); write_timing_sheet()
    (config.BASELINE_DIR / "workload_stats.json").write_text(json.dumps(w, indent=2))
    _print(w)


def cmd_rules_baseline(a):
    from irm.extraction_runner import run
    from irm.ocr import get_provider
    out = {}
    for v in ("v1", "v2"):
        out[v] = run(get_provider("textlayer", version=v), _files("all"), f"w1_textlayer_{v}_all_digital")
    if config.SCAN_DIR.exists() and any(config.SCAN_DIR.glob("*-scan.pdf")):
        out["v2_scan"] = run(get_provider("textlayer", version="v2"), _files("scan"),
                             "w2_textlayer_v2_scan", digital_twin_dir=config.RESUME_DIR)
    _print({k: {"overall": v["overall"], "errors": v["errors"], "parsed_ok": v["parsed_ok"],
                "documents": v["documents"], "match_accuracy": v["match_accuracy"]} for k, v in out.items()})


def cmd_extract(a):
    from irm.extraction_runner import run
    from irm.ocr import get_provider
    if a.model is None and a.provider != "textlayer":
        a.model = config.CLAUDE_OCR_MODEL if a.provider == "claude" else config.GEMINI_OCR_MODEL
    kw = {"model": a.model, "prompt_version": a.prompt} if a.provider in ("gemini", "claude") else {}
    name = a.name or f"{a.provider}_{a.model or ''}_{a.prompt}_{a.set}".replace(".", "-")
    twin = config.RESUME_DIR if "scan" in a.set else None
    _print(run(get_provider(a.provider, **kw), _files(a.set), name, digital_twin_dir=twin))


def cmd_compare_tiers(a):
    import pandas as pd
    from irm.extraction_runner import run
    from irm.ocr import get_provider
    rows = []
    for m in config.GEMINI_OCR_CANDIDATES:
        s = run(get_provider("gemini", model=m, prompt_version=a.prompt), _files(a.set),
                f"w1_tier_{m}_{a.prompt}".replace(".", "-"))
        rows.append({"model": m, "prompt": a.prompt, "micro_f1": s["overall"]["f1"],
                     "precision": s["overall"]["precision"], "recall": s["overall"]["recall"],
                     "parsed_ok": s["parsed_ok"], "match_accuracy": s["match_accuracy"],
                     "mean_latency_s": s["mean_latency_s"], "input_tokens": s["total_input_tokens"],
                     "output_tokens": s["total_output_tokens"], "cost_usd": s["total_cost_usd"],
                     **{f"err_{k}": v for k, v in s["errors"].items()}})
    df = pd.DataFrame(rows)
    config.EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.EVAL_RESULTS / f"tier_comparison_{a.prompt}.csv", index=False)
    print(df.to_string())


def cmd_before_after(a):
    from irm.extraction_runner import run
    from irm.ocr import get_provider
    res = {p: run(get_provider("gemini", model=a.model, prompt_version=p), _files(a.set),
                  f"w1_fix_{a.model}_{p}".replace(".", "-")) for p in ("v1", "v2")}
    _print({p: {"overall": r["overall"], "errors": r["errors"]} for p, r in res.items()})


def cmd_eda(a):
    from irm import eda
    o = eda.run()
    _print({"profile_consistency": o["profile"]["consistency"], "resume_pages": o["resumes"]["pages"],
            "skill_values_changed": o["resumes"]["skill_values_changed"]})


def cmd_scans(a):
    import pandas as pd
    from irm import scans
    from irm.extraction_runner import select_sample
    from irm.ground_truth import load_manifest
    s, man = select_sample(), load_manifest()
    extra = man[~man.Code.isin(s.Code)].sample(10, random_state=7).Code.tolist()
    log = scans.build(s.Code.tolist() + extra)
    print(pd.DataFrame(log).groupby("profile").agg(files=("file", "count"), mean_bytes=("bytes", "mean"),
                                                   max_text_chars=("text_layer_chars", "max")))


def cmd_build_db(a):
    from irm import database
    _print(database.build())


def cmd_index(a):
    from irm.chunking import build_corpus, corpus_frame
    from irm.retrieval.embeddings import get_embedder
    from irm.retrieval.index import BM25Index, build_chroma
    chunks = build_corpus()
    corpus_frame(chunks).to_csv(config.PROCESSED / "chunks_inventory.csv", index=False)
    emb = get_embedder(a.embedder).fit([c.text for c in chunks])
    col = build_chroma(chunks, emb)
    BM25Index(chunks).save()
    print(f"chunks={len(chunks)} chroma[{col.name}]={col.count()} bm25 saved")


def cmd_eval_set(a):
    import pandas as pd
    from irm.evaluation import eval_set
    q = eval_set.build(); eval_set.save(q)
    print(pd.Series([x["category"] for x in q]).value_counts().to_string(), f"\ntotal {len(q)}")


def cmd_retrieval_check(a):
    from irm.evaluation import eval_set, retrieval_eval
    from irm.retrieval.embeddings import get_embedder
    from irm.retrieval.index import BM25Index, HybridRetriever, open_chroma
    emb = get_embedder(a.embedder)
    r = HybridRetriever(open_chroma(a.embedder), emb, BM25Index.load())
    df = retrieval_eval.evaluate(r, eval_set.load(), eval_set.RETRIEVAL)
    df.to_csv(config.PROCESSED / f"retrieval_check_{a.embedder}.csv", index=False)
    print(df.groupby("mode")[["hit@5", "mrr@10", "recall@10"]].mean().round(3).to_string())
    print(df.pivot_table(index="category", columns="mode", values="hit@5").round(2).to_string())


def cmd_answer_eval(a):
    import pandas as pd
    from irm.evaluation import answer_eval, eval_set
    from irm.retrieval.embeddings import get_embedder
    from irm.retrieval.index import BM25Index, HybridRetriever, open_chroma
    am = a.answer_model or (config.CLAUDE_ANSWER_MODEL if a.answer_provider == "claude" else config.GEMINI_ANSWER_MODEL)
    jm = a.judge_model or (config.GEMINI_JUDGE_MODEL if a.judge_provider == "gemini" else config.CLAUDE_JUDGE_MODEL)
    r = HybridRetriever(open_chroma(a.embedder), get_embedder(a.embedder), BM25Index.load())
    qs = answer_eval.subset(eval_set.load(), a.per_category)
    df, s = answer_eval.run(r, qs, a.answer_provider, am, a.judge_provider, jm, a.k)
    answer_eval.save(df, s, f"answer_eval_{am}__judge_{jm}".replace(".", "-"))
    if "refused" in df:
        print(df.groupby("category")[["groundedness", "answer_relevance", "context_relevance"]]
              .mean().round(2).to_string())
    _print(s)


def cmd_week1(a):
    for f in (cmd_sample, cmd_baseline, cmd_rules_baseline):
        f(a)


def cmd_week2(a):
    for f in (cmd_eda, cmd_scans, cmd_build_db, cmd_index, cmd_eval_set, cmd_retrieval_check, cmd_rules_baseline):
        f(a)


def main():
    p = argparse.ArgumentParser(description="IRM pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["check-apis", "baseline", "rules-baseline", "eda", "scans", "build-db", "eval-set", "week1", "week2"]:
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=20)
        sp.add_argument("--embedder", default="lsa")
    sp = sub.add_parser("sample"); sp.add_argument("--n", type=int, default=20)
    sp = sub.add_parser("extract")
    sp.add_argument("--provider", default="gemini", choices=["gemini", "claude", "textlayer"])
    sp.add_argument("--model", default=None, help="default: GEMINI_OCR_MODEL / CLAUDE_OCR_MODEL")
    sp.add_argument("--prompt", default="v1")
    sp.add_argument("--set", default="sample", choices=["sample", "all", "scan", "sample-scan"])
    sp.add_argument("--name", default=None)
    sp = sub.add_parser("compare-tiers"); sp.add_argument("--prompt", default="v1"); sp.add_argument("--set", default="sample")
    sp = sub.add_parser("before-after"); sp.add_argument("--model", default=config.GEMINI_OCR_MODEL); sp.add_argument("--set", default="sample")
    sp = sub.add_parser("answer-eval", help="answer the eval queries and score them with a judge model")
    sp.add_argument("--answer-provider", default="claude", choices=["claude", "gemini"])
    sp.add_argument("--answer-model", default=None)
    sp.add_argument("--judge-provider", default="gemini", choices=["gemini", "claude"])
    sp.add_argument("--judge-model", default=None)
    sp.add_argument("--embedder", default="lsa", choices=["lsa", "gemini", "bge"])
    sp.add_argument("--per-category", type=int, default=2, help="queries per category (0 = all 79)")
    sp.add_argument("--k", type=int, default=8, help="chunks given to the answer model")
    for name in ["index", "retrieval-check"]:
        sp = sub.add_parser(name); sp.add_argument("--embedder", default="lsa", choices=["lsa", "gemini", "bge"])
    a = p.parse_args()
    if not hasattr(a, "n"):
        a.n = 20
    if not hasattr(a, "embedder"):
        a.embedder = "lsa"
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
