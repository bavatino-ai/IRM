"""Field-level extraction evaluation and error taxonomy (Week 1).

Scoring unit = one field slot of one resume. Scalars/skills are single slots;
list fields (languages, certificates, projects) contribute one slot per item.

Outcome per slot
  TP  predicted == expected
  FN  expected present, predicted missing          -> error 'missing'
  FP+FN expected present, predicted different        -> 'misplaced' if the predicted value is the
                                                        expected value of another field, else 'wrong_value'
  FP  predicted present, nothing expected            -> 'invented' if not grounded in the source text,
                                                        else 'unexpected_extra'
'scan_only' is assigned later by comparing a scan variant against its digital twin.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict

import pandas as pd

from ..ground_truth import norm

LIST_FIELDS = ("languages", "certificates", "projects")


@dataclass
class SlotResult:
    code: int
    field: str
    expected: object
    predicted: object
    outcome: str          # TP / FN / FP / FPFN
    error: str            # '' | missing | wrong_value | misplaced | invented | unexpected_extra


def field_group(f: str) -> str:
    if f.startswith("skill:"):
        return "skill_years"
    return f.split(".")[0] if f.startswith("education") else f


def _grounded(value, source_text: str) -> bool:
    if not source_text:
        return False
    src = norm(source_text)
    if isinstance(value, tuple):
        return all(_grounded(v, source_text) for v in value if v is not None)
    if isinstance(value, float):
        return f"{value:g}" in src
    return norm(value) in src


def compare(code: int, expected: dict, predicted: dict | None, source_text: str = "") -> list[SlotResult]:
    predicted = predicted or {}
    out: list[SlotResult] = []
    exp_scalars = {k: v for k, v in expected.items() if k not in LIST_FIELDS and v is not None}
    other_vals = lambda f: {norm(v) for k, v in exp_scalars.items() if k != f}  # noqa: E731

    for f in sorted(set(expected) | set(predicted)):
        e, p = expected.get(f), predicted.get(f)
        if f in LIST_FIELDS:
            e, p = e or frozenset(), p or frozenset()
            for item in e & p:
                out.append(SlotResult(code, f, item, item, "TP", ""))
            for item in e - p:
                out.append(SlotResult(code, f, item, None, "FN", "missing"))
            for item in p - e:
                err = "unexpected_extra" if _grounded(item, source_text) else "invented"
                out.append(SlotResult(code, f, None, item, "FP", err))
            continue
        if e is None and p is None:
            continue
        if e is not None and p is None:
            out.append(SlotResult(code, f, e, None, "FN", "missing"))
        elif e is None:
            err = "unexpected_extra" if _grounded(p, source_text) else "invented"
            out.append(SlotResult(code, f, None, p, "FP", err))
        elif (e == p) or (isinstance(e, str) and norm(e) == norm(p)):
            out.append(SlotResult(code, f, e, p, "TP", ""))
        else:
            err = "misplaced" if norm(p) in other_vals(f) else (
                "invented" if source_text and not _grounded(p, source_text) else "wrong_value")
            out.append(SlotResult(code, f, e, p, "FPFN", err))
    return out


def prf(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "slots": 0}
    tp = int((df.outcome == "TP").sum())
    fp = int(df.outcome.isin(["FP", "FPFN"]).sum())
    fn = int(df.outcome.isin(["FN", "FPFN"]).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "slots": int(len(df))}


def summarise(slots: list[SlotResult]) -> dict:
    df = pd.DataFrame([asdict(s) for s in slots])
    if df.empty:
        return {"overall": prf(df), "by_group": {}, "errors": {}}
    df["group"] = df.field.map(field_group)
    by_group = {g: prf(d) for g, d in df.groupby("group")}
    errors = Counter(df.loc[df.error != "", "error"])
    return {"overall": prf(df), "by_group": by_group, "errors": dict(errors),
            "resumes": int(df.code.nunique())}


def scan_only_errors(digital: list[SlotResult], scan: list[SlotResult]) -> list[SlotResult]:
    """Errors present on the scan variant but not on its digital twin (same code, field, expected)."""
    ok_digital = {(s.code, s.field, repr(s.expected)) for s in digital if s.outcome == "TP"}
    res = []
    for s in scan:
        if s.error and (s.code, s.field, repr(s.expected)) in ok_digital:
            res.append(SlotResult(s.code, s.field, s.expected, s.predicted, s.outcome, "scan_only"))
    return res
