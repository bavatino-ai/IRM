"""Record matching: resume -> existing employee record.

Keys (Week 1 decision):
  1. employee Code printed after 'ID:' (primary) - accepted only if the printed name agrees.
  2. normalised full name (fallback) - accepted only if unique in Employee_profile.
  3. fuzzy name (difflib ratio >= 0.85, single candidate above cutoff) - always routed to HR confirmation.
Email is NOT a usable key: contact details are excluded from the resumes.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

import pandas as pd

from .ground_truth import norm
from .schema import ResumeExtraction


@dataclass
class MatchResult:
    code: int | None
    method: str            # code | code_name_conflict | name | fuzzy_name | unmatched
    confidence: float
    needs_review: bool
    note: str = ""


class RecordMatcher:
    def __init__(self, profile: pd.DataFrame, fuzzy_threshold: float = 0.85):
        self.by_code = {int(c): norm(n) for c, n in zip(profile.Code, profile.Fullname)}
        names = pd.Series([norm(n) for n in profile.Fullname])
        self.name_counts = names.value_counts().to_dict()
        self.by_name = {norm(n): int(c) for c, n in zip(profile.Code, profile.Fullname)}
        self.threshold = fuzzy_threshold

    def match(self, x: ResumeExtraction) -> MatchResult:
        name = norm(x.full_name)
        if x.employee_code is not None and x.employee_code in self.by_code:
            if not name or self.by_code[x.employee_code] == name:
                return MatchResult(x.employee_code, "code", 1.0, False)
            ratio = difflib.SequenceMatcher(None, self.by_code[x.employee_code], name).ratio()
            if ratio >= self.threshold:
                return MatchResult(x.employee_code, "code", ratio, True, "name differs slightly (OCR?)")
            return MatchResult(x.employee_code, "code_name_conflict", ratio, True,
                               f"code {x.employee_code} belongs to '{self.by_code[x.employee_code]}'")
        if name and self.name_counts.get(name) == 1:
            return MatchResult(self.by_name[name], "name", 0.95, True, "matched on name only")
        if name:
            cands = difflib.get_close_matches(name, list(self.by_name), n=2, cutoff=self.threshold)
            if len(cands) == 1:
                return MatchResult(self.by_name[cands[0]], "fuzzy_name", 0.8, True, f"fuzzy: {cands[0]}")
        return MatchResult(None, "unmatched", 0.0, True, "no reliable key")
