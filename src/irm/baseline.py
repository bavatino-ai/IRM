"""Manual-baseline support (Week 1).

The HR time per resume cannot be derived from data - it must be timed with HR.
This module (1) quantifies the manual workload from the data (fields to read,
values that actually change) and (2) writes the timing protocol sheet.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .ground_truth import changed_skills, load_manifest, load_profile, expected_for


def workload() -> dict:
    prof, man = load_profile(), load_manifest()
    slots, changes, changed_docs = [], [], 0
    for code in man.Code:
        exp = expected_for(int(code), prof, man)
        n = sum(len(v) if isinstance(v, frozenset) else 1 for v in exp.values() if v is not None)
        slots.append(n)
        c = len(changed_skills(int(code), man))
        changes.append(c)
        changed_docs += c > 0
    s, c = pd.Series(slots), pd.Series(changes)
    return {"resumes": len(man), "field_slots_total": int(s.sum()), "field_slots_mean": round(float(s.mean()), 1),
            "field_slots_min": int(s.min()), "field_slots_max": int(s.max()),
            "skill_changes_total": int(c.sum()), "skill_changes_mean": round(float(c.mean()), 2),
            "resumes_with_changes": int(changed_docs),
            "skills_capped_total": int(man.SkillsCapped.sum())}


def write_timing_sheet(n: int = 10) -> pd.DataFrame:
    """Sheet for HR to time the current manual process on n resumes (stopwatch, per step)."""
    man = load_manifest().sample(n=n, random_state=config.SEED).sort_values("Code")
    df = pd.DataFrame({"file": man.File, "code": man.Code,
                       "t_find_employee_s": "", "t_read_resume_s": "", "t_compare_fields_s": "",
                       "t_type_updates_s": "", "t_total_s": "", "fields_updated": "", "errors_found_later": "",
                       "hr_operator": "", "date": ""})
    config.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.BASELINE_DIR / "manual_baseline_timesheet.csv"
    df.to_csv(out, index=False)
    return df
