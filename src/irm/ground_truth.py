"""Ground truth for extraction evaluation.

Expected values come from Employee_profile (unchanged fields) and the resume
manifest (fields the generator changed or derived: position, current project,
updated skill years). Both are flattened into the same `field -> value` form as
an extraction so they can be compared slot by slot.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from . import config
from .schema import ResumeExtraction


def norm(v) -> str:
    """Case/whitespace/punctuation-insensitive form used for string comparison."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = unicodedata.normalize("NFKC", str(v)).lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".")


def split(s) -> list[str]:
    return [x.strip() for x in str(s).split("|") if x.strip()] if pd.notna(s) else []


def load_profile() -> pd.DataFrame:
    return pd.read_csv(config.PROFILE_CSV, dtype={"Phone": str})


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(config.MANIFEST_CSV)


def flatten(x: ResumeExtraction) -> dict[str, object]:
    """Flatten an extraction into comparable slots. Lists become frozensets."""
    out: dict[str, object] = {
        "employee_code": x.employee_code,
        "full_name": norm(x.full_name) or None,
        "position": norm(x.position) or None,
        "current_project": norm(x.current_project) or None,
        "primary_stack": norm(x.primary_stack) or None,
        "total_experience_years": x.total_experience_years,
        "business_domain": norm(x.business_domain) or None,
        "education.degree_level": norm(x.education.degree_level) or None,
        "education.major": norm(x.education.major) or None,
        "education.school": norm(x.education.school) or None,
        "education.graduation_year": x.education.graduation_year,
    }
    for s in x.skills:
        out[f"skill:{norm(s.name)}"] = float(s.years)
    out["languages"] = frozenset((norm(l.language), norm(l.level)) for l in x.languages)
    out["certificates"] = frozenset((norm(c.name), c.year) for c in x.certificates)
    out["projects"] = frozenset(norm(p.name) for p in x.projects)
    return out


def expected_for(code: int, profile: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    r = profile.loc[profile.Code == code].iloc[0]
    m = manifest.loc[manifest.Code == code].iloc[0]
    exp: dict[str, object] = {
        "employee_code": int(r.Code),
        "full_name": norm(r.Fullname),
        "position": norm(m.Position),
        "current_project": norm(m.CurrentProject),
        "primary_stack": norm(split(r.Techstack)[0]),
        "total_experience_years": int(r.CompanyTenure) // 12 if int(r.CompanyTenure) >= 12 else None,
        "business_domain": norm(r.Domain),
        "education.degree_level": norm(r.Education),
        "education.major": norm(r.Major),
        "education.school": norm(r.School),
        "education.graduation_year": int(r.Graduation),
    }
    for sk in config.YEAR_SKILLS:
        val = float(m[f"{sk}_resume"])
        if val > 0:
            exp[f"skill:{norm(sk)}"] = val
    exp["languages"] = frozenset((norm(a), norm(b)) for a, b in zip(split(r.Language), split(r.Level)))
    exp["certificates"] = (frozenset({(norm(r.Certificate), int(r.YearOfIssue))})
                           if pd.notna(r.YearOfIssue) else frozenset())
    exp["projects"] = frozenset(norm(p) for p in split(r.Project)[: int(m.ProjectsRendered)])
    return exp


def changed_skills(code: int, manifest: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Known annual changes (base -> resume) used for change-detection evaluation."""
    m = manifest.loc[manifest.Code == code].iloc[0]
    return {sk: (float(m[f"{sk}_base"]), float(m[f"{sk}_resume"]))
            for sk in config.YEAR_SKILLS if float(m[f"{sk}_base"]) != float(m[f"{sk}_resume"])}
