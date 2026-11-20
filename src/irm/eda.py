"""Week 2 EDA on Employee_profile and the Employee_resume corpus. Writes figures + eda_summary.json."""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pymupdf  # noqa: E402

from . import config  # noqa: E402
from .ground_truth import load_manifest, load_profile, split  # noqa: E402

SECTIONS = ["EXPERIENCE SUMMARY", "TECHNICAL SUMMARY", "EDUCATION & QUALIFICATIONS",
            "PROFESSIONAL EXPERIENCE", "End of Document"]
INK, ACCENT, GREY = "#1F3B5C", "#2E75B6", "#A6A6A6"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors="#404040", labelsize=9)


def _save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=160)
    plt.close(fig)


def profile_eda(df: pd.DataFrame) -> dict:
    raw = pd.read_csv(config.PROFILE_CSV)                 # default parsing, to expose type issues
    res = {"rows": len(df), "columns": df.shape[1],
           "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
           "duplicates": {"Code": int(df.Code.duplicated().sum()), "Email": int(df.Email.duplicated().sum()),
                          "Fullname": int(df.Fullname.duplicated().sum())},
           "phone_parsed_as_number_by_default": bool(pd.api.types.is_numeric_dtype(raw.Phone)),
           "code_range": [int(df.Code.min()), int(df.Code.max())]}
    for c in ["Title", "Domain", "Education", "PerformanceRating", "TeamSize", "Language"]:
        res[f"dist_{c}"] = df[c].value_counts().to_dict()
    yrs = df[config.YEAR_SKILLS]
    res["skill_holders"] = {c: int((yrs[c] > 0).sum()) for c in config.YEAR_SKILLS}
    res["skill_years_mean_nonzero"] = {c: round(float(yrs.loc[yrs[c] > 0, c].mean()), 2) for c in config.YEAR_SKILLS}
    tenure_y = df.CompanyTenure / 12
    res["consistency"] = {
        "AtCompany_gt_CompanyTenure": int((df.AtCompany > df.CompanyTenure).sum()),
        "skill_years_gt_tenure": int((yrs.max(axis=1) > tenure_y + 1e-9).sum()),
        "CSharp_ne_DotNET": int((df["C#"] != df[".NET"]).sum()),
        "HTML_ne_CSS": int((df["HTML"] != df["CSS"]).sum()),
        "graduation_after_2026": int((df.Graduation > 2026).sum()),
        "age_at_graduation_lt_18": int(((df.Graduation - (2026 - df.Age)) < 18).sum()),
        "certificate_year_before_graduation_minus_4": int((df.YearOfIssue < df.Graduation - 4).sum()),
        "list_length_mismatch_Project_Role_Technology": int(sum(
            not (len(split(p)) == len(split(r)) == len(split(t)))
            for p, r, t in zip(df.Project, df.Role, df.Technology))),
        "list_length_mismatch_Language_Level": int(sum(len(split(a)) != len(split(b))
                                                       for a, b in zip(df.Language, df.Level))),
    }
    res["projects_per_employee"] = df.Project.map(lambda s: len(split(s))).describe().round(2).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    order = ["Fresher", "Junior", "Mid-level", "Senior", "Architect"]
    df.Title.value_counts().reindex(order).plot.bar(ax=axes[0], color=ACCENT); _style(axes[0], "Employees by title")
    df.Domain.value_counts().plot.bar(ax=axes[1], color=ACCENT); _style(axes[1], "Employees by business domain")
    df.Education.value_counts().plot.bar(ax=axes[2], color=ACCENT); _style(axes[2], "Highest education")
    for a in axes:
        a.set_xlabel(""); a.tick_params(axis="x", rotation=30)
    _save(fig, "w2_profile_distributions.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    pd.Series(res["skill_holders"]).sort_values().plot.barh(ax=ax, color=ACCENT)
    _style(ax, "Employees holding each skill (years > 0), n = 999"); ax.set_xlabel("employees")
    _save(fig, "w2_skill_holders.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(tenure_y, bins=30, color=ACCENT); _style(ax, "Total experience (CompanyTenure, years)")
    ax.set_xlabel("years"); ax.set_ylabel("employees")
    _save(fig, "w2_tenure_hist.png")
    return res


def resume_eda(man: pd.DataFrame) -> dict:
    rows = []
    for f in sorted(config.RESUME_DIR.glob("*.pdf")):
        with pymupdf.open(f) as d:
            txt = "".join(p.get_text() for p in d)
            rows.append({"file": f.name, "code": int(f.name[:5]), "pages": d.page_count, "chars": len(txt),
                         "words": len(txt.split()), "bytes": f.stat().st_size,
                         **{f"has_{s}": s in txt for s in SECTIONS},
                         "projects_on_resume": len(re.findall(r"Project: ", txt))})
    r = pd.DataFrame(rows)
    r.to_csv(config.PROCESSED / "resume_inventory.csv", index=False)
    deltas = pd.concat([man[f"{s}_resume"] - man[f"{s}_base"] for s in config.YEAR_SKILLS])
    changed = int((deltas != 0).sum())
    res = {"files": len(r), "codes_in_profile": int(r.code.isin(load_profile().Code).sum()),
           "pages": r.pages.value_counts().sort_index().to_dict(), "total_bytes": int(r.bytes.sum()),
           "words": r.words.describe().round(1).to_dict(),
           "section_coverage": {s: int(r[f"has_{s}"].sum()) for s in SECTIONS},
           "projects_on_resume_eq_manifest": int((r.set_index("code").projects_on_resume ==
                                                 man.set_index("Code").ProjectsRendered).sum()),
           "projects_truncated_resumes": int((man.ProjectsRendered < man.ProjectsInSource).sum()),
           "projects_dropped_total": int((man.ProjectsInSource - man.ProjectsRendered).sum()),
           "skill_values_changed": changed,
           "skill_delta_distribution": deltas[deltas != 0].value_counts().sort_index().to_dict(),
           "resumes_with_capped_skills": int((man.SkillsCapped > 0).sum()),
           "title_distribution_resumes": man.Title.value_counts().to_dict()}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    r.pages.value_counts().sort_index().plot.bar(ax=axes[0], color=ACCENT)
    _style(axes[0], "Resume length (pages), n = 281"); axes[0].set_xlabel("pages"); axes[0].tick_params(axis="x", rotation=0)
    pd.Series(res["skill_delta_distribution"]).plot.bar(ax=axes[1], color=ACCENT)
    _style(axes[1], f"Skill-year changes vs. dataset ({changed} values)")
    axes[1].set_xlabel("years added"); axes[1].tick_params(axis="x", rotation=0)
    _save(fig, "w2_resume_profile.png")
    return res


def run() -> dict:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    df, man = load_profile(), load_manifest()
    out = {"profile": profile_eda(df), "resumes": resume_eda(man)}
    (config.PROCESSED / "eda_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
