"""Week 2: final cleaned dataset + SQLite system of record for the standalone pilot.

Design
  employees              one row per employee, non-sensitive profile + employment fields
  employee_skill_years   long table of the 9 skill-year columns (value > 0 only)
  employee_projects      Project / Role / Technology lists exploded, ordinal 1 = most recent
  employee_languages     Language + Level pairs
  employee_frameworks    Framework list
  hr_restricted          performance / satisfaction fields (role-based access; never embedded)
  hr_sensitive           income, age, gender, marital, dependents, attrition, contact (never sent to an LLM)
  change_log             audit trail for HR-approved updates (populated from Week 3)
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from . import config
from .ground_truth import split

DDL_EXTRA = """
CREATE TABLE IF NOT EXISTS change_log (
  change_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  code          INTEGER NOT NULL REFERENCES employees(code),
  field         TEXT NOT NULL,
  old_value     TEXT,
  new_value     TEXT,
  source_file   TEXT,
  source_sha256 TEXT,
  extractor     TEXT,
  status        TEXT CHECK(status IN ('proposed','approved','rejected')) DEFAULT 'proposed',
  reviewer      TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
  decided_at    TEXT
);
"""


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return the cleaned frame and a log of every transformation applied."""
    log: list[dict] = []
    df = df.copy()
    # Phone: read as text; restore the international '+' lost when parsed as a number.
    fixed = (~df.Phone.astype(str).str.startswith("+")).sum()
    df["Phone"] = df.Phone.astype(str).str.strip().map(lambda p: p if p.startswith("+") else f"+{p}")
    log.append({"step": "Phone kept as text, '+' prefix enforced", "rows_changed": int(fixed)})
    for c in ["Fullname", "School", "Major", "Domain", "Title", "Certificate"]:
        before = df[c].copy()
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        log.append({"step": f"{c}: trim / collapse whitespace",
                    "rows_changed": int((before.fillna("") != df[c].fillna("")).sum())})
    df["YearOfIssue"] = df.YearOfIssue.astype("Int64")
    log.append({"step": "YearOfIssue float -> nullable integer; empty certificate kept as NULL "
                        "(no imputation)", "rows_changed": int(df.YearOfIssue.isna().sum())})
    df["Email"] = df.Email.str.lower().str.strip()
    df["TitleRank"] = df.Title.map({"Fresher": 0, "Junior": 1, "Mid-level": 2, "Senior": 3, "Architect": 4})
    log.append({"step": "TitleRank ordinal added for filtering", "rows_changed": len(df)})
    dup = int(df.Code.duplicated().sum())
    log.append({"step": "Code uniqueness check", "duplicates": dup})
    assert dup == 0
    return df, log


def build(db_path=config.SQLITE_DB) -> dict:
    raw = pd.read_csv(config.PROFILE_CSV, dtype={"Phone": str})
    df, log = clean(raw)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED / "employee_profile_clean.csv", index=False)

    core_cols = ["Code", "Fullname", "School", "Major", "Graduation", "Education", "Certificate", "YearOfIssue",
                 "AtCompany", "CompanyTenure", "Domain", "Title", "TitleRank"]
    employees = df[core_cols].rename(columns=str.lower)
    employees["primary_stack"] = df.Techstack.map(lambda s: split(s)[0])
    employees["techstack_history"] = df.Techstack
    employees["tenure_years"] = (df.CompanyTenure / 12).round(1)

    skills = df.melt(id_vars="Code", value_vars=config.YEAR_SKILLS, var_name="skill", value_name="years")
    skills = skills[skills.years > 0].rename(columns={"Code": "code"})
    proj_rows, lang_rows, fw_rows = [], [], []
    for r in df.itertuples():
        for i, (p, ro, t) in enumerate(zip(split(r.Project), split(r.Role), split(r.Technology)), 1):
            proj_rows.append({"code": r.Code, "ordinal": i, "project": p, "role": ro, "technologies": t})
        for a, b in zip(split(r.Language), split(r.Level)):
            lang_rows.append({"code": r.Code, "language": a, "level": b})
        for f in split(r.Framework):
            fw_rows.append({"code": r.Code, "framework": f})
    restricted = df[["Code"] + config.RESTRICTED_FIELDS].rename(columns=str.lower)
    sensitive = df[["Code"] + config.SENSITIVE_FIELDS].rename(columns=str.lower)

    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    employees.to_sql("employees", con, index=False)
    skills.to_sql("employee_skill_years", con, index=False)
    pd.DataFrame(proj_rows).to_sql("employee_projects", con, index=False)
    pd.DataFrame(lang_rows).to_sql("employee_languages", con, index=False)
    pd.DataFrame(fw_rows).to_sql("employee_frameworks", con, index=False)
    restricted.to_sql("hr_restricted", con, index=False)
    sensitive.to_sql("hr_sensitive", con, index=False)
    con.executescript(DDL_EXTRA + """
        CREATE UNIQUE INDEX ix_emp_code ON employees(code);
        CREATE INDEX ix_skill ON employee_skill_years(skill, years);
        CREATE INDEX ix_proj ON employee_projects(code);
    """)
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
              ["employees", "employee_skill_years", "employee_projects", "employee_languages",
               "employee_frameworks", "hr_restricted", "hr_sensitive", "change_log"]}
    con.commit(); con.close()
    report = {"db": db_path.name, "tables": counts, "cleaning_log": log}
    (config.PROCESSED / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
