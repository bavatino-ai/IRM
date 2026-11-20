"""Extraction schema aligned with Employee_profile (Week 1 deliverable).

The same Pydantic model is (1) passed to Gemini as `response_schema`, (2) used
to validate every OCR output, and (3) mapped field-by-field to the
Employee_profile columns for evaluation and change detection.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SkillYears(BaseModel):
    """A programming language / framework with years of experience as printed."""
    name: str = Field(description="Skill name exactly as printed, e.g. 'Python', 'C#', '.NET', 'React.js'")
    years: float = Field(description="Years of experience printed in brackets, e.g. 2.5")


class LanguageCert(BaseModel):
    language: str = Field(description="Spoken language, e.g. 'English'")
    level: str = Field(description="Proficiency test and score, e.g. 'IELTS 7.0', 'JLPT N2'")


class Certificate(BaseModel):
    name: str = Field(description="Professional certificate name (not a language test)")
    year: Optional[int] = Field(default=None, description="Year of issue")


class Education(BaseModel):
    degree_level: Optional[str] = Field(default=None, description="One of: Associate Degree, Bachelor's Degree, Master's Degree, PhD")
    major: Optional[str] = None
    school: Optional[str] = None
    start_year: Optional[int] = None
    graduation_year: Optional[int] = None


class ProjectEntry(BaseModel):
    name: str
    role: Optional[str] = None
    technologies: list[str] = Field(default_factory=list)
    period: Optional[str] = Field(default=None, description="As printed, e.g. '10/2025 – Now'")


class ResumeExtraction(BaseModel):
    """Structured content of one annual resume."""
    employee_code: Optional[int] = Field(default=None, description="Value after 'ID:'")
    full_name: Optional[str] = Field(default=None, description="Value after 'NAME:'")
    position: Optional[str] = Field(default=None, description="Value after 'POSITION/ROLE:'")
    current_project: Optional[str] = Field(default=None, description="Value after 'PROJECT:'")
    primary_stack: Optional[str] = Field(default=None, description="Role in 'N+ years of experience as a <role>'")
    total_experience_years: Optional[int] = Field(default=None, description="N in 'N+ years of experience'")
    business_domain: Optional[str] = Field(default=None, description="Value after 'Business domain experience:'")
    skills: list[SkillYears] = Field(default_factory=list, description="Only items that carry '(N yrs)'")
    frameworks: list[str] = Field(default_factory=list, description="Frameworks & Libraries without years")
    databases: list[str] = Field(default_factory=list)
    languages: list[LanguageCert] = Field(default_factory=list)
    certificates: list[Certificate] = Field(default_factory=list)
    education: Education = Field(default_factory=Education)
    projects: list[ProjectEntry] = Field(default_factory=list, description="Most recent first")

    @field_validator("full_name", "position", "current_project", "business_domain", "primary_stack")
    @classmethod
    def _strip(cls, v):
        return v.strip().rstrip(".") if isinstance(v, str) else v

    @field_validator("skills")
    @classmethod
    def _years_range(cls, v):
        for s in v:
            if not 0 <= s.years <= 60:
                raise ValueError(f"implausible years for {s.name}: {s.years}")
        return v


# ---------------------------------------------------------------- mapping table
# Extraction field -> Employee_profile column (or manifest column) and how it is compared.
FIELD_MAP = [
    # (eval_field, source, column, comparison, notes)
    ("employee_code", "profile", "Code", "exact", "Primary matching key"),
    ("full_name", "profile", "Fullname", "normalised", "Fallback matching key"),
    ("position", "manifest", "Position", "normalised", "Title label + primary stack"),
    ("current_project", "manifest", "CurrentProject", "normalised", "First item of Project"),
    ("primary_stack", "profile", "Techstack[0]", "normalised", "First item of Techstack"),
    ("total_experience_years", "profile", "CompanyTenure // 12", "exact", "Months -> whole years"),
    ("business_domain", "profile", "Domain", "normalised", ""),
    ("education.degree_level", "profile", "Education", "normalised", "Derived from degree wording"),
    ("education.major", "profile", "Major", "normalised", ""),
    ("education.school", "profile", "School", "normalised", ""),
    ("education.graduation_year", "profile", "Graduation", "exact", ""),
    ("skill:<name>", "manifest", "<skill>_resume", "numeric ±0", "9 skill-year columns; updated values"),
    ("languages", "profile", "Language + Level", "set", "Pairs (language, test)"),
    ("certificates", "profile", "Certificate + YearOfIssue", "set", "Nullable (19 records empty)"),
    ("projects", "profile", "Project[:ProjectsRendered]", "set", "Project names rendered on the resume"),
]
