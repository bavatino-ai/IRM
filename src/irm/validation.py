"""Post-extraction checks: business rules + grounding. Failing fields go to HR review."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ground_truth import norm
from .schema import ResumeExtraction

DEGREES = {"associate degree", "bachelor's degree", "master's degree", "phd"}


@dataclass
class ValidationReport:
    review_fields: list[str] = field(default_factory=list)   # fields HR must confirm
    notes: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_fields)


def validate(x: ResumeExtraction, source_text: str = "") -> ValidationReport:
    rep = ValidationReport()
    if x.employee_code is None:
        rep.review_fields.append("employee_code"); rep.notes.append("no employee code - name fallback needed")
    if x.education.degree_level and norm(x.education.degree_level) not in DEGREES:
        rep.review_fields.append("education.degree_level")
    if (x.education.start_year and x.education.graduation_year
            and x.education.start_year >= x.education.graduation_year):
        rep.review_fields.append("education.graduation_year")
    if x.total_experience_years is not None:
        for s in x.skills:
            if s.years > x.total_experience_years + 1:
                rep.review_fields.append(f"skill:{s.name}")
                rep.notes.append(f"{s.name}: {s.years} yrs exceeds total experience")
    if not source_text:            # scanned PDF: no independent text to ground against
        rep.notes.append("grounding unavailable (no text layer) - sample-based human check applies")
        return rep
    src = norm(source_text)
    for fld in ("full_name", "position", "current_project", "business_domain"):
        v = getattr(x, fld)
        if v and norm(v) not in src:
            rep.review_fields.append(fld); rep.notes.append(f"{fld} not found in source text")
    for s in x.skills:
        if f"{norm(s.name)} ({s.years:g} yrs" not in src:
            rep.review_fields.append(f"skill:{s.name}"); rep.notes.append(f"{s.name} years not grounded")
    for c in x.certificates:
        if norm(c.name) not in src:
            rep.review_fields.append("certificates"); rep.notes.append(f"certificate '{c.name}' not grounded")
    return rep
