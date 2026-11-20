"""Deterministic text-layer parser (non-LLM reference baseline).

Purpose: validate the ground truth and the evaluation harness with a known,
reproducible extractor, and quantify why a rules-based approach is not enough:
it is tied to one template and returns nothing for scanned PDFs (no text layer).
It is NOT the production OCR function.
"""
from __future__ import annotations

import re
import time

from ..ingestion import ResumeDocument, text_layer
from ..schema import (Certificate, Education, LanguageCert, ProjectEntry, ResumeExtraction,
                      SkillYears)
from .base import OCRProvider, OCRResult

DEGREE = [(r"^Bachelor of (.+)$", "Bachelor's Degree"), (r"^Master of (.+)$", "Master's Degree"),
          (r"^PhD in (.+)$", "PhD"), (r"^Associate Degree in (.+)$", "Associate Degree")]
_SKILL = re.compile(r"([A-Za-z#.+\-]+(?:\.js)?)\s*\((\d+(?:\.\d+)?)\s*yrs?\)")


def _flat(text: str) -> str:
    """Join wrapped lines; turn bullets into a '§' separator."""
    lines = [l.strip() for l in text.splitlines()]
    out: list[str] = []
    for l in lines:
        if not l:
            continue
        if l in ("•", "o"):
            out.append("§")
        else:
            out.append(l)
    s = " ".join(out)
    return re.sub(r"\s+", " ", s)


def _between(s: str, start: str, end: str) -> str:
    m = re.search(re.escape(start) + r"(.*?)" + end, s)
    return m.group(1).strip() if m else ""


def _education(it: str, version: str):
    """'YYYY – YYYY: <degree> in <school>'.
    v1 split on the first ' in ' -> wrong for 'PhD in <major> in <school>' and
    'Associate Degree in <major> in <school>' (Week 1 error analysis, issue IRM-W1-01).
    v2 strips the degree prefix first, then splits major / school on the next ' in '."""
    m = re.match(r"^(\d{4}) [–-] (\d{4}): (.+?)\.?$", it)
    if not m:
        return None
    start, grad, rest = int(m.group(1)), int(m.group(2)), m.group(3)
    lvl = major = school = None
    if version == "v1":
        mm = re.match(r"^(.+?) in (.+)$", rest)
        deg_txt, school = (mm.group(1), mm.group(2)) if mm else (rest, None)
        major = deg_txt
        for pat, label in DEGREE:
            if md := re.match(pat, deg_txt):
                lvl, major = label, md.group(1)
    else:
        for prefix, label in (("Bachelor of ", "Bachelor's Degree"), ("Master of ", "Master's Degree"),
                              ("PhD in ", "PhD"), ("Associate Degree in ", "Associate Degree")):
            if rest.startswith(prefix):
                lvl, rest = label, rest[len(prefix):]
                break
        major, _, school = rest.partition(" in ")
        school = school or None
    return Education(degree_level=lvl, major=major, school=school, start_year=start, graduation_year=grad)


def parse_text(text: str, version: str = "v2") -> ResumeExtraction:
    s = _flat(text)
    hdr = lambda k, nxt: _between(s, f"{k}:", nxt) or None  # noqa: E731
    x = ResumeExtraction(
        employee_code=int(c) if (c := hdr("ID", r" NAME:")) and c.isdigit() else None,
        full_name=hdr("NAME", r" POSITION/ROLE:"),
        position=hdr("POSITION/ROLE", r" PROJECT:"),
        current_project=hdr("PROJECT", r" EXPERIENCE SUMMARY"),
    )
    m = re.search(r"§ (?:(\d+)\+ years|(\d+) months) of experience as an? (.+?):", s)
    if m:
        x.total_experience_years = int(m.group(1)) if m.group(1) else None
        x.primary_stack = m.group(3)
    x.business_domain = _between(s, "Business domain experience:", r" TECHNICAL SUMMARY").rstrip(".") or None

    tech = _between(s, "TECHNICAL SUMMARY", r"EDUCATION & QUALIFICATIONS")
    items = {k.strip(): v.strip() for k, v in
             (it.split(":", 1) for it in tech.split("§") if ":" in it)}
    for key in ("Languages", "Frameworks & Libraries"):
        x.skills += [SkillYears(name=n, years=float(y)) for n, y in _SKILL.findall(items.get(key, ""))]
    x.frameworks = [f.strip() for f in items.get("Frameworks & Libraries", "").split(",")
                    if f.strip() and "yrs" not in f]
    x.databases = [d.strip() for d in items.get("Databases", "").split(",") if d.strip()]

    edu = _between(s, "EDUCATION & QUALIFICATIONS", r"PROFESSIONAL EXPERIENCE")
    for it in [i.strip() for i in edu.split("§") if i.strip()]:
        if re.match(r"^\d{4} [–-] \d{4}: ", it):
            x.education = _education(it, version) or x.education
        elif m := re.match(r"^(\d{4}): (.+)$", it):
            x.certificates.append(Certificate(name=m.group(2).strip(), year=int(m.group(1))))
        elif " - " in it:
            lang, lvl = it.split(" - ", 1)
            x.languages.append(LanguageCert(language=lang.strip(), level=lvl.strip()))

    exp = s.split("PROFESSIONAL EXPERIENCE", 1)[-1]
    for blk in re.split(r"(?=\[\d{2}/\d{4} [–-] (?:Now|\d{2}/\d{4})\] § Project:)", exp)[1:]:
        period = re.match(r"\[(.+?)\]", blk).group(1)
        name = _between(blk, "§ Project:", r" § Project description:")
        role = _between(blk, "§ Role:", r" § Responsibilities:") or None
        techs = _between(blk, "§ Technologies:", r"(?:\.\s|\.$)")
        x.projects.append(ProjectEntry(name=name, role=role, period=period,
                                       technologies=[t.strip() for t in techs.split(",") if t.strip()]))
    return x


class TextLayerParser(OCRProvider):
    name = "textlayer"

    def __init__(self, version: str = "v2"):
        self.version, self.model = version, f"rules-{version}"

    def extract(self, doc: ResumeDocument) -> OCRResult:
        t0 = time.perf_counter()
        txt = text_layer(doc.path)
        if len(txt.strip()) < 50:
            return OCRResult(None, self.name, self.model, raw_text=txt,
                             latency_s=time.perf_counter() - t0,
                             errors=["no text layer (scanned PDF) - rules parser cannot read it"])
        try:
            x = parse_text(txt, self.version)
            return OCRResult(x, self.name, self.model, raw_text=txt, latency_s=time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            return OCRResult(None, self.name, self.model, raw_text=txt, errors=[repr(e)],
                             latency_s=time.perf_counter() - t0)
