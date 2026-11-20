"""Schema, rules baseline, evaluation harness, validation."""
import pytest
from pydantic import ValidationError

from irm import config
from irm.evaluation.extraction_eval import compare, summarise
from irm.ground_truth import expected_for, flatten, load_manifest, load_profile
from irm.ingestion import load, text_layer
from irm.ocr.textlayer import parse_text
from irm.schema import ResumeExtraction, SkillYears
from irm.validation import validate

PROF, MAN = load_profile(), load_manifest()
F_20003 = config.RESUME_DIR / "20003-Ta_Long_Binh.pdf"
F_PHD = config.RESUME_DIR / "20034-Mai_Huu_Khoa.pdf"          # 'PhD in <major> in <school>'


def test_schema_rejects_implausible_years():
    with pytest.raises(ValidationError):
        ResumeExtraction(skills=[SkillYears(name="Python", years=99)])


def test_ingestion_detects_text_layer():
    doc = load(F_20003, render=False)
    assert doc.has_text_layer and doc.code_hint == 20003 and doc.n_pages >= 1


def test_rules_v2_matches_ground_truth_exactly():
    x = parse_text(text_layer(F_20003), "v2")
    slots = compare(20003, expected_for(20003, PROF, MAN), flatten(x), text_layer(F_20003))
    assert all(s.outcome == "TP" for s in slots)
    assert x.skills and {s.name for s in x.skills} >= {"PHP", "Golang"}


def test_rules_v1_reproduces_education_issue():
    v1 = parse_text(text_layer(F_PHD), "v1")
    v2 = parse_text(text_layer(F_PHD), "v2")
    assert v1.education.degree_level is None            # bug IRM-W1-01
    assert v2.education.degree_level == "PhD" and " in " not in v2.education.school


def test_error_taxonomy():
    exp = {"full_name": "a b", "position": "senior dev", "current_project": "crm system",
           "skill:python": 3.0, "projects": frozenset({"crm system"})}
    pred = {"full_name": None, "position": "crm system", "current_project": "crm system",
            "skill:python": 4.0, "skill:golang": 2.0, "projects": frozenset({"crm system", "mars rover"})}
    out = {(s.field, s.error) for s in compare(1, exp, pred, source_text="... crm system ... python (3 yrs)")}
    assert ("full_name", "missing") in out
    assert ("position", "misplaced") in out
    assert ("skill:python", "invented") in out        # 4.0 is not in the source
    assert ("skill:golang", "invented") in out
    assert ("projects", "invented") in out
    s = summarise(compare(1, exp, pred))
    assert 0 < s["overall"]["f1"] < 1


def test_validation_flags_ungrounded_values():
    x = parse_text(text_layer(F_20003), "v2")
    assert not validate(x, text_layer(F_20003)).needs_review
    x.skills[0].years += 3
    rep = validate(x, text_layer(F_20003))
    assert rep.needs_review and any(f.startswith("skill:") for f in rep.review_fields)
