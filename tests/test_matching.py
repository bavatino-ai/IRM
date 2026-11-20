from irm.ground_truth import load_profile
from irm.matching import RecordMatcher
from irm.schema import ResumeExtraction

M = RecordMatcher(load_profile())


def test_code_and_name_agree():
    r = M.match(ResumeExtraction(employee_code=20003, full_name="Ta Long Binh"))
    assert r.code == 20003 and r.method == "code" and not r.needs_review


def test_code_name_conflict_goes_to_review():
    r = M.match(ResumeExtraction(employee_code=20003, full_name="Le Kien Minh"))
    assert r.method == "code_name_conflict" and r.needs_review


def test_name_fallback_and_fuzzy():
    assert M.match(ResumeExtraction(full_name="Ta Long Binh")).code == 20003
    r = M.match(ResumeExtraction(full_name="Ta Long Bihn"))      # OCR transposition
    assert r.code == 20003 and r.method == "fuzzy_name" and r.needs_review
    assert M.match(ResumeExtraction(full_name="Nobody Known")).method == "unmatched"
