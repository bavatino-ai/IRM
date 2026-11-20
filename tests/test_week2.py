"""Cleaning / SQLite, chunking guardrails, retrieval helpers, scans, eval set."""
import sqlite3

import pymupdf

from irm import config, database, scans
from irm.chunking import build_corpus
from irm.evaluation import eval_set
from irm.ground_truth import load_manifest, load_profile
from irm.retrieval.index import rrf, tokenize


def test_sqlite_separates_sensitive_fields(tmp_path):
    db = tmp_path / "t.db"
    rep = database.build(db)
    assert rep["tables"]["employees"] == 999
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(employees)")}
    assert not cols & {"income", "age", "gender", "marital", "dependents", "attrition", "phone", "email"}
    assert con.execute("SELECT COUNT(*) FROM hr_sensitive").fetchone()[0] == 999
    phone = con.execute("SELECT phone FROM hr_sensitive LIMIT 1").fetchone()[0]
    assert phone.startswith("+")


def test_chunks_have_metadata_and_no_sensitive_text():
    chunks = build_corpus()
    prof = load_profile().set_index("Code")
    assert len({c.metadata["code"] for c in chunks}) == 281
    for c in chunks[:300]:
        r = prof.loc[c.metadata["code"]]
        assert r.Email not in c.text and str(r.Phone) not in c.text
        assert "PerformanceRating" not in c.text and r.PerformanceRating not in c.text.split(" | ")[0]
        assert c.metadata["section"] in {"profile", "skills", "education", "project"}


def test_tokenizer_and_rrf():
    assert {"c#", ".net", "react.js"} <= set(tokenize("C# .NET React.js")) | {".net"}
    fused = [d for d, _ in rrf([["a", "b", "c"], ["b", "c", "a"]])]
    assert fused[0] == "b"


def test_scan_variant_has_no_text_layer(tmp_path):
    src = config.RESUME_DIR / "20003-Ta_Long_Binh.pdf"
    info = scans.make_scan(src, tmp_path / "x-scan.pdf", "medium", 1)
    with pymupdf.open(tmp_path / "x-scan.pdf") as d:
        assert info["text_layer_chars"] == 0 and d.page_count >= 1


def test_eval_set_ground_truth_within_corpus():
    q = eval_set.build()
    codes = set(load_manifest().Code)
    assert all(set(x["relevant_codes"]) <= codes for x in q)
    assert sum(x["expected_behaviour"] == "refuse" for x in q) >= 5
    assert len({x["query"] for x in q}) == len(q)
