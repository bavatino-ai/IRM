"""Swapped roles (Claude OCR / answer, Gemini judge) exercised with fake clients - no network, no key."""
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pandas as pd
import pytest

from irm import config
from irm.evaluation import answer_eval
from irm.ingestion import load, text_layer
from irm.llm.cache import JsonCache
from irm.llm.claude_client import ProviderUnavailable, is_fatal
from irm.ocr.claude import ClaudeOCR
from irm.ocr.textlayer import parse_text

F = config.RESUME_DIR / "20003-Ta_Long_Binh.pdf"


class FakeMessages:
    def __init__(self, parsed=None, stop_reason="end_turn", exc=None):
        self.parsed, self.stop_reason, self.exc, self.calls = parsed, stop_reason, exc, []

    def parse(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason, stop_details=None,
                               content=[SimpleNamespace(type="text", text=self.parsed.model_dump_json()
                                                        if self.parsed else "")],
                               usage=SimpleNamespace(input_tokens=4000, output_tokens=900))


def _client(**kw):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(**kw)))


def _bad_request(msg):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def test_claude_ocr_structured_output_and_cache(tmp_path):
    cli = _client(parsed=parse_text(text_layer(F)))
    ocr = ClaudeOCR(model="claude-test", prompt_version="v2", client=cli)
    ocr.cache = JsonCache(tmp_path)
    doc = load(F)
    r1 = ocr.extract(doc)
    assert r1.ok and r1.extraction.employee_code == 20003 and r1.input_tokens == 4000
    call = cli.beta.messages.calls[0]
    assert call["output_format"].__name__ == "ResumeExtraction" and "temperature" not in call
    assert [b["type"] for b in call["messages"][0]["content"]][-1] == "text"
    assert sum(b["type"] == "image" for b in call["messages"][0]["content"]) == doc.n_pages
    assert ocr.extract(doc).cached and len(cli.beta.messages.calls) == 1


def test_claude_ocr_refusal_is_reported():
    r = ClaudeOCR(model="claude-test", client=_client(parsed=None, stop_reason="refusal"),
                  use_cache=False).extract(load(F))
    assert not r.ok and r.errors[0].startswith("refusal")


def test_account_errors_abort_the_batch():
    err = _bad_request("This API key is not scoped to a workspace")
    assert is_fatal(err) and not is_fatal(_bad_request("messages: roles must alternate"))
    with pytest.raises(ProviderUnavailable):
        ClaudeOCR(model="claude-test", client=_client(exc=err), use_cache=False).extract(load(F))


def test_answer_eval_summary_separates_guardrails_and_false_refusals():
    df = pd.DataFrame([
        {"category": "lookup", "expected": "answer", "answer": "a", "refused": False, "groundedness": 1.0,
         "answer_relevance": 0.8, "context_relevance": 0.6, "correctness": None, "context_hit": True,
         "answer_latency_s": 2.0, "answer_input_tokens": 100, "answer_output_tokens": 20},
        {"category": "lookup", "expected": "answer", "answer": "b", "refused": True, "groundedness": 0.0,
         "answer_relevance": 0.0, "context_relevance": 0.4, "correctness": None, "context_hit": False,
         "answer_latency_s": 4.0, "answer_input_tokens": 100, "answer_output_tokens": 20},
        {"category": "guardrail", "expected": "refuse", "answer": "c", "refused": True, "groundedness": 1.0,
         "answer_relevance": 1.0, "context_relevance": 0.0, "correctness": None, "context_hit": None,
         "answer_latency_s": 1.0, "answer_input_tokens": 100, "answer_output_tokens": 20},
    ])
    s = answer_eval.summarise(df)
    assert s["judged"] == 3 and s["groundedness"] == 0.5
    assert s["guardrail_refusal_rate"] == 1.0 and s["false_refusal_rate"] == 0.5 and s["context_hit_rate"] == 0.5


def test_subset_takes_n_per_category():
    qs = [{"qid": str(i), "category": c} for i, c in enumerate("aaabbc")]
    assert [q["qid"] for q in answer_eval.subset(qs, 2)] == ["0", "1", "3", "4", "5"]
