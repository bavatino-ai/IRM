"""Gemini OCR provider exercised with a fake client - no network, no key."""
import json
from types import SimpleNamespace

from irm import config
from irm.ingestion import load, text_layer
from irm.llm.cache import JsonCache
from irm.ocr.gemini import GeminiOCR
from irm.ocr.textlayer import parse_text

F = config.RESUME_DIR / "20003-Ta_Long_Binh.pdf"


class FakeModels:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        assert config.response_schema is not None and config.temperature == 0.0
        assert any(getattr(p, "inline_data", None) for p in contents[:-1])      # page images attached
        return SimpleNamespace(text=self.payload,
                               usage_metadata=SimpleNamespace(prompt_token_count=1500, candidates_token_count=700))


def _client(payload):
    return SimpleNamespace(models=FakeModels(payload))


def test_gemini_valid_json_and_cache(tmp_path):
    payload = parse_text(text_layer(F)).model_dump_json()
    cli = _client(payload)
    ocr = GeminiOCR(model="fake-model", prompt_version="v2", client=cli)
    ocr.cache = JsonCache(tmp_path)
    doc = load(F)
    r1 = ocr.extract(doc)
    assert r1.ok and r1.extraction.employee_code == 20003 and r1.input_tokens == 1500
    r2 = ocr.extract(doc)
    assert r2.cached and cli.models.calls == 1


def test_gemini_schema_violation_is_reported(tmp_path):
    bad = json.dumps({"employee_code": "not-a-number", "skills": [{"name": "Python", "years": 500}]})
    ocr = GeminiOCR(model="fake-model", client=_client(bad), use_cache=False)
    r = ocr.extract(load(F))
    assert not r.ok and r.extraction is None and r.errors[0].startswith("schema")


def test_error_policy():
    from irm.llm.gemini_client import classify
    daily = Exception("429 RESOURCE_EXHAUSTED ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
    minute = Exception("429 RESOURCE_EXHAUSTED ... Please retry in 12.5s.")
    assert classify(daily)[0] == "abort"
    assert classify(minute) == ("retry", 13.5)
    assert classify(Exception("400 INVALID_ARGUMENT. JSON mode is not enabled"))[0] == "fail"
    assert classify(Exception("404 NOT_FOUND"))[0] == "fail"


def test_client_pins_v1beta(monkeypatch):
    from irm import config
    from irm.llm import gemini_client
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    c = gemini_client.make_client()
    assert c._api_client._http_options.api_version == "v1beta"


def test_daily_quota_aborts_batch_without_scoring_unrun_docs(tmp_path, monkeypatch):
    from irm import extraction_runner
    calls = {"n": 0}

    class Boom:
        def generate_content(self, **kw):
            calls["n"] += 1
            raise Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    monkeypatch.setattr(config, "EXTRACTION_DIR", tmp_path)
    ocr = GeminiOCR(model="fake", client=SimpleNamespace(models=Boom()), use_cache=False)
    files = sorted(config.RESUME_DIR.glob("*.pdf"))[:3]
    s = extraction_runner.run(ocr, files, "quota_test")
    assert calls["n"] == 1                       # no retries, batch stopped at first daily-quota error
    assert s["aborted"] and s["documents"] == 0 and s["documents_planned"] == 3
