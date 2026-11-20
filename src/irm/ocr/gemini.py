"""Gemini vision model as the OCR + extraction function (short-term).

One call per resume: all rendered pages + the versioned prompt; the response is
constrained to the `ResumeExtraction` JSON schema and validated with Pydantic.
Responses are cached by (model, prompt version, file hash); tokens, latency and
cost are recorded for the model-tier comparison.
"""
from __future__ import annotations

import json
import time

from pydantic import ValidationError

from .. import config
from ..ingestion import ResumeDocument
from ..llm.cache import JsonCache, cost
from ..llm.gemini_client import Pacer, QuotaExhausted, classify
from ..prompts import DEFAULT_VERSION, PROMPTS
from ..schema import ResumeExtraction
from .base import OCRProvider, OCRResult


class GeminiOCR(OCRProvider):
    name = "gemini"

    def __init__(self, model: str = config.GEMINI_OCR_MODEL, prompt_version: str = DEFAULT_VERSION,
                 client=None, use_cache: bool = True, max_retries: int = 3):
        self.model, self.prompt_version = model, prompt_version
        self.cache = JsonCache() if use_cache else None
        self.max_retries = max_retries
        self._client = client            # injectable for offline tests
        self._pace = client is None      # pace only real API calls

    @property
    def client(self):
        if self._client is None:
            from ..llm.gemini_client import make_client
            self._client = make_client()          # pinned to API version v1beta (JSON mode)
        return self._client

    def _call(self, doc: ResumeDocument):
        from google.genai import types
        parts = [types.Part.from_bytes(data=png, mime_type="image/png") for png in doc.pages_png]
        parts.append(PROMPTS[self.prompt_version])
        return self.client.models.generate_content(
            model=self.model, contents=parts,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                               response_schema=ResumeExtraction))

    def extract(self, doc: ResumeDocument) -> OCRResult:
        key = JsonCache.key(self.model, self.prompt_version, doc.sha256)
        if self.cache and (hit := self.cache.get(key)):
            return OCRResult(ResumeExtraction.model_validate(hit["extraction"]) if hit["extraction"] else None,
                             self.name, self.model, self.prompt_version, hit["raw_text"], hit["latency_s"],
                             hit["input_tokens"], hit["output_tokens"],
                             cost(self.model, hit["input_tokens"], hit["output_tokens"]), True, hit["errors"])
        if not doc.pages_png:
            raise ValueError("document was loaded with render=False")
        errors, raw, resp, t0 = [], "", None, time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                if self._pace:
                    Pacer.wait()
                resp = self._call(doc)
                break
            except Exception as e:  # noqa: BLE001
                errors = [f"api: {e!r}"]
                action, wait = classify(e)
                if action == "abort":
                    raise QuotaExhausted(str(e)[:300]) from e
                if action == "fail" or attempt == self.max_retries - 1:
                    break                      # 4xx: retrying only burns quota
                time.sleep(wait)
        latency = time.perf_counter() - t0
        extraction = None
        in_tok = out_tok = 0
        if resp is not None:
            errors = []
            raw = resp.text or ""
            um = getattr(resp, "usage_metadata", None)
            in_tok = int(getattr(um, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(um, "candidates_token_count", 0) or 0)
            try:
                extraction = ResumeExtraction.model_validate(json.loads(raw))
            except (ValidationError, json.JSONDecodeError) as e:
                errors.append(f"schema: {e}")
        res = OCRResult(extraction, self.name, self.model, self.prompt_version, raw, latency, in_tok,
                        out_tok, cost(self.model, in_tok, out_tok), False, errors)
        if self.cache and resp is not None:
            self.cache.put(key, {"extraction": extraction.model_dump() if extraction else None,
                                 "raw_text": raw, "latency_s": latency, "input_tokens": in_tok,
                                 "output_tokens": out_tok, "errors": errors})
        return res
