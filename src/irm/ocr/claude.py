"""Claude vision model as the OCR + extraction function (swapped-role experiment).

Same contract as `GeminiOCR`: one call per resume with all rendered pages plus the
versioned prompt; the response is constrained to the `ResumeExtraction` schema
(structured outputs) and validated with Pydantic. Responses are cached by
(model, prompt version, file hash); tokens, latency and cost are recorded.
"""
from __future__ import annotations

import base64
import json
import time

from pydantic import ValidationError

from .. import config
from ..ingestion import ResumeDocument
from ..llm.cache import JsonCache, cost
from ..llm.claude_client import FALLBACK_BETA, ProviderUnavailable, is_fatal, make_client, text_of
from ..prompts import DEFAULT_VERSION, PROMPTS
from ..schema import ResumeExtraction
from .base import OCRProvider, OCRResult


class ClaudeOCR(OCRProvider):
    name = "claude"

    def __init__(self, model: str = config.CLAUDE_OCR_MODEL, prompt_version: str = DEFAULT_VERSION,
                 client=None, use_cache: bool = True, effort: str = "medium"):
        self.model, self.prompt_version, self.effort = model, prompt_version, effort
        self.cache = JsonCache() if use_cache else None
        self._client = client            # injectable for offline tests

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def _call(self, doc: ResumeDocument):
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                "data": base64.standard_b64encode(png).decode("ascii")}}
                   for png in doc.pages_png]
        content.append({"type": "text", "text": PROMPTS[self.prompt_version]})
        return self.client.beta.messages.parse(
            model=self.model, max_tokens=16000,
            betas=[FALLBACK_BETA], fallbacks="default",
            output_config={"effort": self.effort},
            output_format=ResumeExtraction,
            messages=[{"role": "user", "content": content}])

    def extract(self, doc: ResumeDocument) -> OCRResult:
        key = JsonCache.key(self.model, self.prompt_version, doc.sha256)
        if self.cache and (hit := self.cache.get(key)):
            return OCRResult(ResumeExtraction.model_validate(hit["extraction"]) if hit["extraction"] else None,
                             self.name, self.model, self.prompt_version, hit["raw_text"], hit["latency_s"],
                             hit["input_tokens"], hit["output_tokens"],
                             cost(self.model, hit["input_tokens"], hit["output_tokens"]), True, hit["errors"])
        if not doc.pages_png:
            raise ValueError("document was loaded with render=False")
        t0 = time.perf_counter()
        try:
            resp = self._call(doc)              # the SDK already retries 429 / 5xx / connection errors
        except Exception as e:  # noqa: BLE001
            if is_fatal(e):
                raise ProviderUnavailable(str(e)[:300]) from e
            return OCRResult(None, self.name, self.model, self.prompt_version, "",
                             time.perf_counter() - t0, errors=[f"api: {e!r}"[:500]])
        latency = time.perf_counter() - t0
        raw = text_of(resp)
        in_tok, out_tok = int(resp.usage.input_tokens or 0), int(resp.usage.output_tokens or 0)
        errors, extraction = [], None
        if resp.stop_reason == "refusal":
            errors.append(f"refusal: {getattr(resp.stop_details, 'category', None)}")
        elif resp.stop_reason == "max_tokens":
            errors.append("truncated: max_tokens reached")
        else:
            try:
                extraction = resp.parsed_output or ResumeExtraction.model_validate(json.loads(raw))
            except (ValidationError, json.JSONDecodeError) as e:
                errors.append(f"schema: {e}")
        res = OCRResult(extraction, self.name, self.model, self.prompt_version, raw, latency, in_tok,
                        out_tok, cost(self.model, in_tok, out_tok), False, errors)
        if self.cache:
            self.cache.put(key, {"extraction": extraction.model_dump() if extraction else None,
                                 "raw_text": raw, "latency_s": latency, "input_tokens": in_tok,
                                 "output_tokens": out_tok, "errors": errors})
        return res
