"""Shared Gemini client + error policy.

Lessons from the first local run (13-Nov-2026, issue IRM-W1-11):
  * JSON / schema-constrained output is only enabled on API version v1beta -> pin it.
  * Free tier = 20 generate requests per model per day. Retrying non-retryable errors
    burned that quota, so 4xx errors are never retried and a *daily* quota error
    stops the whole batch instead of failing every remaining document.
"""
from __future__ import annotations

import re
import threading
import time

from .. import config

API_VERSION = "v1beta"


class QuotaExhausted(RuntimeError):
    """Daily / project quota reached - abort the batch, resume after the reset."""


def make_client():
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set - copy .env.example to .env and add your key")
    from google import genai
    from google.genai import types
    return genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(api_version=API_VERSION))


def classify(err: Exception) -> tuple[str, float]:
    """Return (action, wait_seconds): action in {'retry', 'abort', 'fail'}."""
    msg = str(err)
    code = getattr(err, "code", None)
    if code is None:
        m = re.search(r"\b(4\d\d|5\d\d)\b", msg)
        code = int(m.group(1)) if m else None
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        if "PerDay" in msg or "per day" in msg.lower():
            return "abort", 0.0
        m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay'?: '?(\d+)s", msg)
        return "retry", min(float(m.group(1)) + 1, 120.0) if m else 30.0
    if code is not None and 500 <= code < 600:
        return "retry", 5.0
    if code is None:                       # network / timeout
        return "retry", 5.0
    return "fail", 0.0                     # 400 / 401 / 403 / 404: retrying cannot help


class Pacer:
    """Minimum interval between requests (free-tier requests-per-minute limits)."""
    _lock = threading.Lock()
    _last = 0.0

    @classmethod
    def wait(cls, interval: float = config.GEMINI_MIN_INTERVAL_S):
        with cls._lock:
            delay = cls._last + interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            cls._last = time.monotonic()


def generate(model: str, contents, gen_config, max_retries: int = 3, client=None):
    """One paced `generate_content` call with the error policy above (used by the answer / judge roles)."""
    client = client or make_client()
    for attempt in range(max_retries):
        Pacer.wait()
        try:
            return client.models.generate_content(model=model, contents=contents, config=gen_config)
        except Exception as e:  # noqa: BLE001
            action, wait = classify(e)
            if action == "abort":
                raise QuotaExhausted(str(e)[:300]) from e
            if action == "fail" or attempt == max_retries - 1:
                raise
            time.sleep(wait)
