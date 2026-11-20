"""API setup checks (Week 1) and the Claude LLM-as-judge client (used from Week 3).

`python run.py check-apis` sends one tiny request to each provider and writes the
model IDs that answered to docs/evaluation/model_pins.json - the pinned versions for the project.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .. import config


def check_gemini(model: str) -> dict:
    from .gemini_client import make_client
    client = make_client()
    t0 = time.perf_counter()
    r = client.models.generate_content(model=model, contents="Reply with the single word OK.")
    return {"provider": "google", "model": model, "model_version": getattr(r, "model_version", None),
            "reply": (r.text or "").strip()[:20], "latency_s": round(time.perf_counter() - t0, 2)}


def check_claude(model: str) -> dict:
    from .claude_client import make_client
    client = make_client()
    t0 = time.perf_counter()
    r = client.messages.create(model=model, max_tokens=5,
                               messages=[{"role": "user", "content": "Reply with the single word OK."}])
    return {"provider": "anthropic", "model": model, "model_version": r.model,
            "reply": r.content[0].text.strip()[:20], "latency_s": round(time.perf_counter() - t0, 2)}


def check_gemini_embedding(model: str) -> dict:
    from .gemini_client import make_client
    client = make_client()
    t0 = time.perf_counter()
    r = client.models.embed_content(model=model, contents="OK")
    return {"provider": "google", "model": model, "dimensions": len(r.embeddings[0].values),
            "latency_s": round(time.perf_counter() - t0, 2)}


def configured_models() -> dict[tuple[str, str], list[str]]:
    """(provider, model) -> roles, for every model the configuration can call."""
    roles: dict[tuple[str, str], list[str]] = {}
    for m in config.GEMINI_OCR_CANDIDATES:
        roles.setdefault(("google", m), []).append("ocr_candidate")
    for key, role in [(("google", config.GEMINI_OCR_MODEL), "ocr"),
                      (("google", config.GEMINI_ANSWER_MODEL), "answer"),
                      (("google", config.GEMINI_JUDGE_MODEL), "judge (swapped roles)"),
                      (("google-embed", config.GEMINI_EMBED_MODEL), "embeddings"),
                      (("anthropic", config.CLAUDE_JUDGE_MODEL), "judge"),
                      (("anthropic", config.CLAUDE_OCR_MODEL), "ocr (swapped roles)"),
                      (("anthropic", config.CLAUDE_ANSWER_MODEL), "answer (swapped roles)")]:
        roles.setdefault(key, []).append(role)
    return roles


def check_all() -> list[dict]:
    """One tiny request per configured model (each counts against the free-tier daily quota)."""
    checks = {"google": check_gemini, "google-embed": check_gemini_embedding, "anthropic": check_claude}
    results = []
    for (provider, m), roles in configured_models().items():
        try:
            r = checks[provider](m)
        except Exception as e:  # noqa: BLE001
            r = {"provider": provider.split("-")[0], "model": m, "error": repr(e)[:300]}
        results.append({**r, "roles": roles})
    pins = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "anthropic_workspace_header": bool(config.ANTHROPIC_WORKSPACE_ID), "results": results}
    config.EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    (config.EVAL_RESULTS / "model_pins.json").write_text(json.dumps(pins, indent=2), encoding="utf-8")
    return results


def judge(question: str, context: str, answer: str, model: str = config.CLAUDE_JUDGE_MODEL,
          provider: str = "claude") -> dict:
    """Independent LLM-as-judge; see `evaluation.answer_eval.judge` (either provider)."""
    from ..evaluation.answer_eval import judge as _judge
    return _judge(question, context, answer, provider=provider, model=model).model_dump()
