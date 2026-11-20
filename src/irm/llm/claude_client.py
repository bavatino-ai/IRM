"""Anthropic client shared by the Claude roles (OCR / answer / judge).

* Keys are read from the environment only (`ANTHROPIC_API_KEY`).
* A key that is not scoped to a workspace needs the `anthropic-workspace-id`
  header -> set `ANTHROPIC_WORKSPACE_ID` in `.env`.
* Errors that will hit every request (bad key, no workspace, no credit) abort the
  batch instead of being recorded once per document.
"""
from __future__ import annotations

from .. import config
from .gemini_client import QuotaExhausted

# Server-side refusal fallback: on a policy decline the API re-runs the request on a
# fallback model inside the same call (routing chosen by the API).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ProviderUnavailable(QuotaExhausted):
    """Account-level failure (authentication, workspace, credit): stop the batch."""


def make_client():
    import anthropic
    headers = {"anthropic-workspace-id": config.ANTHROPIC_WORKSPACE_ID} if config.ANTHROPIC_WORKSPACE_ID else None
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY or None, default_headers=headers)


def is_fatal(err: Exception) -> bool:
    import anthropic
    if isinstance(err, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return True
    if isinstance(err, anthropic.BadRequestError):
        msg = str(err).lower()
        return any(s in msg for s in ("credit balance", "workspace", "api key"))
    return False


def text_of(message) -> str:
    """Concatenate the text blocks of a response (thinking blocks are skipped)."""
    return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
