"""Provider-agnostic OCR / extraction interface.

Short-term: Gemini vision model (`gemini`). Long-term: a local document-OCR
model plugs in behind the same interface without changing the pipeline.
`textlayer` is a deterministic, non-LLM reference used to validate the
evaluation harness; it cannot read scanned PDFs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..ingestion import ResumeDocument
from ..schema import ResumeExtraction


@dataclass
class OCRResult:
    extraction: ResumeExtraction | None
    provider: str
    model: str
    prompt_version: str = "-"
    raw_text: str = ""                      # model output / parsed source text, kept for grounding checks
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    cached: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.extraction is not None and not self.errors


class OCRProvider(ABC):
    name: str = "base"
    model: str = "-"

    @abstractmethod
    def extract(self, doc: ResumeDocument) -> OCRResult: ...


def get_provider(name: str, **kw) -> OCRProvider:
    if name == "gemini":
        from .gemini import GeminiOCR
        return GeminiOCR(**kw)
    if name == "claude":
        from .claude import ClaudeOCR
        return ClaudeOCR(**kw)
    if name == "textlayer":
        from .textlayer import TextLayerParser
        return TextLayerParser(**kw)
    raise ValueError(f"unknown OCR provider: {name}")
