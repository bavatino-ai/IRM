"""Ingestion: collect PDF resumes from local storage and prepare them for OCR.

PyMuPDF only *prepares* the input (page rendering, text-layer probe). The OCR
itself is done by a provider behind `irm.ocr.base.OCRProvider`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from . import config

SUPPORTED = {".pdf"}          # short-term scope: PDF only (DOCX / images = long-term roadmap)
_FILE_RE = re.compile(r"^(?P<code>\d{5})-(?P<name>.+?)(?:-scan)?\.pdf$", re.I)


@dataclass
class ResumeDocument:
    path: Path
    sha256: str
    n_pages: int
    has_text_layer: bool
    code_hint: int | None = None          # from file name; never trusted for matching on its own
    pages_png: list[bytes] = field(default_factory=list, repr=False)

    @property
    def kind(self) -> str:
        return "digital" if self.has_text_layer else "scanned"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover(folder: Path = config.RESUME_DIR) -> tuple[list[Path], list[Path]]:
    """Return (supported, rejected) files. Rejected ones are logged for HR, not dropped silently."""
    ok, rejected = [], []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file():
            (ok if p.suffix.lower() in SUPPORTED else rejected).append(p)
    return ok, rejected


def text_layer(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def load(path: Path, dpi: int = config.RENDER_DPI, render: bool = True) -> ResumeDocument:
    path = Path(path)
    with pymupdf.open(path) as doc:
        txt = "".join(page.get_text("text") for page in doc).strip()
        pages = [page.get_pixmap(dpi=dpi).tobytes("png") for page in doc] if render else []
        n = doc.page_count
    m = _FILE_RE.match(path.name)
    return ResumeDocument(path=path, sha256=file_sha256(path), n_pages=n,
                          has_text_layer=len(txt) > 50,
                          code_hint=int(m.group("code")) if m else None, pages_png=pages)
