"""Section-aware, overlapping chunking of resumes with employee metadata (Week 2).

One resume -> profile chunk (header + experience summary), skills chunk, education chunk,
and one chunk per project. Every chunk text starts with a short context header
("Employee 20003 Ta Long Binh | Senior Backend Developer | section") so it stays
meaningful when retrieved on its own. Sections longer than `max_words` are split
with `overlap` words carried over. Sensitive fields never enter chunk text;
restricted fields (performance band) are kept as metadata for role-based filtering only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .ground_truth import load_manifest, load_profile
from .ingestion import text_layer
from .ocr.textlayer import _flat


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


SECTION_SPLIT = re.compile(r"(EXPERIENCE SUMMARY|TECHNICAL SUMMARY|EDUCATION & QUALIFICATIONS|PROFESSIONAL EXPERIENCE)")


def _window(words: list[str], max_words: int, overlap: int) -> list[str]:
    if len(words) <= max_words:
        return [" ".join(words)]
    out, step = [], max_words - overlap
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
    return out


def sections(text: str) -> dict[str, list[str]]:
    s = _flat(text).replace("§ ", "• ").replace("---------- End of Document ---------", "").strip()
    parts = SECTION_SPLIT.split(s)
    sec = {"HEADER": parts[0].strip()}
    for name, body in zip(parts[1::2], parts[2::2]):
        sec[name] = body.strip()
    exp = sec.get("PROFESSIONAL EXPERIENCE", "")
    stint_re = re.compile(r"\[(\d{2}/\d{4} [–-] (?:Now|\d{2}/\d{4}))\]: (.+?)(?= \[| •|$)")
    stints = [(m.start(), m.end(), f"Career stint {m.group(1)} as {m.group(2)}") for m in stint_re.finditer(exp)]
    projects = []
    for m in re.finditer(r"\[\d{2}/\d{4} [–-] (?:Now|\d{2}/\d{4})\] • Project:", exp):
        end = min([s for s, _, _ in stints if s > m.start()] + [
            n.start() for n in re.finditer(r"\[\d{2}/\d{4} [–-] (?:Now|\d{2}/\d{4})\] • Project:", exp)
            if n.start() > m.start()] + [len(exp)])
        stint = [lbl for s, e, lbl in stints if e <= m.start()]
        projects.append((stint[-1] + " | " if stint else "") + exp[m.start():end].strip())
    return {
        "profile": [sec["HEADER"] + " | Experience summary: " + sec.get("EXPERIENCE SUMMARY", "")],
        "skills": [sec.get("TECHNICAL SUMMARY", "")],
        "education": [sec.get("EDUCATION & QUALIFICATIONS", "")],
        "project": projects,
    }


def chunk_resume(path, meta: dict, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    code = meta["code"]
    head = f"Employee {code} {meta['name']} | {meta['position']}"
    chunks: list[Chunk] = []
    for sec, bodies in sections(text_layer(path)).items():
        for bi, body in enumerate(bodies):
            proj = re.search(r"Project: (.+?) •", body)
            for wi, piece in enumerate(_window(body.split(), max_words, overlap)):
                md = {**meta, "section": sec, "project": proj.group(1) if proj else "", "part": wi}
                chunks.append(Chunk(f"{code}-{sec}-{bi}-{wi}", f"{head} | {sec}: {piece}", md))
    return chunks


def build_corpus() -> list[Chunk]:
    prof, man = load_profile(), load_manifest()
    perf = prof.set_index("Code").PerformanceRating
    dom = prof.set_index("Code").Domain
    chunks: list[Chunk] = []
    for _, m in man.iterrows():
        skills = ",".join(s for s in config.YEAR_SKILLS if float(m[f"{s}_resume"]) > 0)
        meta = {"code": int(m.Code), "name": m.Fullname, "position": m.Position, "title": m.Title,
                "domain": dom[m.Code], "skills": skills,
                "performance_band": perf[m.Code],             # restricted: metadata filter only, never in text
                "source_file": m.File}
        chunks += chunk_resume(config.RESUME_DIR / m.File, meta)
    return chunks


def corpus_frame(chunks: list[Chunk]) -> pd.DataFrame:
    return pd.DataFrame([{"chunk_id": c.chunk_id, "words": len(c.text.split()), **c.metadata} for c in chunks])
