"""Scan-style (image-only) PDF variants to test OCR robustness on scanned resumes (Week 2).

Each page is rendered, converted to greyscale, slightly rotated, blurred, noised and
JPEG-compressed, then re-assembled into a PDF with NO text layer - so only a real OCR
step (Gemini vision now, a local model later) can read it.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

from . import config

PROFILES = {
    "light": {"dpi": 150, "angle": 0.6, "blur": 0.4, "noise": 6, "jpeg": 75},
    "medium": {"dpi": 130, "angle": 1.2, "blur": 0.7, "noise": 12, "jpeg": 55},
    "heavy": {"dpi": 110, "angle": 2.0, "blur": 1.0, "noise": 20, "jpeg": 40},
}


def degrade(img: Image.Image, p: dict, rng: random.Random) -> Image.Image:
    img = img.convert("L")
    img = img.rotate(rng.uniform(-p["angle"], p["angle"]), resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    arr = np.asarray(img).astype(np.int16)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    arr = np.clip(arr + nrng.normal(0, p["noise"], arr.shape), 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=p["jpeg"])
    return Image.open(io.BytesIO(buf.getvalue()))


def make_scan(src: Path, dst: Path, profile: str, seed: int) -> dict:
    p, rng = PROFILES[profile], random.Random(seed)
    out = pymupdf.open()
    with pymupdf.open(src) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=p["dpi"])
            img = degrade(Image.open(io.BytesIO(pix.tobytes("png"))), p, rng)
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=p["jpeg"])
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=buf.getvalue())
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, deflate=True)
    out.close()
    with pymupdf.open(dst) as chk:
        text_chars = sum(len(pg.get_text().strip()) for pg in chk)
    return {"file": dst.name, "source": src.name, "profile": profile, "dpi": p["dpi"],
            "text_layer_chars": text_chars, "bytes": dst.stat().st_size}


def build(codes: list[int], seed: int = config.SEED) -> list[dict]:
    """Assign light / medium / heavy in rotation so each degradation level is represented."""
    files = {int(f.name[:5]): f for f in config.RESUME_DIR.glob("*.pdf")}
    names = list(PROFILES)
    log = []
    for i, code in enumerate(sorted(codes)):
        src = files[code]
        log.append(make_scan(src, config.SCAN_DIR / src.name.replace(".pdf", "-scan.pdf"),
                             names[i % len(names)], seed + code))
    (config.SCAN_DIR / "_scan_manifest.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log
