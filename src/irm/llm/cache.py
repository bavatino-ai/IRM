"""File-based response cache: identical (model, prompt version, document hash) is never paid twice."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import config


class JsonCache:
    def __init__(self, folder: Path = config.CACHE_DIR):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]

    def get(self, key: str) -> dict | None:
        p = self.folder / f"{key}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, key: str, value: dict) -> None:
        (self.folder / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def cost(model: str, in_tok: int, out_tok: int) -> float | None:
    pin, pout = config.PRICING.get(model, (None, None))
    if pin is None or pout is None:
        return None
    return round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 6)
