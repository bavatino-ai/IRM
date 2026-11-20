"""Embedding providers behind one interface.

gemini  Gemini API embeddings (baseline in the proposal)            - needs GEMINI_API_KEY
bge     open-source BAAI/bge-small-en-v1.5 via sentence-transformers - needs Hugging Face download once
lsa     TF-IDF + truncated SVD, fitted locally                         - offline fallback used to build
        and verify the index pipeline when no API key / model download is available. It is a
        lexical-semantic baseline, not a candidate for the final system.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import joblib
import numpy as np

from .. import config


class Embedder(ABC):
    name = "base"
    dim = 0

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1, n)


class LSAEmbedder(Embedder):
    name = "lsa"

    def __init__(self, dim: int = 256, path=config.INDEX_DIR / "lsa.joblib"):
        self.dim, self.path, self.pipe = dim, path, None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, token_pattern=r"[A-Za-z0-9#.+]+"),
            TruncatedSVD(n_components=self.dim, random_state=config.SEED))
        self.pipe.fit(texts)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, self.path)
        return self

    def _ensure(self):
        if self.pipe is None:
            self.pipe = joblib.load(self.path)

    def embed_documents(self, texts):
        self._ensure()
        return _l2(self.pipe.transform(texts))

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class GeminiEmbedder(Embedder):
    name = "gemini"

    def __init__(self, model: str = config.GEMINI_EMBED_MODEL, batch: int = 100, client=None):
        self.model, self.batch, self._client = model, batch, client

    @property
    def client(self):
        if self._client is None:
            from ..llm.gemini_client import make_client
            self._client = make_client()
        return self._client

    def _embed(self, texts, task):
        from google.genai import types
        out = []
        for i in range(0, len(texts), self.batch):
            r = self.client.models.embed_content(model=self.model, contents=texts[i:i + self.batch],
                                                 config=types.EmbedContentConfig(task_type=task))
            out += [e.values for e in r.embeddings]
        return _l2(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class BGEEmbedder(Embedder):
    name = "bge"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model: str = config.BGE_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    def embed_query(self, text):
        return self.model.encode([self.QUERY_PREFIX + text], normalize_embeddings=True)[0]


def get_embedder(name: str) -> Embedder:
    return {"lsa": LSAEmbedder, "gemini": GeminiEmbedder, "bge": BGEEmbedder}[name]()
