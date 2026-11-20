"""Vector (ChromaDB) + keyword (BM25) indexes and Reciprocal Rank Fusion (Week 2).

The cross-encoder reranker and LlamaIndex orchestration are added on top of
`HybridRetriever` in Week 3.
"""
from __future__ import annotations

import pickle
import re

import chromadb

from .. import config
from ..chunking import Chunk
from .embeddings import Embedder

TOKEN = re.compile(r"[a-z0-9#+]+(?:\.[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _clean_meta(md: dict) -> dict:
    return {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in md.items()}


def build_chroma(chunks: list[Chunk], embedder: Embedder, path=config.INDEX_DIR / "chroma"):
    client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
    name = f"irm_resumes_{embedder.name}"
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    col = client.create_collection(name, metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200,
                                                 "hnsw:search_ef": 200, "hnsw:M": 32, "embedder": embedder.name})
    embs = embedder.embed_documents([c.text for c in chunks])
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        col.add(ids=[c.chunk_id for c in part], documents=[c.text for c in part],
                metadatas=[_clean_meta(c.metadata) for c in part], embeddings=embs[i:i + 500].tolist())
    return col


def open_chroma(embedder_name: str, path=config.INDEX_DIR / "chroma"):
    client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
    return client.get_collection(f"irm_resumes_{embedder_name}")


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self.ids = [c.chunk_id for c in chunks]
        self.meta = {c.chunk_id: c.metadata for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, path=config.INDEX_DIR / "bm25.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.INDEX_DIR / "bm25.pkl") -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, collection, embedder: Embedder, bm25: BM25Index, k_each: int = 50):
        self.col, self.emb, self.bm25, self.k = collection, embedder, bm25, k_each

    def dense(self, query: str, k: int | None = None, where: dict | None = None) -> list[str]:
        r = self.col.query(query_embeddings=[self.emb.embed_query(query).tolist()], n_results=k or self.k,
                           where=where)
        return r["ids"][0]

    def sparse(self, query: str, k: int | None = None) -> list[str]:
        return [i for i, _ in self.bm25.search(query, k or self.k)]

    def hybrid(self, query: str, k: int | None = None) -> list[str]:
        return [i for i, _ in rrf([self.dense(query), self.sparse(query)])][: k or self.k]
