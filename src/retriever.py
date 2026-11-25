"""Retriever module for agentic RAG evaluation harness.

Provides:
- Vector store backed by FAISS (CPU)
- BM25 sparse retrieval (via sklearn-style token counting)
- Hybrid fusion (reciprocal rank fusion)
- Source-grounded retrieval with citations
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - dependency may be absent in minimal installs
    faiss = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class Chunk:
    """A single text chunk with metadata and citation ID."""
    text: str
    doc_id: str
    chunk_id: int
    citation: str  # e.g. "[1]" or "[AML-02, para 3]"
    metadata: dict


class FaissVectorStore:
    """Minimal FAISS-backed vector store for LangChain-style embeddings."""

    def __init__(self, embeddings: Embeddings, dimension: int):
        if faiss is None:
            raise ImportError("faiss-cpu is required for FaissVectorStore")
        self.embeddings = embeddings
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []
        self._id_to_row: dict[str, int] = {}

    def _embed_texts(self, texts: Sequence[str]):
        vectors = self.embeddings.embed_documents(list(texts))
        return np.array(vectors, dtype=np.float32)

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        vectors = self._embed_texts([c.text for c in chunks])
        faiss.normalize_L2(vectors)  # cosine similarity with inner product
        start = len(self.chunks)
        self.index.add(vectors)
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self._id_to_row[f"{chunk.doc_id}:{chunk.chunk_id}"] = start + i

    def search(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        q_vec = self._embed_texts([query])
        faiss.normalize_L2(q_vec)
        scores, ids = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results: List[tuple[Chunk, float]] = []
        for sid, score in zip(ids[0], scores[0]):
            if sid == -1:
                continue
            results.append((self.chunks[sid], float(score)))
        return results


class BM25Retriever:
    """Simple BM25 sparse retriever over pre-indexed chunks."""

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_count = len(chunks)
        self.avgdl = 0.0
        self.term_freqs: List[Counter] = []
        self.doc_freqs: Counter[str] = Counter()
        self._build()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build(self):
        if self.doc_count == 0:
            return
        total_terms = 0
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_freqs.update(tf.keys())
            total_terms += sum(tf.values())
        self.avgdl = total_terms / self.doc_count

    def score(self, query: str, chunk_idx: int) -> float:
        if self.doc_count == 0:
            return 0.0
        tokens = self._tokenize(query)
        tf = self.term_freqs[chunk_idx]
        dl = sum(tf.values())
        score = 0.0
        for term in tokens:
            n_t = self.doc_freqs.get(term, 0)
            if n_t == 0:
                continue
            idf = math.log((self.doc_count - n_t + 0.5) / (n_t + 0.5) + 1.0)
            f_t_d = tf.get(term, 0)
            score += idf * (
                f_t_d * (self.k1 + 1)
                / (f_t_d + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            )
        return score

    def search(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        if self.doc_count == 0:
            return []
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], s) for i, s in scores[:top_k] if s > 0]


def reciprocal_rank_fusion(
    ranked_lists: List[List[tuple[Chunk, float]]], k: int = 60
) -> List[tuple[Chunk, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked):
            key = f"{chunk.doc_id}:{chunk.chunk_id}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            best_chunk[key] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(best_chunk[key], score) for key, score in fused]


class HybridRetriever:
    """Combines dense (FAISS) and sparse (BM25) retrieval with RRF fusion."""

    def __init__(self, vector_store: FaissVectorStore, bm25: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25

    def search(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        dense = self.vector_store.search(query, top_k * 2)
        sparse = self.bm25.search(query, top_k * 2)
        fused = reciprocal_rank_fusion([dense, sparse])
        return fused[:top_k]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int


class GroundedRetrieval:
    """Wraps retriever output with citation tracking for grounded answers."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        raw = self.retriever.search(query, top_k)
        results: List[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(raw, start=1):
            results.append(
                RetrievalResult(chunk=chunk, score=score, rank=rank)
            )
        return results

    def context_string(self, results: List[RetrievalResult]) -> str:
        """Build a context string with citations from retrieval results."""
        parts: List[str] = []
        for r in results:
            parts.append(f"[{r.chunk.citation}] {r.chunk.text}")
        return "\n".join(parts)

    def cited_chunks(self, results: List[RetrievalResult]) -> List[Chunk]:
        return [r.chunk for r in results]