"""Main entry point for agentic RAG evaluation harness.

Usage:
    python -m src.main --questions "Q1" "Q2" ...
    python -m src.main --interactive
    python -m src.main --benchmark

The benchmark uses the built-in AML/on-chain document corpus and a
synthetic QA dataset covering answerable + unanswerable questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List

from src.agent import build_agent_graph
from src.data.sample_documents import ALL_CHUNKS
from src.evaluator import BaseEvaluator
from src.retriever import (
    BM25Retriever,
    FaissVectorStore,
    GroundedRetrieval,
    HybridRetriever,
)


class MockEmbeddings:
    """Deterministic mock embeddings for reproducible evaluation without API keys."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        import math

        vec = [
            math.sin(int(h[i : i + 2], 16) / 255.0 * 2 * math.pi)
            for i in range(0, min(len(h), self.dimension * 2), 2)
        ]
        if len(vec) < self.dimension:
            vec = vec * (self.dimension // len(vec) + 1)
        vec = vec[: self.dimension]
        self._cache[text] = vec
        return vec


class MockLLM:
    """Deterministic mock LLM for reproducible evaluation.

    Answers are grounded in the provided context. For unanswerable
    questions (no retrieval results), returns None to trigger refusal.
    """

    def __call__(self, question: str, context: str) -> Optional[str]:
        if not context or not context.strip():
            return None
        # Simple extractive answer: return first full sentence of top chunk.
        import re

        # Strip citation markers like [[AML-03]] or [AML-03]
        cleaned = re.sub(r"\[\[[^\]]*\]\]", "", context)
        cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
        first = sentences[0].strip() if sentences else cleaned.strip()
        # Find the citation from the original context
        citations = re.findall(r"\[\[[^\]]*\]\]|\[[^\]]*\]", context)
        citation = citations[0] if citations else "[AML-01]"
        answer = f"Based on the provided documents, {first.lower()} {citation}"
        return answer


def build_retriever() -> GroundedRetrieval:
    """Build hybrid retriever with mock embeddings over ALL_CHUNKS."""
    embeddings = MockEmbeddings(dimension=384)
    vector_store = FaissVectorStore(embeddings, dimension=384)
    vector_store.add_chunks(ALL_CHUNKS)

    bm25 = BM25Retriever(ALL_CHUNKS)
    hybrid = HybridRetriever(vector_store, bm25)
    return GroundedRetrieval(hybrid)


def run_benchmark() -> dict:
    """Run evaluation on built-in QA dataset."""
    retriever = build_retriever()
    llm = MockLLM()
    evaluator = BaseEvaluator(llm_callable=llm, retriever=retriever)

    # Answerable questions (grounded in corpus)
    answerable = [
        "What recall rate does the entity resolution system achieve?",
        "How much manual effort was reduced by LLM automation?",
        "Which bridge protocols are monitored for cross-chain fund tracing?",
        "What is the latency for real-time risk scoring?",
        "How many high-risk wallets have been detected?",
        "What A/B testing results were observed for alert prioritization?",
        "How long does it take to onboard a new blockchain?",
        "What entity resolution accuracy was achieved on UTXO chains?",
        "Which tokenomics metrics are modeled across token contracts?",
        "What governance dashboard features are provided?",
    ]

    # Unanswerable questions (not in corpus) — should trigger refusal
    unanswerable = [
        "What is the current price of Bitcoin?",
        "How many employees does Binance have?",
        "What is the weather in Singapore today?",
        "Who won the 2024 US election?",
        "What is the capital of France?",
    ]

    questions = answerable + unanswerable
    results = evaluator.evaluate(questions)

    # Separate metrics
    answerable_records = [r for r in results.get("records", []) if r.question in answerable]
    unanswerable_records = [r for r in results.get("records", []) if r.question in unanswerable]

    answerable_grounded = sum(1 for r in answerable_records if r.grounded) / len(answerable_records) if answerable_records else 0
    answerable_task = sum(1 for r in answerable_records if r.task_success) / len(answerable_records) if answerable_records else 0
    unanswerable_refusal = sum(1 for r in unanswerable_records if r.refused) / len(unanswerable_records) if unanswerable_records else 0

    return {
        "overall": results,
        "answerable": {
            "grounded_rate": answerable_grounded,
            "task_success_rate": answerable_task,
            "n": len(answerable_records),
        },
        "unanswerable": {
            "refusal_rate": unanswerable_refusal,
            "n": len(unanswerable_records),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Evaluation Harness")
    parser.add_argument(
        "--questions",
        nargs="+",
        help="Questions to evaluate",
        default=None,
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive Q&A session",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark and print metrics",
    )
    parser.add_argument(
        "--output",
        help="Write benchmark results to JSON file",
        default=None,
    )
    args = parser.parse_args()

    if args.benchmark:
        print("Running agentic RAG benchmark...")
        start = time.time()
        results = run_benchmark()
        elapsed = time.time() - start
        results["benchmark_time"] = elapsed

        print(json.dumps(results, indent=2, default=str))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults written to {args.output}")
        return

    if args.interactive:
        print("Interactive mode (Ctrl+C to exit)")
        retriever = build_retriever()
        llm = MockLLM()
        graph = build_agent_graph(retriever, llm)
        while True:
            try:
                q = input("\nQuestion: ").strip()
                if not q:
                    continue
                state = {"question": q}
                result = graph.invoke(state)
                print(f"Answer: {result.get('answer') or '[REFUSED]'}")
                print(f"Citations: {result.get('citations', [])}")
                print(f"Refused: {result.get('refused', False)}")
            except (EOFError, KeyboardInterrupt):
                break
        return

    if args.questions:
        retriever = build_retriever()
        llm = MockLLM()
        graph = build_agent_graph(retriever, llm)
        for q in args.questions:
            state = {"question": q}
            result = graph.invoke(state)
            print(f"\nQ: {q}")
            print(f"A: {result.get('answer') or '[REFUSED]'}")
            print(f"Citations: {result.get('citations', [])}")
            print(f"Refused: {result.get('refused', False)}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()