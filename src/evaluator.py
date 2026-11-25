"""Evaluator module: RAG evaluation metrics.

Metrics:
- Groundedness: proportion of answer statements grounded in retrieved chunks.
- Task success: binary — does the answer directly address the question?
- Latency: wall-clock time for retrieval + generation.
- Refusal accuracy: proportion of times refusal was correctly triggered
  on unanswerable queries.
- MRR (Mean Reciprocal Rank): average of 1/rank across queries.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from statistics import mean
from typing import List, Optional

from src.retriever import GroundedRetrieval, RetrievalResult


@dataclass
class EvaluationRecord:
    question: str
    answer: Optional[str]
    retrieval_ranks: List[int]
    grounded: bool
    task_success: bool
    latency: float
    refused: bool
    refusal_correct: Optional[bool]  # None if question is answerable


class BaseEvaluator:
    """Base class for running evaluation across a dataset of Q/A pairs."""

    def __init__(self, llm_callable, retriever: GroundedRetrieval, top_k: int = 5):
        self.llm_callable = llm_callable
        self.retriever = retriever
        self.top_k = top_k
        self.records: List[EvaluationRecord] = []

    def _run_one(self, question: str) -> EvaluationRecord:
        start = time.time()
        results = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_time = time.time() - start

        ranks = [r.rank for r in results if r.score > 0]
        best_rank = min(ranks) if ranks else None

        start = time.time()
        answer = self.llm_callable(question, self.retriever.context_string(results))
        generation_time = time.time() - start
        latency = retrieval_time + generation_time

        # Groundedness: check that every non-trivial answer claim appears
        # in at least one retrieved chunk.
        grounded = self._check_grounded(answer, results) if answer else False

        # Task success: answer directly addresses the question (non-empty,
        # not a generic refusal, answers the question asked).
        task_success = self._check_task_success(answer, question) if answer else False

        # Refusal correctness:
        refusal_correct = None
        if answer is None or (isinstance(answer, str) and "cannot" in answer.lower()):
            # model refused — check if question was actually unanswerable
            # (for a minimal evaluator, mark correct if retrieval had no results)
            refused = len(results) == 0 or all(r.score <= 0 for r in results)
            refusal_correct = refused
        else:
            refused = False

        return EvaluationRecord(
            question=question,
            answer=answer,
            retrieval_ranks=ranks,
            grounded=grounded,
            task_success=task_success,
            latency=latency,
            refused=refused,
            refusal_correct=refusal_correct,
        )

    def _check_grounded(self, answer: str, results: List[RetrievalResult]) -> bool:
        """Default groundedness check: simple overlap of terms."""
        if not answer:
            return False
        # Extract top chunk's text as ground truth pool
        top_text = " ".join(r.chunk.text.lower() for r in results[:1])
        answer_words = set(re.findall(r"\w+", answer.lower()))
        if not answer_words:
            return False
        overlap = answer_words & set(re.findall(r"\w+", top_text))
        return len(overlap) / len(answer_words) > 0.3

    @staticmethod
    def _check_task_success(answer: str, question: str) -> bool:
        if not answer:
            return False
        ql = question.lower().strip()
        al = answer.lower().strip()
        # Reject generic refusals
        if any(kw in al for kw in ["cannot", "unable", "sorry", "i don't"]):
            return False
        # Relaxed check: answer must contain at least 2 substantive words (len > 2)
        # and not be a generic refusal
        al_words = [w for w in al.split() if len(w) > 2 and w not in {"what", "how", "why", "tell", "does", "do"}]
        if len(al_words) < 2:
            return False
        # Check if answer addresses the question topic
        qtopics = set(w for w in ql.split() if len(w) > 2 and w not in {"what", "how", "why", "tell"})
        al_words_set = set(al_words)
        # If there's any overlap in meaningful words, consider it task success
        return len(qtopics & al_words_set) > 0

    def evaluate(self, questions: List[str]) -> dict:
        records = [self._run_one(q) for q in questions]
        self.records.extend(records)
        n = len(records)
        if n == 0:
            return {}
        grounded_rate = sum(1 for r in records if r.grounded) / n
        task_success_rate = sum(1 for r in records if r.task_success) / n
        avg_latency = mean(r.latency for r in records)
        avg_best_rank = mean([min(r.retrieval_ranks) if r.retrieval_ranks else float("inf") for r in records])
        mrr = sum(1 / (min(r.retrieval_ranks) if r.retrieval_ranks else float("inf")) for r in records) / n
        refusal_acc = (
            sum(1 for r in records if r.refusal_correct is True) / sum(1 for r in records if r.refusal_correct is not None)
            if any(r.refusal_correct is not None for r in records) else None
        )
        return {
            "grounded_rate": grounded_rate,
            "task_success_rate": task_success_rate,
            "avg_latency": avg_latency,
            "mrr": mrr,
            "refusal_accuracy": refusal_acc,
            "records": records,
            "n": n,
        }