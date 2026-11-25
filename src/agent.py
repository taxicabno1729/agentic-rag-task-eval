"""Agent module: LangGraph agentic RAG pipeline.

Builds a ReAct-style agent:
- Decide node: plans retrieval steps based on the question.
- Retrieve node: runs hybrid retriever with RRF fusion (placeholder for LLM tool).
- Agent node: LLM decides to retrieve then answer with citations.
- Grounding / refusal: answer must be grounded in retrieved chunks or refuse.

The agent state tracks: question, retrieval results, answer, citations, refused.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.retriever import Chunk, GroundedRetrieval, RetrievalResult


class AgentState(TypedDict):
    question: str
    retrieval_results: List[Dict[str, Any]]
    answer: Optional[str]
    citations: List[str]
    refused: bool
    error: Optional[str]


def build_agent_graph(
    retriever: GroundedRetrieval,
    llm_callable,
    top_k: int = 5,
):
    """Build an agent graph with decide -> retrieve -> ground -> answer flow.

    Args:
        retriever: GroundedRetrieval instance wrapping the hybrid retriever.
        llm_callable: Callable(question, context) -> answer string or None.
        top_k: Number of chunks to retrieve per query.
    """

    def decide(state: AgentState) -> AgentState:
        question = state.get("question") or ""
        # Heuristic: any non-empty question triggers retrieval.
        return {**state, "question": question}

    def retrieve(state: AgentState) -> AgentState:
        question = state["question"]
        results: List[RetrievalResult] = retriever.retrieve(question, top_k=top_k)
        retrieval_payload = [
            {
                "doc_id": r.chunk.doc_id,
                "chunk_id": r.chunk.chunk_id,
                "citation": r.chunk.citation,
                "score": r.score,
                "rank": r.rank,
                "text": r.chunk.text,
            }
            for r in results
        ]
        return {**state, "retrieval_results": retrieval_payload}

    def ground_and_answer(state: AgentState) -> AgentState:
        results = state.get("retrieval_results") or []
        if not results:
            return {
                **state,
                "answer": None,
                "citations": [],
                "refused": True,
            }
        context = "\n".join(
            f"[{r['citation']}] {r['text']}" for r in results
        )
        answer = llm_callable(state["question"], context)
        if not answer:
            return {
                **state,
                "answer": None,
                "citations": [],
                "refused": True,
            }
        # Extract citations from answer text (look for [n] or [ID]).
        citations = _extract_citations(answer)
        # Enforce grounding: if answer cites nothing, treat as refused.
        if not citations:
            return {
                **state,
                "answer": answer,
                "citations": [],
                "refused": True,
            }
        return {
            **state,
            "answer": answer,
            "citations": citations,
            "refused": False,
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("decide", decide)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", ground_and_answer)
    workflow.add_edge(START, "decide")
    workflow.add_edge("decide", "retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


def _extract_citations(answer: str) -> List[str]:
    """Extract citation markers like [1], [AML-02], [para 3] from answer text."""
    import re

    return re.findall(r"\[[^\]]+\]", answer)