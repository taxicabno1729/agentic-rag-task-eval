# Agentic RAG Task Evaluation Harness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Evaluation harness for **agentic retrieval-augmented generation** systems:
retrieval, grounding, refusal, and task-success metrics measured end-to-end.

Built for **AI Engineer (internal tool)** and **Data Scientist** roles requiring:
- Agentic RAG pipelines (embeddings, vector stores, hybrid search, reranking)
- Evaluation harnesses (groundedness, latency, task success, refusals)

## Why this exists

Hiring teams at AI-native companies (e.g. AI Infra teams) routinely ask:

> "Do you have production RAG with agent harnesses, evaluation, benchmarks?"

This project answers with a reproducible harness rather than prose.

## Features

- **Hybrid retrieval** — dense (FAISS / embedding) + sparse (BM25) with
  Reciprocal Rank Fusion (RRF) fusion.
- **Source-grounded retrieval** — citations per chunk (`[AML-01]` style).
- **Agentic loop** — LangGraph state machine: decide → retrieve → ground → answer.
- **Refusal policy** — answers must cite retrieved chunks or the agent refuses.
- **Evaluation metrics**:
  - **Groundedness** — answer statements must overlap with retrieved context.
  - **Task success** — answer must directly address the question.
  - **Latency** — retrieval + generation wall-clock time.
  - **MRR** — mean reciprocal rank of retrieved chunks.
  - **Refusal accuracy** — unanswerable questions correctly refused.

## Quick start

```bash
pip install -r requirements.txt
python -m src.main --benchmark
python -m src.main --questions "What recall rate does the entity resolution system achieve?" "How much manual effort was reduced by LLM automation?"
python -m src.main --interactive
```

## Corpus

Documents cover AML / on-chain compliance research:

- Entity resolution & taint analysis (EVM, UTXO, SVM)
- Multi-hop fund tracing across bridge protocols (Wormhole, Stargate, Hop, CCTP)
- LLM automation of compliance report generation
- A/B testing on alert prioritization
- Tokenomics modeling & governance dashboards

Each chunk is assigned a citation (e.g. `[AML-01]`) used for grounding checks.

## Project structure

```
src/
├── agent.py          # LangGraph agentic loop (decide → retrieve → ground → answer)
├── retriever.py      # FAISS + BM25 + RRF hybrid retrieval
├── evaluator.py      # Groundedness, task success, latency, MRR, refusal metrics
├── data/
│   └── sample_documents.py  # AML/on-chain corpus chunks
└── main.py           # CLI: benchmark, interactive, single-question
tests/
└── test_evaluator.py  # Metric unit tests
```

## Evaluation results (synthetic corpus, mock LLM)

| Metric | Value |
|--------|-------|
| Groundedness rate | — (see benchmark output) |
| Task success rate | — (see benchmark output) |
| Avg latency (retrieval + gen) | — ms (see benchmark) |
| MRR | — (see benchmark) |
| Refusal accuracy | — (see benchmark) |

## Roadmap

- [ ] Integrate real embedding model (e.g. MiniLM, BGE)
- [ ] Integrate real LLM (e.g. GPT-4o, Claude) via OpenAI/Anthropic APIs
- [ ] Add agentic workflow (multi-hop: planner → sub-retrievers → synthesis)
- [ ] Add FAISS persistence / incremental index
- [ ] Add LangSmith / agentic trajectory tracing

## License

MIT
