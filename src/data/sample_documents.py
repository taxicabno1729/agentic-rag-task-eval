"""Sample on-chain / AML documents for RAG evaluation.

These documents mimic the kind of blockchain analytics / compliance text
that the Binance/OKX/Ripjar roles would reference. Each chunk has a
unique citation for grounding checks.

Files are imported via `src.data.sample_documents` or directly via the
Chunk / FaissVectorStore constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.retriever import Chunk


# ── AML / On-chain compliance documents ────────────────────────────────

DOC_AML_01 = """
Entity resolution & taint analysis form the backbone of blockchain compliance.
We trace funds across UTXO, account, and message-passing architectures,
identifying related wallet clusters with graph theory and machine learning.
Our system achieves 95% recall in risk detection and <1s latency for
real-time scoring, enabling clients to block suspicious transfers before
completion. Over 100M+ high-risk wallets have been flagged across 20+
blockchains since deployment.
"""

DOC_AML_02 = """
Multi-hop fund tracing follows lock/burn events across bridge protocols
(Wormhole, Stargate, Hop Protocol, Circle CCTP) to trace destination-side
mint/release events. Matching heuristics pair source-side lock events with
destination-side mints, making fund flows traceable through bridge hops.
Government agencies and 15+ financial institutions rely on this capability
for sanctions screening and entity resolution at scale.
"""

DOC_AML_03 = """
LLM capabilities automate compliance report generation, reducing manual
effort by 70% and improving customer experience by 35%. Integrated into
legacy compliance systems, the LLM extracts risk insights from transaction
flows and produces structured reports. Developer productivity increased
by 35% via MCP/LLM integration, and transaction monitoring accuracy
improved by 40% compared to rule-based systems alone using ML and graph
analytics.
"""

DOC_AML_04 = """
Entity resolution across 20+ blockchains identifies related wallet clusters
to detect criminal activity networks. Addresses are resolved into entities
and multi-hop fund flows are traced with proportional taint attribution
across EVM, UTXO, and SVM chains. Transaction monitoring accuracy improved
by 40% compared to rule-based systems alone using ML and graph analytics.
Real-time (<1s) risk scoring allows clients to block suspicious transfers
before completion. 100M+ high-risk wallets involved in fraud, money
laundering and ransomware attacks have been detected.
"""

DOC_AML_05 = """
A/B testing on alert prioritization algorithms resulted in 35% reduction
in false positive alerts and 20% faster case resolution times for
compliance teams. Analyst workflow efficiency improved by 100% in 2
years through model tuning. Data quality policies eliminated human errors
with 100% improvement, and global data collection coverage increased by
40% through engineered crawlers and scrapers.
"""

DOC_AML_06 = """
Led the design, development, and maintenance of 3 products through setting
up product analytics, A/B testing, client engagement, and product
recommendations. A risk intelligence system utilized by four government
agencies monitors illicit blockchain activities. LLM capabilities automate
compliance report generation, reducing manual effort by 70% and improving
customer experience by 35%. The first data warehouse integrates over 20
blockchains, efficiently processing over 100 million transactions daily.
"""

DOC_AML_07 = """
Bridge protocol analytics: event-based ELTs for Wormhole, Stargate, Hop
Protocol, and Circle CCTP cover EVMs, Optimistic Rollups, SVM, and UTXO
blockchain families. Onboarding time for new blockchain support reduced
from three months to two weeks via modular architecture. Comprehensive ELT
processes over 15 bridge protocols across EVM, SVM, UTXO, Stellar, and
XRPL architectures.
"""

# ── Chunk definitions ────────────────────────────────────────────────────

AML_CHUNKS: List[Chunk] = [
    Chunk(
        text=DOC_AML_01,
        doc_id="AML-01",
        chunk_id=1,
        citation="[AML-01]",
        metadata={"topic": "entity-resolution", "latency_ms": 850},
    ),
    Chunk(
        text=DOC_AML_02,
        doc_id="AML-02",
        chunk_id=1,
        citation="[AML-02]",
        metadata={"topic": "multi-hop-tracing", "latency_ms": 1200},
    ),
    Chunk(
        text=DOC_AML_03,
        doc_id="AML-03",
        chunk_id=1,
        citation="[AML-03]",
        metadata={"topic": "llm-automation", "latency_ms": 450},
    ),
    Chunk(
        text=DOC_AML_04,
        doc_id="AML-04",
        chunk_id=1,
        citation="[AML-04]",
        metadata={"topic": "entity-resolution", "latency_ms": 600},
    ),
    Chunk(
        text=DOC_AML_05,
        doc_id="AML-05",
        chunk_id=1,
        citation="[AML-05]",
        metadata={"topic": "a-b-testing", "latency_ms": 300},
    ),
    Chunk(
        text=DOC_AML_06,
        doc_id="AML-06",
        chunk_id=1,
        citation="[AML-06]",
        metadata={"topic": "product-development", "latency_ms": 1100},
    ),
    Chunk(
        text=DOC_AML_07,
        doc_id="AML-07",
        chunk_id=1,
        citation="[AML-07]",
        metadata={"topic": "bridge-protocols", "latency_ms": 950},
    ),
]


# ── On-chain research / product development documents ───────────────────

DOC_ONCHAIN_01 = """
EVM architectures (Ethereum, Rollups, PoA) support smart contract execution
and token transfers. We process over 100M daily transactions across 20+
EVM-compatible chains, building ELTs that normalize and enrich transaction
data for downstream analytics. Matching heuristics align lock/burn events
across bridge protocols to enable fund-flow tracing. Entity resolution
across UTXO (Bitcoin, Stellar) and account (XRPL, Polkadot) architectures
extends coverage beyond EVM-only ecosystems.
"""

DOC_ONCHAIN_02 = """
Solana SVM architecture processes transactions at high throughput with
low latency. Anchor framework and program-driven development enable
on-chain invariants and runtime assertions. Taint analysis on Solana
accounts follows account-complete and state-complete models. Cluster
analysis identifies related wallet groups involved in coordinated
attack patterns, including rent-exempt minimum optimization and
priority-fee manipulation.
"""

DOC_ONCHAIN_03 = """
Cross-chain protocol monitoring: Wormhole, Across, AllBridge, Circle CCTP,
Hop Protocol, and Thorswap are monitored for liquidity shifts, volume
changes, and fee-route optimizations. Fee/swap-rate elasticity against
adoption is quantified across sample user bases over rolling 30-day
windows. Monitoring dashboards integrate with Dune, Quicknode, and
Infura for real-time visualization. 100M+ high-risk wallets detected
across all monitored protocols since Q2 2023.
"""

DOC_ONCHAIN_04 = """
Fund tracing across UTXO architectures (Bitcoin, Stellar Soroban) follows
lock/spend patterns with graph-based clustering. Address resolution into
entities enables multi-hop tracing through message-passing architectures
(XRPL, Polkadot, Cosmos). Entity resolution accuracy of 92%+ on labeled
datasets has been achieved via graph neural networks and traditional
link-analysis (PageRank, HITS). Latency <2s per fund-trace query in
production.
"""

DOC_ONCHAIN_05 = """
Tokenomics modeling: supply schedules, emission curves, and vesting
contracts are parsed and modeled across 50+ token contracts. A/B tests
on swap-rate changes influence user adoption and retention. Fee-tier
elasticity identified where retention drops 15% above threshold, informing
pricing changes that resulted in 8% adoption lift. Models integrated
with Airflow + dbt for production reporting.
"""

DOC_ONCHAIN_06 = """
Governance dashboard: real-time TVL, active addresses, and fee-rate
visualizations for staking contracts across EVM and Solana. Grafana +
Prometheus backend enables sub-second query latency. Alerting on
anomalous TVL drops or fee-spike patterns triggered 3x faster case
resolution in compliance audits. Dashboard supports 20+ blockchain
networks with per-chain breakdowns.
"""

# ── On-chain chunk definitions ─────────────────────────────────────────

ONCHAIN_CHUNKS: List[Chunk] = [
    Chunk(
        text=DOC_ONCHAIN_01,
        doc_id="ONC-01",
        chunk_id=1,
        citation="[ONC-01]",
        metadata={"topic": "evm-architecture", "latency_ms": 700},
    ),
    Chunk(
        text=DOC_ONCHAIN_02,
        doc_id="ONC-02",
        chunk_id=1,
        citation="[ONC-02]",
        metadata={"topic": "solana-svm", "latency_ms": 550},
    ),
    Chunk(
        text=DOC_ONCHAIN_03,
        doc_id="ONC-03",
        chunk_id=1,
        citation="[ONC-03]",
        metadata={"topic": "cross-chain-protocols", "latency_ms": 900},
    ),
    Chunk(
        text=DOC_ONCHAIN_04,
        doc_id="ONC-04",
        chunk_id=1,
        citation="[ONC-04]",
        metadata={"topic": "utxo-architectures", "latency_ms": 800},
    ),
    Chunk(
        text=DOC_ONCHAIN_05,
        doc_id="ONC-05",
        chunk_id=1,
        citation="[ONC-05]",
        metadata={"topic": "tokenomics-modeling", "latency_ms": 400},
    ),
    Chunk(
        text=DOC_ONCHAIN_06,
        doc_id="ONC-06",
        chunk_id=1,
        citation="[ONC-06]",
        metadata={"topic": "governance-dashboard", "latency_ms": 650},
    ),
]


# ── Mixed AML + On-chain dataset ────────────────────────────────────────

ALL_CHUNKS: List[Chunk] = AML_CHUNKS + ONCHAIN_CHUNKS

# Citation order for RRF / grounding
CITATION_ORDER = [c.citation for c in ALL_CHUNKS]