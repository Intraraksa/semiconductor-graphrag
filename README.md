# Semiconductor Supply Chain GraphRAG

A GraphRAG system for semiconductor supply chain intelligence, built on the [ETO ChipExplorer](https://eto.tech/dataset-docs/chipexplorer/) dataset from Georgetown CSET. Answers complex multi-hop questions about supplier monopolies, production pipelines, geopolitical risk, and tool taxonomies — queries that vector RAG alone cannot handle.

## Why GraphRAG?

| Question type | Vector RAG | GraphRAG |
|---|---|---|
| "Who supplies EUV tools?" | Maybe | Yes (PROVIDES edge + share_provided) |
| "What tools feed into Photolithography?" | No | Yes (GOES_INTO traversal) |
| "What breaks if ASML stops supplying?" | No | Yes (blast-radius multi-hop) |
| "List all subtypes of Lithography tools" | No | Yes (IS_TYPE_OF taxonomy) |
| "Which inputs are single-country sourced?" | No | Yes (aggregation + country filter) |
| "Find inputs similar to photomask inspection" | Yes | Hybrid (vector fallback) |

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini `gemini-2.5-flash` |
| Embeddings | `gemini-embedding-001` (768-dim) |
| Graph DB | Neo4j 5.26 Community (Docker) |
| Orchestration | LangGraph `StateGraph` |
| LLM Framework | LangChain + `langchain-neo4j` |
| Vector Index | Neo4j built-in (on `SpecChunk` nodes) |

## Dataset

ETO ChipExplorer — Georgetown CSET semiconductor supply chain:

| File | Nodes/Edges | Description |
|---|---|---|
| `data/inputs.csv` | 126 Input nodes | Tools, materials, processes, design resources |
| `data/providers.csv` | 393 Provider nodes | Companies + country aggregates |
| `data/provision.csv` | 1,276 PROVIDES edges | Market share % per provider→input |
| `data/sequence.csv` | 53 GOES_INTO + 86 IS_TYPE_OF | Pipeline flow + tool taxonomy |
| `data/stages.csv` | 3 Stage nodes | S1 Design, S2 Fabrication, S3 ATP |

Notable data patterns:
- **ASML (NLD)** — 100% of EUV lithography tools (absolute monopoly)
- **Netherlands** — controls entire EUV supply chain
- **Japan** — dominant in crystal machining (99.6%), resist processing (93.9%)
- **USA** — dominant in EDA software (96%), electrochemical plating (99.8%)

## Graph Schema

```
(:Input)    input_id*, input_name, type, description
(:Provider) provider_id*, provider_name, provider_type, country
(:Stage)    stage_id*, stage_name
(:SpecChunk) text, embedding   ← vector search side

(Provider)-[:PROVIDES {share_provided, year}]->(Input)
(Input)-[:GOES_INTO]->(Input)       ← production pipeline
(Input)-[:IS_TYPE_OF]->(Input)      ← taxonomy hierarchy
(Input)-[:IN_STAGE]->(Stage)
(SpecChunk)-[:DESCRIBES]->(Input)
```

## Project Structure

```
manufacturing-graphrag/
├── .env                        ← credentials (never commit)
├── docker-compose.yml          ← Neo4j 5.26-community container
├── requirements.txt
├── pytest.ini
│
├── data/                       ← ETO ChipExplorer CSV files
│   ├── inputs.csv
│   ├── providers.csv
│   ├── provision.csv
│   ├── sequence.csv
│   └── stages.csv
│
├── scripts/
│   ├── setup_schema.py         ← Step 1: constraints + indexes + vector index
│   └── ingest.py               ← Step 2: load CSVs + generate embeddings
│
├── graphrag/
│   ├── config.py               ← env loading
│   ├── graph_db.py             ← Neo4j driver + Neo4jGraph helper
│   ├── prompts/
│   │   ├── cypher_generation.py  ← domain-tuned Text-to-Cypher prompt
│   │   ├── extraction.py         ← intent + entity extraction
│   │   └── answering.py          ← final answer generation
│   ├── retrieval/
│   │   ├── graph_retriever.py    ← GraphCypherQAChain (Text-to-Cypher)
│   │   ├── vector_retriever.py   ← Neo4jVector on SpecChunk nodes
│   │   └── hybrid_retriever.py   ← combines both retrievers
│   └── agents/
│       ├── state.py              ← SemiconductorGraphRAGState TypedDict
│       ├── nodes.py              ← 5 async node functions
│       ├── routers.py            ← route_by_intent()
│       └── graph.py              ← build_agent() + ask() wrapper
│
├── tests/
│   ├── test_retrieval.py       ← Neo4j connectivity + Cypher correctness (10 tests)
│   ├── test_queries.py         ← 14 benchmark questions
│   └── test_agent.py           ← LangGraph integration tests
│
└── notebooks/
    └── exploration.ipynb       ← ad-hoc Cypher + retrieval testing
```

## Setup

### Prerequisites

- Python 3.12+
- Docker Desktop
- Google API key (Gemini)

### 1. Configure environment

Create `.env` in the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

GOOGLE_API_KEY=your_google_api_key

LANGCHAIN_TRACING_V2=false
```

### 2. Start Neo4j

```bash
docker compose up -d
```

Neo4j browser available at `http://localhost:7474`.

### 3. Create virtual environment and install dependencies

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Create Neo4j schema

Run this **before** loading any data:

```bash
python scripts/setup_schema.py
```

Creates uniqueness constraints, performance indexes, and the 768-dim cosine vector index for SpecChunk nodes.

### 5. Ingest data + generate embeddings

```bash
python scripts/ingest.py
```

Loads all 5 CSV files via `MERGE` (idempotent — safe to re-run) and generates Gemini embeddings for all Input descriptions. Takes ~2 minutes for 126 embedding API calls.

### 6. Verify

```bash
python -m pytest tests/test_retrieval.py -v
```

Expected: **10/10 passed** — checks node counts, edge counts, ASML monopoly detection, lithography taxonomy, vector index.

---

## Usage

### Python API

```python
import asyncio
from graphrag.agents.graph import ask

# Pipeline traversal
print(asyncio.run(ask(
    "Show me the full production pipeline from Chip design to Finished logic chip."
)))
# → Chip design → Deposition → Photolithography → Etch and clean →
#   Ion implantation → Chemical mechanical planarization → Assembly and packaging
#   → Testing → Finished logic chip

# Monopoly detection
print(asyncio.run(ask(
    "Which inputs have a single organization controlling 100% of the market?"
)))
# → EUV lithography tools: ASML (NLD)
#   Crystal growing furnaces: PVA TePla (DEU)
#   Direct write systems (adv. pkg.): Applied Materials (USA)

# Taxonomy
print(asyncio.run(ask(
    "What are all the subtypes of Lithography tools?"
)))
# → EUV, ArF immersion (DUV), ArF dry (DUV), KrF (DUV),
#   i-line, Mask aligners, Imprint lithography, Maskless lithography tools...

# Blast-radius
print(asyncio.run(ask(
    "What would break in the fabrication pipeline if ASML stopped supplying?"
)))
# → Direct impacts: EUV tools, ArF immersion, KrF, i-line...
#   Cascade: Lithography → Photolithography → Etch → CMP → ATP → Finished chip
```

### Run all benchmarks

```bash
python tests/test_queries.py
```

Runs all 14 benchmark questions and prints full answers.

---

## Agent Architecture

```
START
  │
  ▼
classify_and_extract        ← Gemini classifies intent + extracts entities
  │
  ├──[pipeline|provider|taxonomy|hybrid]──► graph_traversal     ← Text-to-Cypher
  ├──[risk]────────────────────────────────► risk_assessment     ← hardcoded risk Cypher
  └──[semantic]────────────────────────────► vector_search       ← SpecChunk embedding search
                                                    │
                                                    ▼
                                           fuse_and_generate     ← Gemini final answer
                                                    │
                                                   END
```

**Intent routing:**

| Intent | Trigger examples | Retriever |
|---|---|---|
| `pipeline` | "what feeds into X", "production steps" | `graph_traversal` |
| `provider` | "who supplies X", "market share" | `graph_traversal` |
| `taxonomy` | "subtypes of X", "what is a type of" | `graph_traversal` |
| `risk` | "concentration risk", "single country", "blast radius" | `risk_assessment` |
| `semantic` | "find inputs related to", "similar to" | `vector_search` |
| `hybrid` | multi-part questions | `graph_traversal` + vector fallback |

---

## Key Cypher Patterns

```cypher
-- Monopoly: single org controls 100%
MATCH (p:Provider {provider_type: 'organization'})-[r:PROVIDES]->(i:Input)
WHERE r.share_provided = 100
RETURN i.input_name, p.provider_name, p.country

-- Single-country sourcing
MATCH (p:Provider {provider_type: 'country'})-[r:PROVIDES]->(i:Input)
WHERE r.share_provided >= 100
RETURN i.input_name, p.provider_name AS country, r.share_provided

-- Pipeline upstream of X
MATCH (target:Input)
WHERE toLower(target.input_name) CONTAINS 'photolithography'
MATCH (upstream:Input)-[:GOES_INTO*1..6]->(target)
RETURN DISTINCT upstream.input_name, upstream.type

-- Taxonomy subtypes
MATCH (sub:Input)-[:IS_TYPE_OF*1..3]->(parent:Input)
WHERE toLower(parent.input_name) CONTAINS 'lithography'
RETURN sub.input_name, parent.input_name AS parent

-- Blast radius
MATCH (p:Provider)-[:PROVIDES]->(i:Input)
WHERE toLower(p.provider_name) CONTAINS 'asml'
OPTIONAL MATCH (i)-[:GOES_INTO*1..5]->(downstream:Input)
RETURN i.input_name AS direct_impact,
       COLLECT(DISTINCT downstream.input_name) AS cascade_effects

-- Geopolitical risk (>80% single country)
MATCH (p:Provider {provider_type: 'country'})-[r:PROVIDES]->(i:Input)
WHERE r.share_provided > 80
RETURN i.input_name, p.provider_name AS country, r.share_provided
ORDER BY r.share_provided DESC
```

---

## Tests

```bash
# Neo4j schema + data verification (fast, no LLM calls)
python -m pytest tests/test_retrieval.py -v

# Full benchmark (requires GOOGLE_API_KEY + running Neo4j)
python tests/test_queries.py

# LangGraph integration tests
python -m pytest tests/test_agent.py -v -s
```

---

## Notes

- **Ingestion is idempotent** — all writes use `MERGE`, safe to re-run
- **8 provision rows** reference input `N68` which is absent from `inputs.csv` — these edges are silently skipped
- **`provider_type='country'`** rows in `providers.csv` represent country-level aggregates (e.g., `NLD` = all Dutch suppliers combined); `provider_type='organization'` rows are individual firms
- The `SpecChunk` vector index uses 768-dim cosine similarity matching `gemini-embedding-001` output
