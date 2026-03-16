# CLAUDE.md — Manufacturing GraphRAG Project

This file tells Claude Code how to work in this codebase.
Read this entire file before writing any code or making any changes.
You have graphrag-builder skill to work this project
---

## Project Overview

Building a **GraphRAG system for semiconductor supply chain intelligence** using:
- **LangChain + LangGraph** — orchestration and agent framework
- **Neo4j** — knowledge graph database
- **Google Gemini** — LLM (API calls)
- **Google embedding** — Google's embeddings (API calls)

Dataset: [ETO ChipExplorer](https://eto.tech/dataset-docs/chipexplorer/) — Georgetown CSET semiconductor supply chain (144 inputs, 398 providers, 1,305 market-share edges).

The system answers complex supply chain questions that vector RAG alone cannot handle:
- Single-country monopoly detection (ASML/NLD controls 100% of EUV)
- Production pipeline traversal (GOES_INTO chain from Chip design → Finished chip)
- Tool taxonomy lookup (all subtypes of Lithography tools via IS_TYPE_OF)
- Geopolitical blast-radius analysis (what breaks if Japan is cut off)

---

## Skills
- **graphrag-builder** - to construction graphRag

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Google Gemini | Default model: `gemini-2.5-flash` or `gemini-2.5-pro` |
| Embeddings | `gemini-embedding-001` | API key |
| Graph DB | Neo4j 5.15 Community | bolt://localhost:7687 |
| Orchestration | LangGraph | StateGraph pattern |
| LLM Framework | LangChain |  |
| Vector Index | Neo4j built-in vector index | On `SpecChunk` nodes |
| API Server | FastAPI | Optional, for serving |


## Project Structure

```
manufacturing-graphrag/
├── CLAUDE.md                  ← this file
├── .env                       ← environment variables (never commit)
├── docker-compose.yml         ← Neo4j container
├── requirements.txt
│
├── data/                      ← ETO semiconductor supply chain CSV files
│   ├── inputs.csv             ← 144 Input nodes (tools, materials, processes)
│   ├── providers.csv          ← 398 Provider nodes (companies + countries)
│   ├── provision.csv          ← 1,305 PROVIDES edges with market share %
│   ├── sequence.csv           ←   139 GOES_INTO + IS_TYPE_OF edges
│   └── stages.csv             ←     3 Stage nodes (S1 Design, S2 Fab, S3 ATP)
│
├── scripts/                   ← one-time setup scripts
│   ├── setup_schema.py        ← create Neo4j constraints and indexes
│   ├── generate_data.py       ← generate synthetic manufacturing data
│   └── ingest.py              ← load data into Neo4j
│
├── graphrag/                  ← main application package
│   ├── __init__.py
│   ├── config.py              ← settings and env loading
│   ├── graph_db.py            ← Neo4j connection and query helpers
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── graph_retriever.py     ← Cypher-based retrieval
│   │   ├── vector_retriever.py    ← semantic search over SpecChunk nodes
│   │   └── hybrid_retriever.py    ← combines both
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py               ← LangGraph TypedDict state
│   │   ├── nodes.py               ← all agent node functions
│   │   ├── routers.py             ← conditional edge functions
│   │   └── graph.py               ← StateGraph assembly
│   │
│   └── prompts/
│       ├── cypher_generation.py   ← prompt templates for Cypher gen
│       ├── extraction.py          ← entity extraction prompts
│       └── answering.py           ← final answer generation prompts
│
├── tests/
│   ├── test_queries.py            ← benchmark questions
│   ├── test_retrieval.py          ← unit tests for retrievers
│   └── test_agent.py              ← integration tests for full agent
│
└── notebooks/
    └── exploration.ipynb          ← for ad-hoc Cypher and retrieval testing
```

---

## Environment Variables

`.env` file (never commit this):

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=TaTc@4221010011

GOOGLE_API_KEY=

LANGCHAIN_TRACING_V2=false
```

Loading in code — always use `config.py`, never hardcode:

```python
# graphrag/config.py
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mfg_pass_2024")
```

---

## Neo4j Graph Schema

### Node Types

| Label | Key Property | Description |
|---|---|---|
| `Input` | `input_id` (unique) | Tool, material, process, design resource, or ultimate output (e.g. EUV lithography tools, Photolithography, Wafer) |
| `Provider` | `provider_id` (unique) | Company (`provider_type = 'organization'`) or country aggregate (`provider_type = 'country'`) |
| `Stage` | `stage_id` (unique) | Production stage: S1 Design · S2 Fabrication · S3 ATP |
| `SpecChunk` | — | Text chunks with embeddings for vector search (from `Input.description`) |

**Input types** (stored as `Input.type` property):
```
process           11  ← pipeline steps: Chip design, Photolithography, Etch and clean…
tool_resource     90  ← physical equipment: EUV scanner, CMP tool, CVD tools…
material_resource 17  ← chemicals, wafers, gases, photoresists…
design_resource    7  ← EDA software, IP cores, CPU/GPU chip designs
ultimate_output    1  ← Finished logic chip
```

### Relationship Types

| Relationship | Direction | Key Properties | Source |
|---|---|---|---|
| `GOES_INTO` | Input → Input | _(none)_ | sequence.csv — 53 edges (pipeline flow) |
| `IS_TYPE_OF` | Input → Input | _(none)_ | sequence.csv — 86 edges (taxonomy hierarchy) |
| `PROVIDES` | Provider → Input | `share_provided` (0–100), `year` | provision.csv — 1,305 edges |
| `IN_STAGE` | Input → Stage | _(none)_ | inputs.csv stage_id — 29 edges |
| `DESCRIBES` | SpecChunk → Input | _(none)_ | Links text chunk to its Input node |

### Important Data Patterns

These are real patterns in the ETO dataset — use them for testing:

- **ASML absolute monopoly:** `share_provided = 100` on EUV lithography tools — one firm, one country (NLD)
- **ASML near-monopolies:** ArF immersion 98.7%, ArF dry 94.3%, KrF 79.2% — dominant across all lithography
- **Single-country NLD:** Netherlands controls the entire EUV supply chain (ASML is only global supplier)
- **Japan concentration (DEU):** Crystal growing furnaces 100% Germany; many wafer/material inputs Japan-concentrated
- **CHN provider cluster:** 93 Chinese organizations supply tools and materials across Fabrication stage

---

## Benchmark Questions

Use these in `tests/test_queries.py` to validate the agent end-to-end:

```python
BENCHMARK_QUESTIONS = [
    # Graph-primary: pipeline traversal (GOES_INTO)
    "What inputs are needed to reach Photolithography in the fabrication pipeline?",
    "Show me the full production pipeline from Chip design to Finished logic chip.",
    "Which process steps directly feed into Chemical mechanical planarization?",

    # Graph-primary: provider / market share (PROVIDES)
    "Who supplies EUV lithography tools and what is their market share?",
    "Which inputs have a single organization controlling 100% of the market?",
    "Which inputs are sourced entirely from one country?",

    # Graph-primary: taxonomy (IS_TYPE_OF)
    "What are all the subtypes of Lithography tools?",
    "List every tool that is a type of Chemical vapor deposition tools.",

    # Vector-primary: semantic on Input.description
    "Find inputs related to photomask defect inspection.",
    "Which materials are used in wafer surface preparation?",

    # Hybrid (graph traversal + vector)
    "Which single-sourced inputs have the highest geopolitical risk?",
    "What would break in the fabrication pipeline if ASML stopped supplying?",

    # Multi-hop risk
    "Which countries have concentration risk across multiple fabrication stages?",
    "Show all inputs where a single country controls more than 80% market share.",
]
```

---

## Code Style

- **Python 3.12+**
- **Async everywhere** — all LLM calls, all Neo4j calls in agent nodes use `async/await`
- **Pydantic for data models** — use `BaseModel` for structured outputs from LLM
- **Type hints on all functions**
- **Docstring on every node function** — one line describing what it does and what it returns
- **No hardcoded credentials** — always from `config.py`
- **No hardcoded Cypher values** — always parameterized with `$variable`
- **Partial state returns from nodes** — return only the keys the node updates, not full state

---

## What NOT to Do

- Do not use `ChatOpenAI`, `ChatAnthropic`, or any external LLM, use `google gemini` only
- Do not call external embedding APIs — use `Google's embedding` locally
- Do not use `CREATE` in ingestion code — always `MERGE`
- Do not hardcode values inside Cypher strings — always use `$params`
- Do not return the full state from a node — return only the keys you updated
- Do not run synchronous LLM or DB calls inside async node functions
- Do not skip `LIMIT` on retrieval Cypher queries during development
- Do not commit `.env` or any file containing credentials
