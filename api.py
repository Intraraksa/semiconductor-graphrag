"""
api.py — FastAPI server for the Semiconductor GraphRAG system.

Endpoints
---------
GET  /health           — liveness probe + Neo4j connection check
POST /query            — full agent pipeline (classify → retrieve → answer)
POST /graph            — graph-only retrieval (Text-to-Cypher)
POST /vector           — vector-only semantic search
POST /hybrid           — explicit hybrid retrieval (graph + vector)
GET  /schema           — live Neo4j graph schema
GET  /benchmark        — list of benchmark questions

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from graphrag.agents.graph import agent, ask
from graphrag.config import NEO4J_URI, NEO4J_USER
from graphrag.graph_db import get_driver, get_neo4j_graph
from graphrag.retrieval.graph_retriever import get_graph_retriever, graph_query
from graphrag.retrieval.vector_retriever import get_vector_store, vector_search

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("graphrag.api")

# ---------------------------------------------------------------------------
# Application state (singletons initialised once at startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise Neo4j driver and retriever singletons on startup."""
    log.info("Starting up — connecting to Neo4j at %s", NEO4J_URI)
    try:
        _state["driver"] = get_driver()
        _state["graph_chain"] = get_graph_retriever()
        _state["vector_store"] = get_vector_store()
        _state["neo4j_graph"] = get_neo4j_graph()
        log.info("All singletons ready")
    except Exception as exc:
        log.error("Startup failed: %s", exc)
        raise

    yield  # server is running

    log.info("Shutting down — closing Neo4j driver")
    driver = _state.pop("driver", None)
    if driver:
        driver.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Semiconductor Supply Chain GraphRAG",
    description=(
        "GraphRAG API backed by Neo4j + LangGraph + Google Gemini. "
        "Answers complex semiconductor supply chain questions that vector RAG alone cannot handle."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1_000,
        examples=["Who supplies EUV lithography tools and what is their market share?"],
    )


class AgentResponse(BaseModel):
    question: str
    answer: str
    intent: str
    extracted_entities: list[str]
    cypher_used: str
    graph_results: list[dict]
    spec_results: list[str]
    iterations: int
    latency_ms: float


class GraphResponse(BaseModel):
    question: str
    cypher: str
    answer: str
    raw_results: list[dict]
    latency_ms: float


class VectorResult(BaseModel):
    text: str


class VectorResponse(BaseModel):
    question: str
    results: list[VectorResult]
    latency_ms: float


class HybridResponse(BaseModel):
    question: str
    cypher: str
    graph_results: list[dict]
    spec_results: list[str]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    neo4j_uri: str
    neo4j_user: str
    node_count: int | None = None


class SchemaResponse(BaseModel):
    schema_text: str


class BenchmarkResponse(BaseModel):
    questions: list[str]


# ---------------------------------------------------------------------------
# Benchmark questions (from CLAUDE.md)
# ---------------------------------------------------------------------------
BENCHMARK_QUESTIONS: list[str] = [
    "What inputs are needed to reach Photolithography in the fabrication pipeline?",
    "Show me the full production pipeline from Chip design to Finished logic chip.",
    "Which process steps directly feed into Chemical mechanical planarization?",
    "Who supplies EUV lithography tools and what is their market share?",
    "Which inputs have a single organization controlling 100% of the market?",
    "Which inputs are sourced entirely from one country?",
    "What are all the subtypes of Lithography tools?",
    "List every tool that is a type of Chemical vapor deposition tools.",
    "Find inputs related to photomask defect inspection.",
    "Which materials are used in wafer surface preparation?",
    "Which single-sourced inputs have the highest geopolitical risk?",
    "What would break in the fabrication pipeline if ASML stopped supplying?",
    "Which countries have concentration risk across multiple fabrication stages?",
    "Show all inputs where a single country controls more than 80% market share.",
]


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness probe. Runs a lightweight Cypher query to verify Neo4j is up."""
    driver = _state.get("driver")
    neo4j_ok = False
    node_count: int | None = None

    if driver:
        try:
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) AS cnt")
                node_count = result.single()["cnt"]
            neo4j_ok = True
        except Exception as exc:
            log.warning("Neo4j health check failed: %s", exc)

    return HealthResponse(
        status="ok" if neo4j_ok else "degraded",
        neo4j_connected=neo4j_ok,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        node_count=node_count,
    )


@app.get("/schema", response_model=SchemaResponse, tags=["ops"])
async def schema():
    """Return the live Neo4j graph schema (node labels, relationship types, properties)."""
    neo4j_graph = _state.get("neo4j_graph")
    if not neo4j_graph:
        raise HTTPException(status_code=503, detail="Neo4j graph not initialised")
    try:
        neo4j_graph.refresh_schema()
        return SchemaResponse(schema_text=neo4j_graph.schema)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/benchmark", response_model=BenchmarkResponse, tags=["ops"])
async def benchmark():
    """Return the list of benchmark questions for testing the system."""
    return BenchmarkResponse(questions=BENCHMARK_QUESTIONS)


@app.post("/query", response_model=AgentResponse, tags=["query"])
async def query(body: QuestionRequest):
    """Full agent pipeline: classify intent → retrieve (graph / vector / risk) → generate answer.

    This is the primary endpoint. The agent automatically picks the best retrieval
    strategy (graph traversal, vector search, risk assessment, or hybrid) based on
    the question's intent.
    """
    log.info("POST /query  question=%r", body.question[:80])
    t0 = time.perf_counter()

    try:
        result = await agent.ainvoke({
            "question": body.question,
            "graph_results": [],
            "spec_results": [],
            "extracted_entities": [],
            "cypher_used": "",
            "combined_context": "",
            "answer": "",
            "iterations": 0,
        })
    except Exception as exc:
        log.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = (time.perf_counter() - t0) * 1000
    log.info("POST /query  intent=%s  latency=%.0f ms", result.get("intent"), latency)

    return AgentResponse(
        question=body.question,
        answer=result.get("answer", ""),
        intent=result.get("intent", ""),
        extracted_entities=result.get("extracted_entities", []),
        cypher_used=result.get("cypher_used", ""),
        graph_results=result.get("graph_results", []),
        spec_results=result.get("spec_results", []),
        iterations=result.get("iterations", 0),
        latency_ms=round(latency, 1),
    )


@app.post("/graph", response_model=GraphResponse, tags=["query"])
async def graph_endpoint(body: QuestionRequest):
    """Graph-only retrieval via Text-to-Cypher (skips vector search and intent routing).

    Useful for debugging Cypher generation or for questions you know are graph-primary.
    Returns the generated Cypher, the LLM answer, and the raw Neo4j rows.
    """
    log.info("POST /graph  question=%r", body.question[:80])
    t0 = time.perf_counter()

    chain = _state.get("graph_chain")
    if not chain:
        raise HTTPException(status_code=503, detail="Graph chain not initialised")

    try:
        result = graph_query(chain, body.question)
    except Exception as exc:
        log.exception("Graph retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = (time.perf_counter() - t0) * 1000
    return GraphResponse(
        question=body.question,
        cypher=result.get("cypher", ""),
        answer=result.get("answer", ""),
        raw_results=result.get("raw_results", []),
        latency_ms=round(latency, 1),
    )


@app.post("/vector", response_model=VectorResponse, tags=["query"])
async def vector_endpoint(body: QuestionRequest):
    """Vector-only semantic search over SpecChunk nodes (skips graph traversal).

    Returns the top-k semantically similar input descriptions with their provider context.
    """
    log.info("POST /vector  question=%r", body.question[:80])
    t0 = time.perf_counter()

    store = _state.get("vector_store")
    if not store:
        raise HTTPException(status_code=503, detail="Vector store not initialised")

    try:
        texts = vector_search(store, body.question, k=6)
    except Exception as exc:
        log.exception("Vector search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = (time.perf_counter() - t0) * 1000
    return VectorResponse(
        question=body.question,
        results=[VectorResult(text=t) for t in texts],
        latency_ms=round(latency, 1),
    )


@app.post("/hybrid", response_model=HybridResponse, tags=["query"])
async def hybrid_endpoint(body: QuestionRequest):
    """Explicit hybrid retrieval — runs both graph and vector in sequence.

    Returns raw results from both retrievers without LLM answer synthesis.
    Use /query for the full pipeline with a generated answer.
    """
    log.info("POST /hybrid  question=%r", body.question[:80])
    t0 = time.perf_counter()

    chain = _state.get("graph_chain")
    store = _state.get("vector_store")
    if not chain or not store:
        raise HTTPException(status_code=503, detail="Retrievers not initialised")

    try:
        g = graph_query(chain, body.question)
        v = vector_search(store, body.question, k=5)
    except Exception as exc:
        log.exception("Hybrid retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = (time.perf_counter() - t0) * 1000
    return HybridResponse(
        question=body.question,
        cypher=g.get("cypher", ""),
        graph_results=g.get("raw_results", []),
        spec_results=v,
        latency_ms=round(latency, 1),
    )
