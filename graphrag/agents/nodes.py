"""
nodes.py — All LangGraph node functions for the semiconductor GraphRAG agent.

Each node is an async function that accepts the full state and returns a
PARTIAL state dict containing only the keys it updates.
"""
from __future__ import annotations

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from graphrag.config import GOOGLE_API_KEY, GEMINI_LLM_MODEL
from graphrag.graph_db import get_driver, run_query
from graphrag.prompts.extraction import INTENT_EXTRACTION_PROMPT
from graphrag.prompts.answering import ANSWER_GENERATION_PROMPT
from graphrag.agents.state import SemiconductorGraphRAGState

# ── Lazy singletons (created once per process) ────────────────────────────────

_llm: ChatGoogleGenerativeAI | None = None
_graph_chain = None
_vector_store = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )
    return _llm


def _get_graph_chain():
    global _graph_chain
    if _graph_chain is None:
        from graphrag.retrieval.graph_retriever import get_graph_retriever
        _graph_chain = get_graph_retriever()
    return _graph_chain


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from graphrag.retrieval.vector_retriever import get_vector_store
        _vector_store = get_vector_store()
    return _vector_store


# ── Node functions ─────────────────────────────────────────────────────────────

async def classify_and_extract(state: SemiconductorGraphRAGState) -> dict:
    """Classify question intent and extract named entities.

    Returns: intent, extracted_entities, iterations +1.
    """
    llm = _get_llm()
    chain = INTENT_EXTRACTION_PROMPT | llm

    response = await chain.ainvoke({"question": state["question"]})

    # Parse JSON from LLM response — strip markdown fences if present
    text = response.content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        intent = parsed.get("intent", "hybrid")
        entities = parsed.get("extracted_entities", [])
    except (json.JSONDecodeError, AttributeError):
        intent = "hybrid"
        entities = []

    return {
        "intent": intent,
        "extracted_entities": entities,
        "iterations": 1,
    }


async def graph_traversal(state: SemiconductorGraphRAGState) -> dict:
    """Run the Text-to-Cypher chain for structured graph questions.

    Returns: graph_results, cypher_used, iterations +1.
    """
    from graphrag.retrieval.graph_retriever import graph_query

    chain = _get_graph_chain()
    result = graph_query(chain, state["question"])

    return {
        "graph_results": result["raw_results"],
        "cypher_used": result["cypher"],
        "iterations": 1,
    }


async def vector_search(state: SemiconductorGraphRAGState) -> dict:
    """Run semantic search over SpecChunk nodes for description-based questions.

    Returns: spec_results, iterations +1.
    """
    from graphrag.retrieval.vector_retriever import vector_search as vs

    store = _get_vector_store()
    spec_results = vs(store, state["question"], k=6)

    return {
        "spec_results": spec_results,
        "iterations": 1,
    }


async def risk_assessment(state: SemiconductorGraphRAGState) -> dict:
    """Run specialized multi-hop concentration-risk Cypher queries.

    Executes a fixed set of risk-focused queries against Neo4j and returns
    the raw results as graph_results. Falls back to Text-to-Cypher for
    question-specific details.

    Returns: graph_results, cypher_used, iterations +1.
    """
    driver = get_driver()
    try:
        # 1. Single-organization monopolies (100% share)
        monopolies = run_query(driver, """
            MATCH (p:Provider {provider_type: 'organization'})-[r:PROVIDES]->(i:Input)
            WHERE r.share_provided = 100
            RETURN i.input_name AS input, p.provider_name AS sole_provider,
                   p.country AS country, r.year AS year
            ORDER BY i.input_name LIMIT 25
        """)

        # 2. Single-country control (≥ 95% share)
        country_control = run_query(driver, """
            MATCH (p:Provider {provider_type: 'country'})-[r:PROVIDES]->(i:Input)
            WHERE r.share_provided >= 95
            RETURN i.input_name AS input, p.provider_name AS country,
                   r.share_provided AS share, r.year AS year
            ORDER BY r.share_provided DESC LIMIT 25
        """)

        # 3. Multi-stage concentration by country
        stage_concentration = run_query(driver, """
            MATCH (p:Provider {provider_type: 'country'})-[r:PROVIDES]->(i:Input)-[:IN_STAGE]->(s:Stage)
            WHERE r.share_provided >= 50
            WITH p.provider_name AS country, s.stage_name AS stage,
                 COUNT(DISTINCT i) AS concentrated_inputs,
                 AVG(r.share_provided) AS avg_share
            WHERE concentrated_inputs >= 2
            RETURN country, stage, concentrated_inputs, round(avg_share, 1) AS avg_share
            ORDER BY concentrated_inputs DESC LIMIT 20
        """)

        combined = {
            "monopolies": monopolies,
            "country_control": country_control,
            "stage_concentration": stage_concentration,
        }
    finally:
        driver.close()

    # Also run the Text-to-Cypher chain for the specific question
    from graphrag.retrieval.graph_retriever import graph_query
    chain = _get_graph_chain()
    specific = graph_query(chain, state["question"])

    all_results = specific["raw_results"] + [combined]

    return {
        "graph_results": all_results,
        "cypher_used": specific["cypher"],
        "iterations": 1,
    }


async def fuse_and_generate(state: SemiconductorGraphRAGState) -> dict:
    """Merge graph and vector context, then generate the final answer.

    For semantic/hybrid intents with no graph results, automatically runs
    vector search as a fallback so semantic questions always get an answer.

    Returns: combined_context, answer, iterations +1.
    """
    llm = _get_llm()

    # Fallback: if graph returned nothing and intent suggests semantic match,
    # run vector search to populate spec_results
    intent = state.get("intent", "hybrid")
    if not state.get("graph_results") and intent in ("semantic", "hybrid") and not state.get("spec_results"):
        from graphrag.retrieval.vector_retriever import vector_search as vs
        store = _get_vector_store()
        extra = vs(store, state["question"], k=6)
        state = {**state, "spec_results": extra}

    # Format graph results
    graph_text = ""
    if state.get("graph_results"):
        graph_text = "\n".join(
            str(r) for r in state["graph_results"][:30]
        )
    else:
        graph_text = "(no graph data retrieved)"

    # Format vector results
    spec_text = ""
    if state.get("spec_results"):
        spec_text = "\n\n".join(state["spec_results"][:5])
    else:
        spec_text = "(no semantic matches)"

    combined = f"GRAPH RESULTS:\n{graph_text}\n\nSEMANTIC RESULTS:\n{spec_text}"

    chain = ANSWER_GENERATION_PROMPT | llm
    response = await chain.ainvoke({
        "graph_results": graph_text,
        "spec_results": spec_text,
        "question": state["question"],
    })

    return {
        "combined_context": combined,
        "answer": response.content,
        "iterations": 1,
    }
