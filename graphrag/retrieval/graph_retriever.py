"""
graph_retriever.py — Text-to-Cypher chain for structured supply chain questions.

Uses GraphCypherQAChain with a domain-tuned prompt and Gemini LLM.
"""
from __future__ import annotations

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI

from graphrag.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GOOGLE_API_KEY, GEMINI_LLM_MODEL,
)
from graphrag.prompts.cypher_generation import CYPHER_GENERATION_PROMPT


def get_graph_retriever() -> GraphCypherQAChain:
    """Return a GraphCypherQAChain backed by Gemini and a domain-tuned prompt.

    Returns a chain that accepts {"query": "..."} and returns the LLM answer
    plus intermediate_steps containing the generated Cypher and raw results.
    """
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        enhanced_schema=True,
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        validate_cypher=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        return_intermediate_steps=True,
        top_k=25,
        allow_dangerous_requests=True,
    )
    return chain


def graph_query(chain: GraphCypherQAChain, question: str) -> dict:
    """Run a natural-language question through the Text-to-Cypher chain.

    Returns a dict with 'answer', 'cypher', and 'raw_results' keys.
    """
    result = chain.invoke({"query": question})
    steps = result.get("intermediate_steps", [])
    cypher = steps[0].get("query", "") if steps else ""
    raw = steps[1].get("context", []) if len(steps) > 1 else []
    return {
        "answer": result.get("result", ""),
        "cypher": cypher,
        "raw_results": raw,
    }
