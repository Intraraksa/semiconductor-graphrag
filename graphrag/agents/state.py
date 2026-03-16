"""
state.py — LangGraph TypedDict state for the semiconductor GraphRAG agent.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict


class SemiconductorGraphRAGState(TypedDict):
    """Full state passed between agent nodes.

    Nodes return PARTIAL state — only the keys they update.
    The 'iterations' field uses LangGraph's reducer pattern.
    """
    question: str
    intent: Literal["pipeline", "provider", "taxonomy", "risk", "semantic", "hybrid"]
    extracted_entities: list[str]   # Input names or Provider names from the question
    graph_results: list[dict]       # Raw Cypher query results
    spec_results: list[str]         # Formatted vector search context strings
    cypher_used: str                # Last Cypher query generated (for debugging)
    combined_context: str           # Merged context passed to answer generation
    answer: str                     # Final answer from the LLM
    iterations: Annotated[int, operator.add]   # Incremented by each node visit
