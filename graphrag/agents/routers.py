"""
routers.py — Conditional edge functions for the LangGraph StateGraph.
"""
from __future__ import annotations

from graphrag.agents.state import SemiconductorGraphRAGState


def route_by_intent(state: SemiconductorGraphRAGState) -> str:
    """Map the classified intent to the appropriate retrieval node name.

    Returns one of: 'graph_traversal', 'vector_search', 'risk_assessment'.
    """
    intent = state.get("intent", "hybrid")

    routing = {
        "pipeline":  "graph_traversal",
        "provider":  "graph_traversal",
        "taxonomy":  "graph_traversal",
        "hybrid":    "graph_traversal",
        "risk":      "risk_assessment",
        "semantic":  "vector_search",
    }

    return routing.get(intent, "graph_traversal")
