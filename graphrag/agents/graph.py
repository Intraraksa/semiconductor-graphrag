"""
graph.py — LangGraph StateGraph assembly for the semiconductor GraphRAG agent.

Graph topology:

  START
    │
    ▼
  classify_and_extract
    │
    ├──[pipeline|provider|taxonomy|hybrid]──► graph_traversal ──┐
    ├──[risk]────────────────────────────────► risk_assessment──┤
    └──[semantic]────────────────────────────► vector_search ───┤
                                                                 │
                                                                 ▼
                                                        fuse_and_generate
                                                                 │
                                                                END
"""
from langgraph.graph import StateGraph, START, END

from graphrag.agents.state import SemiconductorGraphRAGState
from graphrag.agents.nodes import (
    classify_and_extract,
    graph_traversal,
    vector_search,
    risk_assessment,
    fuse_and_generate,
)
from graphrag.agents.routers import route_by_intent


def build_agent():
    """Compile and return the semiconductor GraphRAG LangGraph agent."""
    builder = StateGraph(SemiconductorGraphRAGState)

    # Register nodes
    builder.add_node("classify_and_extract", classify_and_extract)
    builder.add_node("graph_traversal",      graph_traversal)
    builder.add_node("vector_search",         vector_search)
    builder.add_node("risk_assessment",       risk_assessment)
    builder.add_node("fuse_and_generate",     fuse_and_generate)

    # Entry edge
    builder.add_edge(START, "classify_and_extract")

    # Conditional routing based on intent
    builder.add_conditional_edges(
        "classify_and_extract",
        route_by_intent,
        {
            "graph_traversal": "graph_traversal",
            "vector_search":   "vector_search",
            "risk_assessment": "risk_assessment",
        },
    )

    # All retrieval nodes feed into the answer generation node
    builder.add_edge("graph_traversal",  "fuse_and_generate")
    builder.add_edge("vector_search",     "fuse_and_generate")
    builder.add_edge("risk_assessment",   "fuse_and_generate")

    # Exit
    builder.add_edge("fuse_and_generate", END)

    return builder.compile()


# Module-level compiled agent (import and use directly)
agent = build_agent()


async def ask(question: str) -> str:
    """Convenience wrapper — invoke the agent with a question, return the answer.

    Usage:
        import asyncio
        from graphrag.agents.graph import ask
        print(asyncio.run(ask("Who supplies EUV lithography tools?")))
    """
    result = await agent.ainvoke({
        "question": question,
        "graph_results": [],
        "spec_results": [],
        "extracted_entities": [],
        "cypher_used": "",
        "combined_context": "",
        "answer": "",
        "iterations": 0,
    })
    return result["answer"]
