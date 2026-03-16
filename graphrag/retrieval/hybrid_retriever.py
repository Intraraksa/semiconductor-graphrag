"""
hybrid_retriever.py — Combines graph traversal and vector search results.

For hybrid-intent questions, runs both retrievers and merges their outputs
before passing to the answer generation node.
"""
from __future__ import annotations

from langchain_neo4j import GraphCypherQAChain, Neo4jVector

from graphrag.retrieval.graph_retriever import graph_query
from graphrag.retrieval.vector_retriever import vector_search


class HybridRetriever:
    """Wraps both retrievers and exposes a unified retrieve() interface."""

    def __init__(
        self,
        graph_chain: GraphCypherQAChain,
        vector_store: Neo4jVector,
    ) -> None:
        self.graph_chain = graph_chain
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        *,
        graph: bool = True,
        vector: bool = True,
        vector_k: int = 5,
    ) -> dict:
        """Run graph and/or vector retrieval and return merged results.

        Returns a dict with keys 'graph_results', 'spec_results', 'cypher'.
        """
        graph_results: list[dict] = []
        spec_results: list[str] = []
        cypher: str = ""

        if graph:
            g = graph_query(self.graph_chain, question)
            graph_results = g["raw_results"]
            cypher = g["cypher"]

        if vector:
            spec_results = vector_search(self.vector_store, question, k=vector_k)

        return {
            "graph_results": graph_results,
            "spec_results": spec_results,
            "cypher": cypher,
        }
