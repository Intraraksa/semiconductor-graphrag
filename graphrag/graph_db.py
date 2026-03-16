"""
graph_db.py — Neo4j driver helpers and Neo4jGraph wrapper.
"""
from __future__ import annotations

from neo4j import GraphDatabase, Driver
from langchain_neo4j import Neo4jGraph

from graphrag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def get_driver() -> Driver:
    """Return a Neo4j synchronous driver (caller must close it)."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_neo4j_graph() -> Neo4jGraph:
    """Return a LangChain Neo4jGraph with enhanced_schema=True.

    The enhanced schema includes property types and is fed directly into
    the Cypher generation prompt so the LLM knows exactly what to query.
    """
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        enhanced_schema=True,
    )


def run_query(driver: Driver, cypher: str, params: dict | None = None) -> list[dict]:
    """Execute a read query and return results as a list of dicts."""
    params = params or {}
    with driver.session() as session:
        result = session.run(cypher, **params)
        return [record.data() for record in result]
