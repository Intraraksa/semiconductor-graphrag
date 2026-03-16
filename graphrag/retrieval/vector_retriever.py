"""
vector_retriever.py — Semantic search over SpecChunk nodes using Neo4j vector index.

The retrieval_query traverses SpecChunk → Input → Provider to return
rich metadata alongside each matched chunk.
"""
from __future__ import annotations

from langchain_neo4j import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from graphrag.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GOOGLE_API_KEY, GEMINI_EMBED_MODEL,
)

# After matching a SpecChunk by vector similarity, traverse to its Input
# and collect provider context.
_RETRIEVAL_QUERY = """
MATCH (node)-[:DESCRIBES]->(i:Input)
OPTIONAL MATCH (p:Provider)-[r:PROVIDES]->(i)
WHERE r.share_provided >= 50
RETURN node.text AS text,
       score,
       {
         input_id:    i.input_id,
         input_name:  i.input_name,
         type:        i.type,
         top_providers: COLLECT({
           name:          p.provider_name,
           country:       p.country,
           provider_type: p.provider_type,
           share:         r.share_provided
         })[0..5]
       } AS metadata
"""


def get_vector_store() -> Neo4jVector:
    """Connect to the existing spec_embeddings vector index in Neo4j.

    Returns a Neo4jVector retriever ready for similarity_search calls.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="spec_embeddings",
        node_label="SpecChunk",
        text_node_property="text",
        embedding_node_property="embedding",
        retrieval_query=_RETRIEVAL_QUERY,
    )


def vector_search(store: Neo4jVector, question: str, k: int = 5) -> list[str]:
    """Return formatted context strings from top-k semantic matches.

    Each string contains the chunk text plus Input name and top providers.
    """
    docs = store.similarity_search(question, k=k)
    results: list[str] = []
    for doc in docs:
        meta = doc.metadata
        providers_str = ", ".join(
            f"{p.get('name', '?')} ({p.get('country', '?')}) {p.get('share', '?')}%"
            for p in (meta.get("top_providers") or [])
        )
        results.append(
            f"[{meta.get('input_name', '?')} | type: {meta.get('type', '?')}]\n"
            f"{doc.page_content}\n"
            + (f"Top providers: {providers_str}" if providers_str else "")
        )
    return results
