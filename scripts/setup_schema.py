"""
setup_schema.py — Create Neo4j constraints, indexes, and vector index.
Run this ONCE before ingesting any data.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neo4j import GraphDatabase
from graphrag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

CONSTRAINTS = [
    "CREATE CONSTRAINT input_id IF NOT EXISTS FOR (n:Input) REQUIRE n.input_id IS UNIQUE",
    "CREATE CONSTRAINT provider_id IF NOT EXISTS FOR (n:Provider) REQUIRE n.provider_id IS UNIQUE",
    "CREATE CONSTRAINT stage_id IF NOT EXISTS FOR (n:Stage) REQUIRE n.stage_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX input_type IF NOT EXISTS FOR (n:Input) ON (n.type)",
    "CREATE INDEX input_name IF NOT EXISTS FOR (n:Input) ON (n.input_name)",
    "CREATE INDEX provider_country IF NOT EXISTS FOR (n:Provider) ON (n.country)",
    "CREATE INDEX provider_type_prop IF NOT EXISTS FOR (n:Provider) ON (n.provider_type)",
    "CREATE INDEX provider_name IF NOT EXISTS FOR (n:Provider) ON (n.provider_name)",
]

# Vector index for SpecChunk nodes (gemini-embedding-001 → 768 dimensions)
VECTOR_INDEX = """
CREATE VECTOR INDEX spec_embeddings IF NOT EXISTS
FOR (n:SpecChunk) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
"""


def setup(driver: GraphDatabase.driver) -> None:
    with driver.session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)
            print(f"  OK constraint: {stmt[:80]}")

        for stmt in INDEXES:
            session.run(stmt)
            print(f"  OK index:      {stmt[:80]}")

        # Vector index — skip if already exists
        existing = session.run("SHOW INDEXES YIELD name").value("name")
        if "spec_embeddings" in existing:
            print("  OK vector index: spec_embeddings already exists")
        else:
            session.run(VECTOR_INDEX)
            print("  OK vector index: spec_embeddings created (768-dim cosine)")


def main() -> None:
    print(f"Connecting to {NEO4J_URI} …")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected.\n")
        setup(driver)
        print("\nSchema setup complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
