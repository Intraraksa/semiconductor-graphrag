"""
test_retrieval.py — Unit tests for graph and vector retrievers.

Requires a running Neo4j instance with ingested data.
Run: python -m pytest tests/test_retrieval.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphrag.graph_db import get_driver, run_query


def test_neo4j_connectivity() -> None:
    """Neo4j should be reachable."""
    driver = get_driver()
    driver.verify_connectivity()
    driver.close()


def test_input_count() -> None:
    """Should have 126 Input nodes (actual CSV count)."""
    driver = get_driver()
    result = run_query(driver, "MATCH (n:Input) RETURN count(n) AS cnt")
    driver.close()
    assert result[0]["cnt"] == 126, f"Expected 126 Input nodes, got {result[0]['cnt']}"


def test_provider_count() -> None:
    """Should have ~393 Provider nodes (397 rows, 4 duplicate IDs collapsed by MERGE)."""
    driver = get_driver()
    result = run_query(driver, "MATCH (n:Provider) RETURN count(n) AS cnt")
    driver.close()
    cnt = result[0]["cnt"]
    assert 390 <= cnt <= 400, f"Expected ~393 Provider nodes, got {cnt}"


def test_provides_count() -> None:
    """Should have 1276 PROVIDES (Input) edges + 21 PROVIDES_STAGE edges = 1297 total.
    8 provision rows reference N68 which is absent from inputs.csv."""
    driver = get_driver()
    r1 = run_query(driver, "MATCH ()-[r:PROVIDES]->() RETURN count(r) AS cnt")
    r2 = run_query(driver, "MATCH ()-[r:PROVIDES_STAGE]->() RETURN count(r) AS cnt")
    driver.close()
    total = r1[0]["cnt"] + r2[0]["cnt"]
    assert total == 1297, f"Expected 1297 total provision edges, got {total}"


def test_goes_into_edges() -> None:
    """Should have ~53 GOES_INTO edges."""
    driver = get_driver()
    result = run_query(driver, "MATCH ()-[r:GOES_INTO]->() RETURN count(r) AS cnt")
    driver.close()
    assert result[0]["cnt"] >= 50, f"Too few GOES_INTO edges: {result[0]['cnt']}"


def test_is_type_of_edges() -> None:
    """Should have ~86 IS_TYPE_OF edges."""
    driver = get_driver()
    result = run_query(driver, "MATCH ()-[r:IS_TYPE_OF]->() RETURN count(r) AS cnt")
    driver.close()
    assert result[0]["cnt"] >= 80, f"Too few IS_TYPE_OF edges: {result[0]['cnt']}"


def test_euv_monopoly_cypher() -> None:
    """ASML should control 100% of EUV lithography tools."""
    driver = get_driver()
    result = run_query(driver, """
        MATCH (p:Provider)-[r:PROVIDES]->(i:Input)
        WHERE toLower(i.input_name) CONTAINS 'euv'
          AND r.share_provided = 100
        RETURN p.provider_name AS provider, p.country AS country
        LIMIT 5
    """)
    driver.close()
    assert result, "No 100%-share provider found for EUV inputs"
    providers = [r["provider"] for r in result]
    assert any("ASML" in p for p in providers), f"ASML not found; got: {providers}"


def test_lithography_taxonomy() -> None:
    """IS_TYPE_OF traversal should find EUV/ArF/KrF under Lithography tools."""
    driver = get_driver()
    result = run_query(driver, """
        MATCH (sub:Input)-[:IS_TYPE_OF*1..3]->(parent:Input)
        WHERE toLower(parent.input_name) CONTAINS 'lithography'
        RETURN sub.input_name AS subtype LIMIT 20
    """)
    driver.close()
    subtypes = [r["subtype"] for r in result]
    assert subtypes, "No subtypes found for Lithography tools"
    assert any("EUV" in s or "ArF" in s or "KrF" in s for s in subtypes), \
        f"Expected EUV/ArF/KrF subtypes, got: {subtypes}"


def test_spec_chunks_exist() -> None:
    """SpecChunk nodes should exist after ingestion."""
    driver = get_driver()
    result = run_query(driver, "MATCH (n:SpecChunk) RETURN count(n) AS cnt")
    driver.close()
    assert result[0]["cnt"] > 0, "No SpecChunk nodes found — run ingest.py first"


def test_vector_index_exists() -> None:
    """The spec_embeddings vector index should be present."""
    driver = get_driver()
    result = run_query(driver, "SHOW INDEXES YIELD name WHERE name = 'spec_embeddings'")
    driver.close()
    assert result, "Vector index 'spec_embeddings' not found — run setup_schema.py"
