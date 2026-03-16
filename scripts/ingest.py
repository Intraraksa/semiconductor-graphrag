"""
ingest.py - Load ETO ChipExplorer CSV data into Neo4j and generate embeddings.

Run order (all idempotent via MERGE):
  1. Stage nodes
  2. Input nodes
  3. Provider nodes
  4. Input-[:IN_STAGE]->Stage edges
  5. Provider-[:PROVIDES {share, year}]->Input edges
  6. Input-[:GOES_INTO]->Input  &  Input-[:IS_TYPE_OF]->Input edges
  7. SpecChunk nodes with gemini embeddings  +  SpecChunk-[:DESCRIBES]->Input
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from graphrag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY

DATA_DIR = Path(__file__).parent.parent / "data"

# -- helpers ------------------------------------------------------------------

def run_many(driver: GraphDatabase.driver, query: str, rows: list[dict]) -> None:
    """Execute a parameterized write for each row in a single transaction."""
    if not rows:
        return
    with driver.session() as session:
        session.execute_write(lambda tx: [tx.run(query, row) for row in rows])


def clean(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, replacing NaN with None."""
    return df.where(pd.notna(df), None).to_dict("records")


# -- node ingestion ------------------------------------------------------------

def ingest_stages(driver: GraphDatabase.driver) -> None:
    df = pd.read_csv(DATA_DIR / "stages.csv")
    rows = clean(df[["stage_id", "stage_name", "description"]])
    run_many(driver, """
        MERGE (s:Stage {stage_id: $stage_id})
        SET s.stage_name = $stage_name,
            s.description = $description
    """, rows)
    print(f"  Stages:    {len(rows)} upserted")


def ingest_inputs(driver: GraphDatabase.driver) -> None:
    df = pd.read_csv(DATA_DIR / "inputs.csv")
    rows = clean(df[["input_id", "input_name", "type", "stage_id", "description", "year"]])
    run_many(driver, """
        MERGE (i:Input {input_id: $input_id})
        SET i.input_name  = $input_name,
            i.type        = $type,
            i.stage_id    = $stage_id,
            i.description = $description,
            i.year        = $year
    """, rows)
    print(f"  Inputs:    {len(rows)} upserted")


def ingest_providers(driver: GraphDatabase.driver) -> None:
    df = pd.read_csv(DATA_DIR / "providers.csv")
    rows = clean(df[["provider_id", "provider_name", "alias", "provider_type", "country"]])
    run_many(driver, """
        MERGE (p:Provider {provider_id: $provider_id})
        SET p.provider_name  = $provider_name,
            p.alias          = $alias,
            p.provider_type  = $provider_type,
            p.country        = $country
    """, rows)
    print(f"  Providers: {len(rows)} upserted")


# -- edge ingestion ------------------------------------------------------------

def ingest_in_stage(driver: GraphDatabase.driver) -> None:
    """Link Input nodes to their Stage via IN_STAGE (from inputs.csv stage_id)."""
    df = pd.read_csv(DATA_DIR / "inputs.csv")
    df = df[df["stage_id"].notna()][["input_id", "stage_id"]]
    rows = clean(df)
    run_many(driver, """
        MATCH (i:Input {input_id: $input_id})
        MATCH (s:Stage {stage_id: $stage_id})
        MERGE (i)-[:IN_STAGE]->(s)
    """, rows)
    print(f"  IN_STAGE:  {len(rows)} edges")


def ingest_provides(driver: GraphDatabase.driver) -> None:
    """Provider-[:PROVIDES {share_provided, year}]->Input from provision.csv."""
    df = pd.read_csv(DATA_DIR / "provision.csv")
    all_rows = clean(df[["provider_id", "provided_id", "share_provided", "year"]])

    # Split: Provider->Input vs Provider->Stage (S1/S2/S3 stage-level market share)
    stage_ids = {"S1", "S2", "S3"}
    input_rows = [r for r in all_rows if r["provided_id"] not in stage_ids]
    stage_rows = [r for r in all_rows if r["provided_id"] in stage_ids]

    run_many(driver, """
        MATCH (p:Provider {provider_id: $provider_id})
        MATCH (i:Input    {input_id:    $provided_id})
        MERGE (p)-[r:PROVIDES]->(i)
        SET r.share_provided = $share_provided,
            r.year           = $year
    """, input_rows)

    run_many(driver, """
        MATCH (p:Provider {provider_id: $provider_id})
        MATCH (s:Stage    {stage_id:    $provided_id})
        MERGE (p)-[r:PROVIDES_STAGE]->(s)
        SET r.share_provided = $share_provided,
            r.year           = $year
    """, stage_rows)

    print(f"  PROVIDES:        {len(input_rows)} Input edges")
    print(f"  PROVIDES_STAGE:  {len(stage_rows)} Stage edges")


def ingest_sequence(driver: GraphDatabase.driver) -> None:
    """GOES_INTO and IS_TYPE_OF edges from sequence.csv."""
    df = pd.read_csv(DATA_DIR / "sequence.csv")

    goes_into = clean(
        df[df["goes_into_id"].notna()][["input_id", "goes_into_id"]].rename(
            columns={"goes_into_id": "target_id"}
        )
    )
    run_many(driver, """
        MATCH (a:Input {input_id: $input_id})
        MATCH (b:Input {input_id: $target_id})
        MERGE (a)-[:GOES_INTO]->(b)
    """, goes_into)
    print(f"  GOES_INTO: {len(goes_into)} edges")

    is_type_of = clean(
        df[df["is_type_of_id"].notna()][["input_id", "is_type_of_id"]].rename(
            columns={"is_type_of_id": "target_id"}
        )
    )
    run_many(driver, """
        MATCH (a:Input {input_id: $input_id})
        MATCH (b:Input {input_id: $target_id})
        MERGE (a)-[:IS_TYPE_OF]->(b)
    """, is_type_of)
    print(f"  IS_TYPE_OF:{len(is_type_of)} edges")


# -- embeddings ----------------------------------------------------------------

def ingest_spec_chunks(driver: GraphDatabase.driver) -> None:
    """Generate embeddings for Input.description and store as SpecChunk nodes."""
    df = pd.read_csv(DATA_DIR / "inputs.csv")
    df = df[df["description"].notna()][["input_id", "input_name", "description"]]

    print(f"  Generating embeddings for {len(df)} Input descriptions ...")
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    texts = df["description"].tolist()

    # Batch in groups of 10 to avoid rate-limit bursts
    batch_size = 10
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(embedder.embed_documents(batch))
        print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        if i + batch_size < len(texts):
            time.sleep(1)  # polite pause between batches

    rows = [
        {
            "input_id": row["input_id"],
            "text": row["description"],
            "embedding": all_embeddings[idx],
        }
        for idx, (_, row) in enumerate(df.iterrows())
    ]

    run_many(driver, """
        MERGE (chunk:SpecChunk {input_id: $input_id})
        SET chunk.text      = $text,
            chunk.embedding = $embedding
        WITH chunk
        MATCH (i:Input {input_id: $input_id})
        MERGE (chunk)-[:DESCRIBES]->(i)
    """, rows)
    print(f"  SpecChunks:{len(rows)} nodes + DESCRIBES edges")


# -- main ----------------------------------------------------------------------

def main() -> None:
    print(f"Connecting to {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected.\n")

        print("--- Nodes ---")
        ingest_stages(driver)
        ingest_inputs(driver)
        ingest_providers(driver)

        print("\n-- Edges -------------------------------")
        ingest_in_stage(driver)
        ingest_provides(driver)
        ingest_sequence(driver)

        print("\n-- Embeddings --------------------------")
        ingest_spec_chunks(driver)

        print("\nIngestion complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
