"""
cypher_generation.py — Domain-tuned Cypher generation prompt for the ETO
semiconductor supply chain dataset.
"""
from langchain_core.prompts import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query writer for a semiconductor supply chain knowledge graph built from the ETO ChipExplorer dataset.

GRAPH SCHEMA:
{schema}

NODE LABELS AND KEY PROPERTIES:
- Input       — input_id (unique), input_name, type (process|tool_resource|material_resource|design_resource|ultimate_output), description
- Provider    — provider_id (unique), provider_name, provider_type ('organization' or 'country'), country, alias
- Stage       — stage_id (unique), stage_name ('Design'|'Fabrication'|'Assembly, testing, and packaging (ATP)')
- SpecChunk   — text, embedding  [vector-only; do not include in structural queries]

RELATIONSHIP TYPES:
- (Provider)-[:PROVIDES {{share_provided: float, year: int}}]->(Input)   — market share 0-100
- (Input)-[:GOES_INTO]->(Input)     — production pipeline flow (upstream → downstream)
- (Input)-[:IS_TYPE_OF]->(Input)    — taxonomy/hierarchy (specific → general)
- (Input)-[:IN_STAGE]->(Stage)      — which production stage an input belongs to
- (SpecChunk)-[:DESCRIBES]->(Input) — grounding link for vector search

IMPORTANT DATA FACTS:
- ASML (Netherlands) controls 100% of EUV lithography tools — the only global supplier
- Country-type Providers have provider_type='country'; company-type have provider_type='organization'
- share_provided=100 on a country-type Provider means single-country control
- GOES_INTO traversal follows production flow: e.g., Lithography tools → Photolithography → Finished chip
- IS_TYPE_OF traversal finds specific types: e.g., EUV scanner IS_TYPE_OF Lithography tools

CYPHER WRITING RULES:
1. Always add LIMIT 25 unless the question explicitly asks for all results
2. Use case-insensitive inline string literals in WHERE clauses — extract the key term from the question and embed it directly, e.g.: WHERE toLower(n.input_name) CONTAINS 'photolithography'
3. NEVER use $variable parameters — there is no parameter substitution. Use literal strings only.
4. For pipeline UPSTREAM of X: MATCH (x)<-[:GOES_INTO*1..6]-(upstream)
5. For pipeline DOWNSTREAM of X: MATCH (x)-[:GOES_INTO*1..6]->(downstream)
6. For taxonomy subtypes: MATCH (sub)-[:IS_TYPE_OF*1..4]->(parent)
7. For market share aggregation: WITH + COUNT/SUM after MATCH, then filter with WHERE
8. For single-source detection: WITH c, COUNT(DISTINCT p) AS n WHERE n = 1
9. For geopolitical risk: filter on Provider WHERE provider_type = 'country'
10. NEVER use MERGE, DELETE, CREATE, or SET — read-only queries only
11. OPTIONAL MATCH when a relationship may not exist for all nodes
12. Always ORDER BY and LIMIT for performance
13. Always include i.input_name or i.input_name in RETURN so the answering model knows what is being referenced

EXAMPLE PATTERNS:

Q: Who supplies EUV lithography tools and what is their market share?
Cypher: MATCH (p:Provider)-[r:PROVIDES]->(i:Input)
        WHERE toLower(i.input_name) CONTAINS 'euv'
        RETURN i.input_name, p.provider_name, p.country, p.provider_type, r.share_provided, r.year
        ORDER BY r.share_provided DESC LIMIT 25

Q: Which inputs have a single organization controlling 100% of the market?
Cypher: MATCH (p:Provider {{provider_type: 'organization'}})-[r:PROVIDES]->(i:Input)
        WHERE r.share_provided = 100
        RETURN i.input_name, p.provider_name, p.country, r.year
        ORDER BY i.input_name LIMIT 25

Q: Which inputs are sourced entirely from one country?
Cypher: MATCH (p:Provider {{provider_type: 'country'}})-[r:PROVIDES]->(i:Input)
        WHERE r.share_provided >= 100
        RETURN i.input_name, p.provider_name AS country, r.share_provided, r.year
        ORDER BY i.input_name LIMIT 25

Q: What inputs are needed to reach Photolithography in the fabrication pipeline?
Cypher: MATCH (target:Input)
        WHERE toLower(target.input_name) CONTAINS 'photolithography'
        MATCH (upstream:Input)-[:GOES_INTO*1..6]->(target)
        RETURN DISTINCT upstream.input_name, upstream.type
        ORDER BY upstream.type LIMIT 25

Q: Show the full production pipeline from Chip design to Finished logic chip.
Cypher: MATCH (start:Input), (end:Input)
        WHERE toLower(start.input_name) CONTAINS 'chip design'
          AND toLower(end.input_name) CONTAINS 'finished'
        MATCH path = (start)-[:GOES_INTO*1..10]->(end)
        RETURN [n IN nodes(path) | n.input_name] AS pipeline, length(path) AS steps
        LIMIT 5

Q: What are all the subtypes of Lithography tools?
Cypher: MATCH (sub:Input)-[:IS_TYPE_OF*1..3]->(parent:Input)
        WHERE toLower(parent.input_name) CONTAINS 'lithography'
        RETURN sub.input_name, sub.type, parent.input_name AS parent
        ORDER BY sub.input_name LIMIT 25

Q: What would break in the fabrication pipeline if ASML stopped supplying?
Cypher: MATCH (p:Provider)-[:PROVIDES]->(i:Input)
        WHERE toLower(p.provider_name) CONTAINS 'asml'
        OPTIONAL MATCH (i)-[:GOES_INTO*1..5]->(downstream:Input)
        RETURN i.input_name AS direct_impact,
               COLLECT(DISTINCT downstream.input_name) AS cascade_effects
        LIMIT 25

Q: Which countries have concentration risk across multiple fabrication stages?
Cypher: MATCH (p:Provider {{provider_type: 'country'}})-[r:PROVIDES]->(i:Input)-[:IN_STAGE]->(s:Stage)
        WHERE r.share_provided >= 50
        WITH p.provider_name AS country, s.stage_name AS stage, COUNT(DISTINCT i) AS concentrated_inputs
        WHERE concentrated_inputs >= 2
        RETURN country, stage, concentrated_inputs
        ORDER BY concentrated_inputs DESC LIMIT 25

QUESTION: {question}
CYPHER (output the Cypher query only, no prefix, no explanation):"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)
