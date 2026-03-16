"""
extraction.py — Prompts for entity and intent extraction from user questions.
"""
from langchain_core.prompts import ChatPromptTemplate

INTENT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are classifying a semiconductor supply chain question for a GraphRAG routing system.

Classify the question into EXACTLY ONE intent:

- pipeline   : asks about production flow, what goes into what, upstream/downstream steps
- provider   : asks about who supplies something, market share, provider details
- taxonomy   : asks about subtypes, categories, tool families, IS_TYPE_OF relationships
- risk       : asks about supply risk, concentration, geopolitical exposure, blast radius
- semantic   : asks about characteristics, properties, what something is/does (best for vector search)
- hybrid     : needs BOTH graph traversal AND semantic matching (e.g., risk for specific specs)

Also extract named entities: Input names (tools, materials, processes) and Provider names (companies, countries).

Question: {question}

Reply ONLY with valid JSON:
{{
  "intent": "<one of: pipeline|provider|taxonomy|risk|semantic|hybrid>",
  "extracted_entities": ["<entity1>", "<entity2>"]
}}
""")
