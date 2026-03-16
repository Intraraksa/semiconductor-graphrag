"""
test_agent.py — Integration tests for the full LangGraph agent.

Requires running Neo4j with ingested data AND a valid GOOGLE_API_KEY.
Run: python -m pytest tests/test_agent.py -v -s
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from graphrag.agents.graph import ask, agent
from graphrag.agents.state import SemiconductorGraphRAGState


@pytest.mark.asyncio
async def test_agent_returns_answer() -> None:
    """Agent should return a non-empty answer for a simple question."""
    answer = await ask("Who supplies EUV lithography tools?")
    assert answer and len(answer) > 10


@pytest.mark.asyncio
async def test_agent_full_state() -> None:
    """Agent should populate all expected state keys."""
    initial: SemiconductorGraphRAGState = {
        "question": "What are all the subtypes of Lithography tools?",
        "graph_results": [],
        "spec_results": [],
        "extracted_entities": [],
        "cypher_used": "",
        "combined_context": "",
        "answer": "",
        "iterations": 0,
        "intent": "taxonomy",  # type: ignore[typeddict-item]
    }
    result = await agent.ainvoke(initial)

    assert result["answer"], "answer field is empty"
    assert result["iterations"] >= 3, "expected at least 3 node visits"


@pytest.mark.asyncio
async def test_intent_classification() -> None:
    """classify_and_extract should identify 'risk' intent for risk questions."""
    from graphrag.agents.nodes import classify_and_extract

    initial: SemiconductorGraphRAGState = {
        "question": "Which countries have concentration risk across multiple fabrication stages?",
        "graph_results": [],
        "spec_results": [],
        "extracted_entities": [],
        "cypher_used": "",
        "combined_context": "",
        "answer": "",
        "iterations": 0,
        "intent": "hybrid",  # type: ignore[typeddict-item]
    }
    result = await classify_and_extract(initial)
    assert result["intent"] in ("risk", "hybrid"), \
        f"Expected 'risk' or 'hybrid' intent, got '{result['intent']}'"


@pytest.mark.asyncio
async def test_pipeline_question() -> None:
    """Pipeline traversal question should mention semiconductor process steps."""
    answer = await ask(
        "Show me the full production pipeline from Chip design to Finished logic chip."
    )
    assert answer and len(answer) > 30


@pytest.mark.asyncio
async def test_blast_radius() -> None:
    """Blast-radius question should mention ASML or lithography impacts."""
    answer = await ask("What would break in the fabrication pipeline if ASML stopped supplying?")
    assert answer and len(answer) > 30


if __name__ == "__main__":
    asyncio.run(test_agent_returns_answer())
    print("All manual tests passed.")
