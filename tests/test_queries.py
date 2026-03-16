"""
test_queries.py — Benchmark questions for the semiconductor GraphRAG agent.

Run with:  python -m pytest tests/test_queries.py -v -s
or directly:  python tests/test_queries.py

These tests require a running Neo4j instance with ingested data.
"""
import asyncio
import sys
import os

# Force UTF-8 output on Windows so LLM responses with → etc. don't crash
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphrag.agents.graph import ask

BENCHMARK_QUESTIONS = [
    # ── Graph-primary: pipeline traversal (GOES_INTO) ──────────────────────
    (
        "pipeline",
        "What inputs are needed to reach Photolithography in the fabrication pipeline?",
    ),
    (
        "pipeline",
        "Show me the full production pipeline from Chip design to Finished logic chip.",
    ),
    (
        "pipeline",
        "Which process steps directly feed into Chemical mechanical planarization?",
    ),
    # ── Graph-primary: provider / market share (PROVIDES) ──────────────────
    (
        "provider",
        "Who supplies EUV lithography tools and what is their market share?",
    ),
    (
        "provider",
        "Which inputs have a single organization controlling 100% of the market?",
    ),
    (
        "provider",
        "Which inputs are sourced entirely from one country?",
    ),
    # ── Graph-primary: taxonomy (IS_TYPE_OF) ───────────────────────────────
    (
        "taxonomy",
        "What are all the subtypes of Lithography tools?",
    ),
    (
        "taxonomy",
        "List every tool that is a type of Chemical vapor deposition tools.",
    ),
    # ── Vector-primary: semantic on Input.description ──────────────────────
    (
        "semantic",
        "Find inputs related to photomask defect inspection.",
    ),
    (
        "semantic",
        "Which materials are used in wafer surface preparation?",
    ),
    # ── Hybrid ─────────────────────────────────────────────────────────────
    (
        "hybrid",
        "Which single-sourced inputs have the highest geopolitical risk?",
    ),
    (
        "hybrid",
        "What would break in the fabrication pipeline if ASML stopped supplying?",
    ),
    # ── Multi-hop risk ──────────────────────────────────────────────────────
    (
        "risk",
        "Which countries have concentration risk across multiple fabrication stages?",
    ),
    (
        "risk",
        "Show all inputs where a single country controls more than 80% market share.",
    ),
]


async def run_benchmarks(verbose: bool = True) -> None:
    """Run all benchmark questions and print answers."""
    print("\n" + "=" * 70)
    print("SEMICONDUCTOR GRAPHRAG — BENCHMARK EVALUATION")
    print("=" * 70)

    for i, (category, question) in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n[{i:02d}] [{category.upper()}] {question}")
        print("-" * 60)
        try:
            answer = await ask(question)
            print(answer)
        except Exception as exc:
            print(f"ERROR: {exc}")
        print()


async def _run_single(question: str) -> str:
    """Run a single question inside an existing event loop."""
    return await ask(question)


# pytest async tests (use pytest-asyncio)
import pytest

@pytest.mark.asyncio
async def test_pipeline_traversal() -> None:
    """Verify that pipeline questions return non-empty answers."""
    answer = await ask("What inputs are needed to reach Photolithography in the fabrication pipeline?")
    assert answer and len(answer) > 20


@pytest.mark.asyncio
async def test_euv_monopoly() -> None:
    """Verify ASML EUV monopoly is detected."""
    answer = await ask("Who supplies EUV lithography tools and what is their market share?")
    assert "ASML" in answer or "asml" in answer.lower(), "ASML not found in answer"


@pytest.mark.asyncio
async def test_taxonomy() -> None:
    """Verify IS_TYPE_OF traversal for Lithography subtypes."""
    answer = await ask("What are all the subtypes of Lithography tools?")
    assert any(term in answer for term in ["EUV", "ArF", "KrF"]), \
        "Expected lithography subtypes (EUV/ArF/KrF) not found"


@pytest.mark.asyncio
async def test_single_country() -> None:
    """Verify single-country supply risk detection."""
    answer = await ask("Which inputs are sourced entirely from one country?")
    assert answer and len(answer) > 20


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
