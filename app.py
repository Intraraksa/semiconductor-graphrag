"""
app.py — Streamlit demo for the Semiconductor Supply Chain GraphRAG system.

Demonstrates GraphRAG intelligence powered by Neo4j + LangGraph + Google Gemini.
"""

import asyncio
import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Semiconductor Supply Chain Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0e1117; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Intent badge */
.intent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-pipeline  { background:#1a3a5c; color:#5ba4ff; border:1px solid #2d6ca8; }
.badge-provider  { background:#1a3d2b; color:#4cd97b; border:1px solid #2d7a4f; }
.badge-taxonomy  { background:#3d2b1a; color:#f0a33a; border:1px solid #8a5a1a; }
.badge-risk      { background:#3d1a1a; color:#f05252; border:1px solid #8a2020; }
.badge-semantic  { background:#2b1a3d; color:#c77dff; border:1px solid #6a1a8a; }
.badge-hybrid    { background:#1a3d3d; color:#4ce8e8; border:1px solid #1a8a8a; }

/* Agent step pill */
.step-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
    background: #1f2d3d;
    color: #8ab4f8;
    border: 1px solid #2d4a6e;
}
.step-active {
    background: #1a3a5c;
    color: #5ba4ff;
    border-color: #5ba4ff;
}

/* Chat bubbles */
.chat-user {
    background: #1a2744;
    border: 1px solid #2d4070;
    border-radius: 12px 12px 2px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 0.5rem 15%;
    color: #e8eaf6;
}
.chat-assistant {
    background: #1a2b1a;
    border: 1px solid #2d502d;
    border-radius: 12px 12px 12px 2px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 15% 0.5rem 0;
    color: #e8f5e9;
}

/* Cypher code block */
.cypher-block {
    background: #12161f;
    border: 1px solid #2d3561;
    border-left: 3px solid #5ba4ff;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-family: 'Fira Code', 'Courier New', monospace;
    font-size: 0.82rem;
    color: #a8d8f0;
    overflow-x: auto;
    white-space: pre-wrap;
}

/* Sidebar */
.sidebar-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7280;
    margin: 1rem 0 0.4rem 0;
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
.dot-gray  { background: #6b7280; }
</style>
""", unsafe_allow_html=True)


# ── Benchmark questions by category ──────────────────────────────────────────
EXAMPLE_QUESTIONS = {
    "🔗 Pipeline Traversal": [
        "Show me the full production pipeline from Chip design to Finished logic chip.",
        "What inputs are needed to reach Photolithography in the fabrication pipeline?",
        "Which process steps directly feed into Chemical mechanical planarization?",
    ],
    "🏭 Provider & Market Share": [
        "Who supplies EUV lithography tools and what is their market share?",
        "Which inputs have a single organization controlling 100% of the market?",
        "Which inputs are sourced entirely from one country?",
    ],
    "🌳 Tool Taxonomy": [
        "What are all the subtypes of Lithography tools?",
        "List every tool that is a type of Chemical vapor deposition tools.",
    ],
    "⚠️ Geopolitical Risk": [
        "Which single-sourced inputs have the highest geopolitical risk?",
        "What would break in the fabrication pipeline if ASML stopped supplying?",
        "Which countries have concentration risk across multiple fabrication stages?",
        "Show all inputs where a single country controls more than 80% market share.",
    ],
    "🔍 Semantic Search": [
        "Find inputs related to photomask defect inspection.",
        "Which materials are used in wafer surface preparation?",
    ],
}

INTENT_COLORS = {
    "pipeline": "badge-pipeline",
    "provider": "badge-provider",
    "taxonomy": "badge-taxonomy",
    "risk":     "badge-risk",
    "semantic": "badge-semantic",
    "hybrid":   "badge-hybrid",
}

INTENT_LABELS = {
    "pipeline": "Pipeline Traversal",
    "provider": "Provider / Market Share",
    "taxonomy": "Tool Taxonomy",
    "risk":     "Geopolitical Risk",
    "semantic": "Semantic Search",
    "hybrid":   "Hybrid",
}

ROUTE_MAP = {
    "pipeline": "graph_traversal",
    "provider": "graph_traversal",
    "taxonomy": "graph_traversal",
    "hybrid":   "graph_traversal",
    "risk":     "risk_assessment",
    "semantic": "vector_search",
}


# ── Async helpers ─────────────────────────────────────────────────────────────
def _run_async(coro):
    """Run an async coroutine from a synchronous Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # In some environments (e.g. Jupyter), use thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return loop.run_until_complete(coro)


@st.cache_resource(show_spinner=False)
def load_agent():
    """Load and cache the LangGraph agent (runs once per session)."""
    from graphrag.agents.graph import agent as _agent
    return _agent


@st.cache_resource(show_spinner=False)
def check_neo4j() -> tuple[bool, str]:
    """Check Neo4j connectivity and return (ok, message)."""
    try:
        from graphrag.graph_db import get_driver, run_query
        driver = get_driver()
        rows = run_query(driver, "RETURN 1 AS ok")
        driver.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)[:60]


@st.cache_resource(show_spinner=False)
def get_db_stats() -> dict:
    """Fetch high-level stats from Neo4j."""
    try:
        from graphrag.graph_db import get_driver, run_query
        driver = get_driver()
        inputs    = run_query(driver, "MATCH (n:Input)    RETURN count(n) AS c")[0]["c"]
        providers = run_query(driver, "MATCH (n:Provider) RETURN count(n) AS c")[0]["c"]
        edges     = run_query(driver, "MATCH ()-[r:PROVIDES]->() RETURN count(r) AS c")[0]["c"]
        chunks    = run_query(driver, "MATCH (n:SpecChunk) RETURN count(n) AS c")[0]["c"]
        driver.close()
        return {"inputs": inputs, "providers": providers, "edges": edges, "chunks": chunks}
    except Exception:
        return {"inputs": "—", "providers": "—", "edges": "—", "chunks": "—"}


async def _invoke_agent(question: str) -> dict:
    """Invoke the agent and return the full state dict."""
    agent = load_agent()
    result = await agent.ainvoke({
        "question":          question,
        "graph_results":     [],
        "spec_results":      [],
        "extracted_entities": [],
        "cypher_used":       "",
        "combined_context":  "",
        "answer":            "",
        "iterations":        0,
    })
    return result


def ask_agent(question: str) -> dict:
    """Synchronous wrapper for the async agent call."""
    return _run_async(_invoke_agent(question))


# ── Render helpers ────────────────────────────────────────────────────────────
def render_intent_badge(intent: str) -> str:
    cls   = INTENT_COLORS.get(intent, "badge-hybrid")
    label = INTENT_LABELS.get(intent, intent.title())
    return f'<span class="intent-badge {cls}">{label}</span>'


def render_agent_trace(state: dict):
    """Show which nodes the agent traversed."""
    intent    = state.get("intent", "hybrid")
    retriever = ROUTE_MAP.get(intent, "graph_traversal")
    nodes_run = ["classify_and_extract", retriever, "fuse_and_generate"]

    all_nodes = [
        "classify_and_extract",
        "graph_traversal",
        "risk_assessment",
        "vector_search",
        "fuse_and_generate",
    ]
    pills = []
    for node in all_nodes:
        active = node in nodes_run
        cls = "step-pill step-active" if active else "step-pill"
        label = node.replace("_", " ").title()
        arrow = " → " if node != all_nodes[-1] else ""
        pills.append(f'<span class="{cls}">{label}</span>{arrow}')

    st.markdown("**Agent Pipeline:**  " + "".join(pills), unsafe_allow_html=True)


def render_debug_panel(state: dict):
    """Render expandable debug details for a completed query."""
    with st.expander("🔍 Agent Debug Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Intent**")
            st.markdown(render_intent_badge(state.get("intent", "—")), unsafe_allow_html=True)

            entities = state.get("extracted_entities", [])
            st.markdown("**Extracted Entities**")
            if entities:
                for e in entities:
                    st.markdown(f"- `{e}`")
            else:
                st.caption("None extracted")

        with col2:
            st.markdown(f"**Graph Results:** {len(state.get('graph_results', []))} rows")
            st.markdown(f"**Semantic Chunks:** {len(state.get('spec_results', []))} hits")
            st.markdown(f"**Agent Iterations:** {state.get('iterations', '—')}")

        cypher = state.get("cypher_used", "").strip()
        if cypher:
            st.markdown("**Generated Cypher Query**")
            st.markdown(f'<div class="cypher-block">{cypher}</div>', unsafe_allow_html=True)

        graph_results = state.get("graph_results", [])
        if graph_results:
            st.markdown("**Raw Graph Results** (first 10)")
            import pandas as pd
            try:
                flat = [r for r in graph_results if isinstance(r, dict)]
                if flat:
                    st.dataframe(pd.DataFrame(flat[:10]), use_container_width=True, hide_index=True)
                else:
                    st.json(graph_results[:5])
            except Exception:
                st.json(graph_results[:5])

        spec_results = state.get("spec_results", [])
        if spec_results:
            st.markdown("**Semantic Search Hits**")
            for chunk in spec_results[:3]:
                st.code(chunk[:300] + ("…" if len(chunk) > 300 else ""), language=None)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 GraphRAG Demo")
        st.caption("Semiconductor Supply Chain Intelligence")
        st.divider()

        # System status
        st.markdown('<p class="sidebar-header">System Status</p>', unsafe_allow_html=True)
        neo4j_ok, neo4j_msg = check_neo4j()
        dot_cls = "dot-green" if neo4j_ok else "dot-red"
        st.markdown(
            f'<span class="status-dot {dot_cls}"></span> **Neo4j** — {neo4j_msg}',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="status-dot dot-green"></span> **Gemini** — gemini-2.5-flash',
            unsafe_allow_html=True,
        )
        st.divider()

        # Dataset stats
        st.markdown('<p class="sidebar-header">Dataset (ETO ChipExplorer)</p>', unsafe_allow_html=True)
        stats = get_db_stats()
        col_a, col_b = st.columns(2)
        col_a.metric("Inputs",    stats["inputs"])
        col_b.metric("Providers", stats["providers"])
        col_a.metric("Edges",     stats["edges"])
        col_b.metric("SpecChunks", stats["chunks"])
        st.divider()

        # Example questions
        st.markdown('<p class="sidebar-header">Example Questions</p>', unsafe_allow_html=True)
        for category, questions in EXAMPLE_QUESTIONS.items():
            with st.expander(category, expanded=False):
                for q in questions:
                    if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
                        st.session_state.pending_question = q

        st.divider()
        st.caption("Powered by LangGraph · Neo4j · Google Gemini")


# ── Chat tab ──────────────────────────────────────────────────────────────────
def render_chat_tab():
    st.markdown("### Ask the Supply Chain Agent")
    st.caption(
        "Ask any question about semiconductor supply chain dependencies, "
        "monopolies, pipelines, or geopolitical risks."
    )

    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []   # list of {role, content, state}
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    # Display history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">💬 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            state = msg.get("state", {})
            intent = state.get("intent", "")
            badge = render_intent_badge(intent) if intent else ""

            st.markdown(
                f'<div class="chat-assistant">'
                f'{badge} <br><br>{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if state:
                render_agent_trace(state)
                render_debug_panel(state)

        st.markdown("")  # spacer

    # Input area
    pending = st.session_state.pop("pending_question", "") if "pending_question" in st.session_state else ""
    user_input = st.chat_input(
        "Ask about supply chain dependencies, monopolies, pipelines…",
    ) or pending

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Run agent
        with st.spinner("🤖 Analyzing supply chain graph…"):
            try:
                state = ask_agent(user_input)
                answer = state.get("answer", "No answer returned.")
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
                state = {}

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "state": state,
        })
        st.rerun()

    # Clear button
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ── Explorer tab ──────────────────────────────────────────────────────────────
def render_explorer_tab():
    import pandas as pd

    st.markdown("### Supply Chain Explorer")
    st.caption("Pre-computed insights from the ETO ChipExplorer dataset.")

    try:
        from graphrag.graph_db import get_driver, run_query
        driver = get_driver()

        view = st.selectbox(
            "Choose a view",
            [
                "🏭 Single-Org Monopolies (100% share)",
                "🌍 Country Control (≥ 95% share)",
                "📊 Stage Concentration by Country",
                "🔗 Pipeline Edges (GOES_INTO)",
                "🌳 Taxonomy (IS_TYPE_OF)",
            ],
        )

        if view.startswith("🏭"):
            rows = run_query(driver, """
                MATCH (p:Provider {provider_type: 'organization'})-[r:PROVIDES]->(i:Input)
                WHERE r.share_provided = 100
                RETURN i.input_name  AS Input,
                       p.provider_name AS Sole_Provider,
                       p.country        AS Country,
                       r.year           AS Year
                ORDER BY i.input_name
            """)
            st.markdown(f"**{len(rows)} inputs** with a single organization controlling 100% supply")
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Mini bar chart — providers sorted by count
                provider_counts = df["Sole_Provider"].value_counts().reset_index()
                provider_counts.columns = ["Provider", "Monopolies"]
                st.markdown("**Monopoly count by provider**")
                st.bar_chart(provider_counts.set_index("Provider")["Monopolies"].head(15))

        elif view.startswith("🌍"):
            rows = run_query(driver, """
                MATCH (p:Provider {provider_type: 'country'})-[r:PROVIDES]->(i:Input)
                WHERE r.share_provided >= 95
                RETURN i.input_name   AS Input,
                       p.provider_name AS Country,
                       r.share_provided AS Share_Pct,
                       r.year           AS Year
                ORDER BY r.share_provided DESC
            """)
            st.markdown(f"**{len(rows)} inputs** where a single country holds ≥ 95% supply")
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                country_counts = df["Country"].value_counts().reset_index()
                country_counts.columns = ["Country", "Controlled_Inputs"]
                st.markdown("**Inputs under country control (≥ 95%)**")
                st.bar_chart(country_counts.set_index("Country")["Controlled_Inputs"])

        elif view.startswith("📊"):
            rows = run_query(driver, """
                MATCH (p:Provider {provider_type: 'country'})-[r:PROVIDES]->(i:Input)-[:IN_STAGE]->(s:Stage)
                WHERE r.share_provided >= 50
                WITH p.provider_name AS Country, s.stage_name AS Stage,
                     COUNT(DISTINCT i) AS Concentrated_Inputs,
                     AVG(r.share_provided) AS Avg_Share
                WHERE Concentrated_Inputs >= 2
                RETURN Country, Stage, Concentrated_Inputs,
                       round(Avg_Share, 1) AS Avg_Share_Pct
                ORDER BY Concentrated_Inputs DESC
            """)
            st.markdown("**Countries with ≥ 50% share in 2+ inputs per stage**")
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

        elif view.startswith("🔗"):
            rows = run_query(driver, """
                MATCH (a:Input)-[:GOES_INTO]->(b:Input)
                RETURN a.input_name AS From_Input, b.input_name AS To_Input
                ORDER BY a.input_name
                LIMIT 100
            """)
            st.markdown(f"**{len(rows)} GOES_INTO edges** (production pipeline flow)")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        elif view.startswith("🌳"):
            rows = run_query(driver, """
                MATCH (a:Input)-[:IS_TYPE_OF]->(b:Input)
                RETURN a.input_name AS Subtype, b.input_name AS Parent_Type
                ORDER BY b.input_name, a.input_name
                LIMIT 100
            """)
            st.markdown(f"**{len(rows)} IS_TYPE_OF edges** (tool taxonomy)")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        driver.close()

    except Exception as e:
        st.error(f"Could not connect to Neo4j: {e}")
        st.info("Make sure Neo4j is running: `docker compose up -d`")


# ── About tab ─────────────────────────────────────────────────────────────────
def render_about_tab():
    st.markdown("### About This System")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**GraphRAG** combines structured graph traversal with semantic vector search to answer
complex supply chain questions that traditional RAG cannot handle.

#### Why GraphRAG?

| Capability | Vector RAG | GraphRAG |
|---|---|---|
| Semantic similarity | ✅ | ✅ |
| Multi-hop traversal | ❌ | ✅ |
| Market share queries | ❌ | ✅ |
| Monopoly detection | ❌ | ✅ |
| Pipeline flow | ❌ | ✅ |
| Taxonomy lookup | ❌ | ✅ |

#### Agent Pipeline

```
User Question
      │
      ▼
  classify_and_extract    ← Intent: pipeline / provider / taxonomy
      │                             risk / semantic / hybrid
      ├──[pipeline|provider|taxonomy|hybrid]──► graph_traversal
      │                                         (Text-to-Cypher)
      ├──[risk]───────────────────────────────► risk_assessment
      │                                         (Hardcoded risk Cypher)
      └──[semantic]───────────────────────────► vector_search
                                                (SpecChunk embeddings)
                                    │
                                    ▼
                            fuse_and_generate
                            (LLM answer synthesis)
                                    │
                                    ▼
                               Final Answer
```

#### Dataset
[ETO ChipExplorer](https://eto.tech/dataset-docs/chipexplorer/) — Georgetown CSET
- **126 Inputs** — tools, materials, processes, design resources
- **393+ Providers** — companies and country aggregates
- **1,276+ PROVIDES edges** — with `share_provided` (0–100%)
- **139 sequence edges** — GOES_INTO (pipeline) + IS_TYPE_OF (taxonomy)
- **3 Stages** — S1 Design · S2 Fabrication · S3 ATP
        """)

    with col2:
        st.markdown("#### Tech Stack")
        st.markdown("""
| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | gemini-embedding-001 |
| Graph DB | Neo4j 5.x |
| Orchestration | LangGraph |
| Framework | LangChain |
| Vector Index | Neo4j built-in |
| UI | Streamlit |
        """)

        st.markdown("#### Example Capabilities")
        examples = [
            ("ASML Monopoly", "EUV lithography tools — ASML (NLD) 100%"),
            ("NLD Control", "Netherlands controls entire EUV supply chain"),
            ("Pipeline", "Chip design → Mask making → Photolithography → …"),
            ("Taxonomy", "EUV ⊂ ArF immersion ⊂ ArF dry ⊂ KrF → Lithography"),
            ("Blast Radius", "If ASML stops → entire fab pipeline at risk"),
        ]
        for title, desc in examples:
            st.markdown(f"**{title}:** {desc}")


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Header
    st.markdown("""
<div style="padding: 1rem 0 0.5rem 0;">
  <h1 style="margin:0; font-size:2rem; font-weight:700; color:#e8eaf6;">
    🔬 Semiconductor Supply Chain Intelligence
  </h1>
  <p style="color:#9ca3af; margin-top:0.3rem; font-size:1rem;">
    GraphRAG · Neo4j · LangGraph · Google Gemini
  </p>
</div>
""", unsafe_allow_html=True)

    # Tabs
    tab_chat, tab_explorer, tab_about = st.tabs([
        "💬 Ask the Agent",
        "📊 Supply Chain Explorer",
        "ℹ️ About",
    ])

    with tab_chat:
        render_chat_tab()

    with tab_explorer:
        render_explorer_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
