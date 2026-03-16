"""
config.py — Load environment variables from .env.
Always import settings from here; never hardcode credentials elsewhere.
"""
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI: str      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# LLM model identifiers
GEMINI_LLM_MODEL: str       = "gemini-2.5-flash"
GEMINI_EMBED_MODEL: str     = "models/gemini-embedding-001"
EMBED_DIMENSIONS: int       = 768
