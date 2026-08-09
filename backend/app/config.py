# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

# .env lives at the project root (two levels above this file: app/ → backend/ → project root)
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    # ── LLM (OpenAI) ──────────────────────────────────────────
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"

    # ── Web Search (Tavily) ───────────────────────────────────
    tavily_api_key: str = ""        # optional — web search disabled if empty

    # ── Orchestrator ─────────────────────────────────────────
    agent_max_rounds: int = 4
    web_search_score_threshold: float = 0.30
    session_memory_db: str = "./data/memory.db"

    # ── Embeddings & Reranking ────────────────────────────────
    embedding_model: str
    reranker_model: str
    nli_model: str

    # ── Vector Store ─────────────────────────────────────────
    chroma_persist_dir: str

    # ── Chunking ─────────────────────────────────────────────
    default_chunking: str
    chunk_size: int
    chunk_overlap: int
    semantic_similarity_threshold: float

    # ── Retrieval ────────────────────────────────────────────
    retrieval_top_k: int
    rerank_top_n: int
    rrf_k: int

    # ── Hallucination Detection ──────────────────────────────
    hallucination_threshold: float

    # ── Evaluation / CI ──────────────────────────────────────
    eval_regression_threshold: float
    eval_results_dir: str
    golden_qa_path: str
    baseline_path: str

    # ── API ──────────────────────────────────────────────────
    api_host: str
    api_port: int
    cors_origins: str

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
