from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    print("Ask My Docs API starting up…")

    # Initialise SQLite session memory
    from app.memory import init_memory
    init_memory(settings.session_memory_db)

    # Kick off model preloading in the background — doesn't block /health
    async def preload_models():
        loop = asyncio.get_running_loop()
        def _load():
            from app.api.routes_query import _hybrid_retriever, _reranker, _detector
            print("Preloading ML models in background…")
            _hybrid_retriever._dense._get_model()
            _reranker._get_model()
            _detector._get_model()
            print("ML models ready!")
        await loop.run_in_executor(None, _load)

    asyncio.create_task(preload_models())

    yield
    # Shutdown cleanup
    print("Ask My Docs API shutting down…")


app = FastAPI(
    title="Ask My Docs",
    description="Citation-grounded RAG API with hallucination detection and RAGAS evaluation.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
from app.api import ingest_router, query_router, eval_router

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(eval_router, prefix="/eval", tags=["Evaluation"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": app.version}
