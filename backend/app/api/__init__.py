"""
app.api — FastAPI route handlers

Routers (included in app.main):
    routes_ingest  → POST /ingest
    routes_query   → POST /query
    routes_eval    → POST /eval/run, GET /eval/results, GET /eval/ablation/{experiment}
"""
from .routes_ingest import router as ingest_router
from .routes_query import router as query_router
from .routes_eval import router as eval_router

__all__ = ["ingest_router", "query_router", "eval_router"]
