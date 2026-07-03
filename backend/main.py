import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from config import CORS_ORIGINS
from database import init_db
from core.entity_resolution import entity_registry
from entity_interface.live_entity import LiveEntity
from routers import dashboard, entities, axiom, zola, chat, stream, intel

DEMO_API_KEY = os.getenv("DEMO_API_KEY", "sera-demo-2026")

# Gödel Loop auto-scheduling constants
_godel_auto_step_counter = 0
GODEL_AUTO_TRIGGER_EVERY = 50  # Run one Gödel generation every 50 optimize calls


async def auto_godel_loop():
    """Background task: runs one Gödel generation every 5 minutes when a loop is active."""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        try:
            from routers.zola import _godel_loop, _godel_results, entity_ai
            if isinstance(entity_ai, LiveEntity) and _godel_loop is not None:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, _godel_loop.step_generation)
                _godel_results.append(result)
                print(
                    f"[AUTO-GODEL] Generation {result.get('generation')} complete. "
                    f"Fitness: {result.get('best_fitness', 0.0):.4f}"
                )
        except Exception as e:
            print(f"[AUTO-GODEL] Error: {e}")


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/ws"):
            return await call_next(request)
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key != DEMO_API_KEY:
            return JSONResponse({"detail": "Unauthorized. Provide X-API-Key header."}, status_code=401)
        return await call_next(request)

app = FastAPI(
    title="SERA Intelligence Platform",
    description="Real-time behavioral intelligence API",
    version="1.0.0"
)

app.add_middleware(APIKeyMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(entities.router, prefix="/api/entities", tags=["entities"])
app.include_router(axiom.router)
app.include_router(zola.router)
app.include_router(chat.router)
app.include_router(stream.router)
app.include_router(intel.router)

@app.on_event("startup")
async def startup():
    await init_db()
    await entity_registry._bootstrap_async()
    asyncio.create_task(auto_godel_loop())

@app.get("/")
async def root():
    return {"message": "SERA Intelligence Platform API", "status": "online", "version": "1.0.0"}