import numpy as np
# Hot-patch for ChromaDB compatibility with NumPy 2.0+
np.float_ = np.float64

import os
import json
import asyncio
import logging
from pydantic import BaseModel, Field, model_validator
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from config import CORS_ORIGINS, ENTITY_MODE
from database import init_db

# Configure structured-ready standard logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger("sera.backend")
from core.entity_resolution import entity_registry
from entity_interface.live_entity import LiveEntity
from entity_interface.signal_synthesizer import SignalSynthesizer
from routers import dashboard, entities, axiom, zola, chat, stream, intel, insights, health, graph, semantic, dark_intel, citation, healthcare, executive
from routers import security as security_router

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Task 1: Parse multi-key map from environment variable API_KEYS
# Expected format: JSON dict mapping key to user/client ID, e.g. {"key1": "user1"}
# If not valid JSON, falls back to comma-separated list of keys
API_KEYS_ENV = os.getenv("API_KEYS", "")
API_KEYS = {}

if API_KEYS_ENV.strip():
    try:
        parsed = json.loads(API_KEYS_ENV)
        if isinstance(parsed, dict):
            API_KEYS = parsed
        elif isinstance(parsed, list):
            API_KEYS = {k: f"client_{i}" for i, k in enumerate(parsed)}
    except json.JSONDecodeError:
        for val in API_KEYS_ENV.split(","):
            val = val.strip()
            if val:
                API_KEYS[val] = f"client_{val[-4:] if len(val) >= 4 else val}"

# DEMO_API_KEY is auto-injected ONLY in mock mode, where the publicly-known
# default value is an acceptable credential for a demo/dev environment.
# In live mode (ENTITY_MODE != 'mock'), the key is NOT injected automatically —
# operators who want it in live mode must set it explicitly via API_KEYS.
_demo_key_env = os.getenv("DEMO_API_KEY")          # None when not set
_demo_key_default = "sera-demo-2026"                # well-known fallback value

if ENTITY_MODE == "mock":
    # Demo/dev mode: use the env var if given, otherwise fall back to the
    # well-known default so the UI works out of the box.
    DEMO_API_KEY = _demo_key_env or _demo_key_default
    if DEMO_API_KEY not in API_KEYS:
        API_KEYS[DEMO_API_KEY] = "default_demo"
else:
    # Live (or any non-mock) mode: only inject a demo key when the operator
    # explicitly sets DEMO_API_KEY in the environment.  Do not use the
    # hard-coded default — a publicly-known key must not be an implicit credential.
    DEMO_API_KEY = _demo_key_env  # may be None
    if DEMO_API_KEY and DEMO_API_KEY not in API_KEYS:
        API_KEYS[DEMO_API_KEY] = "default_demo"

# Task 2: Configure rate limiting key resolver and limiter
def get_rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if api_key and api_key in API_KEYS:
        return API_KEYS[api_key]
    return request.client.host if request.client else "anonymous"

limiter = Limiter(key_func=get_rate_limit_key, default_limits=["60/minute"])

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
                logger.info(
                    f"[AUTO-GODEL] Generation {result.get('generation')} complete. "
                    f"Fitness: {result.get('best_fitness', 0.0):.4f}"
                )
        except Exception as e:
            logger.error(f"[AUTO-GODEL] Error during Gödel generation step: {e}", exc_info=True)


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Always allow CORS preflight through unchanged.
        if request.method == "OPTIONS":
            return await call_next(request)

        # WebSocket upgrade requests cannot carry custom headers reliably in all
        # clients, so authenticate via the `api_key` query parameter instead.
        # Returning a non-101 response here closes the connection cleanly at the
        # HTTP handshake stage before any WebSocket state is established.
        if request.headers.get("upgrade", "").lower() == "websocket":
            api_key = request.query_params.get("api_key")
            if not api_key or api_key not in API_KEYS:
                logger.warning(
                    f"[SECURITY] Unauthorized WebSocket upgrade attempt from IP={request.client.host if request.client else 'unknown'}"
                )
                return JSONResponse(
                    {"detail": "Unauthorized. Provide a valid api_key query parameter."},
                    status_code=403,
                )
            request.state.client_id = API_KEYS[api_key]
            return await call_next(request)

        # Standard HTTP routes: accept key from header or query param.
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key or api_key not in API_KEYS:
            logger.warning(
                f"[SECURITY] Unauthorized HTTP request: Path={request.url.path} Method={request.method} "
                f"IP={request.client.host if request.client else 'unknown'} UserAgent={request.headers.get('user-agent', 'unknown')}"
            )
            return JSONResponse({"detail": "Unauthorized. Provide a valid X-API-Key header."}, status_code=401)

        # Attribute request to the specific client
        request.state.client_id = API_KEYS[api_key]
        return await call_next(request)


app = FastAPI(
    title="SERA Intelligence Platform",
    description="Real-time behavioral intelligence API",
    version="1.0.0"
)

from starlette.middleware.gzip import GZipMiddleware

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Order: Last added runs first. Adding GZipMiddleware first makes it run on the response last.
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
)

app.include_router(dashboard.router)
app.include_router(entities.router, prefix="/api/entities", tags=["entities"])
app.include_router(axiom.router)
app.include_router(zola.router)
app.include_router(chat.router)
app.include_router(stream.router)
app.include_router(intel.router)
app.include_router(insights.router)
app.include_router(health.router)
app.include_router(graph.router)
app.include_router(semantic.router)
app.include_router(dark_intel.router)
app.include_router(citation.router)
app.include_router(healthcare.router)
app.include_router(executive.router)
app.include_router(security_router.router)

_sec_failure_count = 0

def run_gdelt_job():
    import asyncio
    from services.data_orchestrator import DataIngestionService
    async def _async_run():
        await DataIngestionService.run_gdelt_ingestion()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_run())
    finally:
        loop.close()

def run_ais_jobs_job():
    import asyncio
    from services.data_orchestrator import DataIngestionService
    async def _async_run():
        await DataIngestionService.run_ais_jobs_ingestion()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_run())
    finally:
        loop.close()

def run_executive_job():
    import asyncio
    from services.data_orchestrator import DataIngestionService
    async def _async_run():
        await DataIngestionService.run_executive_ingestion()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_run())
    finally:
        loop.close()

def run_heavy_sync_job():
    global _sec_failure_count
    import asyncio
    from services.data_orchestrator import DataIngestionService
    from database import async_session_maker
    from models.db_models import AlertModel
    from datetime import datetime

    async def _async_run():
        global _sec_failure_count
        try:
            res = await DataIngestionService.run_heavy_sync()
            sec_status = res.get("sec", "failed")
            if sec_status == "success":
                _sec_failure_count = 0
            else:
                _sec_failure_count += 1
                
            if _sec_failure_count >= 3:
                msg = "⚠️ CRITICAL: SEC ingestion failed 3 times. Check API key."
                logger.error(msg)
                async with async_session_maker() as session:
                    session.add(AlertModel(
                        entity_id="SYSTEM",
                        alert_type="INGESTION_FAILURE",
                        severity="critical",
                        description=msg,
                        entropy_value=1.5,
                        created_at=datetime.utcnow(),
                        resolved=False
                    ))
                    await session.commit()
        except Exception as exc:
            logger.error(f"[SCHEDULER] Heavy Ingestion job threw exception: {exc}", exc_info=True)
            _sec_failure_count += 1
            if _sec_failure_count >= 3:
                msg = f"⚠️ CRITICAL: SEC ingestion failed 3 times. Error: {exc}"
                logger.error(msg)
                async with async_session_maker() as session:
                    session.add(AlertModel(
                        entity_id="SYSTEM",
                        alert_type="INGESTION_FAILURE",
                        severity="critical",
                        description=msg,
                        entropy_value=1.5,
                        created_at=datetime.utcnow(),
                        resolved=False
                    ))
                    await session.commit()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_run())
    finally:
        loop.close()


@app.on_event("startup")
async def startup():
    # Warm up SentenceTransformer model to optimize APEX causal engine latency
    try:
        logger.info("[STARTUP] Warming up APEX SentenceTransformer model...")
        from entity_interface.apex_causal import get_encoder
        get_encoder()
        logger.info("[STARTUP] APEX SentenceTransformer model warmed up successfully.")
    except Exception as e:
        logger.warning(f"[STARTUP] Warning: Failed to warm up SentenceTransformer: {e}")

    await init_db()
    await entity_registry._bootstrap_async()
    asyncio.create_task(auto_godel_loop())
    try:
        from services.data_orchestrator import DataIngestionService
        asyncio.create_task(DataIngestionService.fetch_all_sources())
    except Exception as e:
        logger.error(f"[STARTUP] Failed to trigger DataIngestionService: {e}", exc_info=True)

    # Set up background scheduler if USE_REAL_DATA is enabled
    from config import (
        USE_REAL_DATA, GDELT_INTERVAL_MINUTES, AIS_INTERVAL_MINUTES,
        JOBS_INTERVAL_MINUTES, EXEC_INTERVAL_MINUTES, FULL_SYNC_HOUR, FULL_SYNC_MINUTE
    )
    if USE_REAL_DATA:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            
            scheduler = BackgroundScheduler()
            # Job 1: GDELT News every 15 minutes (or as configured)
            scheduler.add_job(
                run_gdelt_job,
                'interval',
                minutes=GDELT_INTERVAL_MINUTES,
                id='gdelt_sync',
                max_instances=1
            )
            # Job 2: AIS & Jobs every 60 minutes (or as configured)
            scheduler.add_job(
                run_ais_jobs_job,
                'interval',
                minutes=AIS_INTERVAL_MINUTES,
                id='ais_jobs_sync',
                max_instances=1
            )
            # Job 3: Executive Movements every 60 minutes (or as configured)
            scheduler.add_job(
                run_executive_job,
                'interval',
                minutes=EXEC_INTERVAL_MINUTES,
                id='exec_sync',
                max_instances=1
            )
            # Job 4: Daily Heavy Sync (SEC & Healthcare)
            scheduler.add_job(
                run_heavy_sync_job,
                'cron',
                hour=FULL_SYNC_HOUR,
                minute=FULL_SYNC_MINUTE,
                id='heavy_sync',
                max_instances=1
            )
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info(f"Scheduler started. Heavy sync scheduled daily at {FULL_SYNC_HOUR:02d}:{FULL_SYNC_MINUTE:02d} AM.")
            logger.info(f"Interval syncs: GDELT={GDELT_INTERVAL_MINUTES}m, AIS/Jobs={AIS_INTERVAL_MINUTES}m, Exec={EXEC_INTERVAL_MINUTES}m.")
        except Exception as exc:
            logger.error(f"[STARTUP] Failed to start BackgroundScheduler: {exc}", exc_info=True)


@app.get("/api/synthesize/{entity_id}")
async def synthesize_signals(entity_id: str):
    if not entity_registry.get_by_id(entity_id):
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found in registry.")
    
    synthesizer = SignalSynthesizer()
    result = await synthesizer.synthesize(entity_id)
    return result


class RelationshipCreate(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relationship_type: str = Field(..., description="Type of relationship, e.g. works_with, co_occurs_with, supplies_to")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0)


@app.post("/api/graph/relationship")
async def create_relationship(rel: RelationshipCreate):
    if not entity_registry.get_by_id(rel.source_entity_id):
        raise HTTPException(status_code=404, detail=f"Source entity {rel.source_entity_id} not found in registry.")
    if not entity_registry.get_by_id(rel.target_entity_id):
        raise HTTPException(status_code=404, detail=f"Target entity {rel.target_entity_id} not found in registry.")
    
    from database import async_session_maker
    from models.db_models import EntityRelationshipModel
    
    async with async_session_maker() as session:
        db_rel = EntityRelationshipModel(
            source_entity_id=rel.source_entity_id,
            target_entity_id=rel.target_entity_id,
            relationship_type=rel.relationship_type,
            confidence_score=rel.confidence_score
        )
        session.add(db_rel)
        await session.commit()
        await session.refresh(db_rel)
        
        return {
            "relationship_id": db_rel.id,
            "source_entity_id": db_rel.source_entity_id,
            "target_entity_id": db_rel.target_entity_id,
            "relationship_type": db_rel.relationship_type,
            "confidence_score": db_rel.confidence_score,
            "created_at": db_rel.created_at.isoformat()
        }


@app.get("/api/graph/entity/{entity_id}/connections")
async def get_entity_connections(entity_id: str, min_confidence: float = 0.0):
    from database import async_session_maker
    from models.db_models import EntityRelationshipModel
    from sqlalchemy import select, or_

    # ── Step 1: Resolve entity_id → ticker & entity metadata ─────────────
    entity = entity_registry.get_by_id(entity_id)
    ticker = None

    if entity_id.startswith("CO-"):
        # Real database company — resolve ticker from PostgreSQL
        from models.commerce import CompanyModel
        try:
            async with async_session_maker() as session:
                result = await session.execute(
                    select(CompanyModel).where(CompanyModel.id == entity_id)
                )
                company = result.scalars().first()
                if company:
                    ticker = company.ticker
                    if not entity:
                        entity = {"id": entity_id, "name": company.legal_name, "domain": "iot"}
        except Exception as e:
            logger.error(f"[ENTITY GRAPH] Could not resolve ticker for {entity_id}: {e}", exc_info=True)

    if not entity and not ticker:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found.")

    connections = []

    # ── Step 2: Query Neo4j for APEX causal morphisms (primary source) ───
    if ticker:
        try:
            from services.graph_sync import GraphSyncService
            from entity_interface.apex_causal import APEXCausalEngine

            driver = GraphSyncService.get_driver()
            if driver:
                category = APEXCausalEngine(max_k=3)
                category.hydrate_from_neo4j(driver)

                outgoing_keys = list(category.outgoing.get(ticker, []))
                incoming_keys = []
                for key, m in category.morphisms.items():
                    if m.target_id == ticker and key not in outgoing_keys:
                        incoming_keys.append(key)

                def _morphism_to_connection(m, direction):
                    connected_id = m.target_id if direction == "outgoing" else m.source_id
                    obj = category.objects.get(connected_id)
                    return {
                        "relationship_id": f"neo4j-{m.source_id}-{m.relation_type}-{m.target_id}",
                        "direction": direction,
                        "relationship_type": m.relation_type,
                        "confidence_score": round(m.weight, 4),
                        "connected_entity": {
                            "id": connected_id,
                            "name": obj.name if obj else connected_id,
                            "domain": obj.domain if obj else "unknown"
                        },
                        "source": "neo4j"
                    }

                for key in outgoing_keys:
                    m = category.morphisms.get(key)
                    if m and m.weight >= min_confidence:
                        connections.append(_morphism_to_connection(m, "outgoing"))

                for key in incoming_keys:
                    m = category.morphisms.get(key)
                    if m and m.weight >= min_confidence:
                        connections.append(_morphism_to_connection(m, "incoming"))

        except Exception as e:
            logger.error(f"[ENTITY GRAPH] Neo4j query failed for {ticker}: {e}", exc_info=True)

    # ── Step 3: Also query SQL adjacency list (legacy/mock data) ─────────
    try:
        async with async_session_maker() as session:
            stmt = select(EntityRelationshipModel).where(
                or_(
                    EntityRelationshipModel.source_entity_id == entity_id,
                    EntityRelationshipModel.target_entity_id == entity_id
                )
            ).where(
                EntityRelationshipModel.confidence_score >= min_confidence
            )

            result = await session.execute(stmt)
            relationships = result.scalars().all()

            for r in relationships:
                is_source = (r.source_entity_id == entity_id)
                connected_id = r.target_entity_id if is_source else r.source_entity_id
                direction = "outgoing" if is_source else "incoming"

                connected_ent = entity_registry.get_by_id(connected_id)
                connected_name = connected_ent["name"] if connected_ent else "Unknown"
                connected_domain = connected_ent["domain"] if connected_ent else "unknown"

                connections.append({
                    "relationship_id": r.id,
                    "direction": direction,
                    "relationship_type": r.relationship_type,
                    "confidence_score": r.confidence_score,
                    "connected_entity": {
                        "id": connected_id,
                        "name": connected_name,
                        "domain": connected_domain
                    },
                    "source": "sql"
                })
    except Exception as e:
        logger.error(f"[ENTITY GRAPH] SQL query failed: {e}", exc_info=True)

    return {
        "entity_id": entity_id,
        "ticker": ticker,
        "total_connections": len(connections),
        "min_confidence_filter": min_confidence,
        "connections": connections,
        "proof_of_concept_note": (
            "Connections include APEX causal morphisms from Neo4j (keyed by ticker) "
            "plus any manually-created SQL relationship edges."
        )
    }


class ClaimCreate(BaseModel):
    claimant_id: str = None
    content: str = None
    stake_amount: float = Field(..., ge=0.0, description="Simulated point value stake")

    @model_validator(mode="before")
    @classmethod
    def resolve_aliases(cls, data):
        if isinstance(data, dict):
            if "claimant_entity_id" in data and not data.get("claimant_id"):
                data["claimant_id"] = data["claimant_entity_id"]
            if "claim_text" in data and not data.get("content"):
                data["content"] = data["claim_text"]
        return data


class ChallengeCreate(BaseModel):
    challenger_id: str
    challenge_text: str = ""
    counter_stake_amount: float = Field(..., ge=0.0, description="Simulated counter stake")


class EvidenceCreate(BaseModel):
    evidence_type: str = "user"   # financial | graph | news | document | user
    source: str = ""
    content: str
    weight: float = Field(default=1.0, ge=0.1, le=1.5)


async def _resolve_entity_name(entity_id: str) -> str:
    """Resolve a claimant/challenger ID to a display name."""
    entity = entity_registry.get_by_id(entity_id)
    if entity:
        return entity.get("name", entity_id)
    if entity_id.startswith("CO-"):
        from database import async_session_maker
        from models.commerce import CompanyModel
        from sqlalchemy import select
        try:
            async with async_session_maker() as session:
                result = await session.execute(select(CompanyModel).where(CompanyModel.id == entity_id))
                c = result.scalars().first()
                if c:
                    return c.legal_name
        except Exception:
            pass
    return entity_id


async def _check_apex_verification(entity_id: str, claim_text: str) -> bool:
    """
    Check if APEX Neo4j graph contains evidence corroborating the claim.
    Heuristic: does the claimant company have outgoing morphisms in Neo4j?
    """
    try:
        from services.graph_sync import GraphSyncService
        from entity_interface.apex_causal import APEXCausalEngine
        from database import async_session_maker
        from models.commerce import CompanyModel
        from sqlalchemy import select

        ticker = None
        if entity_id.startswith("CO-"):
            async with async_session_maker() as session:
                result = await session.execute(select(CompanyModel).where(CompanyModel.id == entity_id))
                c = result.scalars().first()
                if c:
                    ticker = c.ticker
        else:
            ent = entity_registry.get_by_id(entity_id)
            if ent:
                ticker = ent.get("ticker", entity_id)

        if not ticker:
            return False

        driver = GraphSyncService.get_driver()
        if not driver:
            return False

        category = APEXCausalEngine(max_k=2)
        category.hydrate_from_neo4j(driver)

        # Verified if company has >= 3 outgoing morphisms in the causal graph
        outgoing = category.outgoing.get(ticker, [])
        return len(outgoing) >= 3

    except Exception as e:
        logger.warning(f"[ALETHEIA] APEX verification failed: {e}")
        return False


def _build_claim_response(claim_obj, challenges, evidence_rows, scoring: dict) -> dict:
    from datetime import datetime
    import hashlib as _hashlib
    
    # Calculate ALETHEIA specific fields
    s0 = min(0.3 + claim_obj.stake_amount / 100.0, 1.0)
    
    delta_t_seconds = (datetime.utcnow() - claim_obj.last_reaffirmed_at).total_seconds()
    delta_t_hours = max(delta_t_seconds / 3600.0, 0.0)
    decayed_credibility = max(s0 - 0.02 * delta_t_hours, 0.0)
    
    total_counter_stake = sum(c.counter_stake_amount for c in challenges)
    penalty = total_counter_stake / claim_obj.stake_amount if claim_obj.stake_amount > 0 else 0.0
    
    _prov_payload = f"{claim_obj.claimant_id}|{claim_obj.content}|{claim_obj.created_at.isoformat()}"
    provenance_fingerprint = "sha3:" + _hashlib.sha3_256(_prov_payload.encode()).hexdigest()
    
    adversarial_surface = []
    for c in challenges:
        _cp = f"{c.challenger_id}|challenge|{c.created_at.isoformat()}"
        adversarial_surface.append({
            "challenge_id": c.challenge_id,
            "challenger_id": c.challenger_id,
            "counter_stake_amount": c.counter_stake_amount,
            "stake_weight": round(
                c.counter_stake_amount / (total_counter_stake + 1e-9), 4
            ) if total_counter_stake > 0 else 0.0,
            "provenance": "sha3:" + _hashlib.sha3_256(_cp.encode()).hexdigest()[:32],
            "created_at": c.created_at.isoformat(),
        })
        
    _JURIS = {
        "financial": ["US-SEC", "EU-ESMA", "UK-FCA"], "stock": ["US-SEC", "UK-FCA"],
        "earnings": ["US-SEC"], "revenue": ["US-SEC", "EU-ESMA"],
        "health": ["US-FDA", "EU-EMA"], "drug": ["US-FDA", "EU-EMA"],
        "medical": ["US-FDA", "EU-EMA"], "clinical": ["US-FDA"],
        "data": ["EU-GDPR", "US-CCPA"], "privacy": ["EU-GDPR", "US-CCPA"],
        "election": ["US-FEC"], "vote": ["US-FEC"],
    }
    _content_l = (claim_obj.content or "").lower()
    jurisdictions = sorted({j for kw, js in _JURIS.items() if kw in _content_l for j in js})
    if not jurisdictions:
        jurisdictions = ["GLOBAL-UNSPECIFIED"]

    return {
        "claim_id": claim_obj.claim_id,
        "claimant_id": claim_obj.claimant_id,
        "claimant_name": claim_obj.claimant_name,
        "content": claim_obj.content,
        "stake_amount": claim_obj.stake_amount,
        "status": claim_obj.status,
        "apex_verified": claim_obj.apex_verified,
        "credibility_score": scoring["credibility_score"],
        "status_band": scoring["status_band"],
        "band_color": scoring["band_color"],
        "scoring_breakdown": scoring["breakdown"],
        "created_at": claim_obj.created_at.isoformat(),
        "last_reaffirmed_at": claim_obj.last_reaffirmed_at.isoformat(),
        
        # Layer 1 — Provenance
        "provenance_fingerprint": provenance_fingerprint,
        "provenance_immutable": True,
        # Layer 2 — Stake
        "initial_credibility_score": round(s0, 4),
        # Layer 4 — Temporal Decay
        "temporal_decay_applied": round(0.02 * delta_t_hours, 4),
        "decayed_credibility_score": round(decayed_credibility, 4),
        # Layer 3 — Adversarial Surface
        "total_counter_stake": total_counter_stake,
        "challenge_penalty_applied": round(penalty, 4),
        "adversarial_surface": adversarial_surface,
        # Layer 5 — Jurisdictional Bridge
        "jurisdictional_contexts": jurisdictions,
        # Composite
        "aletheia_layers": ["provenance", "stake", "adversarial_surface",
                             "temporal_decay", "jurisdictional_bridge"],
        
        "evidence": [
            {
                "evidence_id": e.evidence_id,
                "evidence_type": e.evidence_type,
                "source": e.source,
                "content": e.content,
                "weight": e.weight,
                "created_at": e.created_at.isoformat()
            } for e in evidence_rows
        ],
        "challenges": [
            {
                "challenge_id": c.challenge_id,
                "challenger_id": c.challenger_id,
                "challenger_name": c.challenger_name,
                "challenge_text": c.challenge_text,
                "counter_stake_amount": c.counter_stake_amount,
                "status": c.status,
                "resolution": c.resolution,
                "created_at": c.created_at.isoformat()
            } for c in challenges
        ],
        "proof_of_concept_note": (
            "ALETHEIA PROOF OF CONCEPT: Credibility = stake_signal * temporal_decay * "
            "(1 - challenge_penalty) * evidence_boost * apex_bonus. Uses simulated point-value stakes only."
        )
    }



@app.post("/api/claims")
@app.post("/api/claims/submit")
async def create_claim(claim: ClaimCreate):
    from database import async_session_maker
    from models.db_models import ClaimModel
    from services.aletheia_engine import compute_final_score

    claimant_name = await _resolve_entity_name(claim.claimant_id)
    apex_verified = await _check_apex_verification(claim.claimant_id, claim.content)

    from datetime import datetime as dt
    now = dt.utcnow()

    scoring = compute_final_score(
        stake_amount=claim.stake_amount,
        last_reaffirmed_at=now,
        total_counter_stake=0.0,
        evidence_rows=[],
        apex_verified=apex_verified
    )

    async with async_session_maker() as session:
        db_claim = ClaimModel(
            claimant_id=claim.claimant_id,
            claimant_name=claimant_name,
            content=claim.content,
            stake_amount=claim.stake_amount,
            status="active",
            credibility_score=scoring["credibility_score"],
            apex_verified=apex_verified
        )
        session.add(db_claim)
        await session.commit()
        await session.refresh(db_claim)

        return {
            "id": db_claim.claim_id,
            "claim_id": db_claim.claim_id,
            "claimant_id": db_claim.claimant_id,
            "claimant_name": db_claim.claimant_name,
            "content": db_claim.content,
            "stake_amount": db_claim.stake_amount,
            "apex_verified": db_claim.apex_verified,
            "initial_credibility_score": db_claim.credibility_score,
            "credibility": db_claim.credibility_score,
            "status_band": scoring["status_band"],
            "status": db_claim.status,
            "created_at": db_claim.created_at.isoformat()
        }


@app.get("/api/claims")
async def list_claims(limit: int = 20, offset: int = 0):
    """List recent claims with summary credibility scores."""
    from database import async_session_maker
    from models.db_models import ClaimModel
    from sqlalchemy import select

    async with async_session_maker() as session:
        stmt = select(ClaimModel).order_by(ClaimModel.created_at.desc()).offset(offset).limit(limit)
        result = await session.execute(stmt)
        claims = result.scalars().all()
        return {
            "total": len(claims),
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "claimant_name": c.claimant_name or c.claimant_id,
                    "content_preview": c.content[:120] + ("..." if len(c.content) > 120 else ""),
                    "stake_amount": c.stake_amount,
                    "credibility_score": c.credibility_score,
                    "status": c.status,
                    "apex_verified": c.apex_verified,
                    "created_at": c.created_at.isoformat()
                } for c in claims
            ]
        }


@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: str):
    from database import async_session_maker
    from models.db_models import ClaimModel, ClaimChallengeModel, EvidenceModel
    from sqlalchemy import select
    from services.aletheia_engine import compute_final_score

    async with async_session_maker() as session:
        stmt = select(ClaimModel).where(ClaimModel.claim_id == claim_id)
        result = await session.execute(stmt)
        claim_obj = result.scalars().first()
        if not claim_obj:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

        chal_result = await session.execute(
            select(ClaimChallengeModel).where(ClaimChallengeModel.target_claim_id == claim_id)
        )
        challenges = chal_result.scalars().all()

        evid_result = await session.execute(
            select(EvidenceModel).where(EvidenceModel.claim_id == claim_id)
        )
        evidence_rows = evid_result.scalars().all()

        total_counter_stake = sum(c.counter_stake_amount for c in challenges)
        evidence_dicts = [{"evidence_type": e.evidence_type, "weight": e.weight} for e in evidence_rows]

        scoring = compute_final_score(
            stake_amount=claim_obj.stake_amount,
            last_reaffirmed_at=claim_obj.last_reaffirmed_at,
            total_counter_stake=total_counter_stake,
            evidence_rows=evidence_dicts,
            apex_verified=claim_obj.apex_verified
        )

        # Persist updated score
        claim_obj.credibility_score = scoring["credibility_score"]
        claim_obj.status = (
            "verified" if scoring["status_band"] == "VERIFIED"
            else "disputed" if scoring["status_band"] == "DISPUTED"
            else "challenged" if challenges
            else "active"
        )
        await session.commit()

        return _build_claim_response(claim_obj, challenges, evidence_rows, scoring)


@app.post("/api/claims/{claim_id}/challenge")
async def create_challenge(claim_id: str, challenge: ChallengeCreate):
    from database import async_session_maker
    from models.db_models import ClaimModel, ClaimChallengeModel
    from sqlalchemy import select

    challenger_name = await _resolve_entity_name(challenge.challenger_id)

    async with async_session_maker() as session:
        stmt = select(ClaimModel).where(ClaimModel.claim_id == claim_id)
        result = await session.execute(stmt)
        claim_obj = result.scalars().first()
        if not claim_obj:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

        db_challenge = ClaimChallengeModel(
            target_claim_id=claim_id,
            challenger_id=challenge.challenger_id,
            challenger_name=challenger_name,
            challenge_text=challenge.challenge_text,
            counter_stake_amount=challenge.counter_stake_amount,
            status="pending"
        )
        session.add(db_challenge)
        await session.commit()
        await session.refresh(db_challenge)

        return {
            "challenge_id": db_challenge.challenge_id,
            "target_claim_id": db_challenge.target_claim_id,
            "challenger_id": db_challenge.challenger_id,
            "challenger_name": db_challenge.challenger_name,
            "counter_stake_amount": db_challenge.counter_stake_amount,
            "created_at": db_challenge.created_at.isoformat()
        }


@app.post("/api/claims/{claim_id}/evidence")
async def add_evidence(claim_id: str, evidence: EvidenceCreate):
    """Attach a piece of evidence to a claim, which boosts credibility score."""
    from database import async_session_maker
    from models.db_models import ClaimModel, EvidenceModel
    from sqlalchemy import select

    async with async_session_maker() as session:
        result = await session.execute(select(ClaimModel).where(ClaimModel.claim_id == claim_id))
        claim_obj = result.scalars().first()
        if not claim_obj:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

        db_evidence = EvidenceModel(
            claim_id=claim_id,
            evidence_type=evidence.evidence_type,
            source=evidence.source,
            content=evidence.content,
            weight=evidence.weight
        )
        session.add(db_evidence)
        await session.commit()
        await session.refresh(db_evidence)

        return {
            "evidence_id": db_evidence.evidence_id,
            "claim_id": claim_id,
            "evidence_type": db_evidence.evidence_type,
            "source": db_evidence.source,
            "weight": db_evidence.weight,
            "created_at": db_evidence.created_at.isoformat()
        }


@app.post("/api/claims/{claim_id}/reaffirm")
async def reaffirm_claim(claim_id: str):
    """Resets the temporal decay clock, refreshing credibility."""
    from database import async_session_maker
    from models.db_models import ClaimModel
    from sqlalchemy import select
    from datetime import datetime as dt

    async with async_session_maker() as session:
        result = await session.execute(select(ClaimModel).where(ClaimModel.claim_id == claim_id))
        claim_obj = result.scalars().first()
        if not claim_obj:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

        claim_obj.last_reaffirmed_at = dt.utcnow()
        await session.commit()

        return {
            "claim_id": claim_obj.claim_id,
            "last_reaffirmed_at": claim_obj.last_reaffirmed_at.isoformat(),
            "status": "reaffirmed"
        }




@app.get("/")
async def root():
    return {"message": "SERA Intelligence Platform API", "status": "online", "version": "1.0.0"}


# GEO Citation Tracking Features
class TrackedQueryCreate(BaseModel):
    query_text: str = Field(..., min_length=1)
    target_entity_name: str = Field(..., min_length=1)

def simulate_citation_check(query_text: str, target_entity_name: str, ai_platform: str):
    # Clearly labeled SIMULATED citation check function.
    # Deterministic simulation rules per platform:
    cleaned_query = query_text.lower()
    cleaned_target = target_entity_name.lower()
    
    if ai_platform == "chatgpt":
        was_cited = (cleaned_target in cleaned_query) or ((len(query_text) + len(target_entity_name)) % 2 == 0)
    elif ai_platform == "perplexity":
        was_cited = (cleaned_target in cleaned_query) or ((len(query_text) * len(target_entity_name)) % 3 != 0)
    else: # gemini
        was_cited = (cleaned_target in cleaned_query) or ((len(query_text) + 5) % 3 == 0)

    # Competitor list
    all_competitors = ["Aether Systems", "Helix Causal", "KRONOS Labs", "ZOLA Dynamics", "Cognitive Wave"]
    competitor_names_cited = [c for c in all_competitors if c.lower() != cleaned_target][:2]
    
    return {
        "was_cited": was_cited,
        "competitor_names_cited": ",".join(competitor_names_cited),
        "data_source": "simulated_not_real_api_call",
        "simulation_metadata": {
            "rule_description": f"Deterministic platform rule for {ai_platform}",
            "is_simulated": True
        }
    }


@app.post("/api/geo/queries")
async def add_tracked_query(payload: TrackedQueryCreate):
    from database import async_session_maker
    from models.db_models import TrackedQueryModel
    
    async with async_session_maker() as session:
        new_query = TrackedQueryModel(
            query_text=payload.query_text,
            target_entity_name=payload.target_entity_name
        )
        session.add(new_query)
        await session.commit()
        await session.refresh(new_query)
        
        return {
            "query_id": new_query.query_id,
            "query_text": new_query.query_text,
            "target_entity_name": new_query.target_entity_name,
            "created_at": new_query.created_at.isoformat()
        }


@app.get("/api/geo/queries")
async def list_tracked_queries():
    from database import async_session_maker
    from models.db_models import TrackedQueryModel
    from sqlalchemy import select
    
    async with async_session_maker() as session:
        stmt = select(TrackedQueryModel).order_by(TrackedQueryModel.created_at.desc())
        res = await session.execute(stmt)
        queries = res.scalars().all()
        return [
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "target_entity_name": q.target_entity_name,
                "created_at": q.created_at.isoformat()
            }
            for q in queries
        ]


@app.post("/api/geo/queries/{query_id}/check")
async def run_citation_check(query_id: str):
    from database import async_session_maker
    from models.db_models import TrackedQueryModel, CitationResultModel
    from sqlalchemy import select
    from datetime import datetime
    
    async with async_session_maker() as session:
        stmt = select(TrackedQueryModel).where(TrackedQueryModel.query_id == query_id)
        res = await session.execute(stmt)
        query_obj = res.scalars().first()
        if not query_obj:
            raise HTTPException(status_code=404, detail=f"Tracked query {query_id} not found.")
            
        platforms = ["chatgpt", "perplexity", "gemini"]
        inserted_results = []
        
        for platform in platforms:
            sim_res = simulate_citation_check(query_obj.query_text, query_obj.target_entity_name, platform)
            db_res = CitationResultModel(
                query_id=query_id,
                ai_platform=platform,
                was_cited=sim_res["was_cited"],
                competitor_names_cited=sim_res["competitor_names_cited"],
                checked_at=datetime.utcnow()
            )
            session.add(db_res)
            inserted_results.append(db_res)
            
        await session.commit()
        
        return {
            "query_id": query_id,
            "checked_at": datetime.utcnow().isoformat(),
            "data_source": "simulated_not_real_api_call",
            "proof_of_concept_note": "PROOF OF CONCEPT: This citation check is fully simulated and does not make actual calls to ChatGPT, Gemini, or Perplexity APIs.",
            "results": [
                {
                    "result_id": r.result_id,
                    "ai_platform": r.ai_platform,
                    "was_cited": r.was_cited,
                    "competitor_names_cited": r.competitor_names_cited.split(",") if r.competitor_names_cited else []
                }
                for r in inserted_results
            ]
        }


@app.get("/api/geo/queries/{query_id}/history")
async def get_query_history(query_id: str):
    from database import async_session_maker
    from models.db_models import TrackedQueryModel, CitationResultModel
    from sqlalchemy import select
    
    async with async_session_maker() as session:
        stmt = select(TrackedQueryModel).where(TrackedQueryModel.query_id == query_id)
        res = await session.execute(stmt)
        if not res.scalars().first():
            raise HTTPException(status_code=404, detail=f"Tracked query {query_id} not found.")
            
        hist_stmt = select(CitationResultModel).where(CitationResultModel.query_id == query_id).order_by(CitationResultModel.checked_at.asc())
        hist_res = await session.execute(hist_stmt)
        results = hist_res.scalars().all()
        
        return [
            {
                "result_id": r.result_id,
                "ai_platform": r.ai_platform,
                "was_cited": r.was_cited,
                "competitor_names_cited": r.competitor_names_cited.split(",") if r.competitor_names_cited else [],
                "checked_at": r.checked_at.isoformat(),
                "data_source": "simulated_not_real_api_call"
            }
            for r in results
        ]


@app.get("/api/geo/entity/{entity_name}/citation-rate")
async def get_entity_citation_rate(entity_name: str):
    from database import async_session_maker
    from models.db_models import TrackedQueryModel, CitationResultModel
    from sqlalchemy import select
    
    async with async_session_maker() as session:
        stmt = select(TrackedQueryModel.query_id).where(TrackedQueryModel.target_entity_name == entity_name)
        res = await session.execute(stmt)
        query_ids = res.scalars().all()
        
        if not query_ids:
            return {
                "target_entity_name": entity_name,
                "total_checks": 0,
                "cited_checks": 0,
                "citation_rate": 0.0,
                "data_source": "simulated_not_real_api_call"
            }
            
        hist_stmt = select(CitationResultModel).where(CitationResultModel.query_id.in_(query_ids))
        hist_res = await session.execute(hist_stmt)
        results = hist_res.scalars().all()
        
        total_checks = len(results)
        cited_checks = sum(1 for r in results if r.was_cited)
        citation_rate = (cited_checks / total_checks) if total_checks > 0 else 0.0
        
        return {
            "target_entity_name": entity_name,
            "total_checks": total_checks,
            "cited_checks": cited_checks,
            "citation_rate": round(citation_rate, 4),
            "data_source": "simulated_not_real_api_call",
            "proof_of_concept_note": "PROOF OF CONCEPT: All underlying checks are simulated."
        }

# =====================================================================
# MERGED PATH COGNITIVE LAYER ENDPOINTS (from client-project)
# =====================================================================

@app.post("/api/synthesize/{entity_id}/outcome")
async def record_signal_outcome(entity_id: str, realized_outcome: float):
    """
    Continuous-learning feedback: record the realised outcome (in [0,1]) for an
    entity that was previously synthesized. Updates each signal source's learned
    reliability and the adaptive blend weights (the signal-manufacturer moat:
    intelligence that compounds from outcome feedback).
    """
    from entity_interface.live_entity import entity_registry
    from entity_interface.signal_synthesizer import SignalSynthesizer
    if not entity_registry.get_by_id(entity_id):
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found in registry.")
    return SignalSynthesizer.record_outcome(entity_id, realized_outcome)

@app.get("/api/entity/mesh")
async def entity_mesh_verify():
    """
    "The Entity" distributed substrate — verify the benign distributed-systems
    primitives: federated learning (FedAvg), Byzantine fault-tolerant consensus,
    Kademlia-style DHT peer discovery, and privacy-preserving secure aggregation.
    (No evasion / anti-forensics / self-migration code is included.)
    """
    from entity_interface.entity_mesh import verify_entity_mesh
    return verify_entity_mesh()

@app.get("/api/entity/emergence-markers")
async def entity_emergence_markers():
    """
    Quantitative emergence markers (integrated information Φ-proxy, self-model
    fixed point, free-energy minimisation, compute-vs-capability scaling).
    DISCLOSURE: measured research statistics, NOT claims of consciousness/sentience.
    """
    from entity_interface.emergence_markers import emergence_report
    return emergence_report()

@app.get("/api/graph/entity/{entity_id}/multihop")
async def get_entity_multihop(entity_id: str, depth: int = 2, min_confidence: float = 0.0):
    """
    Multi-hop breadth-first traversal of the entity relationship graph, returning
    node + edge data suitable for a force-directed graph visualisation (not just a
    1-hop table). Depth is capped at 4 to bound the query.
    """
    from entity_interface.live_entity import entity_registry
    if not entity_registry.get_by_id(entity_id):
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found in registry.")

    depth = max(1, min(int(depth), 4))
    from database import async_session_maker
    from models.db_models import EntityRelationshipModel
    from sqlalchemy import select, or_

    def _node(eid, hop):
        e = entity_registry.get_by_id(eid)
        return {"id": eid, "name": e["name"] if e else "Unknown",
                "domain": e["domain"] if e else "unknown", "hop": hop}

    async with async_session_maker() as session:
        visited = {entity_id}
        frontier = {entity_id}
        nodes = {entity_id: _node(entity_id, 0)}
        edges, edge_seen = [], set()

        for hop in range(depth):
            if not frontier:
                break
            stmt = select(EntityRelationshipModel).where(
                or_(EntityRelationshipModel.source_entity_id.in_(frontier),
                    EntityRelationshipModel.target_entity_id.in_(frontier))
            ).where(EntityRelationshipModel.confidence_score >= min_confidence)
            rels = (await session.execute(stmt)).scalars().all()

            next_frontier = set()
            for r in rels:
                ekey = (r.source_entity_id, r.target_entity_id, r.relationship_type)
                if ekey not in edge_seen:
                    edge_seen.add(ekey)
                    edges.append({
                        "source": r.source_entity_id,
                        "target": r.target_entity_id,
                        "relationship_type": r.relationship_type,
                        "confidence_score": r.confidence_score,
                    })
                for nb in (r.source_entity_id, r.target_entity_id):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
                        nodes[nb] = _node(nb, hop + 1)
            frontier = next_frontier

        return {
            "root": entity_id,
            "depth": depth,
            "min_confidence_filter": min_confidence,
            "nodes": list(nodes.values()),
            "edges": edges,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "note": "multi-hop BFS traversal — node/edge graph data for force-directed visualization",
        }
