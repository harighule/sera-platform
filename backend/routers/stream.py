import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from experimental.fake_data_generator import (
    generate_financial_event, generate_healthcare_event,
    generate_iot_event, generate_social_event, FakeDataGenerator
)
from core.entity_resolution import entity_registry
from core.entropy_engine import entropy_engine
from database import async_session_maker
from models.db_models import EventModel, AlertModel
import asyncio, random, json, os
from datetime import datetime

logger = logging.getLogger("sera.routers.stream")

router = APIRouter(tags=["stream"])
GENERATORS = [generate_financial_event, generate_healthcare_event,
              generate_iot_event, generate_social_event]

# ---------------------------------------------------------------------------
# Resolve the valid API key set at import time — mirrors the logic in main.py
# so this module can authenticate independently (defence-in-depth).
# ---------------------------------------------------------------------------
def _build_ws_api_keys() -> dict:
    import json as _json
    env = os.getenv("API_KEYS", "")
    keys: dict = {}
    if env.strip():
        try:
            parsed = _json.loads(env)
            if isinstance(parsed, dict):
                keys = parsed
            elif isinstance(parsed, list):
                keys = {k: f"client_{i}" for i, k in enumerate(parsed)}
        except _json.JSONDecodeError:
            for val in env.split(","):
                val = val.strip()
                if val:
                    keys[val] = f"client_{val[-4:] if len(val) >= 4 else val}"
    demo = os.getenv("DEMO_API_KEY", "sera-demo-2026")
    if demo and demo not in keys:
        keys[demo] = "default_demo"
    return keys

_WS_API_KEYS = _build_ws_api_keys()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

async def broadcast_real_event(event_dict: dict, metrics: dict):
    """Pushes ingested event updates to all connected clients."""
    message = {
        "type": "event",
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_dict,
        "metrics": metrics
    }
    await manager.broadcast(json.dumps(message))
    await save_event_to_db(event_dict, entropy_delta=metrics.get("entropy", 0.0))

async def save_event_to_db(event_dict: dict, entropy_delta: float = 0.0):
    """Persist a single streamed event to the DB. Errors are swallowed so they
    never interrupt the WebSocket broadcast loop."""
    try:
        async with async_session_maker() as session:
            session.add(EventModel(
                entity_id=event_dict["entity_id"],
                protocol=event_dict.get("protocol", "unknown"),
                event_type=event_dict.get("event_type", "unknown"),
                payload=event_dict.get("payload", {}),
                timestamp=datetime.utcnow(),
                entropy_delta=entropy_delta,
            ))
            await session.commit()
    except Exception as exc:
        logger.error(f"[stream] DB write failed (non-fatal): {exc}", exc_info=True)

async def save_alert_to_db(entity_id: str, entropy_score: float, z_score: float, domain: str):
    """Persist a pre-transition alert to the DB. Errors are swallowed so they
    never interrupt the WebSocket broadcast loop."""
    try:
        async with async_session_maker() as session:
            severity = "critical" if z_score > 3.0 else "high"
            session.add(AlertModel(
                entity_id=entity_id,
                alert_type="PRE_TRANSITION",
                severity=severity,
                description=(
                    f"AXIOM-\u03a6: Entropy spike detected for entity {entity_id}. "
                    f"Z-score: {z_score:.2f} | Domain: {domain}"
                ),
                entropy_value=entropy_score,
                created_at=datetime.utcnow(),
                resolved=False,
            ))
            await session.commit()
    except Exception as exc:
        logger.error(f"[stream] Alert DB write failed (non-fatal): {exc}", exc_info=True)

# ===========================================================================
# MAIN WEBSOCKET ENDPOINT - FIXED VERSION
# ===========================================================================
@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    
    FIX: Accept connection FIRST, then check authentication.
    This follows the WebSocket protocol correctly.
    """
    # ✅ STEP 1: Accept the WebSocket connection first
    await websocket.accept()
    
    # ✅ STEP 2: Now check authentication via query parameters
    api_key = websocket.query_params.get("api_key")
    if not api_key or api_key not in _WS_API_KEYS:
        logger.warning(
            f"[SECURITY] WebSocket upgrade rejected. Invalid api_key: {api_key}. "
            f"Client IP={websocket.client.host if websocket.client else 'unknown'}"
        )
        await websocket.close(code=1008, reason="Unauthorized: invalid or missing api_key")
        return

    # ✅ STEP 3: Log successful connection
    logger.info(f"✅ WebSocket connected - Client: {websocket.client.host if websocket.client else 'unknown'}")
    from config import USE_REAL_DATA
    logger.info(f"📊 USE_REAL_DATA={USE_REAL_DATA} | ENTITY_MODE={os.getenv('ENTITY_MODE', 'mock')}")

    # ✅ STEP 4: Handle based on mode
    if USE_REAL_DATA:
        # =============================================================
        # REAL DATA MODE - Wait for real events from external sources
        # =============================================================
        try:
            # Add to connection manager for broadcasting real events
            await manager.connect(websocket)
            while True:
                # Keep connection alive by waiting for client messages
                # Real events are pushed via broadcast_real_event()
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("❌ WebSocket disconnected (real mode)")
            manager.disconnect(websocket)
    else:
        # =============================================================
        # MOCK DATA MODE - Generate synthetic events every 0.5-2 seconds
        # =============================================================
        try:
            logger.info("🎲 Starting mock event generator...")
            event_count = 0
            
            while True:
                # Get a random entity
                entity = entity_registry.get_random_entity()
                
                # Generate random event
                event = FakeDataGenerator.generate_random_event(
                    entity["id"], 
                    entity["name"]
                )
                
                # Process through entropy engine
                metrics = entropy_engine.ingest(
                    entity["id"], 
                    event["event_type"],
                    event["protocol"]
                )
                
                # Update entity registry
                entity_registry.update_entropy(
                    entity["id"], 
                    metrics["entropy"], 
                    metrics["alert_triggered"], 
                    metrics.get("z_score", 0.0)
                )
                
                # Save event to database (async, non-blocking)
                asyncio.create_task(
                    save_event_to_db(event, entropy_delta=metrics["entropy"])
                )
                
                # Save alert if triggered (async, non-blocking)
                if metrics["alert_triggered"]:
                    asyncio.create_task(
                        save_alert_to_db(
                            entity_id=event["entity_id"],
                            entropy_score=metrics["entropy"],
                            z_score=metrics.get("z_score", 0.0),
                            domain=event.get("domain", "unknown"),
                        )
                    )
                    logger.info(f"🚨 Alert triggered for entity {event['entity_id']}")
                
                # Build and send message
                message = {
                    "type": "event",
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": event,
                    "metrics": metrics
                }
                
                await websocket.send_text(json.dumps(message))
                event_count += 1
                
                # Log every 10 events
                if event_count % 10 == 0:
                    logger.info(f"📊 Sent {event_count} mock events so far")
                
                # Wait 0.5-2 seconds before next event
                await asyncio.sleep(random.uniform(0.5, 2.0))
                
        except WebSocketDisconnect:
            logger.info(f"❌ WebSocket disconnected after {event_count} events (mock mode)")
        except Exception as e:
            logger.error(f"❌ Error in mock event generator: {e}", exc_info=True)
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except:
                pass