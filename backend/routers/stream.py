from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from generators.financial import generate_financial_event
from generators.healthcare import generate_healthcare_event
from generators.iot import generate_iot_event
from generators.social import generate_social_event
from core.entity_resolution import entity_registry
from core.entropy_engine import entropy_engine
from database import async_session_maker
from models.db_models import EventModel, AlertModel
import asyncio, random, json
from datetime import datetime

router = APIRouter(tags=["stream"])
GENERATORS = [generate_financial_event, generate_healthcare_event,
              generate_iot_event, generate_social_event]

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
        print(f"[stream] DB write failed (non-fatal): {exc}")

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
        print(f"[stream] Alert DB write failed (non-fatal): {exc}")

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            entity = entity_registry.get_random_entity()
            gen_fn = random.choice(GENERATORS)
            event = gen_fn(entity["id"], entity["name"])
            metrics = entropy_engine.ingest(entity["id"], event["event_type"],
                                            event["protocol"])
            entity_registry.update_entropy(entity["id"], metrics["entropy"], metrics["alert_triggered"], metrics.get("z_score", 0.0))
            asyncio.create_task(save_event_to_db(event, entropy_delta=metrics["entropy"]))
            if metrics["alert_triggered"]:
                asyncio.create_task(save_alert_to_db(
                    entity_id=event["entity_id"],
                    entropy_score=metrics["entropy"],
                    z_score=metrics.get("z_score", 0.0),
                    domain=event.get("domain", "unknown"),
                ))
            message = {
                "type": "event",
                "timestamp": datetime.utcnow().isoformat(),
                "event": event,
                "metrics": metrics
            }
            await websocket.send_text(json.dumps(message))
            await asyncio.sleep(random.uniform(0.5, 2.0))
    except WebSocketDisconnect:
        pass
            