from fastapi import APIRouter
from core.entity_resolution import entity_registry
from core.entropy_engine import entropy_engine

router = APIRouter(prefix="/api/axiom", tags=["axiom"])

@router.get("/entropy")
async def get_entropy_data():
    entities = entity_registry.get_all()[:10]
    result = []

    for e in entities:
        result.append({
            "entity_id": e["id"],
            "entity_name": e["name"],
            "domain": e["domain"],
            "entropy": e["entropy"],
            "status": e["status"],
            "history": [round(v, 3) for v in entropy_engine.entity_entropy_history.get(e["id"], [])][-20:]
        })
    return result

@router.get("/alerts")
async def get_alerts():
    entities = entity_registry.get_all()
    alerts = []

    for e in entities:
        if e["status"] == "pre-transition":
            alerts.append({
                "entity_id": e["id"],
                "entity_name": e["name"],
                "alert_type": "entropy_spike",
                "severity": "warning" if e["entropy"] < 3.0 else "critical",
                "entropy_value": e["entropy"],
                "description": f"Entity {e['name']} showing entropy spike in {e['domain']} domain"
            })
    return alerts