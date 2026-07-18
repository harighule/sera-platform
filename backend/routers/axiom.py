from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from database import async_session_maker
from models.commerce import CompanyModel
from services.axiom_monitor import AxiomMonitor

router = APIRouter(prefix="/api/axiom", tags=["axiom"])

@router.get("/monitor")
async def get_axiom_monitor():
    try:
        async with async_session_maker() as session:
            # Count total companies first (very fast)
            from sqlalchemy import func
            total_res = await session.execute(select(func.count(CompanyModel.id)))
            total_entities = total_res.scalar() or 0
            
            # Fetch only the first 50 companies for detailed display
            comp_res = await session.execute(select(CompanyModel).limit(50))
            companies = comp_res.scalars().all()

        entropy_summary = []
        active_alerts = 0

        # Build detailed entropy summary for the 50 companies
        for company in companies:
            metrics = await AxiomMonitor.compute_entropy(company.id)
            is_pre = metrics["is_pre_transition"]
            
            if is_pre:
                active_alerts += 1

            status = "pre-transition" if is_pre else "stable"
            
            # Generate history for display
            history = []
            base = metrics["baseline_entropy"]
            curr = metrics["current_entropy"]
            for i in range(10):
                step = base + (curr - base) * (i / 9.0)
                history.append(round(step, 4))

            entropy_summary.append({
                "entity_id": company.id,
                "entity_name": company.legal_name,
                "domain": company.sector or "technology",
                "entropy": metrics["current_entropy"],
                "z_score": metrics["z_score"],
                "status": status,
                "history": history
            })

        # Extract high risk entities from the active monitor slice for speed
        high_risk_entities = [
            {
                "entity_id": item["entity_id"],
                "entity_name": item["entity_name"],
                "risk_factor": "entropy_spike",
                "entropy": item["entropy"]
            }
            for item in entropy_summary if item["status"] == "pre-transition" or item["entropy"] > 0.6
        ]

        return {
            "total_entities": total_entities,
            "active_alerts": active_alerts,
            "high_risk_entities": high_risk_entities,
            "entropy_summary": entropy_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entropy")
async def get_entropy_data():
    # Keep backward compatibility if other places call it
    res = await get_axiom_monitor()
    return res["entropy_summary"]

@router.get("/alerts")
async def get_alerts():
    # Keep backward compatibility if other places call it
    res = await get_axiom_monitor()
    alerts = []
    for item in res["entropy_summary"]:
        if item["status"] == "pre-transition":
            alerts.append({
                "entity_id": item["entity_id"],
                "entity_name": item["entity_name"],
                "alert_type": "entropy_spike",
                "severity": "warning" if item["entropy"] < 2.0 else "critical",
                "entropy_value": item["entropy"],
                "description": f"Entity {item['entity_name']} showing entropy spike in {item['domain']} domain"
            })
    return alerts