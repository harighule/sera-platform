from fastapi import APIRouter
from datetime import datetime, timedelta
from sqlalchemy import select, desc
from database import async_session_maker
from models.commerce import IngestionLogModel
from models.db_models import AlertModel

router = APIRouter(prefix="/api", tags=["health_admin"])

@router.get("/health")
async def health_check():
    """Simple liveness probe for Render / load balancers."""
    return {"status": "ok"}

@router.get("/health/freshness")
async def get_data_freshness():
    """
    Checks the ingestion_logs table.
    If SEC data is older than 24 hours, returns status "stale".
    If GDELT is older than 4 hours, returns a warning.
    """
    async with async_session_maker() as session:
        # Find latest successful SEC run
        sec_stmt = select(IngestionLogModel).where(
            IngestionLogModel.source == "sec",
            IngestionLogModel.status == "success"
        ).order_by(desc(IngestionLogModel.last_run)).limit(1)
        
        sec_res = await session.execute(sec_stmt)
        sec_log = sec_res.scalars().first()
        
        # Find latest successful GDELT run
        gdelt_stmt = select(IngestionLogModel).where(
            IngestionLogModel.source == "gdelt",
            IngestionLogModel.status == "success"
        ).order_by(desc(IngestionLogModel.last_run)).limit(1)
        
        gdelt_res = await session.execute(gdelt_stmt)
        gdelt_log = gdelt_res.scalars().first()

        now = datetime.utcnow()
        status = "fresh"
        alert_msg = None
        
        sec_last = sec_log.last_run if sec_log else None
        gdelt_last = gdelt_log.last_run if gdelt_log else None
        
        if not sec_last or (now - sec_last) > timedelta(hours=24):
            status = "stale"
            alert_msg = f"SEC data not updated since {sec_last.strftime('%Y-%m-%d %H:%M:%S') if sec_last else 'never'}."
        elif not gdelt_last or (now - gdelt_last) > timedelta(hours=4):
            status = "warning"
            alert_msg = f"GDELT data last updated at {gdelt_last.strftime('%Y-%m-%d %H:%M:%S') if gdelt_last else 'never'}."

        return {
            "status": status,
            "alert": alert_msg,
            "sec_last_run": sec_last.isoformat() if sec_last else None,
            "gdelt_last_run": gdelt_last.isoformat() if gdelt_last else None
        }

@router.get("/admin/alerts")
async def get_admin_alerts():
    """
    Retrieves system alerts and ingestion failure alerts from the DB.
    """
    async with async_session_maker() as session:
        stmt = select(AlertModel).order_by(desc(AlertModel.created_at)).limit(50)
        res = await session.execute(stmt)
        alerts = res.scalars().all()
        return [
            {
                "id": a.id,
                "entity_id": a.entity_id,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "description": a.description,
                "entropy_value": a.entropy_value,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "resolved": a.resolved
            }
            for a in alerts
        ]
