from fastapi import APIRouter
from core.entity_resolution import entity_registry
from database import async_session_maker
from models.db_models import EventModel
from sqlalchemy import select, func
from datetime import datetime, timedelta
import time

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
START_TIME = time.time()

@router.get("/stats")
async def get_stats():
    from config import USE_REAL_DATA
    if USE_REAL_DATA:
        from models.commerce import CompanyModel
        try:
            async with async_session_maker() as session:
                comp_res = await session.execute(select(func.count()).select_from(CompanyModel))
                total_companies = comp_res.scalar() or 0
                if total_companies > 0:
                    since = datetime.utcnow() - timedelta(seconds=60)
                    eps_result = await session.execute(
                        select(func.count()).select_from(EventModel).where(EventModel.timestamp >= since)
                    )
                    events_last_60s = eps_result.scalar() or 0
                    events_per_second = round(events_last_60s / 60.0, 2)

                    events_result = await session.execute(
                        select(func.count()).select_from(EventModel)
                    )
                    events_processed = events_result.scalar() or 0

                    proto_result = await session.execute(
                        select(func.count(func.distinct(EventModel.protocol)))
                    )
                    protocols_active = proto_result.scalar() or 4

                    return {
                        "total_entities": total_companies,
                        "active_alerts": 0,
                        "events_per_second": events_per_second,
                        "protocols_active": protocols_active,
                        "events_processed": events_processed,
                        "uptime_seconds": round(time.time() - START_TIME, 1),
                        "entropy_average": 0.5,
                    }
        except Exception as e:
            print(f"[DASHBOARD] DB stats query failed, falling back: {e}")

    entities = entity_registry.get_all()
    pre_transition = [e for e in entities if e["status"] == "pre-transition"]

    async with async_session_maker() as session:
        # Real events-per-second: count events in the last 60 seconds
        since = datetime.utcnow() - timedelta(seconds=60)
        eps_result = await session.execute(
            select(func.count()).select_from(EventModel).where(EventModel.timestamp >= since)
        )
        events_last_60s = eps_result.scalar() or 0
        events_per_second = round(events_last_60s / 60.0, 2)

        # Total events ever processed
        events_result = await session.execute(
            select(func.count()).select_from(EventModel)
        )
        events_processed = events_result.scalar() or 0

        # Distinct active protocols from events table; fallback 4 if no events yet
        proto_result = await session.execute(
            select(func.count(func.distinct(EventModel.protocol)))
        )
        protocols_active = proto_result.scalar() or 4

    return {
        "total_entities": len(entities),
        "active_alerts": len(pre_transition),
        "events_per_second": events_per_second,
        "protocols_active": protocols_active,
        "events_processed": events_processed,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "entropy_average": round(sum(e["entropy"] for e in entities) / max(len(entities), 1), 4),
    }