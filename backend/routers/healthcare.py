from fastapi import APIRouter
from sqlalchemy import select, func
from database import async_session_maker
from models.commerce import HealthcareMetric

router = APIRouter(prefix="/api/healthcare", tags=["healthcare"])

@router.get("/metrics")
async def get_healthcare_metrics():
    """Get the latest healthcare metrics for all states."""
    async with async_session_maker() as session:
        # Get the latest measurement date
        latest_res = await session.execute(select(func.max(HealthcareMetric.measurement_date)))
        latest = latest_res.scalar()
        
        if latest is None:
            return []
            
        stmt = select(HealthcareMetric).where(HealthcareMetric.measurement_date == latest)
        metrics_res = await session.execute(stmt)
        metrics = metrics_res.scalars().all()
        
        return [
            {
                "region": m.region,
                "admission_count": m.admission_count,
                "avg_total_payment": m.avg_total_payment,
                "drug_claim_count": m.drug_claim_count
            }
            for m in metrics
        ]
