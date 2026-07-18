from fastapi import APIRouter
from sqlalchemy import select, func
from database import async_session_maker
from models.commerce import ExecutiveMovement
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/executive", tags=["executive"])

def determine_change_type(old_title: str, new_title: str) -> str:
    if not old_title:
        return "hire"
    if not new_title:
        return "departure"
    new_title_lower = new_title.lower()
    if any(keyword in new_title_lower for keyword in ["retire", "depart", "step down", "advisor"]):
        return "departure"
    return "promotion"

@router.get("/movements")
async def get_executive_movements():
    """Get recent executive movements and metadata."""
    async with async_session_maker() as session:
        # Fetch all movements, ordered by change_date descending
        stmt = select(ExecutiveMovement).order_by(ExecutiveMovement.change_date.desc()).limit(100)
        res = await session.execute(stmt)
        movements = res.scalars().all()
        
        # Calculate count in the last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        count_stmt = select(func.count(ExecutiveMovement.id)).where(ExecutiveMovement.change_date >= seven_days_ago)
        count_res = await session.execute(count_stmt)
        last_7_days_count = count_res.scalar() or 0
        
        movements_list = []
        for m in movements:
            change_type = determine_change_type(m.old_title, m.new_title)
            movements_list.append({
                "id": m.id,
                "ticker": m.ticker,
                "exec_name": m.exec_name,
                "old_title": m.old_title,
                "new_title": m.new_title,
                "change_date": m.change_date.isoformat() if m.change_date else None,
                "change_type": change_type
            })
            
        return {
            "movements": movements_list,
            "last_7_days_count": last_7_days_count
        }
