from fastapi import APIRouter, Query, HTTPException
from services.dark_intel_service import DarkIntelService

router = APIRouter(prefix="/api/dark-intel", tags=["dark-intel"])

@router.get("/briefings")
async def get_briefings(clearance: str = Query(default="ALL")):
    try:
        # Accept 'all' or specific level and query the service
        briefs = await DarkIntelService.get_briefings(clearance)
        return briefs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
