from fastapi import APIRouter, HTTPException, Query
from services.insight_engine import InsightEngine
from database import async_session_maker
from models.commerce import CompanyModel
from sqlalchemy import select

router = APIRouter(prefix="/api/insights", tags=["insights"])

@router.get("/expansion/{ticker}")
async def get_expansion_insight(ticker: str):
    """Retrieves computed expansion score for a given company ticker."""
    async with async_session_maker() as session:
        # Normalize ticker to uppercase
        res = await session.execute(select(CompanyModel).where(CompanyModel.ticker == ticker.upper()))
        company = res.scalars().first()
        if not company:
            raise HTTPException(
                status_code=404, 
                detail=f"Company with ticker {ticker} not found in commercial registry."
            )
        
        score = await InsightEngine.generate_expansion_score(company.id)
        return {
            "ticker": ticker.upper(),
            "company_name": company.legal_name,
            "company_id": company.id,
            "expansion_score": score
        }

@router.get("/purchase/{category}")
async def get_purchase_insight(
    category: str, 
    region: str = Query(default="GLOBAL", description="Target region for trend analysis")
):
    """Retrieves computed purchase intent score for a category and region."""
    score = await InsightEngine.generate_purchase_intent(category, region)
    return {
        "category": category,
        "region": region,
        "purchase_intent_score": score
    }

@router.get("/narrative/expansion/{ticker}")
async def get_narrative_expansion(ticker: str):
    """Generates a structured expansion report for a ticker."""
    from services.narrative_engine import NarrativeEngine
    # Verify company exists
    async with async_session_maker() as session:
        res = await session.execute(select(CompanyModel).where(CompanyModel.ticker == ticker.upper()))
        company = res.scalars().first()
        # Even if company is missing, mock mode may still provide fallback data
        from config import USE_REAL_DATA
        if not company and USE_REAL_DATA:
            raise HTTPException(
                status_code=404, 
                detail=f"Company with ticker {ticker} not found in commercial registry."
            )
            
    report = await NarrativeEngine.generate_expansion_report(ticker)
    return report

@router.get("/narrative/purchase/{category}")
async def get_narrative_purchase(
    category: str, 
    region: str = Query(default="GLOBAL", description="Target region for trend analysis")
):
    """Generates a narrative text report for purchase intent."""
    from services.narrative_engine import NarrativeEngine
    report = await NarrativeEngine.generate_purchase_intent_report(category, region)
    return {"report": report}
