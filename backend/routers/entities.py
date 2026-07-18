from fastapi import APIRouter, Query
from core.entity_resolution import entity_registry

router = APIRouter(redirect_slashes=False)

@router.get("", include_in_schema=True)
@router.get("/", include_in_schema=False)
async def list_entities(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    from config import USE_REAL_DATA
    if USE_REAL_DATA:
        from database import async_session_maker
        from models.commerce import CompanyModel
        from sqlalchemy import select, func
        from sqlalchemy.orm import selectinload
        try:
            async with async_session_maker() as session:
                # Get the total count of companies
                total_res = await session.execute(select(func.count(CompanyModel.id)))
                total_count = total_res.scalar() or 0
                
                # Query only paginated slice of companies
                stmt = select(CompanyModel).options(
                    selectinload(CompanyModel.financial_metrics),
                    selectinload(CompanyModel.job_postings)
                ).offset(offset).limit(limit)
                result = await session.execute(stmt)
                companies = result.scalars().all()
                if companies:
                    entities = []
                    for c in companies:
                        latest_metrics = c.financial_metrics[-1] if c.financial_metrics else None
                        rev = latest_metrics.revenue if latest_metrics else 0.0
                        
                        # In-memory expansion score to avoid N+1 DB queries
                        jobs_count = len(c.job_postings)
                        sec_count = len(c.financial_metrics)
                        new_job_postings_velocity = min(jobs_count / 10.0, 1.0)
                        sec_8k_events = min(sec_count / 5.0, 1.0)
                        github_commit_activity = 0.5  # default baseline
                        score = round(0.5 * new_job_postings_velocity + 0.3 * sec_8k_events + 0.2 * github_commit_activity, 4)
                        
                        entities.append({
                            "id": c.id,
                            "name": c.legal_name,
                            "domain": c.sector or "financial",
                            "status": "stable",
                            "entropy": 0.5,
                            "event_count": jobs_count,
                            "alert_count": 0,
                            "ticker": c.ticker,
                            "revenue": rev,
                            "expansion_score": score,
                            "news_sentiment": c.news_sentiment or 0.0,
                            "news_mentions": c.news_mentions or 0,
                            "reddit_sentiment": c.reddit_sentiment or 0.0,
                            "reddit_mentions": c.reddit_mentions or 0
                        })
                    return {
                        "total": total_count,
                        "limit": limit,
                        "offset": offset,
                        "entities": entities
                    }
        except Exception as e:
            print(f"[ENTITIES] DB query failed, falling back to registry: {e}")

    all_entities = entity_registry.get_all()
    paginated = all_entities[offset:offset+limit]
    # Add mock expansion score to paginated entities if missing
    for e in paginated:
        if "expansion_score" not in e:
            # deterministic fallback based on entropy
            e["expansion_score"] = round(0.3 + (e.get("entropy", 0.5) * 0.2), 4)
            
    return {
        "total": len(all_entities),
        "limit": limit,
        "offset": offset,
        "entities": paginated
    }

@router.get("/{entity_id}")
async def get_entity(entity_id: str):
    from config import USE_REAL_DATA
    if USE_REAL_DATA:
        from database import async_session_maker
        from models.commerce import CompanyModel
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        try:
            async with async_session_maker() as session:
                stmt = select(CompanyModel).where(CompanyModel.id == entity_id).options(
                    selectinload(CompanyModel.financial_metrics),
                    selectinload(CompanyModel.job_postings)
                )
                result = await session.execute(stmt)
                c = result.scalars().first()
                if c:
                    latest_metrics = c.financial_metrics[-1] if c.financial_metrics else None
                    rev = latest_metrics.revenue if latest_metrics else 0.0
                    from services.insight_engine import InsightEngine
                    score = await InsightEngine.generate_expansion_score(c.id)
                    return {
                        "id": c.id,
                        "name": c.legal_name,
                        "domain": c.sector or "financial",
                        "status": "stable",
                        "entropy": 0.5,
                        "event_count": len(c.job_postings),
                        "alert_count": 0,
                        "ticker": c.ticker,
                        "revenue": rev,
                        "expansion_score": score,
                        "news_sentiment": c.news_sentiment or 0.0,
                        "news_mentions": c.news_mentions or 0,
                        "reddit_sentiment": c.reddit_sentiment or 0.0,
                        "reddit_mentions": c.reddit_mentions or 0
                    }
        except Exception as e:
            print(f"[ENTITIES] DB query failed for {entity_id}, falling back: {e}")

    entity = entity_registry.get_by_id(entity_id)
    if not entity:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Entity not found")
    if "expansion_score" not in entity:
        entity["expansion_score"] = round(0.3 + (entity.get("entropy", 0.5) * 0.2), 4)
    return entity

@router.get("/{ticker}/full")
async def get_entity_full_profile(ticker: str):
    """Retrieves full 7-widget aggregated intelligence profile for an entity ticker."""
    from services.entity_aggregator import EntityAggregator
    profile = await EntityAggregator.get_full_profile(ticker)
    if not profile:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Entity with ticker {ticker} not found.")
    return profile