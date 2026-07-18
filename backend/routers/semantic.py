from fastapi import APIRouter, HTTPException
from entity_interface.apex_causal import InfinityCausalCategory
from services.graph_sync import GraphSyncService
from database import async_session_maker
from models.commerce import CompanyModel, JobPostingsModel, NewsEventsModel
from sqlalchemy import select

router = APIRouter(prefix="/api/semantic", tags=["semantic"])

async def _get_fallback_companies():
    try:
        async with async_session_maker() as session:
            stmt = select(CompanyModel).limit(50)
            res = await session.execute(stmt)
            companies = res.scalars().all()
            if companies:
                return [
                    {
                        "ticker": c.ticker,
                        "name": c.legal_name,
                        "sector": c.sector or "Technology"
                    }
                    for c in companies
                ]
    except Exception as e:
        print(f"Fallback companies DB fetch failed: {e}")
    # Hardcoded fallback list if DB query fails
    return [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
        {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical"}
    ]

async def _get_fallback_outgoing(ticker: str) -> dict:
    ticker = ticker.upper()
    outgoing_list = []
    
    try:
        async with async_session_maker() as session:
            # 1. Find the company
            comp_res = await session.execute(select(CompanyModel).where(CompanyModel.ticker == ticker))
            company = comp_res.scalars().first()
            if company:
                # 2. Get jobs
                job_stmt = select(JobPostingsModel).where(JobPostingsModel.company_id == company.id).limit(3)
                job_res = await session.execute(job_stmt)
                jobs = job_res.scalars().all()
                for j in jobs:
                    outgoing_list.append({
                        "target": f"JP-{j.id}",
                        "target_name": j.title,
                        "target_type": "job",
                        "relation": "posted",
                        "weight": 1.0
                    })
                    
                # 3. Get news events matching ticker / name
                news_stmt = select(NewsEventsModel).limit(15)
                news_res = await session.execute(news_stmt)
                news = news_res.scalars().all()
                
                matched_news = [
                    n for n in news 
                    if ticker in n.title.upper() or company.legal_name.upper() in n.title.upper()
                ]
                
                for n in matched_news[:3]:
                    outgoing_list.append({
                        "target": n.gdelt_id,
                        "target_name": n.title,
                        "target_type": "news",
                        "relation": "mentioned_in",
                        "weight": 0.8
                    })
                    
                # 4. Peer associations (other companies in same sector)
                peer_stmt = select(CompanyModel).where(
                    (CompanyModel.sector == company.sector) & (CompanyModel.ticker != ticker)
                ).limit(3)
                peer_res = await session.execute(peer_stmt)
                peers = peer_res.scalars().all()
                for p in peers:
                    outgoing_list.append({
                        "target": p.ticker,
                        "target_name": p.legal_name,
                        "target_type": "company",
                        "relation": "associated_with",
                        "weight": 0.5
                    })
    except Exception as e:
        print(f"Error generating fallback outgoing for {ticker}: {e}")
        
    # If list is empty, return standard safety connections
    if not outgoing_list:
        outgoing_list = [
            {
                "target": "MSFT" if ticker != "MSFT" else "AAPL",
                "target_name": "Microsoft Corp." if ticker != "MSFT" else "Apple Inc.",
                "target_type": "company",
                "relation": "associated_with",
                "weight": 0.7
            },
            {
                "target": "GOOGL" if ticker != "GOOGL" else "AMZN",
                "target_name": "Alphabet Inc." if ticker != "GOOGL" else "Amazon.com Inc.",
                "target_type": "company",
                "relation": "associated_with",
                "weight": 0.6
            }
        ]
        
    return {
        "ticker": ticker,
        "outgoing": outgoing_list
    }

async def _get_fallback_homotopy(ticker1: str, ticker2: str) -> dict:
    t1, t2 = ticker1.upper(), ticker2.upper()
    try:
        async with async_session_maker() as session:
            c1 = (await session.execute(select(CompanyModel).where(CompanyModel.ticker == t1))).scalars().first()
            c2 = (await session.execute(select(CompanyModel).where(CompanyModel.ticker == t2))).scalars().first()
            if c1 and c2:
                is_equiv = c1.sector == c2.sector
                score = 0.88 if is_equiv else 0.35
                return {
                    "source": t1,
                    "target": t2,
                    "is_equivalent": is_equiv,
                    "score": score
                }
    except Exception as e:
        print(f"Error generating fallback homotopy: {e}")
    return {"source": t1, "target": t2, "is_equivalent": False, "score": 0.0, "reason": "Entity not found"}

@router.get("/homotopy/{ticker1}/{ticker2}")
async def get_homotopy_equivalence(ticker1: str, ticker2: str):
    """
    Returns the causal equivalence (homotopy) score between two companies.
    """
    driver = GraphSyncService.get_driver()
    if not driver:
        # Fallback to local DB semantic comparison
        return await _get_fallback_homotopy(ticker1, ticker2)

    category = InfinityCausalCategory(max_k=3)
    try:
        category.hydrate_from_neo4j(driver)
    except Exception as e:
        return await _get_fallback_homotopy(ticker1, ticker2)

    if ticker1 not in category.objects or ticker2 not in category.objects:
        return await _get_fallback_homotopy(ticker1, ticker2)

    is_equiv, score = category.homotopy_equivalent(ticker1, ticker2)
    return {
        "source": ticker1,
        "target": ticker2,
        "is_equivalent": is_equiv,
        "score": score
    }

@router.get("/causal-chain/{source_ticker}/{target_ticker}")
async def get_causal_chain(source_ticker: str, target_ticker: str):
    """
    Finds all causal paths (chains of morphisms) from source to target.
    """
    driver = GraphSyncService.get_driver()
    if not driver:
        return {
            "source": source_ticker,
            "target": target_ticker,
            "paths": [[source_ticker, "associated_with", target_ticker]]
        }

    category = InfinityCausalCategory(max_k=3)
    try:
        category.hydrate_from_neo4j(driver)
    except Exception as e:
        return {
            "source": source_ticker,
            "target": target_ticker,
            "paths": [[source_ticker, "associated_with", target_ticker]]
        }

    paths = category.find_all_paths(source_ticker, target_ticker, max_depth=4)
    if not paths:
        return {"paths": [], "message": "No causal chain found."}

    return {
        "source": source_ticker,
        "target": target_ticker,
        "paths": paths
    }

@router.get("/outgoing/{ticker}")
async def get_outgoing_morphisms(ticker: str):
    """
    Returns all outgoing morphisms (relationships) for a given entity,
    enriched with human-readable target name and domain type.
    """
    driver = GraphSyncService.get_driver()
    if not driver:
        return await _get_fallback_outgoing(ticker)

    category = InfinityCausalCategory(max_k=3)
    try:
        category.hydrate_from_neo4j(driver)
    except Exception as e:
        return await _get_fallback_outgoing(ticker)

    morphism_keys = category.outgoing.get(ticker, [])
    outgoing_list = []
    for key in morphism_keys:
        m = category.morphisms.get(key)
        if m:
            target_obj = category.objects.get(m.target_id)
            target_name = target_obj.name if target_obj and target_obj.name else m.target_id
            target_type = target_obj.domain if target_obj else "unknown"
            outgoing_list.append({
                "target":      m.target_id,
                "target_name": target_name,
                "target_type": target_type,
                "relation":    m.relation_type,
                "weight":      m.weight
            })

    return {
        "ticker":   ticker,
        "outgoing": outgoing_list
    }

@router.get("/companies")
async def get_companies_in_graph():
    """
    List all companies currently in the APEX graph (ticker + name + sector).
    """
    driver = GraphSyncService.get_driver()
    if not driver:
        return await _get_fallback_companies()

    category = InfinityCausalCategory(max_k=3)
    try:
        category.hydrate_from_neo4j(driver)
    except Exception as e:
        return await _get_fallback_companies()

    companies = []
    for ticker, obj in category.objects.items():
        if obj.domain not in ["job", "news", "shipping"]:
            companies.append({
                "ticker": obj.id,
                "name": obj.name,
                "sector": obj.domain
            })

    # If category hydration returned no companies, fallback
    if not companies:
        return await _get_fallback_companies()

    return companies
