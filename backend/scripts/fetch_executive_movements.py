import os
import sys
import logging
import asyncio
from datetime import datetime
from sqlalchemy import select

# Add backend directory to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import async_session_maker, init_db
from models.commerce import CompanyModel, ExecutiveMovement
from config import USE_REAL_DATA, APIFY_TOKEN

logger = logging.getLogger("sera.exec_movements_fetcher")

async def run_executive_movements_ingestion() -> int:
    """
    Fetch executive lists using Apify's LinkedIn Decision Makers Scraper task dataset.
    Detects movements by comparing incoming executive titles with the latest stored titles (delta detection).
    If USE_REAL_DATA is False or APIFY_TOKEN is missing, falls back to seeding mock executive movements.
    """
    logger.info("Starting Executive Movement Ingestion Pipeline...")
    
    # Ensure database tables exist
    await init_db()

    if not USE_REAL_DATA or not APIFY_TOKEN:
        logger.info("USE_REAL_DATA is False or APIFY_TOKEN is missing. Seeding mock executive movements.")
        return await seed_mock_executive_movements()

    # 1. Fetch from Apify using apify-client
    try:
        from apify_client import ApifyClient
        client = ApifyClient(APIFY_TOKEN)
        
        logger.info("Fetching last run dataset from Apify task 'linkedin-decision-makers-scraper'...")
        # Get the last succeeded run client
        last_run_client = client.task("linkedin-decision-makers-scraper").last_run(status="SUCCEEDED")
        if not last_run_client:
            logger.warning("No succeeded run found for task 'linkedin-decision-makers-scraper'. Falling back to mock data.")
            return await seed_mock_executive_movements()
            
        last_run = last_run_client.get()
        if not last_run:
            logger.warning("Failed to retrieve last run metadata. Falling back to mock data.")
            return await seed_mock_executive_movements()
            
        dataset_id = last_run["defaultDatasetId"]
        dataset_items = client.dataset(dataset_id).list_items().items
        logger.info(f"Retrieved {len(dataset_items)} profiles from Apify.")
    except Exception as e:
        logger.error(f"Failed to fetch data from Apify client/task: {e}. Falling back to mock seeding.")
        return await seed_mock_executive_movements()

    if not dataset_items:
        logger.warning("Apify dataset is empty. No executive data retrieved.")
        return 0

    # 2. Get all companies to map company name/ticker -> ticker
    async with async_session_maker() as session:
        stmt = select(CompanyModel)
        res = await session.execute(stmt)
        companies = res.scalars().all()
        
        # Build maps for ticker resolution
        name_to_ticker = {}
        for c in companies:
            if c.legal_name:
                name_to_ticker[c.legal_name.lower().strip()] = c.ticker
            name_to_ticker[c.ticker.upper().strip()] = c.ticker

    # 3. Perform Delta Detection and Save Movements
    movements_detected = 0
    today = datetime.utcnow()
    
    async with async_session_maker() as session:
        with session.no_autoflush:
            for item in dataset_items:
                # Parse fields with fallbacks
                raw_company = item.get("company_ticker") or item.get("ticker") or item.get("current_company_name") or item.get("company_name")
                name = item.get("full_name") or item.get("name")
                new_title = item.get("current_company_title") or item.get("position") or item.get("title")
                
                if not raw_company or not name or not new_title:
                    continue
                    
                # Resolve company ticker
                ticker = name_to_ticker.get(str(raw_company).lower().strip()) or name_to_ticker.get(str(raw_company).upper().strip())
                if not ticker:
                    # If cannot resolve to an existing company in DB, skip or default to raw name
                    continue
    
                # Query the latest record of this executive at this company
                stmt = select(ExecutiveMovement).where(
                    ExecutiveMovement.ticker == ticker,
                    ExecutiveMovement.exec_name == name
                ).order_by(ExecutiveMovement.change_date.desc()).limit(1)
                
                res = await session.execute(stmt)
                latest_movement = res.scalars().first()
                
                if latest_movement:
                    if latest_movement.new_title != new_title:
                        # A change in title has occurred!
                        change = ExecutiveMovement(
                            ticker=ticker,
                            exec_name=name,
                            old_title=latest_movement.new_title,
                            new_title=new_title,
                            change_date=today
                        )
                        session.add(change)
                        movements_detected += 1
                        try:
                            from services.vector_store import VectorStoreService
                            VectorStoreService.index_executive_movement(ticker, name, latest_movement.new_title, new_title, "promotion", today)
                        except Exception as e:
                            logger.error(f"Failed to index executive movement: {e}")
                else:
                    # Baseline entry (first time seeing this executive)
                    change = ExecutiveMovement(
                        ticker=ticker,
                        exec_name=name,
                        old_title=None,
                        new_title=new_title,
                        change_date=today
                    )
                    session.add(change)
                    movements_detected += 1
                    try:
                        from services.vector_store import VectorStoreService
                        VectorStoreService.index_executive_movement(ticker, name, None, new_title, "hire", today)
                    except Exception as e:
                        logger.error(f"Failed to index executive movement: {e}")
                    
            await session.commit()
        
    logger.info(f"Executive Movements tracking completed. Processed and saved {movements_detected} changes.")
    return movements_detected

async def seed_mock_executive_movements() -> int:
    """Seeds realistic mock executive movement data for prominent companies in the DB."""
    # Ensure database is initialized
    await init_db()
    
    mock_movements = [
        {"ticker": "AAPL", "exec_name": "Kevan Parekh", "old_title": "VP of Financial Planning", "new_title": "Chief Financial Officer (CFO)", "days_ago": 12},
        {"ticker": "AAPL", "exec_name": "Luca Maestri", "old_title": "Chief Financial Officer (CFO)", "new_title": "Senior Advisor to CEO", "days_ago": 12},
        {"ticker": "MSFT", "exec_name": "Mustafa Suleyman", "old_title": "CEO of Inflection AI", "new_title": "CEO of Microsoft AI", "days_ago": 45},
        {"ticker": "GOOGL", "exec_name": "Anat Ashkenazi", "old_title": "CFO of Eli Lilly", "new_title": "Chief Financial Officer (CFO) & Senior VP", "days_ago": 30},
        {"ticker": "GOOGL", "exec_name": "Ruth Porat", "old_title": "Chief Financial Officer (CFO)", "new_title": "President & Chief Investment Officer", "days_ago": 30},
        {"ticker": "AMZN", "exec_name": "Doug Herrington", "old_title": "Senior VP of North America Consumer", "new_title": "CEO of Worldwide Amazon Stores", "days_ago": 90},
        {"ticker": "TSLA", "exec_name": "Vaibhav Taneja", "old_title": "Corporate Controller", "new_title": "Chief Financial Officer (CFO) & Chief Accounting Officer", "days_ago": 60}
    ]
    
    records_written = 0
    from datetime import timedelta
    
    async with async_session_maker() as session:
        with session.no_autoflush:
            for move in mock_movements:
                # Check if this movement record already exists to prevent duplicate seeding
                stmt = select(ExecutiveMovement).where(
                    ExecutiveMovement.ticker == move["ticker"],
                    ExecutiveMovement.exec_name == move["exec_name"],
                    ExecutiveMovement.new_title == move["new_title"]
                )
                res = await session.execute(stmt)
                if res.scalars().first():
                    continue
                    
                change_date = datetime.utcnow() - timedelta(days=move["days_ago"])
                db_record = ExecutiveMovement(
                    ticker=move["ticker"],
                    exec_name=move["exec_name"],
                    old_title=move["old_title"],
                    new_title=move["new_title"],
                    change_date=change_date,
                    created_at=change_date
                )
                session.add(db_record)
                records_written += 1
                try:
                    from services.vector_store import VectorStoreService
                    VectorStoreService.index_executive_movement(
                        move["ticker"], move["exec_name"], move["old_title"], move["new_title"], "promotion" if move["old_title"] else "hire", change_date
                    )
                except Exception as e:
                    logger.error(f"Failed to index mock executive movement: {e}")
                
            await session.commit()
        
    logger.info(f"Seeded {records_written} mock executive movements into the database.")
    return records_written

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    asyncio.run(run_executive_movements_ingestion())
