import asyncio
import os
import sys

# Ensure backend folder is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from database import init_db, async_session_maker
from services.data_orchestrator import DataIngestionService
from models.commerce import CompanyModel, FinancialMetricsModel, NewsEventsModel
from sqlalchemy import select, func

async def main():
    print("Initialising Database tables...")
    await init_db()
    
    print("Running DataIngestionService.fetch_all_sources()...")
    try:
        res = await DataIngestionService.fetch_all_sources()
        print("Ingestion execution finished cleanly.")
        print("Response payload:", res)
    except Exception as e:
        print(f"CRITICAL ERROR during ingestion: {e}")
        return

    print("\n--- Ingestion Statistics (Database Validation) ---")
    async with async_session_maker() as session:
        comp_count = (await session.execute(select(func.count()).select_from(CompanyModel))).scalar() or 0
        fin_count = (await session.execute(select(func.count()).select_from(FinancialMetricsModel))).scalar() or 0
        news_count = (await session.execute(select(func.count()).select_from(NewsEventsModel))).scalar() or 0
        
        print(f"✅ Success: {comp_count} Companies, {fin_count} Financial Records, {news_count} News Events")
        
        # Display sample companies
        res_comp = await session.execute(select(CompanyModel).limit(5))
        companies = res_comp.scalars().all()
        print("\nRegistered Companies:")
        for c in companies:
            print(f"  - [{c.ticker}] {c.legal_name} (Sector: {c.sector}, HQ: {c.headquarters})")
            
        # Display sample financials
        res_fin = await session.execute(select(FinancialMetricsModel).limit(5))
        financials = res_fin.scalars().all()
        print("\nFinancial Records:")
        for f in financials:
            print(f"  - Company ID: {f.company_id} | Revenue: ${f.revenue:,.2f} | Deferred Revenue: ${f.deferred_revenue:,.2f}")

        # Display sample news
        res_news = await session.execute(select(NewsEventsModel).limit(5))
        news = res_news.scalars().all()
        print("\nNews Events:")
        for n in news:
            print(f"  - Title: {n.title} | Tone: {n.tone} | Themes: {n.themes}")

if __name__ == "__main__":
    asyncio.run(main())
