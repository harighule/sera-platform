import os
import sys
import asyncio
import logging
from sqlalchemy import select, func

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import async_session_maker, init_db
from models.commerce import HealthcareMetric
from scripts.fetch_indian_healthcare_api import run_indian_healthcare_api_ingestion

async def test_and_verify():
    print("=== STARTING LIVE INDIAN HEALTHCARE API PIPELINE VERIFICATION ===")
    
    # Check if DATAGOV_IN_API_KEY is configured
    from config import DATAGOV_IN_API_KEY
    api_key = DATAGOV_IN_API_KEY.strip() if DATAGOV_IN_API_KEY else None
    
    if not api_key:
        print("[WARNING] DATAGOV_IN_API_KEY is empty in backend/.env.")
        print("[INFO] Testing ingestion using local CSV factsheet fallback mode...")
    else:
        print(f"[INFO] DATAGOV_IN_API_KEY detected: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
        print("[INFO] Attempting to contact data.gov.in API...")

    await init_db()
    
    # Run ingestion
    try:
        records_processed = await run_indian_healthcare_api_ingestion()
        print(f"[SUCCESS] Ingestion completed. Processed {records_processed} state metrics records.")
    except Exception as e:
        print(f"[ERROR] Ingestion job threw exception: {e}")
        return

    # Check database row counts and sample records
    async with async_session_maker() as session:
        count_res = await session.execute(select(func.count(HealthcareMetric.id)))
        count = count_res.scalar()
        print(f"[INFO] Current HealthcareMetric database row count: {count}")
        
        sample_res = await session.execute(select(HealthcareMetric).limit(5))
        samples = sample_res.scalars().all()
        
        print("[INFO] Sample records in database:")
        for metric in samples:
            print(f"  - Region: {metric.region} | Admissions: {metric.admission_count:,} | Treatment Cost: {metric.avg_total_payment:,.2f} | Drug Claims: {metric.drug_claim_count:,}")
            
    print("\n=== PIPELINE VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(test_and_verify())
