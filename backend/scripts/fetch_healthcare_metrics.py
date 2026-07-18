import os
import sys
import logging
import asyncio
import csv
from datetime import datetime
from sqlalchemy import select

# Add backend directory to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import async_session_maker, init_db
from models.commerce import HealthcareMetric
from config import USE_REAL_DATA

logger = logging.getLogger("sera.healthcare_fetcher")

async def run_healthcare_ingestion() -> int:
    """
    Fetch Indian state-level health data from a local NFHS-5 CSV file
    and upsert them to PostgreSQL/SQLite.
    Falls back to mock Indian seeding in mock mode or on error.
    """
    await init_db()
    logger.info("Starting Indian Healthcare Ingestion Pipeline...")
    
    # Resolve CSV file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "../data/indian_health_data.csv")
    
    if not USE_REAL_DATA or not os.path.exists(csv_path):
        logger.info("USE_REAL_DATA is False or local CSV not found. Seeding mock Indian healthcare metrics.")
        return await seed_mock_healthcare_metrics()

    records_written = 0
    today = datetime.utcnow()
    today_start = datetime(today.year, today.month, today.day)
    
    try:
        rows = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                
        async with async_session_maker() as session:
            for row in rows:
                state = row["state"]
                adm_count = int(row["admissions"])
                avg_payment = float(row["avg_payment"])
                drug_count = int(row["drug_claims"])
                
                # Check for existing record
                stmt = select(HealthcareMetric).where(
                    HealthcareMetric.region == state,
                    HealthcareMetric.measurement_date >= today_start
                )
                res = await session.execute(stmt)
                existing = res.scalars().first()
                
                if existing:
                    existing.admission_count = adm_count
                    existing.avg_total_payment = avg_payment
                    existing.drug_claim_count = drug_count
                    existing.updated_at = today
                else:
                    metric = HealthcareMetric(
                        region=state,
                        admission_count=adm_count,
                        avg_total_payment=avg_payment,
                        drug_claim_count=drug_count,
                        measurement_date=today,
                        created_at=today,
                        updated_at=today
                    )
                    session.add(metric)
                records_written += 1
                
            await session.commit()
            
        logger.info(f"Healthcare Ingestion Pipeline completed successfully. Processed {records_written} Indian state records.")
        return records_written
    except Exception as e:
        logger.error(f"Failed to ingest Indian healthcare CSV: {e}. Falling back to mock data.")
        return await seed_mock_healthcare_metrics()

async def seed_mock_healthcare_metrics() -> int:
    """Seeds fallback/mock Indian healthcare metrics for regional states."""
    await init_db()
    mock_states = [
        {"region": "MH", "admission_count": 1450000, "avg_total_payment": 25000.0, "drug_claim_count": 52000000},
        {"region": "UP", "admission_count": 2100000, "avg_total_payment": 12000.0, "drug_claim_count": 78000000},
        {"region": "TN", "admission_count": 1200000, "avg_total_payment": 22000.0, "drug_claim_count": 48000000},
        {"region": "KA", "admission_count": 1050000, "avg_total_payment": 24000.0, "drug_claim_count": 42000000},
        {"region": "DL", "admission_count": 680000, "avg_total_payment": 28000.0, "drug_claim_count": 25000000},
        {"region": "GJ", "admission_count": 980000, "avg_total_payment": 21000.0, "drug_claim_count": 39000000},
        {"region": "WB", "admission_count": 1350000, "avg_total_payment": 16000.0, "drug_claim_count": 55000000},
        {"region": "AP", "admission_count": 890000, "avg_total_payment": 18000.0, "drug_claim_count": 36000000},
        {"region": "KL", "admission_count": 540000, "avg_total_payment": 26000.0, "drug_claim_count": 31000000},
        {"region": "TS", "admission_count": 720000, "avg_total_payment": 20000.0, "drug_claim_count": 29000000},
        {"region": "MP", "admission_count": 1150000, "avg_total_payment": 14000.0, "drug_claim_count": 41000000},
        {"region": "RJ", "admission_count": 1080000, "avg_total_payment": 15000.0, "drug_claim_count": 38000000},
        {"region": "BR", "admission_count": 1650000, "avg_total_payment": 10000.0, "drug_claim_count": 59000000},
        {"region": "HR", "admission_count": 480000, "avg_total_payment": 23000.0, "drug_claim_count": 18000000},
        {"region": "PB", "admission_count": 510000, "avg_total_payment": 22000.0, "drug_claim_count": 21000000},
        {"region": "OD", "admission_count": 780000, "avg_total_payment": 13000.0, "drug_claim_count": 26000000},
        {"region": "AS", "admission_count": 490000, "avg_total_payment": 12000.0, "drug_claim_count": 15000000},
        {"region": "JH", "admission_count": 620000, "avg_total_payment": 11000.0, "drug_claim_count": 19000000},
        {"region": "CG", "admission_count": 450000, "avg_total_payment": 13000.0, "drug_claim_count": 14000000},
        {"region": "JK", "admission_count": 280000, "avg_total_payment": 15000.0, "drug_claim_count": 9500000},
        {"region": "UT", "admission_count": 220000, "avg_total_payment": 17000.0, "drug_claim_count": 8200000},
        {"region": "HP", "admission_count": 180000, "avg_total_payment": 16000.0, "drug_claim_count": 7100000},
        {"region": "GA", "admission_count": 95000, "avg_total_payment": 24000.0, "drug_claim_count": 4200000},
        {"region": "TR", "admission_count": 88000, "avg_total_payment": 12000.0, "drug_claim_count": 3100000},
        {"region": "ML", "admission_count": 75000, "avg_total_payment": 13000.0, "drug_claim_count": 2800000},
        {"region": "MN", "admission_count": 68000, "avg_total_payment": 14000.0, "drug_claim_count": 2500000},
        {"region": "NL", "admission_count": 52000, "avg_total_payment": 13000.0, "drug_claim_count": 1800000},
        {"region": "AR", "admission_count": 45000, "avg_total_payment": 15000.0, "drug_claim_count": 1500000}
    ]
    
    today = datetime.utcnow()
    today_start = datetime(today.year, today.month, today.day)
    records_written = 0
    
    async with async_session_maker() as session:
        for state_data in mock_states:
            state = state_data["region"]
            
            stmt = select(HealthcareMetric).where(
                HealthcareMetric.region == state,
                HealthcareMetric.measurement_date >= today_start
            )
            res = await session.execute(stmt)
            existing = res.scalars().first()
            
            if existing:
                existing.admission_count = state_data["admission_count"]
                existing.avg_total_payment = state_data["avg_total_payment"]
                existing.drug_claim_count = state_data["drug_claim_count"]
                existing.updated_at = today
            else:
                metric = HealthcareMetric(
                    region=state,
                    admission_count=state_data["admission_count"],
                    avg_total_payment=state_data["avg_total_payment"],
                    drug_claim_count=state_data["drug_claim_count"],
                    measurement_date=today,
                    created_at=today,
                    updated_at=today
                )
                session.add(metric)
            records_written += 1
            
        await session.commit()
        
    logger.info(f"Seeded {records_written} mock healthcare metrics.")
    return records_written

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    asyncio.run(run_healthcare_ingestion())
