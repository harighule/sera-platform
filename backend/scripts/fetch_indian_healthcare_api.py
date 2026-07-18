import os
import sys
import logging
import asyncio
import csv
import httpx
from datetime import datetime
from sqlalchemy import select

# Add backend directory to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import async_session_maker, init_db
from models.commerce import HealthcareMetric
from config import DATAGOV_IN_API_KEY, USE_REAL_DATA

logger = logging.getLogger("sera.indian_healthcare_api")

STATE_NAME_TO_CODE = {
    "andhra pradesh": "AP",
    "arunachal pradesh": "AR",
    "assam": "AS",
    "bihar": "BR",
    "chhattisgarh": "CG",
    "goa": "GA",
    "gujarat": "GJ",
    "haryana": "HR",
    "himachal pradesh": "HP",
    "jammu & kashmir": "JK",
    "jammu and kashmir": "JK",
    "jharkhand": "JH",
    "karnataka": "KA",
    "kerala": "KL",
    "madhya pradesh": "MP",
    "maharashtra": "MH",
    "manipur": "MN",
    "meghalaya": "ML",
    "mizoram": "MZ",
    "nagaland": "NL",
    "odisha": "OD",
    "orissa": "OD",
    "punjab": "PB",
    "rajasthan": "RJ",
    "sikkim": "SK",
    "tamil nadu": "TN",
    "telangana": "TS",
    "tripura": "TR",
    "uttar pradesh": "UP",
    "uttarakhand": "UT",
    "west bengal": "WB",
    "delhi": "DL"
}

# Free resource containing state-wise health parameters on data.gov.in
# We will query this default Resource ID. If the key is not active, it falls back to CSV.
DEFAULT_RESOURCE_ID = "3b5bdf47-dc9d-4876-b9cd-5cbf478f3706"

async def run_indian_healthcare_api_ingestion() -> int:
    """
    Fetch Indian state-level health data from live data.gov.in API
    and upsert them to PostgreSQL/SQLite.
    If DATAGOV_IN_API_KEY is not configured or queries fail, falls back to CSV/Mock data.
    """
    await init_db()
    
    # Resolve local CSV file path (fallback)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "../data/indian_health_data.csv")

    api_key = DATAGOV_IN_API_KEY.strip() if DATAGOV_IN_API_KEY else None
    
    if not api_key:
        logger.warning("DATAGOV_IN_API_KEY is not configured. Falling back to local Indian factsheet CSV ingestion.")
        return await ingest_from_csv(csv_path)

    logger.info("Connecting to data.gov.in Indian Healthcare API...")
    url = f"https://api.data.gov.in/resource/{DEFAULT_RESOURCE_ID}"
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 100
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                records = data.get("records", [])
                if not records:
                    logger.warning("No records returned from data.gov.in API. Falling back to CSV.")
                    return await ingest_from_csv(csv_path)
                
                records_written = await process_and_save_api_records(records)
                logger.info(f"Successfully processed {records_written} live records from data.gov.in.")
                return records_written
            else:
                logger.warning(f"data.gov.in API returned HTTP {r.status_code}: {r.text[:200]}. Falling back to CSV.")
                return await ingest_from_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to query data.gov.in API: {e}. Falling back to CSV.")
        return await ingest_from_csv(csv_path)

async def process_and_save_api_records(records: list[dict]) -> int:
    """Processes government records, normalizes state codes, and saves to database."""
    today = datetime.utcnow()
    today_start = datetime(today.year, today.month, today.day)
    records_written = 0
    
    async with async_session_maker() as session:
        for rec in records:
            # Detect state name field (can vary depending on dataset index)
            state_raw = rec.get("state", rec.get("state_ut", rec.get("state_or_ut", "")))
            if not state_raw:
                continue
                
            state_clean = str(state_raw).strip().lower()
            state_code = STATE_NAME_TO_CODE.get(state_clean)
            
            if not state_code:
                # If we can't resolve code, skip or use raw name if 2 letters
                if len(state_clean) == 2:
                    state_code = state_clean.upper()
                else:
                    logger.debug(f"Skipping unmapped state/UT name: {state_raw}")
                    continue
            
            # Extract metrics dynamically or fall back to defaults
            # e.g. admission_count, treatment costs, etc.
            adm_count = int(float(rec.get("hospitalizations", rec.get("admissions", rec.get("admission_count", 500000)))))
            avg_payment = float(rec.get("average_cost", rec.get("avg_payment", rec.get("treatment_cost", 15000.0))))
            drug_count = int(float(rec.get("drug_claims", rec.get("claims", rec.get("drug_claim_count", 25000000)))))
            
            stmt = select(HealthcareMetric).where(
                HealthcareMetric.region == state_code,
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
                    region=state_code,
                    admission_count=adm_count,
                    avg_total_payment=avg_payment,
                    drug_claim_count=drug_count,
                    measurement_date=today,
                    created_at=today,
                    updated_at=today
                )
                session.add(metric)
            try:
                from services.vector_store import VectorStoreService
                VectorStoreService.index_healthcare_metric(state_code, adm_count, avg_payment, drug_count, today)
            except Exception as e:
                logger.error(f"Failed to index healthcare metric: {e}")
            records_written += 1
            
        await session.commit()
    return records_written

async def ingest_from_csv(csv_path: str) -> int:
    """Reads from local NFHS-5 factsheet CSV as a fallback option."""
    if not os.path.exists(csv_path):
        logger.error(f"Fallback CSV not found at: {csv_path}. Seeding default Indian metrics.")
        return await seed_default_indian_metrics()
        
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
                try:
                    from services.vector_store import VectorStoreService
                    VectorStoreService.index_healthcare_metric(state, adm_count, avg_payment, drug_count, today)
                except Exception as e:
                    logger.error(f"Failed to index healthcare metric fallback: {e}")
                records_written += 1
            await session.commit()
        logger.info(f"Processed {records_written} Indian states from local CSV fallback.")
        return records_written
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return await seed_default_indian_metrics()

async def seed_default_indian_metrics() -> int:
    """Seeds default mock Indian metrics when all other ingestion options fail."""
    mock_states = [
        {"region": "MH", "admission_count": 1450000, "avg_total_payment": 25000.0, "drug_claim_count": 52000000},
        {"region": "UP", "admission_count": 2100000, "avg_total_payment": 12000.0, "drug_claim_count": 78000000},
        {"region": "TN", "admission_count": 1200000, "avg_total_payment": 22000.0, "drug_claim_count": 48000000},
        {"region": "KA", "admission_count": 1050000, "avg_total_payment": 24000.0, "drug_claim_count": 42000000},
        {"region": "DL", "admission_count": 680000, "avg_total_payment": 28000.0, "drug_claim_count": 25000000}
    ]
    today = datetime.utcnow()
    today_start = datetime(today.year, today.month, today.day)
    records_written = 0
    
    async with async_session_maker() as session:
        for sd in mock_states:
            state = sd["region"]
            stmt = select(HealthcareMetric).where(
                HealthcareMetric.region == state,
                HealthcareMetric.measurement_date >= today_start
            )
            res = await session.execute(stmt)
            existing = res.scalars().first()
            if existing:
                existing.admission_count = sd["admission_count"]
                existing.avg_total_payment = sd["avg_total_payment"]
                existing.drug_claim_count = sd["drug_claim_count"]
                existing.updated_at = today
            else:
                session.add(HealthcareMetric(
                    region=state,
                    admission_count=sd["admission_count"],
                    avg_total_payment=sd["avg_total_payment"],
                    drug_claim_count=sd["drug_claim_count"],
                    measurement_date=today,
                    created_at=today,
                    updated_at=today
                ))
            try:
                from services.vector_store import VectorStoreService
                VectorStoreService.index_healthcare_metric(state, sd["admission_count"], sd["avg_total_payment"], sd["drug_claim_count"], today)
            except Exception as e:
                logger.error(f"Failed to index healthcare metric fallback seed: {e}")
            records_written += 1
        await session.commit()
    logger.info(f"Seeded {records_written} fallback Indian metrics.")
    return records_written

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(run_indian_healthcare_api_ingestion())
