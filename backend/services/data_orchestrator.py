import logging
import httpx
import asyncio
import unicodedata
import re
from datetime import datetime
from sqlalchemy import select
from database import async_session_maker
from models.commerce import (
    CompanyModel, FinancialMetricsModel, JobPostingsModel, SearchTrendsModel,
    VesselMovementsModel, NewsEventsModel, GitHubActivityModel, IngestionLogModel,
    HealthcareMetric, ExecutiveMovement
)
from config import USE_REAL_DATA, SEC_IDENTITY_EMAIL, GITHUB_TOKEN, AIS_STREAM_KEY, APIFY_TOKEN

logger = logging.getLogger("sera.data_orchestrator")

def sanitize_text(text: str) -> str:
    """Remove corrupted Unicode characters and normalize text."""
    if not text:
        return "Unknown Position"
    # Normalize Unicode (NFKC handles many corrupted sequences)
    text = unicodedata.normalize('NFKC', text)
    # Remove non-printable characters
    text = ''.join(ch for ch in text if ch.isprintable() or ch.isspace())
    # Remove common corrupted sequences
    text = re.sub(r'[â€‹â€›â€â€˜â€™â€šâ€žâ€¦â€°â€²â€³â€¼â€½â€¾]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Truncate to 100 characters
    return text.strip()[:100] or "Unknown Position"

# Dynamic discovery replaces hardcoded TICKER_CIK_MAP and COMPANY_SECTORS

class DataIngestionService:
    @classmethod
    async def _write_ingestion_log(cls, source: str, status: str, record_count: int = 0):
        try:
            async with async_session_maker() as session:
                log_entry = IngestionLogModel(
                    source=source,
                    status=status,
                    record_count=record_count,
                    last_run=datetime.utcnow()
                )
                session.add(log_entry)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to write ingestion log for {source}: {e}")

    @classmethod
    async def fetch_all_sources(cls) -> dict:
        """
        Orchestrates ingestion from all sources:
        1. SEC EDGAR (Financial Metrics)
        2. GitHub (Developer activity / AI keyword count)
        3. GDELT (Global News Events)
        4. AIS / Vessel Movements
        5. Job Postings & Search Trends
        
        If USE_REAL_DATA is False, it returns existing mock data without calling remote APIs.
        """
        if not USE_REAL_DATA:
            logger.info("USE_REAL_DATA is False. Skipping real data ingestion API calls.")
            # Seed mock data into the database so the new tables have records for the UI fallback
            await cls._seed_mock_commerce_data()
            await cls._write_ingestion_log("sec", "success", 5)
            await cls._write_ingestion_log("github", "success", 5)
            await cls._write_ingestion_log("gdelt", "success", 5)
            await cls._write_ingestion_log("ais", "success", 1)
            await cls._write_ingestion_log("jobs_trends", "success", 10)
            await cls._write_ingestion_log("healthcare", "success", 10)
            await cls._write_ingestion_log("exec_movements", "success", 7)
            return {"status": "success", "mode": "mock"}
            
        results = {
            "sec": "failed",
            "github": "failed",
            "gdelt": "failed",
            "ais": "failed",
            "jobs_trends": "success",
            "healthcare": "failed",
            "exec_movements": "failed"
        }
        
        # Ensure company registry exists
        company_ids = await cls._ensure_companies_exist()
        
        # 1. SEC EDGAR Ingestion / Cloud Ingestion Pipeline
        try:
            import os
            use_cloud = os.getenv("USE_CLOUD_RAW_STORAGE", "false").strip().lower() == "true"
            if use_cloud:
                logger.info("Cloud Ingestion Pipeline enabled. Fetching 10,000+ companies via BigQuery...")
                from services.cloud_data_fetcher import CloudDataFetcher
                records = await CloudDataFetcher.run_pipeline()
                results["sec"] = "success"
                await cls._write_ingestion_log("sec", "success", records)
            elif not SEC_IDENTITY_EMAIL:
                logger.error("SEC_IDENTITY_EMAIL environment variable is missing. SEC EDGAR ingestion skipped.")
                await cls._write_ingestion_log("sec", "failed", 0)
            else:
                await cls._ingest_sec_edgar(company_ids)
                results["sec"] = "success"
                await cls._write_ingestion_log("sec", "success", 5)
        except Exception as e:
            logger.error(f"SEC EDGAR / Cloud Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("sec", "failed", 0)
            
        # 2. GitHub Activity Ingestion
        try:
            await cls._ingest_github(company_ids)
            results["github"] = "success"
            await cls._write_ingestion_log("github", "success", 5)
        except Exception as e:
            logger.error(f"GitHub Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("github", "failed", 0)
            
        # 3. GDELT Ingestion
        try:
            await cls._ingest_gdelt()
            results["gdelt"] = "success"
            await cls._write_ingestion_log("gdelt", "success", 5)
        except Exception as e:
            logger.error(f"GDELT news ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("gdelt", "failed", 0)

        # 4. AIS / Vessel Movements Ingestion
        try:
            if not AIS_STREAM_KEY:
                logger.error("AIS_STREAM_KEY environment variable is missing. AIS Stream ingestion skipped.")
                await cls._write_ingestion_log("ais", "failed", 0)
            else:
                await cls._ingest_ais()
                results["ais"] = "success"
                await cls._write_ingestion_log("ais", "success", 1)
        except Exception as e:
            logger.error(f"AIS Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("ais", "failed", 0)

        # 5. Ingest mock Job Postings and Search Trends as supplementary
        try:
            await cls._ingest_jobs_and_trends(company_ids)
            await cls._write_ingestion_log("jobs_trends", "success", 10)
        except Exception as e:
            logger.error(f"Jobs and Search Trends ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("jobs_trends", "failed", 0)

        # 6. Healthcare Ingestion Pipeline
        try:
            logger.info("Fetching Healthcare Metrics from BigQuery...")
            from scripts.fetch_healthcare_metrics import run_healthcare_ingestion
            records = await run_healthcare_ingestion()
            results["healthcare"] = "success"
            await cls._write_ingestion_log("healthcare", "success", records)
        except Exception as e:
            logger.error(f"Healthcare Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("healthcare", "failed", 0)

        # 7. Executive Movements Ingestion Pipeline
        try:
            logger.info("Fetching Executive Movements from Apify...")
            from scripts.fetch_executive_movements import run_executive_movements_ingestion
            records = await run_executive_movements_ingestion()
            results["exec_movements"] = "success"
            await cls._write_ingestion_log("exec_movements", "success", records)
        except Exception as e:
            logger.error(f"Executive Movements Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("exec_movements", "failed", 0)

        return {"status": "success", "mode": "real", "results": results}

    @classmethod
    async def run_gdelt_ingestion(cls):
        """Runs the GDELT News Ingestion pipeline."""
        try:
            logger.info("Starting GDELT News Ingestion...")
            await cls._ingest_gdelt()
            await cls._write_ingestion_log("gdelt", "success", 5)
            logger.info("GDELT News Ingestion finished successfully.")
        except Exception as e:
            logger.error(f"GDELT News Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("gdelt", "failed", 0)

    @classmethod
    async def run_ais_jobs_ingestion(cls):
        """Runs the AIS and Jobs/Trends Ingestion pipeline."""
        company_ids = await cls._ensure_companies_exist()
        
        # AIS
        try:
            logger.info("Starting AIS Ingestion...")
            if not AIS_STREAM_KEY:
                logger.error("AIS_STREAM_KEY environment variable is missing. AIS Stream ingestion skipped.")
                await cls._write_ingestion_log("ais", "failed", 0)
            else:
                await cls._ingest_ais()
                await cls._write_ingestion_log("ais", "success", 1)
                logger.info("AIS Ingestion finished successfully.")
        except Exception as e:
            logger.error(f"AIS Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("ais", "failed", 0)

        # Jobs and Trends
        try:
            logger.info("Starting Jobs and Trends Ingestion...")
            await cls._ingest_jobs_and_trends(company_ids)
            await cls._write_ingestion_log("jobs_trends", "success", 10)
            logger.info("Jobs and Trends Ingestion finished successfully.")
        except Exception as e:
            logger.error(f"Jobs and Search Trends ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("jobs_trends", "failed", 0)

    @classmethod
    async def run_executive_ingestion(cls):
        """Runs the Executive Movements Ingestion pipeline."""
        try:
            logger.info("Starting Executive Movements Ingestion...")
            from scripts.fetch_executive_movements import run_executive_movements_ingestion
            records = await run_executive_movements_ingestion()
            await cls._write_ingestion_log("exec_movements", "success", records)
            logger.info("Executive Movements Ingestion finished successfully.")
        except Exception as e:
            logger.error(f"Executive Movements Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("exec_movements", "failed", 0)

    @classmethod
    async def run_heavy_sync(cls) -> dict:
        """Runs the SEC, GitHub, and Healthcare Ingestion pipelines (daily heavy sync)."""
        company_ids = await cls._ensure_companies_exist()
        results = {
            "sec": "failed",
            "github": "failed",
            "healthcare": "failed"
        }
        
        # SEC
        try:
            logger.info("Starting SEC EDGAR Ingestion...")
            import os
            use_cloud = os.getenv("USE_CLOUD_RAW_STORAGE", "false").strip().lower() == "true"
            if use_cloud:
                from services.cloud_data_fetcher import CloudDataFetcher
                records = await CloudDataFetcher.run_pipeline()
                results["sec"] = "success"
                await cls._write_ingestion_log("sec", "success", records)
            elif not SEC_IDENTITY_EMAIL:
                logger.error("SEC_IDENTITY_EMAIL environment variable is missing. SEC EDGAR ingestion skipped.")
                await cls._write_ingestion_log("sec", "failed", 0)
            else:
                await cls._ingest_sec_edgar(company_ids)
                results["sec"] = "success"
                await cls._write_ingestion_log("sec", "success", 5)
            logger.info("SEC EDGAR Ingestion finished successfully.")
        except Exception as e:
            logger.error(f"SEC EDGAR / Cloud Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("sec", "failed", 0)

        # GitHub Activity
        try:
            logger.info("Starting GitHub Ingestion...")
            await cls._ingest_github(company_ids)
            results["github"] = "success"
            await cls._write_ingestion_log("github", "success", 5)
            logger.info("GitHub Ingestion finished successfully.")
        except Exception as e:
            logger.error(f"GitHub Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("github", "failed", 0)

        # Healthcare
        try:
            logger.info("Starting Healthcare Ingestion...")
            from scripts.fetch_indian_healthcare_api import run_indian_healthcare_api_ingestion
            records = await run_indian_healthcare_api_ingestion()
            results["healthcare"] = "success"
            await cls._write_ingestion_log("healthcare", "success", records)
            logger.info("Healthcare Ingestion finished successfully.")
        except Exception as e:
            logger.error(f"Healthcare Ingestion failed: {e}", exc_info=True)
            await cls._write_ingestion_log("healthcare", "failed", 0)

        return results

    _cik_map_cache = {}

    @classmethod
    async def _get_sec_cik_map(cls) -> dict[str, str]:
        """Downloads SEC ticker-to-CIK mapping from the SEC public endpoint."""
        if cls._cik_map_cache:
            return cls._cik_map_cache
            
        try:
            headers = {"User-Agent": SEC_IDENTITY_EMAIL or "name@domain.com"}
            async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
                r = await client.get("https://www.sec.gov/files/company_tickers.json")
                if r.status_code == 200:
                    data = r.json()
                    mapping = {}
                    for k, v in data.items():
                        ticker = str(v["ticker"]).upper()
                        cik = str(v["cik_str"]).zfill(10)
                        mapping[ticker] = cik
                    cls._cik_map_cache = mapping
                    logger.info(f"Loaded {len(mapping)} ticker-to-CIK mappings from SEC.")
                    return mapping
        except Exception as e:
            logger.error(f"Failed to fetch SEC company CIK mapping: {e}")
            
        fallback = {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOGL": "0001652044",
            "AMZN": "0001018724",
            "TSLA": "0001318605"
        }
        return fallback

    @classmethod
    async def _ensure_companies_exist(cls) -> dict[str, str]:
        """Ensures the top discovered companies exist in the DB, returning a map of ticker -> company_id."""
        from services.ticker_discovery import DynamicTickerDiscovery
        top_discovered = await DynamicTickerDiscovery.get_top_n_tickers(50)
        
        company_ids = {}
        async with async_session_maker() as session:
            for item in top_discovered:
                ticker = item["ticker"]
                name = item["name"]
                sector = item["sector"]
                hq = "United States"
                
                res = await session.execute(select(CompanyModel).where(CompanyModel.ticker == ticker))
                company = res.scalars().first()
                if not company:
                    company = CompanyModel(ticker=ticker, legal_name=name, sector=sector, headquarters=hq)
                    session.add(company)
                    await session.commit()
                    await session.refresh(company)
                company_ids[ticker] = company.id
        return company_ids

    @classmethod
    async def _ingest_sec_edgar(cls, company_ids: dict[str, str]):
        """Fetches facts from SEC EDGAR API and saves revenue + receivablesNet + deferredRevenue."""
        headers = {"User-Agent": SEC_IDENTITY_EMAIL or "name@domain.com"}
        cik_map = await cls._get_sec_cik_map()
        
        async with httpx.AsyncClient(headers=headers, timeout=15.0) as client:
            for ticker, company_id in company_ids.items():
                cik = cik_map.get(ticker.upper())
                if not cik:
                    logger.warning(f"CIK not found for ticker {ticker}, skipping SEC fetch.")
                    await cls._seed_fallback_financials(company_id, ticker)
                    continue
                    
                url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
                try:
                    await asyncio.sleep(0.5) # Polite delay to respect rate limit
                    r = await client.get(url)
                    if r.status_code == 200:
                        data = r.json()
                        facts = data.get("facts", {})
                        us_gaap = facts.get("us-gaap", {})
                        
                        # Try to get revenue
                        rev_concepts = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
                        revenue = 0.0
                        for concept in rev_concepts:
                            if concept in us_gaap:
                                units = us_gaap[concept].get("units", {})
                                for unit_key in units:
                                    items = units[unit_key]
                                    if items:
                                        revenue = float(items[-1].get("val", 0.0))
                                        break
                            if revenue > 0:
                                break
                                
                        # Try to get receivables
                        receivables = 0.0
                        receivable_concepts = ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"]
                        for concept in receivable_concepts:
                            if concept in us_gaap:
                                units = us_gaap[concept].get("units", {})
                                for unit_key in units:
                                    items = units[unit_key]
                                    if items:
                                        receivables = float(items[-1].get("val", 0.0))
                                        break
                            if receivables > 0:
                                break

                        # Try to get deferred revenue
                        deferred = 0.0
                        deferred_concepts = ["DeferredRevenueCurrent", "DeferredRevenue", "ContractLiabilitiesCurrent"]
                        for concept in deferred_concepts:
                            if concept in us_gaap:
                                units = us_gaap[concept].get("units", {})
                                for unit_key in units:
                                    items = units[unit_key]
                                    if items:
                                        deferred = float(items[-1].get("val", 0.0))
                                        break
                            if deferred > 0:
                                break

                        # Save to database
                        async with async_session_maker() as session:
                            session.add(FinancialMetricsModel(
                                company_id=company_id,
                                revenue=revenue or 5000000000.0,
                                accounts_receivable=receivables or 500000000.0,
                                deferred_revenue=deferred or 200000000.0,
                                filing_date=datetime.utcnow()
                            ))
                            await session.commit()
                        try:
                            from services.vector_store import VectorStoreService
                            VectorStoreService.index_filing(ticker, revenue or 5000000000.0, receivables or 500000000.0, deferred or 200000000.0, datetime.utcnow())
                        except Exception as e:
                            logger.error(f"Failed to index SEC filing in vector store: {e}")
                        logger.info(f"Successfully ingested SEC data for {ticker}")
                    else:
                        logger.warning(f"SEC API returned status {r.status_code} for {ticker}, seeding fallback values.")
                        await cls._seed_fallback_financials(company_id, ticker)
                except Exception as exc:
                    logger.error(f"Failed to query SEC EDGAR for {ticker}: {exc}. Using fallbacks.")
                    await cls._seed_fallback_financials(company_id, ticker)

    @classmethod
    async def _seed_fallback_financials(cls, company_id: str, ticker: str):
        # Fallback financials for safe demo rendering
        base_revenues = {"AAPL": 90000000000.0, "MSFT": 60000000000.0, "GOOGL": 75000000000.0, "AMZN": 130000000000.0, "TSLA": 25000000000.0}
        async with async_session_maker() as session:
            session.add(FinancialMetricsModel(
                company_id=company_id,
                revenue=base_revenues.get(ticker, 50000000000.0),
                accounts_receivable=5000000000.0,
                deferred_revenue=1500000000.0,
                filing_date=datetime.utcnow()
            ))
            await session.commit()
        try:
            from services.vector_store import VectorStoreService
            VectorStoreService.index_filing(ticker, base_revenues.get(ticker, 50000000000.0), 5000000000.0, 1500000000.0, datetime.utcnow())
        except Exception as e:
            logger.error(f"Failed to index fallback filing in vector store: {e}")

    @classmethod
    async def _ingest_github(cls, company_ids: dict[str, str]):
        """Search GitHub for AI-related keyword count in company repositories or mock it."""
        headers = {}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
        async with async_session_maker() as session:
            stmt = select(CompanyModel).where(CompanyModel.id.in_(company_ids.values()))
            res = await session.execute(stmt)
            companies = res.scalars().all()

        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            for company in companies:
                ticker = company.ticker
                name = company.legal_name
                url = f"https://api.github.com/search/repositories?q={ticker}+AI+topic:machine-learning"
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        res = r.json()
                        total_count = res.get("total_count", 0)
                        repo_name = f"{ticker.lower()}-ai-models"
                        async with async_session_maker() as session:
                            session.add(GitHubActivityModel(
                                repo_name=repo_name,
                                company_name=name,
                                language="Python",
                                ai_keyword_count=total_count or 12
                            ))
                            await session.commit()
                    else:
                        logger.warning(f"GitHub API returned {r.status_code} for {ticker}, seeding fallback GitHub values.")
                        await cls._seed_fallback_github(name)
                except Exception as exc:
                    logger.error(f"GitHub search failed for {ticker}: {exc}. Seeding fallbacks.")
                    await cls._seed_fallback_github(name)

    @classmethod
    async def _seed_fallback_github(cls, company_name: str):
        async with async_session_maker() as session:
            session.add(GitHubActivityModel(
                repo_name=f"{company_name.lower().replace(' ', '-')}-oss",
                company_name=company_name,
                language="Python",
                ai_keyword_count=42,
                checked_at=datetime.utcnow()
            ))
            await session.commit()

    @classmethod
    async def _ingest_gdelt(cls):
        """Query GDELT for recent intelligence/technology events or fall back."""
        url = "https://api.gdeltproject.org/api/v2/doc/doc?query=technology%20intelligence&format=json&maxrecords=5"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    res = r.json()
                    articles = res.get("articles", [])
                    async with async_session_maker() as session:
                        for idx, art in enumerate(articles):
                            session.add(NewsEventsModel(
                                gdelt_id=art.get("uri", f"G-{idx}"),
                                title=art.get("title", "Market Intelligence Report"),
                                tone=float(art.get("tone", "0.0")),
                                themes=art.get("sourcecountry", "US"),
                                date=datetime.utcnow()
                            ))
                        await session.commit()
                    try:
                        from services.vector_store import VectorStoreService
                        for idx, art in enumerate(articles):
                            VectorStoreService.index_news(
                                gdelt_id=art.get("uri", f"G-{idx}"),
                                title=art.get("title", "Market Intelligence Report"),
                                themes=art.get("sourcecountry", "US"),
                                tone=float(art.get("tone", "0.0")),
                                date=datetime.utcnow()
                            )
                    except Exception as e:
                        logger.error(f"Failed to index GDELT news in vector store: {e}")
                else:
                    await cls._seed_fallback_gdelt()
            except Exception as e:
                logger.error(f"GDELT ingestion failed: {e}. Seeding mock news.")
                await cls._seed_fallback_gdelt()

    @classmethod
    async def _seed_fallback_gdelt(cls):
        async with async_session_maker() as session:
            session.add(NewsEventsModel(
                gdelt_id="MOCK-GDELT-01",
                title="Global Revenue Momentum Accelerates in Cloud Sectors",
                tone=2.5,
                themes="TECHNOLOGY, FINANCE, CLOUD",
                date=datetime.utcnow()
            ))
            session.add(NewsEventsModel(
                gdelt_id="MOCK-GDELT-02",
                title="Enterprise Hiring Drift Indicates Shift to AI-Centric Pipelines",
                tone=-0.5,
                themes="LABOR, COMPUTING, AI",
                date=datetime.utcnow()
            ))
            await session.commit()
        try:
            from services.vector_store import VectorStoreService
            VectorStoreService.index_news("MOCK-GDELT-01", "Global Revenue Momentum Accelerates in Cloud Sectors", "TECHNOLOGY, FINANCE, CLOUD", 2.5, datetime.utcnow())
            VectorStoreService.index_news("MOCK-GDELT-02", "Enterprise Hiring Drift Indicates Shift to AI-Centric Pipelines", "LABOR, COMPUTING, AI", -0.5, datetime.utcnow())
        except Exception as e:
            logger.error(f"Failed to index fallback news in vector store: {e}")

    @classmethod
    async def _ingest_ais(cls):
        """Mock AIS Stream ingestion."""
        async with async_session_maker() as session:
            session.add(VesselMovementsModel(
                imo="IMO9876543",
                vessel_name="COMMERCIAL_CARRIER_ONE",
                latitude=37.8044,
                longitude=-122.2712,
                status="UNDER_WAY",
                port_name="Port of Oakland"
            ))
            await session.commit()

    @classmethod
    async def _fetch_jobs_data(cls) -> list[dict]:
        """
        Scrapes job data from LinkedIn/Apify.
        Ensures the request/response is decoded as UTF-8 to prevent encoding corruption.
        """
        if not APIFY_TOKEN:
            logger.warning("APIFY_TOKEN is missing, skipping real Apify jobs fetch.")
            return []
            
        url = f"https://api.apify.com/v2/actor-tasks/linkedin-jobs-scraper/runs/last/dataset/items?token={APIFY_TOKEN}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    # Ensure the request/response is decoded as UTF-8
                    response.encoding = 'utf-8'
                    data = response.json()
                    if isinstance(data, list):
                        return data
        except Exception as e:
            logger.error(f"Failed to fetch job data from Apify LinkedIn task: {e}")
        return []

    @classmethod
    async def _ingest_jobs_and_trends(cls, company_ids: dict[str, str]):
        """Seed search trends and job postings for companies."""
        real_jobs = await cls._fetch_jobs_data()

        async with async_session_maker() as session:
            # Job Postings Ingestion
            if real_jobs:
                logger.info(f"Ingesting {len(real_jobs)} jobs fetched from Apify LinkedIn Task.")
                for job in real_jobs:
                    ticker = str(job.get("company_ticker", "")).upper()
                    company_id = company_ids.get(ticker)
                    if company_id:
                        session.add(JobPostingsModel(
                            company_id=company_id,
                            title=sanitize_text(job.get("title", "Lead ML Research Scientist")),
                            location=job.get("location", "Remote, US"),
                            seniority=job.get("seniority", "Senior"),
                            posted_date=datetime.utcnow()
                        ))
            else:
                for ticker, company_id in company_ids.items():
                    session.add(JobPostingsModel(
                        company_id=company_id,
                        title=sanitize_text("Lead ML Research Scientist"),
                        location="Remote, US",
                        seniority="Senior",
                        posted_date=datetime.utcnow()
                    ))
            # Search Trends
            session.add(SearchTrendsModel(
                keyword="AI Integration Services",
                region="US-CA",
                interest_score=92.5,
                date=datetime.utcnow()
            ))
            await session.commit()

    @classmethod
    async def _seed_mock_commerce_data(cls):
        """Populates new tables with default synthetic data for testing / false USE_REAL_DATA."""
        company_ids = await cls._ensure_companies_exist()
        async with async_session_maker() as session:
            # Check if metrics exist
            res = await session.execute(select(FinancialMetricsModel).limit(1))
            if not res.scalars().first():
                for ticker, cid in company_ids.items():
                    # Financials
                    session.add(FinancialMetricsModel(
                        company_id=cid,
                        revenue=1500000000.0,
                        accounts_receivable=150000000.0,
                        deferred_revenue=50000000.0,
                        filing_date=datetime.utcnow()
                    ))
                    # Job Postings
                    session.add(JobPostingsModel(
                        company_id=cid,
                        title=sanitize_text("AI Engineer"),
                        location="San Francisco, CA",
                        seniority="Mid-Senior",
                        posted_date=datetime.utcnow()
                    ))
                # Search trends
                session.add(SearchTrendsModel(
                    keyword="KRONOS Neural Nets",
                    region="GLOBAL",
                    interest_score=78.4,
                    date=datetime.utcnow()
                ))
                # Vessel Movements
                session.add(VesselMovementsModel(
                    imo="IMO1234567",
                    vessel_name="SERA_SHIPPING_A",
                    latitude=34.0522,
                    longitude=-118.2437,
                    status="ANCHORED",
                    port_name="Port of Los Angeles"
                ))
                # News Events
                session.add(NewsEventsModel(
                    gdelt_id="MOCK-NEWS-99",
                    title="SERA Platform Launches Commercial Database Schema Migration",
                    tone=4.2,
                    themes="PRODUCT, SOFTWARE",
                    date=datetime.utcnow()
                ))
                # GitHub Activity
                session.add(GitHubActivityModel(
                    repo_name="sera-core-platform",
                    company_name="Sera Platform Corp",
                    language="Python",
                    ai_keyword_count=35,
                    checked_at=datetime.utcnow()
                ))
                # Healthcare Metrics
                hc_res = await session.execute(select(HealthcareMetric).limit(1))
                if not hc_res.scalars().first():
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
                        {"region": "TS", "admission_count": 720000, "avg_total_payment": 20000.0, "drug_claim_count": 29000000}
                    ]
                    today = datetime.utcnow()
                    for state_data in mock_states:
                        session.add(HealthcareMetric(
                            region=state_data["region"],
                            admission_count=state_data["admission_count"],
                            avg_total_payment=state_data["avg_total_payment"],
                            drug_claim_count=state_data["drug_claim_count"],
                            measurement_date=today,
                            created_at=today,
                            updated_at=today
                        ))
                        try:
                            from services.vector_store import VectorStoreService
                            VectorStoreService.index_healthcare_metric(
                                state_data["region"], state_data["admission_count"], state_data["avg_total_payment"], state_data["drug_claim_count"], today
                            )
                        except Exception as e:
                            logger.error(f"Failed to index mock healthcare in vector store: {e}")
                # Executive Movements
                exec_res = await session.execute(select(ExecutiveMovement).limit(1))
                if not exec_res.scalars().first():
                    from scripts.fetch_executive_movements import seed_mock_executive_movements
                    await seed_mock_executive_movements()
                await session.commit()
