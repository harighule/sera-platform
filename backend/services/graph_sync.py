import logging
import unicodedata
import re
from sqlalchemy import select
from neo4j import GraphDatabase
from database import async_session_maker
from models.commerce import (
    CompanyModel, NewsEventsModel, JobPostingsModel, VesselMovementsModel
)
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger("sera.graph_sync")

def sanitize_text(text: str) -> str:
    """Remove corrupted Unicode characters and normalize text."""
    if not text:
        return "Unknown"
    # Normalize Unicode (NFKC handles many corrupted sequences)
    text = unicodedata.normalize('NFKC', text)
    # Remove non-printable characters
    text = ''.join(ch for ch in text if ch.isprintable() or ch.isspace())
    # Remove common corrupted sequences
    text = re.sub(r'[â€‹â€›â€â€˜â€™â€šâ€žâ€¦â€°â€²â€³â€¼â€½â€¾]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Truncate to 100 characters
    return text.strip()[:100] or "Unknown"

class GraphSyncService:
    _driver = None

    @classmethod
    def get_driver(cls):
        """Initializes and returns the Neo4j Bolt driver driver instance."""
        if not cls._driver:
            if not NEO4J_URI or not NEO4J_PASSWORD:
                logger.error("Neo4j environment configurations (NEO4J_URI, NEO4J_PASSWORD) are missing.")
                return None
            try:
                cls._driver = GraphDatabase.driver(
                    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                logger.info("Successfully established connection driver to Neo4j database.")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j Bolt interface: {e}")
                return None
        return cls._driver

    @classmethod
    async def sync_all_entities(cls) -> dict:
        """
        Queries all records from PostgreSQL and synchronizes them to Neo4j.
        Returns a dict of synchronized element counts.
        """
        driver = cls.get_driver()
        if not driver:
            raise ConnectionError("Neo4j database driver is not connected.")

        companies_synced = 0
        jobs_synced = 0
        news_synced = 0
        vessels_synced = 0
        relations_created = 0

        # 1. Fetch data from PostgreSQL
        async with async_session_maker() as session:
            # Query Companies
            comp_stmt = select(CompanyModel)
            comp_res = await session.execute(comp_stmt)
            postgres_companies = comp_res.scalars().all()

            # Query Job Postings
            job_stmt = select(JobPostingsModel, CompanyModel.ticker).join(CompanyModel)
            job_res = await session.execute(job_stmt)
            postgres_jobs = job_res.all() # returns list of tuples: (JobPostingsModel, ticker)

            # Query News Events
            news_stmt = select(NewsEventsModel)
            news_res = await session.execute(news_stmt)
            postgres_news = news_res.scalars().all()

            # Query Vessel Movements
            vessel_stmt = select(VesselMovementsModel)
            vessel_res = await session.execute(vessel_stmt)
            postgres_vessels = vessel_res.scalars().all()

        # 2. Write/Merge properties to Neo4j using transactions
        with driver.session() as neo_session:
            # A. Sync Companies
            for c in postgres_companies:
                neo_session.run(
                    """
                    MERGE (comp:Company {ticker: $ticker})
                    SET comp.legal_name = $legal_name,
                        comp.sector = $sector,
                        comp.headquarters = $headquarters
                    """,
                    ticker=c.ticker,
                    legal_name=c.legal_name,
                    sector=c.sector or "Technology",
                    headquarters=c.headquarters or "USA"
                )
                companies_synced += 1

            # B. Sync Jobs
            for j_model, ticker in postgres_jobs:
                neo_session.run(
                    """
                    MERGE (job:Job {id: $id})
                    SET job.title = $title,
                        job.location = $location,
                        job.seniority = $seniority
                    WITH job
                    MATCH (comp:Company {ticker: $ticker})
                    MERGE (comp)-[:POSTED]->(job)
                    """,
                    id=j_model.id,
                    title=sanitize_text(j_model.title),
                    location=j_model.location or "Remote",
                    seniority=j_model.seniority or "Mid-Senior",
                    ticker=ticker
                )
                jobs_synced += 1

            # C. Sync News Events
            for n in postgres_news:
                # Format datetime to iso format string for Neo4j compatibility
                date_str = n.date.isoformat() if n.date else None
                neo_session.run(
                    """
                    MERGE (news:News {gdelt_id: $gdelt_id})
                    SET news.title = $title,
                        news.tone = $tone,
                        news.date = $date
                    """,
                    gdelt_id=n.gdelt_id,
                    title=sanitize_text(n.title),
                    tone=float(n.tone or 0.0),
                    date=date_str
                )
                news_synced += 1

            # D. Sync Vessels
            for v in postgres_vessels:
                neo_session.run(
                    """
                    MERGE (vessel:Vessel {imo: $imo})
                    SET vessel.name = $vessel_name,
                        vessel.status = $status
                    WITH vessel
                    MERGE (port:Port {name: $port_name})
                    MERGE (vessel)-[:DOCKED_AT]->(port)
                    """,
                    imo=v.imo,
                    vessel_name=v.vessel_name,
                    status=v.status or "UNDER_WAY",
                    port_name=v.port_name or "Port of Oakland"
                )
                vessels_synced += 1

            # E. Establish Complex Semantic Relationships (Mentions & Co-occurrence Associations)
            # Create (Company)-[:MENTIONED_IN]->(News)
            res_mentions = neo_session.run(
                """
                MATCH (c:Company), (n:News)
                WHERE n.title CONTAINS c.ticker OR n.title CONTAINS c.legal_name
                MERGE (c)-[r:MENTIONED_IN]->(n)
                RETURN count(r) as count
                """
            )
            relations_created += res_mentions.single()["count"]

            # Create (Company)-[:ASSOCIATED_WITH]->(Company)
            res_assoc = neo_session.run(
                """
                MATCH (c1:Company)-[:MENTIONED_IN]->(n:News)<-[:MENTIONED_IN]-(c2:Company)
                WHERE c1.ticker < c2.ticker
                MERGE (c1)-[r:ASSOCIATED_WITH]->(c2)
                RETURN count(r) as count
                """
            )
            relations_created += res_assoc.single()["count"]

        logger.info(
            f"Graph database sync complete. Synced: {companies_synced} Companies, {jobs_synced} Jobs, "
            f"{news_synced} News, {vessels_synced} Vessels. Created {relations_created} associations."
        )

        return {
            "companies": companies_synced,
            "jobs": jobs_synced,
            "news": news_synced,
            "vessels": vessels_synced,
            "relationships": relations_created
        }
