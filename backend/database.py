import asyncio
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from config import DATABASE_URL

logger = logging.getLogger(__name__)

# Configure engine kwargs dynamically for production PostgreSQL vs SQLite
engine_kwargs = {"echo": False, "future": True}
if DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("cockroachdb"):
    engine_kwargs.update({
        "pool_size": 20,
        "max_overflow": 10,
        "pool_recycle": 1800,
        "pool_pre_ping": True
    })
elif DATABASE_URL.startswith("sqlite"):
    engine_kwargs.update({
        "connect_args": {"timeout": 30.0}
    })

engine = create_async_engine(DATABASE_URL, **engine_kwargs)

# For SQLite, enable WAL (Write-Ahead Logging) mode to prevent database locks during concurrent operations
if DATABASE_URL.startswith("sqlite"):
    from sqlalchemy import event
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        except Exception as e:
            logger.warning(f"Failed to set SQLite PRAGMA journal_mode/synchronous: {e}")
        finally:
            cursor.close()

async_session_maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
Base = declarative_base()

async def verify_db_connection() -> None:
    """
    Resilient connection check. Confirms DATABASE_URL is not empty,
    and attempts to connect with exponential backoff retries.
    """
    if not DATABASE_URL or not DATABASE_URL.strip():
        raise RuntimeError("DATABASE_URL is not configured. Check DATABASE_URL in your .env file.")
    
    try:
        url_obj = make_url(DATABASE_URL)
        redacted_url = url_obj.render_as_string(hide_password=True)
    except Exception:
        redacted_url = str(DATABASE_URL)
        
    max_retries = 5
    retry_delay = 1.0
    
    for attempt in range(1, max_retries + 1):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info(f"[DATABASE] Connected successfully to database at {redacted_url}")
            return
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Cannot connect to database at {redacted_url} after {max_retries} attempts. Details: {e}"
                ) from e
            logger.warning(
                f"[DATABASE] Connection attempt {attempt}/{max_retries} failed. Retrying in {retry_delay}s... Error: {e}"
            )
            await asyncio.sleep(retry_delay)
            retry_delay *= 2.0

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db() -> None:
    # Fail fast early check before creating tables
    await verify_db_connection()
    from models.db_models import EntityModel, EventModel, AlertModel, PredictionModel, EntityRelationshipModel, ClaimModel, ClaimChallengeModel, TrackedQueryModel, CitationResultModel
    from models.commerce import CompanyModel, FinancialMetricsModel, JobPostingsModel, SearchTrendsModel, VesselMovementsModel, NewsEventsModel, GitHubActivityModel, IngestionLogModel, TickerPriorityCacheModel, HealthcareMetric, ExecutiveMovement
    from models.claims import TrackedQuery, Claim, Evidence, Challenge
    from models.security import SecurityEngagement, SecurityFinding, EngagementPhaseLog
    
    # Run Table Schema Creation in a transaction block
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # Run Alter table modifications gracefully
    try:
        async with engine.begin() as conn:
            await conn.execute(text("ALTER TABLE news_events ADD COLUMN IF NOT EXISTS tickers TEXT"))
    except Exception:
        try:
            async with engine.begin() as conn:
                await conn.execute(text("ALTER TABLE news_events ADD COLUMN tickers TEXT"))
        except Exception:
            pass

    # Sentiment columns migration for companies table
    for col_name, col_type in [
        ("news_sentiment", "FLOAT"),
        ("news_mentions", "INTEGER"),
        ("reddit_sentiment", "FLOAT"),
        ("reddit_mentions", "INTEGER")
    ]:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f"ALTER TABLE companies ADD COLUMN {col_name} {col_type}"))
        except Exception:
            pass