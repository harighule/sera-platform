from datetime import datetime
import uuid
from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, Integer, JSON
from sqlalchemy.orm import relationship
from database import Base

def gen_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

class CompanyModel(Base):
    __tablename__ = "companies"
    id = Column(String, primary_key=True, default=lambda: gen_id("CO"))
    ticker = Column(String, unique=True, nullable=False)
    legal_name = Column(String, nullable=False)
    sector = Column(String, nullable=True)
    headquarters = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Sentiment analysis signals
    news_sentiment = Column(Float, default=0.0)
    news_mentions = Column(Integer, default=0)
    reddit_sentiment = Column(Float, default=0.0)
    reddit_mentions = Column(Integer, default=0)

    # Relationships
    financial_metrics = relationship("FinancialMetricsModel", back_populates="company", cascade="all, delete-orphan")
    job_postings = relationship("JobPostingsModel", back_populates="company", cascade="all, delete-orphan")

class FinancialMetricsModel(Base):
    __tablename__ = "financial_metrics"
    id = Column(String, primary_key=True, default=lambda: gen_id("FM"))
    company_id = Column(String, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    revenue = Column(Float, nullable=False)
    accounts_receivable = Column(Float, nullable=False)
    deferred_revenue = Column(Float, nullable=False)
    filing_date = Column(DateTime, default=datetime.utcnow)

    company = relationship("CompanyModel", back_populates="financial_metrics")

class JobPostingsModel(Base):
    __tablename__ = "job_postings"
    id = Column(String, primary_key=True, default=lambda: gen_id("JP"))
    company_id = Column(String, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    location = Column(String, nullable=False)
    posted_date = Column(DateTime, default=datetime.utcnow)
    seniority = Column(String, nullable=False)

    company = relationship("CompanyModel", back_populates="job_postings")

class SearchTrendsModel(Base):
    __tablename__ = "search_trends"
    id = Column(String, primary_key=True, default=lambda: gen_id("ST"))
    keyword = Column(String, nullable=False)
    region = Column(String, nullable=False)
    interest_score = Column(Float, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)

class VesselMovementsModel(Base):
    __tablename__ = "vessel_movements"
    id = Column(String, primary_key=True, default=lambda: gen_id("VM"))
    imo = Column(String, nullable=False)
    vessel_name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    port_name = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class NewsEventsModel(Base):
    __tablename__ = "news_events"
    id = Column(String, primary_key=True, default=lambda: gen_id("NE"))
    gdelt_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    tone = Column(Float, nullable=False)
    themes = Column(String, nullable=True)
    tickers = Column(String, nullable=True)
    date = Column(DateTime, default=datetime.utcnow)

class GitHubActivityModel(Base):
    __tablename__ = "github_activity"
    id = Column(String, primary_key=True, default=lambda: gen_id("GH"))
    repo_name = Column(String, nullable=False)
    company_name = Column(String, nullable=False)
    language = Column(String, nullable=True)
    ai_keyword_count = Column(Integer, default=0)
    checked_at = Column(DateTime, default=datetime.utcnow)

class IngestionLogModel(Base):
    __tablename__ = "ingestion_logs"
    id = Column(String, primary_key=True, default=lambda: gen_id("IL"))
    source = Column(String, nullable=False) # e.g. "sec", "github", "gdelt", "ais"
    last_run = Column(DateTime, default=datetime.utcnow)
    status = Column(String, nullable=False) # e.g. "success", "failed"
    record_count = Column(Integer, default=0)

class TickerPriorityCacheModel(Base):
    __tablename__ = "ticker_priority_cache"
    ticker = Column(String, primary_key=True)
    company_name = Column(String, nullable=False)
    sector = Column(String, nullable=False)
    relevance_score = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow)

class HealthcareMetric(Base):
    __tablename__ = "healthcare_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    region = Column(String(50), nullable=False)  # State code (e.g., 'CA', 'NY')
    admission_count = Column(Integer, default=0)
    avg_total_payment = Column(Float, default=0.0)  # Average Medicare payment per stay
    drug_claim_count = Column(Integer, default=0)
    measurement_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ExecutiveMovement(Base):
    __tablename__ = "executive_movements"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(50), nullable=False)
    exec_name = Column(String(200), nullable=False)
    old_title = Column(String(200), nullable=True)
    new_title = Column(String(200), nullable=True)
    change_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)



