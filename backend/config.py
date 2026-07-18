"""
SERA Platform — Central Configuration
======================================
All configurable values live here. We use environment variables
with Pydantic validation to fail fast on invalid or missing configurations.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError

# Load environment variables from .env file if it exists
load_dotenv()

class AppSettings(BaseModel):
    # Running Environment
    PRODUCTION: bool = Field(default=False)

    # Database
    DATABASE_URL: str = Field(default="postgresql+asyncpg://localhost:5432/sera_db")

    # Entity AI Layer
    ENTITY_MODE: str = Field(default="mock")
    USE_NOETHER: bool = Field(default=False)
    USE_PRETRAINED_CIFN: bool = Field(default=True)
    ENTITY_API_URL: str = Field(default="http://localhost:8000")

    # AI Chat Assistant
    AI_API_KEY: str = Field(default="")
    AI_MODEL: str = Field(default="grok-3-mini-fast")
    AI_BASE_URL: str = Field(default="https://api.x.ai/v1")

    # Synthetic Data Generation rates
    FINANCIAL_EVENTS_PER_SEC: float = Field(default=2.0, ge=0.0)
    HEALTHCARE_EVENTS_PER_SEC: float = Field(default=1.5, ge=0.0)
    IOT_EVENTS_PER_SEC: float = Field(default=3.0, ge=0.0)
    SOCIAL_EVENTS_PER_SEC: float = Field(default=2.5, ge=0.0)

    # Entropy Engine
    ENTROPY_WINDOW_SIZE: int = Field(default=50, gt=0)
    ENTROPY_ALERT_THRESHOLD: float = Field(default=2.0, gt=0.0)

    # Server settings
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000, ge=1, le=65535)
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:5173"])

    # API credentials and sync toggles
    USE_REAL_DATA: bool = Field(default=False)
    SEC_IDENTITY_EMAIL: str = Field(default="")
    GITHUB_TOKEN: str = Field(default="")
    AIS_STREAM_KEY: str = Field(default="")
    APIFY_TOKEN: str = Field(default="")
    DATAGOV_IN_API_KEY: str = Field(default="")

    # Neo4j settings
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="")

    # Background sync intervals (minutes)
    GDELT_INTERVAL_MINUTES: int = Field(default=15, gt=0)
    AIS_INTERVAL_MINUTES: int = Field(default=60, gt=0)
    JOBS_INTERVAL_MINUTES: int = Field(default=60, gt=0)
    EXEC_INTERVAL_MINUTES: int = Field(default=60, gt=0)
    FULL_SYNC_HOUR: int = Field(default=6, ge=0, le=23)
    FULL_SYNC_MINUTE: int = Field(default=0, ge=0, le=59)

    # ─── SECURITY AGENT CONFIG ───
    KALI_IMAGE: str = Field(default="custom-kali:latest")
    ZERO_INPUT_ENABLED: bool = Field(default=False)
    NETWORK_INTERFACE: str = Field(default="eth0")
    EXPLOIT_SERVER_IP: str = Field(default="192.168.1.100")
    
    # ─── LLM PROVIDER CONFIG ───
    LLM_PROVIDER: str = Field(default="local")  # anthropic, openai, local
    ANTHROPIC_API_KEY: str = Field(default="")
    OPENAI_API_KEY: str = Field(default="")
    LOCAL_LLM_URL: str = Field(default="http://localhost:11434/api/generate")
    LOCAL_LLM_MODEL: str = Field(default="qwen2.5:1.5b")

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if not v.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError("DATABASE_URL must start with a valid postgresql or sqlite protocol driver prefix.")
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


def _load_settings() -> AppSettings:
    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    
    # Helper to parse string bools
    def get_bool(key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.strip().lower() in ("true", "1", "yes")

    try:
        settings = AppSettings(
            PRODUCTION=get_bool("PRODUCTION", False) or (os.getenv("ENTITY_MODE", "mock").lower() != "mock"),
            DATABASE_URL=os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost:5432/sera_db"),
            ENTITY_MODE=os.getenv("ENTITY_MODE", "mock"),
            USE_NOETHER=get_bool("USE_NOETHER", False),
            USE_PRETRAINED_CIFN=get_bool("USE_PRETRAINED_CIFN", True),
            ENTITY_API_URL=os.getenv("ENTITY_API_URL", "http://localhost:8000"),
            AI_API_KEY=os.getenv("AI_API_KEY", ""),
            AI_MODEL=os.getenv("AI_MODEL", "grok-3-mini-fast"),
            AI_BASE_URL=os.getenv("AI_BASE_URL", "https://api.x.ai/v1"),
            FINANCIAL_EVENTS_PER_SEC=float(os.getenv("FINANCIAL_EVENTS_PER_SEC", "2.0")),
            HEALTHCARE_EVENTS_PER_SEC=float(os.getenv("HEALTHCARE_EVENTS_PER_SEC", "1.5")),
            IOT_EVENTS_PER_SEC=float(os.getenv("IOT_EVENTS_PER_SEC", "3.0")),
            SOCIAL_EVENTS_PER_SEC=float(os.getenv("SOCIAL_EVENTS_PER_SEC", "2.5")),
            ENTROPY_WINDOW_SIZE=int(os.getenv("ENTROPY_WINDOW_SIZE", "50")),
            ENTROPY_ALERT_THRESHOLD=float(os.getenv("ENTROPY_ALERT_THRESHOLD", "2.0")),
            HOST=os.getenv("HOST", "0.0.0.0"),
            PORT=int(os.getenv("PORT", "8000")),
            CORS_ORIGINS=raw_origins,
            USE_REAL_DATA=get_bool("USE_REAL_DATA", False),
            SEC_IDENTITY_EMAIL=os.getenv("SEC_IDENTITY_EMAIL", ""),
            GITHUB_TOKEN=os.getenv("GITHUB_TOKEN", ""),
            AIS_STREAM_KEY=os.getenv("AIS_STREAM_KEY", ""),
            APIFY_TOKEN=os.getenv("APIFY_TOKEN", ""),
            DATAGOV_IN_API_KEY=os.getenv("DATAGOV_IN_API_KEY", ""),
            NEO4J_URI=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            NEO4J_USER=os.getenv("NEO4J_USER", "neo4j"),
            NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD", ""),
            GDELT_INTERVAL_MINUTES=int(os.getenv("GDELT_INTERVAL_MINUTES", "15")),
            AIS_INTERVAL_MINUTES=int(os.getenv("AIS_INTERVAL_MINUTES", "60")),
            JOBS_INTERVAL_MINUTES=int(os.getenv("JOBS_INTERVAL_MINUTES", "60")),
            EXEC_INTERVAL_MINUTES=int(os.getenv("EXEC_INTERVAL_MINUTES", "60")),
            FULL_SYNC_HOUR=int(os.getenv("FULL_SYNC_HOUR", "6")),
            FULL_SYNC_MINUTE=int(os.getenv("FULL_SYNC_MINUTE", "0")),
            # ─── SECURITY AGENT CONFIG ───
            KALI_IMAGE=os.getenv("KALI_IMAGE", "custom-kali:latest"),
            ZERO_INPUT_ENABLED=get_bool("ZERO_INPUT_ENABLED", False),
            NETWORK_INTERFACE=os.getenv("NETWORK_INTERFACE", "eth0"),
            EXPLOIT_SERVER_IP=os.getenv("EXPLOIT_SERVER_IP", "192.168.1.100"),
            LLM_PROVIDER=os.getenv("LLM_PROVIDER", "local"),
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            LOCAL_LLM_URL=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate"),
            LOCAL_LLM_MODEL=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:1.5b"),
        )

        # In production mode, enforce security checks
        if settings.PRODUCTION:
            if settings.NEO4J_URI.startswith("bolt://localhost") and not os.getenv("ALLOW_LOCALHOST_DB_IN_PROD"):
                logging.warning("[CONFIG] Running in PRODUCTION but NEO4J_URI is local database.")
            if not settings.DATABASE_URL or "localhost" in settings.DATABASE_URL:
                if not os.getenv("ALLOW_LOCALHOST_DB_IN_PROD"):
                    logging.warning("[CONFIG] Running in PRODUCTION but DATABASE_URL points to localhost.")

        return settings

    except ValidationError as e:
        print(f"CRITICAL CONFIGURATION ERROR:\n{e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"CRITICAL CONFIGURATION INITIALIZATION FAILURE: {e}")
        raise SystemExit(1)


# Instance loaded at runtime
_config_instance = _load_settings()

# Export all settings attributes directly for compatibility
PRODUCTION = _config_instance.PRODUCTION
DATABASE_URL = _config_instance.DATABASE_URL
ENTITY_MODE = _config_instance.ENTITY_MODE
USE_NOETHER = _config_instance.USE_NOETHER
USE_PRETRAINED_CIFN = _config_instance.USE_PRETRAINED_CIFN
ENTITY_API_URL = _config_instance.ENTITY_API_URL
AI_API_KEY = _config_instance.AI_API_KEY
AI_MODEL = _config_instance.AI_MODEL
AI_BASE_URL = _config_instance.AI_BASE_URL
FINANCIAL_EVENTS_PER_SEC = _config_instance.FINANCIAL_EVENTS_PER_SEC
HEALTHCARE_EVENTS_PER_SEC = _config_instance.HEALTHCARE_EVENTS_PER_SEC
IOT_EVENTS_PER_SEC = _config_instance.IOT_EVENTS_PER_SEC
SOCIAL_EVENTS_PER_SEC = _config_instance.SOCIAL_EVENTS_PER_SEC
ENTROPY_WINDOW_SIZE = _config_instance.ENTROPY_WINDOW_SIZE
ENTROPY_ALERT_THRESHOLD = _config_instance.ENTROPY_ALERT_THRESHOLD
HOST = _config_instance.HOST
PORT = _config_instance.PORT
CORS_ORIGINS = _config_instance.CORS_ORIGINS
USE_REAL_DATA = _config_instance.USE_REAL_DATA
SEC_IDENTITY_EMAIL = _config_instance.SEC_IDENTITY_EMAIL
GITHUB_TOKEN = _config_instance.GITHUB_TOKEN
AIS_STREAM_KEY = _config_instance.AIS_STREAM_KEY
APIFY_TOKEN = _config_instance.APIFY_TOKEN
DATAGOV_IN_API_KEY = _config_instance.DATAGOV_IN_API_KEY
NEO4J_URI = _config_instance.NEO4J_URI
NEO4J_USER = _config_instance.NEO4J_USER
NEO4J_PASSWORD = _config_instance.NEO4J_PASSWORD
GDELT_INTERVAL_MINUTES = _config_instance.GDELT_INTERVAL_MINUTES
AIS_INTERVAL_MINUTES = _config_instance.AIS_INTERVAL_MINUTES
JOBS_INTERVAL_MINUTES = _config_instance.JOBS_INTERVAL_MINUTES
EXEC_INTERVAL_MINUTES = _config_instance.EXEC_INTERVAL_MINUTES
FULL_SYNC_HOUR = _config_instance.FULL_SYNC_HOUR
FULL_SYNC_MINUTE = _config_instance.FULL_SYNC_MINUTE

# ─── SECURITY AGENT EXPORTS ───
KALI_IMAGE = _config_instance.KALI_IMAGE
ZERO_INPUT_ENABLED = _config_instance.ZERO_INPUT_ENABLED
NETWORK_INTERFACE = _config_instance.NETWORK_INTERFACE
EXPLOIT_SERVER_IP = _config_instance.EXPLOIT_SERVER_IP
LLM_PROVIDER = _config_instance.LLM_PROVIDER
ANTHROPIC_API_KEY = _config_instance.ANTHROPIC_API_KEY
OPENAI_API_KEY = _config_instance.OPENAI_API_KEY
LOCAL_LLM_URL = _config_instance.LOCAL_LLM_URL
LOCAL_LLM_MODEL = _config_instance.LOCAL_LLM_MODEL