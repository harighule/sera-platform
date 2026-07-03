"""
SERA Platform — Central Configuration
======================================
All configurable values live here. We use environment variables
with sensible defaults so the app works out of the box.
"""


import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

#____DATABASE_____

# PostgreSQL database connection URL

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://localhost:5432/sera_db"
)

# ENTITY AI LAYER

ENTITY_MODE = os.getenv("ENTITY_MODE", "mock")

ENTITY_API_URL = os.getenv("ENTITY_API_URL", "http://localhost:8000")

# AI CHAT ASSISTANT

AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "grok-3-mini-fast")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.x.ai/v1")

# SYNTHETIC DATA GENERATION

FINANCIAL_EVENTS_PER_SEC = float(os.getenv("FINANCIAL_EVENTS_PER_SEC", "2.0"))
HEALTHCARE_EVENTS_PER_SEC = float(os.getenv("HEALTHCARE_EVENTS_PER_SEC", "1.5"))
IOT_EVENTS_PER_SEC = float(os.getenv("IOT_EVENTS_PER_SEC", "3.0"))
SOCIAL_EVENTS_PER_SEC = float(os.getenv("SOCIAL_EVENTS_PER_SEC", "2.5"))

# AXIOM-Φ ENTROPY ENGINE

# How many recent events to look at when calculating entropy
ENTROPY_WINDOW_SIZE = int(os.getenv("ENTROPY_WINDOW_SIZE", "50"))

# How sensitive the anomaly detection is
# 2.0 = alert when entropy is 2 standard deviations above normal
# Lower = more alerts (sensitive), Higher = fewer alerts (strict)
ENTROPY_ALERT_THRESHOLD = float(os.getenv("ENTROPY_ALERT_THRESHOLD", "2.0"))

# SERVER

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
).split(",")



