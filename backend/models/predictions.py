"""
SERA Platform — Prediction & Alert Models
============================================
These models represent outputs from AXIOM-Φ (entropy alerts)
and ZOLA (commercial intelligence predictions).
"""

from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class AxiomAlert(BaseModel):
    """
    An AXIOM-Φ alert — triggered when an entity's behavioral entropy
    deviates significantly from its baseline.
    
    This means: something is about to change for this entity.
    The entropy spike is the "thermodynamic signature" of an
    impending state transition.
    """

    alert_id: str = Field(default_factory=lambda: f"AX-{uuid.uuid4().hex[:8]}")
    entity_id: str
    entity_name: str = ""
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    entropy_deviation: float      # How many std devs above baseline
    predicted_transition: str     # "financial_decision", "health_event", etc.
    confidence: float             # 0.0 to 1.0
    estimated_timing: str         # "24-48 hours", "1-2 weeks"
    status: str = "active"

class PredictionBrief(BaseModel):
    """
    A ZOLA prediction brief — the commercial intelligence product.
    
    This tells an institutional client: what's about to happen,
    why it's happening, what to do about it, and how likely
    the intervention is to succeed.
    """
    prediction_id: str = Field(default_factory=lambda: f"ZP-{uuid.uuid4().hex[:8]}")
    entity_id: str
    entity_name: str = ""
    transition_type: str
    causal_mechanism: str
    commercial_opportunity: str
    optimal_intervention: str
    success_probability: float  # 0.0 to 1.0
    recommended_timing: str      # "immediate", "within 24 hours", etc.
    consequence_chain: list[str] = []
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class DashboardStats(BaseModel):
    """
    Aggregate statistics shown on the main dashboard.
    """

    total_events: int = 0
    total_entities: int = 0
    active_alerts: int = 0
    events_per_second: float = 0.0
    protocols_active: int = 0
    predictions_generated: int = 0
    uptime_seconds: float = 0.0
    