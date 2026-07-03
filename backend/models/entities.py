"""
SERA Platform — Entity Data Models
=====================================
An "entity" is any resolved identity in the system — a person,
organization, device, or location that SERA has identified
across multiple data sources.
The key innovation: entities are resolved by BEHAVIORAL SIMILARITY
in the semantic manifold, not just by matching emails or IDs.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict
import uuid

class Entity(BaseModel):
    """
    A unified cross-domain identity.
    
    This represents a single real-world entity (person, company, device)
    that may appear across financial, healthcare, IoT, and social data.
    """

    entity_id: str = Field(default_factory=lambda: f"E-{uuid.uuid4().hex[:6]}")
    entity_type: str              # "person", "organization", "device", "location"
    display_name: str
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    event_count: int = 0
    domains: list[str] = []       # ["financial", "healthcare", "iot", "social"]
    current_embedding: list[float] = []  # Current position in semantic manifold
    entropy_baseline: float = 0.0
    current_entropy: float = 0.0
    transition_state: str = "stable"  # "stable", "pre-transition", "transitioning"
    risk_score: float = 0.0

class EntityRelationship(BaseModel):
    """
    A connection between two entities in the knowledge graph.
    
    Example: Entity A "sends_payment_to" Entity B with weight 0.8
    """
    source_entity: str
    target_entity: str
    relationship_type: str  # e.g., "sends_payment_to", "is_employed_by"
    weight: float = 1.0
    detected_at: datetime = Field(default_factory=datetime.utcnow)

class EntityProfile(BaseModel):
    """
    Full detailed profile of an entity — returned when you click
    on an entity in the Entity Explorer page.
    """
    
    entity: Entity
    recent_events: list[dict] = []
    relationships: list[EntityRelationship] = []
    entropy_history: list[dict] = []  # [{timestamp, entropy_value}, ...]
    domain_breakdown: dict = {}