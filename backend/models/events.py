"""
SERA Platform — Event Data Models
===================================
These models define the structure of every event flowing through SERA.
Key concept: Every raw event from any protocol (SWIFT, FHIR, MQTT, HTTP)
gets converted into a single unified NormalizedEvent format. This is
SERA's core job — making different data look the same.
"""


from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict
import uuid

class SemanticTriple(BaseModel):
     """
    A Subject-Predicate-Object relationship extracted from data.
    
    Example: ("Entity-4829", "initiated", "wire-transfer-to-Frankfurt")
    This is how PRAGMA understands relationships between things.
    """
     subject: str
     predicate: str
     object: str
     weight: float = Field(default=1.0, ge=0.0, le=1.0)

class NormalizedEvent(BaseModel):
     """
    The universal internal format for ALL events in SERA.
    
    No matter if the original data was a SWIFT bank transfer,
    a FHIR health record, or an MQTT sensor reading — it all
    becomes this single format after SERA processes it.
    """
     
     event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
     timestamp: datetime = Field(default_factory=datetime.utcnow)
     source_protocol: str
     source_id: str
     entity_id: Optional[str] = None
     event_type: str
     category: str = ""
     payload: dict = {}
     embedding: list[float] = []
     triples: list[SemanticTriple] = []
     anomaly_score: float = 0.0

class StreamEvent(BaseModel):
     """
    What gets sent over WebSocket to the frontend.
    Wraps a NormalizedEvent with a message type.
    """
     type: str = "event"
     data: dict = {}