from datetime import datetime
import uuid
from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base

def gen_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

class EntityModel(Base):
    __tablename__ = "entities"
    id = Column(String, primary_key=True, default=lambda: gen_id("E"))
    name = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    status = Column(String, default="stable")
    entropy = Column(Float, default=0.0)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    events = relationship("EventModel", back_populates="entity", cascade="all, delete-orphan")
    alerts = relationship("AlertModel", back_populates="entity", cascade="all, delete-orphan")
    predictions = relationship("PredictionModel", back_populates="entity", cascade="all, delete-orphan")

class EventModel(Base):
    __tablename__ = "events"
    id = Column(String, primary_key=True, default=lambda: gen_id("EV"))
    entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    protocol = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    entropy_delta = Column(Float, default=0.0)
    entity = relationship("EntityModel", back_populates="events")

class AlertModel(Base):
    __tablename__ = "alerts"
    id = Column(String, primary_key=True, default=lambda: gen_id("AL"))
    entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    alert_type = Column(String, nullable=False)
    description = Column(String, nullable=False)
    severity = Column(String, default="warning")
    entropy_value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    entity = relationship("EntityModel", back_populates="alerts")

class PredictionModel(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True, default=lambda: gen_id("ZP"))
    entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    transition_type = Column(String, nullable=False)
    causal_mechanism = Column(String, nullable=False)
    optimal_intervention = Column(String, nullable=False)
    success_probability = Column(Float, nullable=False)
    recommended_timing = Column(String, nullable=False)
    consequence_chain = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    entity = relationship("EntityModel", back_populates="predictions")