from datetime import datetime
import uuid
from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, JSON, Text
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


class EntityRelationshipModel(Base):
    __tablename__ = "entity_relationships"
    id = Column(String, primary_key=True, default=lambda: gen_id("ER"))
    source_entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String, nullable=False) # e.g. "works_with", "co_occurs_with", "supplies_to"
    confidence_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ClaimModel(Base):
    __tablename__ = "claims"
    claim_id = Column(String, primary_key=True, default=lambda: gen_id("CL"))
    claimant_id = Column(String, nullable=False)
    claimant_name = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    stake_amount = Column(Float, nullable=False)
    status = Column(String, default="active")  # active, challenged, verified, disputed
    credibility_score = Column(Float, default=0.0)
    apex_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_reaffirmed_at = Column(DateTime, default=datetime.utcnow)
    
    challenges = relationship("ClaimChallengeModel", back_populates="claim", cascade="all, delete-orphan")
    evidence = relationship("EvidenceModel", back_populates="claim", cascade="all, delete-orphan")


class ClaimChallengeModel(Base):
    __tablename__ = "claim_challenges"
    challenge_id = Column(String, primary_key=True, default=lambda: gen_id("CC"))
    target_claim_id = Column(String, ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False)
    challenger_id = Column(String, nullable=False)
    challenger_name = Column(String, nullable=True)
    challenge_text = Column(Text, nullable=True)
    counter_stake_amount = Column(Float, nullable=False)
    status = Column(String, default="pending")  # pending, resolved
    resolution = Column(String, nullable=True)  # claimant_won, challenger_won
    created_at = Column(DateTime, default=datetime.utcnow)
    
    claim = relationship("ClaimModel", back_populates="challenges")


class EvidenceModel(Base):
    __tablename__ = "evidence"
    evidence_id = Column(String, primary_key=True, default=lambda: gen_id("EV"))
    claim_id = Column(String, ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False)
    evidence_type = Column(String, default="user")  # financial, graph, news, document, user
    source = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    weight = Column(Float, default=1.0)  # 0.1 to 1.5
    created_at = Column(DateTime, default=datetime.utcnow)
    
    claim = relationship("ClaimModel", back_populates="evidence")


class TrackedQueryModel(Base):
    __tablename__ = "legacy_tracked_queries"
    query_id = Column(String, primary_key=True, default=lambda: gen_id("TQ"))
    query_text = Column(String, nullable=False)
    target_entity_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    citation_results = relationship("CitationResultModel", back_populates="tracked_query", cascade="all, delete-orphan")


class CitationResultModel(Base):
    __tablename__ = "citation_results"
    result_id = Column(String, primary_key=True, default=lambda: gen_id("CR"))
    query_id = Column(String, ForeignKey("legacy_tracked_queries.query_id", ondelete="CASCADE"), nullable=False)
    ai_platform = Column(String, nullable=False) # e.g. "chatgpt", "perplexity", "gemini"
    was_cited = Column(Boolean, nullable=False)
    competitor_names_cited = Column(String, nullable=True) # comma-separated list
    checked_at = Column(DateTime, default=datetime.utcnow)

    tracked_query = relationship("TrackedQueryModel", back_populates="citation_results")