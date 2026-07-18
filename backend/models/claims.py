import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from database import Base

class Claim(Base):
    __tablename__ = "claims_aletheia"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claimant_entity_id = Column(String(50), nullable=False)  # ticker or CO-XXXX
    claimant_name = Column(String(200), nullable=True)
    claim_text = Column(Text, nullable=False)
    stake_amount = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), default="active")  # active, challenged, verified, rejected
    credibility_score = Column(Float, default=0.0)
    apex_verified = Column(Boolean, default=False)  # True if APEX finds supporting graph evidence
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Evidence(Base):
    __tablename__ = "evidence_aletheia"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims_aletheia.id", ondelete="CASCADE"))
    evidence_type = Column(String(30))  # news, financial, graph, user, document
    source = Column(String(200))
    content = Column(Text)
    weight = Column(Float, default=1.0)  # 0.1 to 1.5, based on source authority
    created_at = Column(DateTime, default=datetime.utcnow)

class Challenge(Base):
    __tablename__ = "challenges_aletheia"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims_aletheia.id", ondelete="CASCADE"))
    challenger_entity_id = Column(String(50))
    challenger_name = Column(String(200), nullable=True)
    challenge_text = Column(Text)
    counter_stake = Column(Float, default=0.0)
    status = Column(String(20), default="pending")  # pending, resolved
    resolution = Column(String(20), nullable=True)  # claimant_won, challenger_won
    created_at = Column(DateTime, default=datetime.utcnow)

class TrackedQuery(Base):
    __tablename__ = "tracked_queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(String(500), nullable=False)
    target_entity_id = Column(String(50), nullable=False)
    target_entity_name = Column(String(250), nullable=False)
    citation_count = Column(Integer, default=0)
    share_of_voice = Column(Float, default=0.0)
    last_run = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
