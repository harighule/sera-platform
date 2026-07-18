"""
SERA Security Assessment Models
=================================
SQLAlchemy models for the multi-agent authorized security assessment pipeline.
Stores engagements, findings, approval gates, and audit logs.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, Boolean, DateTime, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


def _gen_uuid() -> str:
    return str(uuid.uuid4())


class SecurityEngagement(Base):
    """Represents one authorized security assessment engagement."""
    __tablename__ = "security_engagements"

    id = Column(String, primary_key=True, default=_gen_uuid)
    auth_reference_id = Column(String, nullable=False, index=True)   # Signed authorization ref
    target_scope = Column(Text, nullable=False)                        # IP ranges, domains, etc.
    engagement_window = Column(String, nullable=False)                 # e.g. "2026-07-17 to 2026-07-18"
    operator_id = Column(String, nullable=False, default="system")     # Who started it

    phase = Column(String, nullable=False, default="PENDING")
    # PENDING | RECON | ANALYSIS | VALIDATION | AWAITING_APPROVAL | REPORTING | COMPLETE | ABORTED

    recon_output = Column(JSON, nullable=True)        # Raw recon JSON from ReconAgent
    analysis_output = Column(JSON, nullable=True)     # Hypotheses from AnalystAgent
    report_output = Column(Text, nullable=True)       # Final markdown report

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    findings = relationship("SecurityFinding", back_populates="engagement", cascade="all, delete-orphan")
    phase_log = relationship("EngagementPhaseLog", back_populates="engagement", cascade="all, delete-orphan")


class SecurityFinding(Base):
    """
    One confirmed finding or pending hypothesis from the assessment pipeline.
    Moves through: hypothesis → validated_passive | needs_active_exploit → (approved) → confirmed
    """
    __tablename__ = "security_findings"

    id = Column(String, primary_key=True, default=_gen_uuid)
    engagement_id = Column(String, ForeignKey("security_engagements.id"), nullable=False)

    hypothesis = Column(Text, nullable=False)
    basis = Column(Text, nullable=True)
    confidence = Column(String, default="low")         # low | medium | high
    priority = Column(Integer, default=3)              # 1 (highest) - 5 (lowest)
    verification_method = Column(String, default="passive")  # passive | requires_active_exploit

    # Validation result
    status = Column(String, default="pending")
    # pending | confirmed_passive | needs_active_exploit | rejected_false_positive | confirmed_active

    validation_evidence = Column(Text, nullable=True)  # Exact request/response used
    validation_reasoning = Column(Text, nullable=True)

    # Approval gate (for needs_active_exploit items)
    approval_requested_at = Column(DateTime, nullable=True)
    approval_granted_at = Column(DateTime, nullable=True)
    approval_granted_by = Column(String, nullable=True)
    proposed_action = Column(Text, nullable=True)
    proposed_tool = Column(String, nullable=True)
    risk_level = Column(String, nullable=True)

    # Report fields (populated by ReportAgent)
    severity = Column(String, nullable=True)           # Critical | High | Medium | Low
    cvss_vector = Column(String, nullable=True)
    cvss_score = Column(Float, nullable=True)
    title = Column(String, nullable=True)
    description_plain = Column(Text, nullable=True)
    business_impact = Column(Text, nullable=True)
    remediation = Column(Text, nullable=True)
    cve_references = Column(JSON, nullable=True)
    owasp_category = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    engagement = relationship("SecurityEngagement", back_populates="findings")


class EngagementPhaseLog(Base):
    """Immutable audit log — every phase transition and approval event is recorded here."""
    __tablename__ = "engagement_phase_logs"

    id = Column(String, primary_key=True, default=_gen_uuid)
    engagement_id = Column(String, ForeignKey("security_engagements.id"), nullable=False)

    event_type = Column(String, nullable=False)   # phase_transition | approval_requested | approval_granted | finding_added
    from_phase = Column(String, nullable=True)
    to_phase = Column(String, nullable=True)
    actor = Column(String, nullable=False, default="system")
    detail = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    engagement = relationship("SecurityEngagement", back_populates="phase_log")
