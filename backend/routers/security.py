"""
SERA Security Assessment Router
================================
REST API endpoints for the multi-agent authorized security assessment pipeline.

Endpoints:
  POST   /api/security/engage              — Start new engagement
  GET    /api/security/engagements         — List all engagements
  GET    /api/security/engage/{id}         — Get engagement status + findings
  POST   /api/security/engage/{id}/run     — Advance pipeline to next phase
  POST   /api/security/approve/{eid}/{fid} — Human approval gate (active exploit authorization)
  POST   /api/security/engage/{id}/report  — Generate final report
  GET    /api/security/engage/{id}/report  — Download final report JSON
  POST   /api/security/engage/{id}/abort   — Abort engagement
"""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select
from database import async_session_maker
from models.security import SecurityEngagement, SecurityFinding, EngagementPhaseLog

logger = logging.getLogger("sera.security_router")
router = APIRouter(prefix="/api/security", tags=["security"])


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class EngagementCreate(BaseModel):
    target_scope: str = Field(..., description="IP ranges, domains, or CIDRs in scope. E.g. '10.0.1.0/24, api.example.com'")
    auth_reference_id: str = Field(..., description="Signed authorization reference ID from client contract")
    engagement_window: str = Field(..., description="Authorized testing window. E.g. '2026-07-17 09:00 UTC to 2026-07-17 18:00 UTC'")
    operator_id: str = Field(default="system", description="Operator identifier initiating the assessment")


class ApprovalDecision(BaseModel):
    approved: bool = Field(..., description="True to approve active exploitation for this finding")
    approver_id: str = Field(..., description="Identity of the human operator granting or denying approval")
    notes: str = Field(default="", description="Optional notes for the audit log")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _log_phase(session, engagement_id: str, event_type: str,
                     from_phase: str, to_phase: str, actor: str, detail: str = ""):
    entry = EngagementPhaseLog(
        engagement_id=engagement_id,
        event_type=event_type,
        from_phase=from_phase,
        to_phase=to_phase,
        actor=actor,
        detail=detail,
        timestamp=datetime.utcnow()
    )
    session.add(entry)


def _serialise_finding(f: SecurityFinding) -> dict:
    return {
        "id": f.id,
        "hypothesis": f.hypothesis,
        "basis": f.basis,
        "confidence": f.confidence,
        "priority": f.priority,
        "verification_method": f.verification_method,
        "status": f.status,
        "validation_evidence": f.validation_evidence,
        "validation_reasoning": f.validation_reasoning,
        "approval_requested_at": f.approval_requested_at.isoformat() if f.approval_requested_at else None,
        "approval_granted_at": f.approval_granted_at.isoformat() if f.approval_granted_at else None,
        "approval_granted_by": f.approval_granted_by,
        "proposed_action": f.proposed_action,
        "proposed_tool": f.proposed_tool,
        "risk_level": f.risk_level,
        "severity": f.severity,
        "cvss_vector": f.cvss_vector,
        "cvss_score": f.cvss_score,
        "title": f.title,
        "description_plain": f.description_plain,
        "business_impact": f.business_impact,
        "remediation": f.remediation,
        "cve_references": f.cve_references,
        "owasp_category": f.owasp_category,
        "created_at": f.created_at.isoformat() if f.created_at else None,
    }


def _serialise_engagement(eng: SecurityEngagement, include_findings: bool = True) -> dict:
    result = {
        "id": eng.id,
        "auth_reference_id": eng.auth_reference_id,
        "target_scope": eng.target_scope,
        "engagement_window": eng.engagement_window,
        "operator_id": eng.operator_id,
        "phase": eng.phase,
        "created_at": eng.created_at.isoformat() if eng.created_at else None,
        "updated_at": eng.updated_at.isoformat() if eng.updated_at else None,
        "completed_at": eng.completed_at.isoformat() if eng.completed_at else None,
        "recon_summary": eng.analysis_output.get("plain_summary", "") if eng.analysis_output else None,
        "report_available": eng.report_output is not None,
    }
    if include_findings:
        result["findings"] = [_serialise_finding(f) for f in (eng.findings or [])]
        result["findings_count"] = len(eng.findings or [])
        awaiting = [f for f in (eng.findings or []) if f.status == "needs_active_exploit_to_confirm"
                    and f.approval_granted_at is None]
        result["awaiting_approval_count"] = len(awaiting)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/engage", summary="Start a new authorized security assessment engagement")
async def create_engagement(body: EngagementCreate, background_tasks: BackgroundTasks):
    """
    Creates a new engagement record and immediately triggers Phase 1 (Recon + Analysis)
    in a background task. Returns the engagement ID so the client can poll status.
    """
    if not body.target_scope.strip():
        raise HTTPException(status_code=422, detail="target_scope must not be empty.")
    if not body.auth_reference_id.strip():
        raise HTTPException(status_code=422, detail="auth_reference_id is required — engagement cannot proceed without authorization.")
    if not body.engagement_window.strip():
        raise HTTPException(status_code=422, detail="engagement_window is required.")

    async with async_session_maker() as session:
        eng = SecurityEngagement(
            target_scope=body.target_scope,
            auth_reference_id=body.auth_reference_id,
            engagement_window=body.engagement_window,
            operator_id=body.operator_id,
            phase="RECON"
        )
        session.add(eng)
        await session.flush()  # get ID before commit
        await _log_phase(session, eng.id, "phase_transition", "PENDING", "RECON",
                         body.operator_id, f"Engagement created. Target: {body.target_scope[:120]}")
        await session.commit()
        engagement_id = eng.id

    logger.info(f"[SECURITY] Engagement {engagement_id} created. Starting recon in background.")
    background_tasks.add_task(_run_recon_phase, engagement_id)

    return {
        "engagement_id": engagement_id,
        "phase": "RECON",
        "message": "Engagement started. Recon & Analysis running in background. Poll GET /api/security/engage/{id} for status.",
        "auth_reference_id": body.auth_reference_id,
        "target_scope": body.target_scope,
    }


@router.get("/engagements", summary="List all engagements")
async def list_engagements():
    async with async_session_maker() as session:
        result = await session.execute(
            select(SecurityEngagement).order_by(SecurityEngagement.created_at.desc()).limit(50)
        )
        engagements = result.scalars().all()
        return {"engagements": [_serialise_engagement(e, include_findings=False) for e in engagements]}


@router.get("/engage/{engagement_id}", summary="Get engagement status, findings, and audit log")
async def get_engagement(engagement_id: str):
    from sqlalchemy.orm import selectinload
    
    async with async_session_maker() as session:
        # Eagerly load findings and phase_log to avoid lazy loading errors
        result = await session.execute(
            select(SecurityEngagement)
            .where(SecurityEngagement.id == engagement_id)
            .options(
                selectinload(SecurityEngagement.findings),
                selectinload(SecurityEngagement.phase_log)
            )
        )
        eng = result.scalars().first()
        if not eng:
            raise HTTPException(status_code=404, detail=f"Engagement {engagement_id} not found.")

        # Build the response manually (safe, no lazy loading)
        data = {
            "id": eng.id,
            "auth_reference_id": eng.auth_reference_id,
            "target_scope": eng.target_scope,
            "engagement_window": eng.engagement_window,
            "operator_id": eng.operator_id,
            "phase": eng.phase,
            "created_at": eng.created_at.isoformat() if eng.created_at else None,
            "updated_at": eng.updated_at.isoformat() if eng.updated_at else None,
            "completed_at": eng.completed_at.isoformat() if eng.completed_at else None,
            "recon_summary": eng.analysis_output.get("plain_summary", "") if eng.analysis_output else None,
            "report_available": eng.report_output is not None,
            "findings": [
                {
                    "id": f.id,
                    "hypothesis": f.hypothesis,
                    "basis": f.basis,
                    "confidence": f.confidence,
                    "priority": f.priority,
                    "verification_method": f.verification_method,
                    "status": f.status,
                    "validation_evidence": f.validation_evidence,
                    "validation_reasoning": f.validation_reasoning,
                    "approval_requested_at": f.approval_requested_at.isoformat() if f.approval_requested_at else None,
                    "approval_granted_at": f.approval_granted_at.isoformat() if f.approval_granted_at else None,
                    "approval_granted_by": f.approval_granted_by,
                    "proposed_action": f.proposed_action,
                    "proposed_tool": f.proposed_tool,
                    "risk_level": f.risk_level,
                    "severity": f.severity,
                    "cvss_vector": f.cvss_vector,
                    "cvss_score": f.cvss_score,
                    "title": f.title,
                    "description_plain": f.description_plain,
                    "business_impact": f.business_impact,
                    "remediation": f.remediation,
                    "cve_references": f.cve_references,
                    "owasp_category": f.owasp_category,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in (eng.findings or [])
            ],
            "findings_count": len(eng.findings or []),
            "awaiting_approval_count": len([
                f for f in (eng.findings or []) 
                if f.status == "needs_active_exploit_to_confirm" and f.approval_granted_at is None
            ]),
            "phase_log": [
                {
                    "event_type": l.event_type,
                    "from_phase": l.from_phase,
                    "to_phase": l.to_phase,
                    "actor": l.actor,
                    "detail": l.detail,
                    "timestamp": l.timestamp.isoformat() if l.timestamp else None
                }
                for l in (eng.phase_log or [])
            ],
            "approval_gate": [
                {
                    "finding_id": f.id,
                    "target": eng.target_scope,
                    "finding": f.hypothesis,
                    "confidence": f.confidence,
                    "proposed_action": f.proposed_action or "Active exploit confirmation required",
                    "tool": f.proposed_tool or "TBD",
                    "risk": f.risk_level or "High",
                    "requested_at": f.approval_requested_at.isoformat() if f.approval_requested_at else None,
                }
                for f in (eng.findings or [])
                if f.status == "needs_active_exploit_to_confirm" and f.approval_granted_at is None
            ]
        }

        return data


@router.post("/approve/{engagement_id}/{finding_id}", summary="Human approval gate — authorize or deny active exploit confirmation")
async def approve_finding(engagement_id: str, finding_id: str, decision: ApprovalDecision, background_tasks: BackgroundTasks):
    """
    Human approval gate. The operator explicitly approves or denies active exploitation
    for a specific finding. This decision is logged immutably in the phase audit log.
    """
    async with async_session_maker() as session:
        eng_result = await session.execute(
            select(SecurityEngagement).where(SecurityEngagement.id == engagement_id)
        )
        eng = eng_result.scalars().first()
        if not eng:
            raise HTTPException(status_code=404, detail=f"Engagement {engagement_id} not found.")

        finding_result = await session.execute(
            select(SecurityFinding).where(
                SecurityFinding.id == finding_id,
                SecurityFinding.engagement_id == engagement_id
            )
        )
        finding = finding_result.scalars().first()
        if not finding:
            raise HTTPException(status_code=404, detail=f"Finding {finding_id} not found in engagement {engagement_id}.")

        if finding.status != "needs_active_exploit_to_confirm":
            raise HTTPException(
                status_code=409,
                detail=f"Finding is in status '{finding.status}' — only 'needs_active_exploit_to_confirm' findings can be approved."
            )

        if decision.approved:
            finding.status = "confirmed_active"  # Marked as approved — would proceed with human-supervised active testing
            finding.approval_granted_at = datetime.utcnow()
            finding.approval_granted_by = decision.approver_id

            await _log_phase(
                session, engagement_id,
                "approval_granted", "AWAITING_APPROVAL", "AWAITING_APPROVAL",
                decision.approver_id,
                f"APPROVED active testing for finding {finding_id}: {finding.hypothesis[:120]}. Notes: {decision.notes}"
            )
            logger.info(f"[SECURITY][APPROVAL] Finding {finding_id} APPROVED by {decision.approver_id}")
            status_msg = "Approved. Finding marked for active exploit confirmation. Human-supervised tool execution may now proceed."
        else:
            finding.status = "rejected_false_positive"
            await _log_phase(
                session, engagement_id,
                "approval_denied", "AWAITING_APPROVAL", "AWAITING_APPROVAL",
                decision.approver_id,
                f"DENIED active testing for finding {finding_id}: {finding.hypothesis[:120]}. Notes: {decision.notes}"
            )
            logger.info(f"[SECURITY][APPROVAL] Finding {finding_id} DENIED by {decision.approver_id}")
            status_msg = "Denied. Finding will not be actively tested. Marked as rejected."

        await session.commit()

    return {
        "finding_id": finding_id,
        "decision": "approved" if decision.approved else "denied",
        "approver": decision.approver_id,
        "timestamp": datetime.utcnow().isoformat(),
        "message": status_msg
    }


@router.post("/engage/{engagement_id}/report", summary="Generate final security report (Phase 5)")
async def generate_report(engagement_id: str, background_tasks: BackgroundTasks):
    """
    Triggers ReportAgent to generate the final professional security report.
    Only includes confirmed findings. Pending approvals go to appendix.
    """
    async with async_session_maker() as session:
        result = await session.execute(
            select(SecurityEngagement).where(SecurityEngagement.id == engagement_id)
        )
        eng = result.scalars().first()
        if not eng:
            raise HTTPException(status_code=404, detail=f"Engagement {engagement_id} not found.")

        if eng.phase == "RECON":
            raise HTTPException(status_code=409, detail="Recon/Analysis still running. Wait for VALIDATION phase before generating report.")

    background_tasks.add_task(_run_report_phase, engagement_id)
    return {"message": "Report generation started in background.", "engagement_id": engagement_id}


@router.get("/engage/{engagement_id}/report", summary="Download the final security report")
async def get_report(engagement_id: str):
    async with async_session_maker() as session:
        result = await session.execute(
            select(SecurityEngagement).where(SecurityEngagement.id == engagement_id)
        )
        eng = result.scalars().first()
        if not eng:
            raise HTTPException(status_code=404, detail=f"Engagement {engagement_id} not found.")
        if not eng.report_output:
            raise HTTPException(status_code=404, detail="Report not yet generated. POST to /report first.")

        import json as _json
        try:
            return _json.loads(eng.report_output)
        except Exception:
            return {"raw_report": eng.report_output}


@router.post("/engage/{engagement_id}/abort", summary="Abort an active engagement")
async def abort_engagement(engagement_id: str, operator_id: str = "system"):
    async with async_session_maker() as session:
        result = await session.execute(
            select(SecurityEngagement).where(SecurityEngagement.id == engagement_id)
        )
        eng = result.scalars().first()
        if not eng:
            raise HTTPException(status_code=404, detail=f"Engagement {engagement_id} not found.")
        if eng.phase == "COMPLETE":
            raise HTTPException(status_code=409, detail="Engagement already complete.")

        prev_phase = eng.phase
        eng.phase = "ABORTED"
        eng.completed_at = datetime.utcnow()
        await _log_phase(session, engagement_id, "phase_transition", prev_phase, "ABORTED",
                         operator_id, "Engagement manually aborted by operator.")
        await session.commit()

    logger.info(f"[SECURITY] Engagement {engagement_id} ABORTED by {operator_id}")
    return {"message": "Engagement aborted.", "engagement_id": engagement_id}


# ─────────────────────────────────────────────────────────────────────────────
# Background phase runners
# ─────────────────────────────────────────────────────────────────────────────

async def _run_recon_phase(engagement_id: str):
    """Background task: Phase 1+2 (Recon + Analysis) → Phase 3 (Validation)."""
    import json as _json
    from ai.security_orchestrator import run_recon_and_analysis, run_validation

    logger.info(f"[SECURITY][BG] Starting recon for engagement {engagement_id}")

    async with async_session_maker() as session:
        result = await session.execute(
            select(SecurityEngagement).where(SecurityEngagement.id == engagement_id)
        )
        eng = result.scalars().first()
        if not eng:
            logger.error(f"[SECURITY][BG] Engagement {engagement_id} not found")
            return

        try:
            # ── Phase 1+2: Recon & Analysis ────────────────────────────────
            recon_data = await run_recon_and_analysis(
                eng.target_scope, eng.auth_reference_id, eng.engagement_window
            )
            eng.recon_output = recon_data
            await _log_phase(session, engagement_id, "phase_transition", "RECON", "ANALYSIS",
                             "system", f"Recon complete. {len(recon_data.get('hypotheses', []))} hypotheses generated.")
            eng.phase = "ANALYSIS"
            await session.commit()

            # ── Phase 3: Validation ────────────────────────────────────────
            hypotheses = recon_data.get("hypotheses", [])
            validation_results = await run_validation(hypotheses, eng.target_scope, eng.auth_reference_id)
            eng.analysis_output = recon_data  # store with plain_summary

            # Create SecurityFinding records from hypotheses + validation results
            val_map = {v["hypothesis_id"]: v for v in validation_results if isinstance(v, dict)}
            hyp_map = {h["id"]: h for h in hypotheses if isinstance(h, dict)}

            for hyp_id, hyp in hyp_map.items():
                val = val_map.get(hyp_id, {})
                status = val.get("status", "pending")
                finding = SecurityFinding(
                    engagement_id=engagement_id,
                    hypothesis=hyp.get("hypothesis", ""),
                    basis=hyp.get("basis", ""),
                    confidence=hyp.get("confidence", "low"),
                    priority=hyp.get("priority", 3),
                    verification_method=hyp.get("verification_method", "passive"),
                    status=status,
                    validation_evidence=val.get("evidence", ""),
                    validation_reasoning=val.get("reasoning", ""),
                )

                # For needs_active_exploit — set approval gate fields
                if status == "needs_active_exploit_to_confirm":
                    finding.approval_requested_at = datetime.utcnow()
                    finding.proposed_action = f"Active exploit confirmation for: {hyp.get('hypothesis', '')[:200]}"
                    finding.proposed_tool = "SQLMap (safe detection mode) / custom non-destructive probe"
                    finding.risk_level = hyp.get("severity_estimate", "High")

                session.add(finding)

            # Check if any findings need human approval
            needs_approval = [h for h in hypotheses
                              if val_map.get(h.get("id", ""), {}).get("status") == "needs_active_exploit_to_confirm"]

            if needs_approval:
                eng.phase = "AWAITING_APPROVAL"
                await _log_phase(session, engagement_id, "phase_transition", "ANALYSIS", "AWAITING_APPROVAL",
                                 "system",
                                 f"Validation complete. {len(needs_approval)} finding(s) require human approval before active testing. "
                                 f"Use POST /api/security/approve/{{engagement_id}}/{{finding_id}} to approve or deny.")
            else:
                eng.phase = "VALIDATION"
                await _log_phase(session, engagement_id, "phase_transition", "ANALYSIS", "VALIDATION",
                                 "system",
                                 "Validation complete. All hypotheses resolved passively. Ready to generate report.")

            await session.commit()
            logger.info(f"[SECURITY][BG] Engagement {engagement_id} reached phase {eng.phase}")

        except Exception as exc:
            logger.error(f"[SECURITY][BG] Recon phase failed for {engagement_id}: {exc}", exc_info=True)
            async with async_session_maker() as err_session:
                err_result = await err_session.execute(
                    select(SecurityEngagement).where(SecurityEngagement.id == engagement_id)
                )
                err_eng = err_result.scalars().first()
                if err_eng:
                    err_eng.phase = "ABORTED"
                    err_eng.completed_at = datetime.utcnow()
                    await _log_phase(err_session, engagement_id, "phase_transition", "RECON", "ABORTED",
                                     "system", f"Recon phase failed with error: {str(exc)[:500]}")
                    await err_session.commit()


async def _run_report_phase(engagement_id: str):
    """Background task: Phase 5 — ReportAgent generates final report."""
    import json as _json
    from sqlalchemy.orm import selectinload
    from ai.security_orchestrator import run_report

    logger.info(f"[SECURITY][BG] Starting report for engagement {engagement_id}")

    async with async_session_maker() as session:
        # Eagerly load findings to avoid lazy loading errors
        result = await session.execute(
            select(SecurityEngagement)
            .where(SecurityEngagement.id == engagement_id)
            .options(selectinload(SecurityEngagement.findings))
        )
        eng = result.scalars().first()
        if not eng:
            logger.error(f"[SECURITY][BG] Engagement {engagement_id} not found")
            return

        eng.phase = "REPORTING"
        await _log_phase(session, engagement_id, "phase_transition",
                         "AWAITING_APPROVAL", "REPORTING", "system", "Report generation started.")
        await session.commit()

        try:
            # Now findings are loaded, serialize them safely
            findings_data = [_serialise_finding(f) for f in (eng.findings or [])]
            asset_inventory = (eng.recon_output or {}).get("asset_inventory", [])

            report = await run_report(
                findings_data, asset_inventory,
                eng.target_scope, eng.auth_reference_id, eng.engagement_window
            )

            eng.report_output = _json.dumps(report)
            eng.phase = "COMPLETE"
            eng.completed_at = datetime.utcnow()
            await _log_phase(session, engagement_id, "phase_transition",
                             "REPORTING", "COMPLETE", "system", "Report generated successfully.")
            await session.commit()
            logger.info(f"[SECURITY][BG] Engagement {engagement_id} COMPLETE.")

        except Exception as exc:
            logger.error(f"[SECURITY][BG] Report phase failed for {engagement_id}: {exc}", exc_info=True)
            eng.phase = "ABORTED"
            await _log_phase(session, engagement_id, "phase_transition",
                             "REPORTING", "ABORTED", "system", f"Report failed: {str(exc)[:300]}")
            await session.commit()