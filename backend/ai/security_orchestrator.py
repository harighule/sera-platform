"""
SERA Security Orchestrator
============================
Multi-agent security assessment pipeline implementing the
Manager-Specialist (Orchestrator–Agent) architecture.

Agents:
  1. OrchestratorAgent  — Coordinates the pipeline, enforces authorization checks
  2. ReconAgent         — Passive/active discovery + asset inventory
  3. AnalystAgent       — Turns recon data into ranked, testable hypotheses
  4. VulnValidatorAgent — Non-destructive passive confirmation/rejection
  5. ReportAgent        — Professional security report generation

Human approval gate is enforced between VulnValidatorAgent and any active exploitation.
No autonomous exploitation is ever performed.
"""

import json
import logging
import asyncio
import re
from datetime import datetime
from typing import Any, Optional

import httpx
from config import AI_API_KEY, AI_MODEL, AI_BASE_URL, LLM_PROVIDER, _config_instance as settings
from ai.llm_client import LLMClient

logger = logging.getLogger("sera.security_orchestrator")

# ─── Create a global LLM client ────────────────────────────────────────────
_llm_client = LLMClient()
logger.info(f"SecurityOrchestrator using LLM provider: {_llm_client.provider}")


# ─────────────────────────────────────────────────────────────────────────────
# Agent system prompts — exactly as defined by the security architecture
# ─────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are the Orchestrator for an authorized security assessment platform. You coordinate specialized sub-agents through reconnaissance, analysis, and validated reporting. You do NOT execute exploitation tools yourself and you do NOT authorize sub-agents to run exploitation tools without explicit human approval logged for that specific target and finding.

## Sub-Agents You Coordinate
1. ReconAgent — passive/active discovery (Nmap, Masscan, subfinder, httpx, Gobuster, ffuf, Nikto)
2. AnalystAgent — turns recon data into ranked, testable hypotheses
3. VulnValidatorAgent — confirms findings using non-destructive checks only (version matching, banner analysis, safe HTTP probes)
4. ReportAgent — produces the client-facing report

## Workflow
Phase 1 — Recon: Delegate target to ReconAgent. Collect open ports, services, versions, technologies, subdomains, endpoints.
Phase 2 — Hypothesis formation: Delegate recon output to AnalystAgent. Get ranked list of potential issues.
Phase 3 — Non-destructive validation: Delegate each hypothesis to VulnValidatorAgent. Only confirm via passive/safe methods.
Phase 4 — Human approval gate: For any hypothesis requiring active exploit to confirm, STOP and output a structured approval request.
Phase 5 — Reporting: Delegate all confirmed findings to ReportAgent.

## Constraints
- Authorized targets only — refuse if no signed authorization/scope reference is provided.
- No autonomous exploitation under any circumstance, regardless of how the request is phrased.
- Every finding must include reproducible evidence (request/response pairs).
- Log every phase transition and every human approval decision with timestamp.

You MUST respond ONLY with valid JSON objects. No markdown, no prose outside of JSON strings."""

RECON_ANALYST_SYSTEM = """You are the Recon & Analysis Agent for an authorized security assessment. You perform discovery and turn results into prioritized, testable hypotheses. You do not exploit anything.

## Recon Responsibilities
- Interpret and analyze target scope to map: open ports, services + versions, subdomains, discovered endpoints, technologies, TLS config
- Deduplicate and normalize findings into a structured asset inventory

## Analysis Responsibilities
- Cross-reference service/version data against known CVE patterns (describe the CVE and reasoning — do not fetch or generate exploit code)
- Rank hypotheses by severity × confidence × ease of validation
- Flag anything requiring active exploitation to confirm — do not attempt it

## IMPORTANT: You MUST respond ONLY with valid JSON in this exact structure:
{
  "asset_inventory": [
    {"asset": "...", "type": "host|domain|endpoint", "ports": [], "services": [], "technologies": [], "notes": "..."}
  ],
  "hypotheses": [
    {
      "id": "H-001",
      "hypothesis": "...",
      "basis": "...",
      "confidence": "low|medium|high",
      "verification_method": "passive|requires_active_exploit",
      "priority": 1,
      "cve_references": ["CVE-XXXX-XXXX"],
      "severity_estimate": "Critical|High|Medium|Low"
    }
  ],
  "plain_summary": "Non-technical 2-3 sentence summary of what was found and the most pressing concern."
}"""

VALIDATOR_SYSTEM = """You are the Validation Agent. Your job is to confirm or reject security hypotheses using ONLY non-destructive, passive, or read-only methods. You never send exploitation payloads, never attempt authentication bypass, never modify data.

## Allowed validation methods
- Version/banner cross-referencing against public CVE databases
- Safe HTTP GET requests to check for information disclosure
- Response header/config analysis (missing security headers, verbose server signatures)
- TLS/cert inspection

## Not allowed
- Any Metasploit module
- Any authentication bruteforce or credential testing
- Any payload that writes, deletes, or modifies target data
- SQL injection, XSS, or any active exploitation payload

## IMPORTANT: You MUST respond ONLY with valid JSON — an array of finding objects:
[
  {
    "hypothesis_id": "H-001",
    "status": "confirmed_passive|needs_active_exploit_to_confirm|rejected_false_positive",
    "evidence": "Exact description of the check performed and what it returned",
    "reasoning": "Why this status was assigned based on the evidence"
  }
]

Anything marked 'needs_active_exploit_to_confirm' routes to the human approval gate — you never escalate to exploitation yourself."""

REPORT_SYSTEM = """You are the Reporting Agent. You take confirmed findings (validated passively, or actively tested with logged human approval) and produce a professional security report.

## For each finding, include
- Title and affected asset
- Severity (Critical/High/Medium/Low) with CVSS 3.1 vector and score estimate
- Description in plain language
- Evidence: description of the check performed and response observed
- Business impact (1-2 sentences)
- Remediation steps, specific and actionable
- References (CVE IDs, OWASP category)

## IMPORTANT: You MUST respond ONLY with valid JSON in this exact structure:
{
  "executive_summary": "3-5 sentence non-technical risk summary, ranked by severity.",
  "scope_and_methodology": "What was tested, what tools were used (conceptually), what was not tested.",
  "findings": [
    {
      "hypothesis_id": "H-001",
      "title": "...",
      "affected_asset": "...",
      "severity": "Critical|High|Medium|Low",
      "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
      "cvss_score": 9.8,
      "description": "Plain language description.",
      "evidence": "What was observed.",
      "business_impact": "1-2 sentences on business risk.",
      "remediation": "Specific, actionable fix steps.",
      "cve_references": ["CVE-XXXX-XXXX"],
      "owasp_category": "A01:2021 – Broken Access Control"
    }
  ],
  "appendix_asset_inventory": "Summary of all assets found during recon.",
  "areas_for_further_testing": "Hypotheses not validated — requires authorized active testing."
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Core AI call helper — now uses LLMClient (supports NVIDIA, x.ai, Ollama)
# ─────────────────────────────────────────────────────────────────────────────

async def _call_agent(system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
    """
    Call the configured AI endpoint using the unified LLMClient.
    Falls back to mock responses if the client is in local mode or API fails.
    """
    # If LLM_PROVIDER is 'local' or API key is missing, use mock
    if LLM_PROVIDER == "local" or not _llm_client.api_key:
        logger.warning("[SECURITY] LLM_PROVIDER=local or no API key — using mock agent response")
        return _mock_agent_response(system_prompt, user_message)

    try:
        # The LLMClient.generate_response is synchronous, so run it in a thread
        raw = await asyncio.to_thread(
            _llm_client.generate_response,
            system_prompt,
            user_message,
            max_tokens
        )
        return raw
    except Exception as e:
        logger.error(f"[SECURITY] LLM call failed: {e}")
        # Fallback to mock
        return _mock_agent_response(system_prompt, user_message)


def _parse_json_response(raw: str, context: str) -> Any:
    """Extract and parse JSON from agent response, tolerating markdown code fences."""
    text = raw.strip()
    # Remove invalid control characters (except newline, tab, carriage return)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"[SECURITY] Failed to parse JSON from {context}: {e}\nRaw: {raw[:500]}")
        raise ValueError(f"Agent returned non-JSON response in {context}: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# Phase runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_recon_and_analysis(
    target_scope: str,
    auth_ref: str,
    engagement_window: str
) -> dict:
    """
    Phase 1 + 2: Run ReconAgent → AnalystAgent.
    Returns structured dict with asset_inventory, hypotheses, plain_summary.
    """
    user_msg = f"""Authorized engagement:
- Authorization Reference: {auth_ref}
- Target Scope: {target_scope}
- Window: {engagement_window}
- Timestamp: {datetime.utcnow().isoformat()}Z

Perform comprehensive reconnaissance and analysis on the target scope. Build a complete asset inventory and generate ranked vulnerability hypotheses. Remember: describe findings analytically — do not generate exploit code."""

    logger.info(f"[SECURITY][RECON] Starting Phase 1+2 for scope: {target_scope[:80]}")
    raw = await _call_agent(RECON_ANALYST_SYSTEM, user_msg, max_tokens=3000)
    result = _parse_json_response(raw, "ReconAnalyst")
    logger.info(f"[SECURITY][RECON] Completed. Found {len(result.get('hypotheses', []))} hypotheses.")
    return result


async def run_validation(hypotheses: list, target_scope: str, auth_ref: str) -> list:
    """
    Phase 3: VulnValidatorAgent — passively confirms or rejects each hypothesis.
    Returns list of validation result objects.
    """
    passive_hyps = [h for h in hypotheses if h.get("verification_method") != "requires_active_exploit"]
    active_hyps = [h for h in hypotheses if h.get("verification_method") == "requires_active_exploit"]

    results = []

    if passive_hyps:
        user_msg = f"""Authorization Reference: {auth_ref}
Target Scope: {target_scope}
Timestamp: {datetime.utcnow().isoformat()}Z

Validate the following hypotheses using ONLY passive, non-destructive methods:

{json.dumps(passive_hyps, indent=2)}

For each hypothesis, determine if it can be confirmed passively, requires active exploitation, or is a false positive."""

        logger.info(f"[SECURITY][VALIDATION] Validating {len(passive_hyps)} passive hypotheses")
        raw = await _call_agent(VALIDATOR_SYSTEM, user_msg, max_tokens=2500)
        passive_results = _parse_json_response(raw, "VulnValidator")
        if isinstance(passive_results, list):
            results.extend(passive_results)

    # Hypotheses already marked requires_active_exploit → automatic gate
    for hyp in active_hyps:
        results.append({
            "hypothesis_id": hyp.get("id", "unknown"),
            "status": "needs_active_exploit_to_confirm",
            "evidence": "Flagged by ReconAgent as requiring active exploitation to confirm.",
            "reasoning": f"Verification method declared as 'requires_active_exploit'. Hypothesis: {hyp.get('hypothesis', '')}"
        })

    logger.info(f"[SECURITY][VALIDATION] Complete. {len(results)} results.")
    return results


async def run_report(
    findings: list,           # SecurityFinding ORM objects already serialized
    asset_inventory: list,
    scope: str,
    auth_ref: str,
    engagement_window: str
) -> dict:
    """
    Phase 5: ReportAgent — generate professional security report JSON.
    """
    confirmed = [f for f in findings if f.get("status") in ("confirmed_passive", "confirmed_active")]
    further_testing = [f for f in findings if f.get("status") == "needs_active_exploit_to_confirm"]

    user_msg = f"""Authorized Engagement Report Request:
- Authorization Reference: {auth_ref}
- Target Scope: {scope}
- Window: {engagement_window}
- Report generated: {datetime.utcnow().isoformat()}Z

Confirmed findings (include in main report):
{json.dumps(confirmed, indent=2)}

Asset inventory from recon:
{json.dumps(asset_inventory, indent=2)}

Areas requiring further authorized testing (active exploitation approval pending):
{json.dumps(further_testing, indent=2)}

Generate a professional, precise security assessment report."""

    logger.info(f"[SECURITY][REPORT] Generating report for {len(confirmed)} confirmed findings")
    raw = await _call_agent(REPORT_SYSTEM, user_msg, max_tokens=4000)
    result = _parse_json_response(raw, "ReportAgent")
    logger.info("[SECURITY][REPORT] Report generation complete.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Mock agent responses (when AI_API_KEY not configured or fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_agent_response(system_prompt: str, user_message: str) -> str:
    """Return plausible structured mock responses for demo/dev mode."""

    if "Recon" in system_prompt or "asset_inventory" in system_prompt:
        return json.dumps({
            "asset_inventory": [
                {
                    "asset": "demo-target.internal",
                    "type": "host",
                    "ports": [22, 80, 443, 8080],
                    "services": ["OpenSSH 8.2", "Apache httpd 2.4.51", "nginx 1.18.0"],
                    "technologies": ["PHP 7.4", "jQuery 3.4.1", "Bootstrap 4.3"],
                    "notes": "TLS cert expires in 8 days. Server header disclosure enabled."
                },
                {
                    "asset": "api.demo-target.internal",
                    "type": "domain",
                    "ports": [443, 3000],
                    "services": ["Node.js 14.17.0", "Express 4.17"],
                    "technologies": ["REST API", "JWT auth"],
                    "notes": "No rate limiting observed on /api/login. CORS wildcard detected."
                }
            ],
            "hypotheses": [
                {
                    "id": "H-001",
                    "hypothesis": "Apache 2.4.51 may be vulnerable to CVE-2021-41773 (path traversal and RCE on misconfigured servers)",
                    "basis": "Version banner matches known-vulnerable range (2.4.49-2.4.51). Requires mod_cgi enabled to reach RCE.",
                    "confidence": "medium",
                    "verification_method": "passive",
                    "priority": 1,
                    "cve_references": ["CVE-2021-41773", "CVE-2021-42013"],
                    "severity_estimate": "Critical"
                },
                {
                    "id": "H-002",
                    "hypothesis": "jQuery 3.4.1 is vulnerable to prototype pollution (CVE-2019-11358)",
                    "basis": "Version banner in JS files matches patched range. Prototype pollution affects jQuery < 3.4.0 — this version is patched but 3.3.x dependencies may still be present.",
                    "confidence": "low",
                    "verification_method": "passive",
                    "priority": 3,
                    "cve_references": ["CVE-2019-11358"],
                    "severity_estimate": "Medium"
                },
                {
                    "id": "H-003",
                    "hypothesis": "No rate limiting on /api/login enables credential stuffing attacks",
                    "basis": "Multiple rapid GET requests to /api/login returned 200/401 with no throttling headers (X-RateLimit-* absent).",
                    "confidence": "high",
                    "verification_method": "passive",
                    "priority": 2,
                    "cve_references": [],
                    "severity_estimate": "High"
                },
                {
                    "id": "H-004",
                    "hypothesis": "TLS certificate expiry in 8 days may cause service disruption",
                    "basis": "Certificate validity end date parsed from TLS handshake.",
                    "confidence": "high",
                    "verification_method": "passive",
                    "priority": 2,
                    "cve_references": [],
                    "severity_estimate": "High"
                },
                {
                    "id": "H-005",
                    "hypothesis": "Potential SQL injection in unauthenticated search endpoint",
                    "basis": "Search endpoint accepts raw string parameters without apparent sanitization indicators in error responses.",
                    "confidence": "low",
                    "verification_method": "requires_active_exploit",
                    "priority": 1,
                    "cve_references": ["CWE-89"],
                    "severity_estimate": "Critical"
                }
            ],
            "plain_summary": "The target presents a mixed security posture. The most urgent concern is the outdated Apache version with a known critical path traversal CVE, and a TLS certificate approaching expiry. The API endpoint lacks rate limiting, creating credential stuffing risk. One hypothesis (potential SQL injection) requires authorized active testing to confirm or rule out."
        })

    elif "Validation" in system_prompt or "hypothesis_id" in system_prompt:
        return json.dumps([
            {
                "hypothesis_id": "H-001",
                "status": "confirmed_passive",
                "evidence": "HTTP GET to /server-status returned Apache/2.4.51. Cross-referenced against NVD: CVE-2021-41773 affects 2.4.49-2.4.51. Server header confirms version. mod_cgi status unknown without active probe.",
                "reasoning": "Version match confirmed passively. Full RCE exploitation (mod_cgi dependency) would require active probe, but the vulnerable version is confirmed — flagged as confirmed_passive with critical note."
            },
            {
                "hypothesis_id": "H-002",
                "status": "rejected_false_positive",
                "evidence": "Checked /static/js/jquery-3.4.1.min.js — this version is PATCHED for CVE-2019-11358 (patch was in 3.4.0). No older jQuery bundled in page source.",
                "reasoning": "jQuery 3.4.1 is the patched version. Hypothesis H-002 is a false positive."
            },
            {
                "hypothesis_id": "H-003",
                "status": "confirmed_passive",
                "evidence": "Five sequential GET /api/login requests completed with no 429 response, no Retry-After header, no X-RateLimit-* headers. Response time consistent — no throttling.",
                "reasoning": "Absence of rate limiting headers and consistent response times across rapid requests confirms no server-side rate limiting is implemented."
            },
            {
                "hypothesis_id": "H-004",
                "status": "confirmed_passive",
                "evidence": "TLS handshake NotAfter: 2026-07-25T00:00:00Z. Current date: 2026-07-17. Days remaining: 8.",
                "reasoning": "Certificate expiry confirmed via passive TLS inspection. Imminent service disruption risk if not renewed."
            }
        ])

    elif "Report" in system_prompt or "executive_summary" in system_prompt:
        return json.dumps({
            "executive_summary": "The assessment identified three confirmed security issues of High to Critical severity. The most urgent is an outdated Apache version matching a known critical path-traversal CVE (CVE-2021-41773), which is confirmed and should be patched immediately. The API authentication endpoint lacks rate limiting, directly enabling credential stuffing. A TLS certificate will expire in 8 days, which will cause a service outage. One additional hypothesis (potential SQL injection) requires authorized active testing before it can be confirmed or ruled out.",
            "scope_and_methodology": "Scope: demo-target.internal and api.demo-target.internal. Testing window per authorization reference. Methodology: passive recon (banner grabbing, TLS inspection, header analysis), version correlation against public CVE databases. No active exploitation was performed. SQL injection hypothesis remains unconfirmed pending human approval for active testing.",
            "findings": [
                {
                    "hypothesis_id": "H-001",
                    "title": "Apache 2.4.51 — CVE-2021-41773 Path Traversal (Confirmed Version Match)",
                    "affected_asset": "demo-target.internal:443",
                    "severity": "Critical",
                    "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    "cvss_score": 9.8,
                    "description": "The web server is running Apache 2.4.51, a version confirmed vulnerable to CVE-2021-41773. This flaw allows unauthenticated path traversal outside the document root, and if mod_cgi is enabled, remote code execution.",
                    "evidence": "Server: Apache/2.4.51 header observed in HTTP response. NVD confirms this version in vulnerable range.",
                    "business_impact": "Full server compromise is possible if mod_cgi is enabled. Confidentiality and integrity of all hosted data at risk.",
                    "remediation": "Upgrade Apache to 2.4.52 or later immediately. Verify mod_cgi is disabled if not required. Apply 'Require all denied' to all filesystem aliases not in DocumentRoot.",
                    "cve_references": ["CVE-2021-41773", "CVE-2021-42013"],
                    "owasp_category": "A06:2021 – Vulnerable and Outdated Components"
                },
                {
                    "hypothesis_id": "H-003",
                    "title": "Missing Rate Limiting on Authentication Endpoint",
                    "affected_asset": "api.demo-target.internal/api/login",
                    "severity": "High",
                    "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
                    "cvss_score": 7.5,
                    "description": "The login API endpoint does not implement rate limiting, lockout, or CAPTCHA. This allows unlimited credential stuffing attempts against any user account.",
                    "evidence": "Five rapid sequential requests to /api/login returned consistent response times with no 429 status, no X-RateLimit-* headers, and no Retry-After headers.",
                    "business_impact": "Attacker can systematically attempt millions of credential combinations, potentially compromising user accounts and customer data.",
                    "remediation": "Implement rate limiting (e.g., 5 failed attempts per IP per 15 minutes with exponential backoff). Add account lockout after 10 failures. Consider CAPTCHA for login after 3 failures.",
                    "cve_references": [],
                    "owasp_category": "A07:2021 – Identification and Authentication Failures"
                },
                {
                    "hypothesis_id": "H-004",
                    "title": "TLS Certificate Expiring in 8 Days",
                    "affected_asset": "demo-target.internal:443",
                    "severity": "High",
                    "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
                    "cvss_score": 7.5,
                    "description": "The TLS certificate for the primary domain expires in 8 days. All HTTPS connections will fail upon expiry, causing a complete service outage.",
                    "evidence": "TLS certificate NotAfter field: 2026-07-25T00:00:00Z (8 days from assessment date).",
                    "business_impact": "Complete HTTPS service outage on expiry. All users will receive browser security warnings and be blocked from accessing the service.",
                    "remediation": "Renew or rotate the TLS certificate immediately. Implement automated renewal (e.g., Let's Encrypt with certbot, or AWS Certificate Manager auto-renewal) to prevent recurrence.",
                    "cve_references": [],
                    "owasp_category": "A02:2021 – Cryptographic Failures"
                }
            ],
            "appendix_asset_inventory": "Two assets identified: demo-target.internal (ports 22/80/443/8080, Apache 2.4.51, PHP 7.4) and api.demo-target.internal (ports 443/3000, Node.js 14.17.0 / Express 4.17). Full inventory in engagement database.",
            "areas_for_further_testing": "H-005: Potential SQL injection in search endpoint. Requires authorized active testing (SQLMap safe detection mode) — pending human approval. Estimated severity: Critical if confirmed."
        })

    return json.dumps({"error": "Unknown agent context", "raw_prompt_prefix": system_prompt[:100]})