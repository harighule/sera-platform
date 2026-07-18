"""
ALETHEIA Scoring Engine — Credibility score computation for SERA claims.

Formula:
  S_final = clamp(S_base × decay_factor × (1 - challenge_ratio) × evidence_boost × apex_bonus)

Where:
  S_base          = stake_signal = min(0.3 + stake / 100, 0.9)
  decay_factor    = max(1 - 0.02 × hours_since_reaffirm, 0)
  challenge_ratio = min(total_counter_stake / stake, 1.0) × 0.8
  evidence_boost  = 1 + 0.1 × clamp(weighted_evidence_sum, 0, 5)
  apex_bonus      = 1.15 if apex_verified else 1.0
"""

from datetime import datetime
import math


# ── Evidence weight table by source type ─────────────────────────────────────
EVIDENCE_WEIGHTS = {
    "financial":  1.4,   # audited financials / SEC filings — high authority
    "graph":      1.3,   # APEX Neo4j relationship verified
    "news":       1.0,   # corroborated press / media
    "document":   1.1,   # uploaded documents
    "user":       0.6,   # user-supplied; lower authority
}


def compute_stake_signal(stake_amount: float) -> float:
    """Base credibility from stake: logarithmic growth capped at 0.9."""
    if stake_amount <= 0:
        return 0.0
    return round(min(0.3 + math.log1p(stake_amount) / 10.0, 0.90), 4)


def compute_decay_factor(last_reaffirmed_at: datetime) -> tuple[float, float]:
    """
    Temporal decay: 2% per hour since last reaffirmation.
    Returns (factor, hours_elapsed).
    """
    delta_seconds = (datetime.utcnow() - last_reaffirmed_at).total_seconds()
    hours = max(delta_seconds / 3600.0, 0.0)
    factor = max(1.0 - 0.02 * hours, 0.0)
    return round(factor, 4), round(hours, 2)


def compute_challenge_ratio(total_counter_stake: float, stake_amount: float) -> float:
    """
    Challenge penalty: proportional to counter-stake vs stake.
    Maxes out at 80% reduction even if counter_stake >> stake.
    """
    if stake_amount <= 0 or total_counter_stake <= 0:
        return 0.0
    raw = min(total_counter_stake / stake_amount, 1.0)
    return round(raw * 0.8, 4)


def compute_evidence_boost(evidence_rows: list) -> tuple[float, float]:
    """
    Evidence multiplier: each piece of evidence adds weight up to a ceiling.
    Returns (boost_multiplier, weighted_sum).
    """
    if not evidence_rows:
        return 1.0, 0.0
    total_weight = sum(
        EVIDENCE_WEIGHTS.get(e.get("evidence_type", "user"), 0.6) * e.get("weight", 1.0)
        for e in evidence_rows
    )
    clamped = min(total_weight, 7.0)
    multiplier = 1.0 + 0.08 * clamped
    return round(multiplier, 4), round(total_weight, 4)


def compute_apex_bonus(apex_verified: bool) -> float:
    """APEX causal graph corroboration adds a 15% credibility lift."""
    return 1.15 if apex_verified else 1.0


def compute_final_score(
    stake_amount: float,
    last_reaffirmed_at: datetime,
    total_counter_stake: float,
    evidence_rows: list,
    apex_verified: bool
) -> dict:
    """
    Full ALETHEIA credibility score computation.
    Returns a breakdown dict with all intermediate values and the final score.
    """
    s_base = compute_stake_signal(stake_amount)
    decay_factor, hours_elapsed = compute_decay_factor(last_reaffirmed_at)
    challenge_ratio = compute_challenge_ratio(total_counter_stake, stake_amount)
    evidence_boost, weighted_evidence_sum = compute_evidence_boost(evidence_rows)
    apex_bonus = compute_apex_bonus(apex_verified)

    raw = s_base * decay_factor * (1.0 - challenge_ratio) * evidence_boost * apex_bonus
    final = round(max(min(raw, 1.0), 0.0), 4)

    # Classify status band
    if final >= 0.75:
        status_band = "VERIFIED"
        band_color = "cyan"
    elif final >= 0.50:
        status_band = "CREDIBLE"
        band_color = "blue"
    elif final >= 0.25:
        status_band = "CONTESTED"
        band_color = "amber"
    else:
        status_band = "DISPUTED"
        band_color = "red"

    return {
        "credibility_score": final,
        "status_band": status_band,
        "band_color": band_color,
        "breakdown": {
            "stake_signal": s_base,
            "decay_factor": decay_factor,
            "hours_since_reaffirm": hours_elapsed,
            "challenge_ratio": challenge_ratio,
            "challenge_penalty": round(challenge_ratio, 4),
            "evidence_boost": evidence_boost,
            "weighted_evidence_sum": weighted_evidence_sum,
            "apex_bonus": apex_bonus,
            "apex_verified": apex_verified,
        }
    }
