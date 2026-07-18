import logging
import random
from core.entity_resolution import entity_registry
from core.entropy_engine import entropy_engine

logger = logging.getLogger(__name__)

# ── Continuous-learning state (persists across calls / instances) ──────────
# The signal manufacturer tracks each source signal against its subsequent
# realised outcome and re-weights the sources by their historical predictive
# reliability — the "continuous learning engine" from the spec. Priors match
# the original fixed 0.5 / 0.3 / 0.2 blend, so behaviour is unchanged until
# outcomes are recorded via record_outcome().
_SIGNAL_PRIORS = {"model": 0.5, "entropy": 0.3, "experience": 0.2}
_SIGNAL_RELIABILITY = {"model": 0.5, "entropy": 0.5, "experience": 0.5}  # EMA accuracy
_LAST_SIGNALS: dict = {}       # entity_id -> {"model":s1,"entropy":s2,"experience":s3}
_OUTCOME_COUNT = {"n": 0}


def _adaptive_weights() -> dict:
    """Blend the priors with learned per-signal reliability, renormalised."""
    raw = {k: _SIGNAL_PRIORS[k] * _SIGNAL_RELIABILITY[k] for k in _SIGNAL_PRIORS}
    z = sum(raw.values()) or 1.0
    return {k: raw[k] / z for k in raw}


class SignalSynthesizer:
    """
    Synthesizes multiple signal sources for an entity into a unified intelligence output.
    Combines model prediction confidence, entropy anomaly scores, and data stream density,
    weighted by each source's continuously-learned predictive reliability.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def record_outcome(entity_id: str, realized_outcome: float) -> dict:
        """
        Continuous-learning update: given the realised outcome in [0,1] for an
        entity that was previously synthesized, update each signal's reliability
        (EMA of its accuracy = 1 − |signal − outcome|) and thereby the adaptive
        weights. This is the outcome-feedback loop: signals that historically
        predicted outcomes better get more weight over time.
        """
        prev = _LAST_SIGNALS.get(entity_id)
        if prev is None:
            return {"updated": False, "reason": "no prior synthesize() for entity"}
        realized = max(0.0, min(1.0, float(realized_outcome)))
        beta = 0.2
        for k in _SIGNAL_RELIABILITY:
            accuracy = 1.0 - abs(prev[k] - realized)         # ∈ [0,1]
            _SIGNAL_RELIABILITY[k] = (1 - beta) * _SIGNAL_RELIABILITY[k] + beta * accuracy
        _OUTCOME_COUNT["n"] += 1
        return {
            "updated": True,
            "outcomes_recorded": _OUTCOME_COUNT["n"],
            "signal_reliability": {k: round(v, 4) for k, v in _SIGNAL_RELIABILITY.items()},
            "adaptive_weights": {k: round(v, 4) for k, v in _adaptive_weights().items()},
        }

    async def synthesize(self, entity_id: str) -> dict:
        # 1. Resolve entity
        entity = entity_registry.get_by_id(entity_id)
        if not entity:
            logger.error(f"Entity resolution failed for entity_id: {entity_id}")
            return {}

        # 2. Get entropy score (fall back to resolved entity baseline if sliding window is empty)
        entropy_val = entropy_engine.get_entity_entropy(entity_id)
        if entropy_val <= 0.0:
            entropy_val = float(entity.get("entropy", 0.0))

        # 3. Get prediction and model confidence
        from routers.zola import entity_ai
        
        pred_dict = await entity_ai.predict(entity_id, {})
        model_confidence = float(pred_dict.get("confidence", 0.0))
        prediction = pred_dict.get("prediction", "unknown")

        # 4. Extract stream density
        event_count = float(entity.get("event_count", 0))

        # 5. Signal combination formula:
        # s1: Model Confidence (PyTorch softmax margin) - scale [0, 1]
        # s2: Entropy Score contribution (normalized to 1.5 max) - scale [0, 1]
        # s3: Experience/Data Density contribution (event count normalized to 100 max) - scale [0, 1]
        s1 = model_confidence
        s2 = min(entropy_val / 1.5, 1.0)
        s3 = min(event_count / 100.0, 1.0)

        # Adaptive weighting: priors (0.5/0.3/0.2) modulated by each signal's
        # continuously-learned predictive reliability (see record_outcome()).
        w = _adaptive_weights()
        synthesized_confidence = w["model"] * s1 + w["entropy"] * s2 + w["experience"] * s3
        synthesized_confidence = round(synthesized_confidence, 4)

        # Store per-signal values so a later record_outcome() can score them.
        _LAST_SIGNALS[entity_id] = {"model": s1, "entropy": s2, "experience": s3}

        # 6. Determine dominant contributor
        contrib_s1 = 0.5 * s1
        contrib_s2 = 0.3 * s2
        contrib_s3 = 0.2 * s3

        max_contrib = max(contrib_s1, contrib_s2, contrib_s3)
        if max_contrib == contrib_s1:
            dominant_source = "Model Confidence"
            dominant_contrib = contrib_s1
        elif max_contrib == contrib_s2:
            dominant_source = "Entropy Anomaly"
            dominant_contrib = contrib_s2
        else:
            dominant_source = "Data Experience"
            dominant_contrib = contrib_s3

        explanation = (
            f"Synthesized confidence is primarily driven by {dominant_source} "
            f"(weight-adjusted contribution: {dominant_contrib:.4f})."
        )

        return {
            "entity_id": entity_id,
            "entity_name": entity.get("name", "Unknown"),
            "domain": entity.get("domain", "unknown"),
            "prediction": prediction,
            "synthesized_confidence": synthesized_confidence,
            "contributing_signals": {
                "model_confidence": round(s1, 4),
                "entropy_anomaly": round(s2, 4),
                "data_experience": round(s3, 4),
                "raw_entropy": round(entropy_val, 4),
                "raw_event_count": int(event_count)
            },
            "dominant_contributor": dominant_source,
            "explanation": explanation,
            "proof_of_concept_note": (
                "PROOF OF CONCEPT: This is a synthesized confidence of 3 existing internal signals "
                "(PyTorch CIFN margin, AXIOM-Φ entropy, and stream event counts). It does NOT integrate "
                "external alternative data panels (satellite, credit card panels, etc.) mentioned in the concept roadmaps."
            )
        }
