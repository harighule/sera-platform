import random
from .base import EntityInterface

TRANSITION_TYPES = ["account_churn", "health_deterioration", "financial_stress",
                    "device_failure", "behavioral_shift", "credit_default"]
MECHANISMS = [
    "Entropy spike in transaction frequency detected over 72h window",
    "Thermodynamic phase transition in behavioral sequence model",
    "Cross-domain signal correlation indicates causal state shift",
    "GNN embedding drift exceeds manifold stability threshold",
]
INTERVENTIONS = [
    "Deploy personalized retention offer within 24 hours",
    "Escalate to clinical care team for preventive consultation",
    "Trigger automated credit line adjustment protocol",
    "Schedule predictive maintenance within 48-hour window",
    "Initiate behavioral nudge via preferred communication channel",
]

class MockEntity(EntityInterface):
    """Mock Entity AI — returns realistic-looking random predictions."""

    async def predict(self, entity_id: str, context: dict) -> dict:
        return {
            "entity_id": entity_id,
            "transition_type": random.choice(TRANSITION_TYPES),
            "causal_mechanism": random.choice(MECHANISMS),
            "optimal_intervention": random.choice(INTERVENTIONS),
            "success_probability": round(random.uniform(0.55, 0.95), 3),
            "recommended_timing": random.choice(["24-48 hours", "1 week", "2-3 days"]),
            "consequence_chain": [
                "Behavioral state normalizes",
                "Cross-domain entropy decreases",
                "Entity re-enters stable manifold region"
            ]
        }

    async def counterfactual(self, entity_id: str, intervention: dict) -> dict:
        return {
            "entity_id": entity_id,
            "intervention": intervention,
            "simulated_outcome": "Positive behavioral realignment",
            "probability_change": round(random.uniform(0.05, 0.30), 3),
            "confidence": round(random.uniform(0.70, 0.95), 3)
        }

    async def get_causal_graph(self, entity_id: str) -> dict:
        return {
            "entity_id": entity_id,
            "nodes": [
                {"id": "financial_stress", "weight": round(random.random(), 3)},
                {"id": "behavioral_anomaly", "weight": round(random.random(), 3)},
                {"id": "state_transition", "weight": round(random.random(), 3)},
            ],
            "edges": [
                {"from": "financial_stress", "to": "behavioral_anomaly", "strength": 0.72},
                {"from": "behavioral_anomaly", "to": "state_transition", "strength": 0.85},
            ]
        }