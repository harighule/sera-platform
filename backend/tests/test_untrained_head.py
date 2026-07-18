import sys
import os
import pytest
from entity_interface.live_entity import LiveEntity
from core.entity_resolution import entity_registry

@pytest.mark.anyio
async def test_untrained_head_honesty():
    entity_registry.entities["test-untrained"] = {
        "id": "test-untrained",
        "name": "Untrained Test",
        "domain": "financial",
        "status": "pre-transition",
        "entropy": 1.0,
        "event_count": 10,
        "alert_count": 2,
    }
    le = LiveEntity()
    pred = await le.predict("test-untrained", {})
    
    assert pred["success_probability"] is None
    assert pred["untrained_heuristic"] is True
    assert pred["grounding_source"] == "synthetic_heuristic_recipe"
    
    # Assert counterfactual output has raw_intervened_prob for debugging but raw prediction/confidence is correct
    cf = await le.counterfactual("test-untrained", {"action": "test"})
    assert "raw_intervened_prob" in cf
    assert "confidence" in cf
