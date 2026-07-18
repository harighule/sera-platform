import sys
import os
import pytest
from entity_interface.live_entity import LiveEntity
from core.entity_resolution import entity_registry

@pytest.mark.anyio
async def test_apex_entity_specific():
    # Two entities with sharply different inputs
    entity_registry.entities["apex-t1"] = {
        "id": "apex-t1", "name": "t1", "domain": "financial", "status": "pre-transition",
        "entropy": 0.05, "event_count": 2, "alert_count": 0,
    }
    entity_registry.entities["apex-t2"] = {
        "id": "apex-t2", "name": "t2", "domain": "financial", "status": "pre-transition",
        "entropy": 2.95, "event_count": 200, "alert_count": 20,
    }
    
    le = LiveEntity()
    le.stats["cifn_classifier_trained"] = True
    
    g1 = await le.get_causal_graph("apex-t1")
    g2 = await le.get_causal_graph("apex-t2")
    
    assert g1["graph_scope"] == "entity_specific_activations"
    assert g2["graph_scope"] == "entity_specific_activations"
    
    w1 = [n["weight"] for n in g1["nodes"]]
    w2 = [n["weight"] for n in g2["nodes"]]
    
    # Assert weights are not identical between the two entities
    assert w1 != w2
