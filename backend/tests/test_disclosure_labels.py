import sys
import os
import pytest
from unittest.mock import patch
from entity_interface.live_entity import LiveEntity
from core.entity_resolution import entity_registry
from fastapi.testclient import TestClient
import main

@pytest.mark.anyio
async def test_disclosure_labels():
    entity_registry.entities["disc-t"] = {
        "id": "disc-t", "name": "disc", "domain": "financial", "status": "pre-transition",
        "entropy": 1.0, "event_count": 10, "alert_count": 2,
    }
    le = LiveEntity()
    
    # 1. predict() response contains verification_score_source and sheaf_grounding_source
    pred = await le.predict("disc-t", {})
    assert pred["verification_score_source"] == "kronos_untrained_synthetic"
    # CSIE is now grounded on the REAL 9-pillar KRONOS logits (trained on
    # synthetic next-token data — the "synthetic" disclosure is preserved).
    assert pred["sheaf_grounding_source"] == "kronos_9pillar_logits_synthetic"
    
    # 2. godel endpoints contain fitness_type
    client = TestClient(main.app)
    with patch("routers.zola.entity_ai", le):
        r = client.get("/api/zola/godel/auto/status", headers={"X-API-Key": "sera-demo-2026"})
        assert r.status_code == 200
        assert r.json()["fitness_type"] == "structural_topology_only_no_task_data"
    
    # 3. get_full_architecture_report()'s top_5_morphisms entries contain morphism_source
    report = le.get_full_architecture_report()
    apex = report["apex"]
    assert apex["morphism_source"] == "cifn_activation_derived"
    for m in apex["top_5_morphisms"]:
        assert m["morphism_source"] == "cifn_activation_derived"
