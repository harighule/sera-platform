import sys
import os
import pytest
from unittest.mock import patch
import torch

from entity_interface.live_entity import LiveEntity
from core.entity_resolution import entity_registry

@pytest.mark.anyio
async def test_confidence_formula():
    entity_registry.entities["test-cf"] = {
        "id": "test-cf",
        "name": "CF Test",
        "domain": "financial",
        "status": "pre-transition",
        "entropy": 1.0,
        "event_count": 10,
        "alert_count": 2,
    }
    le = LiveEntity()
    
    # Assert confidence = abs(intervened_prob - 0.5) * 2 for at least 3 known input/output pairs
    for base_p, int_p, expected_conf in [
        (0.8, 0.5, 0.0),
        (0.2, 0.0, 1.0),
        (0.4, 1.0, 1.0),
        (0.5, 0.75, 0.5),
    ]:
        with patch.object(le.model, 'forward') as mock_forward:
            mock_forward.return_value = {
                "transition_logits": torch.zeros(1, 6),
                "success_prob": torch.tensor([int_p]),
                "optimal_intervention_logits": torch.zeros(1, 5),
                "timing_logits": torch.zeros(1, 3),
            }
            # Mock base model call success_prob inside counterfactual:
            # The method does self.model(features) first, then self.model(intervened_features)
            # So we make mock_forward return base_p then int_p
            mock_forward.side_effect = [
                {"success_prob": torch.tensor([base_p])},
                {"success_prob": torch.tensor([int_p])}
            ]
            res = await le.counterfactual("test-cf", {"action": "test"})
            assert abs(res["confidence"] - expected_conf) < 1e-5
            # Assert confidence is NEVER simply equal to raw intervened_prob (guards against regression)
            if int_p not in (0.0, 1.0):
                assert res["confidence"] != int_p
