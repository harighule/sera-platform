import sys
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from routers.zola import save_prediction_to_db, entity_ai
import main

class BadContextManager:
    async def __aenter__(self):
        raise RuntimeError("Mock DB Connection Failure")
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.anyio
async def test_save_prediction_to_db_failure():
    pred = {"entity_id": "test-entity", "type": "behavioral", "prediction": "credit_default"}
    with patch("routers.zola.async_session_maker", return_value=BadContextManager()):
        res = await save_prediction_to_db(pred)
        assert res is False

@pytest.mark.anyio
async def test_predictions_route_db_failure_surfacing():
    # If DB fails, the GET /api/zola/predictions endpoint should still succeed (200),
    # return the predictions array, and mark "persisted": false.
    client = TestClient(main.app)
    
    with patch("routers.zola.async_session_maker", return_value=BadContextManager()):
        response = client.get("/api/zola/predictions", headers={"X-API-Key": "sera-demo-2026"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for pred in data:
            assert pred["persisted"] is False
            assert "entity_id" in pred
            assert "transition_type" in pred
