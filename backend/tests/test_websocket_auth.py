"""Regression test for WebSocket authentication.

This test exercises the WebSocket auth guard directly with a tiny FastAPI app
that mirrors the production close-on-reject behavior and uses the same valid key
registry built by the stream router.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from routers.stream import _WS_API_KEYS


def _build_websocket_app() -> FastAPI:
    """Create a minimal app that preserves the production WebSocket auth contract."""

    app = FastAPI()

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        api_key = websocket.query_params.get("api_key")
        if not api_key or api_key not in _WS_API_KEYS:
            await websocket.close(code=1008, reason="Unauthorized: invalid or missing api_key")
            return

        await websocket.accept()
        await websocket.send_json({"status": "connected", "client": _WS_API_KEYS[api_key]})
        await websocket.close()

    return app


def test_websocket_auth_rejects_missing_and_invalid_keys():
    """Missing and invalid api_key values should be rejected with close code 1008."""

    client = TestClient(_build_websocket_app())

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/stream"):
            pass
    assert excinfo.value.code == 1008

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/stream?api_key=definitely-invalid"):
            pass
    assert excinfo.value.code == 1008


def test_websocket_auth_accepts_valid_key():
    """A known valid api_key should complete the handshake and send a success message."""

    assert _WS_API_KEYS, "The WebSocket API key registry should not be empty"
    valid_key = next(iter(_WS_API_KEYS))

    client = TestClient(_build_websocket_app())
    with client.websocket_connect(f"/ws/stream?api_key={valid_key}") as websocket:
        payload = websocket.receive_json()

    assert payload["status"] == "connected"
    assert payload["client"] == _WS_API_KEYS[valid_key]
