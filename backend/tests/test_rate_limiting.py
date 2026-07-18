"""Regression test for rate limiting behavior.

The app under test mirrors the production middleware ordering so the SlowAPI
limiter executes before the API-key gate, matching the real request path.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import main
from main import APIKeyMiddleware


def _build_rate_limited_app() -> FastAPI:
    """Create a small app with the same middleware order used in production."""

    limiter = Limiter(key_func=main.get_rate_limit_key, default_limits=["60/minute"])
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(SlowAPIMiddleware)

    limited_route = limiter.limit("60/minute")

    @app.get("/limited")
    @limited_route
    def limited_endpoint(request: Request):
        return {"ok": True}

    return app


def test_rate_limiting_cuts_off_after_sixty_requests(monkeypatch):
    """Requests 1-60 should succeed, 61-70 should be rate-limited, and a second client should still work."""

    monkeypatch.setattr(
        main,
        "API_KEYS",
        {
            "rate-limit-alpha": "client-alpha",
            "rate-limit-beta": "client-beta",
        },
        raising=False,
    )

    client = TestClient(_build_rate_limited_app())

    for request_number in range(1, 71):
        response = client.get("/limited", headers={"X-API-Key": "rate-limit-alpha"})
        if request_number <= 60:
            assert response.status_code == 200, f"request {request_number} should have been allowed"
        else:
            assert response.status_code == 429, f"request {request_number} should have been rate-limited"

    other_client_response = client.get("/limited", headers={"X-API-Key": "rate-limit-beta"})
    assert other_client_response.status_code == 200
