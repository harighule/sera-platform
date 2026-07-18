import sys
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import main

def test_api_auth():
    client = TestClient(main.app)
    # Invalid key -> 401
    r = client.get("/", headers={"X-API-Key": "invalid-key"})
    assert r.status_code == 401
    # Missing key -> 401
    r = client.get("/")
    assert r.status_code == 401
    # Valid key -> 200
    r = client.get("/", headers={"X-API-Key": "sera-demo-2026"})
    assert r.status_code == 200

def test_rate_limiting_order():
    # Test rate limiting middleware runs first (meaning 429 happens even for invalid keys)
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from main import APIKeyMiddleware, get_rate_limit_key
    
    lim = Limiter(key_func=get_rate_limit_key, default_limits=["2/minute"])
    app = FastAPI()
    app.state.limiter = lim
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Adding SlowAPIMiddleware after APIKeyMiddleware makes it outermost (runs first)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(SlowAPIMiddleware)
    
    @app.get("/test")
    def test_ep():
        return {"ok": True}
        
    client = TestClient(app)
    
    # Fire 3 requests with an invalid key
    r1 = client.get("/test", headers={"X-API-Key": "bad-key"})
    r2 = client.get("/test", headers={"X-API-Key": "bad-key"})
    r3 = client.get("/test", headers={"X-API-Key": "bad-key"})
    
    assert r1.status_code == 401
    assert r2.status_code == 401
    assert r3.status_code == 429

def test_cors_restrictions():
    client = TestClient(main.app)
    # preflight with DELETE -> 400
    headers_del = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "DELETE",
        "Access-Control-Request-Headers": "X-API-Key"
    }
    r = client.options("/", headers=headers_del)
    assert r.status_code == 400
    
    # preflight with GET -> 200
    headers_get = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-API-Key"
    }
    r = client.options("/", headers=headers_get)
    assert r.status_code == 200
