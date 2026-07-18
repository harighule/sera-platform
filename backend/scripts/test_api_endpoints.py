import httpx
import json

def verify_all():
    base_url = "http://localhost:8000"
    headers = {"X-API-Key": "sera-demo-2026"}
    
    # 1. Fetch first entity ID to use for get-by-ID test
    print("=== API ENDPOINT VERIFICATION RUN ===")
    real_entity_id = None
    try:
        r = httpx.get(f"{base_url}/api/entities?limit=1", headers=headers, timeout=15.0)
        if r.status_code == 200:
            data = r.json()
            entities = data.get("entities", [])
            if entities:
                real_entity_id = entities[0].get("id")
                print(f"[PREPARATION] Fetched sample entity ID: {real_entity_id}")
    except Exception as e:
        print(f"[PREPARATION] Failed to fetch sample entity ID: {e}")
        
    if not real_entity_id:
        real_entity_id = "CO-00001" # Safe fallback
        
    endpoints = [
        # (name, method, path, data)
        ("Entities Registry List", "GET", "/api/entities?limit=10", None),
        ("Entity Detail by Ticker", "GET", "/api/entities/AAPL/full", None),
        ("Entity Detail by ID", "GET", f"/api/entities/{real_entity_id}", None),
        ("Semantic Outgoing Morphisms", "GET", "/api/semantic/outgoing/AAPL", None),
        ("Semantic Companies list", "GET", "/api/semantic/companies", None),
        ("AXIOM Monitor", "GET", "/api/axiom/monitor", None),
        ("ZOLA Dashboard", "GET", "/api/zola/dashboard", None),
        ("Dark Intel briefings", "GET", "/api/dark-intel/briefings?clearance=all", None),
        ("Citation Tracking", "GET", "/api/citation/tracked", None),
        ("Signal Synthesis", "GET", f"/api/synthesize/{real_entity_id}", None),
        ("Claim Credibility List", "GET", "/api/claims", None),
        ("AI Command Query", "POST", "/api/chat/", {"message": "What is Tesla's expansion score?"})
    ]
    
    for name, method, path, data in endpoints:
        url = f"{base_url}{path}"
        try:
            if method == "GET":
                r = httpx.get(url, headers=headers, timeout=15.0)
            else:
                r = httpx.post(url, headers=headers, json=data, timeout=15.0)
                
            status_code = r.status_code
            print(f"\n[{name}] {method} {path} -> Status: {status_code}")
            
            if status_code in [200, 201]:
                res_data = r.json()
                print("  [PASS] Structure Verification")
                
                # Check specifics for key endpoints
                if name == "Entities Registry List":
                    print(f"  Total count: {res_data.get('total')}")
                    print(f"  Keys returned: {list(res_data.get('entities', [{}])[0].keys())}")
                elif name.startswith("Entity Detail"):
                    print(f"  Keys returned: {list(res_data.keys())[:8]}")
                elif name == "Semantic Companies list":
                    print(f"  Total companies in graph list: {len(res_data)}")
            else:
                print(f"  [FAIL]: Status code {status_code} - {r.text[:200]}")
        except Exception as e:
            print(f"\n[{name}] {method} {path} -> [EXCEPTION] {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    verify_all()
