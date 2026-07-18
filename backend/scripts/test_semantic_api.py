import requests
import json

BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "sera-demo-2026"}

print("======================================================================")
print("TESTING APEX SEMANTIC REASONING REST API")
print("======================================================================")

# Test 1: Homotopy
resp = requests.get(f"{BASE_URL}/api/semantic/homotopy/AAPL/MSFT", headers=HEADERS)
print("Homotopy (AAPL ↔ MSFT):")
print(json.dumps(resp.json(), indent=2))

# Test 2: Causal Chain
resp = requests.get(f"{BASE_URL}/api/semantic/causal-chain/AAPL/MSFT", headers=HEADERS)
print("\nCausal Chain (AAPL → MSFT):")
print(json.dumps(resp.json(), indent=2))

# Test 3: Outgoing
resp = requests.get(f"{BASE_URL}/api/semantic/outgoing/AAPL", headers=HEADERS)
print("\nOutgoing (AAPL):")
print(json.dumps(resp.json(), indent=2))

# Test 4: Companies list
resp = requests.get(f"{BASE_URL}/api/semantic/companies", headers=HEADERS)
print(f"\nCompanies in Graph (Count: {len(resp.json())}):")
print(json.dumps(resp.json()[:3], indent=2)) # Print first 3 companies as a sample
print("======================================================================")
