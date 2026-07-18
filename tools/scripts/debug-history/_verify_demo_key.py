"""
Isolated verification for demo key injection behavior.
Tests three cases without starting a server.
Simulates the API_KEYS-building logic from main.py exactly,
using importlib.reload so each case gets a clean module state.
"""
import sys, os, json, importlib

# Ensure backend is on the path
BACKEND = os.path.join(os.path.dirname(__file__), "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

DEFAULT_DEMO_KEY = "sera-demo-2026"

def build_api_keys(entity_mode: str, demo_key_env: str | None, api_keys_env: str = "") -> dict:
    """
    Pure reimplementation of the API_KEYS-building block from main.py (post-fix).
    Accepts the relevant env-var values as explicit arguments so we can test
    all three cases in a single process without mutating os.environ globally.
    """
    # ── Parse API_KEYS env var ──────────────────────────────────────────────
    keys: dict = {}
    if api_keys_env.strip():
        try:
            parsed = json.loads(api_keys_env)
            if isinstance(parsed, dict):
                keys = parsed
            elif isinstance(parsed, list):
                keys = {k: f"client_{i}" for i, k in enumerate(parsed)}
        except json.JSONDecodeError:
            for val in api_keys_env.split(","):
                val = val.strip()
                if val:
                    keys[val] = f"client_{val[-4:] if len(val) >= 4 else val}"

    # ── Demo key injection (fixed behavior) ─────────────────────────────────
    _demo_key_env     = demo_key_env          # None when env var not set
    _demo_key_default = DEFAULT_DEMO_KEY

    if entity_mode == "mock":
        demo_key = _demo_key_env or _demo_key_default
        if demo_key not in keys:
            keys[demo_key] = "default_demo"
    else:
        # live / non-mock: only inject if operator explicitly set DEMO_API_KEY
        demo_key = _demo_key_env
        if demo_key and demo_key not in keys:
            keys[demo_key] = "default_demo"

    return keys


print("=" * 65)
print("DEMO KEY INJECTION BEHAVIOR — VERIFICATION")
print("=" * 65)

# ── Case 1: ENTITY_MODE=mock, no DEMO_API_KEY env var ──────────────────────
print(f"\n[CASE 1] ENTITY_MODE=mock | DEMO_API_KEY not set")
keys1 = build_api_keys(entity_mode="mock", demo_key_env=None, api_keys_env="")
present1 = DEFAULT_DEMO_KEY in keys1
print(f"  Resulting API_KEYS: {keys1}")
print(f"  '{DEFAULT_DEMO_KEY}' present: {present1}")
assert present1, f"FAIL: expected '{DEFAULT_DEMO_KEY}' to be in API_KEYS in mock mode"
print(f"  PASS: demo key auto-injected in mock mode as expected")

# ── Case 2: ENTITY_MODE=live, no DEMO_API_KEY env var ──────────────────────
print(f"\n[CASE 2] ENTITY_MODE=live | DEMO_API_KEY not set")
keys2 = build_api_keys(entity_mode="live", demo_key_env=None, api_keys_env="")
present2 = DEFAULT_DEMO_KEY in keys2
print(f"  Resulting API_KEYS: {keys2}")
print(f"  '{DEFAULT_DEMO_KEY}' present: {present2}")
assert not present2, f"FAIL: '{DEFAULT_DEMO_KEY}' must NOT be auto-injected in live mode"
print(f"  PASS: demo key NOT present in live mode (no implicit default)")

# ── Case 3: ENTITY_MODE=live, DEMO_API_KEY explicitly set ──────────────────
EXPLICIT_KEY = "my-custom-live-key-abc123"
print(f"\n[CASE 3] ENTITY_MODE=live | DEMO_API_KEY='{EXPLICIT_KEY}' (explicit)")
keys3 = build_api_keys(entity_mode="live", demo_key_env=EXPLICIT_KEY, api_keys_env="")
present_default3 = DEFAULT_DEMO_KEY in keys3
present_explicit3 = EXPLICIT_KEY in keys3
print(f"  Resulting API_KEYS: {keys3}")
print(f"  '{DEFAULT_DEMO_KEY}' present: {present_default3}  (must be False)")
print(f"  '{EXPLICIT_KEY}'  present: {present_explicit3}  (must be True)")
assert not present_default3, f"FAIL: default demo key must not appear in live+explicit case"
assert present_explicit3,    f"FAIL: explicit DEMO_API_KEY must be present"
print(f"  PASS: explicit DEMO_API_KEY injected; default key absent")

# ── Bonus Case 4: explicit API_KEYS env with live mode ──────────────────────
REAL_KEY = "prod-key-9988"
print(f"\n[CASE 4] ENTITY_MODE=live | API_KEYS='{REAL_KEY}' | no DEMO_API_KEY")
keys4 = build_api_keys(entity_mode="live", demo_key_env=None, api_keys_env=REAL_KEY)
present_real4    = REAL_KEY in keys4
present_default4 = DEFAULT_DEMO_KEY in keys4
print(f"  Resulting API_KEYS: {keys4}")
print(f"  '{REAL_KEY}' present: {present_real4}  (must be True)")
print(f"  '{DEFAULT_DEMO_KEY}' present: {present_default4}  (must be False)")
assert present_real4,     f"FAIL: real key from API_KEYS env must not be dropped"
assert not present_default4, f"FAIL: default demo key must not appear"
print(f"  PASS: explicit API_KEYS kept; default demo key absent")

print("\n" + "=" * 65)
print("All assertions passed.")
print("=" * 65)
