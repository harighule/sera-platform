"""
Verification for NOETHER_KRONOS wiring into LiveEntity.

Test 1: USE_NOETHER=true  → type(entity_ai.kronos_model).__name__ == 'NOETHER_KRONOS'
         Run predict() once, confirm no exceptions, print result.

Test 2: USE_NOETHER=false (default) → type(entity_ai.kronos_model).__name__ == 'KRONOS'
         No behavior change from original.

LiveEntity runs an internal 1200-step CIFN training on init, so we mock-out
_train_cifn_classifier and _run_internal_training_step with minimal no-ops to
keep the test fast — we're testing wiring, not training correctness.
"""
import sys, os, types, asyncio, unittest.mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def make_live_entity(use_noether: bool):
    """Instantiate LiveEntity with the specified USE_NOETHER value, patching
    out the slow training methods so the test completes in seconds."""
    import importlib

    # Force-set the config flag before live_entity reads it at import time
    os.environ['USE_NOETHER'] = 'true' if use_noether else 'false'

    # Reload config and live_entity so they pick up the new env var
    import config
    importlib.reload(config)

    # We need to reload live_entity and all modules that import USE_NOETHER from config
    import entity_interface.live_entity as le_mod
    importlib.reload(le_mod)
    LiveEntity = le_mod.LiveEntity

    # Patch the slow init methods to no-ops (we test wiring, not training)
    with (
        unittest.mock.patch.object(LiveEntity, '_run_internal_training_step', lambda self, *a, **kw: None),
        unittest.mock.patch.object(LiveEntity, '_train_cifn_classifier', lambda self: None),
    ):
        entity = LiveEntity()
    return entity, le_mod


print("=" * 65)
print("NOETHER_KRONOS WIRING — VERIFICATION")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# TEST 1: USE_NOETHER=true → NOETHER_KRONOS is active
# ─────────────────────────────────────────────────────────────────
print("\n[TEST 1] USE_NOETHER=true — expect kronos_model = NOETHER_KRONOS")
entity_noether, le_mod = make_live_entity(use_noether=True)
model_class = type(entity_noether.kronos_model).__name__
print(f"  type(entity.kronos_model).__name__ = '{model_class}'")
assert model_class == 'NOETHER_KRONOS', f"FAIL: expected NOETHER_KRONOS, got {model_class}"
print("  PASS: NOETHER_KRONOS is active")

# Confirm architecture_layers reflects NOETHER
arch_layers = entity_noether.stats['architecture_layers']
print(f"  architecture_layers = {arch_layers}")
assert any('NOETHER' in x for x in arch_layers), "FAIL: architecture_layers does not mention NOETHER"
print("  PASS: architecture_layers lists NOETHER-KRONOS-13-Component")

# Confirm get_full_architecture_report
report = entity_noether.get_full_architecture_report()
print(f"  report['kronos']['model']          = {report['kronos']['model']}")
print(f"  report['kronos']['pillars']        = {report['kronos']['pillars']}")
print(f"  report['kronos']['noether_active'] = {report['kronos']['noether_active']}")
assert report['kronos']['model'] == 'NOETHER_KRONOS', "FAIL: architecture report does not say NOETHER_KRONOS"
assert report['kronos']['pillars'] == 13, "FAIL: pillar count should be 13"
assert report['kronos']['noether_active'] is True, "FAIL: noether_active should be True"
print("  PASS: get_full_architecture_report() returns correct NOETHER data")

# Run predict() once to confirm no exceptions
print("\n  Running predict() with NOETHER_KRONOS active...")
async def run_predict():
    # We need at least one entity in the registry for predict to work
    from core.entity_resolution import entity_registry
    if not entity_registry.entities:
        entity_registry.entities['E-TEST0001'] = {
            'id': 'E-TEST0001', 'name': 'Test User', 'domain': 'financial',
            'status': 'pre-transition', 'entropy': 1.2,
            'event_count': 5, 'alert_count': 2,
        }
    result = await entity_noether.predict('E-TEST0001', {'entropy': 1.2})
    return result

result = asyncio.run(run_predict())
print(f"  predict() returned (no exception):")
print(f"    entity_id          : {result.get('entity_id')}")
print(f"    transition_type    : {result.get('transition_type')}")
print(f"    confidence         : {result.get('confidence')}")
print(f"    untrained_heuristic: {result.get('untrained_heuristic')}")
print("  PASS: predict() completed without exception under NOETHER_KRONOS")

# ─────────────────────────────────────────────────────────────────
# TEST 2: USE_NOETHER=false (default) → plain KRONOS, no change
# ─────────────────────────────────────────────────────────────────
print("\n[TEST 2] USE_NOETHER=false (default) — expect kronos_model = KRONOS")
entity_plain, _ = make_live_entity(use_noether=False)
model_class_plain = type(entity_plain.kronos_model).__name__
print(f"  type(entity.kronos_model).__name__ = '{model_class_plain}'")
assert model_class_plain == 'KRONOS', f"FAIL: expected KRONOS, got {model_class_plain}"
print("  PASS: plain KRONOS is active (default behavior unchanged)")

arch_layers_plain = entity_plain.stats['architecture_layers']
print(f"  architecture_layers = {arch_layers_plain}")
assert 'KRONOS-9-Pillar' in arch_layers_plain, "FAIL: architecture_layers should say KRONOS-9-Pillar"
print("  PASS: architecture_layers lists KRONOS-9-Pillar")

report_plain = entity_plain.get_full_architecture_report()
print(f"  report['kronos']['model']          = {report_plain['kronos']['model']}")
print(f"  report['kronos']['pillars']        = {report_plain['kronos']['pillars']}")
print(f"  report['kronos']['noether_active'] = {report_plain['kronos']['noether_active']}")
assert report_plain['kronos']['model'] == 'KRONOS'
assert report_plain['kronos']['pillars'] == 9
assert report_plain['kronos']['noether_active'] is False
print("  PASS: get_full_architecture_report() correctly shows plain KRONOS")

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("All assertions passed.")
print("  USE_NOETHER=true  → NOETHER_KRONOS wired in, predict() runs, report correct")
print("  USE_NOETHER=false → plain KRONOS, no behavior change from before")
print("=" * 65)
