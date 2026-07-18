# -*- coding: utf-8 -*-
"""
SERA Platform AI Entity Layer -- Comprehensive Smoke Test
Run from backend/ directory: python _smoke_test.py
"""
import os, sys, asyncio, traceback

def main():
    # Force UTF-8 output for Windows cp1252 consoles
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # ---- Setup ----
    os.environ["ENTITY_MODE"] = "live"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    results = []

    def check(label, value, condition, note=""):
        status = PASS if condition else FAIL
        results.append((status, label, value, note))
        print(f"  {status}  {label}: {repr(value)[:80]}  {note}")

    print()
    print("=" * 70)
    print("STEP 3 -- LiveEntity End-to-End Smoke Test")
    print("=" * 70)

    # -- Check 1: Import LiveEntity --
    try:
        from entity_interface.live_entity import LiveEntity
        entity_ai = LiveEntity()
        check("CHECK 1: Import + Instantiate LiveEntity", type(entity_ai).__name__, True)
    except Exception as e:
        check("CHECK 1: Import + Instantiate LiveEntity", str(e), False, "(IMPORT FAILED)")
        traceback.print_exc()
        print("Cannot continue -- bailing out.")
        sys.exit(1)

    # -- Check 2: entity_ai.predict() --
    print()
    try:
        async def run_predict():
            result = await entity_ai.predict(
                entity_id="test-001",
                context={"entropy": 1.2, "event_count": 42, "alert_count": 3, "domain": "financial"}
            )
            return result

        pred = asyncio.run(run_predict())
        actual_keys = set(pred.keys())
        print(f"  [INFO] predict() returned keys: {sorted(actual_keys)}")

        # Map original spec keys -> what the live entity actually returns
        has_prediction   = "prediction" in actual_keys or "transition_type" in actual_keys
        has_confidence   = "confidence" in actual_keys or "success_probability" in actual_keys
        has_drsn         = "drsn_world_state" in actual_keys
        has_sheaf        = "sheaf_top_concepts" in actual_keys
        has_causal_depth = "causal_depth" in actual_keys

        check("CHECK 2a: predict() -> prediction (transition_type)",
              pred.get("transition_type", pred.get("prediction", "MISSING")), has_prediction)
        check("CHECK 2b: predict() -> confidence (success_probability)",
              pred.get("success_probability", pred.get("confidence", "MISSING")), has_confidence)
        check("CHECK 2c: predict() -> drsn_world_state",
              "len=" + str(len(pred.get("drsn_world_state", []))), has_drsn)
        check("CHECK 2d: predict() -> sheaf_top_concepts",
              "len=" + str(len(pred.get("sheaf_top_concepts", []))), has_sheaf)
        check("CHECK 2e: predict() -> causal_depth",
              pred.get("causal_depth", "MISSING"), has_causal_depth)

    except Exception as e:
        traceback.print_exc()
        check("CHECK 2: predict()", str(e), False, "(EXCEPTION)")

    # -- Check 3: _run_internal_training_step() --
    print()
    try:
        entity_ai._run_internal_training_step()
        latest_loss = entity_ai.stats.get("latest_loss")
        is_real_float = (
            latest_loss is not None and
            isinstance(latest_loss, float) and
            latest_loss != 0.0
        )
        check("CHECK 3: _run_internal_training_step() -> latest_loss is real non-zero float",
              latest_loss, is_real_float)
    except Exception as e:
        traceback.print_exc()
        check("CHECK 3: _run_internal_training_step()", str(e), False, "(EXCEPTION)")

    # -- Check 4: get_full_architecture_report() --
    print()
    try:
        report = entity_ai.get_full_architecture_report()
        required_top_keys = {"kronos", "apex", "sheaf", "drsn", "stats"}
        actual_top_keys = set(report.keys())
        missing = required_top_keys - actual_top_keys
        all_present = len(missing) == 0
        all_non_empty = all(bool(report.get(k)) for k in required_top_keys if k in report)

        check("CHECK 4a: Architecture report has all 5 top-level keys",
              sorted(actual_top_keys), all_present,
              "(missing: " + str(missing) + ")" if missing else "")
        check("CHECK 4b: All 5 keys have non-empty data",
              {k: bool(report.get(k)) for k in sorted(required_top_keys)}, all_non_empty)

        # Print sub-details for confirmation
        if "kronos" in report:
            print(f"  [INFO] kronos report: {report['kronos']}")
        if "apex" in report:
            apex_keys = list(report["apex"].keys()) if isinstance(report["apex"], dict) else "non-dict"
            print(f"  [INFO] apex keys: {apex_keys}")
        if "sheaf" in report:
            sheaf_keys = list(report["sheaf"].keys()) if isinstance(report["sheaf"], dict) else "non-dict"
            print(f"  [INFO] sheaf keys: {sheaf_keys}")
        if "drsn" in report:
            drsn_keys = list(report["drsn"].keys()) if isinstance(report["drsn"], dict) else "non-dict"
            print(f"  [INFO] drsn keys: {drsn_keys}")

    except Exception as e:
        traceback.print_exc()
        check("CHECK 4: get_full_architecture_report()", str(e), False, "(EXCEPTION)")

    # ---- Final Summary ----
    print()
    print("=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for s, *_ in results if s == PASS)
    total = len(results)
    for status, label, value, note in results:
        print(f"  {status}  {label}")
    print()
    print(f"  {passed}/{total} checks passed")
    print("=" * 70)

if __name__ == "__main__":
    main()
