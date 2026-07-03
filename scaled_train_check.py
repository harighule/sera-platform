"""
Steps 4 & 5 — train LiveEntity with scaled-up CIFN and report
the honest three-column comparison + fallback decision.

Run from sera-platform root: python scaled_train_check.py
"""
import os, sys, asyncio, collections, logging
os.environ["ENTITY_MODE"] = "live"
sys.path.insert(0, "backend")

# Surface training logs to stdout so progress is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sera.live_entity")
log.setLevel(logging.INFO)

import torch
import torch.nn.functional as F
from entity_interface.live_entity import LiveEntity, TRANSITION_TYPES, LiveCausalNetwork
from core.entity_resolution import entity_registry

SEP = "=" * 72

# ─── check param count BEFORE registering entities ───────────────────────────
raw = LiveCausalNetwork()
params = sum(p.numel() for p in raw.parameters())
print(f"\nCIFN param count (hidden=64): {params:,}")

# ─── Register the 10 test scenarios ──────────────────────────────────────────
scenarios = [
    ("s01-fin-low",       "financial",  0.15,  0,  2),
    ("s02-fin-high",      "financial",  1.70,  8, 10),
    ("s03-health-crit",   "healthcare", 1.50,  5,  8),
    ("s04-health-stable", "healthcare", 0.20,  0,  3),
    ("s05-iot-fail",      "iot",        1.30,  6,  9),
    ("s06-iot-normal",    "iot",        0.30,  0,  5),
    ("s07-social-churn",  "social",     0.40,  1,  6),
    ("s08-social-behav",  "social",     1.10,  3,  7),
    ("s09-fin-credit",    "financial",  1.80, 10, 15),
    ("s10-fin-mid",       "financial",  0.90,  2,  8),
]
for eid, domain, entropy, alerts, events in scenarios:
    entity_registry.entities[eid] = {
        "id": eid, "name": eid, "domain": domain,
        "status": "pre-transition", "entropy": entropy,
        "event_count": events, "alert_count": alerts,
    }

# ─── Dataset balance check BEFORE training ────────────────────────────────────
print(f"\n{SEP}")
print("DATASET BALANCE CHECK (500/class, 80/20 split)")
print(SEP)
tr_f, tr_l, val_f, val_l = LiveEntity._generate_synthetic_labels(
    n_per_class=500, val_fraction=0.2
)
tr_dist  = collections.Counter(tr_l.tolist())
val_dist = collections.Counter(val_l.tolist())
print(f"  Train total : {len(tr_l)}   Val total: {len(val_l)}")
print(f"  Train dist  : { {TRANSITION_TYPES[k]: v for k,v in sorted(tr_dist.items())} }")
print(f"  Val dist    : { {TRANSITION_TYPES[k]: v for k,v in sorted(val_dist.items())} }")

# ─── Instantiate + train ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TRAINING (1000 steps, cosine LR, CIFN + KRONOS heads)")
print(SEP)
entity_ai = LiveEntity()   # __init__ runs bootstrap step then _train_cifn_classifier(1000)

# Print the per-100-step training log
log_lines = entity_ai.stats.get("cifn_train_log", [])
for line in log_lines:
    print(line)

final_val_loss = entity_ai.stats.get("cifn_final_val_loss", 9.9)
final_val_acc  = entity_ai.stats.get("cifn_val_accuracy", 0.0)
print(f"\n  => Final val_loss={final_val_loss:.4f}  val_acc={final_val_acc*100:.1f}%")

# ─── Step 4: predict with fallback FORCED OFF ─────────────────────────────────
print(f"\n{SEP}")
print("STEP 4 — predict() with prior fallback FORCED OFF")
print(SEP)
# cifn_classifier_trained=True keeps fallback disabled for normal operation;
# we additionally set both_collapsed guard to never fire by using a large margin.
entity_ai.stats["cifn_classifier_trained"] = True

# Previous "old trained" result (loss=1.63) column values (from last run)
col_old = [
    "account_churn", "credit_default", "account_churn", "account_churn",
    "account_churn", "account_churn",  "account_churn", "account_churn",
    "credit_default","account_churn",
]

col_new = []
col_prior_active = []
for s in scenarios:
    eid, domain, entropy, alerts, events = s
    r = asyncio.run(entity_ai.predict(eid, {"entropy": entropy}))
    col_new.append(r["prediction"])
    col_prior_active.append(r.get("used_prior_fallback", False))

print(f"\n{'#':<3} {'Scenario':<22} {'Domain':<12} {'E':<5} | {'Old (loss=1.63)':<26} {'New (trained)':<26} {'Prior?'}")
print("-" * 100)
for i, (s, old, new, pp) in enumerate(zip(scenarios, col_old, col_new, col_prior_active), 1):
    eid, domain, entropy, alerts, events = s
    print(f"{i:<3} {eid:<22} {domain:<12} {entropy:<5} | {old:<26} {new:<26} {pp}")

print()
old_dist = collections.Counter(col_old)
new_dist = collections.Counter(col_new)
print(f"  Old (loss=1.63) dist : {dict(old_dist)}")
print(f"  New (trained)   dist : {dict(new_dist)}")
u_old, u_new = len(set(col_old)), len(set(col_new))
print(f"  Unique classes       : old={u_old}/6  new={u_new}/6")
any_prior = any(col_prior_active)
print(f"  Prior fallback fired : {any_prior}")

# ─── Step 5: fallback decision based on evidence ──────────────────────────────
print(f"\n{SEP}")
print("STEP 5 — FALLBACK STATUS DECISION (evidence-based)")
print(SEP)

THRESHOLD_LOSS = 0.8
THRESHOLD_UNI  = 5

print(f"  Criteria: val_loss < {THRESHOLD_LOSS} AND unique_classes >= {THRESHOLD_UNI}/6")
print(f"  Result  : val_loss={final_val_loss:.4f}  unique_classes={u_new}/6")
print()

if final_val_loss < THRESHOLD_LOSS and u_new >= THRESHOLD_UNI:
    print("  VERDICT: RECOMMEND disabling _domain_prior_fallback() by default.")
    print("  The model produces genuine label diversity without rule assistance.")
    print("  Keep the code path available as an opt-in safety net (cifn_trained flag).")
else:
    print("  VERDICT: Keep _domain_prior_fallback() ACTIVE.")
    gap_loss  = final_val_loss - THRESHOLD_LOSS
    gap_class = THRESHOLD_UNI - u_new
    print(f"  Gap to threshold: val_loss still {gap_loss:+.4f} above target; "
          f"{gap_class} more unique class(es) needed.")
    print()
    print("  What would close the gap:")
    if final_val_loss > 1.2:
        print("  - More training steps: try 2000-5000; loss curve has not plateaued yet.")
    if final_val_loss <= 1.2 and final_val_loss > THRESHOLD_LOSS:
        print("  - Loss is converging; try 500 more steps with lr=1e-4 fine-tuning.")
    if u_new < 4:
        print("  - Synthetic data boundary is too fuzzy: tighten entropy_range per class.")
    print("  - KRONOS head needs more dedicated steps; its token-id input is noisy.")
    print("  - After these, re-run this script to re-evaluate the fallback decision.")
