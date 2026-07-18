"""
Independent verification script:
1. Re-runs the full two-phase training (Sub-Phase A + B) across 10 NEW random seeds (11 to 20).
2. Reports the accuracy and loss distribution across all 10 seeds.
3. Tests predictions on the 10 scenarios (s01 to s10) and reports correctness against ground truth.
4. Explains training set coverage for s04, s06, s07.
5. Revised recommendation on fallback.
"""
import os, sys, asyncio, collections, logging
os.environ["ENTITY_MODE"] = "live"
sys.path.insert(0, "backend")

# Suppress logs for neat output
logging.getLogger("sera.live_entity").setLevel(logging.WARNING)

import torch
import torch.nn.functional as F
from entity_interface.live_entity import LiveEntity, TRANSITION_TYPES
from core.entity_resolution import entity_registry

SEP = "=" * 75

print(f"\n{SEP}")
print("1. EVALUATING THE CORRECTED ARCHITECTURAL INITIALIZATION ACROSS 10 NEW SEEDS")
print(SEP)

# Generate a single static evaluation dataset to compare seeds fairly
eval_f, eval_l, _, _ = LiveEntity._generate_synthetic_labels(n_per_class=100, val_fraction=0.0)

seeds = list(range(11, 21))
results = {}

for seed in seeds:
    print(f"Training with Seed {seed}...")
    torch.manual_seed(seed)
    
    # LiveEntity init automatically runs Phase A (1000 joint) + Phase B (200 clean isolation)
    entity_ai = LiveEntity()
    
    # Evaluate final model
    entity_ai.model.eval()
    with torch.no_grad():
        logits = entity_ai.model(eval_f)["transition_logits"]
        val_loss = F.cross_entropy(logits, eval_l).item()
        val_acc = (logits.argmax(dim=1) == eval_l).float().mean().item()
        
    results[seed] = {"loss": val_loss, "acc": val_acc}
    print(f"  Seed {seed} final: loss={val_loss:.4f}  accuracy={val_acc*100:.1f}%")

print(f"\n{SEP}")
print("SUMMARY OF NEW 10-SEED ACCURACY & LOSS DISTRIBUTION")
print(SEP)
accs = [results[s]["acc"] for s in seeds]
losses = [results[s]["loss"] for s in seeds]

print(f"Accuracies  : {[f'{a*100:.1f}%' for a in accs]}")
print(f"Losses      : {[f'{l:.4f}' for l in losses]}")
print(f"Average Acc : {sum(accs)/len(accs)*100:.1f}%")
print(f"Min Acc     : {min(accs)*100:.1f}%")
print(f"Max Acc     : {max(accs)*100:.1f}%")
print(f"Average Loss: {sum(losses)/len(losses):.4f}")

# ─── Register scenarios for prediction test ───────────────────────────
scenarios = [
    # (eid,                 domain,       entropy, alerts, events,  gt,                    note)
    ("s01-fin-low",        "financial",   0.15,    0,   2, "account_churn",      "recipe 0 exact"),
    ("s02-fin-high",       "financial",   1.70,    8,  10, "credit_default",     "recipe 5 exact (e>1.2, alert=0.80)"),
    ("s03-health-crit",    "healthcare",  1.50,    5,   8, "health_deterioration","recipe 1 exact"),
    ("s04-health-stable",  "healthcare",  0.20,    0,   3, "behavioral_shift",   "* absent from training (hc low-e)"),
    ("s05-iot-fail",       "iot",         1.30,    6,   9, "device_failure",     "recipe 3 exact"),
    ("s06-iot-normal",     "iot",         0.30,    0,   5, "behavioral_shift",   "* absent from training (iot low-e)"),
    ("s07-social-churn",   "social",      0.40,    1,   6, "account_churn",      "* absent from training (social low-e)"),
    ("s08-social-behav",   "social",      1.10,    3,   7, "behavioral_shift",   "recipe 4 exact"),
    ("s09-fin-credit",     "financial",   1.80,   10,  15, "credit_default",     "recipe 5 exact (e>1.2, alert=0.67)"),
    ("s10-fin-mid",        "financial",   0.90,    2,   8, "financial_stress",   "recipe 2 exact (e 0.6-1.2, alert=0.25)"),
]

for eid, domain, entropy, alerts, events, gt, _ in scenarios:
    entity_registry.entities[eid] = {
        "id": eid, "name": eid, "domain": domain,
        "status": "pre-transition", "entropy": entropy,
        "event_count": events, "alert_count": alerts,
    }

# Run scenario evaluation on the model trained under seed 11 (a random new seed)
print(f"\n{SEP}")
print("2. EVALUATING SCENARIOS ON A REPRESENTATIVE MODEL (trained on new Seed 11)")
print(SEP)
torch.manual_seed(11)
eval_entity = LiveEntity()
eval_entity.stats["cifn_classifier_trained"] = True # disable fallback prior for model predictions

col_pred = []
col_match = []
for eid, domain, entropy, alerts, events, gt, note in scenarios:
    r = asyncio.run(eval_entity.predict(eid, {"entropy": entropy}))
    pred = r["prediction"]
    col_pred.append(pred)
    col_match.append(pred == gt)

print(f"\n{'#':<3} {'Scenario':<22} {'GT Label':<26} {'Predicted':<26} {'Match?'}")
print("-" * 85)
for i, (s, pred, match) in enumerate(zip(scenarios, col_pred, col_match), 1):
    eid, domain, entropy, alerts, events, gt, note = s
    flag = "YES" if match else "NO"
    print(f"{i:<3} {eid:<22} {gt:<26} {pred:<26} {flag}")

correct = sum(col_match)
recipe_correct = sum(col_match[i] for i, (_,_,_,_,_,_,n) in enumerate(scenarios) if "recipe" in n)
print(f"\n  Total scenarios correct        : {correct}/10")
print(f"  Recipe-exact scenarios correct : {recipe_correct}/7")
print(f"  Predicted unique classes       : {len(set(col_pred))}/6")
