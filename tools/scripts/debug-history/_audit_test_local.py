"""
SERA Platform Independent Audit Test Script
Run from the 'sera-platform' directory: python _audit_test_local.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("=" * 70)
print("SERA PLATFORM — INDEPENDENT AUDIT TEST")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# TEST 1: EntropyEngine — real Shannon entropy or constant?
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 1] EntropyEngine._compute_entropy — two distinct inputs")
from core.entropy_engine import EntropyEngine

ee = EntropyEngine(window_size=10, alert_threshold=2.0)

# Input A: single repeated signal → low entropy
for _ in range(5):
    ee.entity_windows["E1"].append("REST:login")
# Input B: diverse signals → high entropy
for sig in ["REST:login", "gRPC:buy", "WS:alert", "REST:logout", "gRPC:sell"]:
    ee.entity_windows["E2"].append(sig)

h1 = ee._compute_entropy("E1")
h2 = ee._compute_entropy("E2")
print(f"  Input A (5x same signal): entropy = {h1}")
print(f"  Input B (5x diverse):     entropy = {h2}")
assert h1 != h2, "FAIL: entropy is not input-sensitive"
print(f"  PASS: entropy differs ({h1:.4f} vs {h2:.4f})")

# ─────────────────────────────────────────────────────────────────────
# TEST 2: DRSN — does it produce input-sensitive spiking?
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 2] DRSNNetwork.encode_features — two distinct inputs")
from entity_interface.drsn_node import DRSNNetwork

drsn = DRSNNetwork(n_nodes=16, d_hidden=8)

# Input A: strong drive (high values)
result_a = drsn.encode_features([50.0] * 16, n_steps=10)
drsn.reset()
# Input B: no drive (zeros)
result_b = drsn.encode_features([0.0] * 16, n_steps=10)
drsn.reset()

print(f"  Input A (50.0 * 16): total_spikes={result_a['total_spikes']}, active_nodes={result_a['active_nodes']}")
print(f"  Input B (0.0 * 16):  total_spikes={result_b['total_spikes']}, active_nodes={result_b['active_nodes']}")
print(f"  World state A (first 4): {[round(v,4) for v in result_a['world_state'][:4]]}")
print(f"  World state B (first 4): {[round(v,4) for v in result_b['world_state'][:4]]}")
if result_a['total_spikes'] != result_b['total_spikes']:
    print("  PASS: DRSN is input-sensitive (spike counts differ)")
else:
    print("  PARTIAL: spike counts same but world_state may differ")

# ─────────────────────────────────────────────────────────────────────
# TEST 3: CSIE Sheaf — input-sensitive grounding?
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 3] CSIESheafLayer.ground_kronos_output — two distinct inputs")
from entity_interface.csie_sheaf import CSIESheafLayer

sheaf = CSIESheafLayer(d_model=64, n_concepts=32)

logits_a = [10.0, 1.0, 0.5, 0.1] + [0.0] * 28  # clearly peaked at index 0
logits_b = [0.1, 0.1, 0.1, 10.0] + [0.0] * 28  # clearly peaked at index 3

result_a = sheaf.ground_kronos_output(logits_a, context_id="ctx_A")
result_b = sheaf.ground_kronos_output(logits_b, context_id="ctx_B")

print(f"  Input A (peak at idx 0) top concept: {result_a['top_concepts'][0]}")
print(f"  Input B (peak at idx 3) top concept: {result_b['top_concepts'][0]}")
top_a = result_a['top_concepts'][0]['concept_id']
top_b = result_b['top_concepts'][0]['concept_id']
if top_a != top_b:
    print(f"  PASS: Different peak -> different top concept: {top_a} vs {top_b}")
else:
    print(f"  WARN: Same top concept despite different peak inputs: {top_a}")

# ─────────────────────────────────────────────────────────────────────
# TEST 4: APEX Causal — compose + path_integral real computation?
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 4] APEXCausalEngine — compose + path_integral")
from entity_interface.apex_causal import APEXCausalEngine, CausalObject, KMorphism

apex = APEXCausalEngine(max_k=5)
apex.add_object(CausalObject(id="A", name="A", domain="financial"))
apex.add_object(CausalObject(id="B", name="B", domain="financial"))
apex.add_object(CausalObject(id="C", name="C", domain="financial"))

m1 = KMorphism(k=1, source_id="A", target_id="B", relation_type="causes", weight=0.8)
m2 = KMorphism(k=1, source_id="B", target_id="C", relation_type="causes", weight=0.6)
apex.add_morphism(m1)
apex.add_morphism(m2)

composed = apex.compose("A->B:causes:1", "B->C:causes:1")
print(f"  compose(w=0.8, w=0.6): weight = {composed.weight:.4f} (expect 0.4800)")

pi = apex.path_integral("A", "C")
print(f"  path_integral(A->C): n_paths={pi['n_paths']}, total_weight={pi['total_weight']:.4f}")

apex2 = APEXCausalEngine(max_k=5)
apex2.add_object(CausalObject(id="A", name="A", domain="financial"))
apex2.add_object(CausalObject(id="B", name="B", domain="financial"))
apex2.add_object(CausalObject(id="C", name="C", domain="financial"))
m1b = KMorphism(k=1, source_id="A", target_id="B", relation_type="causes", weight=0.1)
m2b = KMorphism(k=1, source_id="B", target_id="C", relation_type="causes", weight=0.1)
apex2.add_morphism(m1b)
apex2.add_morphism(m2b)
composed2 = apex2.compose("A->B:causes:1", "B->C:causes:1")
print(f"  compose(w=0.1, w=0.1): weight = {composed2.weight:.4f} (expect 0.0100)")
if composed.weight != composed2.weight:
    print("  PASS: compose is input-sensitive")
else:
    print("  FAIL: compose is constant")

# ─────────────────────────────────────────────────────────────────────
# TEST 5: CIFNLinear — does forward pass vary with input?
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 5] CIFNLinear.forward — two distinct inputs")
import torch
from entity_interface.kronos.cifn import CIFNLinear

torch.manual_seed(0)
layer = CIFNLinear(in_features=8, out_features=15, basis_count=128)

x1 = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])  # financial entity
x2 = torch.tensor([[0.5, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]])  # iot entity

out1 = layer(x1)
out2 = layer(x2)

print(f"  Input 1 (financial-like) out[:5]: {[round(v,4) for v in out1[0][:5].tolist()]}")
print(f"  Input 2 (iot-like)       out[:5]: {[round(v,4) for v in out2[0][:5].tolist()]}")
print("  PASS: outputs differ" if not torch.allclose(out1, out2) else "  FAIL: outputs identical")

# ─────────────────────────────────────────────────────────────────────
# TEST 6: KRONOS — forward pass input-sensitive?
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 6] KRONOS.forward — two distinct inputs")
from entity_interface.kronos.kronos_architecture import KRONOS

torch.manual_seed(42)
kronos = KRONOS(
    vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=256,
    max_seq_len=32, memory_size=64, z_dim=64, n_slots=4, n_wave_freqs=16,
    dropout=0.0, kl_weight=0.05, notears_weight=0.01, notears_coeff=0.01
)
kronos.eval()

ids_a = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
ids_b = torch.tensor([[200, 210, 220, 230, 240, 250, 100, 110]])

with torch.no_grad():
    out_a = kronos(ids_a)
    out_b = kronos(ids_b)

la = out_a['logits'][0, 0, :6].tolist()
lb = out_b['logits'][0, 0, :6].tolist()
print(f"  Input A (low token ids)  logits[:6]: {[round(v,4) for v in la]}")
print(f"  Input B (high token ids) logits[:6]: {[round(v,4) for v in lb]}")
print("  PASS: outputs differ" if la != lb else "  FAIL: identical outputs")

# ─────────────────────────────────────────────────────────────────────
# TEST 7: KRONOS training — noise as both input AND label?
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 7] KRONOS _run_internal_training_step — are labels real or noise?")
# Reproducing lines 480-481 from live_entity.py:
torch.manual_seed(99)
input_ids = torch.randint(0, 256, (1, 8))
labels    = torch.randint(0, 256, (1, 8))
print(f"  input_ids: {input_ids.tolist()}")
print(f"  labels:    {labels.tolist()}")
overlap = (input_ids == labels).sum().item()
print(f"  Overlap between input_ids and labels: {overlap}/8 positions")
print("  VERDICT: labels = random integers 0-255, INDEPENDENT of input_ids — pure noise supervision")

# ─────────────────────────────────────────────────────────────────────
# TEST 8: Parameter counts — computed independently
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 8] LiveCausalNetwork parameter count — computed independently")
from entity_interface.live_entity import LiveCausalNetwork

net = LiveCausalNetwork(in_features=8, hidden_features=64)
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"  LiveCausalNetwork total parameters:     {total_params:,}")
print(f"  LiveCausalNetwork trainable parameters: {trainable_params:,}")
for name, p in net.named_parameters():
    print(f"    {name}: shape={list(p.shape)}, numel={p.numel()}, requires_grad={p.requires_grad}")

print("\n[TEST 9] KRONOS parameter count — computed independently")
from entity_interface.kronos.kronos_architecture import KRONOS

kronos_small = KRONOS(
    vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=256,
    max_seq_len=32, memory_size=64, z_dim=64, n_slots=4, n_wave_freqs=16,
    dropout=0.0, kl_weight=0.05, notears_weight=0.01, notears_coeff=0.01
)
k_params = sum(p.numel() for p in kronos_small.parameters())
k_trainable = sum(p.numel() for p in kronos_small.parameters() if p.requires_grad)
print(f"  KRONOS total parameters:     {k_params:,}")
print(f"  KRONOS trainable parameters: {k_trainable:,}")

# ─────────────────────────────────────────────────────────────────────
# TEST 10: CIFN synthetic training data source
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 10] CIFN _generate_synthetic_labels — training data source")
from entity_interface.live_entity import LiveEntity
tr_f, tr_l, val_f, val_l = LiveEntity._generate_synthetic_labels(n_per_class=10, val_fraction=0.2, seed=1)
print(f"  Train set size: {len(tr_l)} samples")
print(f"  Val set size:   {len(val_l)} samples")
print(f"  Labels in train (first 10): {tr_l[:10].tolist()}")
print(f"  Unique label values: {sorted(set(tr_l.tolist()))}")
print(f"  Sample feature row 0: {[round(v,3) for v in tr_f[0].tolist()]}")
print("  VERDICT: Labels are RULE-DERIVED from domain/entropy recipes, NOT from real-world outcomes or external datasets")

# ─────────────────────────────────────────────────────────────────────
# TEST 11: Config defaults
# ─────────────────────────────────────────────────────────────────────
print("\n[TEST 11] Config defaults")
from config import ENTITY_MODE, DATABASE_URL, CORS_ORIGINS
print(f"  ENTITY_MODE   = '{ENTITY_MODE}'")
print(f"  DATABASE_URL  = '{DATABASE_URL}'")
print(f"  CORS_ORIGINS  = {CORS_ORIGINS}")

env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
print(f"\n  Reading backend/.env:")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key = line.split('=')[0]
                # Redact actual secret values
                if any(x in key.upper() for x in ['KEY', 'SECRET', 'PASS', 'TOKEN']):
                    print(f"    {key}=<REDACTED>")
                else:
                    print(f"    {line}")
else:
    print("  backend/.env not found")

print("\n" + "=" * 70)
print("AUDIT TEST COMPLETE")
print("=" * 70)
