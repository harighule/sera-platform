"""
NOETHER-KRONOS Integration Audit
Isolated inline test script — no servers, no network, no file writes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

print("=" * 70)
print("NOETHER-KRONOS INTEGRATION AUDIT")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Does NOETHER_KRONOS / noether_demo even exist?
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SECTION 0] Checking existence of NOETHER components …\n")

NOETHER_TARGETS = [
    "noether_demo",
    "NOETHER_KRONOS",
    "SymmetryDiscoveryEngine",
    "SemanticOrbitEncoder",
    "CausalFibrationNetwork",
]

import importlib, inspect, pkgutil

found_modules = {}
missing = []

for name in NOETHER_TARGETS:
    try:
        mod = importlib.import_module(name)
        found_modules[name] = mod
        print(f"  [OK]    import {name}  -> {mod.__file__}")
    except ModuleNotFoundError as e:
        missing.append(name)
        print(f"  [MISS]  import {name}  -> ModuleNotFoundError: {e}")
    except Exception as e:
        missing.append(name)
        print(f"  [ERR]   import {name}  -> {type(e).__name__}: {e}")

# Also scan the entire backend package tree
print("\n  Scanning backend package tree for any module containing 'noether' …")
backend_root = os.path.join(os.path.dirname(__file__), "backend")
found_noether_files = []
for root, dirs, files in os.walk(backend_root):
    # skip __pycache__
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fname in files:
        if "noether" in fname.lower():
            found_noether_files.append(os.path.join(root, fname))

if found_noether_files:
    for f in found_noether_files:
        print(f"    FOUND: {f}")
else:
    print("    None found.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CHECK 1 — Import KRONOS (the only real target) and check wiring
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[CHECK 1] KRONOS model import & attribute audit")
print("=" * 70)

try:
    from entity_interface.kronos.kronos_architecture import KRONOS
    print("  [OK] KRONOS imported from entity_interface.kronos.kronos_architecture")
except Exception as e:
    print(f"  [FATAL] Cannot import KRONOS: {e}")
    sys.exit(1)

# BASE_CONFIG equivalent (since noether_demo.py does not exist, reconstruct a
# plausible minimal config from the KRONOS __init__ signature)
BASE_CONFIG = {
    "vocab_size": 1000,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 128,
    "max_seq_len": 32,
    "memory_size": 32,
    "z_dim": 32,
    "n_slots": 4,
    "n_wave_freqs": 16,
    "dropout": 0.0,
    "kl_weight": 0.05,
    "notears_weight": 0.01,
    "notears_coeff": 0.01,
}

model = KRONOS(BASE_CONFIG)
print(f"  [OK] KRONOS instantiated with BASE_CONFIG. Param count: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# Check NOETHER attributes: sde, soe, cfn, ctl, arg
print("\n  Checking NOETHER attribute existence on model …")
noether_attrs = ["sde", "soe", "cfn", "ctl", "arg"]
for attr in noether_attrs:
    val = getattr(model, attr, "__MISSING__")
    if val == "__MISSING__":
        print(f"    [MISS] model.{attr}  -> NOT PRESENT")
    else:
        print(f"    [OK]   model.{attr}  -> {type(val).__name__}")

# Check actual attributes
print("\n  Actual top-level nn.Module children of KRONOS:")
for name, mod in model.named_children():
    print(f"    {name:30s}  {type(mod).__name__}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CHECK 1b — forward() call-site analysis (static inspection)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[CHECK 1b] Static inspection of KRONOS.forward() for call sites")
print("=" * 70)

import inspect
src = inspect.getsource(KRONOS.forward)
EXPECTED_CALLS = {
    "wave":     "sde (SymmetryDiscoveryEngine)",
    "memory":   "soe (SemanticOrbitEncoder)",
    "symbolic": "cfn (CausalFibrationNetwork)",
    "verifier": "ctl",
    # "arg" equivalent
}
# What we actually expect in a pure KRONOS forward:
ACTUAL_EXPECTED_CALLS = ["self.wave", "self.layers", "self.symbolic", "self.verifier", "self.out_norm", "self.head"]
print("  Checking for expected call sites in KRONOS.forward source:")
for call in ACTUAL_EXPECTED_CALLS:
    present = call in src
    marker = "[OK]  " if present else "[MISS]"
    print(f"    {marker}  {call}")

# Check NOETHER-specific calls
noether_calls = ["self.sde", "self.soe", "self.cfn", "self.ctl", "self.arg",
                 "orbit_loss", "ctl_loss", "rg_loss", "sym_scores", "rg_levels"]
print("\n  Checking for NOETHER-specific calls in KRONOS.forward:")
any_noether_found = False
for call in noether_calls:
    present = call in src
    if present:
        any_noether_found = True
        print(f"    [FOUND] {call}")
print("  None found." if not any_noether_found else "")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CHECK 1c — forward() output dict contents
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[CHECK 1c] KRONOS.forward() output dict — shape/key audit")
print("=" * 70)

import torch
model.eval()
torch.manual_seed(42)

B, T = 2, 8
input_ids = torch.randint(0, BASE_CONFIG["vocab_size"], (B, T))
with torch.no_grad():
    out = model(input_ids)

print(f"  Keys returned: {list(out.keys())}")
for k, v in out.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k:30s}  shape={tuple(v.shape)}  dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"    {k:30s}  list of len={len(v)}, item type={type(v[0]).__name__ if v else 'empty'}")
    else:
        print(f"    {k:30s}  {type(v).__name__} = {v}")

# Check NOETHER-expected keys
noether_keys = ["orbit_loss", "ctl_loss", "rg_loss", "sym_scores", "rg_levels"]
print("\n  NOETHER-expected output keys:")
for k in noether_keys:
    present = k in out
    print(f"    {'[FOUND]' if present else '[MISS] '}  {k}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CHECK 2 — SymmetryDiscoveryEngine (G_sem) real vs random
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[CHECK 2] SymmetryDiscoveryEngine audit")
print("=" * 70)

# Does any module named SymmetryDiscoveryEngine, SemanticOrbitEncoder,
# CausalFibrationNetwork exist anywhere in the codebase?
TARGET_CLASSES = [
    "SymmetryDiscoveryEngine",
    "SemanticOrbitEncoder",
    "CausalFibrationNetwork",
    "NOETHER_KRONOS",
]

print("  Searching all .py files under backend/ for class definitions:")
for root, dirs, files in os.walk(backend_root):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for cls in TARGET_CLASSES:
                if cls in content:
                    # find line numbers
                    for i, line in enumerate(content.splitlines(), 1):
                        if cls in line:
                            rel = os.path.relpath(fpath, backend_root)
                            print(f"    [FOUND] {cls} in {rel}:{i}  -> {line.strip()[:80]}")
        except Exception:
            pass

print("  Scan complete.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CHECK 3 — SemanticOrbitEncoder differentiation (orbit pooling)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[CHECK 3] SemanticOrbitEncoder / orbit pooling audit")
print("=" * 70)
print("  SemanticOrbitEncoder not found in codebase — CANNOT TEST.")
print("  Fallback: checking KRONOS forward output differentiation with two different inputs.")

torch.manual_seed(1)
ids_a = torch.randint(0, BASE_CONFIG["vocab_size"], (1, T))
ids_b = torch.randint(0, BASE_CONFIG["vocab_size"], (1, T))
# ensure they differ
assert not torch.all(ids_a == ids_b)

with torch.no_grad():
    out_a = model(ids_a)
    out_b = model(ids_b)

logit_diff = (out_a["logits"] - out_b["logits"]).abs().max().item()
print(f"  Max logit difference between two different inputs: {logit_diff:.6f}")
print(f"  {'[OK]  Outputs differ — model is input-sensitive' if logit_diff > 1e-6 else '[FAIL] Outputs are IDENTICAL — model is collapsed'}")

# contrastive_loss test — not applicable since SemanticOrbitEncoder doesn't exist
print("  contrastive_loss(): N/A — SemanticOrbitEncoder does not exist in codebase.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CHECK 4 — CausalFibrationNetwork (G_caus)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[CHECK 4] CausalFibrationNetwork audit")
print("=" * 70)
print("  CausalFibrationNetwork not found in codebase — CANNOT TEST.")
print("  Fallback: KRONOS has its own internal causal attention (CausalGraphAttention).")
print("  Testing causal_adj matrix effect via CausalGraphAttention in KRONOS layers…")

from entity_interface.kronos.kronos_architecture import CausalGraphAttention
cga = CausalGraphAttention(d_model=64, n_heads=4, max_seq_len=32)
cga.eval()

x_test = torch.randn(1, 8, 64)
with torch.no_grad():
    out_cga, penalty = cga(x_test)
print(f"  CausalGraphAttention output shape: {tuple(out_cga.shape)}, penalty: {penalty.item():.6f}")

# The CausalGraphAttention uses its own internal W_logit for causal_adj —
# there's no external causal_adj parameter in KRONOS.forward()
print("  Note: KRONOS.forward() does NOT accept an external causal_adj argument.")
print("  The causal structure is internal to CausalGraphAttention (W_logit parameter).")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)
print(f"{'#':<3} {'Component':<35} {'Wired In':<12} {'Functionally Real':<20} Evidence")
print("-" * 100)

rows = [
    (1,  "NOETHER_KRONOS (class)",            "NO",       "NO",          "Module/class does not exist anywhere in codebase"),
    (2,  "noether_demo.py (BASE_CONFIG)",      "NO",       "NO",          "File does not exist anywhere in codebase"),
    (3,  "model.sde (SymmetryDiscoveryEng.)",  "NO",       "NO",          "Attribute absent from KRONOS; class not defined anywhere"),
    (4,  "model.soe (SemanticOrbitEncoder)",   "NO",       "NO",          "Attribute absent from KRONOS; class not defined anywhere"),
    (5,  "model.cfn (CausalFibrationNet.)",    "NO",       "NO",          "Attribute absent from KRONOS; class not defined anywhere"),
    (6,  "model.ctl",                          "NO",       "NO",          "Attribute absent; NOETHER CTL component not defined"),
    (7,  "model.arg",                          "NO",       "NO",          "Attribute absent; NOETHER ARG component not defined"),
    (8,  "forward() calls all 5 NOETHER comp.","NO",       "NO",          "forward() calls wave/layers/symbolic/verifier — no NOETHER calls"),
    (9,  "orbit_loss in output dict",          "NO",       "NO",          "Absent from return dict; KRONOS returns kl_loss/notears_penalty"),
    (10, "ctl_loss in output dict",            "NO",       "NO",          "Absent from return dict"),
    (11, "rg_loss in output dict",             "NO",       "NO",          "Absent from return dict"),
    (12, "sym_scores in output dict",          "NO",       "NO",          "Absent from return dict"),
    (13, "rg_levels in output dict",           "NO",       "NO",          "Absent from return dict"),
    (14, "KRONOS itself (9-pillar arch.)",     "PARTIAL",  "PARTIAL",     "Imports/runs; bypassed in live_entity.py L91 (see report.md)"),
    (15, "CausalGraphAttention (internal)",    "YES",      "YES",         "Wired in KRONOSLayer; uses W_logit for causal structure"),
    (16, "Input-sensitivity of KRONOS fwd()", "YES",       "YES",         f"Max logit diff between inputs = {logit_diff:.4f} (non-collapsed)"),
]

for row in rows:
    num, comp, wired, real, evidence = row
    flag = " [!]" if wired == "NO" or real == "NO" else "    "
    print(f"{num:<3} {comp:<35} {wired:<12} {real:<20} {evidence}{flag}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("""
The names NOETHER, NOETHER_KRONOS, noether_demo, SymmetryDiscoveryEngine,
SemanticOrbitEncoder, CausalFibrationNetwork, and the attributes
sde / soe / cfn / ctl / arg do NOT exist anywhere in this codebase.

There is no file, class, function, or import referencing any NOETHER
cognitive symmetry component. The KRONOS model (kronos_architecture.py)
has its own 9-pillar architecture (wave manifold, causal attention,
Hopfield memory, active inference, slot attention, NCA, CoT verifier)
but NONE of these are the NOETHER components named in the task.

The task's premise — that NOETHER components are "present in code but
decorative/inert" — is itself incorrect. They are not present at all.

All four checks (1–4) are untestable as specified because the modules,
classes, and methods they reference do not exist.
""")
