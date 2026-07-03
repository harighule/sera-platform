import os, sys, torch
os.environ["ENTITY_MODE"] = "live"
sys.path.insert(0, "backend")

from entity_interface.live_entity import LiveCausalNetwork, LiveEntity
from entity_interface.kronos.cifn import CIFNWeightField
import torch.nn as nn
import torch.nn.functional as F

tr_f, tr_l, val_f, val_l = LiveEntity._generate_synthetic_labels(n_per_class=500, val_fraction=0.2)

# Save original __init__
orig_init = CIFNWeightField.__init__

def test_frozen_waves_init(self, out_features: int, in_features: int, basis_count: int = 512):
    orig_init(self, out_features, in_features, basis_count)
    # Set requires_grad = False for omega and theta
    self.omega_out.requires_grad = False
    self.omega_in.requires_grad = False
    self.theta_out.requires_grad = False
    self.theta_in.requires_grad = False

CIFNWeightField.__init__ = test_frozen_waves_init

print("Training with frozen wave frequencies/phases:")
accuracies = []
seeds = list(range(1, 11))
for seed in seeds:
    torch.manual_seed(seed)
    m = LiveCausalNetwork()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-5)
    
    for step in range(1, 1001):
        opt.zero_grad()
        idx = torch.randint(0, len(tr_f), (32,))
        xb, yb = tr_f[idx], tr_l[idx]
        out = m(xb)
        loss = F.cross_entropy(out["transition_logits"], yb)
        loss.backward()
        opt.step()
        sch.step()
        
    with torch.no_grad():
        val_logits = m(val_f)["transition_logits"]
        vl = F.cross_entropy(val_logits, val_l).item()
        va = (val_logits.argmax(1) == val_l).float().mean().item()
    accuracies.append(va)
    print(f"Seed {seed}: val_loss={vl:.4f} val_acc={va*100:.1f}%")
    
print(f"\nAverage accuracy: {sum(accuracies)/len(accuracies)*100:.1f}%")
