import os, sys, torch
os.environ["ENTITY_MODE"] = "live"
sys.path.insert(0, "backend")

from entity_interface.live_entity import LiveCausalNetwork, LiveEntity
from entity_interface.kronos.cifn import CIFNWeightField
import torch.nn as nn
import torch.nn.functional as F
import math

tr_f, tr_l, val_f, val_l = LiveEntity._generate_synthetic_labels(n_per_class=500, val_fraction=0.2)

# We want to test different frequency/phase initialization configurations.
# Let's dynamically modify CIFNWeightField's __init__ method for testing.

configs = {
    "orig": lambda self: None, # original
    "omega_scaled_0.5": lambda self: setattr(self, "omega_out", nn.Parameter(torch.randn(self.basis_count) * 0.5)) or setattr(self, "omega_in", nn.Parameter(torch.randn(self.basis_count) * 0.5)),
    "omega_scaled_0.2": lambda self: setattr(self, "omega_out", nn.Parameter(torch.randn(self.basis_count) * 0.2)) or setattr(self, "omega_in", nn.Parameter(torch.randn(self.basis_count) * 0.2)),
    "omega_uniform_1": lambda self: setattr(self, "omega_out", nn.Parameter(torch.rand(self.basis_count) * 2 - 1)) or setattr(self, "omega_in", nn.Parameter(torch.rand(self.basis_count) * 2 - 1)),
    "omega_uniform_pi": lambda self: setattr(self, "omega_out", nn.Parameter((torch.rand(self.basis_count) * 2 - 1) * math.pi)) or setattr(self, "omega_in", nn.Parameter((torch.rand(self.basis_count) * 2 - 1) * math.pi)),
}

# Save original __init__
orig_init = CIFNWeightField.__init__

for config_name, init_modifier in configs.items():
    print(f"\n--- Testing Config: {config_name} ---")
    
    def test_init(self, out_features: int, in_features: int, basis_count: int = 512):
        orig_init(self, out_features, in_features, basis_count)
        init_modifier(self)
        
    CIFNWeightField.__init__ = test_init
    
    accuracies = []
    # Test on 5 seeds first to quickly identify good configs
    seeds = [1, 2, 3, 4, 5]
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
            va = (val_logits.argmax(1) == val_l).float().mean().item()
        accuracies.append(va)
        
    print(f"Seeds {seeds} accuracies: {[round(a*100, 1) for a in accuracies]}")
    print(f"Average accuracy: {sum(accuracies)/len(accuracies)*100:.1f}%")
