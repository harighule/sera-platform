"""
Trace weights during Phase A with KRONOS updates.
"""
import sys
import os
import random
import numpy as np
import torch
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def main():
    os.environ['USE_NOETHER'] = 'false'
    os.environ['USE_PRETRAINED_CIFN'] = 'true'
    
    import config
    importlib.reload(config)
    import entity_interface.live_entity as le_mod
    importlib.reload(le_mod)
    
    def traced_train_with_kronos(self):
        import torch.nn.functional as F
        print(f"Weights at step 0: {self.model.cifn1.weight_field.a[:5].tolist()}")
        
        tr_f, tr_l, val_f, val_l = self._generate_synthetic_labels(
            n_per_class=500, val_fraction=0.2
        )
        cifn_opt_A = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        if hasattr(self.kronos_model, "head"):
            kronos_head = list(self.kronos_model.head.parameters())
        else:
            kronos_head = list(self.kronos_model.kronos.head.parameters())
        k_opt_A = torch.optim.Adam(kronos_head, lr=1e-3)
        N = len(tr_f)
        n_cls = len(le_mod.TRANSITION_TYPES)
        weights = torch.tensor([1.0]*n_cls) # simplified
        
        self.model.train()
        self.kronos_model.train()
        
        for step in range(1, 4):
            idx = torch.randint(0, N, (32,))
            xb, yb = tr_f[idx], tr_l[idx]
            
            # CIFN update
            out_c = self.model(xb)
            loss_c = F.cross_entropy(out_c["transition_logits"], yb)
            cifn_opt_A.zero_grad()
            loss_c.backward()
            cifn_opt_A.step()
            
            # KRONOS update
            with torch.no_grad():
                ids = (xb * 255).long().clamp(0, 255)
            with torch.enable_grad():
                kout = self.kronos_model(ids)
                klog = kout["logits"][:, 0, :n_cls]
                loss_k = F.cross_entropy(klog, yb)
            k_opt_A.zero_grad()
            loss_k.backward()
            k_opt_A.step()
            
            print(f"Weights at step {step}: {self.model.cifn1.weight_field.a[:5].tolist()}")

    le_mod.LiveEntity._train_cifn_classifier = traced_train_with_kronos

    # Set seed 42
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    entity = le_mod.LiveEntity()

if __name__ == '__main__':
    main()
